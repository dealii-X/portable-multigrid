// Timing of the nonlinear Saint Venant-Kirchhoff residual R(u) of
// Portable::ElasticityOperator (or, for comparison, the linear operator),
// FESystem(FE_Q(degree), 3) and degree + 1 Gauss points per direction, on one
// mesh per program run (one rank per GPU), all vmult variants:
//   vmult_dealii     deal.II's Portable::FEEvaluation, one cell per team
//   vmult_batched    the batched FEEvaluation of this repository
//   vmult_fused      vmult_batched with the stress inside evaluate_and_apply()
//   vmult_benchmark  the kernel of the dealii-X benchmarks
//   vmult_bk3        the BK3-style counterpart of the benchmark kernel
//   residual_store   vmult_batched storing grad u for the linearized operator
//   residual_fused   the same as vmult_fused
//   linearized       the operator linearized at u = src, applied to src: as
//                    vmult_batched, plus the stored grad u (9 doubles per
//                    quadrature point) read
//   linearized_fused the same as vmult_fused
//   recompute        linearized with grad u recomputed from u (a second
//                    evaluation in a second set of team memory), not stored
//   recompute_fused  the same as linearized_fused
// For the linear material the linearized operator is the operator itself; it
// ignores grad u, so the compiler drops its loads.
// The mesh is a box of 2^k cubes (k = 3 refinements + 0, 1 or 2 directions
// split once more), deformed so that every cell is a general one and all
// variants read the geometry per quadrature point; homogeneous Dirichlet
// conditions on the boundary. Times are the maximum over the ranks, after
// warm-up; |A src - A_dealii src| / |A_dealii src| checks that the variants agree
// (for the linearized variants: against linearized).
// All variants but vmult_dealii (one cell per team) run with the same batch:
// cells_per_batch, or with 0 the benchmark's rule (10 kB of shared values per
// batch, 1 to 32 cells). Every line prints the batch it ran with.
// With svk, src is scaled by 0.05 (strains of a few percent). With ogden
// (Singh et al. parameters) src is the interpolant of a smooth field
// vanishing on the boundary, |grad u| < 0.2, so that J > 0 everywhere
// (J <= 0 returns NaN early).
// Every line prints the number of non-finite entries of dst (should be 0).
//
//   ./program degree [n_dofs (default 1e7)] [n_repetitions (default 20)]
//             [cells_per_batch (default 0)] [material svk|linear|ogden (default svk)]
//
// Only the materials and degrees of the CMake variables
// ELASTICITY_RESIDUAL_MATERIALS and ELASTICITY_RESIDUAL_DEGREES are compiled
// (instantiations.cmake), all of them by default.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#  include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "operators/portable_elasticity_operator.h"

#ifndef ELASTICITY_RESIDUAL_SVK
#  define ELASTICITY_RESIDUAL_SVK 1
#endif
#ifndef ELASTICITY_RESIDUAL_LINEAR
#  define ELASTICITY_RESIDUAL_LINEAR 1
#endif
#ifndef ELASTICITY_RESIDUAL_OGDEN
#  define ELASTICITY_RESIDUAL_OGDEN 1
#endif
#ifndef ELASTICITY_RESIDUAL_DEGREE_1
#  define ELASTICITY_RESIDUAL_DEGREE_1 1
#endif
#ifndef ELASTICITY_RESIDUAL_DEGREE_2
#  define ELASTICITY_RESIDUAL_DEGREE_2 1
#endif
#ifndef ELASTICITY_RESIDUAL_DEGREE_3
#  define ELASTICITY_RESIDUAL_DEGREE_3 1
#endif
#ifndef ELASTICITY_RESIDUAL_DEGREE_4
#  define ELASTICITY_RESIDUAL_DEGREE_4 1
#endif

using namespace dealii;

using Number     = double;
using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;



// device memory in use on this rank, in GiB (0 without CUDA)
double
device_memory_used()
{
#ifdef KOKKOS_ENABLE_CUDA
  std::size_t free = 0, total = 0;
  cudaMemGetInfo(&free, &total);
  return (total - free) / double(1u << 30);
#else
  return 0.;
#endif
}



// u_c = 0.05 sin(2 x_(c+1) + x_(c+2)) prod_d sin^4(pi x_d / L_d): the bubble
// keeps the jump to the zero boundary values small, |grad u| < 0.2, J > 0
template <int dim>
struct SmoothDisplacement : public Function<dim>
{
  SmoothDisplacement(const Point<dim> &extent)
    : Function<dim>(dim)
    , extent(extent)
  {}

  double
  value(const Point<dim> &p, const unsigned int c) const override
  {
    double bubble = 1.;
    for (unsigned int d = 0; d < dim; ++d)
      bubble *= Utilities::fixed_power<4>(std::sin(numbers::PI * p[d] / extent[d]));
    return 0.05 * bubble * std::sin(2. * p[(c + 1) % dim] + p[(c + 2) % dim]);
  }

  const Point<dim> extent;
};



template <typename Material>
Material
make_material()
{
  if constexpr (std::is_same_v<Material, Portable::OgdenMaterial<3, Number>>)
    {
      return Material{6.24, 16.714, 935., 4.456, 2.1e4};
    }
  else
    {
      return Material{1.7, 0.6};
    }
}



template <typename Material>
std::string
material_name()
{
  if constexpr (std::is_same_v<Material, Portable::LinearElasticMaterial<3, Number>>)
    {
      return "linear";
    }
  else if constexpr (std::is_same_v<Material, Portable::OgdenMaterial<3, Number>>)
    {
      return "ogden";
    }
  else
    {
      return "svk";
    }
}



template <int degree, typename Material>
int
run(const double target_n_dofs, const int n_repetitions, const unsigned int cells_per_batch)
{
  constexpr int  dim    = 3;
  constexpr bool linear = std::is_same_v<Material, Portable::LinearElasticMaterial<dim, Number>>;
  constexpr bool ogden  = std::is_same_v<Material, Portable::OgdenMaterial<dim, Number>>;
  using Operator        = Portable::ElasticityOperator<dim, degree, Number, Material, degree + 1>;

  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // 2^k cells with about 3 degree^3 dofs each
  const double              cells = target_n_dofs / (3. * degree * degree * degree);
  const int                 k     = std::max(3, static_cast<int>(std::round(std::log2(cells))));
  const int                 n_refinements = k / 3;
  std::vector<unsigned int> repetitions(dim, 1);
  for (int d = 0; d < k % 3; ++d)
    repetitions[d] = 2;

  Timer setup_timer;

  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_rectangle(
    tria, repetitions, Point<dim>(), Point<dim>(repetitions[0], repetitions[1], repetitions[2]));
  tria.refine_global(n_refinements);
  GridTools::transform(
    [](const Point<dim> &p)
    {
      Point<dim> result = p;
      result[0] += 0.05 * std::sin(2. * p[1]) * p[2];
      result[1] += 0.05 * std::sin(3. * p[2]) * p[0];
      result[2] += 0.05 * std::sin(2. * p[0]) * p[1];
      return result;
    },
    tria);

  const FESystem<dim> fe(FE_Q<dim>{degree}, dim);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  AffineConstraints<Number> constraints(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    MappingQ1<dim>(), dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints);
  constraints.close();

  const double   memory_before = device_memory_used();
  const Material material      = make_material<Material>();
  Operator       op(dof_handler, constraints, material, false, cells_per_batch);

  VectorType src, dst, dst_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> host(src.get_partitioner());
    if (ogden)
      VectorTools::interpolate(MappingQ1<dim>(),
                               dof_handler,
                               SmoothDisplacement<dim>(
                                 Point<dim>(repetitions[0], repetitions[1], repetitions[2])),
                               host);
    else
      for (unsigned int i = 0; i < host.locally_owned_size(); ++i)
        host.local_element(i) = std::sin(0.001 * (host.get_partitioner()->local_to_global(i)));
    constraints.set_zero(host);
    src.import_elements(host, VectorOperation::insert);
    if (!linear && !ogden)
      src *= 0.05;
  }
  Kokkos::fence();
  const double setup_time    = setup_timer.wall_time();
  const double memory_after  = device_memory_used();
  const double memory_op_max = Utilities::MPI::max(memory_after - memory_before, comm);
  const double memory_max    = Utilities::MPI::max(memory_after, comm);

  const types::global_dof_index n_dofs = dof_handler.n_dofs();
  pcout << material_name<Material>() << " degree=" << degree << " n_q_points_1d=" << degree + 1
        << " ranks=" << Utilities::MPI::n_mpi_processes(comm)
        << " cells=" << tria.n_global_active_cells() << " (" << repetitions[0] << "x"
        << repetitions[1] << "x" << repetitions[2] << ", " << n_refinements
        << " refinements) dofs=" << n_dofs << std::endl
        << "  setup " << std::setprecision(3) << setup_time << " s, device memory per rank: max "
        << memory_max << " GiB, of which operator + vectors " << memory_op_max << " GiB ("
        << memory_op_max * (1u << 30) / (n_dofs / double(Utilities::MPI::n_mpi_processes(comm)))
        << " bytes/dof)" << std::endl;

  // grad u, allocated by the first evaluate_residual()
  {
    const double memory_before_state = device_memory_used();
    op.evaluate_residual(dst, src);
    Kokkos::fence();
    const double memory_state =
      Utilities::MPI::max(device_memory_used() - memory_before_state, comm);
    pcout << "  stored grad u per rank: max " << memory_state << " GiB ("
          << memory_state * (1u << 30) / (n_dofs / double(Utilities::MPI::n_mpi_processes(comm)))
          << " bytes/dof)" << std::endl;
  }
  // u for the recompute variants
  op.set_linearization_point(src);

  // new_reference: compare this and the following variants with this one
  struct Variant
  {
    std::string           name;
    unsigned int          batch;
    std::function<void()> vmult;
    bool                  new_reference = false;
  };
  const std::vector<Variant> variants = {
    {"vmult_dealii", 1, [&]() { op.vmult_dealii(dst, src); }},
    {"vmult_batched", op.cells_per_batch_batched(), [&]() { op.vmult_batched(dst, src); }},
    {"vmult_fused", op.cells_per_batch_batched(), [&]() { op.vmult_fused(dst, src); }},
    {"vmult_benchmark", op.cells_per_batch_benchmark(), [&]() { op.vmult_benchmark(dst, src); }},
    {"vmult_bk3", op.cells_per_batch_bk3(), [&]() { op.vmult_bk3(dst, src); }},
    {"residual_store", op.cells_per_batch_batched(), [&]() { op.evaluate_residual(dst, src); }},
    {"residual_fused",
     op.cells_per_batch_batched(),
     [&]() { op.evaluate_residual_fused(dst, src); }},
    {"linearized", op.cells_per_batch_batched(), [&]() { op.vmult_linearized(dst, src); }, true},
    {"linearized_fused",
     op.cells_per_batch_batched(),
     [&]() { op.vmult_linearized_fused(dst, src); }},
    {"recompute",
     op.cells_per_batch_batched(),
     [&]() { op.vmult_linearized_recompute(dst, src); }},
    {"recompute_fused",
     op.cells_per_batch_batched(),
     [&]() { op.vmult_linearized_recompute_fused(dst, src); }}};

  double reference_norm = 0.;
  for (const auto &[name, batch, vmult, new_reference] : variants)
    {
      for (int r = 0; r < 2; ++r)
        vmult();
      Kokkos::fence();

      std::vector<double> times;
      for (int r = 0; r < n_repetitions; ++r)
        {
          Kokkos::fence();
          MPI_Barrier(comm);
          Timer timer;
          vmult();
          Kokkos::fence();
          times.push_back(Utilities::MPI::max(timer.wall_time(), comm));
        }
      std::sort(times.begin(), times.end());
      const double t_min    = times.front();
      const double t_median = times[times.size() / 2];

      // the first variant (vmult_dealii) is the reference
      if (reference_norm == 0. || new_reference)
        {
          dst_ref        = dst;
          reference_norm = dst_ref.l2_norm();
        }
      unsigned int n_nonfinite = 0;
      {
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> host(dst.get_partitioner());
        host.import_elements(dst, VectorOperation::insert);
        for (unsigned int i = 0; i < host.locally_owned_size(); ++i)
          n_nonfinite += !std::isfinite(host.local_element(i));
        n_nonfinite = Utilities::MPI::sum(n_nonfinite, comm);
      }
      dst -= dst_ref;
      const double difference = dst.l2_norm() / reference_norm;

      pcout << "  " << std::left << std::setw(19) << name << std::right << " batch " << std::setw(2)
            << batch << " median " << std::setw(9) << std::setprecision(4) << 1e3 * t_median
            << " ms, min " << std::setw(9) << 1e3 * t_min << " ms, " << std::setw(7)
            << std::setprecision(4) << 1e-9 * n_dofs / t_median
            << " GDoF/s (median), rel. difference " << std::scientific << std::setprecision(2)
            << difference << std::defaultfloat << ", non-finite " << n_nonfinite << std::endl;
    }
  pcout << std::endl;

  return 0;
}



template <typename Material>
int
run_degree(const int          degree,
           const double       n_dofs,
           const int          n_repetitions,
           const unsigned int cells_per_batch)
{
  switch (degree)
    {
#if ELASTICITY_RESIDUAL_DEGREE_1
      case 1:
        return run<1, Material>(n_dofs, n_repetitions, cells_per_batch);
#endif
#if ELASTICITY_RESIDUAL_DEGREE_2
      case 2:
        return run<2, Material>(n_dofs, n_repetitions, cells_per_batch);
#endif
#if ELASTICITY_RESIDUAL_DEGREE_3
      case 3:
        return run<3, Material>(n_dofs, n_repetitions, cells_per_batch);
#endif
#if ELASTICITY_RESIDUAL_DEGREE_4
      case 4:
        return run<4, Material>(n_dofs, n_repetitions, cells_per_batch);
#endif
      default:
        (void)n_dofs;
        (void)n_repetitions;
        (void)cells_per_batch;
        std::cerr << "degree " << degree
                  << " not compiled (ELASTICITY_RESIDUAL_DEGREES, degrees 1 to 4)" << std::endl;
        return 1;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  if (argc < 2)
    {
      std::cerr << "usage: " << argv[0]
                << " degree [n_dofs (default 1e7)] [n_repetitions (default 20)]"
                << " [cells_per_batch (default 0)] [material svk|linear|ogden (default svk)]"
                << std::endl;
      return 1;
    }

  const int          degree          = std::atoi(argv[1]);
  const double       n_dofs          = (argc > 2) ? std::atof(argv[2]) : 1e7;
  const int          n_repetitions   = (argc > 3) ? std::atoi(argv[3]) : 20;
  const unsigned int cells_per_batch = (argc > 4) ? std::atoi(argv[4]) : 0;
  const std::string  material        = (argc > 5) ? argv[5] : "svk";
  if (material != "linear" && material != "svk" && material != "ogden")
    {
      std::cerr << "material must be linear, svk or ogden" << std::endl;
      return 1;
    }

#if ELASTICITY_RESIDUAL_LINEAR
  if (material == "linear")
    return run_degree<Portable::LinearElasticMaterial<3, Number>>(degree,
                                                                  n_dofs,
                                                                  n_repetitions,
                                                                  cells_per_batch);
#endif
#if ELASTICITY_RESIDUAL_SVK
  if (material == "svk")
    return run_degree<Portable::SaintVenantKirchhoffMaterial<3, Number>>(degree,
                                                                         n_dofs,
                                                                         n_repetitions,
                                                                         cells_per_batch);
#endif
#if ELASTICITY_RESIDUAL_OGDEN
  if (material == "ogden")
    return run_degree<Portable::OgdenMaterial<3, Number>>(degree,
                                                          n_dofs,
                                                          n_repetitions,
                                                          cells_per_batch);
#endif
  std::cerr << "material " << material << " not compiled (ELASTICITY_RESIDUAL_MATERIALS)"
            << std::endl;
  return 1;
}
