// Timing of the vmult variants of Portable::ElasticityOperator, linear
// elasticity with FESystem(FE_Q(degree), 3) and degree + 1 Gauss points per
// direction, on one mesh per program run (one rank per GPU):
//   vmult_dealii     deal.II's Portable::FEEvaluation, one cell per team
//   vmult_batched    the batched FEEvaluation of this repository
//   vmult_fused      vmult_batched with the stress inside evaluate_and_apply()
//   vmult_benchmark  the kernel of the dealii-X benchmarks
// The mesh is a box of 2^k cubes (k = 3 refinements + 0, 1 or 2 directions
// split once more), deformed so that every cell is a general one and all
// variants read the geometry per quadrature point; homogeneous Dirichlet
// conditions on the boundary. Times are the maximum over the ranks, after
// warm-up; |A src| relative to vmult_dealii checks that the variants agree.
//
//   ./program degree [n_dofs (default 1e7)] [n_repetitions (default 20)]

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
#include <vector>

#include "operators/portable_elasticity_operator.h"

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



template <int degree>
int
run(const double target_n_dofs, const int n_repetitions)
{
  constexpr int dim = 3;
  using Material    = Portable::LinearElasticMaterial<dim, Number>;
  using Operator    = Portable::ElasticityOperator<dim, degree, Number, Material, degree + 1>;

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
  const Material material{1.7, 0.6};
  const Operator op(dof_handler, constraints, material, false);

  VectorType src, dst;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> host(src.get_partitioner());
    for (unsigned int i = 0; i < host.locally_owned_size(); ++i)
      host.local_element(i) = std::sin(0.001 * (host.get_partitioner()->local_to_global(i)));
    constraints.set_zero(host);
    src.import_elements(host, VectorOperation::insert);
  }
  Kokkos::fence();
  const double setup_time    = setup_timer.wall_time();
  const double memory_after  = device_memory_used();
  const double memory_op_max = Utilities::MPI::max(memory_after - memory_before, comm);
  const double memory_max    = Utilities::MPI::max(memory_after, comm);

  const types::global_dof_index n_dofs = dof_handler.n_dofs();
  pcout << "degree=" << degree << " n_q_points_1d=" << degree + 1
        << " ranks=" << Utilities::MPI::n_mpi_processes(comm)
        << " cells=" << tria.n_global_active_cells() << " (" << repetitions[0] << "x"
        << repetitions[1] << "x" << repetitions[2] << ", " << n_refinements
        << " refinements) dofs=" << n_dofs << std::endl
        << "  setup " << std::setprecision(3) << setup_time << " s, device memory per rank: max "
        << memory_max << " GiB, of which operator + vectors " << memory_op_max << " GiB ("
        << memory_op_max * (1u << 30) / (n_dofs / double(Utilities::MPI::n_mpi_processes(comm)))
        << " bytes/dof)" << std::endl;

  const std::vector<std::pair<std::string, std::function<void()>>> variants = {
    {"vmult_dealii", [&]() { op.vmult_dealii(dst, src); }},
    {"vmult_batched", [&]() { op.vmult_batched(dst, src); }},
    {"vmult_fused", [&]() { op.vmult_fused(dst, src); }},
    {"vmult_benchmark", [&]() { op.vmult_benchmark(dst, src); }}};

  double reference_norm = 0.;
  for (const auto &[name, vmult] : variants)
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

      const double norm = dst.l2_norm();
      if (reference_norm == 0.)
        reference_norm = norm;

      pcout << "  " << std::left << std::setw(16) << name << std::right << " median "
            << std::setw(9) << std::setprecision(4) << 1e3 * t_median << " ms, min " << std::setw(9)
            << 1e3 * t_min << " ms, " << std::setw(7) << std::setprecision(4)
            << 1e-9 * n_dofs / t_median << " GDoF/s (median), |A src| rel. diff " << std::scientific
            << std::setprecision(2) << std::abs(norm - reference_norm) / reference_norm
            << std::defaultfloat << std::endl;
    }
  pcout << std::endl;

  return 0;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  if (argc < 2)
    {
      std::cerr << "usage: " << argv[0]
                << " degree [n_dofs (default 1e7)] [n_repetitions (default 20)]" << std::endl;
      return 1;
    }

  const int    degree        = std::atoi(argv[1]);
  const double n_dofs        = (argc > 2) ? std::atof(argv[2]) : 1e7;
  const int    n_repetitions = (argc > 3) ? std::atoi(argv[3]) : 20;

  switch (degree)
    {
      case 1:
        return run<1>(n_dofs, n_repetitions);
      case 2:
        return run<2>(n_dofs, n_repetitions);
      case 3:
        return run<3>(n_dofs, n_repetitions);
      case 4:
        return run<4>(n_dofs, n_repetitions);
      default:
        std::cerr << "degree must be in [1, 4]" << std::endl;
        return 1;
    }
}
