// Compare the cell-wise hanging-node kernel of deal.II,
// dealii::Portable::internal::resolve_hanging_nodes(), with the batched one
// of the deal.II branch batched-hanging-nodes,
// Copy::Portable::internal::batched::resolve_hanging_nodes(), on many cells.
//
// The question is whether the batched kernel can replace the cell-wise one
// in Portable::FEEvaluation (batch size 1) without a regression, and what
// batching gains on top.
//
// Every kernel reads the values of its cells from global memory into shared
// memory, resolves the hanging nodes there and writes them back, like
// FEEvaluation::read_dof_values() and distribute_local_to_global() do:
//
//   copy        only the global <-> shared copy, no hanging nodes (baseline)
//   cellwise    one cell per team, the cell-wise kernel if the cell is
//               constrained (what FEEvaluation does today)
//   batched B   B cells per team, the batched kernel
//
// The masks are the ones of tests/matrix_free/hanging_node_kernels_01 (those
// occurring on a mesh with one refined cell), randomly assigned to a given
// fraction of the cells; the others are unconstrained. The 1d interpolation
// weights are the FE_Q ones Portable::MatrixFree uses. All kernels must
// produce identical results.
//
//   ./program dim degree [n_dofs] [n_repetitions]

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_hanging_nodes_internal.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <multigrid_dealii/portable_hanging_nodes_batched.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace dealii;

using ConstraintKinds = dealii::internal::MatrixFreeFunctions::ConstraintKinds;

using ExecutionSpace = MemorySpace::Default::kokkos_space::execution_space;
using TeamPolicy     = Kokkos::TeamPolicy<ExecutionSpace>;
using TeamHandle     = TeamPolicy::member_type;

template <typename T>
using DeviceView = Kokkos::View<T, MemorySpace::Default::kokkos_space>;

using SharedView = Kokkos::
  View<double *, ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;



struct KernelData
{
  DeviceView<double *>          weights;
  DeviceView<ConstraintKinds *> masks;
  DeviceView<double *>          src;
  DeviceView<double *>          dst;
  int                           n_cells;
};



// global -> shared copy of the cells [first_cell, first_cell + n) of a team
template <int n_dofs>
DEAL_II_HOST_DEVICE inline void
read_cells(const TeamHandle &team_member,
           const KernelData &data,
           const int         first_cell,
           const int         n_cells_in_batch,
           SharedView        values)
{
  const int offset = first_cell * n_dofs;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_cells_in_batch * n_dofs),
                       [&](const int i) { values(i) = data.src(offset + i); });
  team_member.team_barrier();
}



template <int n_dofs>
DEAL_II_HOST_DEVICE inline void
write_cells(const TeamHandle &team_member,
            const KernelData &data,
            const int         first_cell,
            const int         n_cells_in_batch,
            SharedView        values)
{
  const int offset = first_cell * n_dofs;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_cells_in_batch * n_dofs),
                       [&](const int i) { data.dst(offset + i) = values(i); });
}



// Separate functors for the three kernels, so that they are separate device
// kernels with their own resource usage (cuobjdump --dump-resource-usage).
template <int dim, int fe_degree, bool transpose>
struct CopyKernel
{
  static constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

  KernelData data;

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    const int  cell = team_member.league_rank();
    SharedView values(team_member.team_shmem(), n_dofs);
    read_cells<n_dofs>(team_member, data, cell, 1, values);
    write_cells<n_dofs>(team_member, data, cell, 1, values);
  }
};



template <int dim, int fe_degree, bool transpose>
struct CellwiseKernel
{
  static constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

  KernelData data;

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    const int  cell = team_member.league_rank();
    SharedView values(team_member.team_shmem(), n_dofs);
    read_cells<n_dofs>(team_member, data, cell, 1, values);

    // as in Portable::FEEvaluation::read_dof_values()
    const ConstraintKinds mask = data.masks(cell);
    if (mask != ConstraintKinds::unconstrained)
      dealii::Portable::internal::resolve_hanging_nodes<dim, fe_degree, transpose, double>(
        team_member, data.weights, mask, values);

    write_cells<n_dofs>(team_member, data, cell, 1, values);
  }
};



template <int dim, int fe_degree, bool transpose>
struct BatchedKernel
{
  static constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

  KernelData data;
  int        cells_per_batch;

  DEAL_II_HOST_DEVICE void
  operator()(const TeamHandle &team_member) const
  {
    const int first_cell = team_member.league_rank() * cells_per_batch;
    const int n_cells_in_batch =
      (data.n_cells - first_cell < cells_per_batch) ? (data.n_cells - first_cell) : cells_per_batch;

    SharedView values(team_member.team_shmem(), cells_per_batch * n_dofs);
    read_cells<n_dofs>(team_member, data, first_cell, n_cells_in_batch, values);

    const auto masks = data.masks;
    Copy::Portable::internal::batched::resolve_hanging_nodes<dim, fe_degree, transpose>(
      team_member,
      data.weights,
      [&](const int cell) { return masks(first_cell + cell); },
      n_cells_in_batch,
      values);

    write_cells<n_dofs>(team_member, data, first_cell, n_cells_in_batch, values);
  }
};



struct Timing
{
  std::string name;
  int         team_size;
  double      median;
  double      min;
  double      max_difference;
};



// Launch a kernel n_repetitions times after two warm-up launches; returns
// the median and minimum time of one launch.
template <typename Functor>
Timing
time_kernel(const std::string          &name,
            const Functor              &functor,
            const int                   n_teams,
            const int                   shared_values,
            const int                   n_repetitions,
            const DeviceView<double *> &dst,
            const DeviceView<double *> &reference)
{
  ExecutionSpace exec;
  TeamPolicy     policy(exec, n_teams, Kokkos::AUTO);
  policy.set_scratch_size(0, Kokkos::PerTeam(SharedView::shmem_size(shared_values)));

  const int team_size = policy.team_size_recommended(functor, Kokkos::ParallelForTag());

  Kokkos::deep_copy(dst, 0.);
  std::vector<double> times;
  for (int r = 0; r < n_repetitions + 2; ++r)
    {
      exec.fence();
      Kokkos::Timer timer;
      Kokkos::parallel_for("hanging_nodes_" + name, policy, functor);
      exec.fence();
      if (r >= 2)
        times.push_back(timer.seconds());
    }
  std::sort(times.begin(), times.end());

  // compare with the reference (the cell-wise kernel)
  double max_difference = 0.;
  if (reference.data() != dst.data())
    Kokkos::parallel_reduce(
      "compare",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, dst.size()),
      KOKKOS_LAMBDA(const int i, double &m) {
        const double d = Kokkos::abs(dst(i) - reference(i));
        m              = (d > m) ? d : m;
      },
      Kokkos::Max<double>(max_difference));

  return {name, team_size, times[times.size() / 2], times.front(), max_difference};
}



template <int dim>
std::vector<ConstraintKinds>
realistic_masks()
{
  using CK = ConstraintKinds;
  if constexpr (dim == 2)
    return {CK::face_x | CK::subcell_x | CK::subcell_y,
            CK::face_x | CK::subcell_x,
            CK::face_x | CK::subcell_y,
            CK::face_x,
            CK::face_y | CK::subcell_y | CK::subcell_x,
            CK::face_y | CK::subcell_y,
            CK::face_y | CK::subcell_x,
            CK::face_y};
  else
    return {CK::edge_x | CK::subcell_y | CK::subcell_z,
            CK::edge_x | CK::subcell_y | CK::subcell_z | CK::subcell_x,
            CK::edge_x | CK::subcell_z,
            CK::edge_x | CK::subcell_z | CK::subcell_x,
            CK::edge_x | CK::subcell_y,
            CK::edge_x | CK::subcell_y | CK::subcell_x,
            CK::edge_x,
            CK::edge_x | CK::subcell_x,
            CK::edge_y | CK::subcell_x | CK::subcell_z,
            CK::edge_y | CK::subcell_x | CK::subcell_z | CK::subcell_y,
            CK::edge_y | CK::subcell_z,
            CK::edge_y | CK::subcell_z | CK::subcell_y,
            CK::edge_y | CK::subcell_x,
            CK::edge_y | CK::subcell_x | CK::subcell_y,
            CK::edge_y,
            CK::edge_y | CK::subcell_y,
            CK::edge_z | CK::subcell_x | CK::subcell_y,
            CK::edge_z | CK::subcell_x | CK::subcell_y | CK::subcell_z,
            CK::edge_z | CK::subcell_y,
            CK::edge_z | CK::subcell_y | CK::subcell_z,
            CK::edge_z | CK::subcell_x,
            CK::edge_z | CK::subcell_x | CK::subcell_z,
            CK::edge_z,
            CK::edge_z | CK::subcell_z,
            CK::face_x | CK::subcell_x,
            CK::face_x,
            CK::face_y | CK::subcell_y,
            CK::face_y,
            CK::face_z | CK::subcell_z,
            CK::face_z};
}



// the 1d interpolation weights Portable::MatrixFree copies to the device
template <int dim>
std::vector<double>
fe_q_weights(const unsigned int degree)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<dim>       fe(degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(MappingQ1<dim>(),
                     dof_handler,
                     constraints,
                     QGauss<1>(degree + 1),
                     typename MatrixFree<dim, double>::AdditionalData());

  const auto &matrix = matrix_free.get_shape_info().data.front().subface_interpolation_matrices[0];
  return std::vector<double>(matrix.begin(), matrix.end());
}



template <int dim, int fe_degree, bool transpose>
void
run(const KernelData                 &data,
    const DeviceView<double *>       &reference,
    const std::vector<int>           &batch_sizes,
    const int                         n_repetitions,
    std::vector<std::vector<Timing>> &results)
{
  constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

  std::vector<Timing> timings;

  // the cell-wise kernel first, it writes the reference
  KernelData reference_data = data;
  reference_data.dst        = reference;
  timings.push_back(time_kernel("cellwise",
                                CellwiseKernel<dim, fe_degree, transpose>{reference_data},
                                data.n_cells,
                                n_dofs,
                                n_repetitions,
                                reference,
                                reference));
  timings.push_back(time_kernel("copy",
                                CopyKernel<dim, fe_degree, transpose>{data},
                                data.n_cells,
                                n_dofs,
                                n_repetitions,
                                data.dst,
                                data.dst));
  for (const int b : batch_sizes)
    timings.push_back(time_kernel("batched " + std::to_string(b),
                                  BatchedKernel<dim, fe_degree, transpose>{data, b},
                                  (data.n_cells + b - 1) / b,
                                  b * n_dofs,
                                  n_repetitions,
                                  data.dst,
                                  reference));

  results.push_back(timings);
}



void
print_table(const std::vector<Timing> &timings, const long long n_cells, const int n_dofs)
{
  const double copy_time = timings[1].median;
  const double cellwise  = timings[0].median;
  const double bytes     = 2. * n_cells * n_dofs * sizeof(double);

  std::cout << std::left << std::setw(12) << "kernel" << std::right << std::setw(7) << "team"
            << std::setw(12) << "median[ms]" << std::setw(10) << "min[ms]" << std::setw(10)
            << "ns/cell" << std::setw(9) << "GB/s" << std::setw(12) << "-copy[ms]" << std::setw(12)
            << "vs cellwise" << std::setw(12) << "-copy vs cw" << std::setw(11) << "max diff"
            << std::endl;

  for (const auto &t : timings)
    {
      const double overhead          = t.median - copy_time;
      const double cellwise_overhead = cellwise - copy_time;
      std::cout << std::left << std::setw(12) << t.name << std::right << std::setw(7) << t.team_size
                << std::fixed << std::setprecision(3) << std::setw(12) << 1e3 * t.median
                << std::setw(10) << 1e3 * t.min << std::setprecision(2) << std::setw(10)
                << 1e9 * t.median / n_cells << std::setprecision(0) << std::setw(9)
                << 1e-9 * bytes / t.median << std::setprecision(3) << std::setw(12)
                << 1e3 * overhead << std::setprecision(2) << std::setw(12) << cellwise / t.median
                << std::setw(12);
      if (t.name == "copy")
        std::cout << "-";
      else
        std::cout << cellwise_overhead / overhead;
      std::cout << std::scientific << std::setprecision(1) << std::setw(11) << t.max_difference
                << std::defaultfloat << std::endl;
    }
  std::cout << std::endl;
}



template <int dim, int fe_degree>
int
benchmark(const long long target_n_dofs, const int n_repetitions)
{
  constexpr int n_dofs  = Utilities::pow(fe_degree + 1, dim);
  const int     n_cells = std::max<long long>(1, target_n_dofs / n_dofs);

  const std::vector<int>    batch_sizes = {1, 2, 4, 8};
  const std::vector<double> fractions   = {1., 0.05};

  std::cout << "dim=" << dim << " degree=" << fe_degree << " n_cells=" << n_cells
            << " dofs/cell=" << n_dofs << " n_dofs=" << static_cast<long long>(n_cells) * n_dofs
            << " repetitions=" << n_repetitions << " execution space=" << ExecutionSpace::name()
            << std::endl
            << std::endl;

  KernelData data;
  data.n_cells = n_cells;
  data.weights = DeviceView<double *>("weights", (fe_degree + 1) * (fe_degree + 1));
  data.masks   = DeviceView<ConstraintKinds *>("masks", n_cells);
  data.src     = DeviceView<double *>(Kokkos::view_alloc("src", Kokkos::WithoutInitializing),
                                  static_cast<std::size_t>(n_cells) * n_dofs);
  data.dst     = DeviceView<double *>("dst", data.src.size());
  DeviceView<double *> reference("reference", data.src.size());

  {
    const auto weights = fe_q_weights<dim>(fe_degree);
    auto       host    = Kokkos::create_mirror_view(data.weights);
    for (unsigned int i = 0; i < weights.size(); ++i)
      host(i) = weights[i];
    Kokkos::deep_copy(data.weights, host);
  }
  {
    Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool(1);
    Kokkos::fill_random(data.src, pool, -1., 1.);
  }

  int failures = 0;
  for (const double fraction : fractions)
    {
      const auto                             masks = realistic_masks<dim>();
      std::mt19937                           rng(42);
      std::uniform_real_distribution<double> uniform(0., 1.);
      std::uniform_int_distribution<int>     pick(0, masks.size() - 1);
      auto                                   host          = Kokkos::create_mirror_view(data.masks);
      long long                              n_constrained = 0;
      for (int c = 0; c < n_cells; ++c)
        if (uniform(rng) < fraction)
          {
            host(c) = masks[pick(rng)];
            ++n_constrained;
          }
        else
          host(c) = ConstraintKinds::unconstrained;
      Kokkos::deep_copy(data.masks, host);

      std::vector<std::vector<Timing>> results;
      run<dim, fe_degree, false>(data, reference, batch_sizes, n_repetitions, results);
      run<dim, fe_degree, true>(data, reference, batch_sizes, n_repetitions, results);

      for (unsigned int transpose = 0; transpose < 2; ++transpose)
        {
          std::cout << "constrained cells: " << n_constrained << " (" << std::fixed
                    << std::setprecision(1) << 100. * n_constrained / n_cells << "%)"
                    << std::defaultfloat << ", "
                    << (transpose ? "transpose (distribute_local_to_global)" :
                                    "non-transpose (read_dof_values)")
                    << std::endl;
          print_table(results[transpose], n_cells, n_dofs);
          for (const auto &t : results[transpose])
            if (t.max_difference != 0.)
              ++failures;
        }
    }

  std::cout << (failures == 0 ? "PASSED: all kernels give identical results" :
                                "FAILED: results differ from the cell-wise kernel")
            << std::endl
            << std::endl;
  return failures;
}



template <int dim>
int
dispatch(const int degree, const long long n_dofs, const int n_repetitions)
{
  switch (degree)
    {
      case 1:
        return benchmark<dim, 1>(n_dofs, n_repetitions);
      case 2:
        return benchmark<dim, 2>(n_dofs, n_repetitions);
      case 3:
        return benchmark<dim, 3>(n_dofs, n_repetitions);
      case 4:
        return benchmark<dim, 4>(n_dofs, n_repetitions);
      case 5:
        return benchmark<dim, 5>(n_dofs, n_repetitions);
      case 6:
        return benchmark<dim, 6>(n_dofs, n_repetitions);
      default:
        std::cerr << "degree must be in [1, 6]" << std::endl;
        return 1;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  if (argc < 3)
    {
      std::cerr << "usage: " << argv[0]
                << " dim degree [n_dofs (default 2^27)] [n_repetitions (default 20)]" << std::endl;
      return 1;
    }

  const int       dim           = std::atoi(argv[1]);
  const int       degree        = std::atoi(argv[2]);
  const long long n_dofs        = (argc > 3) ? std::atoll(argv[3]) : (1ll << 27);
  const int       n_repetitions = (argc > 4) ? std::atoi(argv[4]) : 20;

  if (dim == 2)
    return dispatch<2>(degree, n_dofs, n_repetitions);
  else if (dim == 3)
    return dispatch<3>(degree, n_dofs, n_repetitions);

  std::cerr << "dim must be 2 or 3" << std::endl;
  return 1;
}
