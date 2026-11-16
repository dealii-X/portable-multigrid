// Laplace operator (step-64 style) on deal.II's own Portable::MatrixFree and
// Portable::FEEvaluation, applied on an adaptively refined mesh with hanging
// nodes. Two uses:
//
//  1. cuobjdump --dump-resource-usage on the binary gives the register count
//     of the real cell-loop kernels (Portable::internal::ApplyKernel with the
//     local operator below), to judge whether the extra registers of the
//     batched hanging-node kernel can lower the occupancy of an operator.
//  2. Built once against deal.II master and once against the deal.II branch
//     in which Portable::FEEvaluation resolves hanging nodes with the batched
//     kernel (batch size 1), the vmult timings are the end-to-end comparison.
//
// The mesh is a cube, refined globally to roughly the requested number of
// dofs, then once more either inside a ball around the center (mesh 'b':
// the cells with hanging nodes lie on the sphere around the refined region,
// a few percent) or on a random half of the cells (mesh 'r': most cells have
// hanging nodes, the worst case for the hanging-node kernels). Homogeneous
// Dirichlet conditions on the boundary. |A src| is printed to compare the
// results of two builds.
//
//   ./program dim degree [n_dofs (default 2^25)] [n_repetitions (default 50)]
//             [mesh b|r (default b)]

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace dealii;

using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;



template <int dim, int fe_degree>
struct LocalLaplaceOperator
{
  static constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, double>::Data *data,
             const Portable::DeviceVector<double>                   &src,
             Portable::DeviceVector<double>                         &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);
    data->for_each_quad_point([&](const int &q_point)
                              { fe_eval.submit_gradient(fe_eval.get_gradient(q_point), q_point); });
    fe_eval.integrate(EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }
};



template <int dim, int fe_degree>
int
run(const long long target_n_dofs, const int n_repetitions, const char mesh)
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);

  // global refinement such that, with the local refinement below, the mesh
  // has about target_n_dofs dofs (the ball refines about a tenth of
  // the cells, the random mesh half of them) and at most max_cells cells:
  // at low degrees the dofs alone would give tens of millions of cells,
  // whose setup on one rank does not fit into a job
  constexpr double max_cells           = 1 << 24;
  const double     fraction_refined    = (mesh == 'r') ? 0.5 : 0.1;
  const double     growth              = 1. + ((1 << dim) - 1) * fraction_refined;
  const double     cells_per_direction = std::pow(target_n_dofs / growth, 1. / dim) / fe_degree;
  int              n_global_refinements =
    std::max(1, static_cast<int>(std::round(std::log2(cells_per_direction))));
  while (n_global_refinements > 1 && std::pow(2., dim * n_global_refinements) * growth > max_cells)
    --n_global_refinements;
  tria.refine_global(n_global_refinements);

  Point<dim> center;
  for (unsigned int d = 0; d < dim; ++d)
    center[d] = 0.5;
  unsigned int random_state = 1;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        // a fixed linear congruential sequence, the same on every machine
        random_state = 1103515245u * random_state + 12345u;
        const bool refine =
          (mesh == 'r') ? ((random_state >> 16) & 1) : (cell->center().distance(center) < 0.3);
        if (refine)
          cell->set_refine_flag();
      }
  tria.execute_coarsening_and_refinement();

  const FE_Q<dim> fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  AffineConstraints<double> constraints(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // cells with at least one dof constrained by a hanging node (the boundary
  // constraints are only counted through a separate hanging-node object)
  unsigned int n_cells_with_hanging_nodes = 0;
  {
    AffineConstraints<double> hanging_nodes(dof_handler.locally_owned_dofs(),
                                            locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_nodes);
    hanging_nodes.close();
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          if (std::any_of(dof_indices.begin(),
                          dof_indices.end(),
                          [&](const types::global_dof_index i)
                          { return hanging_nodes.is_constrained(i); }))
            ++n_cells_with_hanging_nodes;
        }
  }

  Portable::MatrixFree<dim, double>                          matrix_free;
  typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_gradients | update_JxW_values;
  matrix_free.reinit(
    MappingQ1<dim>(), dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data);

  VectorType src, dst;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> host(src.get_partitioner());
    for (unsigned int i = 0; i < host.locally_owned_size(); ++i)
      host.local_element(i) = std::sin(0.001 * i);
    src.import_elements(host, VectorOperation::insert);
  }

  const auto vmult = [&]()
  {
    dst = 0.;
    matrix_free.cell_loop(LocalLaplaceOperator<dim, fe_degree>(), src, dst);
    matrix_free.copy_constrained_values(src, dst);
  };

  // warm-up
  for (int r = 0; r < 3; ++r)
    vmult();
  Kokkos::fence();

  std::vector<double> times;
  for (int r = 0; r < n_repetitions; ++r)
    {
      Kokkos::fence();
      Kokkos::Timer timer;
      vmult();
      Kokkos::fence();
      times.push_back(timer.seconds());
    }
  std::sort(times.begin(), times.end());
  const double median = times[times.size() / 2];

  const double norm = dst.l2_norm();

  const unsigned int n_cells = tria.n_global_active_cells();
  std::cout << "dim=" << dim << " degree=" << fe_degree
            << " mesh=" << ((mesh == 'r') ? "random half" : "ball") << " hanging-node kernel="
#ifdef HANGING_NODES_OVERLAY
            << "batched (overlay)"
#else
            << "cell-wise"
#endif
            << " execution space=" << MemorySpace::Default::kokkos_space::execution_space::name()
            << std::endl
            << "  cells " << n_cells << " (" << n_global_refinements
            << " global refinements + local), dofs " << dof_handler.n_dofs()
            << ", cells with hanging nodes " << n_cells_with_hanging_nodes << " (" << std::fixed
            << std::setprecision(1) << 100. * n_cells_with_hanging_nodes / n_cells << "%)"
            << std::defaultfloat << std::endl
            << "  vmult median " << std::setprecision(4) << 1e3 * median << " ms, min "
            << 1e3 * times.front() << " ms, max " << 1e3 * times.back() << " ms, "
            << std::setprecision(4) << 1e-9 * dof_handler.n_dofs() / median << " GDoFs/s"
            << std::endl
            << "  |A src| = " << std::setprecision(12) << norm << std::defaultfloat << std::endl
            << std::endl;

  return 0;
}



template <int dim>
int
dispatch(const int degree, const long long n_dofs, const int n_repetitions, const char mesh)
{
  switch (degree)
    {
      case 1:
        return run<dim, 1>(n_dofs, n_repetitions, mesh);
      case 2:
        return run<dim, 2>(n_dofs, n_repetitions, mesh);
      case 3:
        return run<dim, 3>(n_dofs, n_repetitions, mesh);
      case 4:
        return run<dim, 4>(n_dofs, n_repetitions, mesh);
      case 5:
        return run<dim, 5>(n_dofs, n_repetitions, mesh);
      case 6:
        return run<dim, 6>(n_dofs, n_repetitions, mesh);
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
                << " dim degree [n_dofs (default 2^25)]"
                   " [n_repetitions (default 50)] [mesh b|r (default b)]"
                << std::endl;
      return 1;
    }

  const int       dim           = std::atoi(argv[1]);
  const int       degree        = std::atoi(argv[2]);
  const long long n_dofs        = (argc > 3) ? std::atoll(argv[3]) : (1ll << 25);
  const int       n_repetitions = (argc > 4) ? std::atoi(argv[4]) : 50;
  const char      mesh          = (argc > 5) ? argv[5][0] : 'b';

  if (dim == 2)
    return dispatch<2>(degree, n_dofs, n_repetitions, mesh);
  else if (dim == 3)
    return dispatch<3>(degree, n_dofs, n_repetitions, mesh);

  std::cerr << "dim must be 2 or 3" << std::endl;
  return 1;
}
