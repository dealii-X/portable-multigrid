// Correctness check for Portable::VectorLaplaceOperator::compute_rhs_bk4()
// (portable_vector_laplace_operator.h): assembles the constant-unit-load
// vector RHS directly on the GPU via BK4::Parallel::KokkosRHSAbstracted()
// (bk4_kokkos_kernels.h) -- the same Custom::Parallel abstraction layer
// vmult_bk4() is built on -- replacing the CPU FEValues loop
// tests/bp4/program.cc used to run through host memory. Compares against
// that same CPU FEValues assembly (real deal.II, taken as ground truth) on
// a random mesh/degree combination, at both the operator's default
// n_q_points_1d (nq == nm) and an over-integrated one (nq == nm + 1, what
// tests/bp4/program.cc uses).

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>

#include "operators/portable_vector_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;

  // CPU FEValues assembly of the same constant-unit-load RHS, exactly as
  // tests/bp4/program.cc's compute_rhs() did before this GPU path existed
  // -- taken as ground truth here.
  template <int dim, int n_components>
  void
  compute_rhs_fevalues(
    const DoFHandler<dim>                                             &dof_handler,
    const FESystem<dim>                                               &fe,
    const AffineConstraints<double>                                   &constraints,
    const unsigned int                                                 n_q_points_1d,
    const IndexSet                                                    &locally_owned_dofs,
    const IndexSet                                                    &locally_relevant_dofs,
    const MPI_Comm                                                    &mpi_communicator,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>  &rhs_device)
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> rhs_host(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const QGauss<dim> quadrature_formula(n_q_points_1d);
    FEValues<dim>      fe_values(fe, quadrature_formula, update_values | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int c = 0; c < n_components; ++c)
                cell_rhs(i) +=
                  fe_values.shape_value_component(i, q_index, c) * 1.0 * fe_values.JxW(q_index);

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs_host);
        }

    rhs_host.compress(VectorOperation::add);
    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import_elements(rhs_host, VectorOperation::insert);
    rhs_device.import_elements(rw_vector, VectorOperation::insert);
  }

  // Runs the comparison for one (dim, fe_degree, n_components, n_q_points_1d)
  // combination on a small hyper-cube mesh. Returns the relative error
  // between compute_rhs_bk4() and the CPU FEValues assembly, which should
  // be at machine precision.
  template <int dim, int fe_degree, int n_components, int n_q_points_1d = fe_degree + 1>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_components = " << n_components << ", n_q_points_1d = " << n_q_points_1d
          << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);

    const FE_Q<dim> scalar_fe(fe_degree);
    FESystem<dim>   fe(scalar_fe, n_components);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero(n_components);
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    constraints.close();

    Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double, n_q_points_1d> op(
      dof_handler, constraints, /* overlap_communication_computation = */ false);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> rhs_gpu, rhs_cpu;
    op.initialize_dof_vector(rhs_gpu);
    op.initialize_dof_vector(rhs_cpu);

    op.compute_rhs_bk4(rhs_gpu);

    compute_rhs_fevalues<dim, n_components>(dof_handler,
                                            fe,
                                            constraints,
                                            n_q_points_1d,
                                            locally_owned_dofs,
                                            locally_relevant_dofs,
                                            mpi_communicator,
                                            rhs_cpu);

    const double norm_cpu = rhs_cpu.l2_norm();

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = rhs_gpu;
    diff -= rhs_cpu;
    const double rel_err = norm_cpu > 0 ? diff.l2_norm() / norm_cpu : diff.l2_norm();

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |rhs_cpu| = " << norm_cpu
          << ", rel_err(gpu vs cpu) = " << rel_err << std::endl
          << std::endl;

    return rel_err;
  }
} // namespace multigrid

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double max_rel_err = 0.;

  try
    {
      // Default n_q_points_1d (nq == nm), matching vector_laplace_bk4's cases.
      max_rel_err = std::max(max_rel_err, run_test<2, 2, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 2, 3>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 3, 2>(MPI_COMM_WORLD, pcout));

      // Over-integrated (nq == nm + 1), what tests/bp4/program.cc uses.
      max_rel_err =
        std::max(max_rel_err, run_test<3, 2, 3, /* n_q_points_1d = */ 4>(MPI_COMM_WORLD, pcout));
      max_rel_err =
        std::max(max_rel_err, run_test<3, 3, 2, /* n_q_points_1d = */ 5>(MPI_COMM_WORLD, pcout));
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-10;

  if (max_rel_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_rel_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_rel_err << " (tolerance " << tolerance << ")"
        << std::endl;

  return 0;
}
