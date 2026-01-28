#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "operators/portable_laplace_operator.h"
using namespace dealii;

template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(bool overlap_communication_computation = false);

  void
  run();

private:
  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  setup_smoothers();

  void
  assemble_rhs();

  void
  solve();

  void
  output_results(const unsigned int cycle) const;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::set<types::boundary_id> dirichlet_boundary_ids = {0};

  std::unique_ptr<Portable::LaplaceOperator<dim, fe_degree, double>>
    system_matrix;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_device;

  using VectorTypeMG =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  using SmootherType =
    PreconditionChebyshev<Portable::LaplaceOperatorBase<dim, double>,
                          VectorTypeMG>;

  SmootherType smoother;

  bool overlap_communication_computation;

  ConditionalOStream pcout;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(
  bool overlap_communication_computation)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , overlap_communication_computation(overlap_communication_computation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *>
    dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           dirichlet_boundary_functions,
                                           constraints);
  constraints.close();

  pcout << " Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  system_matrix.reset(new Portable::LaplaceOperator<dim, fe_degree, double>(
    dof_handler, constraints, overlap_communication_computation));

  system_matrix->initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_smoothers()
{
  system_matrix->compute_diagonal();

  typename SmootherType::AdditionalData smoother_data;

  smoother_data.smoothing_range     = 15.;
  smoother_data.degree              = 5;
  smoother_data.eig_cg_n_iterations = 10;
  smoother_data.preconditioner = system_matrix->get_matrix_diagonal_inverse();

  smoother.initialize(*system_matrix, smoother_data);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *
                              fe_values.JxW(q_index));

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs_host);
        }
    }

  system_rhs_host.compress(VectorOperation::add);
  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  pcout << "  RHS assembled" << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve()
{
  SolverControl solver_control(system_rhs_device.size(),
                               1e-12 * system_rhs_device.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  cg.solve(*system_matrix, solution_device, system_rhs_device, smoother);

  pcout << "  Solver converged in " << solver_control.last_step()
        << " iterations." << std::endl;

  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(solution_device, VectorOperation::insert);
  ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

  constraints.distribute(ghost_solution_host);

  ghost_solution_host.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  (void)cycle;
  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(ghost_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", cycle, mpi_communicator, 2);

  Vector<float> cellwise_norm(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    ghost_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  const double global_norm =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_norm,
                                      VectorTools::L2_norm);

  pcout << "  solution norm: " << global_norm << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  pcout << "============== fe_degree = " << fe_degree << " ============== \n\n";

  for (unsigned int cycle = 0; cycle < 12 - dim; ++cycle)
    {
      pcout << std::endl << std::endl;
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation, 0., 1.);
          triangulation.refine_global(2);
        }
      else
        {
          triangulation.refine_global(1);
        }


      setup_dofs();
      setup_matrix_free();
      setup_smoothers();
      assemble_rhs();

      solve();
      output_results(cycle);

      pcout << std::endl;
    }
}

template <int dim, int degree>
void
solve_for_degree(int fe_degree)
{
  if (degree == fe_degree)
    {
      LaplaceProblem<dim, degree> laplace_problem(false);
      laplace_problem.run();
    }
  else if constexpr (degree < 5)
    solve_for_degree<dim, degree + 1>(fe_degree);
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const int dim           = 3;
      const int max_fe_degree = 3;

      for (int fe_degree = 1; fe_degree <= max_fe_degree; ++fe_degree)
        {
          solve_for_degree<dim, 1>(fe_degree);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

