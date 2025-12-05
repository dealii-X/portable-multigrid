#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"

using namespace dealii;

template <int dim, int fe_degree, int mg_levels>
class LaplaceProblem
{
public:
  LaplaceProblem(bool overlap_communication_computation = false);

  void
  run();

private:
  void
  setup_system();

  void
  assemble_rhs();

  void
  setup_mg_transfers();

  void
  solve();

  void
  output_results(const unsigned int cycle) const;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  std::vector<FE_Q<dim>>                        fe_collection;
  std::vector<std::shared_ptr<DoFHandler<dim>>> dof_handler_collection;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::vector<AffineConstraints<double>> constraints_collection;

  MGLevelObject<std::unique_ptr<Portable::LaplaceOperatorBase<dim, double>>>
    mg_matrices;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_device;

  MGLevelObject<std::unique_ptr<Portable::PolynomialTransferBase<dim, double>>>
    mg_transfers;

  bool               overlap_communication_computation;
  ConditionalOStream pcout;

  struct LaplaceOperatorRunner
  {
    const int                  level;
    DoFHandler<dim>           &dof_handler;
    AffineConstraints<double> &constraints;
    bool                       overlap_communication_computation;
    LaplaceProblem<dim, fe_degree, mg_levels> &parent_problem;

    template <unsigned int degree>
    void
    run()
    {
      parent_problem.mg_matrices[level] =
        std::make_unique<Portable::LaplaceOperator<dim, degree, double>>(
          dof_handler, constraints, overlap_communication_computation);
    }
  };

  struct PolynomialTransferRunner
  {
    const int                                level;
    const Portable::MatrixFree<dim, double> &mf_coarse;
    const Portable::MatrixFree<dim, double> &mf_fine;
    AffineConstraints<double>               &constraints_coarse;
    AffineConstraints<double>               &constraints_fine;

    LaplaceProblem<dim, fe_degree, mg_levels> &parent_problem;

    template <unsigned int degree_coarse, unsigned int degree_fine>
    void
    run()
    {
      parent_problem.mg_transfers[level] = std::make_unique<
        Portable::
          PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();

      parent_problem.mg_transfers[level]->reinit(mf_coarse,
                                                 mf_fine,
                                                 constraints_coarse,
                                                 constraints_fine);
    }
  };
};

template <int dim, int fe_degree, int mg_levels>
LaplaceProblem<dim, fe_degree, mg_levels>::LaplaceProblem(
  bool overlap_communication_computation)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , overlap_communication_computation(overlap_communication_computation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
  {
    Assert(mg_levels <= fe_degree,
           ExcMessage(
             "The number of levels must be at most as the polynomial degree"));

    fe_collection.reserve(mg_levels);
    dof_handler_collection.reserve(mg_levels);
    constraints_collection.reserve(mg_levels);

    for (int level = 0; level < mg_levels; ++level)
      {
        const int p_degree = fe_degree - (mg_levels - 1 - level);
        if (p_degree > 0)
          {
            fe_collection.emplace_back(p_degree);
            dof_handler_collection.emplace_back(
              std::make_shared<DoFHandler<dim>>(triangulation));
            constraints_collection.emplace_back();
          }
      }
  }
}

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::setup_system()
{
  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *>
    dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  mg_matrices.clear();
  mg_matrices.resize(0, mg_levels - 1);

  for (int level = 0; level < mg_levels; ++level)
    {
      auto &fe          = fe_collection[level];
      auto &dof_handler = *dof_handler_collection[level];
      auto &constraints = constraints_collection[level];

      dof_handler.distribute_dofs(fe);

      const int p_degree = fe_degree - (mg_levels - 1 - level);

      pcout << "level " << level << ": " << "p = " << p_degree << ", "
            << "DoFs = " << dof_handler.n_dofs() << std::endl;

      constraints.clear();
      constraints.reinit(dof_handler.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               dirichlet_boundary_functions,
                                               constraints);
      constraints.close();

      LaplaceOperatorRunner runner{level,
                                   dof_handler,
                                   constraints,
                                   overlap_communication_computation,
                                   *this};

      bool success =
        Portable::OperatorDispatchFactory::dispatch(p_degree, runner);

      Assert(success,
             ExcMessage(
               "Failed to find a matching polynomial degree in dispatcher."));
    }

  locally_owned_dofs = dof_handler_collection.back()->locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(*dof_handler_collection.back());

  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);

  mg_matrices.back()->initialize_dof_vector(solution_device);

  system_rhs_device.reinit(solution_device);
}

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::setup_mg_transfers()
{
  mg_transfers.resize(mg_matrices.min_level(), mg_matrices.max_level());

  for (unsigned int level = mg_matrices.min_level() + 1;
       level <= mg_matrices.max_level();
       ++level)
    {
      PolynomialTransferRunner runner{(int)level,
                                      mg_matrices[level - 1]->get_matrix_free(),
                                      mg_matrices[level]->get_matrix_free(),
                                      constraints_collection[level - 1],
                                      constraints_collection[level],
                                      *this};

      const int p_fine   = fe_degree - (mg_levels - 1 - level);
      const int p_coarse = p_fine - 1;

      bool success =
        Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse,
                                                              p_fine,
                                                              runner);

      Assert(success,
             ExcMessage("Failed to find a matching polynomial degree "
                        "pair in transfer dispatcher."));
    }
}

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::assemble_rhs()
{
  auto &fe          = fe_collection.back();
  auto &dof_handler = *dof_handler_collection.back();
  auto &constraints = constraints_collection.back();

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
}

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::solve()
{
  auto &constraints = constraints_collection.back();

  auto &system_matrix_device = mg_matrices.back();

  using SmootherType = PreconditionChebyshev<
    Portable::LaplaceOperatorBase<dim, double>,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>;

  MGLevelObject<SmootherType> mg_smoothers;
  mg_smoothers.resize(0, mg_levels - 1);

  for (int level = 0; level < mg_levels; ++level)
    {
      typename SmootherType::AdditionalData smoother_data;
      if (level > 0)
        {
          smoother_data.smoothing_range     = 15.;
          smoother_data.degree              = 5;
          smoother_data.eig_cg_n_iterations = 10;
        }
      else
        {
          smoother_data.smoothing_range = 1e-3;
          smoother_data.degree          = numbers::invalid_unsigned_int;
          smoother_data.smoothing_range = mg_matrices[0]->m();
        }
      smoother_data.constraints.copy_from(constraints_collection[level]);
      mg_matrices[level]->compute_diagonal();
      smoother_data.preconditioner =
        mg_matrices[level]->get_matrix_diagonal_inverse();

      mg_smoothers[level].initialize(*mg_matrices[level], smoother_data);
    }

  Portable::
    VCycleMultigrid<dim, double, Portable::PolynomialTransferBase<dim, double>>
      mg_preconditioner(mg_matrices, mg_transfers, mg_smoothers, 3, 3);

  SolverControl solver_control(system_rhs_device.size(),
                               1e-12 * system_rhs_device.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);
  cg.solve(*system_matrix_device,
           solution_device,
           system_rhs_device,
           mg_preconditioner);

  pcout << "  Solver converged in " << solver_control.last_step()
        << " iterations." << std::endl;

  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(solution_device, VectorOperation::insert);
  ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

  constraints.distribute(ghost_solution_host);

  ghost_solution_host.update_ghost_values();
}

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::output_results(
  const unsigned int cycle) const
{
  auto &dof_handler = *dof_handler_collection.back();
  auto &fe          = fe_collection.back();

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(ghost_solution_host, "solution");
  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", cycle, mpi_communicator, 2);

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

template <int dim, int fe_degree, int mg_levels>
void
LaplaceProblem<dim, fe_degree, mg_levels>::run()
{
  for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
    {
      pcout << std::endl << std::endl;
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation, 0., 1.);
          triangulation.refine_global(3 - dim);
        }
      else
        {
          triangulation.refine_global(1);
        }

      setup_system();
      setup_mg_transfers();
      assemble_rhs();
      solve();
      output_results(cycle);

      pcout << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const int dim       = 2;
      const int fe_degree = 7;
      const int mg_levels = 7;

      const bool overlap_communication_computation = false;

      LaplaceProblem<dim, fe_degree, mg_levels> laplace_problem(
        overlap_communication_computation);

      laplace_problem.run();
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
