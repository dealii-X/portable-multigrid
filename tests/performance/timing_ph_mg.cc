#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"



#define ENABLE_MPI
#define ENABLE_PERFORMANCE_TESTS
#define TESTING_ENVIRONMENT heavy

#include "performance_test_driver.h"

using namespace dealii;

dealii::ConditionalOStream debug_output(std::cout, false);

static unsigned int max_levels = 15;



template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int refinement_cycles,
                 const bool         overlap_communication_computation = false);

  Measurement
  run();

  using VectorTypeMG =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  using SmootherType =
    PreconditionChebyshev<Portable::LaplaceOperatorBase<dim, double>,
                          VectorTypeMG>;

private:
  void
  setup_grid();

  void
  create_coarse_triangulations();

  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  setup_mg_transfers();

  void
  setup_smoothers();

  void
  compute_rhs();

  void
  apply_smoother(const unsigned int  level,
                 VectorTypeMG       &dst,
                 const VectorTypeMG &src,
                 const unsigned int  n_smoothing_steps);

  void
  solve();


  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints_fine;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::set<types::boundary_id> dirichlet_boundary_ids;

  // std::unique_ptr<Portable::LaplaceOperator<dim, fe_degree, double>>
  //   system_matrix;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_device;

  std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_triangulations;

  MGLevelObject<DoFHandler<dim>>           level_dof_handlers;
  MGLevelObject<AffineConstraints<double>> level_constraints;

  MGLevelObject<std::unique_ptr<FE_Q<dim>>> p_level_fes;

  MGLevelObject<std::unique_ptr<Portable::LaplaceOperatorBase<dim, double>>>
    level_matrices;

  MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, double>>>
    mg_transfers;



  MGLevelObject<SmootherType> mg_smoothers;

  const unsigned int refinement_cycles;

  const bool overlap_communication_computation;

  ConditionalOStream pcout;

  struct LaplaceOperatorRunner
  {
    const unsigned int              level;
    DoFHandler<dim>                &dof_handler;
    AffineConstraints<double>      &constraints;
    bool                            overlap_communication_computation;
    LaplaceProblem<dim, fe_degree> &parent_problem;

    template <unsigned int degree>
    void
    run()
    {
      parent_problem.level_matrices[level] =
        std::make_unique<Portable::LaplaceOperator<dim, degree, double>>(
          dof_handler, constraints, overlap_communication_computation);
    }
  };

  struct PolynomialTransferRunner
  {
    const unsigned int                       level;
    const Portable::MatrixFree<dim, double> &mf_coarse;
    const Portable::MatrixFree<dim, double> &mf_fine;
    AffineConstraints<double>               &constraints_coarse;
    AffineConstraints<double>               &constraints_fine;

    LaplaceProblem<dim, fe_degree> &parent_problem;

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

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(
  const unsigned int refinement_cycles,
  const bool         overlap_communication_computation)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  // , dirichlet_boundary_ids({0})
  , refinement_cycles(refinement_cycles)
  , overlap_communication_computation(overlap_communication_computation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
  dirichlet_boundary_ids.insert(0);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_grid()
{
  unsigned int       n_refine  = refinement_cycles / 3;
  const unsigned int remainder = refinement_cycles % 3;
  Point<dim>         p1;
  for (unsigned int d = 0; d < dim; ++d)
    p1[d] = -1;
  Point<dim> p2;
  for (unsigned int d = 0; d < remainder; ++d)
    p2[d] = 2.8;
  for (unsigned int d = remainder; d < dim; ++d)
    p2[d] = 0.9;
  std::vector<unsigned int> subdivisions(dim, 1);
  for (unsigned int d = 0; d < remainder; ++d)
    subdivisions[d] = 2;
  // const unsigned int base_refine = (1 << n_refine);
  // projected_size                 = 1;
  // for (unsigned int d = 0; d < dim; ++d)
  //   projected_size *= base_refine * subdivisions[d] * degree_finite_element +
  //   1;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            p1,
                                            p2);

  triangulation.refine_global(n_refine);
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_coarse_triangulations()
{
  coarse_triangulations =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      triangulation,
      RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(16));
}


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

  std::vector<unsigned int> p_levels({fe.degree});

  while (p_levels.back() > 1)
    p_levels.push_back(std::max(p_levels.back() - 1, 1u));

  p_level_fes.resize(0, p_levels.size() - 1);

  for (unsigned int level = 0; level < p_levels.size(); ++level)
    p_level_fes[level] =
      std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

  level_dof_handlers.resize(0,
                            coarse_triangulations.size() - 1 +
                              p_level_fes.max_level());
  level_constraints.resize(0, level_dof_handlers.max_level());

  for (unsigned int level = level_dof_handlers.min_level();
       level <= level_dof_handlers.max_level();
       ++level)
    {
      DoFHandler<dim> &dof_h = level_dof_handlers[level];

      dof_h.reinit(
        *coarse_triangulations[std::min(level,
                                        triangulation.n_global_levels() - 1)]);

      if (level < coarse_triangulations.size())
        dof_h.distribute_dofs(*p_level_fes[0]);
      else
        dof_h.distribute_dofs(
          *p_level_fes[level + 1 - coarse_triangulations.size()]);

      IndexSet level_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_h);

      AffineConstraints<double> &constraints = level_constraints[level];

      constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_h, constraints);

      VectorTools::interpolate_boundary_values(dof_h,
                                               dirichlet_boundary_functions,
                                               constraints);
      constraints.close();
    }
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  constraints_fine.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints_fine.copy_from(level_constraints.back());

  // system_matrix.reset(new Portable::LaplaceOperator<dim, fe_degree, double>(
  //   dof_handler, constraints_fine, overlap_communication_computation));

  // system_matrix->initialize_dof_vector(solution_device);
  // system_rhs_device.reinit(solution_device);
  // ghost_solution_host.reinit(locally_owned_dofs,
  //                            locally_relevant_dofs,
  //                            mpi_communicator);

  level_matrices.resize(0, level_dof_handlers.max_level());
  for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
    {
      if (level < coarse_triangulations.size())
        level_matrices[level] =
          std::make_unique<Portable::LaplaceOperator<dim, 1, double>>(
            level_dof_handlers[level],
            level_constraints[level],
            overlap_communication_computation);

      else
        {
          LaplaceOperatorRunner runner{level,
                                       level_dof_handlers[level],
                                       level_constraints[level],
                                       overlap_communication_computation,
                                       *this};

          bool success = Portable::OperatorDispatchFactory::dispatch(
            p_level_fes[level + 1 - coarse_triangulations.size()]->degree,
            runner);

          Assert(
            success,
            ExcMessage(
              "Failed to find a matching polynomial degree in dispatcher."));
        }
    }


  const auto &system_matrix = *level_matrices.back();
  system_matrix.initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_transfers()
{
  mg_transfers.resize(level_matrices.min_level(), level_matrices.max_level());

  for (unsigned int level = level_matrices.min_level() + 1;
       level <= level_matrices.max_level();
       ++level)
    {
      if (level < coarse_triangulations.size())
        {
          mg_transfers[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          mg_transfers[level]->reinit(
            level_matrices[level - 1]->get_matrix_free(),
            level_matrices[level]->get_matrix_free(),
            level_constraints[level - 1],
            level_constraints[level]);
        }
      else
        {
          const unsigned int p_coarse =
            p_level_fes[level - coarse_triangulations.size()]->degree;
          const unsigned int p_fine =
            p_level_fes[level + 1 - coarse_triangulations.size()]->degree;

          PolynomialTransferRunner runner{
            level,
            level_matrices[level - 1]->get_matrix_free(),
            level_matrices[level]->get_matrix_free(),
            level_constraints[level - 1],
            level_constraints[level],
            *this};


          bool success =
            Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse,
                                                                  p_fine,
                                                                  runner);

          Assert(success,
                 ExcMessage("Failed to find a matching polynomial degree "
                            "pair in transfer dispatcher."));
        }
    }
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_smoothers()
{
  mg_smoothers.resize(level_matrices.min_level(), level_matrices.max_level());

  for (unsigned int level = level_matrices.min_level();
       level <= level_matrices.max_level();
       ++level)
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
          smoother_data.smoothing_range     = 1e-3;
          smoother_data.degree              = numbers::invalid_unsigned_int;
          smoother_data.eig_cg_n_iterations = level_matrices[0]->m();
        }

      level_matrices[level]->compute_diagonal();
      smoother_data.preconditioner =
        level_matrices[level]->get_matrix_diagonal_inverse();

      mg_smoothers[level].initialize(*level_matrices[level], smoother_data);
    }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_rhs()
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
          constraints_fine.distribute_local_to_global(cell_rhs,
                                                      local_dof_indices,
                                                      system_rhs_host);
        }
    }

  system_rhs_host.compress(VectorOperation::add);
  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  system_rhs_device.import_elements(rw_vector, VectorOperation::insert);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::apply_smoother(
  const unsigned int  level,
  VectorTypeMG       &u,
  const VectorTypeMG &rhs,
  const unsigned int  n_smoothing_steps)
{
  if (level == 0)
    mg_smoothers[0].vmult(u, rhs);
  else
    for (unsigned int step = 0; step < n_smoothing_steps; ++step)
      {
        level_matrices[level]->vmult(u, rhs);
        u.sadd(-1., rhs);

        mg_smoothers[level].vmult(u, rhs);

        u += rhs;
      }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve()
{
  const auto &system_matrix = *level_matrices.back();

  Portable::VCycleMultigrid<dim, double, Portable::MGTransferBase<dim, double>>
    mg_preconditioner(level_matrices, mg_transfers, mg_smoothers, 2, 2);

  SolverControl solver_control(system_rhs_device.size(),
                               1e-12 * system_rhs_device.l2_norm());

  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  solution_device = 0;
  cg.solve(system_matrix,
           solution_device,
           system_rhs_device,
           mg_preconditioner);

  // LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  // rw_vector.import_elements(solution_device, VectorOperation::insert);
  // ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

  // constraints_fine.distribute(ghost_solution_host);

  // ghost_solution_host.update_ghost_values();
}

template <int dim, int fe_degree>
Measurement
LaplaceProblem<dim, fe_degree>::run()
{
  std::map<std::string, dealii::Timer> timer;

  timer["setup_grid"].start();
  setup_grid();
  timer["setup_grid"].stop();


  timer["setup_coarse_grids"].start();
  create_coarse_triangulations();
  timer["setup_coarse_grids"].stop();


  timer["setup_dofs"].start();
  setup_dofs();
  timer["setup_dofs"].stop();

  pcout << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  Kokkos::fence();
  timer["setup_matrix_free"].start();
  setup_matrix_free();
  Kokkos::fence();
  timer["setup_matrix_free"].stop();

  Kokkos::fence();
  timer["setup_mg_transfers"].start();
  setup_mg_transfers();
  Kokkos::fence();
  timer["setup_mg_transfers"].stop();

  Kokkos::fence();
  timer["setup_smoothers"].start();
  setup_smoothers();
  Kokkos::fence();
  timer["setup_smoothers"].stop();

  Kokkos::fence();
  timer["compute_rhs"].start();
  compute_rhs();
  Kokkos::fence();
  timer["compute_rhs"].stop();

  Kokkos::fence();
  timer["solve0"].start();
  solve();
  Kokkos::fence();
  timer["solve0"].stop();

  Kokkos::fence();
  timer["solve1"].start();
  solve();
  Kokkos::fence();
  timer["solve1"].stop();

  Kokkos::fence();
  timer["solve2"].start();
  solve();
  Kokkos::fence();
  timer["solve2"].stop();

  const unsigned int n_repeat = 50;
  level_matrices.back()->vmult(system_rhs_device, solution_device);

  Kokkos::fence();
  timer["matvec_double"].start();
  for (unsigned int repeat = 0; repeat < n_repeat; ++repeat)
    level_matrices.back()->vmult(system_rhs_device, solution_device);
  Kokkos::fence();
  timer["matvec_double"].stop();


  const unsigned int n_levels = level_matrices.max_level() + 1;


  // Prepare level-specific vectors
  MGLevelObject<VectorTypeMG> mg_sol, mg_rhs;
  mg_sol.resize(0, n_levels - 1);
  mg_rhs.resize(0, n_levels - 1);
  for (unsigned int l = 0; l < n_levels; ++l)
    {
      level_matrices[l]->initialize_dof_vector(mg_sol[l]);
      mg_rhs[l].reinit(mg_sol[l]);
    }

  for (unsigned int l = 0; l < n_levels; ++l)
    {
      level_matrices[l]->vmult(mg_sol[l], mg_rhs[l]);

      Kokkos::fence();
      timer["matvec_L_" + std::to_string(l)].start();
      for (unsigned int r = 0; r < n_repeat; ++r)
        level_matrices[l]->vmult(mg_sol[l], mg_rhs[l]);
      Kokkos::fence();
      timer["matvec_L_" + std::to_string(l)].stop();

      apply_smoother(l, mg_sol[l], mg_rhs[l], 2);


      unsigned int rep = l == 0 ? 2 : n_repeat;

      Kokkos::fence();
      timer["apply_smoother_L" + std::to_string(l)].start();
      for (unsigned int r = 0; r < rep; ++r)
        apply_smoother(l, mg_sol[l], mg_rhs[l], 2);
      Kokkos::fence();
      timer["apply_smoother_L" + std::to_string(l)].stop();
    }

    double total_prolongate_time = 0.0;
    double total_restrict_time = 0.0;

  for (unsigned int l = 1; l < n_levels; ++l)
    {
      mg_transfers[l]->prolongate_and_add(mg_sol[l], mg_sol[l - 1]);

      Kokkos::fence();
      timer["prolongate_L_" + std::to_string(l)].start();
      for (unsigned int r = 0; r < n_repeat; ++r)
        mg_transfers[l]->prolongate_and_add(mg_sol[l], mg_sol[l - 1]);
      Kokkos::fence();
      timer["prolongate_L_" + std::to_string(l)].stop();

      total_prolongate_time +=
        timer["prolongate_L_" + std::to_string(l)].wall_time();

      mg_transfers[l]->restrict_and_add(mg_rhs[l - 1], mg_rhs[l]);


      Kokkos::fence();
      timer["restrict_L_" + std::to_string(l)].start();
      for (unsigned int r = 0; r < n_repeat; ++r)
        mg_transfers[l]->restrict_and_add(mg_rhs[l - 1], mg_rhs[l]);
      Kokkos::fence();
      timer["restrict_L_" + std::to_string(l)].stop();

      total_restrict_time +=
        timer["restrict_L_" + std::to_string(l)].wall_time();
    }

  // --- Package for Driver (Must match names in describe_measurements) ---
  std::vector<double> results;
  results.push_back(timer["setup_grid"].wall_time());
  results.push_back(timer["setup_coarse_grids"].wall_time());
  results.push_back(timer["setup_dofs"].wall_time());
  results.push_back(timer["setup_matrix_free"].wall_time());
  results.push_back(timer["setup_smoothers"].wall_time());
  results.push_back(timer["setup_mg_transfers"].wall_time());
  results.push_back(timer["compute_rhs"].wall_time());
  results.push_back(timer["solve0"].wall_time());
  results.push_back(timer["solve1"].wall_time());
  results.push_back(timer["solve2"].wall_time());
  results.push_back(timer["matvec_double"].wall_time());
  results.push_back(total_prolongate_time);
  results.push_back(total_restrict_time);

  for (unsigned int l = 0; l < max_levels; ++l)
    {
      results.push_back(l < n_levels ?
                          timer["matvec_L_" + std::to_string(l)].wall_time() :
                          0.0);
    }

  for (unsigned int l = 0; l < max_levels; ++l)
    {
      results.push_back(
        l < n_levels ?
          timer["apply_smoother_L" + std::to_string(l)].wall_time() :
          0.0);
    }

  for (unsigned int l = 1; l < max_levels; ++l)
    {
      results.push_back(l < n_levels ?
                          timer["restrict_L_" + std::to_string(l)].wall_time() :
                          0.0);
    }

  for (unsigned int l = 1; l < max_levels; ++l)
    {
      results.push_back(l < n_levels ?
                          timer["restrict_L_" + std::to_string(l)].wall_time() :
                          0.0);
    }

  debug_output << std::endl;

  return Measurement(results);
}


std::tuple<Metric, unsigned int, std::vector<std::string>>
describe_measurements()
{
  std::vector<std::string> names = {"setup_grid",
                                    "setup_coarse_grids",
                                    "setup_dofs",
                                    "setup_matrix_free",
                                    "setup_smoothers",
                                    "setup_mg_transfers",
                                    "compute_rhs",
                                    "solve0",
                                    "solve1",
                                    "solve2",
                                    "matvec_double",
                                    "prolongate_tot",
                                    "restrict_tot"};



  for (unsigned int l = 0; l < max_levels; ++l)
    {
      names.push_back("matvec_L" + std::to_string(l));
    }
    
for (unsigned int l = 0; l < max_levels; ++l)
    {
      names.push_back("apply_smoother_L" + std::to_string(l));
    }
    

  for (unsigned int l = 1; l < max_levels; ++l)
    {
      names.push_back("prolongate_L_" + std::to_string(l));
    }
  for (unsigned int l = 1; l < max_levels; ++l)
    {
      names.push_back("restrict_L_" + std::to_string(l));
    }

  return {Metric::timing, 5, names};
}


Measurement
perform_single_measurement(const unsigned int refinement_cycles)
{
  // run in 3d with degree 4
  return LaplaceProblem<3, 4>(refinement_cycles, false).run();
}
