#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
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
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_laplace_operator.h"
#include "operators/portable_laplace_operator_bk3.h"
#include "operators/portable_laplace_operator.h"

#include "portable_multigrid_solver.h"


namespace multigrid
{
  using namespace dealii;

  // Here at the top of the file, we collect the main global settings. The
  // degree can be passed as the first argument to the program, but due to the
  // templates we need to precompile the respective programs. Here we specify
  // a minimum and maximum degree we want to support. Degrees outside this
  // range will not do any work.
  const unsigned int dimension      = 3;
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 4;
  const double       wave_number    = 3.;
  const bool         deform_grid    = false;

  // We also select a mixed-precision approach as default. You can
  // independently change the number type for the outer iteration via
  // full_number and the number type for the multigrid v-cycle.
  using vcycle_number = float;
  using full_number   = double;



  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const bool overlap_communication_computation);

    void
    run(const std::size_t  min_size,
        const std::size_t  max_size,
        const unsigned int n_pre_smooth,
        const unsigned int n_post_smooth,
        const bool         use_doubling_mesh);

    // void
    // run();

    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>;

    using SmootherType =
      PreconditionChebyshev<Portable::LaplaceOperatorBase<dim, vcycle_number>, VectorTypeMG>;

    void
    test();

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
    setup_smoothers(const unsigned int n_pre_smooth, const unsigned int n_post_smooth);

    void
    compute_rhs();

    void
    apply_smoother(const unsigned int  level,
                   VectorTypeMG       &dst,
                   const VectorTypeMG &src,
                   const unsigned int  n_smoothing_steps);

    void
    solve(const unsigned int n_pre_smooth, const unsigned int n_post_smooth);

    void
    matvec_ghost_timing();

    void
    vmult_comparison_timing();

    void
    prolong_restrict_comparison_timing();


    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    std::set<types::boundary_id> dirichlet_boundary_ids;

    LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Host>  ghost_solution_host;
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Default> solution_device;
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Default> system_rhs_device;

    std::vector<std::shared_ptr<const Triangulation<dim>>> coarse_triangulations;

    MGLevelObject<DoFHandler<dim>>                  level_dof_handlers;
    MGLevelObject<AffineConstraints<vcycle_number>> level_constraints;

    MGLevelObject<std::unique_ptr<FE_Q<dim>>> p_level_fes;

    MGLevelObject<std::unique_ptr<Portable::LaplaceOperatorBase<dim, vcycle_number>>>
      level_matrices;

    MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, vcycle_number>>> mg_transfers;

    MGLevelObject<SmootherType> mg_smoothers;

    AffineConstraints<full_number> fine_level_constraints;

    std::unique_ptr<Portable::LaplaceOperatorBase<dim, full_number>> fine_level_matrix;

    const unsigned int refinement_cycles = 10;

    const bool overlap_communication_computation;

    double setup_time;

    ConvergenceTable convergence_table;

    ConvergenceTable ghost_timing_table;

    ConvergenceTable vmult_comparison_table;

    ConditionalOStream pcout;

    ConditionalOStream time_details;


    struct LaplaceOperatorRunner
    {
      const unsigned int                level;
      DoFHandler<dim>                  &dof_handler;
      AffineConstraints<vcycle_number> &constraints;
      bool                              overlap_communication_computation;
      LaplaceProblem<dim, fe_degree>   &parent_problem;

      template <unsigned int degree>
      void
      run()
      {
        parent_problem.level_matrices[level] =
          std::make_unique<Portable::LaplaceOperatorBK3<dim, degree, vcycle_number>>(
            dof_handler, constraints, overlap_communication_computation);
      }
    };

    struct PolynomialTransferRunner
    {
      const unsigned int                              level;
      const Portable::MatrixFree<dim, vcycle_number> &mf_coarse;
      const Portable::MatrixFree<dim, vcycle_number> &mf_fine;
      AffineConstraints<vcycle_number>               &constraints_coarse;
      AffineConstraints<vcycle_number>               &constraints_fine;

      LaplaceProblem<dim, fe_degree> &parent_problem;

      template <unsigned int degree_coarse, unsigned int degree_fine>
      void
      run()
      {
        parent_problem.mg_transfers[level] = std::make_unique<
          Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, vcycle_number>>();

        parent_problem.mg_transfers[level]->reinit(mf_coarse,
                                                   mf_fine,
                                                   constraints_coarse,
                                                   constraints_fine);
      }
    };
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem(const bool overlap_communication_computation)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , overlap_communication_computation(overlap_communication_computation)
    , setup_time(0.)

    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , time_details(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)

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
    //   projected_size *= base_refine * subdivisions[d] * degree_finite_element
    //   + 1;
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    triangulation.refine_global(n_refine);
  }



  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::create_coarse_triangulations()
  {
    setup_time = 0;

    Timer time;

    coarse_triangulations = MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      triangulation, RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(16));

    setup_time += time.wall_time();

    time_details << "Coarse triangulations created  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_dofs()
  {
    Timer time;

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
          << ((int)std::pow(dof_handler.n_dofs() * 1.0000001, 1. / dim) - 1) / fe.degree << " x "
          << fe.degree << " + 1)^" << dim << std::endl;

    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


    Functions::ZeroFunction<dim>                        homogeneous_dirichlet_bc;
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

    std::vector<unsigned int> p_levels({fe.degree});

    while (p_levels.back() > 1)
      p_levels.push_back(std::max(p_levels.back() - 2, 1u));

    for (const auto p : p_levels)
      pcout << p << "  ";
    pcout << std::endl;

    p_level_fes.resize(0, p_levels.size() - 1);

    for (unsigned int level = 0; level < p_levels.size(); ++level)
      {
        p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);
      }

    level_dof_handlers.resize(0, coarse_triangulations.size() - 1 + p_level_fes.max_level());
    level_constraints.resize(0, level_dof_handlers.max_level());

    for (unsigned int level = level_dof_handlers.min_level();
         level <= level_dof_handlers.max_level();
         ++level)
      {
        DoFHandler<dim> &dof_h = level_dof_handlers[level];

        dof_h.reinit(*coarse_triangulations[std::min(level, triangulation.n_global_levels() - 1)]);

        if (level < coarse_triangulations.size())
          dof_h.distribute_dofs(*p_level_fes[0]);
        else
          dof_h.distribute_dofs(*p_level_fes[level + 1 - coarse_triangulations.size()]);

        IndexSet level_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_h);

        fine_level_constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);

        DoFTools::make_hanging_node_constraints(dof_h, fine_level_constraints);

        VectorTools::interpolate_boundary_values(dof_h,
                                                 dirichlet_boundary_functions,
                                                 fine_level_constraints);
        fine_level_constraints.close();

        // because we might be initializing with float numbers, we must first
        // create a double-precision constraints object and work from there
        AffineConstraints<vcycle_number> &constraints = level_constraints[level];
        constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);
        constraints.copy_from(fine_level_constraints);
        constraints.close();
      }

    setup_time += time.wall_time();

    time_details << "DoFs and constraint setup  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_matrix_free()
  {
    Kokkos::fence();

    Timer time;
    level_matrices.resize(0, level_dof_handlers.max_level());

    for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
      {
        if (level < coarse_triangulations.size())
          {
            level_matrices[level] =
              std::make_unique<Portable::LaplaceOperatorBK3<dim, 1, vcycle_number>>(
                level_dof_handlers[level],
                level_constraints[level],
                overlap_communication_computation);
          }

        else
          {
            LaplaceOperatorRunner runner{level,
                                         level_dof_handlers[level],
                                         level_constraints[level],
                                         overlap_communication_computation,
                                         *this};

            bool success = Portable::OperatorDispatchFactory::dispatch(
              p_level_fes[level + 1 - coarse_triangulations.size()]->degree, runner);

            Assert(success,
                   ExcMessage("Failed to find a matching polynomial degree in dispatcher."));
          }
      }



    if constexpr (std::is_same_v<full_number, vcycle_number>)
      {
        const auto &system_matrix = *level_matrices.back();
        system_matrix.initialize_dof_vector(solution_device);
      }
    else
      {
        fine_level_matrix =
          std::make_unique<Portable::LaplaceOperatorBK3<dim, fe_degree, full_number>>(
            level_dof_handlers.back(), fine_level_constraints, overlap_communication_computation);

        fine_level_matrix->initialize_dof_vector(solution_device);
      }
    system_rhs_device.reinit(solution_device);
    ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    Kokkos::fence();

    setup_time += time.wall_time();

    time_details << "Setup matrices   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_mg_transfers()
  {
    Kokkos::fence();
    Timer time;
    mg_transfers.resize(level_matrices.min_level(), level_matrices.max_level());

    for (unsigned int level = level_matrices.min_level() + 1; level <= level_matrices.max_level();
         ++level)
      {
        if (level < coarse_triangulations.size())
          {
            mg_transfers[level] =
              std::make_unique<Portable::GeometricTransfer<dim, 1, vcycle_number>>();
            mg_transfers[level]->reinit(level_matrices[level - 1]->get_matrix_free(),
                                        level_matrices[level]->get_matrix_free(),
                                        level_constraints[level - 1],
                                        level_constraints[level]);
          }
        else
          {
            const unsigned int p_coarse = p_level_fes[level - coarse_triangulations.size()]->degree;
            const unsigned int p_fine =
              p_level_fes[level + 1 - coarse_triangulations.size()]->degree;

            PolynomialTransferRunner runner{level,
                                            level_matrices[level - 1]->get_matrix_free(),
                                            level_matrices[level]->get_matrix_free(),
                                            level_constraints[level - 1],
                                            level_constraints[level],
                                            *this};

            bool success =
              Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse, p_fine, runner);

            Assert(success,
                   ExcMessage("Failed to find a matching polynomial degree "
                              "pair in transfer dispatcher."));
          }
      }
    Kokkos::fence();

    pcout << std::endl;

    setup_time += time.wall_time();

    time_details << "Setup transfers   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }



  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_smoothers(const unsigned int n_pre_smooth,
                                                  const unsigned int n_post_smooth)
  {
    Assert(n_pre_smooth == n_post_smooth,
           ExcNotImplemented("Change of pre- and post-smoother degree "
                             "currently not possible with deal.II"));

    Kokkos::fence();
    Timer time;
    mg_smoothers.resize(level_matrices.min_level(), level_matrices.max_level());

    for (unsigned int level = level_matrices.min_level(); level <= level_matrices.max_level();
         ++level)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (level > 0)
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = n_pre_smooth;
            smoother_data.eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = level_matrices[0]->m();
          }

        level_matrices[level]->compute_diagonal();
        smoother_data.preconditioner = level_matrices[level]->get_matrix_diagonal_inverse();

        mg_smoothers[level].initialize(*level_matrices[level], smoother_data);
      }

    Kokkos::fence();
    setup_time += time.wall_time();

    time_details << "Setup smoothers   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::compute_rhs()
  {
    Timer time;

    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host> system_rhs_host(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<full_number> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_rhs = 0;

            fe_values.reinit(cell);

            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

            cell->get_dof_indices(local_dof_indices);
            level_constraints.back().distribute_local_to_global(cell_rhs,
                                                                local_dof_indices,
                                                                system_rhs_host);
          }
      }

    system_rhs_host.compress(VectorOperation::add);
    LinearAlgebra::ReadWriteVector<full_number> rw_vector(locally_owned_dofs);

    rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
    system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

    setup_time += time.wall_time();

    time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve(const unsigned int n_pre_smooth,
                                        const unsigned int n_post_smooth)
  {
    multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType> *solver;

    if constexpr (std::is_same_v<full_number, vcycle_number>)
      solver =
        new multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>(
          level_matrices.back(),
          level_dof_handlers,
          level_constraints,
          mg_transfers,
          level_matrices,
          mg_smoothers,
          system_rhs_device,
          n_pre_smooth,
          n_post_smooth);
    else
      solver =
        new multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>(
          fine_level_matrix,
          level_dof_handlers,
          level_constraints,
          mg_transfers,
          level_matrices,
          mg_smoothers,
          system_rhs_device,
          n_pre_smooth,
          n_post_smooth);



    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

    pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index << "] " << memory.avg
          << " " << memory.max << " [p" << memory.max_index << "]" << std::endl;

    double                          time_cg = 1e10;
    std::pair<unsigned int, double> cg_details;
    for (unsigned int i = 0; i < 10; ++i)
      {
        Kokkos::fence();
        time.restart();
        cg_details = solver->solve_cg();
        Kokkos::fence();
        time_cg = std::min(time.wall_time(), time_cg);
        pcout << "Time solve CG              " << time.wall_time() << "\n";
      }

    solver->print_wall_times();

    double best_mv = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          solver->do_matvec();
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv = std::min(best_mv, stat.max);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                    << " " << stat.max << " [p" << stat.max_index << "]"
                    << " DoFs/s: " << dof_handler.n_dofs() / stat.max << std::endl;
      }

    double best_mvs = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

        Kokkos::fence();
        time.restart();

        for (unsigned int i = 0; i < n_mv; ++i)
          solver->do_matvec_smoother();

        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mvs = std::min(best_mvs, stat.max);
      }

    std::vector<double> prolongate_per_level(level_matrices.max_level());
    std::vector<double> restrict_per_level(level_matrices.max_level());

    for (unsigned int level = 1; level <= level_matrices.max_level(); ++level)
      {
        prolongate_per_level[level - 1] = 1e10;
        restrict_per_level[level - 1]   = 1e10;

        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> vec_fine,
          vec_coarse;

        level_matrices[level - 1]->initialize_dof_vector(vec_coarse);
        level_matrices[level]->initialize_dof_vector(vec_fine);

        for (unsigned int i = 0; i < 5; ++i)
          {
            const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

            Kokkos::fence();
            time.restart();

            for (unsigned int i = 0; i < n_mv; ++i)
              mg_transfers[level]->prolongate_and_add(vec_fine, vec_coarse);

            Kokkos::fence();

            Utilities::MPI::MinMaxAvg stat =
              Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

            prolongate_per_level[level - 1] = std::min(prolongate_per_level[level - 1], stat.max);
          }


        for (unsigned int i = 0; i < 5; ++i)
          {
            const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

            Kokkos::fence();
            time.restart();

            for (unsigned int i = 0; i < n_mv; ++i)
              mg_transfers[level]->restrict_and_add(vec_coarse, vec_fine);

            Kokkos::fence();

            Utilities::MPI::MinMaxAvg stat =
              Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

            restrict_per_level[level - 1] = std::min(restrict_per_level[level - 1], stat.max);
          }
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Best timings for ndof = " << dof_handler.n_dofs() << "   mv " << best_mv
                << "    mv smooth " << best_mvs << "   cg-mg " << time_cg << std::endl;


    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("mv_outer", best_mv);
    convergence_table.add_value("mv_inner", best_mvs);
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", cg_details.first);
    convergence_table.add_value("cg_reduction", cg_details.second);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      for (unsigned int level = 1; level <= level_matrices.max_level(); level++)
        {
          // convergence_table.add_value("restrict_L_" + std::to_string(level),
          //                             restrict_per_level[level - 1]);
          // convergence_table.add_value("prolong_L_" + std::to_string(level),
          //                             restrict_per_level[level - 1]);

          std::cout << "Best timings for ndof = " << level_dof_handlers[level].n_dofs()
                    << "   on level " << level
                    << "|  restriction = " << restrict_per_level[level - 1]
                    << "   prolongation  =  " << prolongate_per_level[level - 1] << std::endl;
        }
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::matvec_ghost_timing()
  {
    const bool ghost_exchange_on = true;
    const bool computation_on    = true;

    MGLevelObject<LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>>
      dummy_solution(0, level_matrices.max_level()), dummy_rhs(0, level_matrices.max_level());

    for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
      {
        level_matrices[level]->initialize_dof_vector(dummy_solution[level]);

        level_matrices[level]->initialize_dof_vector(dummy_rhs[level]);
      }

    Timer time;

    double best_mv_both    = 1e10;
    double best_only_ghost = 1e10;
    double best_only_comp  = 1e10;

    for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
      {
        best_mv_both    = 1e10;
        best_only_ghost = 1e10;
        best_only_comp  = 1e10;

        for (unsigned int i = 0; i < 5; ++i)
          // for (unsigned int i = 0; i < 1; ++i)
          {
            const unsigned int n_mv =
              dof_handler.n_dofs() < 10000000 ? 200 : 50;

            // const unsigned int n_mv = 1;

            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                level_matrices[level]->vmult_dummy(dummy_solution[level],
                                                   dummy_rhs[level],
                                                   ghost_exchange_on,
                                                   computation_on);
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

              best_mv_both = std::min(best_mv_both, stat.max);
            }
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                level_matrices[level]->vmult_dummy(dummy_solution[level],
                                                   dummy_rhs[level],
                                                   ghost_exchange_on,
                                                   !computation_on);
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

              best_only_ghost = std::min(best_only_ghost, stat.max);
            }

            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                level_matrices[level]->vmult_dummy(dummy_solution[level],
                                                   dummy_rhs[level],
                                                   !ghost_exchange_on,
                                                   computation_on);
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

              best_only_comp = std::min(best_only_comp, stat.max);
            }
          }

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Best timings for ndof = " << level_dof_handlers[level].n_dofs()
                    << "   on level " << level << "|  ghost & compute =  " << best_mv_both
                    << "   ghost only      =  " << best_only_ghost
                    << "   compute only    =  " << best_only_comp

                    << std::endl;
      }

    ghost_timing_table.add_value("cells", triangulation.n_global_active_cells());
    ghost_timing_table.add_value("dofs", dof_handler.n_dofs());
    ghost_timing_table.add_value("mv_ghost_and_compute", best_mv_both);
    ghost_timing_table.add_value("mv_compute_only", best_only_comp);
    ghost_timing_table.add_value("mv_ghost_only", best_only_ghost);
  }

  // Compares LaplaceOperatorBK3::vmult() (BK3::Parallel::KokkosKernel, the
  // original hand-written kernel) against vmult_new() (BK3Custom::Parallel::
  // KokkosKernel, composed from the generic building blocks in
  // kernels/portable_tensor_product_kernels.h) at every MG level, to check
  // the refactor in kernels/portable_tensor_product_kernels.h hasn't
  // regressed performance. Same best-of-5 timing methodology and
  // per-level/per-cycle tabulation convention as
  // matvec_ghost_timing() above -- per-level numbers are printed directly,
  // only the finest level's numbers are added to the table (one row per
  // mesh-size cycle in run()).
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::vmult_comparison_timing()
  {
    MGLevelObject<LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>>
      dummy_solution(0, level_matrices.max_level()), dummy_rhs(0, level_matrices.max_level());

    for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
      {
        level_matrices[level]->initialize_dof_vector(dummy_solution[level]);
        level_matrices[level]->initialize_dof_vector(dummy_rhs[level]);
      }

    Timer time;

    double best_vmult     = 1e10;
    double best_vmult_new = 1e10;

    // Per-level speedup, gathered into its own fresh table each cycle
    // rather than extra per-level columns on vmult_comparison_table below:
    // the number of MG levels varies across mesh-size cycles (run()'s
    // sizes[] loop), and ConvergenceTable requires every column to have the
    // same number of rows across the whole table's lifetime -- a column
    // like "speedup_L5" would only get a value on cycles with >= 6 levels
    // and break write_text() on every earlier, shallower cycle. A local
    // table with one row per level, written out immediately after this
    // cycle's per-level loop, sidesteps that entirely.
    ConvergenceTable level_speedup_table;

    for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
      {
        best_vmult     = 1e10;
        best_vmult_new = 1e10;

        for (unsigned int i = 0; i < 5; ++i)
          {
            const unsigned int n_mv = level_dof_handlers[level].n_dofs() < 10000000 ? 200 : 50;

            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                level_matrices[level]->vmult(dummy_solution[level], dummy_rhs[level]);
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

              best_vmult = std::min(best_vmult, stat.max);
            }
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                level_matrices[level]->vmult_new(dummy_solution[level], dummy_rhs[level]);
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

              best_vmult_new = std::min(best_vmult_new, stat.max);
            }
          }

        const double speedup = best_vmult / best_vmult_new;

        // Correctness check on a random vector, independent of the timing
        // loop above (deterministic, run once per level) -- same
        // random-vector-compare idea as
        // correctness_tests/check_correctness_laplace_operator_batched/
        // program.cc, but using the vectors' own l2_norm() (an MPI-collective
        // reduction, so every rank already has the same global value; only
        // the printing/tabulation below is rank-0-gated) rather than a
        // max-abs-diff over locally_owned_elements().
        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> src_random, dst_vmult,
          dst_vmult_new;
        level_matrices[level]->initialize_dof_vector(src_random);
        level_matrices[level]->initialize_dof_vector(dst_vmult);
        level_matrices[level]->initialize_dof_vector(dst_vmult_new);

        {
          std::mt19937                           gen(42 + level);
          std::uniform_real_distribution<double> dist(-1., 1.);

          LinearAlgebra::ReadWriteVector<vcycle_number> rw(src_random.locally_owned_elements());
          for (const auto idx : src_random.locally_owned_elements())
            rw(idx) = dist(gen);
          src_random.import_elements(rw, VectorOperation::insert);
        }

        level_matrices[level]->vmult(dst_vmult, src_random);
        level_matrices[level]->vmult_new(dst_vmult_new, src_random);

        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> err;
        err = dst_vmult;
        err -= dst_vmult_new;

        const double norm    = dst_vmult.l2_norm();
        const double abs_err = err.l2_norm();
        const double rel_err = abs_err / std::max(norm, 1e-30);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::cout << "Best vmult/vmult_new timings for ndof = " << level_dof_handlers[level].n_dofs()
                      << "   on level " << level << "|  vmult = " << best_vmult
                      << "   vmult_new = " << best_vmult_new << "   speedup = " << speedup << std::endl;
            std::cout << "  correctness (random vector) |  norm = " << norm << "   abs_err = " << abs_err
                      << "   rel_err = " << rel_err << std::endl;

            level_speedup_table.add_value("level", level);
            level_speedup_table.add_value("dofs", level_dof_handlers[level].n_dofs());
            level_speedup_table.add_value("vmult", best_vmult);
            level_speedup_table.add_value("vmult_new", best_vmult_new);
            level_speedup_table.add_value("speedup", speedup);
            level_speedup_table.add_value("norm", norm);
            level_speedup_table.add_value("abs_err", abs_err);
            level_speedup_table.add_value("rel_err", rel_err);
          }
      }

    vmult_comparison_table.add_value("cells", triangulation.n_global_active_cells());
    vmult_comparison_table.add_value("dofs", dof_handler.n_dofs());
    vmult_comparison_table.add_value("vmult", best_vmult);
    vmult_comparison_table.add_value("vmult_new", best_vmult_new);
    vmult_comparison_table.add_value("speedup", best_vmult / best_vmult_new);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        level_speedup_table.set_scientific("vmult", true);
        level_speedup_table.set_precision("vmult", 4);
        level_speedup_table.set_scientific("vmult_new", true);
        level_speedup_table.set_precision("vmult_new", 4);
        level_speedup_table.set_precision("speedup", 3);
        level_speedup_table.set_scientific("norm", true);
        level_speedup_table.set_precision("norm", 3);
        level_speedup_table.set_scientific("abs_err", true);
        level_speedup_table.set_precision("abs_err", 3);
        level_speedup_table.set_scientific("rel_err", true);
        level_speedup_table.set_precision("rel_err", 3);

        std::cout << std::endl << "Per-level vmult/vmult_new speedup and correctness:" << std::endl;
        level_speedup_table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

  // Best-of-5 prolongate_and_add()/restrict_and_add() (production
  // BK1::Parallel::KokkosProlongationBatchedKernel/
  // KokkosRestrictionBatchedKernel) vs prolongate_and_add_new()/
  // restrict_and_add_new() (the EvaluatorTensorProduct-based "Abstracted"
  // kernels) timings, plus a random-vector correctness check, per level,
  // same methodology as vmult_comparison_timing() above. Called directly
  // through the MGTransferBase pointer -- prolongate_and_add_new()/
  // restrict_and_add_new() are part of that virtual interface (base/
  // portable_mg_transfer_base.h), so no per-(degree_coarse, degree_fine)
  // dispatch or downcast is needed here, unlike setup_mg_transfers()'s own
  // PolynomialTransferRunner (which has to construct the concrete
  // Portable::PolynomialTransfer<...> object in the first place). Runs over
  // every level, both the PolynomialTransfer levels (p-multigrid) and the
  // GeometricTransfer ones below coarse_triangulations.size()
  // (h-multigrid) -- GeometricTransfer::prolongate_and_add_internal_new()/
  // restrict_and_add_internal_new() (portable_geometric_transfer.h) are a
  // real alternate-kernel implementation too, not a
  // DEAL_II_NOT_IMPLEMENTED() stub, since GeometricTransfer already calls
  // BK1::Parallel::KokkosProlongationBatchedKernel/
  // KokkosRestrictionBatchedKernel under the hood just like
  // PolynomialTransfer does.
  //
  // dst must be zeroed before each call since prolongate_and_add()/
  // restrict_and_add() *add* into dst rather than overwriting it.
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::prolong_restrict_comparison_timing()
  {
    Timer time;

    ConvergenceTable level_transfer_table;

    for (unsigned int level = 1; level <= level_matrices.max_level(); ++level)
      {
        auto &transfer = *mg_transfers[level];

        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> vec_coarse, vec_fine,
          dst_prolong_orig, dst_prolong_new, dst_restrict_orig, dst_restrict_new;

        level_matrices[level - 1]->initialize_dof_vector(vec_coarse);
        level_matrices[level]->initialize_dof_vector(vec_fine);
        level_matrices[level]->initialize_dof_vector(dst_prolong_orig);
        level_matrices[level]->initialize_dof_vector(dst_prolong_new);
        level_matrices[level - 1]->initialize_dof_vector(dst_restrict_orig); 
        level_matrices[level - 1]->initialize_dof_vector(dst_restrict_new);

        {
          std::mt19937                           gen(42 + level);
          std::uniform_real_distribution<double> dist(-1., 1.);

          LinearAlgebra::ReadWriteVector<vcycle_number> rw_coarse(vec_coarse.locally_owned_elements());
          for (const auto idx : vec_coarse.locally_owned_elements())
            rw_coarse(idx) = dist(gen);
          vec_coarse.import_elements(rw_coarse, VectorOperation::insert);

          LinearAlgebra::ReadWriteVector<vcycle_number> rw_fine(vec_fine.locally_owned_elements());
          for (const auto idx : vec_fine.locally_owned_elements())
            rw_fine(idx) = dist(gen);
          vec_fine.import_elements(rw_fine, VectorOperation::insert);
        }

        double best_prolong      = 1e10;
        double best_prolong_new  = 1e10;
        double best_restrict     = 1e10;
        double best_restrict_new = 1e10;

        const unsigned int n_mv = level_dof_handlers[level].n_dofs() < 10000000 ? 200 : 50;

        for (unsigned int i = 0; i < 5; ++i)
          {
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                {
                  dst_prolong_orig = 0.;
                  transfer.prolongate_and_add(dst_prolong_orig, vec_coarse);
                }
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
              best_prolong = std::min(best_prolong, stat.max);
            }
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                {
                  dst_prolong_new = 0.;
                  transfer.prolongate_and_add_new(dst_prolong_new, vec_coarse);
                }
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
              best_prolong_new = std::min(best_prolong_new, stat.max);
            }
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                {
                  dst_restrict_orig = 0.;
                  transfer.restrict_and_add(dst_restrict_orig, vec_fine);
                }
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
              best_restrict = std::min(best_restrict, stat.max);
            }
            {
              Kokkos::fence();
              time.restart();
              for (unsigned int i = 0; i < n_mv; ++i)
                {
                  dst_restrict_new = 0.;
                  transfer.restrict_and_add_new(dst_restrict_new, vec_fine);
                }
              Kokkos::fence();

              Utilities::MPI::MinMaxAvg stat =
                Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
              best_restrict_new = std::min(best_restrict_new, stat.max);
            }
          }

        // Correctness on a random vector, independent of the timing loop
        // above.
        dst_prolong_orig = 0.;
        dst_prolong_new  = 0.;
        transfer.prolongate_and_add(dst_prolong_orig, vec_coarse);
        transfer.prolongate_and_add_new(dst_prolong_new, vec_coarse);

        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> err_prolong;
        err_prolong = dst_prolong_orig;
        err_prolong -= dst_prolong_new;
        const double prolong_norm    = dst_prolong_orig.l2_norm();
        const double prolong_abs_err = err_prolong.l2_norm();
        const double prolong_rel_err = prolong_abs_err / std::max(prolong_norm, 1e-30);

        dst_restrict_orig = 0.;
        dst_restrict_new  = 0.;
        transfer.restrict_and_add(dst_restrict_orig, vec_fine);
        transfer.restrict_and_add_new(dst_restrict_new, vec_fine);

        LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default> err_restrict;
        err_restrict = dst_restrict_orig;
        err_restrict -= dst_restrict_new;
        const double restrict_norm    = dst_restrict_orig.l2_norm();
        const double restrict_abs_err = err_restrict.l2_norm();
        const double restrict_rel_err = restrict_abs_err / std::max(restrict_norm, 1e-30);

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            std::cout << "Best prolong/restrict timings for ndof_coarse = "
                      << level_dof_handlers[level - 1].n_dofs()
                      << ", ndof_fine = " << level_dof_handlers[level].n_dofs() << "   on level "
                      << level << "|  prolong = " << best_prolong
                      << "   prolong_new = " << best_prolong_new
                      << "   speedup = " << (best_prolong / best_prolong_new)
                      << "   restrict = " << best_restrict << "   restrict_new = " << best_restrict_new
                      << "   speedup = " << (best_restrict / best_restrict_new) << std::endl;

            level_transfer_table.add_value("level", level);
            level_transfer_table.add_value("dofs_coarse", level_dof_handlers[level - 1].n_dofs());
            level_transfer_table.add_value("dofs_fine", level_dof_handlers[level].n_dofs());
            level_transfer_table.add_value("prolong", best_prolong);
            level_transfer_table.add_value("prolong_new", best_prolong_new);
            level_transfer_table.add_value("prolong_speedup", best_prolong / best_prolong_new);
            level_transfer_table.add_value("prolong_rel_err", prolong_rel_err);
            level_transfer_table.add_value("restrict", best_restrict);
            level_transfer_table.add_value("restrict_new", best_restrict_new);
            level_transfer_table.add_value("restrict_speedup", best_restrict / best_restrict_new);
            level_transfer_table.add_value("restrict_rel_err", restrict_rel_err);
          }
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        level_transfer_table.set_scientific("prolong", true);
        level_transfer_table.set_precision("prolong", 4);
        level_transfer_table.set_scientific("prolong_new", true);
        level_transfer_table.set_precision("prolong_new", 4);
        level_transfer_table.set_precision("prolong_speedup", 3);
        level_transfer_table.set_scientific("prolong_rel_err", true);
        level_transfer_table.set_precision("prolong_rel_err", 3);
        level_transfer_table.set_scientific("restrict", true);
        level_transfer_table.set_precision("restrict", 4);
        level_transfer_table.set_scientific("restrict_new", true);
        level_transfer_table.set_precision("restrict_new", 4);
        level_transfer_table.set_precision("restrict_speedup", 3);
        level_transfer_table.set_scientific("restrict_rel_err", true);
        level_transfer_table.set_precision("restrict_rel_err", 3);

        std::cout << std::endl
                  << "Per-level prolongate/restrict speedup and correctness:" << std::endl;
        level_transfer_table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::test()
  {
    pcout << std::endl << std::endl;
    // for (unsigned int level = 0; level <= level_matrices.max_level();
    // ++level)
    //   {
    //     LinearAlgebra::distributed::Vector<vcycle_number,
    //     MemorySpace::Default>
    //       src, vec_bk3, vec_dealii, err;


    //     const auto &matrix_bk3    = *level_matrices[level];
    //     const auto &matrix_dealii = *level_matrices_dealii[level];

    //     matrix_bk3.initialize_dof_vector(src);
    //     vec_bk3.reinit(src);
    //     vec_dealii.reinit(src);

    //     src = 1.;

    //     matrix_dealii.vmult(vec_dealii, src);
    //     matrix_bk3.vmult(vec_bk3, src);

    //     err = vec_bk3;
    //     err -= vec_dealii;

    //     pcout << "L = " << level << ": " << vec_bk3.l2_norm() << " | "
    //           << vec_dealii.l2_norm() << " | " << err.l2_norm() << std::endl;
    //   }

    // for (unsigned int level = 0; level <= level_matrices.max_level();
    // ++level)
    //   {
    //     LinearAlgebra::distributed::Vector<vcycle_number,
    //     MemorySpace::Default>
    //       src, vec_bk3, vec_dealii, err;


    //     const auto &matrix_bk3    = *level_matrices[level];
    //     const auto &matrix_dealii = *level_matrices_dealii[level];

    //     matrix_bk3.initialize_dof_vector(src);
    //     vec_bk3.reinit(src);
    //     vec_dealii.reinit(src);

    //     src = 1.;

    //     matrix_dealii.vmult(vec_dealii, src);
    //     matrix_bk3.vmult(vec_bk3, src);

    //     err = vec_bk3;
    //     err -= vec_dealii;

    //     pcout << "L = " << level << ": " << vec_bk3.l2_norm() << " | "
    //           << vec_dealii.l2_norm() << " | " << err.l2_norm() << std::endl;
    //   }
    // pcout << std::endl << std::endl;
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const std::size_t  min_size,
                                      const std::size_t  max_size,
                                      const unsigned int n_pre_smooth,
                                      const unsigned int n_post_smooth,
                                      const bool         use_doubling_mesh)
  {
    pcout << "Testing " << fe.get_name() << std::endl;
    const unsigned int sizes[] = {1,   2,   3,   4,   5,   6,   7,   8,   10,  12,   14,   16,  20,
                                  24,  28,  32,  40,  48,  56,  64,  80,  96,  112,  128,  160, 192,
                                  224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536};



    for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int); ++cycle)
      {
        triangulation.clear();

        setup_time = 0.;

        pcout << "Cycle " << cycle << std::endl;

        std::size_t  projected_size = numbers::invalid_size_type;
        unsigned int n_refine       = 0;

        if (use_doubling_mesh)
          {
            n_refine                     = cycle / 3;
            const unsigned int remainder = cycle % 3;
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
            const unsigned int base_refine = (1 << n_refine);
            projected_size                 = 1;
            for (unsigned int d = 0; d < dim; ++d)
              projected_size *= base_refine * subdivisions[d] * fe_degree + 1;
            GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
          }
        else
          {
            n_refine              = 0;
            unsigned int n_subdiv = sizes[cycle];
            if (n_subdiv > 1)
              while (n_subdiv % 2 == 0)
                {
                  n_refine += 1;
                  n_subdiv /= 2;
                }
            if (dim == 2)
              n_refine += 3;
            GridGenerator::subdivided_hyper_cube(triangulation, n_subdiv, -0.9, 1.0);
            const unsigned int base_refine = (1 << n_refine);
            projected_size = Utilities::pow(base_refine * n_subdiv * fe_degree + 1, dim);
          }

        if (projected_size < min_size)
          continue;

        if (projected_size > max_size)
          {
            pcout << "Projected size " << projected_size << " higher than max size, terminating."
                  << std::endl;
            pcout << std::endl;
            break;
          }


        triangulation.refine_global(n_refine);

        create_coarse_triangulations();

        setup_dofs();

        setup_matrix_free();

        setup_mg_transfers();

        setup_smoothers(n_pre_smooth, n_post_smooth);

        compute_rhs();

        pcout << "Total setup time: " << setup_time << std::endl;

        solve(n_pre_smooth, n_post_smooth);
        pcout << std::endl;

        pcout << std::endl;
        pcout << std::endl;
        matvec_ghost_timing();
        pcout << std::endl;
        pcout << std::endl;

        pcout << std::endl;
        pcout << std::endl;
        vmult_comparison_timing();
        pcout << std::endl;
        pcout << std::endl;

        pcout << std::endl;
        pcout << std::endl;
        prolong_restrict_comparison_timing();
        pcout << std::endl;
        pcout << std::endl;

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            convergence_table.set_scientific("mv_outer", true);
            convergence_table.set_precision("mv_outer", 3);
            convergence_table.set_scientific("mv_inner", true);
            convergence_table.set_precision("mv_inner", 3);
            convergence_table.set_scientific("cg_reduction", true);
            convergence_table.set_precision("cg_reduction", 3);
            convergence_table.set_scientific("cg_time", true);
            convergence_table.set_precision("cg_time", 3);

            convergence_table.write_text(std::cout);

            std::cout << std::endl << std::endl;

            ghost_timing_table.set_scientific("mv_ghost_and_compute", true);
            ghost_timing_table.set_precision("mv_ghost_and_compute", 4);
            ghost_timing_table.set_scientific("mv_compute_only", true);
            ghost_timing_table.set_precision("mv_compute_only", 4);
            ghost_timing_table.set_scientific("mv_ghost_only", true);
            ghost_timing_table.set_precision("mv_ghost_only", 4);

            ghost_timing_table.write_text(std::cout);

            std::cout << std::endl << std::endl;

            vmult_comparison_table.set_scientific("vmult", true);
            vmult_comparison_table.set_precision("vmult", 4);
            vmult_comparison_table.set_scientific("vmult_new", true);
            vmult_comparison_table.set_precision("vmult_new", 4);
            vmult_comparison_table.set_precision("speedup", 3);

            vmult_comparison_table.write_text(std::cout);

            std::cout << std::endl << std::endl;
          }
      }
  }
  template <int dim, int min_degree, int max_degree>
  class LaplaceRunTime
  {
  public:
    LaplaceRunTime(const unsigned int target_degree,
                   const std::size_t  min_size,
                   const std::size_t  max_size,
                   const unsigned int n_pre_smooth,
                   const unsigned int n_post_smooth,
                   const bool         use_doubling_mesh,
                   const bool         overlap_communication_computation)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem(overlap_communication_computation);
          laplace_problem.run(min_size, max_size, n_pre_smooth, n_post_smooth, use_doubling_mesh);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree,
        min_size,
        max_size,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh,
        overlap_communication_computation);
    }
  };
} // namespace multigrid

int
main(int argc, char *argv[])
{
  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree                            = numbers::invalid_unsigned_int;
      std::size_t  maxsize                           = static_cast<std::size_t>(-1);
      std::size_t  minsize                           = 1;
      unsigned int n_pre_smooth                      = 3;
      unsigned int n_post_smooth                     = 3;
      bool         use_doubling_mesh                 = true;
      bool         overlap_communication_computation = false;

      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "Expected at least one argument." << std::endl
              << "Usage:" << std::endl
              << "./program degree minsize maxsize n_pre_smooth n_post_smooth doubling overlap_communication_computation"
              << std::endl
              << "The parameters degree to n_post_smooth are integers, "
              << "the last selects between a square mesh or a doubling mesh" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);
      if (argc > 4)
        n_pre_smooth = std::atoi(argv[4]);
      if (argc > 5)
        n_post_smooth = std::atoi(argv[5]);
      if (argc > 6)
        use_doubling_mesh = argv[6][0] == 'd';
      if (argc > 7)
        overlap_communication_computation = std::atoi(argv[7]) == 1;


      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters:                " << std::endl
                  << "Number of MPI ranks:                   "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Polynomial degree:                     " << degree << std::endl
                  << "Minimum size:                          " << minsize << std::endl
                  << "Maximum size:                          " << maxsize << std::endl
                  << "Number of pre-smoother iters:          " << n_pre_smooth << std::endl
                  << "Number of post-smoother iters:         " << n_post_smooth << std::endl
                  << "Use doubling mesh:                     " << use_doubling_mesh << std::endl
                  << "Use overlap_communication_computation: " << overlap_communication_computation
                  << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(
        degree,
        minsize,
        maxsize,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh,
        overlap_communication_computation);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }


  return 0;
}
