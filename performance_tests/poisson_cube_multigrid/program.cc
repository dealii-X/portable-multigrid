#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "multigrid/portable_continuous_transfer.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_laplace_operator.h"
#include "operators/portable_laplace_operator_bk3.h"
#include "operators/portable_laplace_operator_dg.h"

#include "portable_multigrid_solver.h"
#include "coarsening_sequence.h"


namespace multigrid
{
  using namespace dealii;

  // Degrees 1..4 are compiled, the degree is chosen at run time.
  const unsigned int dimension      = 3;
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 4;

  // Variant, set by CMake (see CMakeLists.txt):
  //   MG_VCYCLE_DOUBLE     V-cycle in double instead of float
  //   MG_GENERIC_OPERATOR  Portable::LaplaceOperator instead of LaplaceOperatorBK3
  //   MG_DG                FE_DGQ fine level (LaplaceOperatorDG, ContinuousTransfer)
  //                        on top of the FE_Q hierarchy
  //   MG_P_STEP            degree reduction per p-level (default 2)
  //   MG_VARIANT_NAME      name written to the JSON output
#ifdef MG_VCYCLE_DOUBLE
  using vcycle_number = double;
#else
  using vcycle_number = float;
#endif
  using full_number = double;

#ifdef MG_GENERIC_OPERATOR
  constexpr bool use_bk3 = false;
#else
  constexpr bool use_bk3 = true;
#endif

#ifdef MG_DG
  constexpr bool use_dg = true;
#else
  constexpr bool use_dg = false;
#endif

#ifndef MG_P_STEP
#  define MG_P_STEP 2
#endif
  constexpr unsigned int p_step = MG_P_STEP;

#ifndef MG_VARIANT_NAME
#  define MG_VARIANT_NAME "poisson_cube_multigrid"
#endif

  template <int dim, int degree, typename number>
  using LevelOperator = std::conditional_t<use_bk3,
                                           Portable::LaplaceOperatorBK3<dim, degree, number>,
                                           Portable::LaplaceOperator<dim, degree, number>>;



  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const bool                     overlap_communication_computation,
                   const std::string             &json_file,
                   const types::global_cell_index n_cells_single_rank);

    void
    run(const std::size_t  min_size,
        const std::size_t  max_size,
        const unsigned int n_pre_smooth,
        const unsigned int n_post_smooth,
        const bool         use_doubling_mesh);

    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>;

    using SmootherType =
      PreconditionChebyshev<Portable::LaplaceOperatorBase<dim, VectorTypeMG>, VectorTypeMG>;

  private:
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
    solve(const unsigned int n_pre_smooth, const unsigned int n_post_smooth);

    void
    matvec_ghost_timing();

    void
    write_json(const unsigned int n_pre_smooth,
               const unsigned int n_post_smooth,
               const bool         use_doubling_mesh);


    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    MappingQ<dim>                                       mapping;
    std::conditional_t<use_dg, FE_DGQ<dim>, FE_Q<dim>> fe;
    DoFHandler<dim>                                     dof_handler;

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

    MGLevelObject<std::unique_ptr<Portable::LaplaceOperatorBase<
      dim,
      LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>>>>
      level_matrices;

    MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, vcycle_number>>> mg_transfers;

    MGLevelObject<SmootherType> mg_smoothers;

    AffineConstraints<full_number> fine_level_constraints;

    std::unique_ptr<Portable::LaplaceOperatorBase<
      dim,
      LinearAlgebra::distributed::Vector<full_number, MemorySpace::Default>>>
      fine_level_matrix;

    const bool overlap_communication_computation;

    // h-levels with fewer cells live on rank 0
    const types::global_cell_index n_cells_single_rank;

    double setup_time;

    // JSON output: setup phases and results of the current cycle, records of
    // all cycles so far
    const std::string                             json_file;
    std::vector<std::pair<std::string, double>>   setup_phases;
    double                                        result_mv_outer = 0., result_mv_inner = 0.,
                                                  result_cg_time = 0., result_cg_reduction = 0.;
    unsigned int                                  result_cg_its = 0;
    std::vector<double>                           result_restrict, result_prolongate;
    std::vector<std::string>                      json_cycles;

    ConvergenceTable convergence_table;

    ConvergenceTable ghost_timing_table;

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
          std::make_unique<LevelOperator<dim, degree, vcycle_number>>(
            dof_handler, constraints, overlap_communication_computation);
      }
    };

    struct PolynomialTransferRunner
    {
      const unsigned int                     level;
      const Portable::OperatorInfoBase<dim> &op_coarse;
      const Portable::OperatorInfoBase<dim> &op_fine;
      AffineConstraints<vcycle_number>      &constraints_coarse;
      AffineConstraints<vcycle_number>      &constraints_fine;

      LaplaceProblem<dim, fe_degree> &parent_problem;

      template <unsigned int degree_coarse, unsigned int degree_fine>
      void
      run()
      {
        parent_problem.mg_transfers[level] = std::make_unique<
          Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, vcycle_number>>();

        parent_problem.mg_transfers[level]->reinit(op_coarse,
                                                   op_fine,
                                                   constraints_coarse,
                                                   constraints_fine);
      }
    };
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem(
    const bool                     overlap_communication_computation,
    const std::string             &json_file,
    const types::global_cell_index n_cells_single_rank)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , mapping(fe_degree)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , overlap_communication_computation(overlap_communication_computation)
    , n_cells_single_rank(n_cells_single_rank)
    , setup_time(0.)
    , json_file(json_file)

    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , time_details(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)

  {
    dirichlet_boundary_ids.insert(0);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::create_coarse_triangulations()
  {
    setup_time = 0;
    setup_phases.clear();

    Timer time;

    const RepartitioningPolicyTools::MinimalGranularityPolicy<dim> granularity_policy(16);
    coarse_triangulations = create_refined_coarsening_sequence(
      triangulation, SingleRankBelowPolicy<dim>(n_cells_single_rank, granularity_policy));

    setup_time += time.wall_time();

    for (unsigned int level = 0; level < coarse_triangulations.size(); ++level)
      {
        const auto tria = dynamic_cast<const parallel::TriangulationBase<dim> *>(
          coarse_triangulations[level].get());
        const unsigned int n_ranks = Utilities::MPI::sum<unsigned int>(
          tria->n_locally_owned_active_cells() > 0 ? 1 : 0, mpi_communicator);
        pcout << "h-level " << level << ": " << tria->n_global_active_cells() << " cells on "
              << n_ranks << " ranks" << std::endl;
      }

    time_details << "Coarse triangulations created  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;
    setup_phases.emplace_back("coarse_triangulations", time.wall_time());
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
      p_levels.push_back(p_levels.back() > p_step ? p_levels.back() - p_step : 1u);

    for (const auto p : p_levels)
      pcout << p << "  ";
    pcout << std::endl;

    p_level_fes.resize(0, p_levels.size() - 1);

    for (unsigned int level = 0; level < p_levels.size(); ++level)
      {
        p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);
      }

    // DG: one more level, the FE_DGQ space on the finest mesh
    level_dof_handlers.resize(0,
                              coarse_triangulations.size() - 1 + p_level_fes.max_level() +
                                (use_dg ? 1 : 0));
    level_constraints.resize(0, level_dof_handlers.max_level());

    for (unsigned int level = level_dof_handlers.min_level();
         level <= level_dof_handlers.max_level();
         ++level)
      {
        DoFHandler<dim> &dof_h = level_dof_handlers[level];

        dof_h.reinit(*coarse_triangulations[std::min(level, triangulation.n_global_levels() - 1)]);

        if (level < coarse_triangulations.size())
          dof_h.distribute_dofs(*p_level_fes[0]);
        else if (!use_dg || level < level_dof_handlers.max_level())
          dof_h.distribute_dofs(*p_level_fes[level + 1 - coarse_triangulations.size()]);
        else
          dof_h.distribute_dofs(fe);

        IndexSet level_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_h);

        fine_level_constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);

        DoFTools::make_hanging_node_constraints(dof_h, fine_level_constraints);

        // DG: weak boundary conditions
        if (!use_dg || level < level_dof_handlers.max_level())
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
    setup_phases.emplace_back("dofs", time.wall_time());
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
            level_matrices[level] = std::make_unique<LevelOperator<dim, 1, vcycle_number>>(
              level_dof_handlers[level],
              level_constraints[level],
              overlap_communication_computation);
          }
        else if (!use_dg || level < level_dof_handlers.max_level())
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
        else if constexpr (use_dg)
          level_matrices[level] =
            std::make_unique<Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, vcycle_number>>(
              mapping,
              level_dof_handlers[level],
              level_constraints[level],
              overlap_communication_computation);
      }



    if constexpr (std::is_same_v<full_number, vcycle_number>)
      {
        const auto &system_matrix = *level_matrices.back();
        system_matrix.initialize_dof_vector(solution_device);
      }
    else
      {
        if constexpr (use_dg)
          fine_level_matrix =
            std::make_unique<Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, full_number>>(
              mapping,
              level_dof_handlers.back(),
              fine_level_constraints,
              overlap_communication_computation);
        else
          fine_level_matrix = std::make_unique<LevelOperator<dim, fe_degree, full_number>>(
            level_dof_handlers.back(), fine_level_constraints, overlap_communication_computation);

        fine_level_matrix->initialize_dof_vector(solution_device);
      }
    system_rhs_device.reinit(solution_device);
    ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    Kokkos::fence();

    setup_time += time.wall_time();

    time_details << "Setup matrices   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
    setup_phases.emplace_back("matrices", time.wall_time());
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
            mg_transfers[level]->reinit(*level_matrices[level - 1],
                                        *level_matrices[level],
                                        level_constraints[level - 1],
                                        level_constraints[level]);
          }
        else if (!use_dg || level < level_matrices.max_level())
          {
            const unsigned int p_coarse = p_level_fes[level - coarse_triangulations.size()]->degree;
            const unsigned int p_fine =
              p_level_fes[level + 1 - coarse_triangulations.size()]->degree;

            PolynomialTransferRunner runner{level,
                                            *level_matrices[level - 1],
                                            *level_matrices[level],
                                            level_constraints[level - 1],
                                            level_constraints[level],
                                            *this};

            bool success =
              Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse, p_fine, runner);

            Assert(success,
                   ExcMessage("Failed to find a matching polynomial degree "
                              "pair in transfer dispatcher."));
          }
        else
          {
            // DG: FE_Q(p) -> FE_DGQ(p) on the same mesh
            mg_transfers[level] =
              std::make_unique<Portable::ContinuousTransfer<dim, fe_degree, vcycle_number>>();
            mg_transfers[level]->reinit(*level_matrices[level - 1],
                                        *level_matrices[level],
                                        level_constraints[level - 1],
                                        level_constraints[level]);
          }
      }
    Kokkos::fence();

    pcout << std::endl;

    setup_time += time.wall_time();

    time_details << "Setup transfers   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
    setup_phases.emplace_back("transfers", time.wall_time());
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
        smoother_data.constraints.copy_from(level_constraints[level]);

        mg_smoothers[level].initialize(*level_matrices[level], smoother_data);
      }

    Kokkos::fence();
    setup_time += time.wall_time();

    time_details << "Setup smoothers   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
    setup_phases.emplace_back("smoothers", time.wall_time());
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::compute_rhs()
  {
    Timer time;

    // (f, phi_i) with f = 1: BK3 assembles it on the device with its cell
    // kernel (constrained DoFs skipped), the other operators on the host
    if constexpr (use_bk3 && !use_dg)
      {
        using FineOperator = LevelOperator<dim, fe_degree, full_number>;
        const FineOperator *fine_operator = nullptr;
        if constexpr (std::is_same_v<full_number, vcycle_number>)
          fine_operator = dynamic_cast<const FineOperator *>(level_matrices.back().get());
        else
          fine_operator = dynamic_cast<const FineOperator *>(fine_level_matrix.get());
        Assert(fine_operator != nullptr,
               ExcMessage("The fine-level operator is not a LaplaceOperatorBK3."));

        fine_operator->compute_rhs(system_rhs_device);
      }
    else
      {
        LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host> system_rhs_host(
          locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

        const QGauss<dim>  quadrature_formula(fe_degree + 1);
        FEValues<dim>      fe_values(fe, quadrature_formula, update_values | update_JxW_values);
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        Vector<full_number>                  cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell_rhs = 0;
              fe_values.reinit(cell);
              for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);
              cell->get_dof_indices(local_dof_indices);
              fine_level_constraints.distribute_local_to_global(cell_rhs,
                                                                local_dof_indices,
                                                                system_rhs_host);
            }
        system_rhs_host.compress(VectorOperation::add);

        LinearAlgebra::ReadWriteVector<full_number> rw_vector(locally_owned_dofs);
        rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
        system_rhs_device.import_elements(rw_vector, VectorOperation::insert);
      }
    Kokkos::fence();

    setup_time += time.wall_time();

    time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
    setup_phases.emplace_back("rhs", time.wall_time());
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve(const unsigned int n_pre_smooth,
                                        const unsigned int n_post_smooth)
  {
    using SolverType =
      multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>;
    std::unique_ptr<SolverType> solver;

    if constexpr (std::is_same_v<full_number, vcycle_number>)
      solver = std::make_unique<SolverType>(level_matrices.back(),
                                            level_dof_handlers,
                                            level_constraints,
                                            mg_transfers,
                                            level_matrices,
                                            mg_smoothers,
                                            system_rhs_device,
                                            n_pre_smooth,
                                            n_post_smooth);
    else
      solver = std::make_unique<SolverType>(fine_level_matrix,
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


    convergence_table.add_value("n_proc", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("mv_outer", best_mv);
    convergence_table.add_value("mv_inner", best_mvs);
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", cg_details.first);
    convergence_table.add_value("cg_reduction", cg_details.second);

    result_mv_outer     = best_mv;
    result_mv_inner     = best_mvs;
    result_cg_time      = time_cg;
    result_cg_its       = cg_details.first;
    result_cg_reduction = cg_details.second;
    result_restrict     = restrict_per_level;
    result_prolongate   = prolongate_per_level;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      for (unsigned int level = 1; level <= level_matrices.max_level(); level++)
        {
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
          {
            const unsigned int n_mv =
              dof_handler.n_dofs() < 10000000 ? 200 : 50;

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


  // One record per cycle; the whole file is rewritten after every cycle.
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::write_json(const unsigned int n_pre_smooth,
                                             const unsigned int n_post_smooth,
                                             const bool         use_doubling_mesh)
  {
    std::ostringstream cycle;
    cycle << std::setprecision(6) << std::scientific;
    cycle << "    {\"cells\": " << triangulation.n_global_active_cells()
          << ", \"dofs\": " << dof_handler.n_dofs() << ", \"setup_time\": " << setup_time
          << ",\n     \"setup\": {";
    for (unsigned int i = 0; i < setup_phases.size(); ++i)
      cycle << (i > 0 ? ", " : "") << '"' << setup_phases[i].first
            << "\": " << setup_phases[i].second;
    cycle << "},\n     \"mv_outer\": " << result_mv_outer << ", \"mv_inner\": " << result_mv_inner
          << ", \"cg_time\": " << result_cg_time << ", \"cg_its\": " << result_cg_its
          << ", \"cg_reduction\": ";
    // 0/0 when CG takes no iteration; JSON has no nan
    if (std::isfinite(result_cg_reduction))
      cycle << result_cg_reduction;
    else
      cycle << "null";
    cycle << ",\n     \"levels\": [";
    for (unsigned int level = 1; level <= level_matrices.max_level(); ++level)
      cycle << (level > 1 ? ", " : "") << "{\"level\": " << level
            << ", \"dofs\": " << level_dof_handlers[level].n_dofs()
            << ", \"restrict\": " << result_restrict[level - 1]
            << ", \"prolongate\": " << result_prolongate[level - 1] << '}';
    cycle << "]}";
    json_cycles.push_back(cycle.str());

    if (json_file.empty() || Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
      return;

    std::ofstream out(json_file);
    out << "{\n  \"driver\": \"poisson_cube_multigrid\",\n  \"variant\": \"" << MG_VARIANT_NAME
        << "\",\n  \"dim\": " << dim << ",\n  \"degree\": " << fe_degree
        << ",\n  \"n_ranks\": " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << ",\n  \"vcycle_number\": \""
        << (std::is_same_v<vcycle_number, float> ? "float" : "double")
        << "\",\n  \"n_pre_smooth\": " << n_pre_smooth << ",\n  \"n_post_smooth\": " << n_post_smooth
        << ",\n  \"doubling_mesh\": " << (use_doubling_mesh ? "true" : "false")
        << ",\n  \"overlap\": " << (overlap_communication_computation ? "true" : "false")
        << ",\n  \"n_cells_single_rank\": " << n_cells_single_rank
        << ",\n  \"cycles\": [\n";
    for (unsigned int i = 0; i < json_cycles.size(); ++i)
      out << json_cycles[i] << (i + 1 < json_cycles.size() ? ",\n" : "\n");
    out << "  ]\n}\n";
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

        write_json(n_pre_smooth, n_post_smooth, use_doubling_mesh);

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
                   const bool         overlap_communication_computation,
                   const std::string &json_file,
                   const types::global_cell_index n_cells_single_rank)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem(overlap_communication_computation,
                                                          json_file,
                                                          n_cells_single_rank);
          laplace_problem.run(min_size, max_size, n_pre_smooth, n_post_smooth, use_doubling_mesh);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree,
        min_size,
        max_size,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh,
        overlap_communication_computation,
        json_file,
        n_cells_single_rank);
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

      // --json <file> and --single-rank-cells <n> anywhere on the command
      // line, the rest is positional
      std::string              json_file;
      types::global_cell_index n_cells_single_rank = 0;
      std::vector<char *>      args;
      for (int i = 0; i < argc; ++i)
        if (std::string(argv[i]) == "--json" && i + 1 < argc)
          json_file = argv[++i];
        else if (std::string(argv[i]) == "--single-rank-cells" && i + 1 < argc)
          n_cells_single_rank = std::atoll(argv[++i]);
        else
          args.push_back(argv[i]);
      argc = args.size();
      argv = args.data();

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
              << "./program degree minsize maxsize n_pre_smooth n_post_smooth doubling overlap_communication_computation [--json file] [--single-rank-cells n]"
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
                  << "Variant:                               " << MG_VARIANT_NAME << std::endl
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
                  << "JSON output:                           " << json_file << std::endl
                  << "h-levels on rank 0 below (cells):      " << n_cells_single_rank << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(
        degree,
        minsize,
        maxsize,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh,
        overlap_communication_computation,
        json_file,
        n_cells_single_rank);
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
