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
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "operators/portable_laplace_operator.h"

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



  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void
    run(const std::size_t min_size, const std::size_t max_size, const bool use_doubling_mesh);


    using VectorTypeMG = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  private:
    void
    setup_grid();

    void
    setup_dofs();

    void
    setup_matrix_free();

    void
    compute_rhs();

    void
    solve();

    // void
    // matvec_ghost_timing();

    void
    vmult_comparison_timing();


    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    std::set<types::boundary_id> dirichlet_boundary_ids;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>    ghost_solution_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> solution_device;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> system_rhs_device;

    std::unique_ptr<Portable::LaplaceOperatorBase<dim, double>> system_matrix;

    const unsigned int refinement_cycles = 10;

    const bool overlap_communication_computation = false;

    double setup_time;

    ConvergenceTable convergence_table;

    ConvergenceTable ghost_timing_table;

    ConvergenceTable vmult_comparison_table;


    ConditionalOStream pcout;

    ConditionalOStream time_details;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(fe_degree)
    , dof_handler(triangulation)
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

    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);
    constraints.close();


    setup_time += time.wall_time();

    time_details << "DoFs and constraint setup  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_matrix_free()
  {
    Timer time;
    Kokkos::fence();

    system_matrix = std::make_unique<Portable::LaplaceOperator<dim, fe_degree, double>>(
      dof_handler, constraints, overlap_communication_computation);

    system_matrix->initialize_dof_vector(solution_device);
    system_rhs_device.reinit(solution_device);
    ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    Kokkos::fence();

    setup_time += time.wall_time();

    time_details << "Setup matrices   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::compute_rhs()
  {
    Timer time;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

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
                cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs_host);
          }
      }

    system_rhs_host.compress(VectorOperation::add);
    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

    rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
    system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

    setup_time += time.wall_time();

    time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve()
  {
    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

    pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index << "] " << memory.avg
          << " " << memory.max << " [p" << memory.max_index << "]" << std::endl;


    unsigned int                    iterations = 0;
    double                          time_cg    = 1e10;
    std::pair<unsigned int, double> cg_details;
    for (unsigned int i = 0; i < 10; ++i)
      {
        Kokkos::fence();
        time.restart();
        solution_device = 0;
        SolverControl solver_control(system_rhs_device.size(), 1e-12 * system_rhs_device.l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
          solver_control);
        cg.solve(*system_matrix, solution_device, system_rhs_device, PreconditionIdentity());
        Kokkos::fence();
        time_cg    = std::min(time.wall_time(), time_cg);
        iterations = solver_control.last_step();
        pcout << "Time solve CG: " << iterations << " iterations,    " << time.wall_time() << "s"
              << std::endl;
      }

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import_elements(solution_device, VectorOperation::insert);
    ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

    constraints.distribute(ghost_solution_host);

    ghost_solution_host.update_ghost_values();

    Vector<float> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      Functions::ZeroFunction<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);

    const double global_norm =
      VectorTools::compute_global_error(triangulation, cellwise_norm, VectorTools::L2_norm);

    pcout << "  solution norm: " << global_norm << std::endl;

    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", iterations);
    convergence_table.add_value("norm", global_norm);

    // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    //   for (unsigned int level = 1; level <= level_matrices.max_level(); level++)
    //     {
    //       std::cout << "Best timings for ndof = " << level_dof_handlers[level].n_dofs()
    //                 << "   on level " << level
    //                 << "|  restriction = " << restrict_per_level[level - 1]
    //                 << "   prolongation  =  " << prolongate_per_level[level - 1] << std::endl;
    //     }
  }


  // template <int dim, int fe_degree>
  // void
  // LaplaceProblem<dim, fe_degree>::matvec_ghost_timing()
  // {
  //   const bool ghost_exchange_on = true;
  //   const bool computation_on    = true;

  //   MGLevelObject<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  //   dummy_solution(
  //     0, level_matrices.max_level()),
  //     dummy_rhs(0, level_matrices.max_level());

  //   for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
  //     {
  //       level_matrices[level]->initialize_dof_vector(dummy_solution[level]);

  //       level_matrices[level]->initialize_dof_vector(dummy_rhs[level]);
  //     }

  //   Timer time;

  //   double best_mv_both    = 1e10;
  //   double best_only_ghost = 1e10;
  //   double best_only_comp  = 1e10;

  //   for (unsigned int level = 0; level <= level_matrices.max_level(); ++level)
  //     {
  //       best_mv_both    = 1e10;
  //       best_only_ghost = 1e10;
  //       best_only_comp  = 1e10;

  //       for (unsigned int i = 0; i < 5; ++i)
  //         {
  //           const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

  //           {
  //             Kokkos::fence();
  //             time.restart();
  //             for (unsigned int i = 0; i < n_mv; ++i)
  //               level_matrices[level]->vmult_dummy(dummy_solution[level],
  //                                                  dummy_rhs[level],
  //                                                  ghost_exchange_on,
  //                                                  computation_on);
  //             Kokkos::fence();

  //             Utilities::MPI::MinMaxAvg stat =
  //               Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

  //             best_mv_both = std::min(best_mv_both, stat.max);
  //           }
  //           {
  //             Kokkos::fence();
  //             time.restart();
  //             for (unsigned int i = 0; i < n_mv; ++i)
  //               level_matrices[level]->vmult_dummy(dummy_solution[level],
  //                                                  dummy_rhs[level],
  //                                                  ghost_exchange_on,
  //                                                  !computation_on);
  //             Kokkos::fence();

  //             Utilities::MPI::MinMaxAvg stat =
  //               Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

  //             best_only_ghost = std::min(best_only_ghost, stat.max);
  //           }

  //           {
  //             Kokkos::fence();
  //             time.restart();
  //             for (unsigned int i = 0; i < n_mv; ++i)
  //               level_matrices[level]->vmult_dummy(dummy_solution[level],
  //                                                  dummy_rhs[level],
  //                                                  !ghost_exchange_on,
  //                                                  computation_on);
  //             Kokkos::fence();

  //             Utilities::MPI::MinMaxAvg stat =
  //               Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

  //             best_only_comp = std::min(best_only_comp, stat.max);
  //           }
  //         }

  //       if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  //         std::cout << "Best timings for ndof = " << level_dof_handlers[level].n_dofs()
  //                   << "   on level " << level << "|  ghost & compute =  " << best_mv_both
  //                   << "   ghost only      =  " << best_only_ghost
  //                   << "   compute only    =  " << best_only_comp

  //                   << std::endl;
  //     }

  //   ghost_timing_table.add_value("cells", triangulation.n_global_active_cells());
  //   ghost_timing_table.add_value("dofs", dof_handler.n_dofs());
  //   ghost_timing_table.add_value("mv_ghost_and_compute", best_mv_both);
  //   ghost_timing_table.add_value("mv_compute_only", best_only_comp);
  //   ghost_timing_table.add_value("mv_ghost_only", best_only_ghost);
  // }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::vmult_comparison_timing()
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dummy_solution, dummy_rhs;
    system_matrix->initialize_dof_vector(dummy_solution);
    system_matrix->initialize_dof_vector(dummy_rhs);

    Timer time;

    double best_vmult     = 1e10;
    double best_vmult_new = 1e10;

    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

        {
          Kokkos::fence();
          time.restart();
          for (unsigned int i = 0; i < n_mv; ++i)
            system_matrix->vmult(dummy_solution, dummy_rhs);
          Kokkos::fence();

          Utilities::MPI::MinMaxAvg stat =
            Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

          best_vmult = std::min(best_vmult, stat.max);
        }
        {
          Kokkos::fence();
          time.restart();
          for (unsigned int i = 0; i < n_mv; ++i)
            system_matrix->vmult_new(dummy_solution, dummy_rhs);
          Kokkos::fence();

          Utilities::MPI::MinMaxAvg stat =
            Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

          best_vmult_new = std::min(best_vmult_new, stat.max);
        }
      }


    vmult_comparison_table.add_value("cells", triangulation.n_global_active_cells());
    vmult_comparison_table.add_value("dofs", dof_handler.n_dofs());
    vmult_comparison_table.add_value("vmult", best_vmult);
    vmult_comparison_table.add_value("vmult_new", best_vmult_new);
    vmult_comparison_table.add_value("speedup", best_vmult / best_vmult_new);
  }


  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const std::size_t min_size,
                                      const std::size_t max_size,
                                      const bool        use_doubling_mesh)
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

        setup_dofs();

        setup_matrix_free();

        compute_rhs();

        pcout << "Total setup time: " << setup_time << std::endl;

        solve();
        pcout << std::endl;

        pcout << std::endl;
        pcout << std::endl;
        // matvec_ghost_timing();
        vmult_comparison_timing();
        pcout << std::endl;
        pcout << std::endl;


        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            convergence_table.set_scientific("cg_time", true);
            convergence_table.set_precision("cg_time", 3);
            convergence_table.set_scientific("norm", true);
            convergence_table.set_precision("norm", 3);

            convergence_table.write_text(std::cout);

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
                   const bool         use_doubling_mesh)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem;
          laplace_problem.run(min_size, max_size, use_doubling_mesh);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree, min_size, max_size, use_doubling_mesh);
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

      unsigned int degree            = numbers::invalid_unsigned_int;
      std::size_t  maxsize           = static_cast<std::size_t>(-1);
      std::size_t  minsize           = 1;
      bool         use_doubling_mesh = true;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program degree minsize maxsize doubling" << std::endl
                      << "The parameters degree to maxsize are integers, "
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
        use_doubling_mesh = argv[4][0] == 'd';

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(degree,
                                                                    minsize,
                                                                    maxsize,
                                                                    use_doubling_mesh);
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
