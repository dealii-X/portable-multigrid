// Performance and correctness comparison of three GPU two-level transfer
// kernels for geometric (h-)coarsening between globally refined meshes, in
// the style of performance_tests/abstracted_batched_laplace:
//
//   dealii    the installed deal.II Portable::MGTwoLevelTransfer, one team per
//             coarse cell, the unbatched tensor-product evaluator;
//   copy      Copy::Portable::MGTwoLevelTransfer (include/multigrid_dealii/),
//             the version to be PR'd to deal.II: batched tensor-product
//             evaluator, one component at a time, no scratch pad, a batch of
//             coarse cells per team (automatic batch size);
//   copy_b1   the same with one coarse cell per team, to separate the effect
//             of the kernel from the effect of batching;
//   bk1       this repo's Portable::GeometricTransfer, which drives the
//             hand-written BK1::Parallel batched prolongation/restriction
//             kernels (kernels/bk1_kokkos_kernels.h).
//
// For every mesh size the fine mesh is the coarse mesh refined once, both
// carry homogeneous Dirichlet constraints (the BK1 kernels do not support
// hanging nodes), and prolongate_and_add() / restrict_and_add() are applied
// to random vectors. The installed deal.II transfer is the correctness
// reference; timings are the best of five repetitions of n_mv applications,
// taking the slowest rank.
//
// Usage: ./program dimension degree minsize maxsize doubling [cells_per_batch]
//                  [n_applications]
// (as for performance_tests/abstracted_batched_laplace; degree is the FE_Q degree of
// both levels, 1 to maximal_degree). The optional cells_per_batch fixes the
// number of coarse cells per team of the `copy` column; 0 or absent selects
// the automatic choice from the shared memory budget. The optional
// n_applications overrides the number of applications per timing repetition
// (default 200, or 50 above 1e7 fine dofs), e.g. for profiling runs. Each
// timed variant runs inside a Kokkos profiling region named after it.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <iostream>
#include <memory>
#include <random>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid_dealii/portable_mg_transfer_global_coarsening.templates.h"

namespace multigrid
{
  using namespace dealii;

  // deal.II's Portable transfer dispatches the embedding kernels for coarse
  // degrees 1 to 6 (CellTransferFactory).
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 6;

  // applications per timing repetition; 0 selects the size-dependent default
  unsigned int n_applications_override = 0;

  template <int dim, int fe_degree>
  class TransferComparison
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

    TransferComparison();

    void
    run(const std::size_t  min_size,
        const std::size_t  max_size,
        const bool         use_doubling_mesh,
        const unsigned int cells_per_batch);

  private:
    void
    setup_dofs();

    void
    setup_transfers(const unsigned int cells_per_batch);

    void
    comparison_timing();

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation_coarse;
    parallel::distributed::Triangulation<dim> triangulation_fine;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler_coarse;
    DoFHandler<dim> dof_handler_fine;

    AffineConstraints<double> constraints_coarse;
    AffineConstraints<double> constraints_fine;

    std::unique_ptr<Portable::MGTwoLevelTransfer<dim, VectorType>>       transfer_dealii;
    std::unique_ptr<Copy::Portable::MGTwoLevelTransfer<dim, VectorType>> transfer_copy;
    std::unique_ptr<Copy::Portable::MGTwoLevelTransfer<dim, VectorType>> transfer_copy_b1;
    std::unique_ptr<Portable::GeometricTransfer<dim, fe_degree, double>> transfer_bk1;

    double setup_time_dealii;
    double setup_time_copy;
    double setup_time_bk1;

    ConvergenceTable timing_table;
    ConvergenceTable speedup_table;
    ConvergenceTable error_table;

    ConditionalOStream pcout;
  };



  template <int dim, int fe_degree>
  TransferComparison<dim, fe_degree>::TransferComparison()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation_coarse(mpi_communicator)
    , triangulation_fine(mpi_communicator)
    , fe(fe_degree)
    , dof_handler_coarse(triangulation_coarse)
    , dof_handler_fine(triangulation_fine)
    , setup_time_dealii(0.)
    , setup_time_copy(0.)
    , setup_time_bk1(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {}



  template <int dim, int fe_degree>
  void
  TransferComparison<dim, fe_degree>::setup_dofs()
  {
    for (auto *const dof_handler : {&dof_handler_coarse, &dof_handler_fine})
      {
        dof_handler->distribute_dofs(fe);

        AffineConstraints<double> &constraints =
          (dof_handler == &dof_handler_coarse) ? constraints_coarse : constraints_fine;
        constraints.clear();
        constraints.reinit(dof_handler->locally_owned_dofs(),
                           DoFTools::extract_locally_relevant_dofs(*dof_handler));
        VectorTools::interpolate_boundary_values(*dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 constraints);
        constraints.close();
      }

    pcout << "Number of degrees of freedom: " << dof_handler_fine.n_dofs() << " (fine), "
          << dof_handler_coarse.n_dofs() << " (coarse)" << std::endl;
  }



  template <int dim, int fe_degree>
  void
  TransferComparison<dim, fe_degree>::setup_transfers(const unsigned int cells_per_batch)
  {
    Timer time;

    Kokkos::fence();
    time.restart();
    transfer_dealii = std::make_unique<Portable::MGTwoLevelTransfer<dim, VectorType>>();
    transfer_dealii->reinit_geometric_transfer(dof_handler_fine,
                                               dof_handler_coarse,
                                               constraints_fine,
                                               constraints_coarse);
    Kokkos::fence();
    setup_time_dealii = time.wall_time();

    time.restart();
    transfer_copy = std::make_unique<Copy::Portable::MGTwoLevelTransfer<dim, VectorType>>();
    transfer_copy->reinit_geometric_transfer(dof_handler_fine,
                                             dof_handler_coarse,
                                             constraints_fine,
                                             constraints_coarse);
    if (cells_per_batch > 0)
      transfer_copy->set_cells_per_batch(cells_per_batch);
    Kokkos::fence();
    setup_time_copy = time.wall_time();

    transfer_copy_b1 = std::make_unique<Copy::Portable::MGTwoLevelTransfer<dim, VectorType>>();
    transfer_copy_b1->reinit_geometric_transfer(dof_handler_fine,
                                                dof_handler_coarse,
                                                constraints_fine,
                                                constraints_coarse);
    transfer_copy_b1->set_cells_per_batch(1);

    time.restart();
    transfer_bk1 = std::make_unique<Portable::GeometricTransfer<dim, fe_degree, double>>();
    transfer_bk1->reinit(dof_handler_coarse,
                         dof_handler_fine,
                         constraints_coarse,
                         constraints_fine);
    Kokkos::fence();
    setup_time_bk1 = time.wall_time();

    pcout << "Setup transfers (wall): dealii " << setup_time_dealii << "s, copy " << setup_time_copy
          << "s, bk1 " << setup_time_bk1 << 's' << std::endl;
  }



  template <int dim, int fe_degree>
  void
  TransferComparison<dim, fe_degree>::comparison_timing()
  {
    const auto initialize_vector = [&](VectorType &vec, const DoFHandler<dim> &dof_handler)
    {
      vec.reinit(dof_handler.locally_owned_dofs(),
                 DoFTools::extract_locally_relevant_dofs(dof_handler),
                 mpi_communicator);
    };

    const auto fill_random = [&](VectorType &vec, const unsigned int seed)
    {
      std::mt19937 gen(seed + Utilities::MPI::this_mpi_process(mpi_communicator));
      std::uniform_real_distribution<double> dist(-1., 1.);

      LinearAlgebra::ReadWriteVector<double> rw(vec.locally_owned_elements());
      for (const auto idx : vec.locally_owned_elements())
        rw(idx) = dist(gen);
      vec.import_elements(rw, VectorOperation::insert);
    };

    // prolongation: coarse -> fine; restriction: fine -> coarse
    VectorType src_coarse, src_fine;
    VectorType dst_fine_dealii, dst_fine_copy, dst_fine_copy_b1, dst_fine_bk1;
    VectorType dst_coarse_dealii, dst_coarse_copy, dst_coarse_copy_b1, dst_coarse_bk1;

    initialize_vector(src_coarse, dof_handler_coarse);
    initialize_vector(src_fine, dof_handler_fine);
    for (auto *const vec : {&dst_fine_dealii, &dst_fine_copy, &dst_fine_copy_b1, &dst_fine_bk1})
      initialize_vector(*vec, dof_handler_fine);
    for (auto *const vec :
         {&dst_coarse_dealii, &dst_coarse_copy, &dst_coarse_copy_b1, &dst_coarse_bk1})
      initialize_vector(*vec, dof_handler_coarse);

    fill_random(src_coarse, 42);
    fill_random(src_fine, 43);

    // -- correctness: the installed deal.II transfer is the reference --
    const auto apply_all = [&]()
    {
      dst_fine_dealii = 0.;
      transfer_dealii->prolongate_and_add(dst_fine_dealii, src_coarse);
      dst_fine_copy = 0.;
      transfer_copy->prolongate_and_add(dst_fine_copy, src_coarse);
      dst_fine_copy_b1 = 0.;
      transfer_copy_b1->prolongate_and_add(dst_fine_copy_b1, src_coarse);
      dst_fine_bk1 = 0.;
      transfer_bk1->prolongate_and_add(dst_fine_bk1, src_coarse);

      dst_coarse_dealii = 0.;
      transfer_dealii->restrict_and_add(dst_coarse_dealii, src_fine);
      dst_coarse_copy = 0.;
      transfer_copy->restrict_and_add(dst_coarse_copy, src_fine);
      dst_coarse_copy_b1 = 0.;
      transfer_copy_b1->restrict_and_add(dst_coarse_copy_b1, src_fine);
      dst_coarse_bk1 = 0.;
      transfer_bk1->restrict_and_add(dst_coarse_bk1, src_fine);
    };
    apply_all();

    const auto relative_error = [](const VectorType &vec, const VectorType &reference)
    {
      VectorType diff = vec;
      diff -= reference;
      const double norm = reference.l2_norm();
      return norm > 0. ? diff.l2_norm() / norm : diff.l2_norm();
    };

    const double err_p_copy    = relative_error(dst_fine_copy, dst_fine_dealii);
    const double err_p_copy_b1 = relative_error(dst_fine_copy_b1, dst_fine_dealii);
    const double err_p_bk1     = relative_error(dst_fine_bk1, dst_fine_dealii);
    const double err_r_copy    = relative_error(dst_coarse_copy, dst_coarse_dealii);
    const double err_r_copy_b1 = relative_error(dst_coarse_copy_b1, dst_coarse_dealii);
    const double err_r_bk1     = relative_error(dst_coarse_bk1, dst_coarse_dealii);

    // -- timing --
    const unsigned int n_mv = n_applications_override > 0 ?
                                n_applications_override :
                                (dof_handler_fine.n_dofs() < 10000000 ? 200u : 50u);

    const auto time_op = [&](const std::string &name, const std::function<void()> &func)
    {
      double best = 1e10;
      Timer  time;
      Kokkos::Profiling::pushRegion(name);
      for (unsigned int i = 0; i < 5; ++i)
        {
          Kokkos::fence();
          time.restart();
          for (unsigned int j = 0; j < n_mv; ++j)
            func();
          Kokkos::fence();

          const Utilities::MPI::MinMaxAvg stat =
            Utilities::MPI::min_max_avg(time.wall_time() / n_mv, mpi_communicator);

          best = std::min(best, stat.max);
        }
      Kokkos::Profiling::popRegion();
      return best;
    };

    const double t_p_dealii =
      time_op("P_dealii",
              [&]() { transfer_dealii->prolongate_and_add(dst_fine_dealii, src_coarse); });
    const double t_p_copy =
      time_op("P_copy", [&]() { transfer_copy->prolongate_and_add(dst_fine_copy, src_coarse); });
    const double t_p_copy_b1 =
      time_op("P_copy_b1",
              [&]() { transfer_copy_b1->prolongate_and_add(dst_fine_copy_b1, src_coarse); });
    const double t_p_bk1 =
      time_op("P_bk1", [&]() { transfer_bk1->prolongate_and_add(dst_fine_bk1, src_coarse); });

    const double t_r_dealii =
      time_op("R_dealii",
              [&]() { transfer_dealii->restrict_and_add(dst_coarse_dealii, src_fine); });
    const double t_r_copy =
      time_op("R_copy", [&]() { transfer_copy->restrict_and_add(dst_coarse_copy, src_fine); });
    const double t_r_copy_b1 =
      time_op("R_copy_b1",
              [&]() { transfer_copy_b1->restrict_and_add(dst_coarse_copy_b1, src_fine); });
    const double t_r_bk1 =
      time_op("R_bk1", [&]() { transfer_bk1->restrict_and_add(dst_coarse_bk1, src_fine); });

    timing_table.add_value("cells", triangulation_fine.n_global_active_cells());
    timing_table.add_value("dofs_fine", dof_handler_fine.n_dofs());
    timing_table.add_value("dofs_coarse", dof_handler_coarse.n_dofs());
    timing_table.add_value("setup_dealii", setup_time_dealii);
    timing_table.add_value("setup_copy", setup_time_copy);
    timing_table.add_value("setup_bk1", setup_time_bk1);
    timing_table.add_value("P_dealii", t_p_dealii);
    timing_table.add_value("P_copy", t_p_copy);
    timing_table.add_value("P_copy_b1", t_p_copy_b1);
    timing_table.add_value("P_bk1", t_p_bk1);
    timing_table.add_value("R_dealii", t_r_dealii);
    timing_table.add_value("R_copy", t_r_copy);
    timing_table.add_value("R_copy_b1", t_r_copy_b1);
    timing_table.add_value("R_bk1", t_r_bk1);

    speedup_table.add_value("cells", triangulation_fine.n_global_active_cells());
    speedup_table.add_value("dofs_fine", dof_handler_fine.n_dofs());
    speedup_table.add_value("P_copy_vs_dealii", t_p_dealii / t_p_copy);
    speedup_table.add_value("P_copy_b1_vs_dealii", t_p_dealii / t_p_copy_b1);
    speedup_table.add_value("P_bk1_vs_dealii", t_p_dealii / t_p_bk1);
    speedup_table.add_value("P_copy_vs_bk1", t_p_bk1 / t_p_copy);
    speedup_table.add_value("R_copy_vs_dealii", t_r_dealii / t_r_copy);
    speedup_table.add_value("R_copy_b1_vs_dealii", t_r_dealii / t_r_copy_b1);
    speedup_table.add_value("R_bk1_vs_dealii", t_r_dealii / t_r_bk1);
    speedup_table.add_value("R_copy_vs_bk1", t_r_bk1 / t_r_copy);

    error_table.add_value("cells", triangulation_fine.n_global_active_cells());
    error_table.add_value("dofs_fine", dof_handler_fine.n_dofs());
    error_table.add_value("P_copy", err_p_copy);
    error_table.add_value("P_copy_b1", err_p_copy_b1);
    error_table.add_value("P_bk1", err_p_bk1);
    error_table.add_value("R_copy", err_r_copy);
    error_table.add_value("R_copy_b1", err_r_copy_b1);
    error_table.add_value("R_bk1", err_r_bk1);
  }



  template <int dim, int fe_degree>
  void
  TransferComparison<dim, fe_degree>::run(const std::size_t  min_size,
                                          const std::size_t  max_size,
                                          const bool         use_doubling_mesh,
                                          const unsigned int cells_per_batch)
  {
    pcout << "Testing " << fe.get_name() << ", copy with "
          << (cells_per_batch > 0 ? std::to_string(cells_per_batch) : std::string("automatic"))
          << " coarse cells per team" << std::endl;

    // same mesh sequence as performance_tests/abstracted_batched_laplace; the fine mesh
    // is the mesh of that sequence and the coarse mesh the same base mesh
    // refined one time less
    const unsigned int sizes[] = {1,   2,   3,   4,   5,   6,   7,   8,   10,  12,   14,   16,  20,
                                  24,  28,  32,  40,  48,  56,  64,  80,  96,  112,  128,  160, 192,
                                  224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536};

    for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int); ++cycle)
      {
        triangulation_coarse.clear();
        triangulation_fine.clear();

        pcout << "Cycle " << cycle << std::endl;

        std::size_t  projected_size = numbers::invalid_size_type;
        unsigned int n_refine       = 0;

        const auto create_base_mesh = [&](Triangulation<dim> &triangulation)
        {
          if (use_doubling_mesh)
            {
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
              GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
            }
          else
            {
              unsigned int n_subdiv = sizes[cycle];
              if (n_subdiv > 1)
                while (n_subdiv % 2 == 0)
                  n_subdiv /= 2;
              GridGenerator::subdivided_hyper_cube(triangulation, n_subdiv, -0.9, 1.0);
            }
        };

        if (use_doubling_mesh)
          {
            n_refine                       = cycle / 3;
            const unsigned int remainder   = cycle % 3;
            const unsigned int base_refine = (1 << n_refine);
            projected_size                 = 1;
            for (unsigned int d = 0; d < dim; ++d)
              projected_size *= base_refine * (d < remainder ? 2 : 1) * fe_degree + 1;
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

        // the coarse mesh needs one refinement less than the fine mesh
        if (n_refine == 0)
          {
            pcout << "Fine mesh is not refined, no coarse mesh; skipping." << std::endl;
            continue;
          }

        create_base_mesh(triangulation_coarse);
        triangulation_coarse.refine_global(n_refine - 1);
        create_base_mesh(triangulation_fine);
        triangulation_fine.refine_global(n_refine);

        setup_dofs();
        setup_transfers(cells_per_batch);
        pcout << std::endl;

        comparison_timing();
        pcout << std::endl;

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          {
            for (const std::string column : {"setup_dealii",
                                             "setup_copy",
                                             "setup_bk1",
                                             "P_dealii",
                                             "P_copy",
                                             "P_copy_b1",
                                             "P_bk1",
                                             "R_dealii",
                                             "R_copy",
                                             "R_copy_b1",
                                             "R_bk1"})
              {
                timing_table.set_scientific(column, true);
                timing_table.set_precision(column, 4);
              }

            for (const std::string column : {"P_copy_vs_dealii",
                                             "P_copy_b1_vs_dealii",
                                             "P_bk1_vs_dealii",
                                             "P_copy_vs_bk1",
                                             "R_copy_vs_dealii",
                                             "R_copy_b1_vs_dealii",
                                             "R_bk1_vs_dealii",
                                             "R_copy_vs_bk1"})
              speedup_table.set_precision(column, 3);

            for (const std::string column :
                 {"P_copy", "P_copy_b1", "P_bk1", "R_copy", "R_copy_b1", "R_bk1"})
              {
                error_table.set_scientific(column, true);
                error_table.set_precision(column, 3);
              }

            std::cout << "-- transfer timing comparison (seconds per application) --" << std::endl;
            timing_table.write_text(std::cout);
            std::cout << std::endl;

            std::cout << "-- transfer speedup comparison --" << std::endl;
            speedup_table.write_text(std::cout);
            std::cout << std::endl;

            std::cout << "-- relative error vs. installed deal.II transfer --" << std::endl;
            error_table.write_text(std::cout);
            std::cout << std::endl << std::endl;
          }
      }
  }



  template <int dim, int min_degree, int max_degree>
  class TransferRunTime
  {
  public:
    TransferRunTime(const unsigned int target_degree,
                    const std::size_t  min_size,
                    const std::size_t  max_size,
                    const bool         use_doubling_mesh,
                    const unsigned int cells_per_batch)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          TransferComparison<dim, min_degree> comparison;
          comparison.run(min_size, max_size, use_doubling_mesh, cells_per_batch);
        }
      TransferRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree>
        m(target_degree, min_size, max_size, use_doubling_mesh, cells_per_batch);
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

      unsigned int dimension         = 3;
      unsigned int degree            = numbers::invalid_unsigned_int;
      std::size_t  maxsize           = static_cast<std::size_t>(-1);
      std::size_t  minsize           = 1;
      bool         use_doubling_mesh = true;
      unsigned int cells_per_batch   = 0;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program dimension degree minsize maxsize doubling "
                      << "[cells_per_batch] [n_applications]" << std::endl
                      << "The parameters degree to maxsize are integers, "
                      << "doubling selects between a square mesh or a doubling mesh, "
                      << "cells_per_batch fixes the batch of the copy (0: automatic)" << std::endl;
          return 1;
        }

      if (argc > 1)
        dimension = std::atoi(argv[1]);
      if (argc > 2)
        degree = std::atoi(argv[2]);
      if (argc > 3)
        minsize = std::atoll(argv[3]);
      if (argc > 4)
        maxsize = std::atoll(argv[4]);
      if (argc > 5)
        use_doubling_mesh = argv[5][0] == 'd';
      if (argc > 6)
        cells_per_batch = std::atoi(argv[6]);
      if (argc > 7)
        n_applications_override = std::atoi(argv[7]);

      AssertThrow(dimension == 2 || dimension == 3, ExcMessage("Dimension must be 2 or 3."));
      AssertThrow(degree >= minimal_degree && degree <= maximal_degree,
                  ExcMessage("The degree must be between " + std::to_string(minimal_degree) +
                             " and " + std::to_string(maximal_degree) + "."));

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Dimension:                      " << dimension << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh << std::endl
                  << "Cells per batch (copy):         "
                  << (cells_per_batch > 0 ? std::to_string(cells_per_batch) : "automatic")
                  << std::endl
                  << "Applications per repetition:    "
                  << (n_applications_override > 0 ? std::to_string(n_applications_override) :
                                                    "default")
                  << std::endl
                  << std::endl;

      if (dimension == 2)
        {
          TransferRunTime<2, minimal_degree, maximal_degree> run(
            degree, minsize, maxsize, use_doubling_mesh, cells_per_batch);
        }
      else
        {
          TransferRunTime<3, minimal_degree, maximal_degree> run(
            degree, minsize, maxsize, use_doubling_mesh, cells_per_batch);
        }
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
