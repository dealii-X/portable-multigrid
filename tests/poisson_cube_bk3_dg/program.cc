// Poisson problem on a cube: CG solver preconditioned by a hybrid multigrid
// (h-levels with Q1, p-levels Q2 and Q4, DG level of degree 4), with optional
// overlap of the ghost exchange with the computation.
//
// Run command (Jupiter, 16 GPUs on 4 nodes):
//
//   export OMP_PROC_BIND=false
//   srun -N 4 -n 16 --ntasks-per-node=4 -c 72 --cpu-bind=cores --gpu-bind=closest
//        --mem-bind=local --gpus-per-task=1
//        ./program 4 1 500000000 5 5 d 16384 1 1000000
//
// (one command line; written on three lines here)
//
// Arguments, in this order:
//   4          degree:              polynomial degree of the DG discretization
//                                   (and of the finest continuous Q4 level)
//   1          minsize:             skip mesh sizes whose projected number of
//                                   continuous Q4 DoFs is smaller than this
//   500000000  maxsize:             stop at the first mesh size whose projected
//                                   number of continuous Q4 DoFs exceeds this
//   5          n_pre_smooth:        degree of the Chebyshev pre-smoother
//   5          n_post_smooth:       degree of the Chebyshev post-smoother (must equal
//                                   n_pre_smooth)
//   d          doubling:            'd' doubling mesh (DoFs grow by 2x per cycle),
//                                   anything else uses the square mesh sizes
//   16384      coarse_granularity:  minimal number of cells per rank on the coarse
//                                   h-levels (fewer ranks on the coarse levels)
//   1          overlap:             1 colors the levels on the finest mesh (L7 to
//                                   L10 and the outer DG operator) and overlaps the
//                                   ghost exchange with the computation of interior
//                                   cells in the FE_Q and DG vmult, 0 disables it
//   1000000    overlap_min_dofs_per_rank: the vmult is overlapped only on levels
//                                   with at least this many DoFs per rank (smaller
//                                   FE_Q levels launch all colors and fence once,
//                                   since overlap costs more than it hides; if no
//                                   FE_Q level reaches it, nothing is colored)

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

#include "multigrid/portable_continuous_transfer.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_laplace_operator_bk3.h"
#include "operators/portable_laplace_operator_dg.h"
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

  // Minimal number of cells per MPI rank on the geometric coarse levels. A
  // larger value keeps the coarse levels on fewer ranks (agglomeration). Can
  // be overwritten by the seventh command line argument.
  unsigned int coarse_granularity = 16384;

  // Overlap the ghost exchange with the computation in the matrix-vector
  // products. On the continuous (FE_Q) levels: update_ghost_values_start(),
  // cells away from the ghost layer, update_ghost_values_finish(), cells at
  // the ghost layer, compress_start(), remaining cells, compress_finish().
  // On the DG levels: update_ghost_values_start(), owned cells and the faces
  // between them, update_ghost_values_finish(), ghost cells and the faces to
  // them (see LaplaceOperatorDG::set_overlap_in_vmult()). Can be
  // overwritten by the eighth command line argument (0 or 1).
  //
  // Overlapping needs a MatrixFree object with three colors, and the
  // polynomial and continuous transfers require the same coloring on both
  // of their levels. Therefore the levels on the finest mesh (the last
  // h-level, the p-levels and the DG level) get the three colors, while the
  // h-levels below keep a single color (the geometric transfer does not
  // depend on the coloring). Of the colored FE_Q levels, only those with at
  // least overlap_min_dofs_per_rank DoFs per rank actually overlap in
  // vmult(); on smaller levels the computation is too short to hide the
  // exchange and the extra kernel launches would only cost time. The same
  // threshold applies to the DG level and the outer DG operator, whose vmult
  // overlaps the exchange with the work on the owned cells (it does not use
  // the coloring). If none of the FE_Q levels reaches the threshold, the
  // levels are not colored either (the DG level may still overlap), so that
  // small problems do not pay for the coloring.
  bool use_overlap = false;

  // Can be overwritten by the ninth command line argument.
  types::global_dof_index overlap_min_dofs_per_rank = 1000000;


  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void
    run(const std::size_t  min_size,
        const std::size_t  max_size,
        const unsigned int n_pre_smooth,
        const unsigned int n_post_smooth,
        const bool         use_doubling_mesh);

    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Default>;

    using SmootherType =
      PreconditionChebyshev<Portable::LaplaceOperatorBase<dim, vcycle_number>, VectorTypeMG>;

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


    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    MappingQ<dim>   mapping;
    FE_DGQ<dim>     fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    std::set<types::boundary_id> dirichlet_boundary_ids;

    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host>    ghost_solution_host;
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

    double setup_time;

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
      bool                              overlap_in_vmult;
      LaplaceProblem<dim, fe_degree>   &parent_problem;

      template <unsigned int degree>
      void
      run()
      {
        parent_problem.level_matrices[level] =
          std::make_unique<Portable::LaplaceOperatorBK3<dim, degree, vcycle_number>>(
            dof_handler, constraints, overlap_communication_computation, overlap_in_vmult);
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
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , mapping(fe_degree)
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
  LaplaceProblem<dim, fe_degree>::create_coarse_triangulations()
  {
    setup_time = 0;

    Timer time;

    coarse_triangulations = MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      triangulation, RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(coarse_granularity));

    setup_time += time.wall_time();

    time_details << "Coarse triangulations created  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;

    for (unsigned int level = 0; level < coarse_triangulations.size(); ++level)
      {
        unsigned int owns_cells = 0;
        for (const auto &cell : coarse_triangulations[level]->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              owns_cells = 1;
              break;
            }
        const unsigned int n_active_ranks = Utilities::MPI::sum(owns_cells, MPI_COMM_WORLD);
        time_details << "Coarse level " << level << ": "
                     << coarse_triangulations[level]->n_global_active_cells() << " cells on "
                     << n_active_ranks << " ranks" << std::endl;
      }
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

    p_level_fes.resize(0, p_levels.size() - 1);

    for (unsigned int level = 0; level < p_levels.size(); ++level)
      p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

    level_dof_handlers.resize(0, coarse_triangulations.size() + p_level_fes.max_level());
    level_constraints.resize(0, level_dof_handlers.max_level());

    for (unsigned int level = level_dof_handlers.min_level();
         level <= level_dof_handlers.max_level();
         ++level)
      {
        DoFHandler<dim> &dof_h = level_dof_handlers[level];

        dof_h.reinit(*coarse_triangulations[std::min(level, triangulation.n_global_levels() - 1)]);

        if (level < coarse_triangulations.size())
          dof_h.distribute_dofs(*p_level_fes[0]);
        else if (level < level_dof_handlers.max_level())
          dof_h.distribute_dofs(*p_level_fes[level + 1 - coarse_triangulations.size()]);
        else
          dof_h.distribute_dofs(fe);

        IndexSet level_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_h);

        fine_level_constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);

        DoFTools::make_hanging_node_constraints(dof_h, fine_level_constraints);

        if (level < level_dof_handlers.max_level())
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

    std::string overlap_levels;

    // Three colors on the levels on the finest mesh, but only if at least one
    // of the FE_Q levels among them is large enough to overlap in vmult(),
    // see use_overlap. The DG level (the last level) overlaps without the
    // coloring, so it only needs to reach the threshold itself.
    const types::global_dof_index min_overlap_dofs =
      overlap_min_dofs_per_rank * Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int first_fine_mesh_level = coarse_triangulations.size() - 1;
    const unsigned int dg_level              = level_dof_handlers.max_level();
    bool               color_fine_mesh_levels = false;
    if (use_overlap)
      for (unsigned int level = first_fine_mesh_level; level < dg_level; ++level)
        if (level_dof_handlers[level].n_dofs() >= min_overlap_dofs)
          color_fine_mesh_levels = true;

    for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
      {
        const bool colored = color_fine_mesh_levels && level >= first_fine_mesh_level;
        const bool overlap_in_vmult = (level == dg_level ? use_overlap : colored) &&
                                      level_dof_handlers[level].n_dofs() >= min_overlap_dofs;
        if (overlap_in_vmult)
          overlap_levels += " " + std::to_string(level);

        if (level < coarse_triangulations.size())
          {
            level_matrices[level] =
              std::make_unique<Portable::LaplaceOperatorBK3<dim, 1, vcycle_number>>(
                level_dof_handlers[level], level_constraints[level], colored, overlap_in_vmult);
          }

        else if (level < level_dof_handlers.max_level())

          {
            LaplaceOperatorRunner runner{level,
                                         level_dof_handlers[level],
                                         level_constraints[level],
                                         colored,
                                         overlap_in_vmult,
                                         *this};

            bool success = Portable::OperatorDispatchFactory::dispatch(
              p_level_fes[level + 1 - coarse_triangulations.size()]->degree, runner);

            Assert(success,
                   ExcMessage("Failed to find a matching polynomial degree in dispatcher."));
          }
        else
          {
            auto dg_matrix = std::make_unique<
              Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, vcycle_number>>(
              mapping,
              level_dof_handlers[level],
              level_constraints[level],
              colored);
            dg_matrix->set_overlap_in_vmult(overlap_in_vmult);
            level_matrices[level] = std::move(dg_matrix);
          }
      }

    if (use_overlap)
      pcout << "Overlapping vmult on levels:"
            << (overlap_levels.empty() ? std::string(" none (all below the threshold)") :
                                         overlap_levels)
            << (color_fine_mesh_levels ? "" : ", levels not colored")
            << std::endl;

    if constexpr (std::is_same_v<full_number, vcycle_number>)
      {
        const auto &system_matrix = *level_matrices.back();
        system_matrix.initialize_dof_vector(solution_device);
      }

    else
      {
        // Downcast the fine-level float matrix to access its topology
        const auto *fine_float_matrix =
          dynamic_cast<const Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, vcycle_number> *>(
            level_matrices.back().get());

        Assert(fine_float_matrix != nullptr, ExcInternalError());

        auto dg_fine_level_matrix =
          std::make_unique<Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, full_number>>();

        dg_fine_level_matrix->reinit_with_shared_topology(*fine_float_matrix,
                                                          mapping,
                                                          level_dof_handlers.back(),
                                                          fine_level_constraints,
                                                          color_fine_mesh_levels);
        dg_fine_level_matrix->set_overlap_in_vmult(fine_float_matrix->get_overlap_in_vmult());
        if (dg_fine_level_matrix->get_overlap_in_vmult())
          pcout << "Overlapping vmult of the outer DG operator" << std::endl;

        fine_level_matrix = std::move(dg_fine_level_matrix);
        fine_level_matrix->initialize_dof_vector(solution_device);
      }

    system_rhs_device.reinit(solution_device);
    //ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
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
        else if (level < level_matrices.max_level())
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
        else
          {
            mg_transfers[level] =
              std::make_unique<Portable::ContinuousTransfer<dim, fe_degree, vcycle_number>>();
            mg_transfers[level]->reinit(level_matrices[level - 1]->get_matrix_free(),
                                        level_matrices[level]->get_matrix_free(),
                                        level_constraints[level - 1],
                                        level_constraints[level]);
          }
      }
    Kokkos::fence();

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

    // The right-hand side f = 1 is integrated on the device by the fine-level
    // DG operator, which already holds the cell measures and DoF indices.
    using FineOperator = Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, full_number>;

    const FineOperator *fine_operator = nullptr;
    if constexpr (std::is_same_v<full_number, vcycle_number>)
      fine_operator = dynamic_cast<const FineOperator *>(level_matrices.back().get());
    else
      fine_operator = dynamic_cast<const FineOperator *>(fine_level_matrix.get());

    AssertThrow(fine_operator != nullptr, ExcInternalError());

    fine_operator->compute_rhs(system_rhs_device, 1.);

    setup_time += time.wall_time();

    time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve(const unsigned int n_pre_smooth,
                                        const unsigned int n_post_smooth)
  {
    std::unique_ptr<
      multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>>
      solver;

    if constexpr (std::is_same_v<full_number, vcycle_number>)
      solver = std::make_unique<
        multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>>(
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
      solver = std::make_unique<
        multigrid::MultigridSolver<dim, fe_degree, vcycle_number, full_number, SmootherType>>(
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
            const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

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
                   const bool         use_doubling_mesh)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem;
          laplace_problem.run(min_size, max_size, n_pre_smooth, n_post_smooth, use_doubling_mesh);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree, min_size, max_size, n_pre_smooth, n_post_smooth, use_doubling_mesh);
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
      unsigned int n_pre_smooth      = 3;
      unsigned int n_post_smooth     = 3;
      bool         use_doubling_mesh = true;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program degree minsize maxsize n_pre_smooth n_post_smooth doubling "
                      << "coarse_granularity overlap overlap_min_dofs_per_rank" << std::endl
                      << "The parameters degree to n_post_smooth are integers, "
                      << "doubling selects between a square mesh or a doubling mesh, "
                      << "coarse_granularity is the minimal number of cells per rank "
                      << "on the coarse levels, overlap (0 or 1) overlaps the ghost "
                      << "exchange with the computation on the levels with at least "
                      << "overlap_min_dofs_per_rank DoFs per rank" << std::endl;
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
        coarse_granularity = std::atoi(argv[7]);
      if (argc > 8)
        use_overlap = std::atoi(argv[8]) == 1;
      if (argc > 9)
        overlap_min_dofs_per_rank = std::atoll(argv[9]);

      AssertThrow(coarse_granularity > 0, ExcMessage("coarse_granularity must be positive."));

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Number of pre-smoother iters:   " << n_pre_smooth << std::endl
                  << "Number of post-smoother iters:  " << n_post_smooth << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh << std::endl
                  << "Coarse level granularity:       " << coarse_granularity << std::endl
                  << "Overlap comm./comp.:            " << use_overlap << std::endl
                  << "Overlap min DoFs per rank:      " << overlap_min_dofs_per_rank << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(
        degree, minsize, maxsize, n_pre_smooth, n_post_smooth, use_doubling_mesh);
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
