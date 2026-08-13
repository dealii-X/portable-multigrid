#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"
#include "domain_decomposition/portable_bddc_preconditioner.h"
#include "domain_decomposition/portable_bnn_preconditioner.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_projected_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "multigrid/portable_projected_chebyshev_smoother.h"
#include "multigrid/portable_projected_diagonal_preconditioner.h"
#include "multigrid/portable_projected_jacobi_smoother.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"
#include "operators/portable_subdomain_laplace_operator.h"
#include "operators/portable_subdomain_neumann_operator_wrapper.h"



using namespace dealii;

// GeometricTransfer/PolynomialTransfer::reinit_primal_pinned() takes the
// extra pinned dofs as a plain host-side list; SubdomainBDDCOperator's own
// mask is a device view (get_primal_pinned_dof_indices_subdomain()) --
// mirrored to host once here rather than duplicating this at each of the
// two call sites (h-level and p-level transfer setup) below.
template <int dim>
std::vector<unsigned int>
primal_pinned_dofs_host(const Portable::SubdomainBDDCOperator<dim, double> &op)
{
  auto host_view = Kokkos::create_mirror_view_and_copy(
    Kokkos::HostSpace(), op.get_primal_pinned_dof_indices_subdomain());

  std::vector<unsigned int> result(host_view.extent(0));
  for (unsigned int i = 0; i < result.size(); ++i)
    result[i] = host_view(i);
  return result;
}

// Same role as primal_pinned_dofs_host() above, but for the corner-only mask
// (SubdomainBDDCOperator::get_corner_pinned_dof_indices_subdomain()).
template <int dim>
std::vector<unsigned int>
corner_pinned_dofs_host(const Portable::SubdomainBDDCOperator<dim, double> &op)
{
  auto host_view = Kokkos::create_mirror_view_and_copy(
    Kokkos::HostSpace(), op.get_corner_pinned_dof_indices_subdomain());

  std::vector<unsigned int> result(host_view.extent(0));
  for (unsigned int i = 0; i < result.size(); ++i)
    result[i] = host_view(i);
  return result;
}


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int n_pre_smooth, const unsigned int n_post_smooth);

  // Selects BDDCPreconditioner's fine-correction algorithm (see
  // BDDCPreconditioner::set_fine_correction_mode()): false = the original
  // Ahat/project() path, true = classical corner-pin + edge/face-Lagrange-
  // multiplier static condensation. run() below solves the SAME problem
  // with both settings and compares -- this member is only read/set
  // internally by run() now, not toggled externally between calls the way
  // the source/bddc_preconditioner/program.cc original uses it.
  bool use_static_condensation_fine_correction = false;

  // Runs the same fixed problem twice (Pi-projected fine correction, then
  // static condensation) and compares the two outer interface-CG solutions:
  // they must agree, since both solve the SAME linear system
  // interface_operator * x = rhs_schur_device -- the preconditioner only
  // affects the convergence path, not the fixed point CG converges to.
  // Also reports both solves' iteration counts (informational: the two
  // fine-correction algorithms are different, not required to need the
  // same number of iterations, just to converge to the same answer).
  // Returns true iff the comparison passes.
  bool
  run();

  void
  test_bddc();

private:
  void
  create_subdomain_triangulations(unsigned int n_refinement_cycles);

  void
  setup_dofs();

  void
  compute_interface_weights();

  void
  setup_matrix_free();

  void
  setup_mg_transfers();

  void
  setup_smoothers();

  void
  setup_mg_preconditioners();

  void
  setup_interface_system();

  void
  setup_bddc_preconditioner();

  void
  assemble_rhs();

  // Returns the outer interface-CG iteration count (solver_control.last_step())
  // -- used by the correctness comparison in run() below.
  unsigned int
  solve_interface();

  void
  matvec_ghost_timing();

  // Microbenchmark isolating the cost of the individual kernels the fine-
  // correction CG solve (and the V-cycle preconditioning it) is built
  // from: the BDDC operator's vmult()/vmult_plain()/project(), each
  // level's diagonal preconditioner vmult() (fused_scale() + project())
  // and fused_scale() alone, the raw Dirichlet operator's vmult(), and the
  // Neumann operator's vmult_neumann() -- looped over every MG level, not
  // just the finest one, since the V-cycle applies all of these at every
  // level. Calls each in isolation (outside any CG solve) n_mv times per
  // rep -- same "best of n_reps" MPI-max pattern as matvec_ghost_timing()
  // -- rather than instrumenting SolverCG's internals, so it approximates
  // per-application cost without threading timers through dealii's solver.
  void
  fine_correction_component_timing(const unsigned int cycle);

  void
  postprocess_subdomain_solution();

  void
  output_results(const unsigned int cycle) const;


  MPI_Comm mpi_communicator;

  parallel::fullydistributed::Triangulation<dim> triangulation;

  FE_Q<dim> fe;

  MGLevelObject<std::unique_ptr<FE_Q<dim>>> p_level_fes;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::vector<std::shared_ptr<SubdomainTriangulation<dim>>> level_subdomain_triangulations;
  std::vector<std::shared_ptr<parallel::fullydistributed::Triangulation<dim>>> level_triangulations;

  MGLevelObject<SubdomainDoFHandler<dim>> level_subdomain_dof_handlers;
  MGLevelObject<DoFHandler<dim>>          level_distributed_dof_handlers;

  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints;
  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints_physical;

  using VectorTypeMG = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  using LevelMatrixType = Portable::SubdomainLaplaceOperatorBase<dim, double>;

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;

  using BddcPreconditionerType =
    Portable::ProjectedDiagonalPreconditioner<LevelMatrixType, VectorTypeMG>;

  using BddcSmootherType =
    Portable::ProjectedChebyshevSmoother<LevelMatrixType, BddcPreconditionerType, VectorTypeMG>;

  using TransferType = Portable::MGTransferBase<dim, double>;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_matrices;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_neumann_matrices;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_bddc_matrices;

  // A_RR (static-condensation) hierarchy: thin adapters (see
  // SubdomainBDDCOperatorARRAdapter in portable_subdomain_bddc_operator_
  // wrapper.h) presenting each level's *existing* level_subdomain_bddc_
  // matrices[level]->vmult_primal_pinned() as vmult() -- no separate
  // operator/MatrixFree is built per level, these wrap the same underlying
  // SubdomainBDDCOperator instances the Pi-projected hierarchy already
  // owns. A plain (non-projected) PreconditionChebyshev/DiagonalMatrix
  // smoother suffices here (see SmootherType below), unlike the BDDC
  // hierarchy's ProjectedChebyshevSmoother, since A_RR is genuinely
  // nonsingular and needs none of the singular-Ahat workarounds.
  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_primal_pinned_matrices;

  // Corner-only-pinned hierarchy (classical BDDC construction): same idea as
  // level_subdomain_primal_pinned_matrices above -- thin adapters (see
  // SubdomainBDDCOperatorCornerPinnedAdapter) over the SAME level_subdomain_
  // bddc_matrices instances, but pinning corners only (edges/faces stay
  // free). Used to realize A_RR^{-1} in BDDCPreconditioner::
  // compute_local_edge_face_schur_complement()/vmult_fine_correction_
  // static_condensation(); see set_fine_correction_mode()'s class-level
  // comment in portable_bddc_preconditioner.h for the surrounding math.
  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_corner_pinned_matrices;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_dirichlet;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_neumann;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_bddc;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_primal_pinned;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_corner_pinned;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_dirichlet;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_neumann;

  MGLevelObject<BddcSmootherType> subdomain_mg_smoothers_bddc;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_primal_pinned;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_corner_pinned;

  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>> subdomain_mg_preconditioner_dirichlet;
  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>> subdomain_mg_preconditioner_neumann;
  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>> subdomain_mg_preconditioner_bddc;
  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>>
    subdomain_mg_preconditioner_primal_pinned;
  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>>
    subdomain_mg_preconditioner_corner_pinned;

  std::unique_ptr<Portable::SchurInterfaceOperator<dim, double>> interface_operator;

  std::unique_ptr<Portable::BNNPreconditioner<dim, double>> bnn_preconditioner;

  std::unique_ptr<Portable::BDDCPreconditioner<dim, double, BddcSmootherType>> bddc_preconditioner;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> global_solution_host,
    subdomain_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> subdomain_solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> subdomain_rhs_device, schur_rhs;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> rhs_schur_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> solution_interface_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> global_interface_weights;

  const unsigned int n_pre_smooth;
  const unsigned int n_post_smooth;

  // Toggle for the subdomain MatrixFree objects (Laplace operator + p-MG
  // transfer) used by the expensive fine correction: when true, a real
  // graph coloring (MatrixFree::AdditionalData::use_coloring) is built so
  // the BK3/BK1 scatter kernels can accumulate directly instead of going
  // through Kokkos::atomic_add. overlap_communication_computation is not a
  // relevant alternative here since these solves are purely local (no MPI
  // communication happens inside a subdomain vmult).
  static constexpr bool use_coloring_for_subdomain_solvers = false;

  double             setup_time;
  ConditionalOStream pcout;
  ConditionalOStream time_details;

  ConvergenceTable timing_table;

  ConvergenceTable timing_table_per_iteration;

  ConvergenceTable ghost_timing_table;

  ConvergenceTable fine_correction_component_timing_table;

  ConvergenceTable bddc_setup_timing_table;

  // Per-rank diagnostics, repopulated (cleared + refilled) each cycle and
  // printed together once at the end of that cycle instead of inline,
  // to avoid interleaving a 1-row-per-rank table into the middle of the
  // per-phase timing log.
  ConvergenceTable per_rank_dof_table;
  ConvergenceTable per_rank_load_table;

  unsigned int n_cells_total;

  struct SubdomainLaplaceOperatorRunner
  {
    const unsigned int              level;
    SubdomainDoFHandler<dim>       &subomain_dof_handler;
    AffineConstraints<double>      &constraints;
    AffineConstraints<double>      &constraints_physical;
    bool                            use_coloring;
    LaplaceProblem<dim, fe_degree> &parent_problem;

    template <unsigned int degree>
    void
    run()
    {
      parent_problem.level_subdomain_matrices[level] =
        std::make_unique<Portable::SubdomainLaplaceOperator<dim, degree, double>>(
          subomain_dof_handler, constraints, constraints_physical, use_coloring);


      parent_problem.level_subdomain_neumann_matrices[level] =
        std::make_unique<typename Portable::SubdomainNeumannOperatorWrapper<dim, degree, double>>(
          *parent_problem.level_subdomain_matrices[level]);

      parent_problem.level_subdomain_bddc_matrices[level] =
        std::make_unique<typename Portable::SubdomainBDDCOperator<dim, double>>(
          *parent_problem.level_subdomain_matrices[level]);

      parent_problem.level_subdomain_primal_pinned_matrices[level] =
        std::make_unique<typename Portable::SubdomainBDDCOperatorARRAdapter<dim, double>>(
          static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
            *parent_problem.level_subdomain_bddc_matrices[level]));

      parent_problem.level_subdomain_corner_pinned_matrices[level] =
        std::make_unique<typename Portable::SubdomainBDDCOperatorCornerPinnedAdapter<dim, double>>(
          static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
            *parent_problem.level_subdomain_bddc_matrices[level]));
    }
  };

  struct PolynomialTransferRunner
  {
    const unsigned int                       level;
    const Portable::MatrixFree<dim, double> &mf_coarse;
    const Portable::MatrixFree<dim, double> &mf_fine;
    AffineConstraints<double>               &constraints_coarse;
    AffineConstraints<double>               &constraints_fine;
    AffineConstraints<double>               &physical_constraints_coarse;
    AffineConstraints<double>               &physical_constraints_fine;

    LaplaceProblem<dim, fe_degree> &parent_problem;

    template <unsigned int degree_coarse, unsigned int degree_fine>
    void
    run()
    {
      parent_problem.subdomain_mg_transfers_dirichlet[level] =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();

      parent_problem.subdomain_mg_transfers_dirichlet[level]->reinit(mf_coarse,
                                                                     mf_fine,
                                                                     constraints_coarse,
                                                                     constraints_fine);

      parent_problem.subdomain_mg_transfers_neumann[level] =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();


      parent_problem.subdomain_mg_transfers_neumann[level]->reinit(mf_coarse,
                                                                   mf_fine,
                                                                   physical_constraints_coarse,
                                                                   physical_constraints_fine);


      parent_problem.subdomain_mg_transfers_bddc[level] =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();

      parent_problem.subdomain_mg_transfers_bddc[level]->reinit(mf_coarse,
                                                                mf_fine,
                                                                physical_constraints_coarse,
                                                                physical_constraints_fine);

      {
        const auto &bddc_coarse = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
          *parent_problem.level_subdomain_bddc_matrices[level - 1]);
        const auto &bddc_fine = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
          *parent_problem.level_subdomain_bddc_matrices[level]);

        auto transfer =
          std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();
        transfer->reinit_primal_pinned(mf_coarse,
                                       mf_fine,
                                       physical_constraints_coarse,
                                       primal_pinned_dofs_host(bddc_coarse),
                                       physical_constraints_fine,
                                       primal_pinned_dofs_host(bddc_fine));
        parent_problem.subdomain_mg_transfers_primal_pinned[level] = std::move(transfer);
      }

      {
        const auto &bddc_coarse = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
          *parent_problem.level_subdomain_bddc_matrices[level - 1]);
        const auto &bddc_fine = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
          *parent_problem.level_subdomain_bddc_matrices[level]);

        auto transfer =
          std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();
        transfer->reinit_primal_pinned(mf_coarse,
                                       mf_fine,
                                       physical_constraints_coarse,
                                       corner_pinned_dofs_host(bddc_coarse),
                                       physical_constraints_fine,
                                       corner_pinned_dofs_host(bddc_fine));
        parent_problem.subdomain_mg_transfers_corner_pinned[level] = std::move(transfer);
      }
    }
  };
};
template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(const unsigned int n_pre_smooth,
                                               const unsigned int n_post_smooth)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , n_pre_smooth(n_pre_smooth)
  , n_post_smooth(n_post_smooth)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout, true && Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
  Assert(n_pre_smooth == n_post_smooth,
         ExcNotImplemented("Change of pre- and post-smoother degree "
                           "currently not possible with deal.II"));
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_subdomain_triangulations(unsigned int n_refinement_cycles)
{
  Timer time;

  const unsigned int n_subdomains = Utilities::MPI::n_mpi_processes(mpi_communicator);

  std::vector<unsigned int> subdomains_per_axis(dim);

  int remaining = n_subdomains;
  for (int d = dim; d > 0; --d)
    {
      int n_this_axis = std::pow(remaining, 1.0 / d);

      int best_divisor = 1;
      for (int j = n_this_axis; j >= 1; --j)
        if (remaining % j == 0)
          {
            best_divisor = j;
            break;
          }
      subdomains_per_axis[d - 1] = best_divisor;

      remaining /= best_divisor;
    }


  Triangulation<dim> coarse_triangulation;

  Point<dim> p1, p2;
  for (int d = 0; d < dim; ++d)
    p2[d] = 1.;

  GridGenerator::subdivided_hyper_rectangle(coarse_triangulation, subdomains_per_axis, p1, p2);

  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  this->level_subdomain_triangulations.clear();
  this->level_triangulations.clear();

  for (unsigned int cycle = 0; cycle <= n_refinement_cycles; ++cycle)
    {
      if (cycle > 0)
        coarse_triangulation.refine_global(1);

      n_cells_total = coarse_triangulation.n_global_active_cells();

      const TriangulationDescription::Description<dim> description =
        TriangulationDescription::Utilities::create_description_from_triangulation(
          coarse_triangulation, mpi_communicator);

      this->triangulation.clear();
      this->triangulation.create_triangulation(description);

      this->level_triangulations.push_back(
        std::make_shared<parallel::fullydistributed::Triangulation<dim>>(mpi_communicator));
      this->level_triangulations.back()->create_triangulation(description);

      this->level_subdomain_triangulations.push_back(
        std::make_shared<SubdomainTriangulation<dim>>());

      if (cycle == 0)
        this->level_subdomain_triangulations.back()->create_subdomain_triangulation(triangulation);
      else
        {
          this->level_subdomain_triangulations.back()->copy_subdomain_triangulation(
            *level_subdomain_triangulations[cycle - 1]);
          this->level_subdomain_triangulations.back()->refine_global(1);
        }
    }
  setup_time += time.wall_time();

  pcout << "                      N_cells = " << triangulation.n_global_active_cells() << std::endl
        << std::endl;

  time_details << "                      Subdomain triangulations extracted        (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  // const double subdomain_diameter = Utilities::MPI::max(
  //   GridTools::diameter(
  //     level_subdomain_triangulations.back()->get_triangulation()),
  //   mpi_communicator);

  // const double subdomain_mesh_size = Utilities::MPI::max(
  //   GridTools::maximal_cell_diameter(
  //     level_subdomain_triangulations.back()->get_triangulation()),
  //   mpi_communicator);


  // pcout << "H/h = " << subdomain_diameter / subdomain_mesh_size << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  Timer time;

  const unsigned int n_h_levels = static_cast<unsigned int>(level_subdomain_triangulations.size());

  std::vector<unsigned int> p_levels({fe.degree});

  while (p_levels.back() > 1)
    p_levels.push_back(std::max(p_levels.back() / 2, 1u));

  p_level_fes.resize(0, p_levels.size() - 1);

  for (unsigned int level = 0; level < p_levels.size(); ++level)
    p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

  level_subdomain_dof_handlers.resize(0, n_h_levels - 1 + p_level_fes.max_level());
  level_distributed_dof_handlers.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_constraints.resize(0, level_subdomain_dof_handlers.max_level());
  level_subdomain_constraints_physical.resize(0, level_subdomain_dof_handlers.max_level());

  Functions::ZeroFunction<dim>                        homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
    {types::boundary_id(0), &homogeneous_dirichlet_bc},
    {level_subdomain_triangulations.back()->get_interface_id(), &homogeneous_dirichlet_bc}};
  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions_physical = {
    {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  for (unsigned int level = 0; level <= level_subdomain_dof_handlers.max_level(); ++level)
    {
      DoFHandler<dim> &dof_h = level_distributed_dof_handlers[level];

      dof_h.reinit(*level_triangulations[std::min(level, n_h_levels - 1)]);

      if (level < n_h_levels)
        dof_h.distribute_dofs(*p_level_fes[0]);
      else
        dof_h.distribute_dofs(*p_level_fes[level + 1 - n_h_levels]);

      SubdomainDoFHandler<dim> &subdomain_dof_h = level_subdomain_dof_handlers[level];

      subdomain_dof_h.reinit(level_subdomain_triangulations[std::min(level, n_h_levels - 1)],
                             dof_h);
      subdomain_dof_h.distribute_subdomain_dofs();


      {
        AffineConstraints<double> &constraints = level_subdomain_constraints[level];

        constraints.clear();

        DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(), constraints);
        VectorTools::interpolate_boundary_values(subdomain_dof_h.get_dof_handler(),
                                                 dirichlet_boundary_functions,
                                                 constraints);
        constraints.close();
      }

      {
        AffineConstraints<double> &constraints_physical =
          level_subdomain_constraints_physical[level];

        constraints_physical.clear();

        DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(),
                                                constraints_physical);
        VectorTools::interpolate_boundary_values(subdomain_dof_h.get_dof_handler(),
                                                 dirichlet_boundary_functions_physical,
                                                 constraints_physical);
        constraints_physical.close();
      }
    }

  locally_owned_dofs = level_distributed_dof_handlers.back().locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(level_distributed_dof_handlers.back());

  pcout << "                      Total number of DoFs: "
        << level_distributed_dof_handlers.back().n_dofs() << std::endl;

  {
    const auto &finest_subdomain_dof_handler = level_subdomain_dof_handlers.back();

    const unsigned int n_subdomain_dofs = finest_subdomain_dof_handler.get_dof_handler().n_dofs();

    const auto &interface_partitioner =
      finest_subdomain_dof_handler.get_interface_vector_partitioner();

    const unsigned int n_interface_owned =
      interface_partitioner ? interface_partitioner->locally_owned_size() : 0;
    const unsigned int n_interface_ghost =
      interface_partitioner ? interface_partitioner->n_ghost_indices() : 0;

    const auto all_n_subdomain_dofs = Utilities::MPI::gather(mpi_communicator, n_subdomain_dofs, 0);
    const auto all_n_interface_owned =
      Utilities::MPI::gather(mpi_communicator, n_interface_owned, 0);
    const auto all_n_interface_ghost =
      Utilities::MPI::gather(mpi_communicator, n_interface_ghost, 0);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        per_rank_dof_table.clear();
        for (unsigned int rank = 0; rank < all_n_subdomain_dofs.size(); ++rank)
          {
            per_rank_dof_table.add_value("rank", rank);
            per_rank_dof_table.add_value("subdomain_dofs", all_n_subdomain_dofs[rank]);
            per_rank_dof_table.add_value("interface_owned", all_n_interface_owned[rank]);
            per_rank_dof_table.add_value("interface_ghost", all_n_interface_ghost[rank]);
          }
      }
  }

  global_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  subdomain_solution_host.reinit(level_subdomain_dof_handlers.back().get_dof_handler().n_dofs());

  setup_time += time.wall_time();
  time_details << "                      Subdomain DoFs setup                      (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_interface_weights()
{
  if (level_subdomain_dof_handlers.back().get_interface_vector_partitioner() == nullptr)
    return;

  const auto &subdomain_dof_h_fine = level_subdomain_dof_handlers.back();

  subdomain_dof_h_fine.initialize_interface_dof_vector(global_interface_weights);

  const unsigned int n_locally_relevant_interface_indices =
    subdomain_dof_h_fine.n_locally_relevant_interface_indices();

  for (unsigned int i = 0; i < n_locally_relevant_interface_indices; ++i)
    global_interface_weights[subdomain_dof_h_fine.local_to_global_interface_partitioner(i)] += 1.0;

  global_interface_weights.compress(VectorOperation::add);

  for (unsigned int i = 0; i < global_interface_weights.locally_owned_size(); ++i)
    global_interface_weights.local_element(i) = 1. / global_interface_weights.local_element(i);

  global_interface_weights.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  const unsigned int n_h_levels = static_cast<unsigned int>(level_triangulations.size());

  Kokkos::fence();
  Timer time;

  level_subdomain_matrices.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_neumann_matrices.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_bddc_matrices.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_primal_pinned_matrices.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_corner_pinned_matrices.resize(0, level_subdomain_dof_handlers.max_level());


  for (unsigned int level = 0; level <= level_subdomain_dof_handlers.max_level(); ++level)
    {
      if (level < n_h_levels)
        {
          level_subdomain_matrices[level] =
            std::make_unique<Portable::SubdomainLaplaceOperator<dim, 1, double>>(
              level_subdomain_dof_handlers[level],
              level_subdomain_constraints[level],
              level_subdomain_constraints_physical[level],
              use_coloring_for_subdomain_solvers);


          level_subdomain_neumann_matrices[level] =
            std::make_unique<typename Portable::SubdomainNeumannOperatorWrapper<dim, 1, double>>(
              *level_subdomain_matrices[level]);

          level_subdomain_bddc_matrices[level] =
            std::make_unique<typename Portable::SubdomainBDDCOperator<dim, double>>(
              *level_subdomain_matrices[level]);

          level_subdomain_primal_pinned_matrices[level] =
            std::make_unique<typename Portable::SubdomainBDDCOperatorARRAdapter<dim, double>>(
              static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
                *level_subdomain_bddc_matrices[level]));

          level_subdomain_corner_pinned_matrices[level] =
            std::make_unique<typename Portable::SubdomainBDDCOperatorCornerPinnedAdapter<dim, double>>(
              static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
                *level_subdomain_bddc_matrices[level]));
        }
      else
        {
          SubdomainLaplaceOperatorRunner runner{level,
                                                level_subdomain_dof_handlers[level],
                                                level_subdomain_constraints[level],
                                                level_subdomain_constraints_physical[level],
                                                use_coloring_for_subdomain_solvers,
                                                *this};


          bool success = Portable::SubdomainOperatorDispatchFactory::dispatch(
            p_level_fes[level + 1 - n_h_levels]->degree, runner);

          Assert(success, ExcMessage("Failed to find a matching polynomial degree in dispatcher."));
        }
    }

  level_subdomain_matrices.back()->initialize_dof_vector(subdomain_solution_device);
  level_subdomain_matrices.back()->initialize_dof_vector(subdomain_rhs_device);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      Matrix-free operators setup               (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_transfers()
{
  Kokkos::fence();
  Timer time;

  const unsigned int n_h_levels = static_cast<unsigned int>(level_subdomain_triangulations.size());

  subdomain_mg_transfers_dirichlet.resize(level_subdomain_matrices.min_level(),
                                          level_subdomain_matrices.max_level());

  subdomain_mg_transfers_neumann.resize(level_subdomain_matrices.min_level(),
                                        level_subdomain_matrices.max_level());


  subdomain_mg_transfers_bddc.resize(level_subdomain_matrices.min_level(),
                                     level_subdomain_matrices.max_level());

  subdomain_mg_transfers_primal_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  subdomain_mg_transfers_corner_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level() + 1;
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      if (level < n_h_levels)
        {
          subdomain_mg_transfers_dirichlet[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          subdomain_mg_transfers_dirichlet[level]->reinit(
            level_subdomain_matrices[level - 1]->get_matrix_free(),
            level_subdomain_matrices[level]->get_matrix_free(),
            level_subdomain_constraints[level - 1],
            level_subdomain_constraints[level]);

          subdomain_mg_transfers_neumann[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          subdomain_mg_transfers_neumann[level]->reinit(
            level_subdomain_matrices[level - 1]->get_matrix_free(),
            level_subdomain_matrices[level]->get_matrix_free(),
            level_subdomain_constraints_physical[level - 1],
            level_subdomain_constraints_physical[level]);

          subdomain_mg_transfers_bddc[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          subdomain_mg_transfers_bddc[level]->reinit(
            level_subdomain_matrices[level - 1]->get_matrix_free(),
            level_subdomain_matrices[level]->get_matrix_free(),
            level_subdomain_constraints_physical[level - 1],
            level_subdomain_constraints_physical[level]);

          {
            const auto &bddc_coarse = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
              *level_subdomain_bddc_matrices[level - 1]);
            const auto &bddc_fine = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
              *level_subdomain_bddc_matrices[level]);

            auto transfer = std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
            transfer->reinit_primal_pinned(level_subdomain_matrices[level - 1]->get_matrix_free(),
                                           level_subdomain_matrices[level]->get_matrix_free(),
                                           level_subdomain_constraints_physical[level - 1],
                                           primal_pinned_dofs_host(bddc_coarse),
                                           level_subdomain_constraints_physical[level],
                                           primal_pinned_dofs_host(bddc_fine));
            subdomain_mg_transfers_primal_pinned[level] = std::move(transfer);
          }

          {
            const auto &bddc_coarse = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
              *level_subdomain_bddc_matrices[level - 1]);
            const auto &bddc_fine = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
              *level_subdomain_bddc_matrices[level]);

            auto transfer = std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
            transfer->reinit_primal_pinned(level_subdomain_matrices[level - 1]->get_matrix_free(),
                                           level_subdomain_matrices[level]->get_matrix_free(),
                                           level_subdomain_constraints_physical[level - 1],
                                           corner_pinned_dofs_host(bddc_coarse),
                                           level_subdomain_constraints_physical[level],
                                           corner_pinned_dofs_host(bddc_fine));
            subdomain_mg_transfers_corner_pinned[level] = std::move(transfer);
          }
        }
      else
        {
          const unsigned int p_coarse = p_level_fes[level - n_h_levels]->degree;
          const unsigned int p_fine   = p_level_fes[level + 1 - n_h_levels]->degree;

          PolynomialTransferRunner runner{level,
                                          level_subdomain_matrices[level - 1]->get_matrix_free(),
                                          level_subdomain_matrices[level]->get_matrix_free(),
                                          level_subdomain_constraints[level - 1],
                                          level_subdomain_constraints[level],
                                          level_subdomain_constraints_physical[level - 1],
                                          level_subdomain_constraints_physical[level],
                                          *this};

          bool success =
            Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse, p_fine, runner);

          Assert(success,
                 ExcMessage("Failed to find a matching polynomial degree "
                            "pair in transfer dispatcher."));
        }
    }

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      MG transfers setup                        (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_smoothers()
{
  Kokkos::fence();
  Timer time;

  subdomain_mg_smoothers_dirichlet.resize(level_subdomain_matrices.min_level(),
                                          level_subdomain_matrices.max_level());

  subdomain_mg_smoothers_neumann.resize(level_subdomain_matrices.min_level(),
                                        level_subdomain_matrices.max_level());


  subdomain_mg_smoothers_bddc.resize(level_subdomain_matrices.min_level(),
                                     level_subdomain_matrices.max_level());

  subdomain_mg_smoothers_primal_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  subdomain_mg_smoothers_corner_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level();
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      typename SmootherType::AdditionalData     smoother_data_dirichlet;
      typename SmootherType::AdditionalData     smoother_data_neumann;
      typename SmootherType::AdditionalData     smoother_data_primal_pinned;
      typename SmootherType::AdditionalData     smoother_data_corner_pinned;
      typename BddcSmootherType::AdditionalData smoother_data_bddc;

      unsigned int bddc_eig_cg_n_iterations;

      if (level > 0)
        {
          smoother_data_dirichlet.smoothing_range     = 15.;
          smoother_data_dirichlet.degree              = n_pre_smooth;
          smoother_data_dirichlet.eig_cg_n_iterations = 10;

          smoother_data_neumann.smoothing_range     = 15.;
          smoother_data_neumann.degree              = n_pre_smooth;
          smoother_data_neumann.eig_cg_n_iterations = 10;

          // A_RR is nonsingular (pinning the primal dofs removes exactly
          // the null space a floating subdomain's plain Neumann operator
          // would otherwise have -- see the design discussion on
          // SubdomainBDDCOperator::vmult_primal_pinned()), so it needs
          // none of the singular-Ahat special-casing BDDC's own smoother
          // below does -- same plain treatment as Dirichlet/Neumann.
          smoother_data_primal_pinned.smoothing_range     = 15.;
          smoother_data_primal_pinned.degree              = n_pre_smooth;
          smoother_data_primal_pinned.eig_cg_n_iterations = 10;

          // Corner-pinned A_RR (corners only) is unconditionally nonsingular
          // too (see vmult_corner_pinned()'s class comment) -- same plain
          // treatment.
          smoother_data_corner_pinned.smoothing_range     = 15.;
          smoother_data_corner_pinned.degree              = n_pre_smooth;
          smoother_data_corner_pinned.eig_cg_n_iterations = 10;

          // Deliberately not the true condition number: smoothing_range
          // 5-20 is the usual MG heuristic to focus the smoother on
          // high-frequency modes only and leave the low frequencies to the
          // coarse-grid correction.
          smoother_data_bddc.smoothing_range = 15.;
          smoother_data_bddc.degree          = n_pre_smooth;
          bddc_eig_cg_n_iterations           = 10;
        }
      else
        {
          smoother_data_dirichlet.smoothing_range     = 1e-3;
          smoother_data_dirichlet.degree              = numbers::invalid_unsigned_int;
          smoother_data_dirichlet.eig_cg_n_iterations = level_subdomain_matrices[0]->m();

          smoother_data_neumann.smoothing_range     = 1e-3;
          smoother_data_neumann.degree              = numbers::invalid_unsigned_int;
          smoother_data_neumann.eig_cg_n_iterations = level_subdomain_matrices[0]->m();

          smoother_data_primal_pinned.smoothing_range     = 1e-3;
          smoother_data_primal_pinned.degree              = numbers::invalid_unsigned_int;
          smoother_data_primal_pinned.eig_cg_n_iterations = level_subdomain_matrices[0]->m();

          smoother_data_corner_pinned.smoothing_range     = 1e-3;
          smoother_data_corner_pinned.degree              = numbers::invalid_unsigned_int;
          smoother_data_corner_pinned.eig_cg_n_iterations = level_subdomain_matrices[0]->m();

          // ProjectedChebyshevSmoother always uses a fixed degree (no
          // degree = invalid_unsigned_int auto-selection); a large fixed
          // degree covers this. smoothing_range is set below from the
          // genuine min/max eigenvalue bounds instead of a guess, since an
          // "exact" solve wants the true spectral range, not just the
          // high-frequency end.
          smoother_data_bddc.degree = 100;
          bddc_eig_cg_n_iterations  = level_subdomain_bddc_matrices[0]->m();
        }

      level_subdomain_matrices[level]->compute_diagonal();

      level_subdomain_bddc_matrices[level]->compute_diagonal();

      smoother_data_dirichlet.preconditioner =
        level_subdomain_matrices[level]->get_matrix_diagonal_inverse();

      smoother_data_neumann.preconditioner =
        level_subdomain_matrices[level]->get_matrix_diagonal_inverse_neumann();

      // Reads the *primal-pinned* diagonal SubdomainBDDCOperator::
      // compute_diagonal() (called above) also builds -- diag=1 forced at
      // primal-pinned dofs, distinct from the plain diagonal the Pi-
      // projected path uses (see SubdomainBDDCOperatorARRAdapter's class
      // comment for why that distinction matters here specifically).
      // compute_diagonal() on the adapter itself is intentionally
      // DEAL_II_NOT_IMPLEMENTED().
      smoother_data_primal_pinned.preconditioner =
        level_subdomain_primal_pinned_matrices[level]->get_matrix_diagonal_inverse();

      // Reads the *corner-pinned* diagonal (diag=1 forced at corner-pinned
      // dofs only) SubdomainBDDCOperator::compute_diagonal() (called above)
      // also builds -- same reasoning as smoother_data_primal_pinned above.
      smoother_data_corner_pinned.preconditioner =
        level_subdomain_corner_pinned_matrices[level]->get_matrix_diagonal_inverse();

      smoother_data_bddc.preconditioner = std::make_shared<BddcPreconditionerType>(
        *level_subdomain_bddc_matrices[level],
        level_subdomain_bddc_matrices[level]->get_matrix_diagonal_inverse());

      // ProjectedChebyshevSmoother takes max_eigenvalue (and, at the
      // coarsest level, smoothing_range) directly rather than estimating
      // them internally. dealii::PreconditionChebyshev's default Lanczos
      // estimator runs an actual CG solve seeded with a generic,
      // non-V-projected vector, which hits an exact-zero denominator and
      // returns NaN since A-hat = Pi*A*Pi is singular outside V = range(Pi).
      // estimate_eigenvalue_bounds() runs the same kind of
      // SolverCG::connect_eigenvalues_slot()-based Lanczos estimate
      // dealii::PreconditionChebyshev uses (with the same safety_factor),
      // but seeded with a probe vector genuinely projected into V first, so
      // the CG solve stays well-posed.
      {
        VectorTypeMG eigenvector;
        level_subdomain_bddc_matrices[level]->initialize_dof_vector(eigenvector);
        dealii::internal::set_initial_guess(eigenvector);

        const auto bounds =
          Portable::estimate_eigenvalue_bounds(*level_subdomain_bddc_matrices[level],
                                               *smoother_data_bddc.preconditioner,
                                               eigenvector,
                                               bddc_eig_cg_n_iterations);

        smoother_data_bddc.max_eigenvalue = bounds.max_eigenvalue;

        if (level == 0)
          smoother_data_bddc.smoothing_range =
            (bounds.min_eigenvalue > 0.) ? (bounds.max_eigenvalue / bounds.min_eigenvalue) : 1.;
      }

      subdomain_mg_smoothers_dirichlet[level].initialize(*level_subdomain_matrices[level],
                                                         smoother_data_dirichlet);

      subdomain_mg_smoothers_neumann[level].initialize(*level_subdomain_neumann_matrices[level],
                                                       smoother_data_neumann);

      subdomain_mg_smoothers_bddc[level].initialize(*level_subdomain_bddc_matrices[level],
                                                    smoother_data_bddc);

      subdomain_mg_smoothers_primal_pinned[level].initialize(
        *level_subdomain_primal_pinned_matrices[level], smoother_data_primal_pinned);

      subdomain_mg_smoothers_corner_pinned[level].initialize(
        *level_subdomain_corner_pinned_matrices[level], smoother_data_corner_pinned);

      // LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src,dst;

      // level_subdomain_bddc_matrices[level]->initialize_dof_vector(src);

      // const auto eig_info = subdomain_mg_smoothers_bddc[level].estimate_eigenvalues(src);
      // src = 1.0;
    }


  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      Smoothers setup                           (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_preconditioners()
{
  Kokkos::fence();
  Timer time;

  subdomain_mg_preconditioner_dirichlet = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_matrices, subdomain_mg_transfers_dirichlet, subdomain_mg_smoothers_dirichlet);

  const bool impose_zero_mean =
    level_subdomain_matrices.back()->get_physical_boundary_dof_indices_subdomain().size() == 0;

  subdomain_mg_preconditioner_neumann = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_neumann_matrices,
    subdomain_mg_transfers_neumann,
    subdomain_mg_smoothers_neumann,
    impose_zero_mean);

  subdomain_mg_preconditioner_bddc = std::make_unique<
    Portable::
      SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, BddcSmootherType>>(
    level_subdomain_bddc_matrices, subdomain_mg_transfers_bddc, subdomain_mg_smoothers_bddc, false);

  // impose_zero_mean=false: unlike Neumann, A_RR is never singular (pinning
  // primal dofs already removes any floating-subdomain null space), so it
  // never needs this.
  subdomain_mg_preconditioner_primal_pinned = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_primal_pinned_matrices,
    subdomain_mg_transfers_primal_pinned,
    subdomain_mg_smoothers_primal_pinned,
    false);

  // impose_zero_mean=false: corner-pinned A_RR is unconditionally nonsingular
  // too (see vmult_corner_pinned()'s class comment), same reasoning as
  // subdomain_mg_preconditioner_primal_pinned above.
  subdomain_mg_preconditioner_corner_pinned = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_corner_pinned_matrices,
    subdomain_mg_transfers_corner_pinned,
    subdomain_mg_smoothers_corner_pinned,
    false);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      MG Preconditioners setup                  (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_interface_system()
{
  Kokkos::fence();
  Timer time;

  interface_operator = std::make_unique<Portable::SchurInterfaceOperator<dim, double>>(
    *level_subdomain_matrices.back(), *subdomain_mg_preconditioner_dirichlet);

  rhs_schur_device.reinit(
    this->level_subdomain_dof_handlers.back().get_interface_vector_partitioner());

  solution_interface_device.reinit(
    this->level_subdomain_dof_handlers.back().get_interface_vector_partitioner());

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      Interface system setup                    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_bddc_preconditioner()
{
  Kokkos::fence();
  Timer time;

  this->bddc_preconditioner =
    std::make_unique<Portable::BDDCPreconditioner<dim, double, BddcSmootherType>>(
      *interface_operator,
      *level_subdomain_matrices.back(),
      *subdomain_mg_preconditioner_bddc,
      level_subdomain_bddc_matrices,
      subdomain_mg_transfers_bddc,
      subdomain_mg_smoothers_bddc,
      subdomain_mg_preconditioner_corner_pinned.get());

  // Needed before set_fine_correction_mode(true) can be used -- see its
  // Assert. Cheap to always call (once per preconditioner setup, not per
  // iteration) even when the flag ends up false.
  this->bddc_preconditioner->compute_local_edge_face_schur_complement();

  this->bddc_preconditioner->set_fine_correction_mode(use_static_condensation_fine_correction);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      BDDC preconditioner setup                  (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  Kokkos::fence();
  time.restart();

  this->bddc_preconditioner->compute_coarse_matrix();

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      Coarse matrix for BDDC computed            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  {
    const std::array<double, 6> &setup_timings = this->bddc_preconditioner->get_setup_timings();

    const double total = setup_timings[0] + setup_timings[1] + setup_timings[2] + setup_timings[3] +
                         setup_timings[4] + setup_timings[5];

    bddc_setup_timing_table.add_value("cells", n_cells_total);
    bddc_setup_timing_table.add_value("dofs", level_distributed_dof_handlers.back().n_dofs());
    bddc_setup_timing_table.add_value("n_global_coarse_dofs",
                                      this->bddc_preconditioner->get_n_global_coarse_dofs());
    bddc_setup_timing_table.add_value("n_local_coarse_dofs",
                                      this->bddc_preconditioner->get_n_local_coarse_dofs());
    bddc_setup_timing_table.add_value("lift", setup_timings[0]);
    bddc_setup_timing_table.add_value("vmult_plain", setup_timings[1]);
    bddc_setup_timing_table.add_value("fine_correction", setup_timings[2]);
    bddc_setup_timing_table.add_value("inner_products", setup_timings[3]);
    bddc_setup_timing_table.add_value("mpi_sum", setup_timings[4]);
    bddc_setup_timing_table.add_value("lu_factorization", setup_timings[5]);
    bddc_setup_timing_table.add_value("total", total);

    // Diagnostic: mpi_sum times a *blocking* collective, so its wall time on
    // this rank includes however long it sits waiting for the slowest rank
    // to arrive. Gather each rank's own local coarse-dof count (split by
    // primal-constraint type: vertex/edge/face -- offsets[0..3] are the
    // cumulative start-of-vertices/edges/faces/end-of-faces indices) and
    // local (pre-reduction) compute time to check whether that wait is
    // actually load imbalance in how many primal constraints each
    // subdomain owns, rather than the reduction itself being expensive.
    const auto &local_coarse_offsets =
      level_subdomain_dof_handlers.back().get_dof_info().local_coarse_offsets;

    const unsigned int local_n_vertices = local_coarse_offsets[1] - local_coarse_offsets[0];
    const unsigned int local_n_edges    = local_coarse_offsets[2] - local_coarse_offsets[1];
    const unsigned int local_n_faces    = local_coarse_offsets[3] - local_coarse_offsets[2];

    const unsigned int local_n_coarse_dofs = this->bddc_preconditioner->get_n_local_coarse_dofs();
    const double       local_compute_time =
      setup_timings[0] + setup_timings[1] + setup_timings[2] + setup_timings[3];

    const auto all_n_vertices    = Utilities::MPI::gather(mpi_communicator, local_n_vertices, 0);
    const auto all_n_edges       = Utilities::MPI::gather(mpi_communicator, local_n_edges, 0);
    const auto all_n_faces       = Utilities::MPI::gather(mpi_communicator, local_n_faces, 0);
    const auto all_n_coarse_dofs = Utilities::MPI::gather(mpi_communicator, local_n_coarse_dofs, 0);
    const auto all_compute_time  = Utilities::MPI::gather(mpi_communicator, local_compute_time, 0);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        per_rank_load_table.clear();
        for (unsigned int rank = 0; rank < all_n_coarse_dofs.size(); ++rank)
          {
            per_rank_load_table.add_value("rank", rank);
            per_rank_load_table.add_value("vertices", all_n_vertices[rank]);
            per_rank_load_table.add_value("edges", all_n_edges[rank]);
            per_rank_load_table.add_value("faces", all_n_faces[rank]);
            per_rank_load_table.add_value("n_local_coarse_dofs", all_n_coarse_dofs[rank]);
            per_rank_load_table.add_value("local_compute_time", all_compute_time[rank]);
          }
        per_rank_load_table.set_scientific("local_compute_time", true);
        per_rank_load_table.set_precision("local_compute_time", 3);
      }
  }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  Timer time;
  Kokkos::fence();

  // Matrix-free, device-side replacement for the host FEValues cell loop +
  // Host-vector-then-import_elements-to-device pattern this used to be:
  // level_subdomain_matrices.back() is guaranteed to be a
  // SubdomainLaplaceOperator<dim, fe_degree, double> (the finest p-level
  // always uses degree == fe_degree, this class's own template parameter --
  // see setup_dofs()'s p_level_fes construction), so the downcast here is
  // safe. compute_rhs() already skips constrained dofs the same way
  // vmult_plain() does (plain_dof_indices_per_color marks them invalid),
  // which is exactly equivalent to the old code's explicit
  // subdomain_physical_boundary_dofs zeroing pass.
  static_cast<const Portable::SubdomainLaplaceOperator<dim, fe_degree, double> &>(
    *level_subdomain_matrices.back())
    .compute_rhs(subdomain_rhs_device, 1.0);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      RHS assembled                             (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  Kokkos::fence();
  time.restart();
  this->interface_operator->assemble_rhs_schur(rhs_schur_device, subdomain_rhs_device);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "                      Schur RHS assembled                       (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
unsigned int
LaplaceProblem<dim, fe_degree>::solve_interface()
{
  Timer time;
  Kokkos::fence();
  SolverControl solver_control(rhs_schur_device.size(), 1e-6 * rhs_schur_device.l2_norm());

  // solve_dd() runs the same PCG recursion as a plain solve(), but routes
  // the per-iteration matrix-vector product through
  // bddc_preconditioner->vmult_interface() instead of a bare
  // interface_operator->vmult() -- vmult_interface() is a timed wrapper
  // around exactly the same Schur-complement action, so the outer-matvec
  // Dirichlet-solve cost lands in bddc_preconditioner->get_timings() (see
  // timings[4] below) under the same instrumentation BNNPreconditioner
  // already uses, making the two preconditioners' timings comparable.
  //
  // (A SolverFlexibleCG A/B test confirmed the Polak-Ribiere vs.
  // Fletcher-Reeves beta formula makes no difference here -- BDDC's vmult()
  // behaves close enough to a fixed linear operator that FCG's correction
  // term is a no-op, so solve_dd()'s plain PCG recursion is not leaving
  // anything on the table.)
  Portable::SolverProjectedCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  bddc_preconditioner->reset_timings();

  solution_interface_device = 0.;
  cg.solve_dd(*interface_operator,
              solution_interface_device,
              rhs_schur_device,
              *bddc_preconditioner);

  solution_interface_device.update_ghost_values();

  Kokkos::fence();
  const double time_solve = time.wall_time();

  pcout << "                      Interface solver converged in " << solver_control.last_step()
        << " iterations.    (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's'
        << std::endl;

  const unsigned int max_mg_iterations_dirichlet =
    Utilities::MPI::max(interface_operator->get_maximum_subdomain_mg_iterations(),
                        mpi_communicator);


  const unsigned int max_mg_iterations_bddc =
    Utilities::MPI::max(bddc_preconditioner->get_maximum_subdomain_mg_iterations(),
                        mpi_communicator);

  pcout << "Subdomain Dirichlet MG iteration / BDDC MG iterations: " << max_mg_iterations_dirichlet
        << "   /    " << max_mg_iterations_bddc << std::endl;

  // Per-rank breakdown: the MPI-max above only tells us the worst subdomain,
  // not whether iteration counts grew uniformly or a few outlier subdomains
  // are dragging the max up. Append to per_rank_load_table (already holds
  // one row per rank from setup_bddc_preconditioner(), populated earlier
  // this cycle) rather than opening a new table, since it already lines up
  // rank <-> coarse-dof-count <-> setup-time.
  const auto all_mg_iterations_dirichlet =
    Utilities::MPI::gather(mpi_communicator,
                           interface_operator->get_maximum_subdomain_mg_iterations(),
                           0);
  const auto all_mg_iterations_bddc =
    Utilities::MPI::gather(mpi_communicator,
                           bddc_preconditioner->get_maximum_subdomain_mg_iterations(),
                           0);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      for (unsigned int rank = 0; rank < all_mg_iterations_dirichlet.size(); ++rank)
        {
          per_rank_load_table.add_value("dirichlet_mg_its", all_mg_iterations_dirichlet[rank]);
          per_rank_load_table.add_value("bddc_mg_its", all_mg_iterations_bddc[rank]);
        }
    }

  const auto iterations = std::max(solver_control.last_step(), 1u);

  // timings[0] = gather_and_weight_global_interface + weight_local_interface_and_scatter
  // timings[1] = vmult_coarse_correction
  // timings[2] = vmult_fine_correction
  // timings[3] = total vmult() wall time
  // timings[4] = vmult_interface (outer Dirichlet solve via S, driven by solve_dd())
  const std::array<double, 5> &timings = bddc_preconditioner->get_timings();

  timing_table.add_value("cells", n_cells_total);
  timing_table.add_value("dofs", level_distributed_dof_handlers.back().n_dofs());
  timing_table.add_value("Dirichlet", timings[4]);
  timing_table.add_value("gather_scatter", timings[0]);
  timing_table.add_value("coarse_correction", timings[1]);
  timing_table.add_value("fine_correction", timings[2]);
  timing_table.add_value("total_vmult", timings[3]);
  timing_table.add_value("CG_time", time_solve);
  timing_table.add_value("Iters", solver_control.last_step());

  timing_table_per_iteration.add_value("cells", n_cells_total);
  timing_table_per_iteration.add_value("dofs", level_distributed_dof_handlers.back().n_dofs());
  timing_table_per_iteration.add_value("Dir_per_iter", timings[4] / iterations);
  timing_table_per_iteration.add_value("gather_scatter_per_iter", timings[0] / iterations);
  timing_table_per_iteration.add_value("coarse_per_iter", timings[1] / iterations);
  timing_table_per_iteration.add_value("fine_per_iter", timings[2] / iterations);
  timing_table_per_iteration.add_value("vmult_per_iter", timings[3] / iterations);
  timing_table_per_iteration.add_value("CG_per_iter", time_solve / iterations);
  timing_table_per_iteration.add_value("dirichlet_mg_its", max_mg_iterations_dirichlet);
  timing_table_per_iteration.add_value("bddc_mg_its", max_mg_iterations_bddc);

  return solver_control.last_step();
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::matvec_ghost_timing()
{
  const bool communication_on = true;
  const bool computation_on   = true;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dummy_solution, dummy_rhs;
  level_subdomain_dof_handlers.back().initialize_interface_dof_vector(dummy_solution);
  dummy_rhs.reinit(dummy_solution);

  dummy_rhs = 1.;

  Timer time;


  std::array<double, 2> best_mv_both{{1e10, 1e10}};
  std::array<double, 2> best_only_ghost{{1e10, 1e10}};
  std::array<double, 2> best_only_comp{{1e10, 1e10}};

  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = 50;

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          communication_on,
                                          communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv_both[0] = std::min(best_mv_both[0], stat.max);
      }


      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->vmult_coarse_correction_dummy(dummy_solution,
                                                            dummy_rhs,
                                                            computation_on,
                                                            communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv_both[1] = std::min(best_mv_both[1], stat.max);
      }
      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          !computation_on,
                                          communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_ghost[0] = std::min(best_only_ghost[0], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->vmult_coarse_correction_dummy(dummy_solution,
                                                            dummy_rhs,
                                                            !computation_on,
                                                            communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_ghost[1] = std::min(best_only_ghost[1], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          computation_on,
                                          !communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_comp[0] = std::min(best_only_comp[0], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->vmult_coarse_correction_dummy(dummy_solution,
                                                            dummy_rhs,
                                                            computation_on,
                                                            !communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_comp[1] = std::min(best_only_comp[1], stat.max);
      }
    }


  ghost_timing_table.add_value("cells", n_cells_total);
  ghost_timing_table.add_value("dofs", level_distributed_dof_handlers.back().n_dofs());

  ghost_timing_table.add_value("subdomain_total", best_mv_both[0]);
  ghost_timing_table.add_value("subdomain_compute", best_only_comp[0]);
  ghost_timing_table.add_value("subdomain_communicate", best_only_ghost[0]);

  ghost_timing_table.add_value("coarse_total", best_mv_both[1]);
  ghost_timing_table.add_value("coarse_compute", best_only_comp[1]);
  ghost_timing_table.add_value("coarse_communicate", best_only_ghost[1]);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::fine_correction_component_timing(const unsigned int cycle)
{
  const unsigned int n_reps = 5;

  Timer time;

  // Times n_mv calls to op(), n_reps times, and returns the minimum
  // per-call wall time across reps (as an MPI max across ranks) -- same
  // "best of repeats" pattern matvec_ghost_timing() uses above.
  auto time_op = [&](const std::function<void()> &op, unsigned int n_mv)
    {
      double best = 1e10;
      for (unsigned int rep = 0; rep < n_reps; ++rep)
        {
          Kokkos::fence();
          time.restart();
          for (unsigned int i = 0; i < n_mv; ++i)
            op();
          Kokkos::fence();

          const Utilities::MPI::MinMaxAvg stat =
            Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
          best = std::min(best, stat.max);
        }
      return best;
    };

  // subdomain_mg_preconditioner_bddc's V-cycle applies exactly these
  // per-level operators/diagonals during vmult_fine_correction()'s CG
  // (smoothing at every level, the operator's own vmult() at every level
  // via the V-cycle's residual/restrict/prolongate chain) -- looping over
  // every level here, rather than just the finest one, is what actually
  // reconstructs where the V-cycle spends its time, not just the top-level
  // CG matvec cost.
  for (unsigned int level = level_subdomain_matrices.min_level();
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      const unsigned int n_mv = level_subdomain_matrices[level]->m() < 1e6 ? 200 : 50;

      VectorTypeMG dummy_src, dummy_dst;
      level_subdomain_matrices[level]->initialize_dof_vector(dummy_src);
      dummy_dst.reinit(dummy_src);
      dummy_src = 1.;

      // Built standalone purely to isolate its vmult()/fused_scale() cost
      // outside the V-cycle -- subdomain_mg_smoothers_bddc[level] holds an
      // equivalent instance internally (see setup_smoothers()).
      BddcPreconditionerType diagonal_preconditioner(
        *level_subdomain_bddc_matrices[level],
        level_subdomain_bddc_matrices[level]->get_matrix_diagonal_inverse());

      const double bddc_vmult =
        time_op([&]() { level_subdomain_bddc_matrices[level]->vmult(dummy_dst, dummy_src); }, n_mv);

      const double bddc_vmult_plain =
        time_op([&]() { level_subdomain_bddc_matrices[level]->vmult_plain(dummy_dst, dummy_src); },
                n_mv);

      const double bddc_project =
        time_op([&]() { level_subdomain_bddc_matrices[level]->project(dummy_dst); }, n_mv);

      const double diag_precond_vmult =
        time_op([&]() { diagonal_preconditioner.vmult(dummy_dst, dummy_src); }, n_mv);

      const double diag_fused_scale = time_op(
        [&]()
          {
            BddcPreconditionerType::fused_scale(
              dummy_dst,
              dummy_src,
              level_subdomain_bddc_matrices[level]->get_matrix_diagonal_inverse()->get_vector());
          },
        n_mv);

      // The Neumann Chebyshev smoother's preconditioner isn't a
      // ProjectedDiagonalPreconditioner at all -- setup_smoothers() hands
      // it dealii::DiagonalMatrix directly (get_matrix_diagonal_inverse_
      // neumann()), since BNN's Neumann problem has no primal constraints
      // to keep homogeneous. But note this number is NOT actually paid as
      // a separate kernel by the real smoother: dealii::PreconditionChebyshev
      // has a MemorySpace::Default + DiagonalMatrix specialization of its
      // internal vector_updates() (see precondition.h) that reads the
      // diagonal straight out of preconditioner.get_vector() *inside* the
      // same Kokkos kernel that computes the Chebyshev recurrence update --
      // DiagonalMatrix::vmult() itself is never called there. So this is a
      // standalone reference number (what an unfused scale would cost), not
      // a cost actually incurred in the Dirichlet/Neumann smoothers -- unlike
      // BDDC's hand-rolled ProjectedChebyshevSmoother, which genuinely does
      // call diagonal_preconditioner->vmult() as a separate step each
      // iteration (see diag_precond_vmult/diag_fused_scale above), so those
      // two numbers ARE real costs paid by the BDDC smoother.
      const double neumann_diag_vmult_unfused_ref = time_op(
        [&]()
          {
            level_subdomain_matrices[level]->get_matrix_diagonal_inverse_neumann()->vmult(
              dummy_dst, dummy_src);
          },
        n_mv);

      const double dirichlet_vmult =
        time_op([&]() { level_subdomain_matrices[level]->vmult(dummy_dst, dummy_src); }, n_mv);

      const double neumann_vmult =
        time_op([&]()
                  { level_subdomain_neumann_matrices[level]->vmult_neumann(dummy_dst, dummy_src); },
                n_mv);

      // apply()'s per-iteration bookkeeping around the operator/precondi-
      // tioner calls above: the residual sadd(), and the two recurrence-
      // update variants (fused_update() for k=0, fused_recurrence_update()
      // for k>=1). None of these have a Dirichlet/Neumann counterpart to
      // compare against -- dealii::PreconditionChebyshev's GPU-specialized
      // vector_updates() (see the neumann_diag_vmult_unfused_ref comment
      // above) computes the residual and the recurrence combine *inside*
      // the same fused kernel that also reads the diagonal, so it never
      // pays a separate sadd() or update kernel either. These three
      // columns are therefore exactly the extra per-iteration overhead our
      // hand-rolled ProjectedChebyshevSmoother pays that the dealii-backed
      // Dirichlet/Neumann smoothers avoid entirely.
      VectorTypeMG dummy_x_prev;
      dummy_x_prev.reinit(dummy_src);

      const double bddc_sadd =
        time_op([&]() { dummy_dst.sadd(-1.0, 1.0, dummy_src); }, n_mv);

      const double bddc_fused_update = time_op(
        [&]() { BddcSmootherType::fused_update(dummy_dst, dummy_x_prev, dummy_src, 1.0); },
        n_mv);

      const double bddc_recurrence_update = time_op(
        [&]() {
          BddcSmootherType::fused_recurrence_update(
            dummy_dst, dummy_x_prev, dummy_src, 1.0, 1.0);
        },
        n_mv);

      // MG transfer cost between this level and level-1 -- the one part of
      // the V-cycle none of the columns above touch at all. There's no
      // transfer object below the coarsest level, so these are left at 0
      // there (no level-1 to transfer to/from).
      double bddc_prolongate = 0.;
      double bddc_restrict   = 0.;

      if (level > level_subdomain_matrices.min_level())
        {
          VectorTypeMG dummy_coarse_src, dummy_coarse_dst;
          level_subdomain_matrices[level - 1]->initialize_dof_vector(dummy_coarse_src);
          dummy_coarse_dst.reinit(dummy_coarse_src);
          dummy_coarse_src = 1.;

          bddc_prolongate = time_op(
            [&]() {
              subdomain_mg_transfers_bddc[level]->prolongate_and_add(dummy_dst, dummy_coarse_src);
            },
            n_mv);

          bddc_restrict = time_op(
            [&]() {
              subdomain_mg_transfers_bddc[level]->restrict_and_add(dummy_coarse_dst, dummy_src);
            },
            n_mv);
        }

      fine_correction_component_timing_table.add_value("cycle", cycle);
      fine_correction_component_timing_table.add_value("level", level);
      fine_correction_component_timing_table.add_value("subdomain_dofs",
                                                       level_subdomain_matrices[level]->m());
      fine_correction_component_timing_table.add_value("bddc_vmult", bddc_vmult);
      fine_correction_component_timing_table.add_value("bddc_vmult_plain", bddc_vmult_plain);
      fine_correction_component_timing_table.add_value("bddc_project", bddc_project);
      fine_correction_component_timing_table.add_value("diag_precond_vmult", diag_precond_vmult);
      fine_correction_component_timing_table.add_value("diag_fused_scale", diag_fused_scale);
      fine_correction_component_timing_table.add_value("neumann_diag_vmult_unfused_ref",
                                                       neumann_diag_vmult_unfused_ref);
      fine_correction_component_timing_table.add_value("dirichlet_vmult", dirichlet_vmult);
      fine_correction_component_timing_table.add_value("neumann_vmult", neumann_vmult);
      fine_correction_component_timing_table.add_value("bddc_sadd", bddc_sadd);
      fine_correction_component_timing_table.add_value("bddc_fused_update", bddc_fused_update);
      fine_correction_component_timing_table.add_value("bddc_recurrence_update",
                                                       bddc_recurrence_update);
      fine_correction_component_timing_table.add_value("bddc_prolongate", bddc_prolongate);
      fine_correction_component_timing_table.add_value("bddc_restrict", bddc_restrict);
    }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::postprocess_subdomain_solution()
{
  const auto &subdomain_dof_handler_fine = level_subdomain_dof_handlers.back();

  Timer time;
  Kokkos::fence();
  this->interface_operator->reconstruct_subdomain_solution_from_interface(subdomain_solution_device,
                                                                          solution_interface_device,
                                                                          subdomain_rhs_device);

  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler_fine.get_dof_handler().n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_solution_host.update_ghost_values();

  level_subdomain_constraints_physical.back().distribute(subdomain_solution_host);

  const auto &subdomain_to_global_dof_map =
    subdomain_dof_handler_fine.get_dof_info().subdomain_to_global_dof_map;

  for (unsigned int i = 0; i < subdomain_to_global_dof_map.size(); ++i)
    {
      const auto global_index = subdomain_to_global_dof_map[i];

      global_solution_host[global_index] = subdomain_solution_host[i];
    }

  global_solution_host.compress(VectorOperation::add);


  for (unsigned int i = 0; i < subdomain_dof_handler_fine.n_locally_relevant_interface_indices();
       ++i)
    {
      const auto subdomain_index = subdomain_dof_handler_fine.local_interface_to_subdomain(i);
      const auto global_index    = subdomain_to_global_dof_map[subdomain_index];
      global_solution_host[global_index] *=
        global_interface_weights[subdomain_dof_handler_fine.local_to_global_interface_partitioner(
          i)];
    }

  global_solution_host.update_ghost_values();

  Kokkos::fence();
  time_details << "                      Subdomain solution post-processed         (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  Kokkos::fence();
  Timer time;
  (void)cycle;

  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(level_distributed_dof_handlers.back());
  // data_out.add_data_vector(global_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2);

  // DataOut<dim> data_out_subdomain;

  // data_out_subdomain.attach_dof_handler(level_subdomain_dof_handlers.back().get_dof_handler());
  // data_out_subdomain.add_data_vector(
  //   subdomain_solution_host,
  //   "solution_subdomain_" +
  //   std::to_string(level_subdomain_dof_handlers.back().get_subdomain_id()));
  // data_out_subdomain.build_patches();

  // data_out_subdomain.set_flags(flags);
  // data_out_subdomain.write_vtu_with_pvtu_record("./",
  //                                               "solution_subdomain_" +std::to_string(
  //                                                 level_subdomain_dof_handlers.back().get_subdomain_id()),
  //                                               cycle,
  //                                               mpi_communicator,
  //                                               1);

  Vector<float> cellwise_norm(triangulation.n_active_cells());
  VectorTools::integrate_difference(level_distributed_dof_handlers.back(),
                                    global_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);
  const double global_norm =
    VectorTools::compute_global_error(triangulation, cellwise_norm, VectorTools::L2_norm);


  Kokkos::fence();
  time_details << "                      Output results                            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  pcout << "                      solution norm: " << global_norm << std::endl;


  // Vector<float> cellwise_norm_subdomain(
  //   subdomain_triangulation.get_triangulation().n_active_cells());
  // VectorTools::integrate_difference(subdomain_dof_handler.get_dof_handler(),
  //                                   subdomain_solution_host,
  //                                   Functions::ZeroFunction<dim>(),
  //                                   cellwise_norm_subdomain,
  //                                   QGauss<dim>(fe.degree + 2),
  //                                   VectorTools::L2_norm);
  // const double subdomain_norm = VectorTools::compute_global_error(
  //   subdomain_triangulation.get_triangulation(),
  //   cellwise_norm_subdomain,
  //   VectorTools::L2_norm);


  // std::cout << " solution norm on subdomain "
  //           << subdomain_dof_handler.get_subdomain_id() << ": "
  //           << subdomain_norm << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test_bddc()
{
  // this->bddc_preconditioner =
  //   std::make_unique<Portable::BDDCPreconditioner<dim, double>>(*interface_operator,
  //                                                               *level_subdomain_matrices.back(),
  //                                                              Portable::BDDCVariant::corner);

  // this->bddc_preconditioner =
  //   std::make_unique<Portable::BDDCPreconditioner<dim, double>>(*interface_operator,
  //                                                               *level_subdomain_matrices.back());

  // using InterfaceVectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  // InterfaceVectorType dst, src;

  // dst.reinit(level_subdomain_dof_handlers.back().get_interface_vector_partitioner());
  // src.reinit(dst);

  // // src = 1.0;

  // Portable::DeviceVector<double> src_view(src.get_values(), src.locally_owned_size());

  // Kokkos::parallel_for(src.locally_owned_size(), KOKKOS_LAMBDA(const int &i) { src_view(i) = i;
  // });

  // src.compress(VectorOperation::insert);

  // // bddc_preconditioner->solve_subdomain_with_constraints(dst, src);

  // bddc_preconditioner->compute_coarse_matrix();

  // Timer time;
  // Kokkos::fence();
  // // SolverControl solver_control(1000, 1e-9 * rhs_schur_device.l2_norm());
  // ReductionControl solver_control(1000, 1e-12, 1e-7);

  // // SolverControl solver_control(10000, 1e-12 * rhs_schur_device.l2_norm());

  // // Portable::SolverProjectedCG<LinearAlgebra::distributed::Vector<double,
  // MemorySpace::Default>>
  // // cg(solver_control);

  // SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(solver_control);

  // solution_interface_device = 0.;

  // cg.solve(*interface_operator, solution_interface_device, rhs_schur_device,
  // *bddc_preconditioner);

  // // cg.solve(*interface_operator, solution_interface_device, rhs_schur_device,
  // // PreconditionIdentity());

  // pcout << "                      Interface solver converged in " << solver_control.last_step()
  //       << " iterations.    (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's'
  //       << std::endl;

  // // SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  // // cg(
  // //   solver_control);

  // solution_interface_device.update_ghost_values();

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dst, src;
  this->level_subdomain_bddc_matrices.back()->initialize_dof_vector(src);
  dst.reinit(src);

  src = 1.;

  // double norm_before = src.l2_norm();

  level_subdomain_bddc_matrices.back()->project(src);

  // double norm_after = src.l2_norm();

  // level_subdomain_bddc_matrices.back()->project(src);

  // double norm_after2 = src.l2_norm();

  // std::cout << "On subdomain " << Utilities::MPI::this_mpi_process(mpi_communicator)
  //           << " norm before = " << norm_before << "  , norm after = " << norm_after
  //           << ", norm after2 = " << norm_after2 << std::endl;



  SolverControl solver_control(src.size(), 1e-12 * src.l2_norm());
  Portable::SolverProjectedCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
    solver_cg(solver_control);
  solver_cg.solve_projected(*level_subdomain_bddc_matrices.back(),
                            dst,
                            src,
                            *subdomain_mg_preconditioner_bddc,
                            *level_subdomain_bddc_matrices.back());
  // solver_cg.solve(*level_subdomain_bddc_matrices.back(), dst, src, PreconditionIdentity());


  // SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
  // SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> solver_cg(
  //   solver_control);
  // solver_cg.solve(*level_subdomain_matrices.back(), dst, src,
  // *subdomain_mg_preconditioner_dirichlet);

  // SolverControl solver_control(src.size(), 1e-12 * src.l2_norm());
  // Portable::SolverProjectedCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  //   solver_cg(solver_control);
  // solver_cg.solve_projected(*level_subdomain_matrices.back(),
  //                           dst,
  //                           src,
  //                           *subdomain_mg_preconditioner_dirichlet,
  //                           *level_subdomain_matrices.back());

  std::cout << "On subdomain " << Utilities::MPI::this_mpi_process(mpi_communicator)
            << " solver converged in " << solver_control.last_step() << "  iterations."
            << std::endl;
}

template <int dim, int fe_degree>
bool
LaplaceProblem<dim, fe_degree>::run()
{
  // Fixed, small problem -- this is a correctness check, not a performance
  // sweep, so one h-refinement level is enough (still multi-level p+h MG,
  // still real edge/face primal constraints for any rank count >= 2 per
  // axis). All of this setup is independent of the fine-correction mode,
  // so it's built once and reused for both solves below.
  const unsigned int n_refinement_cycles = 2;

  create_subdomain_triangulations(n_refinement_cycles);
  setup_dofs();
  compute_interface_weights();
  setup_matrix_free();
  setup_mg_transfers();
  setup_smoothers();
  setup_mg_preconditioners();
  setup_interface_system();

  // Independent of the fine-correction mode (only reads level_subdomain_
  // matrices/interface_operator, never bddc_preconditioner) -- built once,
  // reused for both solves.
  assemble_rhs();

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> solution_pi, solution_sc;
  unsigned int                                                    iterations_pi = 0;
  unsigned int                                                    iterations_sc = 0;

  for (const bool use_static_condensation : {false, true})
    {
      use_static_condensation_fine_correction = use_static_condensation;

      // Rebuilds bddc_preconditioner from scratch each time (a fresh
      // make_unique, see setup_bddc_preconditioner()), including
      // compute_local_edge_face_schur_complement() and
      // set_fine_correction_mode(use_static_condensation_fine_correction).
      setup_bddc_preconditioner();

      const unsigned int iterations = solve_interface();

      pcout << "  fine-correction mode = "
            << (use_static_condensation ? "static condensation" : "Pi-projected (Ahat)")
            << ": interface CG converged in " << iterations << " iterations" << std::endl;

      if (use_static_condensation)
        {
          solution_sc   = solution_interface_device;
          iterations_sc = iterations;
        }
      else
        {
          solution_pi   = solution_interface_device;
          iterations_pi = iterations;
        }
    }

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = solution_pi;
  diff -= solution_sc;

  const double solution_norm   = solution_pi.l2_norm();
  const double relative_diff   = diff.l2_norm() / std::max(solution_norm, 1e-300);

  pcout << std::endl
        << "  ||x_pi||_2 = " << solution_norm << ", ||x_pi - x_sc||_2 / ||x_pi||_2 = "
        << relative_diff << std::endl
        << "  iterations: Pi-projected = " << iterations_pi
        << ", static condensation = " << iterations_sc << std::endl;

  // Pass/fail: both fine-correction modes precondition the SAME linear
  // system (interface_operator * x = rhs_schur_device) -- the
  // preconditioner only changes the convergence path, not the fixed point
  // both CG solves converge to. So the two solutions must agree, to
  // roughly the outer solver's own relative tolerance (1e-6 in
  // solve_interface()) with some slack for two independently-accumulated
  // rounding errors. Iteration counts are reported but not gated on --
  // the two algorithms are genuinely different preconditioners, not
  // required to need the same iteration count, just to converge to the
  // same answer.
  return relative_diff < 1e-4;
}

int
main(int argc, char *argv[])
{
  bool passed = false;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      const unsigned int n_pre_smooth  = 5;
      const unsigned int n_post_smooth = 5;

      constexpr int dim       = 3;
      constexpr int fe_degree = 4;

      LaplaceProblem<dim, fe_degree> laplace_problem(n_pre_smooth, n_post_smooth);
      passed = laplace_problem.run();

      // The comparison in run() is a global (MPI-reduced) vector norm, so
      // every rank already computed the same pass/fail verdict -- no
      // further MPI reduction needed here, unlike check_correctness_bddc_
      // arr_mg's per-rank-local-quantity pattern.
      const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      if (rank == 0)
        std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;
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

  return passed ? 0 : 1;
}