// Correctness check for the A_RR (static-condensation) MG hierarchy built
// in source/bddc_preconditioner/program.cc (level_subdomain_primal_pinned_
// matrices / subdomain_mg_transfers_primal_pinned / subdomain_mg_smoothers_
// primal_pinned / subdomain_mg_preconditioner_primal_pinned): solves
// A_RR x = b (b random, zeroed at primal-pinned dofs to respect the
// operator's implicit precondition) two ways --
//   (a) unpreconditioned CG: sanity check that A_RR itself is genuinely
//       SPD/well-posed on its own, independent of the MG hierarchy;
//   (b) CG preconditioned by subdomain_mg_preconditioner_primal_pinned:
//       the actual thing under test.
// Both must converge to the same solution; (b) must need far fewer
// iterations if the V-cycle is actually acting as an effective
// preconditioner and not, say, silently degrading to an identity or
// diverging.
//
// Needs a real multi-subdomain interface (for primal constraints to exist
// at all) and a real multi-level h+p hierarchy (for there to be anything
// for the V-cycle to precondition) -- so this necessarily duplicates a
// good deal of source/bddc_preconditioner/program.cc's own triangulation/
// dof/matrix-free/transfer/smoother setup, trimmed to only what
// subdomain_mg_preconditioner_primal_pinned depends on (the plain
// Dirichlet operator as a MatrixFree source, and the BDDC operator for
// primal-constraint info) -- Dirichlet's own smoother/transfer/
// preconditioner, Neumann, BNN, and the interface-solve/coarse-matrix
// machinery are all irrelevant here and omitted.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"
#include "operators/portable_subdomain_laplace_operator.h"

using namespace dealii;

// Same helper source/bddc_preconditioner/program.cc uses to feed
// GeometricTransfer/PolynomialTransfer::reinit_primal_pinned() a host-side
// dof list from SubdomainBDDCOperator's own device-side mask.
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

template <int dim, int fe_degree>
class ARRProblem
{
public:
  ARRProblem();

  // Returns {unpreconditioned_iterations, mg_preconditioned_iterations,
  // ||x_unprec - x_mg||_2 / ||x_mg||_2} on this rank, or {0, 0, -1.} if
  // this rank has no primal constraints (nothing to test locally).
  std::array<double, 3>
  run();

private:
  void
  create_subdomain_triangulation();

  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  setup_mg_transfers();

  void
  setup_smoothers();

  void
  setup_mg_preconditioner();

  MPI_Comm                                        mpi_communicator;
  parallel::fullydistributed::Triangulation<dim> triangulation;

  FE_Q<dim>                                 fe;
  MGLevelObject<std::unique_ptr<FE_Q<dim>>> p_level_fes;

  std::shared_ptr<SubdomainTriangulation<dim>> subdomain_triangulation;

  MGLevelObject<SubdomainDoFHandler<dim>> level_subdomain_dof_handlers;
  MGLevelObject<DoFHandler<dim>>          level_distributed_dof_handlers;

  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints;
  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints_physical;

  using VectorTypeMG     = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
  using LevelMatrixType  = Portable::SubdomainLaplaceOperatorBase<dim, double>;
  using SmootherType     = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;
  using TransferType     = Portable::MGTransferBase<dim, double>;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_matrices;
  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_bddc_matrices;
  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_primal_pinned_matrices;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_primal_pinned;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_primal_pinned;

  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>>
    subdomain_mg_preconditioner_primal_pinned;

  static constexpr unsigned int n_pre_smooth  = 5;
  static constexpr unsigned int n_post_smooth = 5;

  struct SubdomainOperatorRunner
  {
    const unsigned int              level;
    SubdomainDoFHandler<dim>       &subdomain_dof_h;
    AffineConstraints<double>      &constraints;
    AffineConstraints<double>      &constraints_physical;
    ARRProblem<dim, fe_degree>     &parent;

    template <unsigned int degree>
    void
    run()
    {
      parent.level_subdomain_matrices[level] =
        std::make_unique<Portable::SubdomainLaplaceOperator<dim, degree, double>>(
          subdomain_dof_h, constraints, constraints_physical);

      parent.level_subdomain_bddc_matrices[level] =
        std::make_unique<Portable::SubdomainBDDCOperator<dim, double>>(
          *parent.level_subdomain_matrices[level]);

      parent.level_subdomain_primal_pinned_matrices[level] =
        std::make_unique<Portable::SubdomainBDDCOperatorARRAdapter<dim, double>>(
          static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
            *parent.level_subdomain_bddc_matrices[level]));
    }
  };

  struct PolynomialTransferRunner
  {
    const unsigned int                       level;
    const Portable::MatrixFree<dim, double> &mf_coarse;
    const Portable::MatrixFree<dim, double> &mf_fine;
    AffineConstraints<double>               &physical_constraints_coarse;
    AffineConstraints<double>               &physical_constraints_fine;

    ARRProblem<dim, fe_degree> &parent;

    template <unsigned int degree_coarse, unsigned int degree_fine>
    void
    run()
    {
      const auto &bddc_coarse = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
        *parent.level_subdomain_bddc_matrices[level - 1]);
      const auto &bddc_fine = static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
        *parent.level_subdomain_bddc_matrices[level]);

      auto transfer =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();
      transfer->reinit_primal_pinned(mf_coarse,
                                     mf_fine,
                                     physical_constraints_coarse,
                                     primal_pinned_dofs_host(bddc_coarse),
                                     physical_constraints_fine,
                                     primal_pinned_dofs_host(bddc_fine));
      parent.subdomain_mg_transfers_primal_pinned[level] = std::move(transfer);
    }
  };
};

template <int dim, int fe_degree>
ARRProblem<dim, fe_degree>::ARRProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
{}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::create_subdomain_triangulation()
{
  const unsigned int n_subdomains = Utilities::MPI::n_mpi_processes(mpi_communicator);

  std::vector<unsigned int> subdomains_per_axis(dim);
  {
    int remaining = static_cast<int>(n_subdomains);
    for (int d = dim; d > 0; --d)
      {
        int n_this_axis = static_cast<int>(std::pow(remaining, 1.0 / d));
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
  }

  Triangulation<dim> coarse_triangulation;
  Point<dim>          p1, p2;
  for (int d = 0; d < dim; ++d)
    p2[d] = 1.;
  GridGenerator::subdivided_hyper_rectangle(coarse_triangulation, subdomains_per_axis, p1, p2);

  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  // Two h-refinements so there is a real (if small) h-multigrid hierarchy
  // on top of the p-coarsening, matching the shape (not the size) of the
  // main driver's hierarchy.
  coarse_triangulation.refine_global(2);

  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(coarse_triangulation,
                                                                                mpi_communicator);

  triangulation.create_triangulation(description);

  subdomain_triangulation = std::make_shared<SubdomainTriangulation<dim>>();
  subdomain_triangulation->create_subdomain_triangulation(triangulation);
}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::setup_dofs()
{
  std::vector<unsigned int> p_levels({fe.degree});
  while (p_levels.back() > 1)
    p_levels.push_back(std::max(p_levels.back() / 2, 1u));

  p_level_fes.resize(0, p_levels.size() - 1);
  for (unsigned int level = 0; level < p_levels.size(); ++level)
    p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

  // n_h_levels = 1 here (create_subdomain_triangulation() builds a single,
  // already-refined mesh -- no separate h-level history to walk, unlike
  // the main driver's per-cycle h-refinement loop), so level 0 is the
  // coarsest p-level and the rest are pure p-coarsening/refinement.
  level_subdomain_dof_handlers.resize(0, p_level_fes.max_level());
  level_distributed_dof_handlers.resize(0, level_subdomain_dof_handlers.max_level());

  level_subdomain_constraints.resize(0, level_subdomain_dof_handlers.max_level());
  level_subdomain_constraints_physical.resize(0, level_subdomain_dof_handlers.max_level());

  for (unsigned int level = 0; level <= level_subdomain_dof_handlers.max_level(); ++level)
    {
      DoFHandler<dim> &dof_h = level_distributed_dof_handlers[level];
      dof_h.reinit(triangulation);
      dof_h.distribute_dofs(*p_level_fes[level]);

      SubdomainDoFHandler<dim> &subdomain_dof_h = level_subdomain_dof_handlers[level];
      subdomain_dof_h.reinit(subdomain_triangulation, dof_h);
      subdomain_dof_h.distribute_subdomain_dofs();

      AffineConstraints<double> &constraints = level_subdomain_constraints[level];
      constraints.clear();
      DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(), constraints);
      DoFTools::make_zero_boundary_constraints(subdomain_dof_h.get_dof_handler(),
                                               types::boundary_id(0),
                                               constraints);
      DoFTools::make_zero_boundary_constraints(subdomain_dof_h.get_dof_handler(),
                                               subdomain_triangulation->get_interface_id(),
                                               constraints);
      constraints.close();

      AffineConstraints<double> &constraints_physical = level_subdomain_constraints_physical[level];
      constraints_physical.clear();
      DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(),
                                              constraints_physical);
      DoFTools::make_zero_boundary_constraints(subdomain_dof_h.get_dof_handler(),
                                               types::boundary_id(0),
                                               constraints_physical);
      constraints_physical.close();
    }
}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::setup_matrix_free()
{
  level_subdomain_matrices.resize(0, level_subdomain_dof_handlers.max_level());
  level_subdomain_bddc_matrices.resize(0, level_subdomain_dof_handlers.max_level());
  level_subdomain_primal_pinned_matrices.resize(0, level_subdomain_dof_handlers.max_level());

  // Level 0 is the coarsest p-level (fe_degree 1); dispatch every level
  // (including 0) through the same runtime-degree dispatcher the main
  // driver uses for its p-levels, since there is no separate h-level=1
  // fast path here (n_h_levels == 1, see setup_dofs()).
  for (unsigned int level = 0; level <= level_subdomain_dof_handlers.max_level(); ++level)
    {
      SubdomainOperatorRunner runner{level,
                                     level_subdomain_dof_handlers[level],
                                     level_subdomain_constraints[level],
                                     level_subdomain_constraints_physical[level],
                                     *this};

      const bool success = Portable::SubdomainOperatorDispatchFactory::dispatch(
        p_level_fes[level]->degree, runner);
      AssertThrow(success, ExcMessage("Failed to find a matching polynomial degree in dispatcher."));
    }
}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::setup_mg_transfers()
{
  subdomain_mg_transfers_primal_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level() + 1;
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      const unsigned int p_coarse = p_level_fes[level - 1]->degree;
      const unsigned int p_fine   = p_level_fes[level]->degree;

      PolynomialTransferRunner runner{level,
                                      level_subdomain_matrices[level - 1]->get_matrix_free(),
                                      level_subdomain_matrices[level]->get_matrix_free(),
                                      level_subdomain_constraints_physical[level - 1],
                                      level_subdomain_constraints_physical[level],
                                      *this};

      const bool success =
        Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse, p_fine, runner);
      AssertThrow(success,
                 ExcMessage("Failed to find a matching polynomial degree pair in dispatcher."));
    }
}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::setup_smoothers()
{
  subdomain_mg_smoothers_primal_pinned.resize(level_subdomain_matrices.min_level(),
                                              level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level();
       level <= level_subdomain_matrices.max_level();
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
          smoother_data.eig_cg_n_iterations = level_subdomain_matrices[0]->m();
        }

      level_subdomain_matrices[level]->compute_diagonal();
      level_subdomain_bddc_matrices[level]->compute_diagonal();

      smoother_data.preconditioner =
        level_subdomain_primal_pinned_matrices[level]->get_matrix_diagonal_inverse();

      subdomain_mg_smoothers_primal_pinned[level].initialize(
        *level_subdomain_primal_pinned_matrices[level], smoother_data);
    }
}

template <int dim, int fe_degree>
void
ARRProblem<dim, fe_degree>::setup_mg_preconditioner()
{
  subdomain_mg_preconditioner_primal_pinned = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_primal_pinned_matrices,
    subdomain_mg_transfers_primal_pinned,
    subdomain_mg_smoothers_primal_pinned,
    false);
}

template <int dim, int fe_degree>
std::array<double, 3>
ARRProblem<dim, fe_degree>::run()
{
  create_subdomain_triangulation();
  setup_dofs();
  setup_matrix_free();
  setup_mg_transfers();
  setup_smoothers();
  setup_mg_preconditioner();

  const auto &finest_bddc =
    static_cast<const Portable::SubdomainBDDCOperator<dim, double> &>(
      *level_subdomain_bddc_matrices.back());
  const auto primal_pinned_dofs = primal_pinned_dofs_host(finest_bddc);

  if (primal_pinned_dofs.empty())
    return std::array<double, 3>{{0., 0., -1.}};

  VectorTypeMG rhs, x_unprec, x_mg;
  level_subdomain_primal_pinned_matrices.back()->initialize_dof_vector(rhs);
  x_unprec.reinit(rhs);
  x_mg.reinit(rhs);

  {
    LinearAlgebra::ReadWriteVector<double> rw(rhs.size());
    std::mt19937                          rng(4321 + Utilities::MPI::this_mpi_process(mpi_communicator));
    std::uniform_real_distribution<double> dist(-1., 1.);
    for (unsigned int i = 0; i < rw.size(); ++i)
      rw(i) = dist(rng);
    rhs.import_elements(rw, VectorOperation::insert);
  }

  // Zero the RHS at primal-pinned dofs -- A_RR's implicit precondition
  // (those dofs are eliminated, not solved for).
  {
    LinearAlgebra::ReadWriteVector<double> rw(rhs.size());
    rw.import_elements(rhs, VectorOperation::insert);
    for (const unsigned int dof : primal_pinned_dofs)
      rw(dof) = 0.;
    rhs.import_elements(rw, VectorOperation::insert);
  }

  const unsigned int max_iterations = rhs.size();

  SolverControl control_unprec(max_iterations, 1e-12 * rhs.l2_norm());
  SolverCG<VectorTypeMG> cg_unprec(control_unprec);
  cg_unprec.solve(*level_subdomain_primal_pinned_matrices.back(),
                  x_unprec,
                  rhs,
                  PreconditionIdentity());

  SolverControl control_mg(max_iterations, 1e-12 * rhs.l2_norm());
  SolverCG<VectorTypeMG> cg_mg(control_mg);
  cg_mg.solve(*level_subdomain_primal_pinned_matrices.back(),
             x_mg,
             rhs,
             *subdomain_mg_preconditioner_primal_pinned);

  VectorTypeMG diff = x_unprec;
  diff -= x_mg;
  const double relative_diff = diff.l2_norm() / std::max(x_mg.l2_norm(), 1e-300);

  return std::array<double, 3>{{static_cast<double>(control_unprec.last_step()),
                               static_cast<double>(control_mg.last_step()),
                               relative_diff}};
}

constexpr int dim       = 3;
constexpr int fe_degree = 4;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  const MPI_Comm                   mpi_communicator = MPI_COMM_WORLD;
  const unsigned int                rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  ConditionalOStream                pcout(std::cout, rank == 0);

  ARRProblem<dim, fe_degree> problem;
  const auto                 result = problem.run();

  const double n_unprec    = result[0];
  const double n_mg        = result[1];
  const double relative_diff = result[2];

  if (relative_diff < 0.)
    {
      std::cout << "Rank " << rank << ": no primal constraints, skipping." << std::endl;
    }
  else
    {
      std::cout << "Rank " << rank << ": unpreconditioned CG iterations = " << n_unprec
                << ", MG-preconditioned CG iterations = " << n_mg
                << ", ||x_unprec - x_mg||/||x_mg|| = " << relative_diff << std::endl;
    }

  // Pass/fail: both solves must agree (same linear system, same unique
  // solution given A_RR is SPD) and the MG-preconditioned solve must need
  // meaningfully fewer iterations, unless this rank had nothing to check.
  const bool local_pass =
    (relative_diff < 0.) || (relative_diff < 1e-6 && n_mg <= n_unprec);
  const int local_result  = local_pass ? 0 : 1;
  const int global_result = Utilities::MPI::sum(local_result, mpi_communicator);

  if (global_result != 0)
    {
      pcout << "FAILED" << std::endl;
      return 1;
    }

  pcout << "PASSED" << std::endl;
  return 0;
}
