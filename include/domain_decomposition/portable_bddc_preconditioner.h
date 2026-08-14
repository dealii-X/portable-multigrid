#ifndef portable_bddc_preconditioner_h
#define portable_bddc_preconditioner_h

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <cmath>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_block_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "multigrid/portable_block_vcycle_adapters.h"
#include "multigrid/portable_projected_chebyshev_smoother.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * BddcSmootherType is the type of subdomain_mg_smoothers_bddc's element
   * in program.cc -- ProjectedChebyshevSmoother<LevelMatrixType,
   * BddcPreconditionerType, VectorTypeMG> in production -- needed here
   * (rather than only through the type-erased VCycleMultigridBase
   * subdomain_mg_preconditioner already stored below) so that
   * compute_local_coarse_matrix() can read each level's AdditionalData
   * (degree/max_eigenvalue/smoothing_range) back out via
   * get_additional_data() and build a *second*, block-shaped V-cycle for
   * the coarse problem -- VCycleMultigridBase's vmult()-only interface
   * has no way to expose the individual levels a block V-cycle needs.
   */
  template <int dim, typename Number, typename BddcSmootherType>
  class BDDCPreconditioner
  {
  public:
    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainPreconditioner = VCycleMultigridBase<dim, Number>;


    BDDCPreconditioner(
      const SchurInterfaceOperator<dim, Number>       &interface_operator,
      const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
      const SubdomainPreconditioner                   &subdomain_mg_preconditione,
      const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
                                                                        &level_bddc_matrices,
      const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers,
      const MGLevelObject<BddcSmootherType>                             &level_bddc_smoothers,
      // Corner-only-pinned MG preconditioner (level_subdomain_corner_pinned_
      // matrices' V-cycle in program.cc) -- optional: only needed when
      // set_fine_correction_mode(true) is used to switch vmult()'s fine
      // correction over to the classical corner-pin + edge/face-Lagrange-
      // multiplier static condensation (see below); nullptr is fine for
      // callers that never do that. Also requires
      // compute_local_edge_face_schur_complement() to be called once during
      // setup before set_fine_correction_mode(true) is used.
      const SubdomainPreconditioner *subdomain_mg_preconditioner_corner_pinned = nullptr,
      // Corner-pinned hierarchy's own level transfers (subdomain_mg_
      // transfers_corner_pinned in program.cc) -- needed to build a
      // block-shaped corner-pinned V-cycle for compute_local_edge_face_
      // schur_complement()'s batched w_l solves, the same way level_bddc_
      // transfers already lets compute_local_coarse_matrix() build one for
      // the Pi-projected path. The block V-cycle's MATRICES and diagonal
      // come from level_bddc_matrices above instead (level_subdomain_
      // corner_pinned_matrices in program.cc holds SubdomainBDDCOperator
      // CornerPinnedAdapter-wrapped views of the SAME underlying
      // SubdomainBDDCOperator objects level_bddc_matrices already holds
      // directly -- casting through the adapter would be undefined
      // behavior, since it doesn't inherit from SubdomainBDDCOperator;
      // get_matrix_diagonal_inverse_corner_pinned() is reachable straight
      // off the concrete SubdomainBDDCOperator, no adapter needed). Its
      // SMOOTHER tuning (degree/smoothing_range) is likewise borrowed from
      // level_bddc_smoothers rather than needing its own array -- see
      // compute_local_edge_face_schur_complement()'s comment. Optional in
      // lockstep with subdomain_mg_preconditioner_corner_pinned -- nullptr
      // is fine for callers that never enable static condensation, but
      // must be non-null together with it whenever compute_local_edge_
      // face_schur_complement() actually runs the block solve (see its
      // Assert).
      const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>>
                        *level_corner_pinned_transfers = nullptr,
      const BDDCVariant variant                        = BDDCVariant::corner_edge_face);

    // Selects which algorithm vmult()'s fine-correction step uses:
    // false (default) = CG against Ahat = Pi*A*Pi (the original, project()-
    //   based path -- project() is a tiny serial per-primal-constraint-group
    //   kernel, poor GPU occupancy, called on every fine-correction CG
    //   iteration).
    // true = classical BDDC static condensation (Klawonn/Widlund/Dohrmann-
    //   style saddle-point fine-grid solve): corner primal dofs are hard-
    //   pinned (Dirichlet-style, baked into the corner-pinned operator's
    //   per-cell dof mask -- no separate kernel launch), which alone is
    //   enough to make the remaining "R" system SPD; edge/face primal
    //   constraints are then enforced weakly via a small, dense, local
    //   Lagrange-multiplier system (see compute_local_edge_face_schur_
    //   complement()/vmult_fine_correction_static_condensation()). This
    //   replaces project()'s per-iteration serial-kernel cost with (a) an
    //   MG-preconditioned CG solve against the corner-pinned operator and
    //   (b) a tiny precomputed-Cholesky-factor triangular solve, both
    //   cheaper on GPU.
    // Coarse correction is unaffected either way -- it's a once-per-setup,
    // MPI-global solve, not the per-iteration GPU bottleneck this flag
    // exists to let around. See [[project-static-condensation-arr]] for the
    // motivation/derivation. Requires a non-null
    // subdomain_mg_preconditioner_corner_pinned to have been passed to the
    // constructor, and compute_local_edge_face_schur_complement() to have
    // been called, when enabling.
    void
    set_fine_correction_mode(bool use_static_condensation);

    void
    vmult(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

    // Timed wrapper around interface_operator->vmult() -- the same Schur-
    // complement action (a subdomain Dirichlet solve under the hood) that
    // SchurInterfaceOperator::vmult() performs as the outer CG's A.vmult().
    // Exists so SolverProjectedCG::solve_dd() can route the per-iteration
    // matvec through the preconditioner and have its Dirichlet-solve cost
    // captured under the same instrumentation BNNPreconditioner's
    // vmult_interface() already provides, making the two comparable.
    void
    vmult_interface(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

    void
    project_to_homogeneous_constraints_interface(InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_subdomain(SubdomainVectorType &subdomain_vector) const;


    void
    lift_coarse_to_subdomain(SubdomainVectorType  &interface_vector,
                             const Vector<Number> &coarse_vector) const;

    void
    compute_coarse_matrix();

    void
    compute_local_coarse_matrix(LAPACKFullMatrix<Number> &local_coarse_matrix);

    // One-time setup for vmult_fine_correction_static_condensation() (call
    // once, alongside compute_coarse_matrix(), before enabling
    // set_fine_correction_mode(true)): for each active edge/face primal
    // constraint group l (corners are excluded -- they're hard-pinned into
    // the corner-pinned operator itself, not part of this Lagrange system),
    // solves A_RR_corner_pinned w_l = c_l via CG preconditioned by
    // subdomain_mg_preconditioner_corner_pinned, where c_l is group l's
    // constraint functional lifted into subdomain-vector space (coarse_
    // weights(l) at the group's dofs, zero elsewhere). Stores the w_l in
    // edge_face_basis_block (packed blocks, one per l) and forms/factorizes
    // the small, dense,
    // LOCAL (no MPI -- unlike the global coarse matrix) SPD Schur matrix
    // edge_face_schur_matrix(k,l) = c_k . w_l = C_R A_RR_corner_pinned^{-1}
    // C_R^t, safe to factorize locally because corner-only pinning already
    // makes A_RR_corner_pinned SPD regardless of bddc_variant/floating-
    // subdomain status (see [[project-static-condensation-arr]]).
    void
    compute_local_edge_face_schur_complement();

    void
    vmult_fine_correction(SubdomainVectorType       &fine_solution,
                          const SubdomainVectorType &fine_residual) const;

    // Static-condensation replacement for vmult_fine_correction(): computes
    // fine_solution = A^{-1} fine_residual restricted to the non-corner
    // ("R") dofs via the classical corner-pin + edge/face-Lagrange-multiplier
    // construction (see set_fine_correction_mode()'s class-level comment):
    //   t_R = A_RR_corner_pinned^{-1} fine_residual                 (CG, MG-preconditioned)
    //   rhs_lambda = C_R t_R                                        (small dense,
    //   apply_edge_face_constraints()) lambda = edge_face_schur_matrix^{-1} rhs_lambda (small
    //   dense, precomputed Cholesky factor) fine_solution = t_R - sum_l lambda(l) *
    //   w_l (block l of edge_face_basis_block), fused into one kernel
    // Corner dofs end up exactly zero in fine_solution automatically (no
    // special-casing needed): both t_R and every basis function are zero
    // there by construction of the corner-pinned operator's identity block.
    // Precondition on fine_residual: zero at CORNER-pinned dof positions
    // only (NOT edge/face -- those are genuine input here, unlike the Ahat
    // path's project()). Requires compute_local_edge_face_schur_complement()
    // to have been called, and a non-null subdomain_mg_preconditioner_
    // corner_pinned to have been passed to the constructor.
    void
    vmult_fine_correction_static_condensation(SubdomainVectorType       &fine_solution,
                                              const SubdomainVectorType &fine_residual) const;

    void
    solve_fine_correction(SubdomainVectorType       &fine_solution,
                          const SubdomainVectorType &fine_residual) const;


    void
    vmult_coarse_correction(SubdomainVectorType       &coarse_solution,
                            const SubdomainVectorType &fine_residual) const;

    void
    coarse_to_global_interface(InterfaceVectorType  &interface_vector,
                               const Vector<Number> &coarse_vector) const;

    void
    global_interface_to_coarse(Vector<Number>            &coarse_vector,
                               const InterfaceVectorType &interface_vector) const;

    void
    gather_and_weight_global_interface(SubdomainVectorType       &dst,
                                       const InterfaceVectorType &src) const;

    void
    weight_local_interface_and_scatter(InterfaceVectorType       &dst,
                                       const SubdomainVectorType &src) const;



    // C_R applied to x: for each active edge/face primal constraint group l
    // (group indices [n_corner_local, n_local_coarse_dofs) -- corners are
    // excluded, see n_corner_local's class-level comment), out(l) =
    // coarse_weights(n_corner_local + l) * sum over the group's dofs of
    // x(dof) -- the group's weighted average, i.e. the same per-group
    // reduction project()/vmult_coarse_correction() already do, just written
    // into a small Vector<Number> (size n_edge_face_local) instead of
    // subtracted back into x. Shared by compute_local_edge_face_schur_
    // complement() (forming a column of edge_face_schur_matrix per basis
    // vector) and vmult_fine_correction_static_condensation() (forming
    // rhs_lambda = C_R t_R).
    void
    apply_edge_face_constraints(Vector<Number> &out, const SubdomainVectorType &x) const;


    void
    reset_timings() const;


    const std::array<double, 5> &
    get_timings() const;

    void
    reset_setup_timings() const;

    const std::array<double, 6> &
    get_setup_timings() const;

    // Reset before a solve_dd()-style outer loop, same convention as
    // reset_timings() -- accumulated across repeated vmult() calls when
    // use_static_condensation_fine_correction is true. All 0 otherwise
    // (never touched by the Ahat path).
    //
    // [0] = CG solve for t_R (A_RR_corner_pinned^{-1} fine_residual)
    // [1] = apply_edge_face_constraints (C_R reduction forming rhs_lambda)
    // [2] = edge_face_schur_matrix.solve (small dense triangular solve)
    // [3] = recombination (fine_solution.add(-lambda(l), w_l) loop)
    void
    reset_static_condensation_timings() const;

    const std::array<double, 4> &
    get_static_condensation_timings() const;

    // One-shot setup-phase timings for compute_local_edge_face_schur_
    // complement(), same convention as setup_timings above (reset
    // internally at the start of that call).
    //
    // [0] = building c_l (the per-group constraint-functional lift kernel), all l
    // [1] = the n_edge_face_local CG solves for w_l = A_RR_corner_pinned^{-1} c_l
    // [2] = apply_edge_face_constraints calls filling edge_face_schur_matrix's columns
    // [3] = set_property(symmetric) + compute_cholesky_factorization()
    const std::array<double, 4> &
    get_edge_face_setup_timings() const;

    unsigned int
    get_n_local_coarse_dofs() const;

    unsigned int
    get_n_global_coarse_dofs() const;


    struct SubdomainProjectorWrapper
    {
    public:
      const BDDCPreconditioner<dim, Number, BddcSmootherType> &parent;

      SubdomainProjectorWrapper(
        const BDDCPreconditioner<dim, Number, BddcSmootherType> &parent_preconditioner)
        : parent(parent_preconditioner)
      {}

      void
      project(SubdomainVectorType &subdomain_vector) const
      {
        parent.project_to_homogeneous_constraints_subdomain(subdomain_vector);
      }
    };

    unsigned int
    get_maximum_subdomain_mg_iterations() const;



  private:
    void
    setup_primal_constraint_views();


    ObserverPointer<const SchurInterfaceOperator<dim, Number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    const SubdomainPreconditioner &subdomain_mg_preconditioner;

    // A_RR's own MG preconditioner (nullptr unless the caller passed one in,
    // see the constructor's comment) -- only vmult_fine_correction_static_
    // condensation() dereferences this, i.e. only when
    // use_static_condensation_fine_correction is true.
    const SubdomainPreconditioner *subdomain_mg_preconditioner_corner_pinned;

    // Set via set_fine_correction_mode(); read by vmult() to pick which
    // algorithm its fine-correction step uses. See set_fine_correction_mode()'s
    // class-level comment.
    bool use_static_condensation_fine_correction = false;

    // Per-level BDDC matrices/transfers/smoothers, needed (alongside
    // subdomain_mg_preconditioner above) to build a second, block-shaped
    // V-cycle for compute_local_coarse_matrix() -- see the class-level
    // comment on BddcSmootherType for why the type-erased
    // subdomain_mg_preconditioner alone isn't enough.
    const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
                                                                      &level_bddc_matrices;
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers;
    const MGLevelObject<BddcSmootherType>                             &level_bddc_smoothers;

    // Corner-pinned hierarchy's own per-level transfers (nullptr unless the
    // caller passed them in -- see the constructor's comment), needed to
    // build a block-shaped corner-pinned V-cycle for compute_local_edge_
    // face_schur_complement()'s batched w_l solves. Same role as level_
    // bddc_transfers above, but for the static-condensation path instead
    // of the Pi-projected one; that block V-cycle's matrices/diagonal come
    // from level_bddc_matrices instead (see the constructor's comment for
    // why), so no separate corner-pinned matrices array is stored here.
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> *level_corner_pinned_transfers;

    SubdomainBDDCOperator<dim, Number> subdomain_bddc_operator;

    // Presents subdomain_bddc_operator.vmult_corner_pinned() (corners hard-
    // pinned, edges/faces free) as vmult() -- the interface SolverCG
    // expects -- for compute_local_edge_face_schur_complement()/
    // vmult_fine_correction_static_condensation(). See the class-level
    // comment on SubdomainBDDCOperatorCornerPinnedAdapter for why this can't
    // just be subdomain_bddc_operator.vmult() itself.
    SubdomainBDDCOperatorCornerPinnedAdapter<dim, Number> subdomain_bddc_operator_corner;

    // The global coarse matrix is a finite-element-like graph (each global
    // coarse dof only couples to the others sharing a subdomain with it),
    // so it's sparse -- assembled as such (see compute_coarse_matrix()) and
    // factorized/solved with UMFPACK instead of a dense LAPACKFullMatrix.
    // BNN's own coarse matrix stays dense: it needs an SVD-based
    // pseudoinverse (compute_inverse_svd()) rather than a direct
    // factorization, since it isn't guaranteed SPD/nonsingular the way
    // BDDC's is (checked below, in DEBUG builds).
    SparsityPattern      coarse_sparsity_pattern;
    SparseMatrix<Number> coarse_matrix;
    SparseDirectUMFPACK  coarse_matrix_solver;

    BDDCVariant bddc_variant;

    unsigned int       n_global_coarse_dofs;
    unsigned int       n_local_coarse_dofs;
    const unsigned int n_subdomain_dofs;
    const unsigned int interface_vector_size;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const DeviceVector<const Number> interface_weights;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    // Flattened fine local interface DoF indices associated with each local primal constraint
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_interface_local;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_subdomain;

    // offset of the primal constraint dofs per each dof entity
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_offsets;

    // subdomain (local) to global constraint map
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> coarse_dofs_local_to_global;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> corner_dofs_subdomain;

    std::vector<unsigned int> coarse_dofs_local_to_global_vector_host;

    std::vector<SubdomainVectorType> coarse_basis_functions;

    // Set by compute_local_edge_face_schur_complement(). n_corner_local is
    // the number of active corner (vertex) primal constraint groups --
    // corner_dofs_subdomain.size(), equivalently local_coarse_offsets[1] --
    // always the leading n_corner_local groups of n_local_coarse_dofs
    // (constraint groups are ordered vertices, then edges, then faces).
    // n_edge_face_local = n_local_coarse_dofs - n_corner_local is the size
    // of the Lagrange system (0 for BDDCVariant::corner, where hard-pinning
    // IS the whole primal set and no Lagrange system is needed at all --
    // vmult_fine_correction_static_condensation() short-circuits in that
    // case).
    unsigned int n_corner_local    = 0;
    unsigned int n_edge_face_local = 0;

    // Guards against enabling the static-condensation fine correction
    // before compute_local_edge_face_schur_complement() has run: without
    // this, n_edge_face_local's default of 0 would be indistinguishable
    // from "computed and turned out to be 0" (BDDCVariant::corner), and
    // vmult_fine_correction_static_condensation() would silently skip the
    // edge/face correction instead of producing a clear error.
    bool edge_face_schur_computed = false;

    // w_l = A_RR_corner_pinned^{-1} c_l for each active edge/face group l,
    // where c_l is that group's constraint functional lifted into
    // subdomain-vector space. Packed as n_edge_face_local blocks of
    // n_subdomain_dofs each in ONE contiguous vector (block l occupies
    // [l*n_subdomain_dofs, (l+1)*n_subdomain_dofs)) -- the same block-vector
    // convention compute_local_coarse_matrix()'s lifted_block uses. This
    // lets vmult_fine_correction_static_condensation()'s recombination step
    // read all blocks from one fused kernel instead of n_edge_face_local
    // separate AXPY launches. See compute_local_edge_face_schur_
    // complement()'s class-level comment.
    SubdomainVectorType edge_face_basis_block;

    // C_R A_RR_corner_pinned^{-1} C_R^t (size n_edge_face_local), Cholesky-
    // factorized in place by compute_local_edge_face_schur_complement() --
    // LOCAL (no MPI), unlike coarse_matrix/coarse_matrix_solver above, since
    // corner-only pinning already makes A_RR_corner_pinned SPD regardless of
    // floating-subdomain status (see compute_local_edge_face_schur_
    // complement()'s class-level comment).
    LAPACKFullMatrix<Number> edge_face_schur_matrix;

    mutable InterfaceVectorType temp_interface;
    mutable Vector<Number>      temp_coarse_local;
    mutable Vector<Number>      temp_coarse_global;


    mutable SubdomainVectorType temp_subdomain_dst;
    mutable SubdomainVectorType temp_subdomain_src;
    mutable SubdomainVectorType temp_subdomain_coarse;
    mutable SubdomainVectorType temp_subdomain_fine;


    mutable std::vector<Number> temp_global_coarse_std;

    /**
     * Per-vmult() solve-phase timings, reset externally (reset_timings())
     * and accumulated across repeated vmult() calls -- e.g. over the whole
     * outer interface CG solve -- so get_timings() reports totals for
     * whatever span the caller bracketed with reset_timings().
     *
     * timings[0] = gather_and_weight_global_interface + weight_local_interface_and_scatter
     * timings[1] = vmult_coarse_correction (global coarse problem solve)
     * timings[2] = vmult_fine_correction (local constrained CG solve)
     * timings[3] = total vmult() wall time
     * timings[4] = vmult_interface (outer Dirichlet solve via S, when driven
     *              through SolverProjectedCG::solve_dd() instead of vmult())
     */
    mutable std::array<double, 5> timings;

    // Set after the first vmult_fine_correction() call on this preconditioner
    // instance (one instance per refinement cycle), so the RHS-norm trace
    // below prints exactly once per cycle instead of once per outer CG
    // iteration.
    mutable bool printed_fine_correction_diagnostics = false;

    /**
     * One-shot setup-phase timings for compute_coarse_matrix() /
     * compute_local_coarse_matrix(), reset internally at the start of each
     * compute_coarse_matrix() call (it runs once per preconditioner setup,
     * not per iteration, so there is no meaningful "per outer solve" span
     * to accumulate over the way there is for timings[] above).
     *
     * setup_timings[0] = lift_coarse_to_subdomain (building lifted constraint vectors)
     * setup_timings[1] = vmult_plain calls (rhs assembly + S*phi_j)
     * setup_timings[2] = vmult_fine_correction (one CG solve per local primal constraint)
     * setup_timings[3] = local coarse matrix inner products
     * setup_timings[4] = MPI sum (global reduction of local coarse matrix contributions)
     * setup_timings[5] = LU factorization of the global coarse matrix
     */
    mutable std::array<double, 6> setup_timings;

    // See get_static_condensation_timings()'s class-level comment for the
    // slot meanings; accumulated only inside vmult_fine_correction_static_
    // condensation(), left at 0 by the Ahat path.
    mutable std::array<double, 4> static_condensation_timings{};

    // See get_edge_face_setup_timings()'s class-level comment; populated
    // once by compute_local_edge_face_schur_complement().
    mutable std::array<double, 4> edge_face_setup_timings{};

    mutable unsigned int max_subdomain_mg_iterations;
  };

  template <int dim, typename Number, typename BddcSmootherType>
  BDDCPreconditioner<dim, Number, BddcSmootherType>::BDDCPreconditioner(
    const SchurInterfaceOperator<dim, Number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
    const SubdomainPreconditioner                   &subdomain_mg_preconditioner,
    const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
                                                                      &level_bddc_matrices,
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers,
    const MGLevelObject<BddcSmootherType>                             &level_bddc_smoothers,
    const SubdomainPreconditioner *subdomain_mg_preconditioner_corner_pinned,
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>>
                      *level_corner_pinned_transfers,
    const BDDCVariant variant)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , subdomain_mg_preconditioner(subdomain_mg_preconditioner)
    , subdomain_mg_preconditioner_corner_pinned(subdomain_mg_preconditioner_corner_pinned)
    , level_bddc_matrices(level_bddc_matrices)
    , level_bddc_transfers(level_bddc_transfers)
    , level_bddc_smoothers(level_bddc_smoothers)
    , level_corner_pinned_transfers(level_corner_pinned_transfers)
    , subdomain_bddc_operator(subdomain_operator)
    , subdomain_bddc_operator_corner(subdomain_bddc_operator)
    , n_subdomain_dofs(subdomain_operator.get_subdomain_dof_handler().get_dof_handler().n_dofs())
    , interface_vector_size(subdomain_operator.get_interface_dof_indices_subdomain().size())
    , coarse_problem_rank(subdomain_dof_handler->n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
  {
    if (dim == 2 && variant == BDDCVariant::corner_edge_face)
      bddc_variant = BDDCVariant::corner_edge;
    else
      bddc_variant = variant;

    // store primal constraints
    {
      const auto &subdomain_dof_info = this->subdomain_dof_handler->get_dof_info();

      if (bddc_variant == BDDCVariant::corner)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[1]; // End of Vertices
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[1];

          if (n_local_coarse_dofs == 0)
            bddc_variant = BDDCVariant::corner_edge;
        }

      if (bddc_variant == BDDCVariant::corner_edge)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[2]; // End of Edges
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[2];

          if (n_local_coarse_dofs == 0)
            bddc_variant = BDDCVariant::corner_edge_face;
        }

      if (bddc_variant == BDDCVariant::corner_edge_face)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[3]; // End of Faces
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[3];
        }

      AssertThrow(n_global_coarse_dofs > 0, ExcMessage("There's zero global constraints"));
      AssertThrow(n_local_coarse_dofs > 0, ExcMessage("There's zero local constraints"));

      setup_primal_constraint_views();
    }

    temp_interface.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    temp_coarse_local.reinit(n_local_coarse_dofs);
    temp_coarse_global.reinit(n_global_coarse_dofs);

    temp_global_coarse_std.resize(n_global_coarse_dofs);

    subdomain_operator.initialize_dof_vector(temp_subdomain_dst);
    temp_subdomain_src.reinit(temp_subdomain_dst);
    temp_subdomain_coarse.reinit(temp_subdomain_dst);
    temp_subdomain_fine.reinit(temp_subdomain_dst);

    max_subdomain_mg_iterations = 0;

    reset_timings();
    reset_setup_timings();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::reset_timings() const
  {
    for (unsigned int i = 0; i < timings.size(); ++i)
      timings[i] = 0.;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::reset_setup_timings() const
  {
    for (unsigned int i = 0; i < setup_timings.size(); ++i)
      setup_timings[i] = 0.;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 6> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_setup_timings() const
  {
    return setup_timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::reset_static_condensation_timings() const
  {
    for (unsigned int i = 0; i < static_condensation_timings.size(); ++i)
      static_condensation_timings[i] = 0.;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 4> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_static_condensation_timings() const
  {
    return static_condensation_timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 4> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_edge_face_setup_timings() const
  {
    return edge_face_setup_timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_n_local_coarse_dofs() const
  {
    return n_local_coarse_dofs;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_n_global_coarse_dofs() const
  {
    return n_global_coarse_dofs;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 5> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_timings() const
  {
    return timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::set_fine_correction_mode(
    bool use_static_condensation)
  {
    Assert(!use_static_condensation || subdomain_mg_preconditioner_corner_pinned != nullptr,
           ExcMessage("This BDDCPreconditioner was constructed without a corner-pinned "
                      "A_RR MG preconditioner -- pass one to the constructor before "
                      "enabling the static-condensation fine correction."));
    Assert(!use_static_condensation || edge_face_schur_computed,
           ExcMessage("compute_local_edge_face_schur_complement() must be called before "
                      "enabling the static-condensation fine correction."));
    use_static_condensation_fine_correction = use_static_condensation;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult(InterfaceVectorType       &dst,
                                                           const InterfaceVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    Kokkos::fence();
    Timer total_time;

    dst = 0;
    src.update_ghost_values();

    Kokkos::fence();
    Timer time;
    gather_and_weight_global_interface(temp_subdomain_src, src);
    Kokkos::fence();
    timings[0] += time.wall_time();

    time.restart();
    vmult_coarse_correction(temp_subdomain_coarse, temp_subdomain_src);
    Kokkos::fence();
    timings[1] += time.wall_time();

    // vmult_coarse_correction() above needs the raw (unprojected) residual --
    // it takes inner products against the coarse basis functions, not Ahat/
    // Chat's sandwiched action -- so the fine-correction precondition is only
    // imposed here, after the coarse correction and before the fine one,
    // rather than earlier. Applying it to temp_subdomain_src in place lets
    // vmult_fine_correction()/vmult_fine_correction_static_condensation()
    // take their RHS by const reference directly (no internal copy) since
    // the precondition is now satisfied by the caller either way -- only
    // which precondition differs: project()'s per-group mean subtraction
    // (Pi, weak, over ALL active primal groups) for the Ahat path, vs
    // zero_corner_pinned_dofs()'s hard zero (Dirichlet, strong, CORNER
    // groups only) for the static-condensation path. See
    // set_fine_correction_mode()'s class-level comment for what selects
    // between the two.
    time.restart();
    if (use_static_condensation_fine_correction)
      {
        // Only CORNER positions need zeroing here (matching x_C = 0 in the
        // classical construction) -- edge/face values must stay intact,
        // they're genuine input to the Lagrange system inside
        // vmult_fine_correction_static_condensation(), not pinned away.
        this->subdomain_bddc_operator.zero_corner_pinned_dofs(temp_subdomain_src);
        vmult_fine_correction_static_condensation(temp_subdomain_fine, temp_subdomain_src);
      }
    else
      {
        this->subdomain_bddc_operator.project(temp_subdomain_src);
        vmult_fine_correction(temp_subdomain_fine, temp_subdomain_src);
      }
    Kokkos::fence();
    timings[2] += time.wall_time();

    temp_subdomain_coarse += temp_subdomain_fine;

    time.restart();
    weight_local_interface_and_scatter(dst, temp_subdomain_coarse);
    Kokkos::fence();
    timings[0] += time.wall_time();

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    Kokkos::fence();
    timings[3] += total_time.wall_time();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_interface(
    InterfaceVectorType       &dst,
    const InterfaceVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    Kokkos::fence();
    Timer time;
    this->interface_operator->vmult(dst, src);
    Kokkos::fence();
    timings[4] += time.wall_time();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_fine_correction(
    SubdomainVectorType       &fine_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(fine_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);

    fine_solution = 0;


    // subdomain_bddc_operator.vmult() == Pi*A*Pi and subdomain_mg_preconditioner.vmult()
    // are both symmetric and map into V = range(Pi) by construction (Pi is baked into
    // vmult() itself, and the BDDC-level smoother's elementary correction is sandwiched
    // as Pi*(omega*D^{-1})*Pi), so plain CG works directly -- only the RHS needs to be
    // in V up front.
    //
    // Precondition: fine_residual must already satisfy Pi*fine_residual == fine_residual
    // on entry -- this function does NOT project it. SolverCG::solve() takes its RHS by
    // const reference and never writes to it internally, so as long as the caller has
    // already projected the vector it's about to pass in (both callers do: vmult()
    // projects temp_subdomain_src in place right before calling this, and
    // compute_local_coarse_matrix() projects its local rhs before calling this), the
    // fine_residual reference can be handed straight to solve() with no extra copy.

    // fine_solution == 0 on entry, so the initial residual ReductionControl
    // measures internally is exactly ||fine_residual|| -- same quantity the
    // trace below reports, and the same one dirichlet_solve_subdomain()'s
    // ReductionControl(100, 1e-16, 1e-12) is built on (see
    // portable_schur_interface_operator.h). Floor kept at 1e-15 rather than
    // that 1e-16 -- one order of magnitude of headroom above double epsilon
    // for CG's accumulated round-off, still to be checked against 1e-16
    // empirically via the Dirichlet trace.
    if (!printed_fine_correction_diagnostics &&
        Utilities::MPI::this_mpi_process(this->subdomain_dof_handler->get_mpi_communicator()) == 0)
      {
        const Number rhs_norm    = fine_residual.l2_norm();
        const Number rel_tol_abs = 1e-12 * rhs_norm;
        std::cout << "[fine-correction RHS trace] ||fine_residual||_2 = " << rhs_norm
                  << ", n_subdomain_dofs = " << n_subdomain_dofs
                  << ", interface_vector_size = " << interface_vector_size
                  << ", sqrt(n_subdomain_dofs) = "
                  << std::sqrt(static_cast<double>(n_subdomain_dofs))
                  << ", sqrt(interface_vector_size) = "
                  << std::sqrt(static_cast<double>(interface_vector_size))
                  << ", 1e-12*||rhs|| = " << rel_tol_abs
                  << ", floor (1e-15) active = " << (rel_tol_abs < 1e-15 ? "YES" : "no")
                  << std::endl;
        printed_fine_correction_diagnostics = true;
      }
    ReductionControl              solver_control(fine_residual.size(), 1e-12, 1e-9);
    SolverCG<SubdomainVectorType> solver(solver_control);
    solver.solve(this->subdomain_bddc_operator,
                 fine_solution,
                 fine_residual,
                 subdomain_mg_preconditioner);
    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));

    // subdomain_mg_preconditioner.vmult(fine_solution, fine_residual);
    // max_subdomain_mg_iterations = std::max(max_subdomain_mg_iterations, 1u);

    // std::cout << "Constrained projected solver converged in " << solver_control.last_step()
    //           << "  iterations." << std::endl;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::apply_edge_face_constraints(
    Vector<Number>            &out,
    const SubdomainVectorType &x) const
  {
    AssertDimension(out.size(), n_edge_face_local);

    if (n_edge_face_local == 0)
      return;

    DeviceVector<const Number> x_view(x.get_values(), n_subdomain_dofs);

    const auto         offsets         = this->primal_constraint_offsets;
    const auto         constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto         weights         = this->coarse_weights;
    const unsigned int n_corner        = this->n_corner_local;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> out_device(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "edge_face_constraint_values"),
      n_edge_face_local);

    Kokkos::parallel_for(
      "apply_edge_face_constraints", n_edge_face_local, KOKKOS_LAMBDA(const int l) {
        const unsigned int k     = n_corner + l;
        const unsigned int start = offsets(k);
        const unsigned int end   = offsets(k + 1);

        Number sum = 0;
        for (unsigned int i = start; i < end; ++i)
          sum += x_view(constraint_dofs(i));

        out_device(l) = sum * weights(k);
      });
    Kokkos::fence();

    auto out_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out_device);
    for (unsigned int l = 0; l < n_edge_face_local; ++l)
      out(l) = out_host(l);
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::compute_local_edge_face_schur_complement()
  {
    Assert(subdomain_mg_preconditioner_corner_pinned != nullptr,
           ExcMessage("This BDDCPreconditioner was constructed without a corner-pinned "
                      "A_RR MG preconditioner -- pass one to the constructor before calling "
                      "compute_local_edge_face_schur_complement()."));

    const auto &subdomain_dof_info = this->subdomain_dof_handler->get_dof_info();
    n_corner_local                 = subdomain_dof_info.local_coarse_offsets[1];
    AssertThrow(n_corner_local <= n_local_coarse_dofs, ExcInternalError());
    n_edge_face_local = n_local_coarse_dofs - n_corner_local;

    edge_face_basis_block.reinit(static_cast<typename SubdomainVectorType::size_type>(
                                    n_edge_face_local) *
                                  n_subdomain_dofs);
    edge_face_schur_matrix.reinit(n_edge_face_local, n_edge_face_local);

    // BDDCVariant::corner: the entire active primal set IS corners, so
    // hard-pinning already handles everything -- no Lagrange system needed
    // at all (this is exactly the case where hard-pin and Pi coincide
    // exactly, singleton groups, see [[project-static-condensation-arr]]).
    if (n_edge_face_local == 0)
      {
        edge_face_schur_computed = true;
        return;
      }

    const auto         offsets         = this->primal_constraint_offsets;
    const auto         constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto         weights         = this->coarse_weights;
    const unsigned int n_corner        = this->n_corner_local;

    Assert(level_corner_pinned_transfers != nullptr,
           ExcMessage("Block-batched edge/face basis solves require the corner-pinned level "
                      "transfers to have been passed to the BDDCPreconditioner constructor (in "
                      "lockstep with subdomain_mg_preconditioner_corner_pinned)."));

    Vector<Number> schur_column(n_edge_face_local);

    for (unsigned int i = 0; i < edge_face_setup_timings.size(); ++i)
      edge_face_setup_timings[i] = 0.;

    Timer time;

    // Build all n_edge_face_local c_l lifts into one packed block vector
    // (same n_edge_face_local-blocks-of-n_subdomain_dofs layout
    // edge_face_basis_block uses), then batch the n_edge_face_local
    // A_RR_corner_pinned^{-1} c_l solves into ONE SolverBlockCG call via a
    // block-shaped corner-pinned V-cycle, instead of n_edge_face_local
    // sequential SolverCG calls -- per JUPITER data this dominates this
    // function's cost (~98%), same motivation/pattern as compute_local_
    // coarse_matrix()'s existing Pi-path block-CG.
    SubdomainVectorType c_block;
    c_block.reinit(static_cast<typename SubdomainVectorType::size_type>(n_edge_face_local) *
                   n_subdomain_dofs);
    c_block = 0;

    {
      DeviceVector<Number> c_block_view(c_block.get_values(),
                                        static_cast<typename SubdomainVectorType::size_type>(
                                          n_edge_face_local) *
                                          n_subdomain_dofs);

      Kokkos::fence();
      time.restart();
      Kokkos::parallel_for(
        "build_edge_face_constraint_lift_block", n_edge_face_local, KOKKOS_LAMBDA(const int l) {
          const unsigned int k     = n_corner + l;
          const unsigned int start = offsets(k);
          const unsigned int end   = offsets(k + 1);
          const Number       w     = weights(k);
          for (unsigned int i = start; i < end; ++i)
            c_block_view(static_cast<std::size_t>(l) * n_subdomain_dofs + constraint_dofs(i)) = w;
        });
      Kokkos::fence();
      edge_face_setup_timings[0] += time.wall_time();
    }

    // Block-shaped V-cycle over the corner-pinned hierarchy: matrices come
    // from level_bddc_matrices (the SAME concrete SubdomainBDDCOperator
    // objects the Pi-path's own block V-cycle below already casts to --
    // level_subdomain_corner_pinned_matrices in program.cc holds
    // SubdomainBDDCOperatorCornerPinnedAdapter-wrapped views of those same
    // objects, not the concrete type itself, so casting through THAT array
    // would be undefined behavior), transfers from level_corner_pinned_
    // transfers (genuinely different constraints than level_bddc_transfers,
    // corner-pinned-aware). Built from BlockCornerPinnedOperatorAdapter (no
    // projection step: A_RR's corner pinning is baked into the mask
    // already) and plain dealii::PreconditionChebyshev as the smoother
    // (matches the corner-pinned SCALAR hierarchy's own choice in
    // program.cc, and for the same reason -- A_RR is genuinely SPD
    // everywhere, unlike Ahat, so a generic internal Lanczos eigenvalue
    // estimate is safe here, no special projected-safe estimator needed).
    using BlockOperatorType = BlockCornerPinnedOperatorAdapter<dim, Number>;
    using BlockTransferType = BlockTransferAdapter<dim, Number>;
    using BlockPreconditionerType =
      BlockProjectedDiagonalPreconditioner<BlockOperatorType, SubdomainVectorType>;
    using BlockSmootherType = PreconditionChebyshev<BlockOperatorType, SubdomainVectorType, BlockPreconditionerType>;

    const unsigned int minlevel = level_bddc_matrices.min_level();
    const unsigned int maxlevel = level_bddc_matrices.max_level();

    MGLevelObject<std::unique_ptr<BlockOperatorType>> block_matrices(minlevel, maxlevel);
    MGLevelObject<std::unique_ptr<BlockTransferType>> block_transfers(minlevel, maxlevel);
    MGLevelObject<BlockSmootherType>                  block_smoothers(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const auto &concrete_matrix =
          static_cast<const SubdomainBDDCOperator<dim, Number> &>(*level_bddc_matrices[level]);

        block_matrices[level] =
          std::make_unique<BlockOperatorType>(concrete_matrix, n_edge_face_local);

        if (level > minlevel)
          block_transfers[level] = std::make_unique<BlockTransferType>(
            *(*level_corner_pinned_transfers)[level], n_edge_face_local);

        typename BlockSmootherType::AdditionalData smoother_data;
        // degree/smoothing_range mirror the corner-pinned SCALAR hierarchy's
        // own per-level convention in program.cc's setup_smoothers()
        // (level 0: near-exact via degree = invalid_unsigned_int; interior
        // levels: smoothing_range = 15, degree = n_pre_smooth) -- n_pre_smooth
        // itself isn't reachable from here (PreconditionChebyshev keeps its
        // AdditionalData private, no accessor), so it's borrowed from the
        // already-accessible BDDC scalar smoother's degree at the same
        // level, which uses the identical n_pre_smooth value at level>0.
        // max_eigenvalue is deliberately left unset: eig_cg_n_iterations>0
        // below triggers PreconditionChebyshev's own internal Lanczos
        // estimate, safe here since A_RR is genuinely SPD (unlike Ahat).
        if (level == minlevel)
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = concrete_matrix.m();
          }
        else
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = level_bddc_smoothers[level].get_additional_data().degree;
            smoother_data.eig_cg_n_iterations = 10;
          }
        smoother_data.preconditioner = std::make_shared<BlockPreconditionerType>(
          *block_matrices[level],
          concrete_matrix.get_matrix_diagonal_inverse_corner_pinned(),
          n_edge_face_local);

        block_smoothers[level].initialize(*block_matrices[level], smoother_data);
      }

    SubdomainVCycleMultigrid<dim, Number, BlockOperatorType, BlockTransferType, BlockSmootherType>
      block_mg_preconditioner(block_matrices, block_transfers, block_smoothers);

    SubdomainVectorType w_block;
    w_block.reinit(c_block);

    ReductionControl                   solver_control(c_block.size(), 1e-14, 1e-11);
    SolverBlockCG<SubdomainVectorType> block_solver(solver_control);

    time.restart();
    block_solver.solve_block(*block_matrices[maxlevel],
                             w_block,
                             c_block,
                             block_mg_preconditioner,
                             n_edge_face_local);
    Kokkos::fence();
    edge_face_setup_timings[1] += time.wall_time();

    // w_block is already laid out exactly as edge_face_basis_block expects
    // (block l = A_RR_corner_pinned^{-1} c_l, at offset l*n_subdomain_dofs)
    // -- no unpack/copy needed, just adopt it directly.
    edge_face_basis_block.swap(w_block);

    // Column l of edge_face_schur_matrix = C_R w_l (entry m = c_m . w_l),
    // read out of the packed block one column at a time -- small/cheap
    // relative to the solve above (per JUPITER data ~2% of this function's
    // cost), so left as a per-l loop rather than also batched.
    // apply_edge_face_constraints() takes a SubdomainVectorType by
    // reference (it owns its Kokkos storage, no aliasing-into-a-block
    // constructor exists), so each column is deep_copy'd into a reused
    // scratch vector first, same idiom the setup loop above used to use
    // for the reverse (scratch -> block) direction.
    SubdomainVectorType w_l_scratch;
    w_l_scratch.reinit(temp_subdomain_dst);

    time.restart();
    for (unsigned int l = 0; l < n_edge_face_local; ++l)
      {
        DeviceVector<const Number> block_view(edge_face_basis_block.get_values() +
                                                static_cast<typename SubdomainVectorType::size_type>(
                                                  l) *
                                                  n_subdomain_dofs,
                                              n_subdomain_dofs);
        DeviceVector<Number> w_l_view(w_l_scratch.get_values(), n_subdomain_dofs);
        Kokkos::deep_copy(w_l_view, block_view);

        apply_edge_face_constraints(schur_column, w_l_scratch);
        for (unsigned int m = 0; m < n_edge_face_local; ++m)
          edge_face_schur_matrix(m, l) = schur_column(m);
      }
    Kokkos::fence();
    edge_face_setup_timings[2] += time.wall_time();

    // compute_cholesky_factorization() asserts property == symmetric.
    time.restart();
    edge_face_schur_matrix.set_property(LAPACKSupport::symmetric);
    edge_face_schur_matrix.compute_cholesky_factorization();
    edge_face_setup_timings[3] += time.wall_time();

    edge_face_schur_computed = true;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_fine_correction_static_condensation(
    SubdomainVectorType       &fine_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(fine_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);
    Assert(subdomain_mg_preconditioner_corner_pinned != nullptr,
           ExcMessage("This BDDCPreconditioner was constructed without a corner-pinned "
                      "A_RR MG preconditioner -- pass one to the constructor to use "
                      "vmult_fine_correction_static_condensation()."));

    fine_solution = 0;

    // t_R = A_RR_corner_pinned^{-1} fine_residual. subdomain_bddc_operator_
    // corner.vmult() == vmult_corner_pinned() is symmetric and, thanks to
    // the identity block at corner-pinned dofs baked into its dof mask,
    // maps a vector that's zero there straight back to another vector
    // that's zero there -- so plain CG works directly, same as
    // vmult_fine_correction()'s Ahat solve.
    //
    // Precondition: fine_residual must already be zero at CORNER-pinned dof
    // positions on entry (matching x_C = 0) -- this function does NOT zero
    // it. Caller: vmult() zeros temp_subdomain_src in place (via
    // zero_corner_pinned_dofs()) right before calling this, when
    // use_static_condensation_fine_correction is true.
    Timer time;

    ReductionControl              solver_control(fine_residual.size(), 1e-12, 1e-9);
    SolverCG<SubdomainVectorType> solver(solver_control);
    solver.solve(this->subdomain_bddc_operator_corner,
                 fine_solution,
                 fine_residual,
                 *subdomain_mg_preconditioner_corner_pinned);
    Kokkos::fence();
    static_condensation_timings[0] += time.wall_time();

    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));

    // BDDCVariant::corner: t_R alone is already the answer -- no Lagrange
    // system was built (see compute_local_edge_face_schur_complement()).
    if (n_edge_face_local == 0)
      return;

    // rhs_lambda = C_R t_R, then lambda = edge_face_schur_matrix^{-1}
    // rhs_lambda (precomputed Cholesky factor, in-place triangular solve).
    // Sign derived directly from the saddle-point system (19): row 4 is
    // C_R x_R = 0; substituting x_R = t_R - A_RR^{-1} C_R^t lambda gives
    // C_R A_RR^{-1} C_R^t lambda = +C_R t_R (the textbook version of this
    // step is sometimes printed with the RHS negated -- checked against the
    // derivation, that sign is wrong; this is the corrected version).
    Vector<Number> lambda(n_edge_face_local);

    time.restart();
    apply_edge_face_constraints(lambda, fine_solution);
    Kokkos::fence();
    static_condensation_timings[1] += time.wall_time();

    time.restart();
    edge_face_schur_matrix.solve(lambda);
    static_condensation_timings[2] += time.wall_time();

    time.restart();

    // Copy lambda to device once (tiny, ~n_edge_face_local doubles), then
    // fuse the whole recombination into ONE kernel over n_subdomain_dofs
    // with an inner loop over the small n_edge_face_local dimension --
    // replaces n_edge_face_local separate full-vector AXPY kernel launches,
    // same fusion idea as fused_recurrence_update() in
    // portable_projected_chebyshev_smoother.h.
    const Kokkos::View<const Number *, Kokkos::HostSpace> lambda_host_view(lambda.data(),
                                                                           n_edge_face_local);
    auto lambda_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(), lambda_host_view);

    DeviceVector<Number>       fine_solution_view(fine_solution.get_values(), n_subdomain_dofs);
    DeviceVector<const Number> basis_block_view(edge_face_basis_block.get_values(),
                                                static_cast<typename SubdomainVectorType::size_type>(
                                                  n_edge_face_local) *
                                                  n_subdomain_dofs);
    const unsigned int n_edge_face = this->n_edge_face_local;
    const unsigned int n_dofs      = this->n_subdomain_dofs;

    Kokkos::parallel_for(
      "static_condensation_recombine", n_subdomain_dofs, KOKKOS_LAMBDA(const int i) {
        Number sum = fine_solution_view(i);
        for (unsigned int l = 0; l < n_edge_face; ++l)
          sum -= lambda_device_view(l) * basis_block_view(l * n_dofs + i);
        fine_solution_view(i) = sum;
      });
    Kokkos::fence();
    static_condensation_timings[3] += time.wall_time();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::solve_fine_correction(
    SubdomainVectorType       &fine_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(fine_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);

    fine_solution = 0;

    SolverControl solver_control(fine_residual.size(), 1e-12 * fine_residual.l2_norm());
    // ReductionControl solver_control(100, 1e-16, 1e-12);

    // subdomain_bddc_operator.vmult() == Pi*A*Pi and subdomain_mg_preconditioner.vmult()
    // are both symmetric and map into V = range(Pi) by construction (Pi is baked into
    // vmult() itself, and the BDDC-level smoother's elementary correction is sandwiched
    // as Pi*(omega*D^{-1})*Pi), so plain CG works directly -- only the RHS needs to be
    // in V up front.
    //
    // Precondition: fine_residual must already satisfy Pi*fine_residual == fine_residual
    // on entry -- this function does NOT project it. SolverCG::solve() takes its RHS by
    // const reference and never writes to it internally, so as long as the caller has
    // already projected the vector it's about to pass in (both callers do: vmult()
    // projects temp_subdomain_src in place right before calling this, and
    // compute_local_coarse_matrix() projects its local rhs before calling this), the
    // fine_residual reference can be handed straight to solve() with no extra copy.

    SolverCG<SubdomainVectorType> solver(solver_control);
    solver.solve(this->subdomain_bddc_operator,
                 fine_solution,
                 fine_residual,
                 subdomain_mg_preconditioner);
    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));


    // std::cout << "Constrained projected solver converged in " << solver_control.last_step()
    //           << "  iterations." << std::endl;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_maximum_subdomain_mg_iterations() const
  {
    return max_subdomain_mg_iterations;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_coarse_correction(
    SubdomainVectorType       &coarse_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(coarse_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);

    coarse_solution = 0;

    const unsigned int n_coarse_local  = this->n_local_coarse_dofs;
    const unsigned int n_coarse_global = this->n_global_coarse_dofs;

    const auto interface_dof_subdomain = this->interface_dof_indices_subdomain;

    temp_coarse_local = 0;

    DeviceVector<const Number> fine_residual_view(fine_residual.get_values(), fine_residual.size());
    DeviceVector<Number> coarse_solution_view(coarse_solution.get_values(), coarse_solution.size());

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        const SubdomainVectorType &basis_function = this->coarse_basis_functions[j];

        DeviceVector<const Number> basis_function_view(basis_function.get_values(),
                                                       basis_function.size());

        Number local_inner_product = 0;

        Kokkos::parallel_reduce(
          "coarse_rhs_inner_product",
          this->interface_vector_size,
          KOKKOS_LAMBDA(const int i, Number &sum) {
            const unsigned int subdomain_idx = interface_dof_subdomain(i);
            sum += fine_residual_view(subdomain_idx) * basis_function_view(subdomain_idx);
          },
          local_inner_product);

        temp_coarse_local(j) = local_inner_product;
      }

    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    const auto &local_to_global = this->coarse_dofs_local_to_global_vector_host;

    for (unsigned int i = 0; i < n_coarse_local; ++i)
      {
        temp_global_coarse_std[local_to_global[i]] = temp_coarse_local(i);
      }

    Utilities::MPI::sum(temp_global_coarse_std,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        temp_global_coarse_std);

    for (unsigned int i = 0; i < n_coarse_global; ++i)
      temp_coarse_global(i) = temp_global_coarse_std[i];

    coarse_matrix_solver.solve(temp_coarse_global);

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        const Number local_coarse_value = temp_coarse_global[local_to_global[j]];

        const SubdomainVectorType &basis_function = this->coarse_basis_functions[j];

        DeviceVector<const Number> basis_function_view(basis_function.get_values(),
                                                       basis_function.size());

        Kokkos::parallel_for(
          "coarse_prolongation", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
            const unsigned int subdomain_idx = interface_dof_subdomain(i);
            coarse_solution_view(subdomain_idx) +=
              local_coarse_value * basis_function_view(subdomain_idx);
          });
        Kokkos::fence();
      }
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::gather_and_weight_global_interface(
    SubdomainVectorType       &dst,
    const InterfaceVectorType &src) const
  {
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(dst.size(), n_subdomain_dofs);

    dst = 0;

    DeviceVector<Number>       dst_view(dst.get_values(), dst.size());
    DeviceVector<const Number> src_view(src.get_values(), this->interface_vector_size);

    const auto &weights                 = interface_weights;
    const auto  interface_dof_subdomain = this->interface_dof_indices_subdomain;

    Kokkos::parallel_for(
      "scale_interface_residual", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
        dst_view(interface_dof_subdomain(i)) = src_view(i) * weights(i);
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::weight_local_interface_and_scatter(
    InterfaceVectorType       &dst,
    const SubdomainVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(src.size(), n_subdomain_dofs);

    dst = 0;

    DeviceVector<Number>       dst_view(dst.get_values(), this->interface_vector_size);
    DeviceVector<const Number> src_view(src.get_values(), src.size());

    const auto &weights = interface_weights;

    const auto interface_dof_subdomain = this->interface_dof_indices_subdomain;

    Kokkos::parallel_for(
      "scale_interface_residual", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
        dst_view(i) = src_view(interface_dof_subdomain(i)) * weights(i);
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::project_to_homogeneous_constraints_interface(
    InterfaceVectorType &interface_vector) const
  {
    auto interface_vector_view = interface_vector.get_values();

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;
    const auto weights               = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              interface_vector_view(local_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const
  {
    subdomain_vector                 = 0;
    const auto interface_vector_view = interface_vector.get_values();
    auto       subdomain_vector_view = subdomain_vector.get_values();

    const auto offsets                   = this->primal_constraint_offsets;
    const auto local_constraint_dofs     = this->primal_constraint_dofs_interface_local;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface_and_scatter_to_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) =
                interface_vector_view(local_constraint_dofs(i)) - average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::project_to_homogeneous_constraints_subdomain(
    SubdomainVectorType &subdomain_vector) const
  {
    DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(),
                                               subdomain_vector.size());

    const auto offsets                   = this->primal_constraint_offsets;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += subdomain_vector_view(subdomain_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::global_interface_to_coarse(
    Vector<Number>            &coarse_vector,
    const InterfaceVectorType &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);
    coarse_vector = 0;

    interface_vector.update_ghost_values();

    auto interface_vector_view = interface_vector.get_values();

    const auto weights = this->coarse_weights;

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> local_coarse_contribution(
      "local_coarse_contribution", this->n_local_coarse_dofs);

    Kokkos::parallel_for(
      "global_interface_to_coarse_local_sum",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        Number average = 0;
        for (int i = start; i < end; ++i)
          average += interface_vector_view(local_constraint_dofs(i));

        average *= weights(coarse_local_idx);

        local_coarse_contribution(coarse_local_idx) = average;
      });

    Kokkos::fence();

    auto local_coarse_contribution_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_coarse_contribution);

    AssertDimension(temp_global_coarse_std.size(), this->n_global_coarse_dofs);
    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    for (unsigned int local_idx = 0; local_idx < this->n_local_coarse_dofs; ++local_idx)
      {
        const unsigned int global_coarse_idx = coarse_dofs_local_to_global_vector_host[local_idx];
        temp_global_coarse_std[global_coarse_idx] = local_coarse_contribution_host(local_idx);
      }

    Utilities::MPI::sum(temp_global_coarse_std,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        coarse_vector.get_values());

    interface_vector.zero_out_ghost_values();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::coarse_to_global_interface(
    InterfaceVectorType  &interface_vector,
    const Vector<Number> &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);

    interface_vector = 0;

    AssertDimension(temp_global_coarse_std.size(), this->n_global_coarse_dofs);
    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    for (unsigned int i = 0; i < this->n_local_coarse_dofs; ++i)
      temp_global_coarse_std[i] = coarse_vector(coarse_dofs_local_to_global_vector_host[i]);

    // copy to host
    Kokkos::View<Number *, Kokkos::HostSpace> local_coarse_host_view(temp_global_coarse_std.data(),
                                                                     this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);

    auto interface_vector_view = interface_vector.get_values();

    const auto weights               = this->coarse_weights;
    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::parallel_for(
      "coarse_to_global_interface_interpolate",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const Number coarse_value =
          weights(coarse_local_idx) * local_coarse_device_view(coarse_local_idx);

        for (unsigned int i = start; i < end; ++i)
          interface_vector_view(local_constraint_dofs(i)) = coarse_value;
      });

    Kokkos::fence();

    // update globally
    interface_vector.compress(VectorOperation::insert);
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::lift_coarse_to_subdomain(
    SubdomainVectorType  &subdomain_vector,
    const Vector<Number> &coarse_vector) const
  {
    AssertDimension(subdomain_vector.size(), this->n_subdomain_dofs);

    AssertDimension(coarse_vector.size(), this->n_local_coarse_dofs);

    subdomain_vector = 0.;

    // copy to host
    const Kokkos::View<const Number *, Kokkos::HostSpace> local_coarse_host_view(
      coarse_vector.data(), this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);
    Kokkos::fence();

    DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(),
                                               subdomain_vector.size());

    const auto constraint_dof_subdomain = this->primal_constraint_dofs_subdomain;
    const auto offsets                  = this->primal_constraint_offsets;
    const auto weights                  = this->coarse_weights;

    Kokkos::parallel_for(
      "lift_coarse_constraints",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int local_coarse_idx) {
        const unsigned int start = offsets(local_coarse_idx);
        const unsigned int end   = offsets(local_coarse_idx + 1);

        const Number coarse_value =
          local_coarse_device_view(local_coarse_idx) * weights(local_coarse_idx);

        for (unsigned int i = start; i < end; ++i)
          {
            subdomain_vector_view(constraint_dof_subdomain(i)) = coarse_value;
          }
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::compute_local_coarse_matrix(
    LAPACKFullMatrix<Number> &local_coarse_matrix)
  {
    const unsigned int n_coarse_local = this->n_local_coarse_dofs;
    local_coarse_matrix.reinit(n_coarse_local, n_coarse_local);

    coarse_basis_functions.resize(n_coarse_local);
    for (unsigned int i = 0; i < n_coarse_local; ++i)
      coarse_basis_functions[i].reinit(temp_subdomain_dst);

    SubdomainVectorType S_per_phi_j;
    S_per_phi_j.reinit(temp_subdomain_dst);

    DeviceVector<Number> temp_interface_view(temp_interface.get_values(), interface_vector_size);

    std::vector<SubdomainVectorType> lifted_constraints(n_coarse_local);

    Vector<Number> e_k(n_coarse_local);

    Kokkos::fence();
    Timer time;

    for (unsigned int k = 0; k < n_coarse_local; ++k)
      {
        e_k    = 0.;
        e_k(k) = Number(1);

        SubdomainVectorType &lift = lifted_constraints[k];

        lift.reinit(temp_subdomain_src);

        this->lift_coarse_to_subdomain(lift, e_k);
      }

    Kokkos::fence();
    setup_timings[0] += time.wall_time();

    // Two ways to extend each l_j into a coarse basis function phi_j =
    // l_j + correction_j, selected by the SAME flag vmult()'s fine
    // correction uses. See [[project-static-condensation-arr]] for the
    // derivation of why both are valid: Pi's correction_j lies in ker(C)
    // because Ahat = Pi*A*Pi always maps into range(Pi) = ker(C), while
    // static condensation's correction_j lies in ker(C) because corners
    // are hard-pinned and the Lagrange constraint enforces C_R x_R = 0 --
    // so C(phi_j) = C(l_j) + C(correction_j) = C(l_j) is IDENTICAL either
    // way (the reproduction property that makes phi_j a valid BDDC coarse
    // basis function), even though the two algorithms solve genuinely
    // different linear systems for correction_j.
    if (use_static_condensation_fine_correction)
      {
        // Static-condensation path: reuses vmult_fine_correction_static_
        // condensation() (and the edge_face_schur_matrix/edge_face_basis_
        // block it depends on, already built by compute_local_edge_face_
        // schur_complement() before this function runs) instead of the
        // Pi-projected block V-cycle below. Eliminates project_block()'s
        // expensive group-mean reduction from a SECOND (setup-time) call
        // site -- replacing it with the much cheaper, purely structural
        // zero_corner_pinned_dofs() -- on top of the per-iteration call
        // site vmult_fine_correction_static_condensation() already
        // replaces it at.
        SubdomainVectorType rhs_j, correction_j;
        rhs_j.reinit(temp_subdomain_dst);
        correction_j.reinit(temp_subdomain_dst);

        time.restart();
        for (unsigned int j = 0; j < n_coarse_local; ++j)
          {
            this->subdomain_bddc_operator.vmult_plain(rhs_j, lifted_constraints[j]);
            rhs_j *= Number(-1);
            this->subdomain_bddc_operator.zero_corner_pinned_dofs(rhs_j);

            correction_j = 0;
            this->vmult_fine_correction_static_condensation(correction_j, rhs_j);

            SubdomainVectorType &phi_j = coarse_basis_functions[j];
            phi_j                      = lifted_constraints[j];
            phi_j.add(Number(1), correction_j);
          }
        Kokkos::fence();
        setup_timings[2] += time.wall_time();

        // Verification: recompute phi_j via the Pi-projected sequential
        // path (project() + solve_fine_correction(), the same reference
        // machinery the block-Pi verification below uses) and diff against
        // the static-condensation result above -- confirms the
        // [[project-static-condensation-arr]] derivation numerically, not
        // just structurally. Cheap enough (one extra sequential CG solve
        // per primal group) to leave on by default; disable (#if 0) like
        // the block-Pi check below if a clean timing run of the static-
        // condensation setup path is needed later.
#if 1
        {
          SubdomainVectorType phi_ref, rhs_ref, correction_ref;
          phi_ref.reinit(temp_subdomain_dst);
          rhs_ref.reinit(temp_subdomain_src);
          correction_ref.reinit(temp_subdomain_dst);

          Number max_phi_diff = 0;
          Number max_phi_norm = 0;

          for (unsigned int j = 0; j < n_coarse_local; ++j)
            {
              phi_ref = lifted_constraints[j];

              this->subdomain_bddc_operator.vmult_plain(rhs_ref, phi_ref);
              rhs_ref *= Number(-1);
              this->subdomain_bddc_operator.project(rhs_ref);

              correction_ref = 0;
              this->solve_fine_correction(correction_ref, rhs_ref);

              phi_ref.add(Number(1), correction_ref);

              SubdomainVectorType diff = phi_ref;
              diff -= coarse_basis_functions[j];
              max_phi_diff = std::max(max_phi_diff, diff.linfty_norm());
              max_phi_norm = std::max(max_phi_norm, phi_ref.linfty_norm());
            }

          std::cout << "[static-condensation coarse-matrix verification] subdomain "
                    << this_subdomain << ": max |phi_sc - phi_pi| = " << max_phi_diff
                    << ", max |phi_pi| = " << max_phi_norm << std::endl;
        }
#endif
      }
    else
      {
        // Pack the n_coarse_local lifted constraint vectors into one block
        // vector (n_coarse_local blocks of n_subdomain_dofs each -- the
        // same layout vmult_plain_block()/project_block()/SolverBlockCG
        // use throughout). phi_j starts out equal to lifted_constraints[j]
        // before any fine correction is added (see the old per-j loop this
        // replaces), so this block vector is both the "phi_j at
        // block-solve time" input to vmult_plain_block() below *and*,
        // unpacked back out further down, becomes coarse_basis_functions'
        // initial value.
        SubdomainVectorType lifted_block;
        lifted_block.reinit(static_cast<typename SubdomainVectorType::size_type>(n_coarse_local) *
                            n_subdomain_dofs);

        for (unsigned int k = 0; k < n_coarse_local; ++k)
          {
            DeviceVector<Number> src_view(lifted_constraints[k].get_values(), n_subdomain_dofs);
            DeviceVector<Number> dst_view(lifted_block.get_values() + k * n_subdomain_dofs,
                                          n_subdomain_dofs);
            Kokkos::deep_copy(dst_view, src_view);
          }
        Kokkos::fence();

        SubdomainVectorType rhs_block;
        rhs_block.reinit(lifted_block, true);

        time.restart();
        this->subdomain_bddc_operator.vmult_plain_block(rhs_block, lifted_block, n_coarse_local);
        Kokkos::fence();
        setup_timings[1] += time.wall_time();

        rhs_block *= Number(-1);

        this->subdomain_bddc_operator.project_block(rhs_block, n_coarse_local);

        // Build a block-shaped V-cycle over the same per-level BDDC
        // matrices/transfers/smoothers the scalar subdomain_mg_
        // preconditioner above was built from (see the class-level
        // comment on BddcSmootherType), via the adapter classes in
        // portable_block_vcycle_adapters.h. Eigenvalue bounds are read
        // back from each level's already-initialized scalar smoother
        // rather than re-estimated: they depend only on the
        // operator/preconditioner pair, not on how many RHS columns are
        // solved at once.
        using BlockOperatorType = BlockBDDCOperatorAdapter<dim, Number>;
        using BlockTransferType = BlockTransferAdapter<dim, Number>;
        using BlockPreconditionerType =
          BlockProjectedDiagonalPreconditioner<BlockOperatorType, SubdomainVectorType>;
        using BlockSmootherType =
          ProjectedChebyshevSmoother<BlockOperatorType, BlockPreconditionerType, SubdomainVectorType>;

        const unsigned int minlevel = level_bddc_matrices.min_level();
        const unsigned int maxlevel = level_bddc_matrices.max_level();

        MGLevelObject<std::unique_ptr<BlockOperatorType>> block_matrices(minlevel, maxlevel);
        MGLevelObject<std::unique_ptr<BlockTransferType>> block_transfers(minlevel, maxlevel);
        MGLevelObject<BlockSmootherType>                  block_smoothers(minlevel, maxlevel);

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            const auto &concrete_matrix =
              static_cast<const SubdomainBDDCOperator<dim, Number> &>(*level_bddc_matrices[level]);

            block_matrices[level] =
              std::make_unique<BlockOperatorType>(concrete_matrix, n_coarse_local);

            if (level > minlevel)
              block_transfers[level] =
                std::make_unique<BlockTransferType>(*level_bddc_transfers[level], n_coarse_local);

            typename BlockSmootherType::AdditionalData smoother_data;
            const auto &scalar_data       = level_bddc_smoothers[level].get_additional_data();
            smoother_data.degree          = scalar_data.degree;
            smoother_data.max_eigenvalue  = scalar_data.max_eigenvalue;
            smoother_data.smoothing_range = scalar_data.smoothing_range;
            smoother_data.preconditioner  = std::make_shared<BlockPreconditionerType>(
              *block_matrices[level],
              level_bddc_matrices[level]->get_matrix_diagonal_inverse(),
              n_coarse_local);

            block_smoothers[level].initialize(*block_matrices[level], smoother_data);
          }

        SubdomainVCycleMultigrid<dim, Number, BlockOperatorType, BlockTransferType, BlockSmootherType>
          block_mg_preconditioner(block_matrices, block_transfers, block_smoothers);

        SubdomainVectorType correction_block;
        correction_block.reinit(rhs_block);

        // Same interface-derived-RHS/precision-floor reasoning as
        // vmult_fine_correction() above -- rhs_block is n_coarse_local
        // stacked lifted-constraint RHS's, same BC-only origin, so it's
        // exposed to the same shrinking-norm-under-refinement effect.
        ReductionControl                   solver_control(rhs_block.size(), 1e-15, 1e-12);
        SolverBlockCG<SubdomainVectorType> block_solver(solver_control);

        time.restart();
        block_solver.solve_block(*block_matrices[maxlevel],
                                 correction_block,
                                 rhs_block,
                                 block_mg_preconditioner,
                                 n_coarse_local);
        Kokkos::fence();
        setup_timings[2] += time.wall_time();

        // std::cout << "On subdomain " << this_subdomain << ", block coarse solve converged in "
        //           << solver_control.last_step() << " iterations." << std::endl;

        // max_subdomain_mg_iterations =
        //   std::max(max_subdomain_mg_iterations, static_cast<unsigned
        //   int>(solver_control.last_step()));

        // phi_j = lifted_constraints[j] + correction_j, unpacked straight
        // into coarse_basis_functions (matches the old loop's
        // `phi_j = lifted_constraints[j]; ...; phi_j.add(1., temp_subdomain_dst);`).
        for (unsigned int j = 0; j < n_coarse_local; ++j)
          {
            SubdomainVectorType &phi_j = coarse_basis_functions[j];

            DeviceVector<Number> lifted_view(lifted_block.get_values() + j * n_subdomain_dofs,
                                             n_subdomain_dofs);
            DeviceVector<Number> correction_view(correction_block.get_values() +
                                                   j * n_subdomain_dofs,
                                                 n_subdomain_dofs);
            DeviceVector<Number> phi_view(phi_j.get_values(), n_subdomain_dofs);

            Kokkos::parallel_for(
              "unpack_phi_j", n_subdomain_dofs, KOKKOS_LAMBDA(const int i) {
                phi_view(i) = lifted_view(i) + correction_view(i);
              });
          }
        Kokkos::fence();

        // Verification: recompute phi_j via the old sequential
        // vmult_fine_correction() loop the block path above replaces, and
        // diff against the block result. Kept permanently (not a one-shot
        // check to delete once confirmed) as a running correctness guard
        // on the block coarse-solve path, since compute_local_coarse_
        // matrix() only runs once per preconditioner setup -- the extra
        // n_coarse_local sequential solves cost comparatively little next
        // to that.
        //
        // Disabled (#if 0) for the Jupiter GPU perf run: it doubles the
        // coarse-solve cost (a full sequential re-solve alongside the
        // block one), which would confound a clean block-vs-sequential
        // timing comparison. Already confirmed at machine precision on
        // CPU -- re-enable once the GPU run itself needs a correctness
        // re-check.
#if 0
        {
          std::vector<SubdomainVectorType> phi_ref(n_coarse_local);
          SubdomainVectorType              rhs_ref, correction_ref;
          rhs_ref.reinit(temp_subdomain_src);
          correction_ref.reinit(temp_subdomain_dst);

          Number max_phi_diff = 0;

          for (unsigned int j = 0; j < n_coarse_local; ++j)
            {
              phi_ref[j] = lifted_constraints[j];

              this->subdomain_bddc_operator.vmult_plain(rhs_ref, phi_ref[j]);
              rhs_ref *= Number(-1);
              this->subdomain_bddc_operator.project(rhs_ref);

              correction_ref = 0;
              this->solve_fine_correction(correction_ref, rhs_ref);

              phi_ref[j].add(Number(1), correction_ref);

              SubdomainVectorType diff = phi_ref[j];
              diff -= coarse_basis_functions[j];
              max_phi_diff = std::max(max_phi_diff, diff.linfty_norm());
            }

          std::cout << "[block coarse-matrix verification] subdomain " << this_subdomain
                    << ": max |phi_block - phi_sequential| = " << max_phi_diff << std::endl;
        }
#endif
      }

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        SubdomainVectorType &phi_j = coarse_basis_functions[j];

        time.restart();
        this->subdomain_bddc_operator.vmult_plain(S_per_phi_j, phi_j);
        Kokkos::fence();
        setup_timings[1] += time.wall_time();

        time.restart();
        for (unsigned int k = 0; k < n_coarse_local; ++k)
          local_coarse_matrix(k, j) = lifted_constraints[k] * S_per_phi_j;
        Kokkos::fence();
        setup_timings[3] += time.wall_time();
      }
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::compute_coarse_matrix()
  {
    reset_setup_timings();

    const unsigned int n_coarse_global = this->n_global_coarse_dofs;
    const unsigned int n_coarse_local  = this->n_local_coarse_dofs;
    const auto        &local_to_global = this->coarse_dofs_local_to_global_vector_host;

    LAPACKFullMatrix<Number> local_coarse_matrix;
    this->compute_local_coarse_matrix(local_coarse_matrix);

    Kokkos::fence();

    // Flatten this rank's local_coarse_matrix entries as
    // (row*n_coarse_global + col, value) pairs -- only the n_coarse_local^2
    // entries this rank actually touches, not a dense n_coarse_global^2
    // array. All-gathering these small per-rank lists (rather than
    // Utilities::MPI::sum-ing a dense n_coarse_global^2 array, as before)
    // is the sparse analogue of the same assembly: every rank ends up with
    // every contribution and can build the same sparse matrix locally, but
    // the communication volume scales with the number of coarse dofs
    // actually touched (O(P * n_coarse_local^2)) rather than the square of
    // the global coarse space size.
    std::vector<unsigned int> own_flat_indices;
    std::vector<Number>       own_values;
    own_flat_indices.reserve(n_coarse_local * n_coarse_local);
    own_values.reserve(n_coarse_local * n_coarse_local);

    for (unsigned int i = 0; i < n_coarse_local; ++i)
      for (unsigned int j = 0; j < n_coarse_local; ++j)
        {
          own_flat_indices.push_back(local_to_global[i] * n_coarse_global + local_to_global[j]);
          own_values.push_back(local_coarse_matrix(i, j));
        }

    Timer                                        mpi_sum_time;
    const std::vector<std::vector<unsigned int>> all_flat_indices =
      Utilities::MPI::all_gather(this->subdomain_dof_handler->get_mpi_communicator(),
                                 own_flat_indices);
    const std::vector<std::vector<Number>> all_values =
      Utilities::MPI::all_gather(this->subdomain_dof_handler->get_mpi_communicator(), own_values);
    setup_timings[4] += mpi_sum_time.wall_time();

    DynamicSparsityPattern dsp(n_coarse_global, n_coarse_global);
    for (const auto &rank_indices : all_flat_indices)
      for (const auto &flat : rank_indices)
        dsp.add(flat / n_coarse_global, flat % n_coarse_global);

    this->coarse_sparsity_pattern.copy_from(dsp);
    this->coarse_matrix.reinit(this->coarse_sparsity_pattern);

    // SparseMatrix::add() accumulates rather than overwrites, so entries
    // contributed by multiple ranks (any pair of global coarse dofs shared
    // by more than one subdomain) sum correctly -- the same semantics the
    // old Utilities::MPI::sum() had.
    for (unsigned int r = 0; r < all_flat_indices.size(); ++r)
      for (unsigned int k = 0; k < all_flat_indices[r].size(); ++k)
        {
          const unsigned int flat = all_flat_indices[r][k];
          this->coarse_matrix.add(flat / n_coarse_global, flat % n_coarse_global, all_values[r][k]);
        }

#ifdef DEBUG
    {
      bool   is_spd     = true;
      Number min_eigenv = std::numeric_limits<Number>::max();

      // Diagonal-positivity is a cheap necessary (not sufficient) proxy for
      // SPD-ness, not an actual eigenvalue computation -- matches what this
      // check already did before (it called compute_eigenvalues() but then
      // read the matrix's own diagonal, not the eigenvalues, so dropping
      // that unused call here changes nothing observable).
      for (unsigned int i = 0; i < n_coarse_global; ++i)
        {
          const Number diag = this->coarse_matrix.diag_element(i);
          if (diag < min_eigenv)
            min_eigenv = diag;

          if (diag <= 1e-10)
            is_spd = false;
        }

      if (Utilities::MPI::this_mpi_process(this->subdomain_dof_handler->get_mpi_communicator()) ==
          0)
        {
          std::cout << "============================================" << std::endl;
          std::cout << "BDDC COARSE MATRIX CHARACTERISTICS:" << std::endl;
          std::cout << "Minimum diagonal entry: " << min_eigenv << std::endl;
          std::cout << "Is Strictly Positive Definite? " << (is_spd ? "YES" : "NO") << std::endl;
          std::cout << "============================================" << std::endl;

          AssertThrow(is_spd, ExcMessage("BDDC Global Coarse Matrix is not positive definite!"));
        }
    }
#endif

    Timer lu_time;
    this->coarse_matrix_solver.initialize(this->coarse_matrix);
    setup_timings[5] += lu_time.wall_time();
  }



  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::setup_primal_constraint_views()
  {
    const auto &dof_info          = this->subdomain_dof_handler->get_dof_info();
    const auto &local_constraints = dof_info.local_primal_constraints;

    std::vector<unsigned int> primal_constraint_dofs_interface_local_v;
    std::vector<unsigned int> primal_constraint_dofs_subdomain_v;
    std::vector<unsigned int> constraint_dofs_offsets_v;
    std::vector<unsigned int> corner_dofs_subdomain_v;

    coarse_dofs_local_to_global_vector_host.clear();

    std::vector<Number> coarse_weights_v;

    constraint_dofs_offsets_v.push_back(0);
    for (const auto &constraint : local_constraints)
      {
        if (bddc_variant == BDDCVariant::corner && constraint.type != PrimalConstraintType::Vertex)
          continue;

        if (bddc_variant == BDDCVariant::corner_edge &&
            constraint.type == PrimalConstraintType::Face)
          continue;

        if (constraint.type == PrimalConstraintType::Vertex)
          {
            AssertDimension(constraint.interface_partitioner_dofs_local.size(), 1);
            AssertDimension(constraint.local_subdomain_dofs.size(), 1);

            corner_dofs_subdomain_v.push_back(constraint.local_subdomain_dofs[0]);
          }

        primal_constraint_dofs_interface_local_v.insert(
          primal_constraint_dofs_interface_local_v.end(),
          constraint.interface_partitioner_dofs_local.begin(),
          constraint.interface_partitioner_dofs_local.end());

        primal_constraint_dofs_subdomain_v.insert(primal_constraint_dofs_subdomain_v.end(),
                                                  constraint.local_subdomain_dofs.begin(),
                                                  constraint.local_subdomain_dofs.end());


        constraint_dofs_offsets_v.push_back(primal_constraint_dofs_interface_local_v.size());

        coarse_dofs_local_to_global_vector_host.push_back(constraint.global_coarse_dof_index);

        Number weight = Number(1) / Number(constraint.interface_partitioner_dofs_local.size());

        AssertThrow(weight > 0, ExcInternalError());
        coarse_weights_v.push_back(weight);
      }

    AssertThrow(primal_constraint_dofs_interface_local_v.size() > 0, ExcInternalError());
    AssertThrow(primal_constraint_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(constraint_dofs_offsets_v.size() > 0, ExcInternalError());
    // AssertThrow(corner_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(coarse_dofs_local_to_global_vector_host.size() > 0, ExcInternalError());
    AssertThrow(coarse_weights_v.size() > 0, ExcInternalError());


    // Allocate and copy to Device Kokkos::Views
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_interface_local_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "primal_constraint_dofs_interface_local_host"),
      primal_constraint_dofs_interface_local_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_dofs_subdomain_host"),
      primal_constraint_dofs_subdomain_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_constraint_offsets_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_constraint_offsets_host"),
      constraint_dofs_offsets_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> coarse_dofs_local_to_global_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_dofs_local_to_global_host"),
      coarse_dofs_local_to_global_vector_host.size());

    Kokkos::View<unsigned int *, Kokkos::HostSpace> corner_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "corner_dofs_subdomain_host"),
      corner_dofs_subdomain_v.size());

    Kokkos::View<Number *, Kokkos::HostSpace> coarse_weights_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_weights_host"),
      coarse_weights_v.size());

    std::copy(primal_constraint_dofs_interface_local_v.begin(),
              primal_constraint_dofs_interface_local_v.end(),
              primal_constraint_dofs_interface_local_host.data());
    std::copy(primal_constraint_dofs_subdomain_v.begin(),
              primal_constraint_dofs_subdomain_v.end(),
              primal_constraint_dofs_subdomain_host.data());
    std::copy(constraint_dofs_offsets_v.begin(),
              constraint_dofs_offsets_v.end(),
              primal_constraint_constraint_offsets_host.data());
    std::copy(coarse_dofs_local_to_global_vector_host.begin(),
              coarse_dofs_local_to_global_vector_host.end(),
              coarse_dofs_local_to_global_host.data());
    std::copy(corner_dofs_subdomain_v.begin(),
              corner_dofs_subdomain_v.end(),
              corner_dofs_subdomain_host.data());
    std::copy(coarse_weights_v.begin(), coarse_weights_v.end(), coarse_weights_host.data());

    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

    this->primal_constraint_dofs_interface_local =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_interface_local_host);
    this->primal_constraint_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_subdomain_host);
    this->primal_constraint_offsets =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_constraint_offsets_host);
    this->coarse_dofs_local_to_global =
      Kokkos::create_mirror_view_and_copy(exec_space, coarse_dofs_local_to_global_host);
    this->corner_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, corner_dofs_subdomain_host);
    this->coarse_weights = Kokkos::create_mirror_view_and_copy(exec_space, coarse_weights_host);

    exec_space.fence();
  }


} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif