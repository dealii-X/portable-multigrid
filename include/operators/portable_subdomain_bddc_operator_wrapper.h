#ifndef portable_subdomain_bddc_operator_wrapper_h
#define portable_subdomain_bddc_operator_wrapper_h

#include <deal.II/base/observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <algorithm>
#include <memory>
#include <unordered_set>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "kernels/portable_block_group_mean_projection.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"
#include "operators/portable_masked_dof_indices.h"
#include "operators/portable_subdomain_laplace_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  enum class BDDCVariant
  {
    corner,
    corner_edge,
    corner_edge_face
  };

  template <int dim, typename Number>
  class SubdomainBDDCOperator : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    SubdomainBDDCOperator(
      const SubdomainLaplaceOperatorBase<dim, Number> &dirichlet_operator,
      const BDDCVariant                                variant = BDDCVariant::corner_edge_face)
      : dirichlet_operator(&dirichlet_operator)
      , subdomain_dof_handler(&dirichlet_operator.get_subdomain_dof_handler())
      , n_subdomain_dofs(dirichlet_operator.get_subdomain_dof_handler().get_dof_handler().n_dofs())
      , interface_vector_size(dirichlet_operator.get_interface_dof_indices_subdomain().size())
      , interface_dof_indices_subdomain(dirichlet_operator.get_interface_dof_indices_subdomain())
      , inverse_diagonal_entries(
          std::make_shared<
            DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>())
      , primal_pinned_inverse_diagonal_entries(
          std::make_shared<
            DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>())
      , corner_pinned_inverse_diagonal_entries(
          std::make_shared<
            DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>())
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

      setup_primal_pinned_dof_indices();
      setup_corner_pinned_dof_indices();
    }

    void
    project(SubdomainVectorType &subdomain_vector) const override
    {
      DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(), n_subdomain_dofs);

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

    // Hard-zeros subdomain_vector at the primal-pinned dof set (physical
    // boundary UNION primal-constraint dofs) -- the precondition
    // vmult_primal_pinned()'s implicit identity block requires of any RHS
    // fed into a CG solve against it, analogous to project()'s "already in
    // V" precondition for the Pi-projected vmult(), but a hard zero (A_RR
    // pins strongly) rather than a per-group mean subtraction (Pi pins
    // weakly).
    void
    zero_primal_pinned_dofs(SubdomainVectorType &subdomain_vector) const
    {
      DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(), n_subdomain_dofs);

      const auto primal_pinned_dofs = this->primal_pinned_boundary_dof_indices;

      Kokkos::parallel_for(
        "zero_primal_pinned_dofs",
        primal_pinned_dofs.size(),
        KOKKOS_LAMBDA(const int i) { subdomain_vector_view(primal_pinned_dofs(i)) = Number(0); });
      Kokkos::fence();
    }

    /**
     * Block Pi: applies project() independently to each of n_rhs blocks
     * of block_vector, which must be sized n_rhs * n_subdomain_dofs and
     * laid out as n_rhs blocks of n_subdomain_dofs each (same convention
     * bk3_kokkos_kernel_block.h's KokkosKernelBlock() and SolverBlockCG
     * use). Does not touch project() itself -- see
     * apply_block_group_mean_projection()'s comment in
     * portable_block_group_mean_projection.h for why this doesn't need
     * the cell/shared-memory batching care the leaf matrix-free kernel
     * did (primal_constraint_offsets/_dofs_subdomain/coarse_weights are
     * the same for every RHS block, and this kernel has no per-cell
     * shared data to reuse in the first place).
     */
    void
    project_block(SubdomainVectorType &block_vector, const unsigned int n_rhs) const override
    {
      AssertDimension(block_vector.size(), static_cast<std::size_t>(n_rhs) * n_subdomain_dofs);

      internal::apply_block_group_mean_projection(block_vector.get_values(),
                                                   this->primal_constraint_offsets,
                                                   this->primal_constraint_dofs_subdomain,
                                                   this->coarse_weights,
                                                   this->n_local_coarse_dofs,
                                                   n_rhs,
                                                   this->n_subdomain_dofs);
    }

    void
    vmult_plain_block(SubdomainVectorType &dst, const SubdomainVectorType &src,
                      const unsigned int n_rhs) const override
    {
      dirichlet_operator->vmult_plain_block(dst, src, n_rhs);
    }

    // Ahat = Pi*A*Pi, the subtract-mean-projected apply. This is what
    // vmult() has always meant on this class -- deliberately UNCHANGED,
    // since level_subdomain_bddc_matrices' entire existing V-cycle wiring
    // in program.cc (ProjectedChebyshevSmoother::apply(), the eigenvalue
    // estimator, ...) reaches this through the SubdomainLaplaceOperatorBase
    // virtual interface, so redefining vmult() here would silently change
    // what every one of those existing call sites computes. The new
    // static-condensation apply below therefore gets its own explicit,
    // non-virtual name instead of overloading vmult()'s meaning.
    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);
      project(dst);
    }

    // A_RR: the plain local operator with primal-constraint dofs *also*
    // pinned to zero (Dirichlet-style), on top of the physical boundary --
    // the "remainder" operator the static-condensation fine correction
    // solves via MG-CG (see setup_primal_pinned_dof_indices() below). Not
    // part of the SubdomainLaplaceOperatorBase interface (no override) --
    // only ever called where the concrete SubdomainBDDCOperator type is
    // already in hand (the new, purpose-built A_RR MG hierarchy), so there
    // is no virtual-dispatch ambiguity with the vmult() above to worry
    // about.
    void
    vmult_primal_pinned(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
    {
      dirichlet_operator->vmult_masked(
        dst, src, primal_pinned_dof_indices_per_color, primal_pinned_boundary_dof_indices);
    }

    // A_RR restricted to CORNER dofs only (not the full active primal set):
    // physical boundary UNION corner_dofs_subdomain, leaving edge/face primal
    // dofs free/unpinned. Pinning corners alone is already enough to remove a
    // floating subdomain's constant null space (a nonzero constant can't
    // vanish at even one vertex), so this is unconditionally SPD regardless
    // of bddc_variant -- the classical BDDC construction (Klawonn/Widlund/
    // Dohrmann-style saddle-point fine-grid solve) builds on this: corners
    // hard-pinned via this operator, edges/faces enforced weakly afterwards
    // via a small dense Lagrange-multiplier system (see
    // BDDCPreconditioner::compute_local_edge_face_schur_complement()).
    void
    vmult_corner_pinned(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
    {
      dirichlet_operator->vmult_masked(
        dst, src, corner_pinned_dof_indices_per_color, corner_pinned_boundary_dof_indices);
    }

    // Block counterpart of vmult_corner_pinned() above: applies
    // A_RR_corner_pinned independently to each of n_rhs blocks -- lets
    // BDDCPreconditioner::compute_local_edge_face_schur_complement() batch
    // its n_edge_face_local sequential w_l solves into one SolverBlockCG
    // call, mirroring compute_local_coarse_matrix()'s existing Pi-path
    // block-CG pattern (BlockBDDCOperatorAdapter's vmult_plain_block()).
    void
    vmult_corner_pinned_block(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      const unsigned int                                                     n_rhs) const
    {
      dirichlet_operator->vmult_masked_block(
        dst, src, corner_pinned_dof_indices_per_color, corner_pinned_boundary_dof_indices, n_rhs);
    }

    // Hard-zeros subdomain_vector at the corner-pinned dof set (physical
    // boundary UNION corner_dofs_subdomain only, NOT edge/face primal dofs)
    // -- the precondition vmult_corner_pinned()'s identity block requires.
    // Narrower than zero_primal_pinned_dofs(): edge/face dof values must
    // stay intact here, since they're genuine input to the Lagrange system,
    // not pinned away.
    void
    zero_corner_pinned_dofs(SubdomainVectorType &subdomain_vector) const
    {
      DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(), n_subdomain_dofs);

      const auto corner_pinned_dofs = this->corner_pinned_boundary_dof_indices;

      Kokkos::parallel_for(
        "zero_corner_pinned_dofs",
        corner_pinned_dofs.size(),
        KOKKOS_LAMBDA(const int i) { subdomain_vector_view(corner_pinned_dofs(i)) = Number(0); });
      Kokkos::fence();
    }

    void
    vmult_masked(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
        &copy_through_dof_indices) const override
    {
      dirichlet_operator->vmult_masked(dst, src, dof_indices_per_color, copy_through_dof_indices);
    }

    void
    vmult_masked_block(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
                        &copy_through_dof_indices,
      const unsigned int n_rhs) const override
    {
      dirichlet_operator->vmult_masked_block(
        dst, src, dof_indices_per_color, copy_through_dof_indices, n_rhs);
    }

    void
    vmult_plain(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);
    }

    void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override
    {
      (void)dst;
      (void)src;
      (void)ghost_exchange_on;
      (void)computation_on;

      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_interface_cell_range(dst, src);
    }

    void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_neumann(dst, src);
    }


    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      this->vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override
    {
      dirichlet_operator->initialize_dof_vector(vec);
    }

    void
    compute_diagonal() override
    {
      inverse_diagonal_entries->reinit(
        dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

      // Same values everywhere except diag=1 at primal-pinned dofs (see
      // primal_pinned_inverse_diagonal_entries's class-level comment for
      // why) -- start from a copy of the already-inverted Neumann diagonal
      // (diag=1 at physical boundary already) and overwrite just the
      // primal-pinned entries directly with 1.0 (no need to re-invert:
      // 1/1 == 1).
      primal_pinned_inverse_diagonal_entries->reinit(
        dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

      Number *raw_diagonal = primal_pinned_inverse_diagonal_entries->get_vector().get_values();
      const auto primal_pinned_dofs = this->primal_pinned_boundary_dof_indices;

      Kokkos::parallel_for(
        "primal_pinned_diagonal_identity_block",
        primal_pinned_dofs.size(),
        KOKKOS_LAMBDA(const int i) { raw_diagonal[primal_pinned_dofs(i)] = Number(1.); });
      Kokkos::fence();

      // Same identity-block convention as primal_pinned_inverse_diagonal_entries
      // above, but for the corner-only mask.
      corner_pinned_inverse_diagonal_entries->reinit(
        dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

      Number *raw_diagonal_corner = corner_pinned_inverse_diagonal_entries->get_vector().get_values();
      const auto corner_pinned_dofs = this->corner_pinned_boundary_dof_indices;

      Kokkos::parallel_for(
        "corner_pinned_diagonal_identity_block",
        corner_pinned_dofs.size(),
        KOKKOS_LAMBDA(const int i) { raw_diagonal_corner[corner_pinned_dofs(i)] = Number(1.); });
      Kokkos::fence();
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override
    {
      return this->inverse_diagonal_entries;
    }

    // The primal-pinned-aware diagonal vmult_primal_pinned()'s own smoother
    // needs -- see primal_pinned_inverse_diagonal_entries's class-level
    // comment. Not part of the polymorphic interface (same reasoning as
    // vmult_primal_pinned() itself): only SubdomainBDDCOperatorARRAdapter
    // calls this, and it already holds the concrete type.
    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_primal_pinned() const
    {
      return this->primal_pinned_inverse_diagonal_entries;
    }

    // Same role as get_matrix_diagonal_inverse_primal_pinned(), but for
    // vmult_corner_pinned()'s smoother (diag=1 at corner-pinned dofs only).
    // Only SubdomainBDDCOperatorCornerPinnedAdapter calls this.
    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_corner_pinned() const
    {
      return this->corner_pinned_inverse_diagonal_entries;
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const override
    {
      return dirichlet_operator->get_matrix_diagonal_inverse_neumann();
    }

    types::global_dof_index
    m() const override
    {
      return dirichlet_operator->m();
    }

    types::global_dof_index
    n() const override
    {
      return dirichlet_operator->n();
    }

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      (void)col;
      Assert(row == col, ExcNotImplemented());

      Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
             ExcNotInitialized());

      return 1.0 / (*inverse_diagonal_entries)(row, row);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const override
    {
      return dirichlet_operator->get_matrix_free();
    }

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override
    {
      return dirichlet_operator->get_vector_partitioner();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_interface_dof_indices_subdomain();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_physical_boundary_dof_indices_subdomain();
    }

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override
    {
      return dirichlet_operator->get_subdomain_dof_handler();
    }

    // The dof-index mask vmult_primal_pinned() uses internally (physical
    // boundary UNION primal-constraint dofs) -- exposed so callers building
    // the A_RR MG hierarchy's own transfer objects (which need the same
    // "additionally pinned" set to correctly re-zero those dofs after
    // interpolation, see GeometricTransfer/PolynomialTransfer::
    // reinit_primal_pinned()) don't need to duplicate this computation.
    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_primal_pinned_dof_indices_subdomain() const
    {
      return primal_pinned_boundary_dof_indices;
    }

    // Same role as get_primal_pinned_dof_indices_subdomain(), but for the
    // corner-only mask vmult_corner_pinned() uses (physical boundary UNION
    // corner_dofs_subdomain only) -- exposed for the corner-pinned MG
    // hierarchy's own transfer objects.
    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_corner_pinned_dof_indices_subdomain() const
    {
      return corner_pinned_boundary_dof_indices;
    }


  private:
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> dirichlet_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    BDDCVariant bddc_variant;

    unsigned int       n_global_coarse_dofs;
    unsigned int       n_local_coarse_dofs;
    const unsigned int n_subdomain_dofs;
    const unsigned int interface_vector_size;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    // Separate from inverse_diagonal_entries above: that one is the plain
    // Neumann diagonal (diag=1 at physical boundary only), correct for the
    // Pi-projected vmult()'s own ProjectedDiagonalPreconditioner, which
    // re-zeros primal dofs via project() regardless of what raw diagonal
    // value sits there. vmult_primal_pinned()'s smoother has no such
    // projection step, so its diagonal needs diag=1 at primal-pinned dofs
    // *itself* -- otherwise dealii::PreconditionChebyshev's own internal
    // eigenvalue estimator (which seeds a generic probe vector, not
    // confined to zero at those indices, unlike the projected path) would
    // read whatever generic stiffness-diagonal value sits there, rather
    // than the "identity block" convention constrained dofs are supposed
    // to present -- same reasoning as why physical/interface-constrained
    // dofs already get diag=1 in SubdomainLaplaceOperator::compute_diagonal().
    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      primal_pinned_inverse_diagonal_entries;

    // Same role as primal_pinned_inverse_diagonal_entries, for the
    // corner-only mask.
    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      corner_pinned_inverse_diagonal_entries;

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

    // A_RR's own dof-index mask: physical boundary dofs (already excluded
    // by dirichlet_operator itself) UNION primal_constraint_dofs_subdomain
    // (excluded here too, unlike the plain/Neumann operator). Built once at
    // construction by setup_primal_pinned_dof_indices(), consumed by this
    // class's own vmult() via dirichlet_operator->vmult_masked().
    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      primal_pinned_dof_indices_per_color;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_pinned_boundary_dof_indices;

    // Same role as primal_pinned_dof_indices_per_color/_boundary_dof_indices
    // above, but for the corner-only mask (physical UNION corner_dofs_subdomain
    // only). Built by setup_corner_pinned_dof_indices(), consumed by
    // vmult_corner_pinned().
    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      corner_pinned_dof_indices_per_color;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      corner_pinned_boundary_dof_indices;

    void
    setup_primal_pinned_dof_indices()
    {
      const auto physical_boundary_dofs_device =
        dirichlet_operator->get_physical_boundary_dof_indices_subdomain();
      auto physical_boundary_dofs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_boundary_dofs_device);

      auto primal_dofs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->primal_constraint_dofs_subdomain);

      std::unordered_set<unsigned int> excluded_dofs;
      excluded_dofs.reserve(physical_boundary_dofs_host.extent(0) + primal_dofs_host.extent(0));

      for (unsigned int i = 0; i < physical_boundary_dofs_host.extent(0); ++i)
        excluded_dofs.insert(physical_boundary_dofs_host(i));
      for (unsigned int i = 0; i < primal_dofs_host.extent(0); ++i)
        excluded_dofs.insert(primal_dofs_host(i));

      primal_pinned_dof_indices_per_color =
        internal::build_masked_dof_indices_per_color<dim, Number>(
          dirichlet_operator->get_matrix_free(),
          this->subdomain_dof_handler->get_dof_handler(),
          [&excluded_dofs](unsigned int dof) { return excluded_dofs.count(dof) > 0; });

      std::vector<unsigned int> excluded_dofs_v(excluded_dofs.begin(), excluded_dofs.end());
      std::sort(excluded_dofs_v.begin(), excluded_dofs_v.end());

      Kokkos::View<unsigned int *, Kokkos::HostSpace> excluded_dofs_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_pinned_boundary_dof_indices_host"),
        excluded_dofs_v.size());
      std::copy(excluded_dofs_v.begin(), excluded_dofs_v.end(), excluded_dofs_host.data());

      dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
      this->primal_pinned_boundary_dof_indices =
        Kokkos::create_mirror_view_and_copy(exec_space, excluded_dofs_host);
      exec_space.fence();
    }

    // Same construction as setup_primal_pinned_dof_indices() above, but
    // excluding only corner_dofs_subdomain (not the full active primal set)
    // -- see vmult_corner_pinned()'s class-level comment for why.
    void
    setup_corner_pinned_dof_indices()
    {
      const auto physical_boundary_dofs_device =
        dirichlet_operator->get_physical_boundary_dof_indices_subdomain();
      auto physical_boundary_dofs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physical_boundary_dofs_device);

      auto corner_dofs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->corner_dofs_subdomain);

      std::unordered_set<unsigned int> excluded_dofs;
      excluded_dofs.reserve(physical_boundary_dofs_host.extent(0) + corner_dofs_host.extent(0));

      for (unsigned int i = 0; i < physical_boundary_dofs_host.extent(0); ++i)
        excluded_dofs.insert(physical_boundary_dofs_host(i));
      for (unsigned int i = 0; i < corner_dofs_host.extent(0); ++i)
        excluded_dofs.insert(corner_dofs_host(i));

      corner_pinned_dof_indices_per_color =
        internal::build_masked_dof_indices_per_color<dim, Number>(
          dirichlet_operator->get_matrix_free(),
          this->subdomain_dof_handler->get_dof_handler(),
          [&excluded_dofs](unsigned int dof) { return excluded_dofs.count(dof) > 0; });

      std::vector<unsigned int> excluded_dofs_v(excluded_dofs.begin(), excluded_dofs.end());
      std::sort(excluded_dofs_v.begin(), excluded_dofs_v.end());

      Kokkos::View<unsigned int *, Kokkos::HostSpace> excluded_dofs_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "corner_pinned_boundary_dof_indices_host"),
        excluded_dofs_v.size());
      std::copy(excluded_dofs_v.begin(), excluded_dofs_v.end(), excluded_dofs_host.data());

      dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
      this->corner_pinned_boundary_dof_indices =
        Kokkos::create_mirror_view_and_copy(exec_space, excluded_dofs_host);
      exec_space.fence();
    }

    void
    setup_primal_constraint_views()
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
          if (bddc_variant == BDDCVariant::corner &&
              constraint.type != PrimalConstraintType::Vertex)
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
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "primal_constraint_constraint_offsets_host"),
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
        Kokkos::create_mirror_view_and_copy(exec_space,
                                            primal_constraint_dofs_interface_local_host);
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
  };

  /**
   * Presents SubdomainBDDCOperator::vmult_primal_pinned() (A_RR) as vmult()
   * -- the exact name PreconditionChebyshev/SolverCG/the MG transfer/
   * V-cycle machinery all expect -- without touching SubdomainBDDCOperator::
   * vmult() itself, which stays the Pi-projected apply the *existing* fine
   * correction's V-cycle wiring already reaches through the same
   * SubdomainLaplaceOperatorBase virtual interface (see the comment on
   * SubdomainBDDCOperator::vmult() above for why that couldn't just be
   * redefined in place).
   *
   * Dof-vector initialization, m()/n()/el(), and the underlying MatrixFree
   * are identical regardless of which vmult() variant is active, so those
   * are plain forwards to the wrapped operator; vmult()/vmult_plain()/
   * Tvmult() are redirected to the primal-pinned apply. The diagonal is
   * NOT a plain forward, though: get_matrix_diagonal_inverse() here reads
   * SubdomainBDDCOperator::get_matrix_diagonal_inverse_primal_pinned(), a
   * second diagonal with diag=1 forced at primal-pinned dofs -- the
   * Pi-projected path's diagonal doesn't need that (project() re-zeros
   * those entries regardless of the raw diagonal value there), but this
   * path's smoother has no such projection step, and dealii::
   * PreconditionChebyshev's internal eigenvalue estimator seeds a generic
   * probe vector that is NOT confined to zero at those indices, so the
   * "identity block" convention has to be enforced by the diagonal itself.
   * compute_diagonal() is intentionally DEAL_II_NOT_IMPLEMENTED() (same
   * convention SubdomainNeumannOperatorWrapper uses): the wrapped
   * SubdomainBDDCOperator's own compute_diagonal() must already have been
   * called externally (as the existing setup already does) before this
   * adapter is used.
   */
  template <int dim, typename Number>
  class SubdomainBDDCOperatorARRAdapter : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    SubdomainBDDCOperatorARRAdapter(const SubdomainBDDCOperator<dim, Number> &op)
      : op(&op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      op->vmult_primal_pinned(dst, src);
    }

    void
    vmult_plain(VectorType &dst, const VectorType &src) const override
    {
      op->vmult_primal_pinned(dst, src);
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const override
    {
      this->vmult(dst, src);
    }

    // A_RR has no primal-constraint projection of its own -- pinning is
    // baked into vmult_primal_pinned()'s dof mask, not a separate step.
    void
    project(VectorType &vec) const override
    {
      (void)vec;
    }

    void
    vmult_plain_block(VectorType &dst, const VectorType &src, const unsigned int n_rhs) const override
    {
      (void)dst;
      (void)src;
      (void)n_rhs;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    project_block(VectorType &vec, const unsigned int n_rhs) const override
    {
      (void)vec;
      (void)n_rhs;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_bk3(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(VectorType &dst,
               const VectorType &src,
               const bool        ghost_exchange_on,
               const bool        computation_on) const override
    {
      (void)dst;
      (void)src;
      (void)ghost_exchange_on;
      (void)computation_on;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_interface_cell_range(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_neumann(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_masked(
      VectorType       &dst,
      const VectorType &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
        &copy_through_dof_indices) const override
    {
      op->vmult_masked(dst, src, dof_indices_per_color, copy_through_dof_indices);
    }

    // Never used in block form (this class was only ever used for scalar
    // CG, see the class-level comment) -- stub matches this class's other
    // _block overrides.
    void
    vmult_masked_block(
      VectorType       &dst,
      const VectorType &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
                        &copy_through_dof_indices,
      const unsigned int n_rhs) const override
    {
      (void)dst;
      (void)src;
      (void)dof_indices_per_color;
      (void)copy_through_dof_indices;
      (void)n_rhs;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    initialize_dof_vector(VectorType &vec) const override
    {
      op->initialize_dof_vector(vec);
    }

    void
    compute_diagonal() override
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse() const override
    {
      return op->get_matrix_diagonal_inverse_primal_pinned();
    }

    std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse_neumann() const override
    {
      return op->get_matrix_diagonal_inverse_neumann();
    }

    types::global_dof_index
    m() const override
    {
      return op->m();
    }

    types::global_dof_index
    n() const override
    {
      return op->n();
    }

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      return op->el(row, col);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const override
    {
      return op->get_matrix_free();
    }

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override
    {
      return op->get_vector_partitioner();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override
    {
      return op->get_interface_dof_indices_subdomain();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override
    {
      return op->get_physical_boundary_dof_indices_subdomain();
    }

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override
    {
      return op->get_subdomain_dof_handler();
    }

  private:
    ObserverPointer<const SubdomainBDDCOperator<dim, Number>> op;
  };

  /**
   * Same role as SubdomainBDDCOperatorARRAdapter, but presents
   * SubdomainBDDCOperator::vmult_corner_pinned() (corners hard-pinned only,
   * edges/faces left free) as vmult() instead -- the operator the
   * corner-pinned MG hierarchy's smoother/transfer/V-cycle machinery is
   * built against, and that BDDCPreconditioner::
   * compute_local_edge_face_schur_complement()/vmult_fine_correction_
   * static_condensation() solve against via MG-preconditioned CG to realize
   * A_RR^{-1} in the classical corner-pin + edge/face-Lagrange-multiplier
   * BDDC construction. See vmult_corner_pinned()'s class-level comment for
   * the math.
   */
  template <int dim, typename Number>
  class SubdomainBDDCOperatorCornerPinnedAdapter : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    SubdomainBDDCOperatorCornerPinnedAdapter(const SubdomainBDDCOperator<dim, Number> &op)
      : op(&op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      op->vmult_corner_pinned(dst, src);
    }

    void
    vmult_plain(VectorType &dst, const VectorType &src) const override
    {
      op->vmult_corner_pinned(dst, src);
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const override
    {
      this->vmult(dst, src);
    }

    // Corner-pinned A_RR has no primal-constraint projection of its own --
    // pinning is baked into vmult_corner_pinned()'s dof mask.
    void
    project(VectorType &vec) const override
    {
      (void)vec;
    }

    // Block counterpart of vmult_plain()/vmult() above -- delegates to the
    // concrete operator's vmult_corner_pinned_block(), letting a caller
    // that only has this scalar-looking SubdomainLaplaceOperatorBase
    // interface in hand still drive a block-batched corner-pinned solve
    // (see BlockCornerPinnedOperatorAdapter in
    // portable_block_vcycle_adapters.h).
    void
    vmult_plain_block(VectorType &dst, const VectorType &src, const unsigned int n_rhs) const override
    {
      op->vmult_corner_pinned_block(dst, src, n_rhs);
    }

    // Corner-pinned A_RR has no primal-constraint projection of its own,
    // in block form either -- same reasoning as project() above.
    void
    project_block(VectorType &vec, const unsigned int n_rhs) const override
    {
      (void)vec;
      (void)n_rhs;
    }

    void
    vmult_bk3(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(VectorType &dst,
               const VectorType &src,
               const bool        ghost_exchange_on,
               const bool        computation_on) const override
    {
      (void)dst;
      (void)src;
      (void)ghost_exchange_on;
      (void)computation_on;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_interface_cell_range(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_neumann(VectorType &dst, const VectorType &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_masked(
      VectorType       &dst,
      const VectorType &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
        &copy_through_dof_indices) const override
    {
      op->vmult_masked(dst, src, dof_indices_per_color, copy_through_dof_indices);
    }

    void
    vmult_masked_block(
      VectorType       &dst,
      const VectorType &src,
      const std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        &dof_indices_per_color,
      const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
                        &copy_through_dof_indices,
      const unsigned int n_rhs) const override
    {
      op->vmult_masked_block(dst, src, dof_indices_per_color, copy_through_dof_indices, n_rhs);
    }

    void
    initialize_dof_vector(VectorType &vec) const override
    {
      op->initialize_dof_vector(vec);
    }

    void
    compute_diagonal() override
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse() const override
    {
      return op->get_matrix_diagonal_inverse_corner_pinned();
    }

    std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse_neumann() const override
    {
      return op->get_matrix_diagonal_inverse_neumann();
    }

    types::global_dof_index
    m() const override
    {
      return op->m();
    }

    types::global_dof_index
    n() const override
    {
      return op->n();
    }

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      return op->el(row, col);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const override
    {
      return op->get_matrix_free();
    }

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override
    {
      return op->get_vector_partitioner();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override
    {
      return op->get_interface_dof_indices_subdomain();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override
    {
      return op->get_physical_boundary_dof_indices_subdomain();
    }

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override
    {
      return op->get_subdomain_dof_handler();
    }

  private:
    ObserverPointer<const SubdomainBDDCOperator<dim, Number>> op;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif