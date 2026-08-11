#ifndef portable_block_group_mean_projection_h
#define portable_block_group_mean_projection_h

#include <deal.II/base/memory_space.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {
    /**
     * Block Pi: applies "for each of n_groups primal-constraint groups,
     * subtract the group's mean from its member dofs" independently to
     * each of n_rhs blocks of a block vector, laid out as n_rhs blocks of
     * dof_stride each (same convention as bk3_kokkos_kernel_block.h's
     * KokkosKernelBlock() and portable_solver_block_cg.h: block k
     * occupies [k*dof_stride, (k+1)*dof_stride)).
     *
     * offsets/member_dofs/weights are exactly what
     * SubdomainBDDCOperatorWrapper::project() and
     * BDDCPreconditioner::project_to_homogeneous_constraints_subdomain()
     * already build (primal_constraint_offsets /
     * primal_constraint_dofs_subdomain / coarse_weights) -- they don't
     * vary with the RHS (same subdomain, same primal-constraint
     * structure for every column), so unlike bk3_kokkos_kernel_block.h
     * there's no cell/shared-memory batching subtlety here: this is a
     * flat index space over (group, k) pairs from the start, no
     * per-team data reuse to worry about.
     */
    // offsets/member_dofs/weights are taken as deduced template types
    // (rather than fixed to Kokkos::View<const .../..., Default::kokkos_space>)
    // so callers can pass either a plain (non-const-element) View -- as
    // SubdomainBDDCOperatorWrapper's own primal_constraint_offsets etc.
    // are stored -- or an explicitly const one, without needing to
    // manually convert first: template argument deduction does not apply
    // Kokkos::View's own const-converting constructor, so a hard-coded
    // Kokkos::View<const unsigned int *, ...> parameter type would reject
    // a Kokkos::View<unsigned int *, ...> argument outright.
    template <typename Number, typename OffsetsView, typename MemberDofsView, typename WeightsView>
    void
    apply_block_group_mean_projection(Number             *block_vector,
                                      const OffsetsView    &offsets,
                                      const MemberDofsView &member_dofs,
                                      const WeightsView    &weights,
                                      const unsigned int    n_groups,
                                      const unsigned int    n_rhs,
                                      const unsigned int    dof_stride)
    {
      if (n_groups == 0 || n_rhs == 0)
        return;

      Kokkos::parallel_for(
        "apply_block_group_mean_projection",
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
          0, static_cast<std::size_t>(n_groups) * n_rhs),
        KOKKOS_LAMBDA(const std::size_t idx) {
          // k-major layout, matching bk3_kokkos_kernel_block.h's
          // global_cell_index = k * n_cells + cell convention.
          const unsigned int group_idx = idx % n_groups;
          const unsigned int k         = idx / n_groups;

          const unsigned int start = offsets(group_idx);
          const unsigned int end   = offsets(group_idx + 1);

          if (end > start)
            {
              const unsigned int rhs_offset = k * dof_stride;

              Number average = 0;
              for (unsigned int i = start; i < end; ++i)
                average += block_vector[rhs_offset + member_dofs(i)];
              average *= weights(group_idx);

              for (unsigned int i = start; i < end; ++i)
                block_vector[rhs_offset + member_dofs(i)] -= average;
            }
        });
      Kokkos::fence();
    }

  } // namespace internal
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
