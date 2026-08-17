#ifndef bk3_kokkos_kernel_common_h
#define bk3_kokkos_kernel_common_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

#include "kernels/common.h"

// Structural twin of BK3::Parallel::KokkosKernel (bk3_kokkos_kernel.h),
// composed from the generic building blocks in kernels/common.h instead of
// ten hand-written steps. Kept in its own file/namespace, alongside the
// original, so the two can be run side by side and diffed for correctness
// before anything switches over to this version -- see the discussion in
// the unify-kernels session about factoring the sum-factorization logic
// shared across include/kernels/ into a small number of generic,
// GPU-batched primitives.
//
// Shared memory is organized as two pools instead of BK3's four named,
// individually-aliased buffers: a single "values" slot (nelmtPerBatch *
// nq^dim elements) and a "gradients" pool of dim consecutive slots of the
// same size. The dof->quad interpolation sweep (steps 2-4) ping-pongs
// through the gradients pool's slots as scratch and lands its result in the
// values slot; the fused collocated-gradient step (5-6) reads/writes the
// values slot in place, using the gradients pool for its own (differently
// purposed) directional-derivative intermediates; the quad->dof integration
// sweep (7-9) ping-pongs back out through the gradients pool. Total
// footprint is (1 + dim) slots -- the same 4 slots BK3 uses for dim == 3,
// and one fewer (3) for dim == 2, since BK3's own 4-slot budget is sized
// for the dim == 3 case regardless of dim.
//
// Steps 1 (copy-in) and 10 (scatter) keep BK3::Parallel::KokkosKernel's
// logic verbatim, just reading/writing whichever pool slot the surrounding
// sequence puts the live data in.
DEAL_II_NAMESPACE_OPEN

namespace BK3Common
{
  namespace Parallel
  {
    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename Number>
    void
    KokkosKernel(const DeviceView<Number> d_shape_values,
                 const DeviceView<Number> d_co_shape_gradients,
                 const DeviceView<Number> d_G,
                 const DeviceView<Number> d_in,
                 DeviceView<Number>       d_out,
                 const DoFIndicesView     dof_indices,
                 const unsigned int       n_cells,
                 const unsigned int       n_blocks          = numbers::invalid_unsigned_int,
                 const unsigned int       threads_per_block = numbers::invalid_unsigned_int,
                 const CellRangeIdView    cell_range_ids    = CellRangeIdView())
    {
      if (n_cells == 0)
        return;

      static_assert(dim == 2 || dim == 3, "dim must be 2 or 3");

      constexpr int nq_total                   = Utilities::pow(nq, dim);
      constexpr int nm_total                   = Utilities::pow(nm, dim);
      constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      // 1 values slot + dim gradients-pool slots (see file comment).
      constexpr int n_scratch_arrays = 1 + dim;

      if (cell_range_ids.size() > 0)
        AssertDimension(cell_range_ids.size(), n_cells);

      // evaluate_and_multiply_tensor() below indexes d_G by
      // eb * nelmtPerBatch + e directly, not remapped through
      // cell_range_ids the way dof_indices is in steps 1/10 -- see its doc
      // comment in kernels/common.h. No current caller passes a non-empty
      // cell_range_ids to this kernel; this catches it loudly rather than
      // silently reading the wrong cell's geometric factors if that ever
      // changes without also revisiting this.
      AssertThrow(cell_range_ids.size() == 0,
                 ExcMessage("BK3Common::Parallel::KokkosKernel's collocated-gradient step does not "
                            "support a non-empty cell_range_ids."));

      const int nelmt = n_cells;

      const int nelmtPerBatch =
        std::max(1,
                 static_cast<int>(shmemPerBlock / (n_scratch_arrays * nq_total) / sizeof(Number)));

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 1) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));

      {
        const int ssize = nm * nq +                                    // shape values
                          nq * nq +                                    // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch * nq_total; // values + gradients pools

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            // Single work pool: offset 0 is the "values" slot, offsets
            // slot, 2*slot, (3*slot for dim == 3) are the "gradients" pool
            // (see file comment for how the two are used in sequence).
            Number *s_work = s_co_shape_gradients + nq * nq;

            const int slot = nelmtPerBatch * nq_total;

            const int off_values = 0;
            const int off_g0     = slot;
            const int off_g1     = 2 * slot;
            const int off_g2     = 3 * slot; // dim == 3 only

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            // element batch iteration
            int eb = team_member.league_rank();

            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                              (nelmt - eb * nelmtPerBatch) :
                                              nelmtPerBatch;

                // step-1 : Copy from in to the scratch values (lands in
                // the gradients pool's first slot -- see steps 2-4 below)
                {
                  Number *s_wsp0 = s_work + off_g0;

                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      unsigned int global_cell_index = eb * nelmtPerBatch + e;

                      if (cell_range_ids.size() > 0)
                        global_cell_index = cell_range_ids(global_cell_index);

                      if (dim == 2)
                        {
                          const int i = tid % nm;

                          for (int j = 0; j < nm; ++j)
                            {
                              const int          local_idx = j * nm + i;
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index == numbers::invalid_unsigned_int)
                                s_wsp0[shared_idx] = 0;
                              else
                                s_wsp0[shared_idx] = d_in[dof_index];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm;
                          const int i = tid % nm;

                          for (int k = 0; k < nm; ++k)
                            {
                              const int          local_idx = k * nm * nm + j * nm + i;
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index == numbers::invalid_unsigned_int)
                                s_wsp0[shared_idx] = 0;
                              else
                                s_wsp0[shared_idx] = d_in[dof_index];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // steps 2-4: interpolate dof -> quad, one direction at a
                // time, ping-ponging through the gradients pool's slots and
                // landing the result in the values slot. Each call ends
                // with its own team_barrier().
                if constexpr (dim == 2)
                  {
                    Common::Parallel::apply_matrix_vector_product<dim, 0, nm, nq, true, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g0,
                      off_g1,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 1, nm, nq, true, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g1,
                      off_values,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);
                  }
                else if constexpr (dim == 3)
                  {
                    Common::Parallel::apply_matrix_vector_product<dim, 0, nm, nq, true, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g0,
                      off_g1,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 1, nm, nq, true, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g1,
                      off_g2,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 2, nm, nq, true, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g2,
                      off_values,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);
                  }

                // steps 5-6: evaluate collocated gradients + apply the
                // geometric-factor tensor, then integrate back. Reads and
                // writes the values slot in place; uses the gradients pool
                // (now free) for its own directional-derivative
                // intermediates. Each call ends with its own
                // team_barrier().
                //
                // evaluate_and_multiply_tensor() takes a flat, already-
                // resolved offset into d_G rather than doing a
                // cell_range_ids lookup itself (see its doc comment in
                // kernels/common.h) -- unlike dof_indices in steps 1/10
                // above, d_G is NOT remapped through cell_range_ids here
                // (enforced by the AssertThrow near the top of this
                // function).
                {
                  const int g_offset =
                    static_cast<int>(eb * nelmtPerBatch) * symmetric_tensor_dimension * nq_total;
                  const int gradient_slot_stride = slot;

                  Common::Parallel::evaluate_and_multiply_tensor<dim, nq>(team_member,
                                                                          s_co_shape_gradients,
                                                                          d_G,
                                                                          g_offset,
                                                                          s_work,
                                                                          off_values,
                                                                          off_g0,
                                                                          gradient_slot_stride,
                                                                          c_nelmtPerBatch,
                                                                          threadIdx,
                                                                          blockSize);

                  Common::Parallel::integrate<dim, nq>(team_member,
                                                       s_co_shape_gradients,
                                                       s_work,
                                                       off_g0,
                                                       gradient_slot_stride,
                                                       off_values,
                                                       c_nelmtPerBatch,
                                                       threadIdx,
                                                       blockSize);
                }

                // steps 7-9: integrate quad -> dof, one direction at a
                // time, in reverse order, ping-ponging back out through the
                // gradients pool. Final result lands in off_g2 (dim == 3)
                // or off_g1 (dim == 2), read by step-10 below.
                int result_offset;
                if constexpr (dim == 2)
                  {
                    Common::Parallel::apply_matrix_vector_product<dim, 1, nm, nq, false, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_values,
                      off_g0,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 0, nm, nq, false, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g0,
                      off_g1,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    result_offset = off_g1;
                  }
                else if constexpr (dim == 3)
                  {
                    Common::Parallel::apply_matrix_vector_product<dim, 2, nm, nq, false, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_values,
                      off_g0,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 1, nm, nq, false, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g0,
                      off_g1,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    Common::Parallel::apply_matrix_vector_product<dim, 0, nm, nq, false, false>(
                      team_member,
                      s_shape_values,
                      s_work,
                      off_g1,
                      off_g2,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);

                    result_offset = off_g2;
                  }

                // step-10 : Copy the final result back to global out vector
                {
                  Number *s_result = s_work + result_offset;

                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      unsigned int global_cell_index = eb * nelmtPerBatch + e;

                      if (cell_range_ids.size() > 0)
                        global_cell_index = cell_range_ids(global_cell_index);

                      if (dim == 2)
                        {
                          const int i = tid % nm;

                          for (int j = 0; j < nm; ++j)
                            {
                              const int          local_idx = j * nm + i;
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                Kokkos::atomic_add(&d_out[dof_index], s_result[shared_idx]);
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm;
                          const int i = tid % nm;

                          for (int k = 0; k < nm; ++k)
                            {
                              const int          local_idx = k * nm * nm + j * nm + i;
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                Kokkos::atomic_add(&d_out[dof_index], s_result[shared_idx]);
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                eb += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3Common

DEAL_II_NAMESPACE_CLOSE

#endif
