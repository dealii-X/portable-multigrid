#ifndef bk3_kokkos_kernels_custom_h
#define bk3_kokkos_kernels_custom_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

#include "kernels/portable_evaluation_kernels.h"
#include "kernels/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace BK3Custom
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

      constexpr int nq_total = Utilities::pow(nq, dim);

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 1 + dim;

      if (cell_range_ids.size() > 0)
        AssertDimension(cell_range_ids.size(), n_cells);

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
        const int ssize =
          nm * nq +                                    // shape values
          nq * nq +                                    // co-shape gradients
          n_scratch_arrays * nelmtPerBatch * nq_total; // values slot + dim gradients-pool slots

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            Number *s_values    = s_co_shape_gradients + nq * nq;
            Number *s_gradients = s_values + nelmtPerBatch * nq_total;

            const int quad_size_per_batch = nelmtPerBatch * nq_total;

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

                const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nm, nq, Number>
                  fe_eval(team_member,
                          s_shape_values,
                          s_co_shape_gradients,
                          c_nelmtPerBatch,
                          threadIdx,
                          blockSize);

                Custom::Parallel::read_dof_values<dim, nm>(team_member,
                                                           d_in,
                                                           dof_indices,
                                                           s_values,
                                                           eb,
                                                           nelmtPerBatch,
                                                           cell_range_ids,
                                                           c_nelmtPerBatch,
                                                           threadIdx,
                                                           blockSize);

                fe_eval.evaluate_values(s_values, s_values, s_gradients, quad_size_per_batch);

                fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(d_G,
                                                                         eb,
                                                                         nelmtPerBatch,
                                                                         cell_range_ids,
                                                                         s_values,
                                                                         s_gradients,
                                                                         quad_size_per_batch);

                fe_eval.integrate_gradients(s_gradients, quad_size_per_batch, s_values);

                fe_eval.integrate_values(s_values, s_values, s_gradients, quad_size_per_batch);

                Custom::Parallel::distribute_local_to_global<dim, nm>(team_member,
                                                                      s_values,
                                                                      d_out,
                                                                      dof_indices,
                                                                      eb,
                                                                      nelmtPerBatch,
                                                                      cell_range_ids,
                                                                      c_nelmtPerBatch,
                                                                      threadIdx,
                                                                      blockSize);

                eb += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3Custom

DEAL_II_NAMESPACE_CLOSE

#endif
