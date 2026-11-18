#ifndef bk1_kokkos_kernels_h
#define bk1_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

#include "matrix_free/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace BK1
{
  namespace Parallel
  {

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <typename Number>
    using WeightsView = Kokkos::View<Number **, MemorySpace::Default::kokkos_space>;


    template <int dim, int nm_coarse, int nm_fine, typename Number>
    void
    KokkosProlongationBatchedKernelAbstracted(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      const unsigned int        n_blocks            = numbers::invalid_unsigned_int,
      const unsigned int        n_threads_per_block = numbers::invalid_unsigned_int)

    {
      if (n_cells == 0)
        return;

      constexpr int nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr int nm_fine_total   = Utilities::pow(nm_fine, dim);

      constexpr int shmemPerBlock = 10800;

      const int nelmt = n_cells;

      int nelmtPerBatch = (shmemPerBlock / (2 * nm_fine_total * sizeof(Number)));

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock = std::max(1,
                                           ((n_threads_per_block == numbers::invalid_unsigned_int) ?
                                              Utilities::pow(nm_fine, dim - 1) * nelmtPerBatch :
                                              static_cast<int>(n_threads_per_block)));

      {
        const unsigned int ssize = nm_coarse * nm_fine +              // kernel matrix
                                   2 * nelmtPerBatch * nm_fine_total; // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_values_coarse = s_shape_values + nm_coarse * nm_fine;
            Number *s_values_fine   = s_values_coarse + nelmtPerBatch * nm_fine_total;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            for (int tid = threadIdx; tid < nm_coarse * nm_fine; tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            int cell_batch_index = team_member.league_rank();

            while (cell_batch_index < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch =
                  (cell_batch_index * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                    (nelmt - cell_batch_index * nelmtPerBatch) :
                    nelmtPerBatch;

                // Gather coarse dof values
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_coarse_total;
                       tid += blockSize)
                    {
                      const int batch_idx = tid / nm_coarse_total;
                      const int local_idx = tid % nm_coarse_total;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_idx;

                      const unsigned int dof_index_coarse =
                        dof_indices_coarse(local_idx, global_cell_index);

                      if (dof_index_coarse != numbers::invalid_unsigned_int)
                        s_values_coarse[tid] = d_in[dof_index_coarse];
                      else
                        s_values_coarse[tid] = 0.;
                    }
                  team_member.team_barrier();
                }

                // Apply prolongation kernel
                {
                  Custom::Parallel::EvaluatorTensorProduct<
                    Custom::Parallel::EvaluatorVariant::evaluate_general,
                    dim,
                    nm_coarse,
                    nm_fine,
                    Number>
                    evaluator(team_member,
                              s_shape_values,
                              nullptr, // no gradients
                              nullptr, // no co-gradients
                              nullptr, // no temp
                              c_nelmtPerBatch,
                              threadIdx,
                              blockSize);

                  if constexpr (dim == 1)
                    {
                      evaluator.template values<0, true, false>(s_values_coarse, s_values_fine);
                    }
                  else if constexpr (dim == 2)
                    {
                      evaluator.template values<0, true, false>(s_values_coarse, s_values_fine);
                      evaluator.template values<1, true, false>(s_values_fine, s_values_coarse);
                    }
                  else // dim == 3
                    {
                      evaluator.template values<0, true, false>(s_values_coarse, s_values_fine);
                      evaluator.template values<1, true, false>(s_values_fine, s_values_coarse);
                      evaluator.template values<2, true, false>(s_values_coarse, s_values_fine);
                    }
                }
                // Apply weights and scatter fine dof values
                {
                  Number *s_result;
                  if constexpr (dim == 2)
                    s_result = s_values_coarse;
                  else
                    s_result = s_values_fine;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_fine_total; tid += blockSize)
                    {
                      const int batch_idx = tid / nm_fine_total;
                      const int local_idx = tid % nm_fine_total;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_idx;

                      const unsigned int dof_index_fine =
                        dof_indices_fine(local_idx, global_cell_index);

                      const Number weight = weights(local_idx, global_cell_index);

                      Kokkos::atomic_add(&d_out[dof_index_fine], weight * s_result[tid]);
                    }
                  team_member.team_barrier();
                }

                cell_batch_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int dim, int nm_coarse, int nm_fine, typename Number>
    void
    KokkosRestrictionBatchedKernelAbstracted(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      const unsigned int        n_blocks            = numbers::invalid_unsigned_int,
      const unsigned int        n_threads_per_block = numbers::invalid_unsigned_int)

    {
      if (n_cells == 0)
        return;

      constexpr int nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr int nm_fine_total   = Utilities::pow(nm_fine, dim);

      constexpr int shmemPerBlock = 10800;

      const int nelmt = n_cells;

      int nelmtPerBatch = (shmemPerBlock / (2 * nm_fine_total * sizeof(Number)));

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock = std::max(1,
                                           ((n_threads_per_block == numbers::invalid_unsigned_int) ?
                                              Utilities::pow(nm_fine, dim - 1) * nelmtPerBatch :
                                              static_cast<int>(n_threads_per_block)));

      {
        const unsigned int ssize = nm_coarse * nm_fine +              // kernel matrix
                                   2 * nelmtPerBatch * nm_fine_total; // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_values_fine   = s_shape_values + nm_coarse * nm_fine;
            Number *s_values_coarse = s_values_fine + nelmtPerBatch * nm_fine_total;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            for (int tid = threadIdx; tid < nm_coarse * nm_fine; tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            int cell_batch_index = team_member.league_rank();

            while (cell_batch_index < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch =
                  (cell_batch_index * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                    (nelmt - cell_batch_index * nelmtPerBatch) :
                    nelmtPerBatch;

                // Gather fine dof values and apply weights
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_fine_total; tid += blockSize)
                    {
                      const int batch_idx = tid / nm_fine_total;
                      const int local_idx = tid % nm_fine_total;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_idx;

                      const unsigned int dof_index_fine =
                        dof_indices_fine(local_idx, global_cell_index);

                      s_values_fine[tid] =
                        weights(local_idx, global_cell_index) * d_in[dof_index_fine];
                    }
                  team_member.team_barrier();
                }

                // apply restriction kernel per dimension
                {
                  Custom::Parallel::EvaluatorTensorProduct<
                    Custom::Parallel::EvaluatorVariant::evaluate_general,
                    dim,
                    nm_coarse,
                    nm_fine,
                    Number>
                    evaluator(team_member,
                              s_shape_values,
                              nullptr, // no gradients
                              nullptr, // no co-gradients
                              nullptr, // no temp
                              c_nelmtPerBatch,
                              threadIdx,
                              blockSize);

                  if constexpr (dim == 1)
                    {
                      evaluator.template values<0, false, false>(s_values_fine, s_values_coarse);
                    }
                  else if constexpr (dim == 2)
                    {
                      evaluator.template values<1, false, false>(s_values_fine, s_values_coarse);
                      evaluator.template values<0, false, false>(s_values_coarse, s_values_fine);
                    }
                  else // dim == 3
                    {
                      evaluator.template values<2, false, false>(s_values_fine, s_values_coarse);
                      evaluator.template values<1, false, false>(s_values_coarse, s_values_fine);
                      evaluator.template values<0, false, false>(s_values_fine, s_values_coarse);
                    }
                }

                {
                  Number *s_result;
                  if constexpr (dim == 2)
                    s_result = s_values_fine;
                  else
                    s_result = s_values_coarse;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_coarse_total;
                       tid += blockSize)
                    {
                      const int batch_idx = tid / nm_coarse_total;
                      const int local_idx = tid % nm_coarse_total;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_idx;

                      const unsigned int dof_index_coarse =
                        dof_indices_coarse(local_idx, global_cell_index);

                      // CRITICAL: Use atomic_add because elements share
                      // nodes!
                      if (dof_index_coarse != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&d_out[dof_index_coarse], s_result[tid]);
                    }
                  team_member.team_barrier();
                }

                cell_batch_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK1

DEAL_II_NAMESPACE_CLOSE

#endif