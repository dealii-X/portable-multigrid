#ifndef bk1_kokkos_kernels_h
#define bk1_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

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
    KokkosProlongationKernel(const DeviceView<Number>  d_shape_values,
                             const DeviceView<Number>  d_in,
                             DeviceView<Number>        d_out,
                             const DoFIndicesView      dof_indices_coarse,
                             const DoFIndicesView      dof_indices_fine,
                             const WeightsView<Number> weights,
                             const unsigned int        n_cells,
                             const unsigned int        n_threads    = numbers::invalid_unsigned_int,
                             const unsigned int n_threads_per_block = numbers::invalid_unsigned_int)

    {
      constexpr int nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr int nm_fine_total   = Utilities::pow(nm_fine, dim);

      const int nelmt = n_cells;

      const int numThreads =
        std::max(1,
                 ((n_threads == numbers::invalid_unsigned_int) ? nelmt * nm_fine_total / 2 :
                                                                 static_cast<int>(n_threads)));
      const int threadsPerBlock = std::max(1,
                                           (n_threads_per_block == numbers::invalid_unsigned_int) ?
                                             nm_fine_total :
                                             static_cast<int>(n_threads_per_block));

      const int numBlocks = std::max(1, numThreads / (std::min(nm_fine_total, threadsPerBlock)));

      {
        const unsigned int ssize = nm_coarse * nm_fine + // kernel matrix
                                   2 * nm_fine_total;    // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nm_fine_total;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            for (int tid = threadIdx; tid < nm_coarse * nm_fine; tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            int cell_index = team_member.league_rank();

            while (cell_index < nelmt)
              {
                // Gather coarse dof values
                {
                  for (int tid = threadIdx; tid < nm_coarse_total; tid += blockSize)
                    {
                      // Fetch the global DoF index
                      const unsigned int dof_index = dof_indices_coarse(tid, cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        s_wsp0[tid] = 0;
                      else
                        s_wsp0[tid] = d_in[dof_index];
                    }
                  team_member.team_barrier();
                }

                // direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < nm_fine * co_dimension_size; tid += blockSize)
                    {
                      if (dim == 2)
                        {
                          const int j = tid / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              tmp += s_shape_values[i * nm_fine + p] * s_wsp0[j * nm_coarse + i];
                            }

                          s_wsp1[j * nm_fine + p] = tmp;
                        }
                      else if (dim == 3)
                        {
                          const int k = tid / (nm_fine * nm_coarse);
                          const int j = (tid % (nm_fine * nm_coarse)) / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              tmp += s_shape_values[i * nm_fine + p] *
                                     s_wsp0[k * nm_coarse * nm_coarse + j * nm_coarse + i];
                            }

                          s_wsp1[k * nm_coarse * nm_fine + j * nm_fine + p] = tmp;
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 1
                {
                  constexpr int co_dimension_size = nm_fine * Utilities::pow(nm_coarse, dim - 2);

                  for (int tid = threadIdx; tid < nm_fine * co_dimension_size; tid += blockSize)
                    {
                      if (dim == 2)
                        {
                          const int q = tid / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;
                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              tmp += s_shape_values[j * nm_fine + q] * s_wsp1[j * nm_fine + p];
                            }
                          s_wsp0[q * nm_fine + p] = tmp;
                        }
                      else if (dim == 3)
                        {
                          const int k = tid / (nm_fine * nm_fine);
                          const int q = (tid % (nm_fine * nm_fine)) / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              tmp += s_shape_values[j * nm_fine + q] *
                                     s_wsp1[k * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          s_wsp0[k * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 2
                if (dim == 3)
                  {
                    for (int tid = threadIdx; tid < nm_fine * nm_fine * nm_fine; tid += blockSize)
                      {
                        const int r = tid / (nm_fine * nm_fine);
                        const int q = (tid % (nm_fine * nm_fine)) / nm_fine;
                        const int p = tid % nm_fine;

                        Number tmp = 0.;

                        for (int k = 0; k < nm_coarse; ++k)
                          {
                            tmp += s_shape_values[k * nm_fine + r] *
                                   s_wsp0[k * nm_fine * nm_fine + q * nm_fine + p];
                          }

                        s_wsp1[r * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                      }
                    team_member.team_barrier();
                  }

                // Apply weights and scatter fine dof values
                if (dim == 2)
                  {
                    for (int tid = threadIdx; tid < nm_fine_total; tid += blockSize)
                      {
                        // Find where this node lives in the global 'd_out'
                        // vector
                        const unsigned int dof_index = dof_indices_fine(tid, cell_index);

                        // apply weights
                        Number value_out = weights(tid, cell_index) * s_wsp0[tid];

                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        Kokkos::atomic_add(&d_out[dof_index], value_out);
                      }
                  }
                else if (dim == 3)
                  {
                    for (int tid = threadIdx; tid < nm_fine_total; tid += blockSize)
                      {
                        // Find where this node lives in the global 'd_out'
                        // vector
                        const unsigned int dof_index = dof_indices_fine(tid, cell_index);

                        // apply weights
                        Number value_out = weights(tid, cell_index) * s_wsp1[tid];

                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        Kokkos::atomic_add(&d_out[dof_index], value_out);
                      }
                  }

                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int dim, int nm_coarse, int nm_fine, typename Number>
    void
    KokkosRestrictionKernel(const DeviceView<Number>  d_shape_values,
                            const DeviceView<Number>  d_in,
                            DeviceView<Number>        d_out,
                            const DoFIndicesView      dof_indices_coarse,
                            const DoFIndicesView      dof_indices_fine,
                            const WeightsView<Number> weights,
                            const unsigned int        n_cells,
                            const unsigned int        n_threads    = numbers::invalid_unsigned_int,
                            const unsigned int n_threads_per_block = numbers::invalid_unsigned_int)

    {
      constexpr int nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr int nm_fine_total   = Utilities::pow(nm_fine, dim);

      const int nelmt = n_cells;

      const int numThreads =
        std::max(1,
                 ((n_threads == numbers::invalid_unsigned_int) ? nelmt * nm_fine_total / 2 :
                                                                 static_cast<int>(n_threads)));

      const int threadsPerBlock = std::max(1,
                                           (n_threads_per_block == numbers::invalid_unsigned_int) ?
                                             nm_fine_total :
                                             static_cast<int>(n_threads_per_block));

      const int numBlocks = std::max(1, numThreads / (std::min(nm_fine_total, threadsPerBlock)));

      {
        const unsigned int ssize = nm_coarse * nm_fine + // kernel matrix
                                   2 * nm_fine_total;    // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nm_fine_total;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            for (int tid = threadIdx; tid < nm_coarse * nm_fine; tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            int cell_index = team_member.league_rank();

            while (cell_index < nelmt)
              {
                // Gather fine dof values and apply weights
                {
                  for (int tid = threadIdx; tid < nm_fine_total; tid += blockSize)
                    {
                      // Fetch the global DoF index
                      const unsigned int dof_index = dof_indices_fine(tid, cell_index);

                      // read dof value and apply weights
                      s_wsp0[tid] = weights(tid, cell_index) * d_in[dof_index];
                    }
                  team_member.team_barrier();
                }

                // step-2 : direction 2
                if (dim == 3)
                  {
                    for (int tid = threadIdx; tid < nm_fine * nm_fine * nm_coarse; tid += blockSize)
                      {
                        const int k = tid / (nm_fine * nm_fine);
                        const int q = (tid % (nm_fine * nm_fine)) / nm_fine;
                        const int p = tid % nm_fine;

                        Number tmp = 0.;
                        for (int r = 0; r < nm_fine; ++r)
                          {
                            tmp += s_shape_values[k * nm_fine + r] *
                                   s_wsp0[r * nm_fine * nm_fine + q * nm_fine + p];
                          }

                        s_wsp1[k * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                      }
                    team_member.team_barrier();
                  }

                // step-3 : direction 1
                {
                  constexpr int co_dimension_size = nm_fine * Utilities::pow(nm_coarse, dim - 2);

                  for (int tid = threadIdx; tid < nm_coarse * co_dimension_size; tid += blockSize)
                    {
                      if (dim == 2)
                        {
                          const int j = tid / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              tmp += s_shape_values[j * nm_fine + q] * s_wsp0[q * nm_fine + p];
                            }
                          s_wsp1[j * nm_fine + p] = tmp;
                        }
                      else if (dim == 3)
                        {
                          const int k = tid / (nm_coarse * nm_fine);
                          const int j = (tid % (nm_coarse * nm_fine)) / nm_fine;
                          const int p = tid % nm_fine;

                          Number tmp = 0.;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              tmp += s_shape_values[j * nm_fine + q] *
                                     s_wsp1[k * nm_fine * nm_fine + q * nm_fine + p];
                            }
                          s_wsp0[k * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                        }
                    }
                  team_member.team_barrier();
                }

                // step-4 : direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < nm_coarse * co_dimension_size; tid += blockSize)
                    {
                      if (dim == 2)
                        {
                          const int j = tid / nm_coarse;
                          const int i = tid % nm_coarse;

                          Number tmp = 0.;

                          for (int p = 0; p < nm_fine; ++p)
                            {
                              tmp += s_shape_values[i * nm_fine + p] * s_wsp1[j * nm_fine + p];
                            }
                          s_wsp0[j * nm_coarse + i] = tmp;
                        }
                      else if (dim == 3)
                        {
                          const int i = tid / (nm_coarse * nm_coarse);
                          const int j = (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                          const int k = tid % nm_coarse;

                          Number tmp = 0.;

                          for (unsigned int p = 0; p < nm_fine; ++p)
                            {
                              tmp += s_shape_values[i * nm_fine + p] *
                                     s_wsp0[k * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          s_wsp1[k * nm_coarse * nm_coarse + j * nm_coarse + i] = tmp;
                        }
                    }
                  team_member.team_barrier();
                }

                // Scatter coarse values
                if (dim == 2)
                  {
                    for (int tid = threadIdx; tid < nm_coarse_total; tid += blockSize)
                      {
                        // Find where this node lives in the global 'd_out'
                        // vector
                        const unsigned int dof_index = dof_indices_coarse(tid, cell_index);

                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        if (dof_index != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&d_out[dof_index], s_wsp0[tid]);
                      }
                  }
                else if (dim == 3)
                  {
                    for (int tid = threadIdx; tid < nm_coarse_total; tid += blockSize)
                      {
                        // Find where this node lives in the global 'd_out'
                        // vector
                        const unsigned int dof_index = dof_indices_coarse(tid, cell_index);

                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        if (dof_index != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&d_out[dof_index], s_wsp1[tid]);
                      }
                  }

                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int dim, int nm_coarse, int nm_fine, typename Number>
    void
    KokkosProlongationBatchedKernel(
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

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nelmtPerBatch * nm_fine_total;

            Number reg[nm_fine];

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
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_id;

                      if (dim == 2)
                        {
                          const int i = tid % nm_coarse;

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              const int local_idx = j * nm_coarse + i;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_coarse(local_idx, global_cell_index);

                              if (dof_index == numbers::invalid_unsigned_int)
                                s_wsp0[batch_id * nm_coarse_total + local_idx] = 0;
                              else
                                s_wsp0[batch_id * nm_coarse_total + local_idx] = d_in[dof_index];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm_coarse;
                          const int i = tid % nm_coarse;

                          for (int k = 0; k < nm_coarse; ++k)
                            {
                              const int local_idx = k * nm_coarse * nm_coarse + j * nm_coarse + i;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_coarse(local_idx, global_cell_index);

                              if (dof_index == numbers::invalid_unsigned_int)
                                s_wsp0[batch_id * nm_coarse_total + local_idx] = 0;
                              else
                                s_wsp0[batch_id * nm_coarse_total + local_idx] = d_in[dof_index];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int j = tid % nm_coarse;

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              reg[i] = s_wsp0[batch_id * nm_coarse * nm_coarse + j * nm_coarse + i];
                            }

                          for (int p = 0; p < nm_fine; ++p)
                            {
                              Number tmp = 0.;
                              for (int i = 0; i < nm_coarse; ++i)
                                {
                                  tmp += s_shape_values[i * nm_fine + p] * reg[i];
                                }

                              s_wsp1[batch_id * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm_coarse;
                          const int j = tid % nm_coarse;

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              reg[i] = s_wsp0[batch_id * nm_coarse * nm_coarse * nm_coarse +
                                              k * nm_coarse * nm_coarse + j * nm_coarse + i];
                            }

                          for (int p = 0; p < nm_fine; ++p)
                            {
                              Number tmp = 0.;
                              for (int i = 0; i < nm_coarse; ++i)
                                {
                                  tmp += s_shape_values[i * nm_fine + p] * reg[i];
                                }

                              s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                                     k * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 1
                {
                  constexpr int co_dimension_size = nm_fine * Utilities::pow(nm_coarse, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int p = tid % nm_fine;

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              reg[j] = s_wsp1[batch_id * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              Number tmp = 0.;
                              for (int j = 0; j < nm_coarse; ++j)
                                {
                                  tmp += s_shape_values[j * nm_fine + q] * reg[j];
                                }

                              s_wsp0[batch_id * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm_fine;
                          const int p = tid % nm_fine;

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              reg[j] = s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                                              k * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              Number tmp = 0.;
                              for (int j = 0; j < nm_coarse; ++j)
                                {
                                  tmp += s_shape_values[j * nm_fine + q] * reg[j];
                                }

                              s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                                     k * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 2
                if (dim == 3)
                  {
                    constexpr int co_dimension_size = Utilities::pow(nm_fine, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int batch_id = tid / co_dimension_size;

                        const int q = (tid % co_dimension_size) / nm_fine;
                        const int p = tid % nm_fine;

                        for (int k = 0; k < nm_coarse; ++k)
                          {
                            reg[k] = s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                                            k * nm_fine * nm_fine + q * nm_fine + p];
                          }

                        for (int r = 0; r < nm_fine; ++r)
                          {
                            Number tmp = 0.;
                            for (int k = 0; k < nm_coarse; ++k)
                              {
                                tmp += s_shape_values[k * nm_fine + r] * reg[k];
                              }

                            s_wsp1[batch_id * nm_fine * nm_fine * nm_fine + r * nm_fine * nm_fine +
                                   q * nm_fine + p] = tmp;
                          }
                      }
                    team_member.team_barrier();
                  }

                // Apply weights and scatter fine dof values
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_fine, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_id;

                      if (dim == 2)
                        {
                          const int p = tid % nm_fine;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              const int local_idx = q * nm_fine + p;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_fine(local_idx, global_cell_index);

                              Number value_out = weights(local_idx, global_cell_index) *
                                                 s_wsp0[batch_id * nm_fine_total + local_idx];

                              // CRITICAL: Use atomic_add because elements share
                              // nodes!
                              Kokkos::atomic_add(&d_out[dof_index], value_out);
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / nm_fine;
                          const int p = tid % nm_fine;

                          for (int r = 0; r < nm_fine; ++r)
                            {
                              const int local_idx = r * nm_fine * nm_fine + q * nm_fine + p;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_fine(local_idx, global_cell_index);

                              Number value_out = weights(local_idx, global_cell_index) *
                                                 s_wsp1[batch_id * nm_fine_total + local_idx];

                              // CRITICAL: Use atomic_add because elements share
                              // nodes!
                              Kokkos::atomic_add(&d_out[dof_index], value_out);
                            }
                        }
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
    KokkosRestrictionBatchedKernel(const DeviceView<Number>  d_shape_values,
                                   const DeviceView<Number>  d_in,
                                   DeviceView<Number>        d_out,
                                   const DoFIndicesView      dof_indices_coarse,
                                   const DoFIndicesView      dof_indices_fine,
                                   const WeightsView<Number> weights,
                                   const unsigned int        n_cells,
                                   unsigned int n_blocks            = numbers::invalid_unsigned_int,
                                   unsigned int n_threads_per_block = numbers::invalid_unsigned_int)

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

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nelmtPerBatch * nm_fine_total;

            Number reg[nm_fine];

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
                  constexpr int co_dimension_size = Utilities::pow(nm_fine, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_id;

                      if (dim == 2)
                        {
                          const int p = tid % nm_fine;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              const int local_idx = q * nm_fine + p;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_fine(local_idx, global_cell_index);

                              s_wsp0[batch_id * nm_fine_total + local_idx] =
                                weights(local_idx, global_cell_index) * d_in[dof_index];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / nm_fine;
                          const int p = tid % nm_fine;

                          for (int r = 0; r < nm_fine; ++r)
                            {
                              const int local_idx = r * nm_fine * nm_fine + q * nm_fine + p;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_fine(local_idx, global_cell_index);

                              s_wsp1[batch_id * nm_fine_total + local_idx] =
                                weights(local_idx, global_cell_index) * d_in[dof_index];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                //  direction 2
                if (dim == 3)
                  {
                    constexpr int co_dimension_size = Utilities::pow(nm_fine, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int batch_id = tid / co_dimension_size;

                        const int q = (tid % co_dimension_size) / nm_fine;
                        const int p = tid % nm_fine;

                        for (int r = 0; r < nm_fine; ++r)
                          {
                            reg[r] = s_wsp1[batch_id * nm_fine * nm_fine * nm_fine +
                                            r * nm_fine * nm_fine + q * nm_fine + p];
                          }

                        for (int k = 0; k < nm_coarse; ++k)
                          {
                            Number tmp = 0.;
                            for (int r = 0; r < nm_fine; ++r)
                              {
                                tmp += s_shape_values[k * nm_fine + r] * reg[r];
                              }

                            s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                                   k * nm_fine * nm_fine + q * nm_fine + p] = tmp;
                          }
                      }
                    team_member.team_barrier();
                  }

                // direction 1
                {
                  constexpr int co_dimension_size = nm_fine * Utilities::pow(nm_coarse, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int p = tid % nm_fine;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              reg[q] = s_wsp0[batch_id * nm_fine * nm_fine + q * nm_fine + p];
                            }

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              Number tmp = 0.;
                              for (int q = 0; q < nm_fine; ++q)
                                {
                                  tmp += s_shape_values[j * nm_fine + q] * reg[q];
                                }

                              s_wsp1[batch_id * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm_fine;
                          const int p = tid % nm_fine;

                          for (int q = 0; q < nm_fine; ++q)
                            {
                              reg[q] = s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                                              k * nm_fine * nm_fine + q * nm_fine + p];
                            }

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              Number tmp = 0.;
                              for (int q = 0; q < nm_fine; ++q)
                                {
                                  tmp += s_shape_values[j * nm_fine + q] * reg[q];
                                }

                              s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                                     k * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int j = tid % nm_coarse;

                          for (int p = 0; p < nm_fine; ++p)
                            {
                              reg[p] = s_wsp1[batch_id * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              Number tmp = 0.;
                              for (int p = 0; p < nm_fine; ++p)
                                {
                                  tmp += s_shape_values[i * nm_fine + p] * reg[p];
                                }

                              s_wsp0[batch_id * nm_coarse * nm_coarse + j * nm_coarse + i] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm_coarse;
                          const int j = tid % nm_coarse;

                          for (int p = 0; p < nm_fine; ++p)
                            {
                              reg[p] = s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                                              k * nm_fine * nm_coarse + j * nm_fine + p];
                            }

                          for (int i = 0; i < nm_coarse; ++i)
                            {
                              Number tmp = 0.;
                              for (int p = 0; p < nm_fine; ++p)
                                {
                                  tmp += s_shape_values[i * nm_fine + p] * reg[p];
                                }

                              s_wsp0[batch_id * nm_coarse * nm_coarse * nm_coarse +
                                     k * nm_coarse * nm_coarse + j * nm_coarse + i] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // Scatter coarse dof values
                {
                  constexpr int co_dimension_size = Utilities::pow(nm_coarse, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int batch_id = tid / co_dimension_size;

                      const unsigned int global_cell_index =
                        cell_batch_index * nelmtPerBatch + batch_id;

                      if (dim == 2)
                        {
                          const int i = tid % nm_coarse;

                          for (int j = 0; j < nm_coarse; ++j)
                            {
                              const int local_idx = j * nm_coarse + i;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_coarse(local_idx, global_cell_index);

                              // CRITICAL: Use atomic_add because elements share
                              // nodes!
                              if (dof_index != numbers::invalid_unsigned_int)
                                Kokkos::atomic_add(&d_out[dof_index],
                                                   s_wsp0[batch_id * nm_coarse_total + local_idx]);
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm_coarse;
                          const int i = tid % nm_coarse;

                          for (int k = 0; k < nm_coarse; ++k)
                            {
                              const int local_idx = k * nm_coarse * nm_coarse + j * nm_coarse + i;

                              // Fetch the global DoF index
                              const unsigned int dof_index =
                                dof_indices_coarse(local_idx, global_cell_index);

                              // CRITICAL: Use atomic_add because elements share
                              // nodes!
                              if (dof_index != numbers::invalid_unsigned_int)
                                Kokkos::atomic_add(&d_out[dof_index],
                                                   s_wsp0[batch_id * nm_coarse_total + local_idx]);
                            }
                        }
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