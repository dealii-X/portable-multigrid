#ifndef bk3_kokkos_kernel_h
#define bk3_kokkos_kernel_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
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
      constexpr int nm_total = Utilities::pow(nm, dim);

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 4;

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
        const int ssize = nm * nq + // shape values
                          nq * nq + // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total; // working scratch arrays: s_wsp0, s_wsp1, rqr, rqs, rqt

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];
            Number r_r[nq];

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            Number *s_wsp0 = s_co_shape_gradients + nq * nq;
            Number *s_wsp1 = s_wsp0 + nelmtPerBatch * nq_total;

            Number *s_rqr = s_wsp1 + nelmtPerBatch * nq_total;
            Number *s_rqs = s_rqr + nelmtPerBatch * nq_total;
            Number *s_rqt = s_wsp0;

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

                {
                  // step-1 : Copy from in to the scratch values
                  {
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
                                // Calculate the flat local index within the 3D
                                // element
                                const int local_idx = j * nm + i;

                                // Fetch the global DoF index
                                const unsigned int dof_index =
                                  dof_indices(local_idx, global_cell_index);

                                // The index in the batched shared memory array
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
                                // Calculate the flat local index within the 3D
                                // element
                                const int local_idx = k * nm * nm + j * nm + i;

                                // Fetch the global DoF index
                                const unsigned int dof_index =
                                  dof_indices(local_idx, global_cell_index);

                                // The index in the batched shared memory array
                                const int shared_idx = e * nm_total + local_idx;

                                if (dof_index == numbers::invalid_unsigned_int)
                                  s_wsp0[shared_idx] = 0;
                                else
                                  s_wsp0[shared_idx] = d_in[dof_index];
                              }
                          }
                      }
                  }
                  team_member.team_barrier();
                }

                // step-2 : direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int j = tid % nm;

                          for (int i = 0; i < nm; ++i)
                            {
                              r_p[i] = s_wsp0[e * nm * nm + j * nm + i];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0.0;

                              for (int i = 0; i < nm; ++i)
                                {
                                  tmp += s_shape_values[i * nq + p] * r_p[i];
                                }

                              s_wsp1[e * nq * nm + j * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm;
                          const int j = tid % nm;

                          for (int i = 0; i < nm; ++i)
                            {
                              r_p[i] = s_wsp0[e * nm * nm * nm + k * nm * nm + j * nm + i];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0.0;

                              for (int i = 0; i < nm; ++i)
                                {
                                  tmp += s_shape_values[i * nq + p] * r_p[i];
                                }

                              s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-3 : direction 1
                {
                  constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int p = tid % nq;

                          for (int j = 0; j < nm; ++j)
                            {
                              r_p[j] = s_wsp1[e * nq * nm + j * nq + p];
                            }

                          for (int q = 0; q < nq; ++q)
                            {
                              Number tmp = 0.0;

                              for (int j = 0; j < nm; ++j)
                                {
                                  tmp += s_shape_values[j * nq + q] * r_p[j];
                                }

                              s_wsp0[e * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int j = 0; j < nm; ++j)
                            {
                              r_p[j] = s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p];
                            }

                          for (int q = 0; q < nq; ++q)
                            {
                              Number tmp = 0.0;

                              for (int j = 0; j < nm; ++j)
                                {
                                  tmp += s_shape_values[j * nq + q] * r_p[j];
                                }

                              s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-4 : direction 2
                if (dim == 3)
                  {
                    constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        int e = tid / co_dimension_size;

                        int q = (tid % co_dimension_size) / nq;
                        int p = tid % nq;

                        for (int k = 0; k < nm; ++k)
                          {
                            r_p[k] = s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p];
                          }
                        for (int r = 0; r < nq; ++r)
                          {
                            Number tmp = 0.0;

                            for (int k = 0; k < nm; ++k)
                              {
                                tmp += s_shape_values[k * nq + r] * r_p[k];
                              }

                            s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                          }
                      }
                    team_member.team_barrier();
                  }

                // step-5: evaluate gradients and apply geometric factors
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      int e = tid / (co_dimension_size);

                      unsigned int global_cell_index = eb * nelmtPerBatch + e;

                      if (cell_range_ids.size() > 0)
                        global_cell_index = cell_range_ids(global_cell_index);

                      if (dim == 2)
                        {
                          const int p = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = s_co_shape_gradients[n * nq + p];
                              r_q[n] = s_wsp0[e * nq * nq + n * nq + p];
                            }

                          Number Grr, Grs, Gss;
                          Number qr, qs;

                          for (int q = 0; q < nq; ++q)
                            {
                              qr = 0;
                              qs = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        0 * nq_total + q * nq + p];

                              Grs = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        1 * nq_total + q * nq + p];

                              Gss = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        2 * nq_total + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; n++)
                                {
                                  qr += r_p[n] * s_wsp0[e * nq * nq + q * nq + n];
                                  qs += s_co_shape_gradients[n * nq + q] * r_q[n];
                                }

                              // Apply chain rule
                              s_rqr[e * nq * nq + q * nq + p] = Grr * qr + Grs * qs;

                              s_rqs[e * nq * nq + q * nq + p] = Grs * qr + Gss * qs;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = tid % (co_dimension_size) / nq;
                          const int p = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = s_co_shape_gradients[n * nq + p];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                              r_r[n] = s_wsp1[e * nq * nq * nq + n * nq * nq + q * nq + p];
                            }

                          Number Grr, Grs, Grt, Gss, Gst, Gtt;
                          Number qr, qs, qt;

                          for (int r = 0; r < nq; ++r)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        0 * nq_total + r * nq * nq + q * nq + p];

                              Grs = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        1 * nq_total + r * nq * nq + q * nq + p];

                              Grt = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        2 * nq_total + r * nq * nq + q * nq + p];

                              Gss = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        3 * nq_total + r * nq * nq + q * nq + p];

                              Gst = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        4 * nq_total + r * nq * nq + q * nq + p];

                              Gtt = d_G[global_cell_index * symmetric_tensor_dimension * nq_total +
                                        5 * nq_total + r * nq * nq + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; n++)
                                {
                                  qr +=
                                    r_p[n] * s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                                  qs +=
                                    r_q[n] * s_wsp1[e * nq * nq * nq + r * nq * nq + n * nq + p];
                                  qt += s_co_shape_gradients[n * nq + r] * r_r[n];
                                }

                              // Apply chain rule
                              s_rqr[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grr * qr + Grs * qs + Grt * qt;

                              s_rqs[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grs * qr + Gss * qs + Gst * qt;

                              s_rqt[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grt * qr + Gst * qs + Gtt * qt;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-6: integrate gradients
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int p = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = s_co_shape_gradients[p * nq + n];
                              r_q[n] = s_rqs[e * nq * nq + n * nq + p];
                            }

                          for (int q = 0; q < nq; ++q)
                            {
                              Number tmp0 = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp0 += s_rqr[e * nq * nq + q * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += r_q[n] * s_co_shape_gradients[q * nq + n];

                              s_wsp0[e * nq * nq + q * nq + p] = tmp0;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = s_co_shape_gradients[p * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];

                              r_r[n] = s_rqt[e * nq * nq * nq + n * nq * nq + q * nq + p];
                            }

                          for (int r = 0; r < nq; ++r)
                            {
                              Number tmp0 = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp0 += s_rqr[e * nq * nq * nq + r * nq * nq + q * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += s_rqs[e * nq * nq * nq + r * nq * nq + n * nq + p] * r_q[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += r_r[n] * s_co_shape_gradients[r * nq + n];

                              s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp0;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                /*
                Interpolate to GLL nodes
                */

                // step-7 : direction 2
                if (dim == 3)
                  {
                    constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        const int q = (tid % co_dimension_size) / nq;
                        const int p = tid % nq;

                        for (int r = 0; r < nq; ++r)
                          {
                            r_p[r] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p];
                          }

                        for (int k = 0; k < nm; ++k)
                          {
                            Number tmp = 0.0;

                            for (int r = 0; r < nq; ++r)
                              {
                                tmp += s_shape_values[k * nq + r] * r_p[r];
                              }

                            s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                          }
                      }
                    team_member.team_barrier();
                  }

                // step-8 : direction 1
                {
                  constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;
                      if (dim == 2)
                        {
                          const int p = tid % nq;

                          for (int q = 0; q < nq; ++q)
                            {
                              r_p[q] = s_wsp0[e * nq * nq + q * nq + p];
                            }

                          for (int j = 0; j < nm; ++j)
                            {
                              Number tmp = 0.0;

                              for (int q = 0; q < nq; ++q)
                                {
                                  tmp += s_shape_values[j * nq + q] * r_p[q];
                                }
                              s_wsp1[e * nq * nm + j * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int q = 0; q < nq; ++q)
                            {
                              r_p[q] = s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p];
                            }

                          for (int j = 0; j < nm; ++j)
                            {
                              Number tmp = 0.0;

                              for (int q = 0; q < nq; ++q)
                                {
                                  tmp += s_shape_values[j * nq + q] * r_p[q];
                                }
                              s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-9 : direction 0
                {
                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int j = tid % nm;

                          for (int p = 0; p < nq; ++p)
                            {
                              r_p[p] = s_wsp1[e * nq * nm + j * nq + p];
                            }

                          for (int i = 0; i < nm; ++i)
                            {
                              Number tmp = 0.0;
                              for (int p = 0; p < nq; ++p)
                                {
                                  tmp += s_shape_values[i * nq + p] * r_p[p];
                                }
                              s_wsp0[e * nm * nm + j * nm + i] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int k = (tid % co_dimension_size) / nm;
                          const int j = tid % nm;

                          for (int p = 0; p < nq; ++p)
                            {
                              r_p[p] = s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p];
                            }

                          for (int i = 0; i < nm; ++i)
                            {
                              Number tmp = 0.0;
                              for (int p = 0; p < nq; ++p)
                                {
                                  tmp += s_shape_values[i * nq + p] * r_p[p];
                                }
                              s_wsp0[e * nm * nm * nm + k * nm * nm + j * nm + i] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-10 : Copy wsp0 (result) back to global out vector
                {
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
                              const int local_idx = j * nm + i;

                              // Find where this node lives in the global 'd_out'
                              // vector
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              // The index in our batched shared memory result
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // CRITICAL: Use atomic_add because elements share
                                  // nodes!
                                  Kokkos::atomic_add(&d_out[dof_index], s_wsp0[shared_idx]);
                                }
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm;
                          const int i = tid % nm;

                          for (int k = 0; k < nm; ++k)
                            {
                              const int local_idx = k * nm * nm + j * nm + i;

                              // Find where this node lives in the global 'd_out'
                              // vector
                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              // The index in our batched shared memory result
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // CRITICAL: Use atomic_add because elements share
                                  // nodes!
                                  Kokkos::atomic_add(&d_out[dof_index], s_wsp0[shared_idx]);
                                }
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

    template <int dim, int n_local_dofs_1d, int n_q_points_1d, typename Number>
    void
    KokkosKernel_1D_Block(const DeviceView<Number> shape_values_device,
                          const DeviceView<Number> co_shape_gradients_device,
                          const DeviceView<Number> G_device,
                          const DeviceView<Number> in_device,
                          DeviceView<Number>       out_device,
                          const DoFIndicesView     dof_indices,
                          const unsigned int       n_cells,
                          unsigned int             numThreads      = numbers::invalid_unsigned_int,
                          unsigned int             threadsPerBlock = numbers::invalid_unsigned_int,
                          const CellRangeIdView    cell_range_ids  = CellRangeIdView())

    {
      if (n_cells == 0)
        return;

      constexpr unsigned n_q_points_total   = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned n_local_dofs_total = Utilities::pow(n_local_dofs_1d, dim);

      const int nelmt = n_cells;

      if (numThreads == numbers::invalid_unsigned_int)
        numThreads = nelmt * n_q_points_total / 2;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = n_q_points_total;

      unsigned int numBlocks = numThreads / (std::min(n_q_points_total, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;

      {
        const unsigned int scratch_pad_size = 5 * n_q_points_total; // working scratch arrays:
                                                                    // s_wsp0, s_wsp1, rqr,rqq, rqt

        unsigned int ssize = n_local_dofs_1d * n_q_points_1d + // shape values
                             n_q_points_1d * n_q_points_1d +   // co-shape gradients
                             scratch_pad_size;                 // at most 5 tmp arrays

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *shape_values_scratch = scratch;
            Number *co_shape_gradients_scratch =
              shape_values_scratch + n_q_points_1d * n_local_dofs_1d;

            Number *s_wsp0 = co_shape_gradients_scratch + n_q_points_1d * n_q_points_1d;
            Number *s_wsp1 = s_wsp0 + n_q_points_total;

            Number *rqr = s_wsp1 + n_q_points_total;
            Number *rqs = rqr + n_q_points_total;
            Number *rqt = rqs + n_q_points_total;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();


            // copy to shared memory
            {
              for (unsigned int tid = threadIdx; tid < n_local_dofs_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  shape_values_scratch[tid] = shape_values_device[tid];
                }

              for (unsigned int tid = threadIdx; tid < n_q_points_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  co_shape_gradients_scratch[tid] = co_shape_gradients_device[tid];
                }
            }

            // std::cout << "BK3 kernel shape_values:";
            // for (unsigned int i = 0; i < n_local_dofs_1d * n_q_points_1d;
            // ++i)
            //   std::cout << shape_values_device(i) << " ";
            // std::cout << std::endl << std::endl;

            // std::cout << "BK3 kernel shape_grads:";
            // for (unsigned int i = 0; i < n_local_dofs_1d * n_q_points_1d;
            // ++i)
            //   std::cout << co_shape_gradients_device(i) << " ";
            // std::cout << std::endl << std::endl;


            // std::cout << "BK3 kernel G_device:";
            // for (unsigned int i = 0; i < G_device.size(); ++i)
            //   std::cout << G_device(i) << " ";
            // std::cout << std::endl << std::endl;


            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            unsigned int cell_index = team_member.league_rank();

            while (cell_index < nelmt)
              {
                unsigned int cell_id = cell_index;

                if (cell_range_ids.size() > 0)
                  cell_id = cell_range_ids(cell_index);

                // std::cout << "cell_id: " << cell_id << std::endl;

                team_member.team_barrier();
                {
                  // step-1 : Copy from in to the scratch values
                  for (unsigned int tid = threadIdx; tid < n_local_dofs_total; tid += blockSize)
                    {
                      const unsigned int dof_index = dof_indices(tid, cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        s_wsp0[tid] = 0;
                      else
                        s_wsp0[tid] = in_device[dof_index];
                    }
                }
                team_member.team_barrier();

                // std::cout << "BK3 kernel cell_id = " << cell_id << ": ";
                // for (unsigned int i = 0; i < n_local_dofs_total; ++i)
                //   std::cout << s_wsp0(i) << " ";
                // std::cout << std::endl << std::endl;


                if constexpr (dim == 3)
                  {
                    // step-2 : direction 0
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_local_dofs_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                        const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) / n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        Number sum = 0.0;
                        for (unsigned int i = 0; i < n_local_dofs_1d; ++i)
                          {
                            sum += s_wsp0[k * n_local_dofs_1d * n_local_dofs_1d +
                                          j * n_local_dofs_1d + i] *
                                   shape_values_scratch[i * n_q_points_1d + p];
                          }
                        s_wsp1[k * n_q_points_1d * n_local_dofs_1d + j * n_q_points_1d + p] = sum;
                      }
                    team_member.team_barrier();

                    // step-3 : direction 1
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int i = tid / (n_q_points_1d * n_local_dofs_1d);
                        const int q = (tid % (n_q_points_1d * n_local_dofs_1d)) / n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        Number sum = 0.0;
                        for (unsigned int j = 0; j < n_local_dofs_1d; j++)
                          {
                            sum +=
                              s_wsp1[k * n_q_points_1d * n_local_dofs_1d + j * n_q_points_1d + i] *
                              shape_values_scratch[j * n_q_points_1d + q];
                          }

                        s_wsp0[k * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + i] = sum;
                      }
                    team_member.team_barrier();

                    // step-4 : direction 2
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                         tid += blockSize)
                      {
                        const int i = tid / (n_q_points_1d * n_q_points_1d);
                        const int j = (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                        const int r = tid % n_q_points_1d;

                        Number sum = 0.0;
                        for (unsigned int k = 0; k < n_local_dofs_1d; ++k)
                          {
                            sum +=
                              s_wsp0[k * n_q_points_1d * n_q_points_1d + j * n_q_points_1d + i] *
                              shape_values_scratch[k * n_q_points_1d + r];
                          }
                        s_wsp1[r * n_q_points_1d * n_q_points_1d + j * n_q_points_1d + i] = sum;
                      }
                    team_member.team_barrier();
                  }

                // std::cout << "BK3 kernel cell_id = " << cell_id << ": ";
                // for (unsigned int i = 0; i < n_local_dofs_total; ++i)
                //   std::cout << s_wsp1(i) << " ";
                // std::cout << std::endl << std::endl;

                // Geometric vals
                Number        Grr, Grs, Grt, Gss, Gst, Gtt;
                Number        qr, qs, qt;
                constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

                // std::cout << "BK3 kernel cell_id = " << cell_id << ": "
                //           << std::endl;

                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q = (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    qr = 0;
                    qs = 0;
                    qt = 0;

                    // step-5 : Load Geometric Factors, coalesced access
                    Grr = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   0 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grs = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   1 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grt = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   2 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gss = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   3 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gst = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   4 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gtt = G_device[cell_index * symmetric_tensor_dimension * n_q_points_total +
                                   5 * n_q_points_total + p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];

                    // step-6 : Multiply by D
                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qr += s_wsp1[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + n] *
                              co_shape_gradients_scratch[n * n_q_points_1d + p];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qs += s_wsp1[r * n_q_points_1d * n_q_points_1d + n * n_q_points_1d + p] *
                              co_shape_gradients_scratch[n * n_q_points_1d + q];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qt += s_wsp1[n * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] *
                              co_shape_gradients_scratch[n * n_q_points_1d + r];
                      }

                    // std::cout << "BK3 kernel cell_id = " << cell_id << ": ";
                    // std::cout << qt << " " << qs << " " << qr << " ";
                    // std::cout << std::endl << std::endl;

                    // step-7 : Apply chain rule
                    rqr[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] =
                      Grr * qr + Grs * qs + Grt * qt;
                    rqs[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] =
                      Grs * qr + Gss * qs + Gst * qt;
                    rqt[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] =
                      Grt * qr + Gst * qs + Gtt * qt;
                  }
                team_member.team_barrier();


                // std::cout << "BK3 kernel cell_id = " << cell_id << ": "
                //           << std::endl
                //           << std::endl;

                // for (unsigned int tid = 0;
                //      tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                //      tid += 1)
                //   {
                //     // std::cout << qt << " " << qs << " " << qr << " ";
                //     std::cout << rqr[tid] << " " << rqs[tid] << " " <<
                //     rqt[tid]
                //               << " ";

                //     std::cout << std::endl << std::endl;
                //   }



                // step-8 : Compute out vector in GL nodes
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q = (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    Number sum = 0;
                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqr[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + n] *
                             co_shape_gradients_scratch[p * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqs[r * n_q_points_1d * n_q_points_1d + n * n_q_points_1d + p] *
                             co_shape_gradients_scratch[q * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqt[n * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] *
                             co_shape_gradients_scratch[r * n_q_points_1d + n];

                    s_wsp1[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                // std::cout << "BK3 kernel cell_id = " << cell_id << ": ";
                // for (unsigned int i = 0; i < n_local_dofs_total; ++i)
                //   std::cout << s_wsp1(i) << " ";
                // std::cout << std::endl << std::endl;

                /*
                Interpolate to GLL nodes
                */



                // step-9 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_local_dofs_1d);
                    const int q = (tid % (n_q_points_1d * n_local_dofs_1d)) / n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    Number sum = 0.0;
                    for (unsigned int r = 0; r < n_q_points_1d; ++r)
                      {
                        sum += s_wsp1[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] *
                               shape_values_scratch[k * n_local_dofs_1d + r];
                      }
                    s_wsp0[k * n_q_points_1d * n_local_dofs_1d + q * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                // step-10 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_local_dofs_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) / n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    Number sum = 0.0;
                    for (unsigned int q = 0; q < n_q_points_1d; q++)
                      {
                        sum += s_wsp0[k * n_q_points_1d * n_q_points_1d + q * n_q_points_1d + p] *
                               shape_values_scratch[j * n_q_points_1d + q];
                      }
                    s_wsp1[k * n_q_points_1d * n_local_dofs_1d + j * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                // step-11 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < n_local_dofs_1d * n_local_dofs_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int i = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) / n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    Number sum = 0.0;
                    for (unsigned int p = 0; p < n_q_points_1d; ++p)
                      {
                        sum += s_wsp1[k * n_q_points_1d * n_local_dofs_1d + j * n_q_points_1d + p] *
                               shape_values_scratch[i * n_q_points_1d + p];
                      }
                    s_wsp0[k * n_local_dofs_1d * n_local_dofs_1d + j * n_local_dofs_1d + i] = sum;
                  }
                team_member.team_barrier();

                // step-12 : Copy wsp0 to out
                for (unsigned int tid = threadIdx; tid < n_local_dofs_total; tid += blockSize)
                  {
                    const unsigned int dof_index = dof_indices(tid, cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      Kokkos::atomic_add(&out_device[dof_index], s_wsp0[tid]);
                  }
                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif