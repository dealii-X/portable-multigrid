#ifndef bk3_kokkos_kernel_block_h
#define bk3_kokkos_kernel_block_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

// Copy of bk3_kokkos_kernel.h's KokkosKernel(), extended to apply the same
// operator to n_rhs right-hand sides in a single kernel launch instead of
// n_rhs sequential calls -- "batch"/"batched" throughout this file (and the
// original) refers only to the pre-existing per-team *cell* batching
// (nelmtPerBatch etc.); the new n_rhs dimension added here is called
// "block" throughout, to keep the two kinds of batching distinguishable by
// name. d_in/d_out are laid out as n_rhs blocks of dof_stride each (RHS k
// occupies [k*dof_stride, (k+1)*dof_stride)); G and dof_indices are
// untouched (same physical mesh, so they don't vary with the RHS index)
// and are looked up via the *physical* cell obtained by
// global_cell_index % n_cells, while global_cell_index / n_cells selects
// which RHS block of d_in/d_out to read/write. See the discussion in the
// bddc-preconditioner session about solving compute_local_coarse_matrix()'s
// n_local_coarse_dofs sequential vmult_fine_correction() solves together as
// one block: this is the leaf-kernel piece of that. KokkosKernel_1D_Block
// (unused even in the original file -- the active call sites only use
// KokkosKernel) was dropped from this copy to keep it focused.
namespace BK3Block
{
  namespace Parallel
  {

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename Number>
    void
    KokkosKernelBlock(const DeviceView<Number> d_shape_values,
                        const DeviceView<Number> d_co_shape_gradients,
                        const DeviceView<Number> d_G,
                        const DeviceView<Number> d_in,
                        DeviceView<Number>       d_out,
                        const DoFIndicesView     dof_indices,
                        const unsigned int       n_cells,
                        const unsigned int       n_rhs,
                        const unsigned int       dof_stride,
                        const unsigned int       n_blocks          = numbers::invalid_unsigned_int,
                        const unsigned int       threads_per_block = numbers::invalid_unsigned_int,
                        const CellRangeIdView    cell_range_ids    = CellRangeIdView())
    {
      if (n_cells == 0 || n_rhs == 0)
        return;

      constexpr int nq_total = Utilities::pow(nq, dim);
      constexpr int nm_total = Utilities::pow(nm, dim);

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 4;

      if (cell_range_ids.size() > 0)
        AssertDimension(cell_range_ids.size(), n_cells);

      // Virtual element count: n_rhs independent copies of the same n_cells
      // mesh, laid out RHS-major (see class-level comment for the indexing
      // convention used to recover the physical cell / RHS index below).
      const int nelmt = static_cast<int>(n_cells) * static_cast<int>(n_rhs);

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

                        // Recover the physical cell (for dof_indices, which
                        // does not vary with the RHS) and the RHS block
                        // (for d_in, which is n_rhs blocks of dof_stride).
                        unsigned int physical_cell = global_cell_index % n_cells;
                        const unsigned int rhs_idx = global_cell_index / n_cells;

                        if (cell_range_ids.size() > 0)
                          physical_cell = cell_range_ids(physical_cell);

                        const unsigned int rhs_offset = rhs_idx * dof_stride;

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
                                  dof_indices(local_idx, physical_cell);

                                // The index in the batched shared memory array
                                const int shared_idx = e * nm_total + local_idx;

                                if (dof_index == numbers::invalid_unsigned_int)
                                  s_wsp0[shared_idx] = 0;
                                else
                                  s_wsp0[shared_idx] = d_in[rhs_offset + dof_index];
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
                                  dof_indices(local_idx, physical_cell);

                                // The index in the batched shared memory array
                                const int shared_idx = e * nm_total + local_idx;

                                if (dof_index == numbers::invalid_unsigned_int)
                                  s_wsp0[shared_idx] = 0;
                                else
                                  s_wsp0[shared_idx] = d_in[rhs_offset + dof_index];
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

                      // G does not vary with the RHS -- only the physical
                      // cell (global_cell_index % n_cells) matters here.
                      unsigned int physical_cell = global_cell_index % n_cells;

                      if (cell_range_ids.size() > 0)
                        physical_cell = cell_range_ids(physical_cell);

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
                              Grr = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        0 * nq_total + q * nq + p];

                              Grs = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        1 * nq_total + q * nq + p];

                              Gss = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
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
                              Grr = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        0 * nq_total + r * nq * nq + q * nq + p];

                              Grs = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        1 * nq_total + r * nq * nq + q * nq + p];

                              Grt = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        2 * nq_total + r * nq * nq + q * nq + p];

                              Gss = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        3 * nq_total + r * nq * nq + q * nq + p];

                              Gst = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
                                        4 * nq_total + r * nq * nq + q * nq + p];

                              Gtt = d_G[physical_cell * symmetric_tensor_dimension * nq_total +
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

                      unsigned int       physical_cell = global_cell_index % n_cells;
                      const unsigned int rhs_idx       = global_cell_index / n_cells;

                      if (cell_range_ids.size() > 0)
                        physical_cell = cell_range_ids(physical_cell);

                      const unsigned int rhs_offset = rhs_idx * dof_stride;

                      if (dim == 2)
                        {
                          const int i = tid % nm;

                          for (int j = 0; j < nm; ++j)
                            {
                              const int local_idx = j * nm + i;

                              // Find where this node lives in the global 'd_out'
                              // vector
                              const unsigned int dof_index =
                                dof_indices(local_idx, physical_cell);

                              // The index in our batched shared memory result
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // CRITICAL: Use atomic_add because elements share
                                  // nodes!
                                  Kokkos::atomic_add(&d_out[rhs_offset + dof_index],
                                                     s_wsp0[shared_idx]);
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
                                dof_indices(local_idx, physical_cell);

                              // The index in our batched shared memory result
                              const int shared_idx = e * nm_total + local_idx;

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // CRITICAL: Use atomic_add because elements share
                                  // nodes!
                                  Kokkos::atomic_add(&d_out[rhs_offset + dof_index],
                                                     s_wsp0[shared_idx]);
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

  } // namespace Parallel
} // namespace BK3Block

DEAL_II_NAMESPACE_CLOSE

#endif