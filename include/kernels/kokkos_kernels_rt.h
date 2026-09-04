#ifndef kokkos_kernels_rt_h
#define kokkos_kernels_rt_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/portable_tensor_product_kernels.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

#include <vector>

#include "matrix_free/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace RT
  {

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    template <typename Number>
    using SharedViewValues =
      Kokkos::View<Number *,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int n_t, int n_q, typename Number>
    void
    mass_operator(const Kokkos::Array<DeviceView<Number>, 2> shape_info,
                  const DeviceView<Number>                   geometric_tensor,
                  const DeviceView<Number>                   vector_in,
                  DeviceView<Number>                         vector_out,
                  const DoFIndicesView                       dof_indices,
                  const unsigned int                         n_cells,
                  const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
                  const unsigned int n_blocks          = numbers::invalid_unsigned_int,
                  const unsigned int threads_per_block = numbers::invalid_unsigned_int)

    {
      if (n_cells == 0)
        return;

      AssertThrow(dim > 1, ExcNotImplemented());

      static_assert(n_t > 1, "Degree 0 not supported");

      AssertThrow(n_q > n_t, ExcNotImplemented());

      constexpr int n_n = n_t + 1;

      constexpr int n_q_total = Utilities::pow(n_q, dim);

      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);

      const int nelmt = n_cells;

      // const size_t shmemPerBlock =
      //   Kokkos::TeamPolicy<>::scratch_size_max(0); // maximum shared memory size per thread block


      int shmemPerBlock = 10800; // total shared memory used per block (KB)


      const int nelmtPerBatch = (n_cells_per_batch == numbers::invalid_unsigned_int) ?
                                  (shmemPerBlock / (5 * n_q_total) / sizeof(Number)) :
                                  n_cells_per_batch;

      const int numBlocks = (n_blocks == numbers::invalid_unsigned_int) ?
                              std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                              n_blocks;

      const int threadsPerBlock =
        (threads_per_block == numbers::invalid_unsigned_int) ?
          std::min(std::max(1, nelmtPerBatch) * Utilities::pow(n_q, dim - 1), 512) :
          threads_per_block;


      const unsigned int ssize = n_n * n_q + n_t * n_q + 5 * nelmtPerBatch * n_q_total;

      const unsigned int shmem_size = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number r_p[n_q];

          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *shape_values_normal  = scratch;
          Number *shape_values_tangent = shape_values_normal + n_n * n_q;

          Number *s_wsp0 = shape_values_tangent + n_t * n_q;
          Number *s_wsp1 = s_wsp0 + nelmtPerBatch * n_q_total;

          Number *s_uq_0 = s_wsp1 + nelmtPerBatch * n_q_total;
          Number *s_uq_1 = s_uq_0 + nelmtPerBatch * n_q_total;
          Number *s_uq_2;
          if constexpr (dim > 2)
            s_uq_2 = s_uq_1 + nelmtPerBatch * n_q_total;

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();


          // copy to shared memory
          {
            for (int tid = threadIdx; tid < n_n * n_q; tid += blockSize)
              {
                shape_values_normal[tid] = shape_info[0][tid];
              }
            for (int tid = threadIdx; tid < n_t * n_q; tid += blockSize)
              {
                shape_values_tangent[tid] = shape_info[1][tid];
              }
            team_member.team_barrier();
          }

          // element batch iteration
          int eb = team_member.league_rank();

          while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              // current nelmtPerBatch (edge case, last batch size can be less)
              const int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

              // ====================================================
              // PHASE 1: Read from global L vector per component
              // ====================================================
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_x != numbers::invalid_unsigned_int)
                        s_uq_0[tid] = vector_in[dof_x];
                      else
                        s_uq_0[tid] = 0;
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        s_uq_1[tid] = vector_in[dof_y];
                      else
                        s_uq_1[tid] = 0;
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          s_uq_2[tid] = vector_in[dof_z];
                        else
                          s_uq_2[tid] = 0;
                      }
                  }
                team_member.team_barrier();
              }

              // ====================================================
              // PHASE 2: Interpolate to quadrature nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = Utilities::pow(n_t, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];

                                s_wsp1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];


                                s_wsp1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_wsp1[e * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_uq_0[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_wsp1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] = s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];
                                s_wsp1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_wsp1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_wsp1[e * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_uq_1[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_wsp1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] =
                                s_wsp0[e * n_dofs_per_component + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }
                {
                  // ------------------------ Component 2 (x-direction) ------------------------
                  // z is normal (basis_n), x and y are tangent (basis_t)
                  if constexpr (dim == 3)
                    {
                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_wsp1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in y direction
                      {
                        constexpr int co_dimension_size = n_q * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_wsp1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_wsp0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in z direction
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_n; ++k)
                              r_p[k] = s_wsp0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_n; ++k)
                                  tmp += shape_values_normal[k * n_q + r] * r_p[k];

                                s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                }
              }

              // ====================================================
              // PHASE 3: Apply Piola Geometry Metric
              // ====================================================
              {
                constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
                constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);

                for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                     tid += blockSize)
                  {
                    const int e = tid / co_dimension_size;

                    //  Base offset for the current element's geometric factors
                    const int e_offset =
                      eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
                      e * symmetric_tensor_dimension * n_q_total;

                    Number d_G[symmetric_tensor_dimension];
                    Number u[dim];

                    if (dim == 2)
                      {
                        const int p = tid % n_q;

                        for (int q = 0; q < n_q; ++q)
                          {
                            for (int d = 0; d < symmetric_tensor_dimension; ++d)
                              d_G[d] = geometric_tensor[e_offset + d * n_q_total + q * n_q + p];

                            const int shm_idx = e * n_q_total + q * n_q + p;

                            u[0] = s_uq_0[shm_idx];
                            u[1] = s_uq_1[shm_idx];

                            s_uq_0[shm_idx] = d_G[0] * u[0] + d_G[1] * u[1];
                            s_uq_1[shm_idx] = d_G[1] * u[0] + d_G[2] * u[1];
                          }
                      }
                    else if (dim == 3)
                      {
                        const int p = tid % (n_q * n_q) / n_q;
                        const int q = tid % n_q;

                        for (int r = 0; r < n_q; ++r)
                          {
                            for (int d = 0; d < symmetric_tensor_dimension; ++d)
                              d_G[d] = geometric_tensor[e_offset + d * n_q_total + r * n_q * n_q +
                                                        q * n_q + p];

                            const int shm_idx = e * n_q_total + r * n_q * n_q + q * n_q + p;

                            u[0] = s_uq_0[shm_idx];
                            u[1] = s_uq_1[shm_idx];
                            u[2] = s_uq_2[shm_idx];

                            s_uq_0[shm_idx] = d_G[0] * u[0] + d_G[1] * u[1] + d_G[2] * u[2];
                            s_uq_1[shm_idx] = d_G[1] * u[0] + d_G[3] * u[1] + d_G[4] * u[2];
                            s_uq_2[shm_idx] = d_G[2] * u[0] + d_G[4] * u[1] + d_G[5] * u[2];
                          }
                      }
                  }
                team_member.team_barrier();
              }


              // ====================================================
              // PHASE 4: Project back to Nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_0[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_wsp1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_wsp1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_t : n_t * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_wsp1[e * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_wsp1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_1[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_wsp1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_wsp0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_wsp1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_wsp1[e * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_wsp1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 2 (z-direction) ------------------------
                // z is normal (basis_n), x and y are tangent (basis_t)
                if constexpr (dim == 3)
                  {
                    // component 2 in z direction
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_n; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_normal[k * n_q + r] * r_p[r];

                              s_wsp0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                    // component 2 in y direction
                    {
                      constexpr int co_dimension_size = n_q * n_n;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          {
                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_wsp0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_wsp1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                          team_member.team_barrier();
                        }

                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_wsp1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                  }
              }

              // ====================================================
              // PHASE 5: Write the results to the global L vector.
              // ====================================================

              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_x != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_x], s_uq_0[tid]);
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_y], s_uq_1[tid]);
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&vector_out[dof_z], s_uq_2[tid]);
                      }
                  }
                team_member.team_barrier();
              }
              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }

    template <int dim, int n_t, int n_q, typename Number>
    void
    stiffness_operator(const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
                       const DeviceView<Number>                   shape_gradients_collocation,
                       const DeviceView<Number>                   geometric_tensor_mass,
                       const DeviceView<Number>                   geometric_tensor_stiffness,
                       const DeviceView<Number>                   vector_in,
                       DeviceView<Number>                         vector_out,
                       const DoFIndicesView                       dof_indices,
                       const unsigned int                         n_cells,
                       const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
                       const unsigned int n_blocks          = numbers::invalid_unsigned_int,
                       const unsigned int threads_per_block = numbers::invalid_unsigned_int)

    {
      if (n_cells == 0)
        return;

      AssertThrow(dim > 1, ExcNotImplemented());

      static_assert(n_t > 1, "Degree 0 not supported");

      AssertThrow(n_q > n_t, ExcNotImplemented());

      constexpr int n_n = n_t + 1;

      constexpr int n_q_total = Utilities::pow(n_q, dim);

      constexpr int n_components = dim;

      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);
      const int     nelmt                = n_cells;

      // const size_t shmemPerBlock =
      //   Kokkos::TeamPolicy<>::scratch_size_max(0); // maximum shared memory size per thread block

      int shmemPerBlock = 10800; // total shared memory used per block (KB)

      const int nelmtPerBatch =
        (n_cells_per_batch == numbers::invalid_unsigned_int) ?
          // at least 1: for large degree/dim the per-cell scratch footprint can
          // exceed the shmemPerBlock budget on its own, flooring this to 0
          // otherwise and dividing by zero below.
          std::max(1,
                  shmemPerBlock / (n_components * (dim + 1) * n_q_total) /
                    static_cast<int>(sizeof(Number))) :
          n_cells_per_batch;

      const int numBlocks = (n_blocks == numbers::invalid_unsigned_int) ?
                              std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                              n_blocks;

      const int threadsPerBlock =
        (threads_per_block == numbers::invalid_unsigned_int) ?
          std::min(std::max(1, nelmtPerBatch) * Utilities::pow(n_q, dim - 1), 512) :
          threads_per_block;


      const unsigned int ssize = n_n * n_q   // normal shape values
                                 + n_t * n_q // tangent shape values
                                 + n_q * n_q // shape gradients at collocation points
                                 + n_components * nelmtPerBatch * n_q_total        // values
                                 + n_components * dim * nelmtPerBatch * n_q_total; // gradients


      const unsigned int shmem_size = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number r_p[n_q];

          Number r_p0[n_q];
          Number r_p1[n_q];
          Number r_p2[n_q];
          Number r_q[n_q];
          Number r_r[n_q];


          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *shape_values_normal  = scratch;
          Number *shape_values_tangent = shape_values_normal + n_n * n_q;
          Number *co_shape_gradients   = shape_values_tangent + n_t * n_q;


          Number *s_uq_0  = co_shape_gradients + n_q * n_q;
          Number *s_duq_0 = s_uq_0 + nelmtPerBatch * n_q_total;
          Number *s_uq_1  = s_duq_0 + nelmtPerBatch * n_q_total * dim;
          Number *s_duq_1 = s_uq_1 + nelmtPerBatch * n_q_total;

          Number *s_uq_2, *s_duq_2;
          if constexpr (dim > 2)
            {
              s_uq_2  = s_duq_1 + nelmtPerBatch * n_q_total * dim;
              s_duq_2 = s_uq_2 + nelmtPerBatch * n_q_total;
            }

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();


          // copy to shared memory
          {
            for (int tid = threadIdx; tid < n_n * n_q; tid += blockSize)
              {
                shape_values_normal[tid] = shape_values_info[0][tid];
              }
            for (int tid = threadIdx; tid < n_t * n_q; tid += blockSize)
              {
                shape_values_tangent[tid] = shape_values_info[1][tid];
              }

            for (int tid = threadIdx; tid < n_q * n_q; tid += blockSize)
              {
                co_shape_gradients[tid] = shape_gradients_collocation[tid];
              }
            team_member.team_barrier();
          }

          // element batch iteration
          int eb = team_member.league_rank();

          while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              // current nelmtPerBatch (edge case, last batch size can be less)
              const int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

              // ====================================================
              // PHASE 1: Read from global L vector per component
              // ====================================================
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);
                      if (dof_x != numbers::invalid_unsigned_int)
                        s_uq_0[tid] = vector_in[dof_x];
                      else
                        s_uq_0[tid] = 0;
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        s_uq_1[tid] = vector_in[dof_y];
                      else
                        s_uq_1[tid] = 0;
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          s_uq_2[tid] = vector_in[dof_z];
                        else
                          s_uq_2[tid] = 0;
                      }
                  }
                team_member.team_barrier();
              }

              // ====================================================
              // PHASE 2: Interpolate to quadrature nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = Utilities::pow(n_t, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];


                                s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_uq_0[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];
                                s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_uq_1[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] =
                                s_duq_0[e * n_dofs_per_component + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }
                {
                  // ------------------------ Component 2 (x-direction) ------------------------
                  // z is normal (basis_n), x and y are tangent (basis_t)
                  if constexpr (dim == 3)
                    {
                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in y direction
                      {
                        constexpr int co_dimension_size = n_q * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in z direction
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_n; ++k)
                              r_p[k] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_n; ++k)
                                  tmp += shape_values_normal[k * n_q + r] * r_p[k];

                                s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                }
              }

              // ====================================================
              // PHASE 3: Evaluate gradients at quadrature nodes
              // ====================================================

              {
                // 1. evaluate gradients in reference space and multiply by stiffness geometric
                // tensor
                {
                  constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      //  Base offset for the current element's geometric factors
                      const int e_offset =
                        eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
                        e * symmetric_tensor_dimension * n_q_total;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = s_uq_0[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = s_uq_1[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number d_G[dim][dim];
                          Number qr[dim];
                          Number qs[dim];

                          for (int p = 0; p < n_q; ++p)
                            {
                              // Load stiffness geometric tensor
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  qr[d1] = 0;
                                  qs[d1] = 0;
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_stiffness[e_offset + index * n_q_total +
                                                                   q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                }

                              // Multiply by D
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];

                                  qs[0] += r_q[n] * s_uq_0[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] * s_uq_1[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
                              const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

                              s_duq_0[idx0] = qr[0] * d_G[0][0] + qs[0] * d_G[1][0];
                              s_duq_0[idx1] = qr[0] * d_G[0][1] + qs[0] * d_G[1][1];

                              s_duq_1[idx0] = qr[1] * d_G[0][0] + qs[1] * d_G[1][0];
                              s_duq_1[idx1] = qr[1] * d_G[0][1] + qs[1] * d_G[1][1];
                            }
                        }
                      else if constexpr (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          for (int n = 0; n < n_q; ++n)

                            {
                              r_p0[n] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];
                              r_p1[n] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];
                              r_p2[n] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                              r_r[n] = co_shape_gradients[n * n_q + r];
                            }

                          Number d_G[dim][dim];
                          Number qr[dim];
                          Number qs[dim];
                          Number qt[dim];

                          for (int p = 0; p < n_q; ++p)
                            {
                              // Load stiffness geometric tensor
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  qr[d1] = 0;
                                  qs[d1] = 0;
                                  qt[d1] = 0;
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_stiffness[e_offset + index * n_q_total +
                                                                   r * n_q * n_q + q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                }
                              // Multiply by D
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] +=
                                    r_q[n] * s_uq_0[e * n_q_total + r * n_q * n_q + n * n_q + p];
                                  qs[1] +=
                                    r_q[n] * s_uq_1[e * n_q_total + r * n_q * n_q + n * n_q + p];
                                  qs[2] +=
                                    r_q[n] * s_uq_2[e * n_q_total + r * n_q * n_q + n * n_q + p];

                                  qt[0] +=
                                    r_r[n] * s_uq_0[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                  qt[1] +=
                                    r_r[n] * s_uq_1[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                  qt[2] +=
                                    r_r[n] * s_uq_2[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx2 =
                                e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q + p;

                              s_duq_0[idx0] =
                                qr[0] * d_G[0][0] + qs[0] * d_G[1][0] + qt[0] * d_G[2][0];
                              s_duq_0[idx1] =
                                qr[0] * d_G[0][1] + qs[0] * d_G[1][1] + qt[0] * d_G[2][1];
                              s_duq_0[idx2] =
                                qr[0] * d_G[0][2] + qs[0] * d_G[1][2] + qt[0] * d_G[2][2];

                              s_duq_1[idx0] =
                                qr[1] * d_G[0][0] + qs[1] * d_G[1][0] + qt[1] * d_G[2][0];
                              s_duq_1[idx1] =
                                qr[1] * d_G[0][1] + qs[1] * d_G[1][1] + qt[1] * d_G[2][1];
                              s_duq_1[idx2] =
                                qr[1] * d_G[0][2] + qs[1] * d_G[1][2] + qt[1] * d_G[2][2];

                              s_duq_2[idx0] =
                                qr[2] * d_G[0][0] + qs[2] * d_G[1][0] + qt[2] * d_G[2][0];
                              s_duq_2[idx1] =
                                qr[2] * d_G[0][1] + qs[2] * d_G[1][1] + qt[2] * d_G[2][1];
                              s_duq_2[idx2] =
                                qr[2] * d_G[0][2] + qs[2] * d_G[1][2] + qt[2] * d_G[2][2];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // 2. multiply by the mass geometric tensor
                {
                  constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      //  Base offset for the current element's geometric factors
                      const int e_offset =
                        eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
                        e * symmetric_tensor_dimension * n_q_total;

                      Number d_G[dim][dim];
                      Number qr[dim];
                      Number qs[dim];

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          for (int p = 0; p < n_q; ++p)
                            {
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_mass[e_offset + index * n_q_total +
                                                              q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }

                                  qr[d1] =
                                    s_duq_0[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
                                  qs[d1] =
                                    s_duq_1[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
                                }

                              const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
                              const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

                              s_duq_0[idx0] = d_G[0][0] * qr[0] + d_G[0][1] * qs[0];
                              s_duq_0[idx1] = d_G[0][0] * qr[1] + d_G[0][1] * qs[1];

                              s_duq_1[idx0] = d_G[1][0] * qr[0] + d_G[1][1] * qs[0];
                              s_duq_1[idx1] = d_G[1][0] * qr[1] + d_G[1][1] * qs[1];
                            }
                        }

                      else if constexpr (dim == 3)
                        {
                          Number qt[dim];

                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          for (int p = 0; p < n_q; ++p)
                            {
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_mass[e_offset + index * n_q_total +
                                                              r * n_q * n_q + q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                  qr[d1] = s_duq_0[e * dim * n_q_total + d1 * n_q_total +
                                                   r * n_q * n_q + q * n_q + p];
                                  qs[d1] = s_duq_1[e * dim * n_q_total + d1 * n_q_total +
                                                   r * n_q * n_q + q * n_q + p];
                                  qt[d1] = s_duq_2[e * dim * n_q_total + d1 * n_q_total +
                                                   r * n_q * n_q + q * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx2 =
                                e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q + p;

                              s_duq_0[idx0] =
                                d_G[0][0] * qr[0] + d_G[0][1] * qs[0] + d_G[0][2] * qt[0];
                              s_duq_0[idx1] =
                                d_G[0][0] * qr[1] + d_G[0][1] * qs[1] + d_G[0][2] * qt[1];
                              s_duq_0[idx2] =
                                d_G[0][0] * qr[2] + d_G[0][1] * qs[2] + d_G[0][2] * qt[2];

                              s_duq_1[idx0] =
                                d_G[1][0] * qr[0] + d_G[1][1] * qs[0] + d_G[1][2] * qt[0];
                              s_duq_1[idx1] =
                                d_G[1][0] * qr[1] + d_G[1][1] * qs[1] + d_G[1][2] * qt[1];
                              s_duq_1[idx2] =
                                d_G[1][0] * qr[2] + d_G[1][1] * qs[2] + d_G[1][2] * qt[2];

                              s_duq_2[idx0] =
                                d_G[2][0] * qr[0] + d_G[2][1] * qs[0] + d_G[2][2] * qt[0];
                              s_duq_2[idx1] =
                                d_G[2][0] * qr[1] + d_G[2][1] * qs[1] + d_G[2][2] * qt[1];
                              s_duq_2[idx2] =
                                d_G[2][0] * qr[2] + d_G[2][1] * qs[2] + d_G[2][2] * qt[2];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // 3. integrate, i.e apply D^T
                {
                  constexpr int co_dimension_size = Utilities::pow(n_q, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if constexpr (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < n_q; ++n)
                            {
                              const int idx_0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + n;

                              r_p0[n] = s_duq_0[idx_0];
                              r_p1[n] = s_duq_1[idx_0];

                              r_q[n] = co_shape_gradients[q * n_q + n];
                            }

                          for (int p = 0; p < n_q; ++p)
                            {
                              Number tmp0 = 0, tmp1 = 0;

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
                                  tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_1 =
                                    e * dim * n_q_total + 1 * n_q_total + n * n_q + p;
                                  tmp0 += s_duq_0[idx_1] * r_q[n];
                                  tmp1 += s_duq_1[idx_1] * r_q[n];
                                }

                              s_uq_0[e * n_q_total + q * n_q + p] = tmp0;
                              s_uq_1[e * n_q_total + q * n_q + p] = tmp1;
                            }
                        }
                      else if constexpr (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          // copy to register
                          for (int n = 0; n < n_q; ++n)
                            {
                              const int idx_0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + n;

                              r_p0[n] = s_duq_0[idx_0];
                              r_p1[n] = s_duq_1[idx_0];
                              r_p2[n] = s_duq_2[idx_0];

                              r_q[n] = co_shape_gradients[q * n_q + n];
                              r_r[n] = co_shape_gradients[r * n_q + n];
                            }

                          for (int p = 0; p < n_q; ++p)
                            {
                              Number tmp0 = 0, tmp1 = 0, tmp2 = 0;

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
                                  tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
                                  tmp2 += r_p2[n] * co_shape_gradients[p * n_q + n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_1 = e * dim * n_q_total + 1 * n_q_total +
                                                    r * n_q * n_q + n * n_q + p;

                                  tmp0 += s_duq_0[idx_1] * r_q[n];
                                  tmp1 += s_duq_1[idx_1] * r_q[n];
                                  tmp2 += s_duq_2[idx_1] * r_q[n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_2 = e * dim * n_q_total + 2 * n_q_total +
                                                    n * n_q * n_q + q * n_q + p;

                                  tmp0 += s_duq_0[idx_2] * r_r[n];
                                  tmp1 += s_duq_1[idx_2] * r_r[n];
                                  tmp2 += s_duq_2[idx_2] * r_r[n];
                                }

                              s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p] = tmp0;
                              s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p] = tmp1;
                              s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p] = tmp2;
                            }
                        }
                    }
                }
                team_member.team_barrier();
              }


              // ====================================================
              // PHASE 4: Project back to Nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_0[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_t : n_t * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_1[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 2 (z-direction) ------------------------
                // z is normal (basis_n), x and y are tangent (basis_t)
                if constexpr (dim == 3)
                  {
                    // component 2 in z direction
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_n; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_normal[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                    // component 2 in y direction
                    {
                      constexpr int co_dimension_size = n_q * n_n;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          {
                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                          team_member.team_barrier();
                        }

                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                  }
              }

              // ====================================================
              // PHASE 5: Write the results to the global L vector.
              // ====================================================

              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_x != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_x], s_uq_0[tid]);
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_y], s_uq_1[tid]);
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&vector_out[dof_z], s_uq_2[tid]);
                      }
                  }
                team_member.team_barrier();
              }

              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }

    template <int dim, int n_t, int n_q, typename Number>
    void
    helmholtz_operator(const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
                       const DeviceView<Number>                   shape_gradients_collocation,
                       const DeviceView<Number>                   geometric_tensor_mass,
                       const DeviceView<Number>                   geometric_tensor_stiffness,
                       const DeviceView<Number>                   vector_in,
                       DeviceView<Number>                         vector_out,
                       const DoFIndicesView                       dof_indices,
                       const unsigned int                         n_cells,
                       const Number                               factor_mass    = Number(1),
                       const Number                               factor_laplace = Number(1),
                       const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
                       const unsigned int n_blocks          = numbers::invalid_unsigned_int,
                       const unsigned int threads_per_block = numbers::invalid_unsigned_int)

    {
      constexpr int n_components = dim;

      if (n_cells == 0)
        return;

      AssertThrow(dim > 1, ExcNotImplemented());

      static_assert(n_t > 1, "Degree 0 not supported");

      AssertThrow(n_q > n_t, ExcNotImplemented());

      constexpr int n_n = n_t + 1;

      constexpr int n_q_total = Utilities::pow(n_q, dim);

      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);
      const int     nelmt                = n_cells;

      // const size_t shmemPerBlock =
      //   Kokkos::TeamPolicy<>::scratch_size_max(0); // maximum shared memory size per thread block

      int shmemPerBlock = 10800; // total shared memory used per block (KB)

      const int nelmtPerBatch =
        (n_cells_per_batch == numbers::invalid_unsigned_int) ?
          // at least 1: for large degree/dim the per-cell scratch footprint can
          // exceed the shmemPerBlock budget on its own, flooring this to 0
          // otherwise and dividing by zero below.
          std::max(1,
                  shmemPerBlock / (n_components * (dim + 1) * n_q_total) /
                    static_cast<int>(sizeof(Number))) :
          n_cells_per_batch;

      const int numBlocks = (n_blocks == numbers::invalid_unsigned_int) ?
                              std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                              n_blocks;

      const int threadsPerBlock =
        (threads_per_block == numbers::invalid_unsigned_int) ?
          std::min(std::max(1, nelmtPerBatch) * Utilities::pow(n_q, dim - 1), 512) :
          threads_per_block;


      const unsigned int ssize = n_n * n_q   // normal shape values
                                 + n_t * n_q // tangent shape values
                                 + n_q * n_q // shape gradients at collocation points
                                 + n_components * nelmtPerBatch * n_q_total        // values
                                 + n_components * dim * nelmtPerBatch * n_q_total; // gradients


      unsigned int shmem_size = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number r_p[n_q];

          Number r_p0[n_q];
          Number r_p1[n_q];
          Number r_p2[n_q];
          Number r_q[n_q];
          Number r_r[n_q];

          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *shape_values_normal  = scratch;
          Number *shape_values_tangent = shape_values_normal + n_n * n_q;
          Number *co_shape_gradients   = shape_values_tangent + n_t * n_q;

          Number *s_uq_0  = co_shape_gradients + n_q * n_q;
          Number *s_duq_0 = s_uq_0 + nelmtPerBatch * n_q_total;
          Number *s_uq_1  = s_duq_0 + nelmtPerBatch * n_q_total * dim;
          Number *s_duq_1 = s_uq_1 + nelmtPerBatch * n_q_total;

          Number *s_uq_2, *s_duq_2;
          if constexpr (dim > 2)
            {
              s_uq_2  = s_duq_1 + nelmtPerBatch * n_q_total * dim;
              s_duq_2 = s_uq_2 + nelmtPerBatch * n_q_total;
            }

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();


          // copy to shared memory
          {
            for (int tid = threadIdx; tid < n_n * n_q; tid += blockSize)
              {
                shape_values_normal[tid] = shape_values_info[0][tid];
              }
            for (int tid = threadIdx; tid < n_t * n_q; tid += blockSize)
              {
                shape_values_tangent[tid] = shape_values_info[1][tid];
              }
            for (int tid = threadIdx; tid < n_q * n_q; tid += blockSize)
              {
                co_shape_gradients[tid] = shape_gradients_collocation[tid];
              }
            team_member.team_barrier();
          }

          // element batch iteration
          int eb = team_member.league_rank();

          while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              // current nelmtPerBatch (edge case, last batch size can be less)
              const int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

              // ====================================================
              // PHASE 1: Read from global L vector per component
              // ====================================================
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);
                      if (dof_x != numbers::invalid_unsigned_int)
                        s_uq_0[tid] = vector_in[dof_x];
                      else
                        s_uq_0[tid] = 0;
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        s_uq_1[tid] = vector_in[dof_y];
                      else
                        s_uq_1[tid] = 0;
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          s_uq_2[tid] = vector_in[dof_z];
                        else
                          s_uq_2[tid] = 0;
                      }
                  }
                team_member.team_barrier();
              }

              // ====================================================
              // PHASE 2: Interpolate to quadrature nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = Utilities::pow(n_t, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_n; ++i)
                              r_p[i] = s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_n; ++i)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[i];


                                s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_uq_0[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];
                                s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_uq_1[e * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int j = 0; j < n_n; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_n; ++j)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in z direction
                  {
                    if constexpr (dim == 3)
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_t; ++k)
                              r_p[k] =
                                s_duq_0[e * n_dofs_per_component + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_t; ++k)
                                  tmp += shape_values_tangent[k * n_q + r] * r_p[k];

                                s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                  }
                }
                {
                  // ------------------------ Component 2 (x-direction) ------------------------
                  // z is normal (basis_n), x and y are tangent (basis_t)
                  if constexpr (dim == 3)
                    {
                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int i = 0; i < n_t; ++i)
                              r_p[i] = s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i];

                            for (int p = 0; p < n_q; ++p)
                              {
                                Number tmp = 0;
                                for (int i = 0; i < n_t; ++i)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[i];

                                s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in y direction
                      {
                        constexpr int co_dimension_size = n_q * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int j = 0; j < n_t; ++j)
                              r_p[j] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int q = 0; q < n_q; ++q)
                              {
                                Number tmp = 0;
                                for (int j = 0; j < n_t; ++j)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[j];

                                s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }

                      // component 2 in z direction
                      {
                        constexpr int co_dimension_size = n_q * n_q;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int p = (tid % co_dimension_size) / n_q;
                            const int q = tid % n_q;

                            for (int k = 0; k < n_n; ++k)
                              r_p[k] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int r = 0; r < n_q; ++r)
                              {
                                Number tmp = 0;
                                for (int k = 0; k < n_n; ++k)
                                  tmp += shape_values_normal[k * n_q + r] * r_p[k];

                                s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                }
              }

              // ====================================================
              // PHASE 3: Evaluate gradients at quadrature nodes
              // ====================================================

              {
                // 1. evaluate gradients in reference space and multiply by stiffness geometric
                // tensor
                {
                  constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      //  Base offset for the current element's geometric factors
                      const int e_offset =
                        eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
                        e * symmetric_tensor_dimension * n_q_total;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = s_uq_0[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = s_uq_1[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number d_G[dim][dim];
                          Number qr[dim];
                          Number qs[dim];

                          for (int p = 0; p < n_q; ++p)
                            {
                              // Load stiffness geometric tensor
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  qr[d1] = 0;
                                  qs[d1] = 0;
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_stiffness[e_offset + index * n_q_total +
                                                                   q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                }

                              // Multiply by D
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];

                                  qs[0] += r_q[n] * s_uq_0[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] * s_uq_1[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
                              const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

                              s_duq_0[idx0] = qr[0] * d_G[0][0] + qs[0] * d_G[1][0];
                              s_duq_0[idx1] = qr[0] * d_G[0][1] + qs[0] * d_G[1][1];

                              s_duq_1[idx0] = qr[1] * d_G[0][0] + qs[1] * d_G[1][0];
                              s_duq_1[idx1] = qr[1] * d_G[0][1] + qs[1] * d_G[1][1];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          for (int n = 0; n < n_q; ++n)

                            {
                              r_p0[n] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];
                              r_p1[n] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];
                              r_p2[n] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                              r_r[n] = co_shape_gradients[n * n_q + r];
                            }

                          Number d_G[dim][dim];
                          Number qr[dim];
                          Number qs[dim];
                          Number qt[dim];

                          for (int p = 0; p < n_q; ++p)
                            {
                              // Load stiffness geometric tensor
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  qr[d1] = 0;
                                  qs[d1] = 0;
                                  qt[d1] = 0;
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_stiffness[e_offset + index * n_q_total +
                                                                   r * n_q * n_q + q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                }
                              // Multiply by D
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] +=
                                    r_q[n] * s_uq_0[e * n_q_total + r * n_q * n_q + n * n_q + p];
                                  qs[1] +=
                                    r_q[n] * s_uq_1[e * n_q_total + r * n_q * n_q + n * n_q + p];
                                  qs[2] +=
                                    r_q[n] * s_uq_2[e * n_q_total + r * n_q * n_q + n * n_q + p];

                                  qt[0] +=
                                    r_r[n] * s_uq_0[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                  qt[1] +=
                                    r_r[n] * s_uq_1[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                  qt[2] +=
                                    r_r[n] * s_uq_2[e * n_q_total + n * n_q * n_q + q * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx2 =
                                e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q + p;

                              s_duq_0[idx0] =
                                qr[0] * d_G[0][0] + qs[0] * d_G[1][0] + qt[0] * d_G[2][0];
                              s_duq_0[idx1] =
                                qr[0] * d_G[0][1] + qs[0] * d_G[1][1] + qt[0] * d_G[2][1];
                              s_duq_0[idx2] =
                                qr[0] * d_G[0][2] + qs[0] * d_G[1][2] + qt[0] * d_G[2][2];

                              s_duq_1[idx0] =
                                qr[1] * d_G[0][0] + qs[1] * d_G[1][0] + qt[1] * d_G[2][0];
                              s_duq_1[idx1] =
                                qr[1] * d_G[0][1] + qs[1] * d_G[1][1] + qt[1] * d_G[2][1];
                              s_duq_1[idx2] =
                                qr[1] * d_G[0][2] + qs[1] * d_G[1][2] + qt[1] * d_G[2][2];

                              s_duq_2[idx0] =
                                qr[2] * d_G[0][0] + qs[2] * d_G[1][0] + qt[2] * d_G[2][0];
                              s_duq_2[idx1] =
                                qr[2] * d_G[0][1] + qs[2] * d_G[1][1] + qt[2] * d_G[2][1];
                              s_duq_2[idx2] =
                                qr[2] * d_G[0][2] + qs[2] * d_G[1][2] + qt[2] * d_G[2][2];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // 2. multiply by the mass geometric tensor
                {
                  constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      //  Base offset for the current element's geometric factors
                      const int e_offset =
                        eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
                        e * symmetric_tensor_dimension * n_q_total;

                      Number d_G[dim][dim];
                      Number qr[dim];
                      Number qs[dim];

                      Number u[dim];

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          for (int p = 0; p < n_q; ++p)
                            {
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_mass[e_offset + index * n_q_total +
                                                              q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }

                                  qr[d1] =
                                    factor_laplace *
                                    s_duq_0[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
                                  qs[d1] =
                                    factor_laplace *
                                    s_duq_1[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
                                }

                              u[0] = factor_mass * s_uq_0[e * n_q_total + q * n_q + p];
                              u[1] = factor_mass * s_uq_1[e * n_q_total + q * n_q + p];

                              const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
                              const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

                              s_duq_0[idx0] = d_G[0][0] * qr[0] + d_G[0][1] * qs[0];
                              s_duq_0[idx1] = d_G[0][0] * qr[1] + d_G[0][1] * qs[1];

                              s_duq_1[idx0] = d_G[1][0] * qr[0] + d_G[1][1] * qs[0];
                              s_duq_1[idx1] = d_G[1][0] * qr[1] + d_G[1][1] * qs[1];

                              // also apply mass tensor to the value itself
                              s_uq_0[e * n_q_total + q * n_q + p] =
                                d_G[0][0] * u[0] + d_G[0][1] * u[1];
                              s_uq_1[e * n_q_total + q * n_q + p] =
                                d_G[1][0] * u[0] + d_G[1][1] * u[1];
                            }
                        }

                      else if (dim == 3)
                        {
                          Number qt[dim];

                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          for (int p = 0; p < n_q; ++p)
                            {
                              int index = 0;
                              for (int d1 = 0; d1 < dim; ++d1)
                                {
                                  for (int d2 = d1; d2 < dim; ++d2)
                                    {
                                      d_G[d1][d2] =
                                        geometric_tensor_mass[e_offset + index * n_q_total +
                                                              r * n_q * n_q + q * n_q + p];
                                      if (d2 != d1)
                                        d_G[d2][d1] = d_G[d1][d2]; // symmetric
                                      ++index;
                                    }
                                  qr[d1] =
                                    factor_laplace * s_duq_0[e * dim * n_q_total + d1 * n_q_total +
                                                             r * n_q * n_q + q * n_q + p];
                                  qs[d1] =
                                    factor_laplace * s_duq_1[e * dim * n_q_total + d1 * n_q_total +
                                                             r * n_q * n_q + q * n_q + p];
                                  qt[d1] =
                                    factor_laplace * s_duq_2[e * dim * n_q_total + d1 * n_q_total +
                                                             r * n_q * n_q + q * n_q + p];
                                }

                              u[0] =
                                factor_mass * s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p];
                              u[1] =
                                factor_mass * s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p];
                              u[2] =
                                factor_mass * s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p];

                              const int idx0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q + p;
                              const int idx2 =
                                e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q + p;

                              s_duq_0[idx0] =
                                d_G[0][0] * qr[0] + d_G[0][1] * qs[0] + d_G[0][2] * qt[0];
                              s_duq_0[idx1] =
                                d_G[0][0] * qr[1] + d_G[0][1] * qs[1] + d_G[0][2] * qt[1];
                              s_duq_0[idx2] =
                                d_G[0][0] * qr[2] + d_G[0][1] * qs[2] + d_G[0][2] * qt[2];

                              s_duq_1[idx0] =
                                d_G[1][0] * qr[0] + d_G[1][1] * qs[0] + d_G[1][2] * qt[0];
                              s_duq_1[idx1] =
                                d_G[1][0] * qr[1] + d_G[1][1] * qs[1] + d_G[1][2] * qt[1];
                              s_duq_1[idx2] =
                                d_G[1][0] * qr[2] + d_G[1][1] * qs[2] + d_G[1][2] * qt[2];

                              s_duq_2[idx0] =
                                d_G[2][0] * qr[0] + d_G[2][1] * qs[0] + d_G[2][2] * qt[0];
                              s_duq_2[idx1] =
                                d_G[2][0] * qr[1] + d_G[2][1] * qs[1] + d_G[2][2] * qt[1];
                              s_duq_2[idx2] =
                                d_G[2][0] * qr[2] + d_G[2][1] * qs[2] + d_G[2][2] * qt[2];

                              // also apply mass tensor to the value itself
                              s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p] =
                                d_G[0][0] * u[0] + d_G[0][1] * u[1] + d_G[0][2] * u[2];
                              s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p] =
                                d_G[1][0] * u[0] + d_G[1][1] * u[1] + d_G[1][2] * u[2];
                              s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p] =
                                d_G[2][0] * u[0] + d_G[2][1] * u[1] + d_G[2][2] * u[2];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // 3. integrate, i.e apply D^T
                {
                  constexpr int co_dimension_size = Utilities::pow(n_q, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if constexpr (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < n_q; ++n)
                            {
                              const int idx_0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + n;

                              r_p0[n] = s_duq_0[idx_0];
                              r_p1[n] = s_duq_1[idx_0];

                              r_q[n] = co_shape_gradients[q * n_q + n];
                            }

                          for (int p = 0; p < n_q; ++p)
                            {
                              Number tmp0 = 0, tmp1 = 0;

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
                                  tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_1 =
                                    e * dim * n_q_total + 1 * n_q_total + n * n_q + p;
                                  tmp0 += s_duq_0[idx_1] * r_q[n];
                                  tmp1 += s_duq_1[idx_1] * r_q[n];
                                }

                              s_uq_0[e * n_q_total + q * n_q + p] += tmp0;
                              s_uq_1[e * n_q_total + q * n_q + p] += tmp1;
                            }
                        }
                      else if constexpr (dim == 3)
                        {
                          const int q = (tid % co_dimension_size) / n_q;
                          const int r = tid % n_q;

                          // copy to register
                          for (int n = 0; n < n_q; ++n)
                            {
                              const int idx_0 =
                                e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q + n;

                              r_p0[n] = s_duq_0[idx_0];
                              r_p1[n] = s_duq_1[idx_0];
                              r_p2[n] = s_duq_2[idx_0];

                              r_q[n] = co_shape_gradients[q * n_q + n];
                              r_r[n] = co_shape_gradients[r * n_q + n];
                            }

                          for (int p = 0; p < n_q; ++p)
                            {
                              Number tmp0 = 0, tmp1 = 0, tmp2 = 0;

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
                                  tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
                                  tmp2 += r_p2[n] * co_shape_gradients[p * n_q + n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_1 = e * dim * n_q_total + 1 * n_q_total +
                                                    r * n_q * n_q + n * n_q + p;

                                  tmp0 += s_duq_0[idx_1] * r_q[n];
                                  tmp1 += s_duq_1[idx_1] * r_q[n];
                                  tmp2 += s_duq_2[idx_1] * r_q[n];
                                }

                              for (unsigned int n = 0; n < n_q; ++n)
                                {
                                  const int idx_2 = e * dim * n_q_total + 2 * n_q_total +
                                                    n * n_q * n_q + q * n_q + p;

                                  tmp0 += s_duq_0[idx_2] * r_r[n];
                                  tmp1 += s_duq_1[idx_2] * r_r[n];
                                  tmp2 += s_duq_2[idx_2] * r_r[n];
                                }

                              s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp0;
                              s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp1;
                              s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp2;
                            }
                        }
                    }
                }
                team_member.team_barrier();
              }


              // ====================================================
              // PHASE 4: Project back to Nodes
              // ====================================================
              {
                // ------------------------ Component 0 (x-direction) ------------------------
                // x is normal (basis_n), y and z are tangent (basis_t)
                {
                  // component 0 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 0 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_0[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 0 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_t : n_t * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_n; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_normal[i * n_q + p] * r_p[p];

                                s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 1 (y-direction) ------------------------
                // y is normal (basis_n), x and z are tangent (basis_t)
                {
                  // component 1 in z direction
                  if constexpr (dim == 3)
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_t; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_tangent[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  // component 1 in y direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_uq_1[e * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int p = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_n; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_normal[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // component 1 in x direction
                  {
                    constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if constexpr (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                        else if constexpr (dim == 3)
                          {
                            const int j = (tid % co_dimension_size) / n_t;
                            const int k = tid % n_t;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // ------------------------ Component 2 (z-direction) ------------------------
                // z is normal (basis_n), x and y are tangent (basis_t)
                if constexpr (dim == 3)
                  {
                    // component 2 in z direction
                    {
                      constexpr int co_dimension_size = n_q * n_q;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int p = (tid % co_dimension_size) / n_q;
                          const int q = tid % n_q;

                          for (int r = 0; r < n_q; ++r)
                            r_p[r] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

                          for (int k = 0; k < n_n; ++k)
                            {
                              Number tmp = 0;
                              for (int r = 0; r < n_q; ++r)
                                tmp += shape_values_normal[k * n_q + r] * r_p[r];

                              s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                    // component 2 in y direction
                    {
                      constexpr int co_dimension_size = n_q * n_n;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          {
                            const int p = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int q = 0; q < n_q; ++q)
                              r_p[q] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p];

                            for (int j = 0; j < n_t; ++j)
                              {
                                Number tmp = 0;
                                for (int q = 0; q < n_q; ++q)
                                  tmp += shape_values_tangent[j * n_q + q] * r_p[q];

                                s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
                              }
                          }
                          team_member.team_barrier();
                        }

                      // component 2 in x direction
                      {
                        constexpr int co_dimension_size = n_t * n_n;

                        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                             tid += blockSize)
                          {
                            const int e = tid / co_dimension_size;

                            const int j = (tid % co_dimension_size) / n_n;
                            const int k = tid % n_n;

                            for (int p = 0; p < n_q; ++p)
                              r_p[p] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p];

                            for (int i = 0; i < n_t; ++i)
                              {
                                Number tmp = 0;
                                for (int p = 0; p < n_q; ++p)
                                  tmp += shape_values_tangent[i * n_q + p] * r_p[p];

                                s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i] = tmp;
                              }
                          }
                        team_member.team_barrier();
                      }
                    }
                  }
              }

              // ====================================================
              // PHASE 5: Write the results to the global L vector.
              // ====================================================

              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    {
                      const unsigned int dof_x =
                        dof_indices(0 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_x != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_x], s_uq_0[tid]);
                    }
                    {
                      const unsigned int dof_y =
                        dof_indices(1 * n_dofs_per_component + local_dof_index_1d, global_cell_id);

                      if (dof_y != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof_y], s_uq_1[tid]);
                    }

                    if constexpr (dim > 2)
                      {
                        const unsigned int dof_z =
                          dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);

                        if (dof_z != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&vector_out[dof_z], s_uq_2[tid]);
                      }
                  }
                team_member.team_barrier();
              }



              team_member.team_barrier();
              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }


    // Combined dof-count extent of the tensor-product axes with role > `dir`,
    // for a Raviart-Thomas component whose normal axis is `normal_dir` (extent
    // `n_normal`); the other axes have extent `n_tangent`. This is the
    // `n_blocks2` argument to Custom::Parallel::apply_anisotropic().
    constexpr int
    rt_perp_dof_extent(int space_dim, int dir, int normal_dir, int n_normal, int n_tangent)
    {
      int result = 1;
      for (int axis = dir + 1; axis < space_dim; ++axis)
        result *= (axis == normal_dir) ? n_normal : n_tangent;
      return result;
    }


    // Buffer index of the (face_in_batch, side, component) trace used by
    // compute_inner_faces, one n_q_face-sized block per (side, component) pair.
    // A plain DEAL_II_HOST_DEVICE function rather than a local lambda: nvcc
    // does not allow defining an extended __host__ __device__ lambda inside
    // another one, which a KOKKOS_LAMBDA-captured local lambda would be.
    template <int dim>
    DEAL_II_HOST_DEVICE inline int
    inner_face_slot_index(int face_in_batch, int side, int component)
    {
      return (face_in_batch * 2 + side) * dim + component;
    }

    // Same, single-sided (no `side`) -- used by compute_boundary_faces.
    template <int dim>
    DEAL_II_HOST_DEVICE inline int
    boundary_face_slot_index(int face_in_batch, int component)
    {
      return face_in_batch * dim + component;
    }

    // tangent_index-th axis that is not normal_dir; dim == 2 has a single
    // tangent axis, dim == 3 looks it up in a lookup table. Shared by
    // compute_inner_faces and compute_boundary_faces.
    template <int dim>
    DEAL_II_HOST_DEVICE inline int
    face_tangent_direction(int normal_dir, int tang_index)
    {
      if constexpr (dim == 2)
        return 1 - normal_dir;
      else
        {
          constexpr int lookup_tangents_3d[3][2] = {
            {1, 2}, // normal_dir == 0
            {0, 2}, // normal_dir == 1
            {0, 1}  // normal_dir == 2
          };
          return lookup_tangents_3d[normal_dir][tang_index];
        }
    }


    template <int dim, int n_t, int n_q, typename Number>
    void
    compute_cell(
      const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
      const DeviceView<Number>                   shape_gradients_collocation,
      const DeviceView<Number>                   geometric_tensor_mass,
      const DeviceView<Number>                   geometric_tensor_stiffness,
      const DeviceView<Number>                   vector_in,
      DeviceView<Number>                         vector_out,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
                                                                    interpolate_quad_to_boundary,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space> face_values_at_quads,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space>
                           face_normal_derivatives_at_quads,
      const DoFIndicesView dof_indices,
      // neighbor_cells(2*d+side, cell) == invalid_unsigned_int marks a domain
      // (Neumann) boundary face -- skip interpolating it, since its contribution
      // to distribute_face_to_global is zero.
      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> neighbor_cells,
      const unsigned int                                                      n_cells,
      const bool                                                              interpolate_to_faces,
      const Number       factor_mass       = Number(1),
      const Number       factor_laplace    = Number(1),
      const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)

    {
      using Custom::Parallel::apply_anisotropic;
      using Custom::Parallel::evaluate_vector_gradients_and_multiply_symmetric_tensor;
      using Custom::Parallel::integrate_vector_gradients;

      constexpr int n_components = dim;

      if (n_cells == 0)
        return;

      AssertThrow(dim > 1, ExcNotImplemented());

      static_assert(n_t > 1, "Degree 0 not supported");

      AssertThrow(n_q > n_t, ExcNotImplemented());

      constexpr int n_n = n_t + 1;

      constexpr int n_q_total = Utilities::pow(n_q, dim);

      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);
      const int     nelmt                = n_cells;

      // const size_t shmemPerBlock =
      //   Kokkos::TeamPolicy<>::scratch_size_max(0); // maximum shared memory size per thread block

      int shmemPerBlock = 10800; // total shared memory used per block (KB)

      const int nelmtPerBatch =
        (n_cells_per_batch == numbers::invalid_unsigned_int) ?
          // at least 1: for large degree/dim the per-cell scratch footprint can
          // exceed the shmemPerBlock budget on its own, flooring this to 0
          // otherwise and dividing by zero below.
          std::max(1,
                  shmemPerBlock / (n_components * (dim + 1) * n_q_total) /
                    static_cast<int>(sizeof(Number))) :
          n_cells_per_batch;

      const int numBlocks = (n_blocks == numbers::invalid_unsigned_int) ?
                              std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                              n_blocks;

      const int threadsPerBlock =
        (threads_per_block == numbers::invalid_unsigned_int) ?
          std::min(std::max(1, nelmtPerBatch) * Utilities::pow(n_q, dim - 1), 512) :
          threads_per_block;


      const unsigned int ssize = n_n * n_q   // normal shape values
                                 + n_t * n_q // tangent shape values
                                 + n_q * n_q // shape gradients at collocation points
                                 + n_components * nelmtPerBatch * n_q_total        // values
                                 + n_components * dim * nelmtPerBatch * n_q_total; // gradients


      unsigned int shmem_size = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *shape_values_normal  = scratch;
          Number *shape_values_tangent = shape_values_normal + n_n * n_q;
          Number *co_shape_gradients   = shape_values_tangent + n_t * n_q;

          // Per-component staging buffers. s_uq[c] holds nelmtPerBatch * n_q_total
          // values (one per quadrature point), s_duq[c] the dim reference-gradient
          // components (dim * n_q_total per cell). Interleaved uq_0, duq_0, uq_1,
          // duq_1, ... to match the ssize / shmem_size computation above.
          Number *s_uq[n_components];
          Number *s_duq[n_components];
          {
            Number *ptr = co_shape_gradients + n_q * n_q;
            for (int c = 0; c < n_components; ++c)
              {
                s_uq[c] = ptr;
                ptr += nelmtPerBatch * n_q_total;
                s_duq[c] = ptr;
                ptr += nelmtPerBatch * n_q_total * dim;
              }
          }

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();


          // copy to shared memory
          {
            for (int tid = threadIdx; tid < n_n * n_q; tid += blockSize)
              {
                shape_values_normal[tid] = shape_values_info[0][tid];
              }
            for (int tid = threadIdx; tid < n_t * n_q; tid += blockSize)
              {
                shape_values_tangent[tid] = shape_values_info[1][tid];
              }
            for (int tid = threadIdx; tid < n_q * n_q; tid += blockSize)
              {
                co_shape_gradients[tid] = shape_gradients_collocation[tid];
              }
            team_member.team_barrier();
          }

          // element batch iteration
          int eb = team_member.league_rank();

          while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              // current nelmtPerBatch (edge case, last batch size can be less)
              const int c_nelmtPerBatch = Kokkos::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

              // ====================================================
              // PHASE 1: Read from global L vector per component
              // ====================================================
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;
                    const int global_cell_id     = eb * nelmtPerBatch + e;

                    for (int comp = 0; comp < n_components; ++comp)
                      {
                        const unsigned int dof =
                          dof_indices(comp * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);
                        s_uq[comp][tid] =
                          (dof != numbers::invalid_unsigned_int) ? vector_in[dof] : Number(0);
                      }
                  }
                team_member.team_barrier();
              }

              // ====================================================
              // PHASE 2: Interpolate to quadrature nodes
              // ====================================================
              {
                // Interpolate dofs -> quadrature points: one anisotropic 1D
                // sum-factorization sweep per (component, direction). For
                // component c the axis c carries the normal shape values
                // (n_n x n_q), the others the tangent ones (n_t x n_q);
                // contract_over_rows = true. Buffers ping-pong s_uq[c] <-> the
                // s_duq scratch so that the final quad values land in s_uq[c].

                const Number *const mat_n = shape_values_normal;
                const Number *const mat_t = shape_values_tangent;
                const int           nb    = c_nelmtPerBatch;

                // component 0 (normal axis 0)
                apply_anisotropic<dim,
                                  0,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 0, n_n, n_t),
                                  true,
                                  false,
                                  Number>(
                  team_member, mat_n, s_uq[0], s_duq[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 0, n_n, n_t),
                                  true,
                                  false,
                                  Number>(team_member,
                                          mat_t,
                                          s_duq[0],
                                          (dim == 2) ? s_uq[0] : s_duq[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 0, n_n, n_t),
                                    true,
                                    false,
                                    Number>(
                    team_member, mat_t, s_duq[1], s_uq[0], nb, threadIdx, blockSize);

                // component 1 (normal axis 1)
                apply_anisotropic<dim,
                                  0,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 1, n_n, n_t),
                                  true,
                                  false,
                                  Number>(
                  team_member, mat_t, s_uq[1], s_duq[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 1, n_n, n_t),
                                  true,
                                  false,
                                  Number>(team_member,
                                          mat_n,
                                          s_duq[0],
                                          (dim == 2) ? s_uq[1] : s_duq[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 1, n_n, n_t),
                                    true,
                                    false,
                                    Number>(
                    team_member, mat_t, s_duq[1], s_uq[1], nb, threadIdx, blockSize);

                // component 2 (normal axis 2)
                if constexpr (dim == 3)
                  {
                    apply_anisotropic<dim,
                                      0,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 0, 2, n_n, n_t),
                                      true,
                                      false,
                                      Number>(
                      team_member, mat_t, s_uq[2], s_duq[0], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      1,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 1, 2, n_n, n_t),
                                      true,
                                      false,
                                      Number>(
                      team_member, mat_t, s_duq[0], s_duq[1], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      2,
                                      n_n,
                                      n_q,
                                      rt_perp_dof_extent(dim, 2, 2, n_n, n_t),
                                      true,
                                      false,
                                      Number>(
                      team_member, mat_n, s_duq[1], s_uq[2], nb, threadIdx, blockSize);
                  }
              }

              if (interpolate_to_faces)
                {
                  // interpolate the value and the reference normal derivative of
                  // every component to the n_q^(dim-1) quadrature points of each
                  // of the 2*dim faces (as the DG Laplace cell kernel does).
                  constexpr int n_q_face       = Utilities::pow(n_q, dim - 1);
                  constexpr int axis_stride[3] = {1, n_q, n_q * n_q};

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * n_q_face; tid += blockSize)
                    {
                      const int e              = tid / n_q_face;
                      const int m              = tid % n_q_face;
                      const int global_cell_id = eb * nelmtPerBatch + e;
                      const int e_val          = e * n_q_total;

                      for (int d = 0; d < dim; ++d) // face-normal direction
                        {
                          // Neumann boundary faces contribute nothing in
                          // distribute_face_to_global -- skip interpolating them.
                          const bool skip0 = neighbor_cells(2 * d + 0, global_cell_id) ==
                                             numbers::invalid_unsigned_int;
                          const bool skip1 = neighbor_cells(2 * d + 1, global_cell_id) ==
                                             numbers::invalid_unsigned_int;
                          if (skip0 && skip1)
                            continue;

                          // base offset of the fiber running along axis d for the
                          // face quadrature point m (the other dim-1 indices)
                          int base = 0;
                          int mm   = m;
                          for (int a = 0; a < dim; ++a)
                            if (a != d)
                              {
                                base += (mm % n_q) * axis_stride[a];
                                mm /= n_q;
                              }

                          for (int comp = 0; comp < n_components; ++comp)
                            {
                              Number v0 = 0, v1 = 0, dn0 = 0, dn1 = 0;
                              for (int n = 0; n < n_q; ++n)
                                {
                                  const Number u = s_uq[comp][e_val + base + n * axis_stride[d]];
                                  if (!skip0)
                                    {
                                      v0 += interpolate_quad_to_boundary(0, n, 0) * u;
                                      dn0 += interpolate_quad_to_boundary(1, n, 0) * u;
                                    }
                                  if (!skip1)
                                    {
                                      v1 += interpolate_quad_to_boundary(0, n, 1) * u;
                                      dn1 += interpolate_quad_to_boundary(1, n, 1) * u;
                                    }
                                }
                              if (!skip0)
                                {
                                  face_values_at_quads(m, 2 * d + 0, comp, global_cell_id) = v0;
                                  face_normal_derivatives_at_quads(m,
                                                                   2 * d + 0,
                                                                   comp,
                                                                   global_cell_id)         = dn0;
                                }
                              if (!skip1)
                                {
                                  face_values_at_quads(m, 2 * d + 1, comp, global_cell_id) = v1;
                                  face_normal_derivatives_at_quads(m,
                                                                   2 * d + 1,
                                                                   comp,
                                                                   global_cell_id)         = dn1;
                                }
                            }
                        }
                    }
                }

              // ====================================================
              // PHASE 3: Evaluate gradients at quadrature nodes
              // ====================================================

              {
                constexpr int n_sym    = (dim * (dim + 1)) / 2;
                constexpr int n_planes = Utilities::pow(n_q, dim - 1);

                // 1. reference-space collocation gradients of every component,
                //    multiplied pointwise by the (symmetric) stiffness metric.
                evaluate_vector_gradients_and_multiply_symmetric_tensor<dim,
                                                                        n_components,
                                                                        n_q,
                                                                        Number>(
                  team_member,
                  co_shape_gradients,
                  geometric_tensor_stiffness.data() + eb * nelmtPerBatch * n_sym * n_q_total,
                  s_uq,
                  s_duq,
                  c_nelmtPerBatch,
                  threadIdx,
                  blockSize);

                // 2. Piola coupling: mix the components with the (symmetric)
                //    mass metric and apply the mass/Laplace coefficients, both
                //    to the values (in s_uq) and to the gradients (in s_duq).
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_planes; tid += blockSize)
                  {
                    const int e     = tid / n_planes;
                    const int plane = tid % n_planes;
                    const int a1    = plane % n_q;
                    const int a2    = (dim == 3) ? (plane / n_q) : 0;

                    const int e_val  = e * n_q_total;
                    const int e_grad = e * dim * n_q_total;
                    const int e_ten  = (eb * nelmtPerBatch + e) * n_sym * n_q_total;

                    for (int a0 = 0; a0 < n_q; ++a0)
                      {
                        const int q = a0 + a1 * n_q + a2 * n_q * n_q;

                        Number G[dim][dim];
                        for (int d0 = 0, s = 0; d0 < dim; ++d0)
                          for (int d1 = d0; d1 < dim; ++d1, ++s)
                            {
                              const Number g = geometric_tensor_mass[e_ten + s * n_q_total + q];
                              G[d0][d1]      = g;
                              G[d1][d0]      = g;
                            }

                        Number val_in[n_components];
                        Number grad_in[n_components][dim];
                        for (int c = 0; c < n_components; ++c)
                          {
                            val_in[c] = factor_mass * s_uq[c][e_val + q];
                            for (int d = 0; d < dim; ++d)
                              grad_in[c][d] = factor_laplace * s_duq[c][e_grad + d * n_q_total + q];
                          }

                        for (int c = 0; c < n_components; ++c)
                          {
                            Number val_out = 0;
                            Number grad_out[dim];
                            for (int d = 0; d < dim; ++d)
                              grad_out[d] = 0;

                            for (int cp = 0; cp < n_components; ++cp)
                              {
                                val_out += G[c][cp] * val_in[cp];
                                for (int d = 0; d < dim; ++d)
                                  grad_out[d] += G[c][cp] * grad_in[cp][d];
                              }

                            s_uq[c][e_val + q] = val_out;
                            for (int d = 0; d < dim; ++d)
                              s_duq[c][e_grad + d * n_q_total + q] = grad_out[d];
                          }
                      }
                  }
                team_member.team_barrier();

                // 3. integrate the gradients (transpose collocation derivative)
                //    and accumulate onto the mass contribution in s_uq.
                integrate_vector_gradients<dim, n_components, n_q, Number>(team_member,
                                                                           co_shape_gradients,
                                                                           s_duq,
                                                                           s_uq,
                                                                           c_nelmtPerBatch,
                                                                           threadIdx,
                                                                           blockSize);
              }


              // ====================================================
              // PHASE 4: Project back to Nodes
              // ====================================================
              {
                // Integrate quadrature values -> dofs: the transpose of PHASE 2.
                // Same matrices, contract_over_rows = false, sweeps run in
                // reverse direction order (dim-1 .. 0). Buffers ping-pong
                // s_uq[c] <-> s_duq scratch, ending back in s_uq[c] in the dof
                // layout PHASE 5 expects.

                const Number *const mat_n = shape_values_normal;
                const Number *const mat_t = shape_values_tangent;
                const int           nb    = c_nelmtPerBatch;

                // component 0 (normal axis 0)
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 0, n_n, n_t),
                                    false,
                                    false,
                                    Number>(
                    team_member, mat_t, s_uq[0], s_duq[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 0, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_t,
                                          (dim == 2) ? s_uq[0] : s_duq[0],
                                          (dim == 2) ? s_duq[0] : s_duq[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                apply_anisotropic<dim,
                                  0,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 0, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_n,
                                          (dim == 2) ? s_duq[0] : s_duq[1],
                                          s_uq[0],
                                          nb,
                                          threadIdx,
                                          blockSize);

                // component 1 (normal axis 1)
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 1, n_n, n_t),
                                    false,
                                    false,
                                    Number>(
                    team_member, mat_t, s_uq[1], s_duq[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 1, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_n,
                                          (dim == 2) ? s_uq[1] : s_duq[0],
                                          (dim == 2) ? s_duq[0] : s_duq[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                apply_anisotropic<dim,
                                  0,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 1, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_t,
                                          (dim == 2) ? s_duq[0] : s_duq[1],
                                          s_uq[1],
                                          nb,
                                          threadIdx,
                                          blockSize);

                // component 2 (normal axis 2)
                if constexpr (dim == 3)
                  {
                    apply_anisotropic<dim,
                                      2,
                                      n_n,
                                      n_q,
                                      rt_perp_dof_extent(dim, 2, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_n, s_uq[2], s_duq[0], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      1,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 1, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_t, s_duq[0], s_duq[1], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      0,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 0, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_t, s_duq[1], s_uq[2], nb, threadIdx, blockSize);
                  }
              }

              // ====================================================
              // PHASE 5: Write the results to the global L vector.
              // ====================================================

              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                     tid += blockSize)
                  {
                    const int e                  = tid / n_dofs_per_component;
                    const int local_dof_index_1d = tid % n_dofs_per_component;
                    const int global_cell_id     = eb * nelmtPerBatch + e;

                    for (int comp = 0; comp < n_components; ++comp)
                      {
                        const unsigned int dof =
                          dof_indices(comp * n_dofs_per_component + local_dof_index_1d,
                                      global_cell_id);
                        if (dof != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&vector_out[dof], s_uq[comp][tid]);
                      }
                  }
                team_member.team_barrier();
              }



              team_member.team_barrier();
              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }


    // ================================================================
    //  Face kernels (SIPG viscous term), DG-style split.
    //  compute_cell(interpolate_to_faces=true) writes the reference face
    //  value / normal derivative of every component into
    //  face_values_at_quads / face_normal_derivatives_at_quads.
    //  compute_inner_faces / compute_boundary_faces overwrite those slots
    //  in place with the (reference-frame) test-function contributions.
    //  distribute_face_to_global interpolates them back to the cell and
    //  integrates onto the global vector.
    //
    //  Affine cells (general -- both the Piola value map and, via the
    //  tangential reference derivatives computed here, the full physical
    //  normal derivative). The Jacobian-gradient (curvature) terms are the
    //  only thing still dropped, so this is exact on parallelepiped meshes.
    // ================================================================

    template <int dim, int n_t, int n_q, typename Number>
    void
    compute_inner_faces(
      const DeviceView<Number>                                            co_shape_gradients,
      const DeviceView<Number>                                            cell_piola,
      const Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jacobians_times_normal,
      const Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jxw_values,
      const DeviceView<Number>                                            penalty_parameters,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space>       face_values_at_quads,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space>
        face_normal_derivatives_at_quads,
      const Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space> face_info,
      const unsigned int                                                        n_inner_faces,
      const Number       factor_laplace    = Number(1),
      const unsigned int n_faces_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      using Custom::Parallel::apply;

      if (n_inner_faces == 0)
        return;

      constexpr int n_components = dim;
      constexpr int n_q_face     = Utilities::pow(n_q, dim - 1);
      constexpr int n_face_dofs  = n_components * 2; // (component, side) tensors per face

      const int nelmt = n_inner_faces;

      // per team scratch is n_q*n_q (fixed) + (2 + dim) * nelmtPerBatch * n_face_dofs *
      // n_q_face (see shmem_size below) -- size nelmtPerBatch off of that per-face term
      // against a shared-memory budget, same idea as compute_cell.
      const int shmemPerBlock = 10800; // total shared memory used per block (bytes)

      const int nelmtPerBatch =
        (n_faces_per_batch == numbers::invalid_unsigned_int) ?
          std::max(1,
                  shmemPerBlock / (n_face_dofs * n_q_face * (2 + dim)) /
                    static_cast<int>(sizeof(Number))) :
          static_cast<int>(n_faces_per_batch);
      const int numBlocks       = (n_blocks == numbers::invalid_unsigned_int) ?
                                    std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                                    static_cast<int>(n_blocks);
      const int threadsPerBlock = (threads_per_block == numbers::invalid_unsigned_int) ?
                                    std::min(nelmtPerBatch * n_q_face, 256) :
                                    static_cast<int>(threads_per_block);

      // per team scratch: collocation-gradient matrix (n_q*n_q)
      //                 + face values buffer            (buffer_size)
      //                 + apply() output buffer         (buffer_size)
      //                 + face gradients buffer         (dim * buffer_size, one slab per axis)
      const int          buffer_size = nelmtPerBatch * n_face_dofs * n_q_face;
      const unsigned int ssize       = n_q * n_q + 2 * buffer_size + dim * buffer_size;
      const unsigned int shmem_size  = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          // scratch layout, in order:
          //   s_co_shape_gradients : collocation 1D derivative matrix, n_q * n_q
          //   s_face_values        : per (face_in_batch, side, component) trace, [slot][q_face]
          //   s_temp               : scratch output of apply() (tangential contraction)
          //   s_face_gradients     : dim slabs, slab `axis` holds d/dxi_axis of the trace
          Number *s_co_shape_gradients = scratch;
          Number *s_face_values        = s_co_shape_gradients + n_q * n_q;
          Number *s_temp               = s_face_values + buffer_size;
          Number *s_face_gradients     = s_temp + buffer_size;

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();

          for (int idx = threadIdx; idx < n_q * n_q; idx += blockSize)
            s_co_shape_gradients[idx] = co_shape_gradients[idx];
          team_member.team_barrier();

          int face_batch = team_member.league_rank();
          while (face_batch < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              const int n_faces_this_batch =
                Kokkos::min(nelmtPerBatch, nelmt - face_batch * nelmtPerBatch);

              // ---- step 1: read the face trace values and the normal reference derivative ----
              for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                   tid += blockSize)
                {
                  const int q_face    = tid % n_q_face;
                  const int dof_slot  = tid / n_q_face;
                  const int component = dof_slot % n_components;
                  const int side =
                    (dof_slot / n_components) % 2; // "minus" or "plus" side of the face
                  const int face_in_batch = dof_slot / n_face_dofs;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell       = face_info(face, side == 0 ? 0 : 1);
                  const unsigned int local_face = face_info(face, side == 0 ? 2 : 3);
                  const int          normal_dir = face_info(face, 2) / 2;
                  const int          slot       = inner_face_slot_index<dim>(face_in_batch, side, component);

                  // m = J^{-1} n, component along the normal reference axis
                  const Number m_normal =
                    jacobians_times_normal(face * dim * n_q_face + normal_dir * n_q_face + q_face,
                                           side);

                  s_face_values[slot * n_q_face + q_face] =
                    face_values_at_quads(q_face, local_face, component, cell);
                  s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face] =
                    face_normal_derivatives_at_quads(q_face, local_face, component, cell) *
                    m_normal;
                }
              team_member.team_barrier();

              // ---- step 2: add the tangential reference derivatives of the trace ----
              for (int tang_index = 0; tang_index < dim - 1; ++tang_index)
                {
                  if (tang_index == 0)
                    apply<dim - 1, 0, n_q, n_q, true, false, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_face_values,
                                                                     s_temp,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                  else if constexpr (dim == 3) // tang_index == 1 only exists in 3D
                    apply<dim - 1, 1, n_q, n_q, true, false, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_face_values,
                                                                     s_temp,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);


                  for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                       tid += blockSize)
                    {
                      const int q_face        = tid % n_q_face;
                      const int dof_slot      = tid / n_q_face;
                      const int component     = dof_slot % n_components;
                      const int side          = (dof_slot / n_components) % 2;
                      const int face_in_batch = dof_slot / n_face_dofs;

                      const int face        = face_batch * nelmtPerBatch + face_in_batch;
                      const int normal_dir  = face_info(face, 2) / 2;
                      const int tangent_dir = face_tangent_direction<dim>(normal_dir, tang_index);

                      const int slot = inner_face_slot_index<dim>(face_in_batch, side, component);

                      const Number m_tangent = jacobians_times_normal(
                        face * dim * n_q_face + tangent_dir * n_q_face + q_face, side);

                      s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face] +=
                        s_temp[slot * n_q_face + q_face] * m_tangent;
                    }
                  team_member.team_barrier();
                }

              // ---- step 3: Piola map, SIPG flux, and Piola-transpose of the test contributions
              // ----
              for (int tid = threadIdx; tid < n_faces_this_batch * n_q_face; tid += blockSize)
                {
                  const int q_face        = tid % n_q_face;
                  const int face_in_batch = tid / n_q_face;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell_minus = face_info(face, 0);
                  const unsigned int cell_plus  = face_info(face, 1);
                  const int          normal_dir = face_info(face, 2) / 2;

                  Number piola_minus[dim][dim], piola_plus[dim][dim];
                  for (int i = 0; i < dim; ++i)
                    for (int j = 0; j < dim; ++j)
                      {
                        piola_minus[i][j] = cell_piola[cell_minus * dim * dim + i * dim + j];
                        piola_plus[i][j]  = cell_piola[cell_plus * dim * dim + i * dim + j];
                      }

                  Number m_minus[dim], m_plus[dim];
                  for (int axis = 0; axis < dim; ++axis)
                    {
                      m_minus[axis] =
                        jacobians_times_normal(face * dim * n_q_face + axis * n_q_face + q_face, 0);
                      m_plus[axis] =
                        jacobians_times_normal(face * dim * n_q_face + axis * n_q_face + q_face, 1);
                    }

                  const Number jxw            = jxw_values(face * n_q_face + q_face, 0);
                  const Number penalty        = penalty_parameters[face];
                  const Number laplace_factor = factor_laplace;

                  Number u_ref_minus[n_components], u_ref_plus[n_components];
                  Number dn_ref_minus[n_components], dn_ref_plus[n_components];

                  for (int component = 0; component < n_components; ++component)
                    {
                      const int slot_minus   = inner_face_slot_index<dim>(face_in_batch, 0, component);
                      const int slot_plus    = inner_face_slot_index<dim>(face_in_batch, 1, component);
                      u_ref_minus[component] = s_face_values[slot_minus * n_q_face + q_face];
                      u_ref_plus[component]  = s_face_values[slot_plus * n_q_face + q_face];
                      // (grad_xi u_ref) . m, summed over reference axes already
                      dn_ref_minus[component] =
                        s_face_gradients[normal_dir * buffer_size + slot_minus * n_q_face + q_face];
                      dn_ref_plus[component] =
                        s_face_gradients[normal_dir * buffer_size + slot_plus * n_q_face + q_face];
                    }

                  Number value_flux[n_components], grad_flux[n_components];
                  {
                    Number u_phys_minus[n_components], u_phys_plus[n_components];
                    Number dn_phys_minus[n_components], dn_phys_plus[n_components];
                    for (int component = 0; component < n_components; ++component)
                      {
                        Number acc_u_minus = 0, acc_u_plus = 0, acc_dn_minus = 0, acc_dn_plus = 0;
                        for (int k = 0; k < n_components; ++k)
                          {
                            acc_u_minus += piola_minus[component][k] * u_ref_minus[k];
                            acc_u_plus += piola_plus[component][k] * u_ref_plus[k];
                            acc_dn_minus += piola_minus[component][k] * dn_ref_minus[k];
                            acc_dn_plus += piola_plus[component][k] * dn_ref_plus[k];
                          }
                        u_phys_minus[component]  = acc_u_minus;
                        u_phys_plus[component]   = acc_u_plus;
                        dn_phys_minus[component] = acc_dn_minus;
                        dn_phys_plus[component]  = acc_dn_plus;
                      }
                    for (int component = 0; component < n_components; ++component)
                      {
                        const Number jump = u_phys_minus[component] - u_phys_plus[component];
                        value_flux[component] =
                          jxw * laplace_factor *
                          (penalty * jump -
                           Number(0.5) * (dn_phys_minus[component] + dn_phys_plus[component]));
                        grad_flux[component] = -jxw * laplace_factor * Number(0.5) * jump;
                      }
                  }

                  for (int component = 0; component < n_components; ++component)
                    {
                      Number pt_value_minus = 0, pt_value_plus = 0;
                      Number pt_grad_minus = 0, pt_grad_plus = 0;
                      for (int k = 0; k < n_components; ++k)
                        {
                          pt_value_minus += piola_minus[k][component] * value_flux[k];
                          pt_value_plus += piola_plus[k][component] * value_flux[k];
                          pt_grad_minus += piola_minus[k][component] * grad_flux[k];
                          pt_grad_plus += piola_plus[k][component] * grad_flux[k];
                        }
                      const int slot_minus = inner_face_slot_index<dim>(face_in_batch, 0, component);
                      const int slot_plus  = inner_face_slot_index<dim>(face_in_batch, 1, component);
                      s_face_values[slot_minus * n_q_face + q_face] =
                        pt_value_minus; // [[v]]: minus +
                      s_face_values[slot_plus * n_q_face + q_face] =
                        -pt_value_plus; //        plus  -
                      for (int axis = 0; axis < dim; ++axis)
                        {
                          s_face_gradients[axis * buffer_size + slot_minus * n_q_face + q_face] =
                            pt_grad_minus * m_minus[axis];
                          s_face_gradients[axis * buffer_size + slot_plus * n_q_face + q_face] =
                            pt_grad_plus * m_plus[axis];
                        }
                    }
                }
              team_member.team_barrier();

              // ---- step 4: integrate the tangential test gradients, then write the traces back
              // ---- (mirror of step 2: axes undone in reverse order)
              for (int tang_index = dim - 2; tang_index >= 0; --tang_index)
                {
                  for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                       tid += blockSize)
                    {
                      const int q_face        = tid % n_q_face;
                      const int dof_slot      = tid / n_q_face;
                      const int component     = dof_slot % n_components;
                      const int side          = (dof_slot / n_components) % 2;
                      const int face_in_batch = dof_slot / n_face_dofs;

                      const int face        = face_batch * nelmtPerBatch + face_in_batch;
                      const int normal_dir  = face_info(face, 2) / 2;
                      const int tangent_dir = face_tangent_direction<dim>(normal_dir, tang_index);

                      // move the tangential test-gradient slab into s_temp for apply()
                      const int slot = inner_face_slot_index<dim>(face_in_batch, side, component);
                      s_temp[slot * n_q_face + q_face] =
                        s_face_gradients[tangent_dir * buffer_size + slot * n_q_face + q_face];
                    }
                  team_member.team_barrier();

                  if (tang_index == 0)
                    apply<dim - 1, 0, n_q, n_q, false, true, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_temp,
                                                                     s_face_values,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                  else if constexpr (dim == 3) // tang_index == 1 only exists in 3D
                    apply<dim - 1, 1, n_q, n_q, false, true, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_temp,
                                                                     s_face_values,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                }

              for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                   tid += blockSize)
                {
                  const int q_face        = tid % n_q_face;
                  const int dof_slot      = tid / n_q_face;
                  const int component     = dof_slot % n_components;
                  const int side          = (dof_slot / n_components) % 2;
                  const int face_in_batch = dof_slot / n_face_dofs;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell       = face_info(face, side == 0 ? 0 : 1);
                  const unsigned int local_face = face_info(face, side == 0 ? 2 : 3);
                  const int          normal_dir = face_info(face, 2) / 2;
                  const int          slot       = inner_face_slot_index<dim>(face_in_batch, side, component);

                  face_values_at_quads(q_face, local_face, component, cell) =
                    s_face_values[slot * n_q_face + q_face];
                  face_normal_derivatives_at_quads(q_face, local_face, component, cell) =
                    s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face];
                }
              team_member.team_barrier();

              face_batch += team_member.league_size();
            }
        });
      Kokkos::fence();
    }


    // Dirichlet boundary faces (boundary_id != 0; g = 0 for now). Neumann boundary
    // faces (boundary_id == 0) need no kernel at all: compute_cell and
    // distribute_face_to_global already skip them via neighbor_cells, since their
    // contribution is exactly zero. face_info here only lists Dirichlet faces (see
    // compute_face_info), so every face this kernel touches gets the real treatment
    // below -- no per-face boundary_id branch needed.
    //
    // Single-sided mirror of compute_inner_faces: the "minus" state is the real
    // interior trace (read back from what compute_cell wrote), and the "plus"
    // (ghost) state is synthesized in PHYSICAL space from the homogeneous Dirichlet
    // condition once the Piola map has been applied, mirroring around g = 0:
    //   u_phys_plus  = -u_phys_minus   (so the SIPG average {{u}} = 0 = g)
    //   dn_phys_plus =  dn_phys_minus  (so the SIPG average {{du/dn}} = dn_phys_minus)
    // Substituting m_plus = m_minus and piola_plus = piola_minus (same cell) into
    // compute_inner_faces' step-3 formula with this mirror reduces to the formula
    // used below; jacobians_times_normal_boundary_face / jxw_boundary_face /
    // penalty_parameters_boundary_face already store only the one (interior) copy
    // of that per-face geometry (penalty_parameters_boundary_face already carries
    // the factor of 2 that comes from m_minus.norm() + m_plus.norm() collapsing to
    // 2 * m_minus.norm()).
    template <int dim, int n_t, int n_q, typename Number>
    void
    compute_boundary_faces(
      const DeviceView<Number> co_shape_gradients,
      const DeviceView<Number> cell_piola,
      const DeviceView<Number> jacobians_times_normal_boundary_face,
      const DeviceView<Number> jxw_boundary_face,
      const DeviceView<Number> penalty_parameters_boundary_face,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space> face_values_at_quads,
      Kokkos::View<Number ****, MemorySpace::Default::kokkos_space>
        face_normal_derivatives_at_quads,
      const Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space> face_info,
      const unsigned int                                                        n_boundary_faces,
      const Number       factor_laplace    = Number(1),
      const unsigned int n_faces_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      using Custom::Parallel::apply;

      if (n_boundary_faces == 0)
        return;

      constexpr int n_components = dim;
      constexpr int n_q_face     = Utilities::pow(n_q, dim - 1);
      constexpr int n_face_dofs  = n_components; // one side only -- (component) tensors per face

      const int nelmt = n_boundary_faces;

      // per team scratch is n_q*n_q (fixed) + (2 + dim) * nelmtPerBatch * n_face_dofs *
      // n_q_face (see shmem_size below) -- size nelmtPerBatch off of that per-face term
      // against a shared-memory budget, same idea as compute_cell.
      const int shmemPerBlock = 10800; // total shared memory used per block (bytes)

      const int nelmtPerBatch =
        (n_faces_per_batch == numbers::invalid_unsigned_int) ?
          std::max(1,
                  shmemPerBlock / (n_face_dofs * n_q_face * (2 + dim)) /
                    static_cast<int>(sizeof(Number))) :
          static_cast<int>(n_faces_per_batch);
      const int numBlocks       = (n_blocks == numbers::invalid_unsigned_int) ?
                                    std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                                    static_cast<int>(n_blocks);
      const int threadsPerBlock = (threads_per_block == numbers::invalid_unsigned_int) ?
                                    std::min(nelmtPerBatch * n_q_face, 256) :
                                    static_cast<int>(threads_per_block);

      // per team scratch: collocation-gradient matrix (n_q*n_q)
      //                 + face values buffer            (buffer_size)
      //                 + apply() output buffer         (buffer_size)
      //                 + face gradients buffer         (dim * buffer_size, one slab per axis)
      const int          buffer_size = nelmtPerBatch * n_face_dofs * n_q_face;
      const unsigned int ssize       = n_q * n_q + 2 * buffer_size + dim * buffer_size;
      const unsigned int shmem_size  = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *s_co_shape_gradients = scratch;
          Number *s_face_values        = s_co_shape_gradients + n_q * n_q;
          Number *s_temp               = s_face_values + buffer_size;
          Number *s_face_gradients     = s_temp + buffer_size;

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();

          for (int idx = threadIdx; idx < n_q * n_q; idx += blockSize)
            s_co_shape_gradients[idx] = co_shape_gradients[idx];
          team_member.team_barrier();

          int face_batch = team_member.league_rank();
          while (face_batch < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              const int n_faces_this_batch =
                Kokkos::min(nelmtPerBatch, nelmt - face_batch * nelmtPerBatch);

              // ---- step 1: read the interior trace value and normal reference derivative ----
              for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                   tid += blockSize)
                {
                  const int q_face        = tid % n_q_face;
                  const int dof_slot      = tid / n_q_face;
                  const int component     = dof_slot % n_components;
                  const int face_in_batch = dof_slot / n_face_dofs;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell       = face_info(face, 0);
                  const unsigned int local_face = face_info(face, 2);
                  const int          normal_dir = face_info(face, 2) / 2;
                  const int          slot       = boundary_face_slot_index<dim>(face_in_batch, component);

                  // m = J^{-1} n, component along the normal reference axis
                  const Number m_normal = jacobians_times_normal_boundary_face(
                    face * dim * n_q_face + normal_dir * n_q_face + q_face);

                  s_face_values[slot * n_q_face + q_face] =
                    face_values_at_quads(q_face, local_face, component, cell);
                  s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face] =
                    face_normal_derivatives_at_quads(q_face, local_face, component, cell) *
                    m_normal;
                }
              team_member.team_barrier();

              // ---- step 2: add the tangential reference derivatives of the trace ----
              for (int tang_index = 0; tang_index < dim - 1; ++tang_index)
                {
                  if (tang_index == 0)
                    apply<dim - 1, 0, n_q, n_q, true, false, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_face_values,
                                                                     s_temp,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                  else if constexpr (dim == 3)
                    apply<dim - 1, 1, n_q, n_q, true, false, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_face_values,
                                                                     s_temp,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);

                  for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                       tid += blockSize)
                    {
                      const int q_face        = tid % n_q_face;
                      const int dof_slot      = tid / n_q_face;
                      const int component     = dof_slot % n_components;
                      const int face_in_batch = dof_slot / n_face_dofs;

                      const int face        = face_batch * nelmtPerBatch + face_in_batch;
                      const int normal_dir  = face_info(face, 2) / 2;
                      const int tangent_dir = face_tangent_direction<dim>(normal_dir, tang_index);

                      const int    slot      = boundary_face_slot_index<dim>(face_in_batch, component);
                      const Number m_tangent = jacobians_times_normal_boundary_face(
                        face * dim * n_q_face + tangent_dir * n_q_face + q_face);
                      s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face] +=
                        s_temp[slot * n_q_face + q_face] * m_tangent;
                    }
                  team_member.team_barrier();
                }

              // ---- step 3: Piola map, Dirichlet mirror, SIPG flux, Piola-transpose ----
              for (int tid = threadIdx; tid < n_faces_this_batch * n_q_face; tid += blockSize)
                {
                  const int q_face        = tid % n_q_face;
                  const int face_in_batch = tid / n_q_face;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell       = face_info(face, 0);
                  const int          normal_dir = face_info(face, 2) / 2;

                  Number piola[dim][dim];
                  for (int i = 0; i < dim; ++i)
                    for (int j = 0; j < dim; ++j)
                      piola[i][j] = cell_piola[cell * dim * dim + i * dim + j];

                  Number m[dim];
                  for (int axis = 0; axis < dim; ++axis)
                    m[axis] = jacobians_times_normal_boundary_face(face * dim * n_q_face +
                                                                   axis * n_q_face + q_face);

                  const Number jxw            = jxw_boundary_face(face * n_q_face + q_face);
                  const Number penalty        = penalty_parameters_boundary_face[face];
                  const Number laplace_factor = factor_laplace;

                  Number u_ref[n_components], dn_ref[n_components];
                  for (int component = 0; component < n_components; ++component)
                    {
                      const int slot   = boundary_face_slot_index<dim>(face_in_batch, component);
                      u_ref[component] = s_face_values[slot * n_q_face + q_face];
                      dn_ref[component] =
                        s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face];
                    }

                  Number value_flux[n_components], grad_flux[n_components];
                  {
                    Number u_phys[n_components], dn_phys[n_components];
                    for (int component = 0; component < n_components; ++component)
                      {
                        Number acc_u = 0, acc_dn = 0;
                        for (int k = 0; k < n_components; ++k)
                          {
                            acc_u += piola[component][k] * u_ref[k];
                            acc_dn += piola[component][k] * dn_ref[k];
                          }
                        u_phys[component]  = acc_u;
                        dn_phys[component] = acc_dn;
                      }
                    // Dirichlet mirror (g = 0): jump = 2 * u_phys, {{du/dn}} = dn_phys.
                    for (int component = 0; component < n_components; ++component)
                      {
                        value_flux[component] =
                          jxw * laplace_factor *
                          (penalty * Number(2) * u_phys[component] - dn_phys[component]);
                        grad_flux[component] = -jxw * laplace_factor * u_phys[component];
                      }
                  }

                  for (int component = 0; component < n_components; ++component)
                    {
                      Number pt_value = 0, pt_grad = 0;
                      for (int k = 0; k < n_components; ++k)
                        {
                          pt_value += piola[k][component] * value_flux[k];
                          pt_grad += piola[k][component] * grad_flux[k];
                        }
                      const int slot = boundary_face_slot_index<dim>(face_in_batch, component);
                      s_face_values[slot * n_q_face + q_face] = pt_value;
                      for (int axis = 0; axis < dim; ++axis)
                        s_face_gradients[axis * buffer_size + slot * n_q_face + q_face] =
                          pt_grad * m[axis];
                    }
                }
              team_member.team_barrier();

              // ---- step 4: integrate the tangential test gradients, then write the trace back
              // ---- (mirror of step 2: axes undone in reverse order)
              for (int tang_index = dim - 2; tang_index >= 0; --tang_index)
                {
                  for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                       tid += blockSize)
                    {
                      const int q_face        = tid % n_q_face;
                      const int dof_slot      = tid / n_q_face;
                      const int component     = dof_slot % n_components;
                      const int face_in_batch = dof_slot / n_face_dofs;

                      const int face        = face_batch * nelmtPerBatch + face_in_batch;
                      const int normal_dir  = face_info(face, 2) / 2;
                      const int tangent_dir = face_tangent_direction<dim>(normal_dir, tang_index);

                      // move the tangential test-gradient slab into s_temp for apply()
                      const int slot = boundary_face_slot_index<dim>(face_in_batch, component);
                      s_temp[slot * n_q_face + q_face] =
                        s_face_gradients[tangent_dir * buffer_size + slot * n_q_face + q_face];
                    }
                  team_member.team_barrier();

                  if (tang_index == 0)
                    apply<dim - 1, 0, n_q, n_q, false, true, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_temp,
                                                                     s_face_values,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                  else if constexpr (dim == 3)
                    apply<dim - 1, 1, n_q, n_q, false, true, Number>(team_member,
                                                                     s_co_shape_gradients,
                                                                     s_temp,
                                                                     s_face_values,
                                                                     n_faces_this_batch *
                                                                       n_face_dofs,
                                                                     threadIdx,
                                                                     blockSize);
                }

              for (int tid = threadIdx; tid < n_faces_this_batch * n_face_dofs * n_q_face;
                   tid += blockSize)
                {
                  const int q_face        = tid % n_q_face;
                  const int dof_slot      = tid / n_q_face;
                  const int component     = dof_slot % n_components;
                  const int face_in_batch = dof_slot / n_face_dofs;

                  const int          face       = face_batch * nelmtPerBatch + face_in_batch;
                  const unsigned int cell       = face_info(face, 0);
                  const unsigned int local_face = face_info(face, 2);
                  const int          normal_dir = face_info(face, 2) / 2;
                  const int          slot       = boundary_face_slot_index<dim>(face_in_batch, component);

                  face_values_at_quads(q_face, local_face, component, cell) =
                    s_face_values[slot * n_q_face + q_face];
                  face_normal_derivatives_at_quads(q_face, local_face, component, cell) =
                    s_face_gradients[normal_dir * buffer_size + slot * n_q_face + q_face];
                }
              team_member.team_barrier();

              face_batch += team_member.league_size();
            }
        });
      Kokkos::fence();
    }


    template <int dim, int n_t, int n_q, typename Number>
    void
    distribute_face_to_global(
      const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
        interpolate_quad_to_boundary,
      const Kokkos::View<Number ****, MemorySpace::Default::kokkos_space> face_values_at_quads,
      const Kokkos::View<Number ****, MemorySpace::Default::kokkos_space>
                           face_normal_derivatives_at_quads,
      const DoFIndicesView dof_indices,
      // neighbor_cells(2*d+side, cell) == invalid_unsigned_int marks a domain
      // (Neumann) boundary face -- its contribution is zero, so skip reading it.
      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> neighbor_cells,
      DeviceView<Number>                                                      vector_out,
      const unsigned int                                                      n_cells,
      const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      using Custom::Parallel::apply_anisotropic;

      if (n_cells == 0)
        return;

      constexpr int n_components         = dim;
      constexpr int n_n                  = n_t + 1;
      constexpr int n_q_total            = Utilities::pow(n_q, dim);
      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);

      const int nelmt = n_cells;

      // per team scratch is (fixed shape-value terms) + (n_components + 2*dim) *
      // nelmtPerBatch * n_q_total (see ssize below) -- size nelmtPerBatch off of that
      // per-cell term against a shared-memory budget, same idea as compute_cell.
      const int shmemPerBlock = 10800; // total shared memory used per block (bytes)

      const int nelmtPerBatch =
        (n_cells_per_batch == numbers::invalid_unsigned_int) ?
          std::max(1,
                  shmemPerBlock / ((n_components + 2 * dim) * n_q_total) /
                    static_cast<int>(sizeof(Number))) :
          static_cast<int>(n_cells_per_batch);
      const int numBlocks       = (n_blocks == numbers::invalid_unsigned_int) ?
                                    std::max(1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch) :
                                    static_cast<int>(n_blocks);
      const int threadsPerBlock = (threads_per_block == numbers::invalid_unsigned_int) ?
                                    std::min(nelmtPerBatch * Utilities::pow(n_q, dim - 1), 512) :
                                    static_cast<int>(threads_per_block);

      const unsigned int ssize      = n_n * n_q + n_t * n_q // shape values
                                      + 4 * n_q             // interpolate_quad_to_boundary
                                      + n_components * nelmtPerBatch * n_q_total // s_values
                                      + 2 * nelmtPerBatch * n_q_total * dim; // ping-pong scratch
      const unsigned int shmem_size = ssize * sizeof(Number);

      typedef Kokkos::TeamPolicy<>::member_type MemberType;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(MemberType team_member) {
          Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

          Number *shape_values_normal  = scratch;
          Number *shape_values_tangent = shape_values_normal + n_n * n_q;
          // flattened as [kind * (2 * n_q) + quad * 2 + side], kind in {0=value, 1=deriv}
          Number *s_interp_to_boundary = shape_values_tangent + n_t * n_q;

          Number *s_values[n_components];
          Number *s_temp[2];
          {
            Number *ptr = s_interp_to_boundary + 4 * n_q;
            for (int c = 0; c < n_components; ++c)
              {
                s_values[c] = ptr;
                ptr += nelmtPerBatch * n_q_total;
              }
            s_temp[0] = ptr;
            ptr += nelmtPerBatch * n_q_total * dim;
            s_temp[1] = ptr;
          }

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();

          for (int t = threadIdx; t < n_n * n_q; t += blockSize)
            shape_values_normal[t] = shape_values_info[0][t];
          for (int t = threadIdx; t < n_t * n_q; t += blockSize)
            shape_values_tangent[t] = shape_values_info[1][t];
          for (int t = threadIdx; t < 2 * n_q; t += blockSize)
            {
              const int kind = t / n_q;
              const int quad = t % n_q;
              s_interp_to_boundary[kind * (2 * n_q) + quad * 2 + 0] =
                interpolate_quad_to_boundary(kind, quad, 0);
              s_interp_to_boundary[kind * (2 * n_q) + quad * 2 + 1] =
                interpolate_quad_to_boundary(kind, quad, 1);
            }
          team_member.team_barrier();

          int eb = team_member.league_rank();
          while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {
              const int c_nelmtPerBatch = Kokkos::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

              // -- step 1: interpolate the 2*dim faces back to the cell quad points --
              // Adjoint of compute_cell's interpolate-to-faces step: there, each cell
              // quad point contributes (via a 1D weight) to every face it lies on; here,
              // for every cell quad point cell_quad, we sum back the contributions of the
              // 2*dim faces it lies on. For each face-normal axis face_axis and side
              // (0/1), the 1D weight at cell_quad's position along face_axis combines
              // that face's (already flux-integrated) trace value and normal derivative,
              // using the same interpolate_quad_to_boundary weights compute_cell used to
              // go the other way. Neumann boundary faces are skipped (zero contribution).
              // The result lands in s_values, ready for the dof-space contraction in step 2.
              for (int tid = threadIdx; tid < c_nelmtPerBatch * n_q_total; tid += blockSize)
                {
                  const int cell_in_batch = tid / n_q_total;
                  const int cell_quad     = tid % n_q_total;
                  const int cell_id       = eb * nelmtPerBatch + cell_in_batch;

                  const int axis_idx[3] = {cell_quad % n_q,
                                           (cell_quad / n_q) % n_q,
                                           (dim == 3) ? (cell_quad / (n_q * n_q)) : 0};

                  Number acc[n_components];
                  for (int c = 0; c < n_components; ++c)
                    acc[c] = 0;

                  for (int face_axis = 0; face_axis < dim; ++face_axis)
                    {
                      const int normal_quad_idx = axis_idx[face_axis];
                      int       face_quad = 0, stride = 1;
                      for (int axis = 0; axis < dim; ++axis)
                        if (axis != face_axis)
                          {
                            face_quad += axis_idx[axis] * stride;
                            stride *= n_q;
                          }

                      for (int side = 0; side < 2; ++side)
                        {
                          // Neumann boundary face: contribution is zero, nothing to read.
                          if (neighbor_cells(2 * face_axis + side, cell_id) ==
                              numbers::invalid_unsigned_int)
                            continue;

                          const Number weight_value =
                            s_interp_to_boundary[0 * (2 * n_q) + normal_quad_idx * 2 + side];
                          const Number weight_deriv =
                            s_interp_to_boundary[1 * (2 * n_q) + normal_quad_idx * 2 + side];
                          for (int c = 0; c < n_components; ++c)
                            acc[c] +=
                              weight_value *
                                face_values_at_quads(face_quad, 2 * face_axis + side, c, cell_id) +
                              weight_deriv * face_normal_derivatives_at_quads(face_quad,
                                                                              2 * face_axis + side,
                                                                              c,
                                                                              cell_id);
                        }
                    }

                  for (int c = 0; c < n_components; ++c)
                    s_values[c][cell_in_batch * n_q_total + cell_quad] = acc[c];
                }
              team_member.team_barrier();

              // -- step 2: integrate cell quad values -> dof values (as compute_cell PHASE 4) --
              {
                const Number *const mat_n = shape_values_normal;
                const Number *const mat_t = shape_values_tangent;
                const int           nb    = c_nelmtPerBatch;

                // component 0
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 0, n_n, n_t),
                                    false,
                                    false,
                                    Number>(
                    team_member, mat_t, s_values[0], s_temp[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 0, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_t,
                                          (dim == 2) ? s_values[0] : s_temp[0],
                                          (dim == 2) ? s_temp[0] : s_temp[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                apply_anisotropic<dim,
                                  0,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 0, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_n,
                                          (dim == 2) ? s_temp[0] : s_temp[1],
                                          s_values[0],
                                          nb,
                                          threadIdx,
                                          blockSize);

                // component 1
                if constexpr (dim == 3)
                  apply_anisotropic<dim,
                                    2,
                                    n_t,
                                    n_q,
                                    rt_perp_dof_extent(dim, 2, 1, n_n, n_t),
                                    false,
                                    false,
                                    Number>(
                    team_member, mat_t, s_values[1], s_temp[0], nb, threadIdx, blockSize);
                apply_anisotropic<dim,
                                  1,
                                  n_n,
                                  n_q,
                                  rt_perp_dof_extent(dim, 1, 1, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_n,
                                          (dim == 2) ? s_values[1] : s_temp[0],
                                          (dim == 2) ? s_temp[0] : s_temp[1],
                                          nb,
                                          threadIdx,
                                          blockSize);
                apply_anisotropic<dim,
                                  0,
                                  n_t,
                                  n_q,
                                  rt_perp_dof_extent(dim, 0, 1, n_n, n_t),
                                  false,
                                  false,
                                  Number>(team_member,
                                          mat_t,
                                          (dim == 2) ? s_temp[0] : s_temp[1],
                                          s_values[1],
                                          nb,
                                          threadIdx,
                                          blockSize);

                // component 2
                if constexpr (dim == 3)
                  {
                    apply_anisotropic<dim,
                                      2,
                                      n_n,
                                      n_q,
                                      rt_perp_dof_extent(dim, 2, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_n, s_values[2], s_temp[0], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      1,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 1, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_t, s_temp[0], s_temp[1], nb, threadIdx, blockSize);
                    apply_anisotropic<dim,
                                      0,
                                      n_t,
                                      n_q,
                                      rt_perp_dof_extent(dim, 0, 2, n_n, n_t),
                                      false,
                                      false,
                                      Number>(
                      team_member, mat_t, s_temp[1], s_values[2], nb, threadIdx, blockSize);
                  }
              }

              // -- step 3: scatter to the global vector (H(div): shared face dofs -> atomic) --
              for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
                   tid += blockSize)
                {
                  const int e              = tid / n_dofs_per_component;
                  const int i              = tid % n_dofs_per_component;
                  const int global_cell_id = eb * nelmtPerBatch + e;

                  for (int comp = 0; comp < n_components; ++comp)
                    {
                      const unsigned int dof =
                        dof_indices(comp * n_dofs_per_component + i, global_cell_id);
                      if (dof != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&vector_out[dof], s_values[comp][tid]);
                    }
                }
              team_member.team_barrier();

              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }



  } // namespace RT
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif