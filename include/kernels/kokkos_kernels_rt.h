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

                    if constexpr (dim == 2)
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
                    else if constexpr (dim == 3)
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
          (shmemPerBlock / (n_components * (dim + 1) * n_q_total) / sizeof(Number)) :
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

                      if constexpr (dim == 2)
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

                      if constexpr (dim == 2)
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
          (shmemPerBlock / (n_components * (dim + 1) * n_q_total) / sizeof(Number)) :
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

                      if constexpr (dim == 2)
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

                      Number u[dim];

                      if constexpr (dim == 2)
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


    template <int dim, int n_t, int n_q, typename Number>
    void
    compute_cell(const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
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
                 const unsigned int   n_cells,
                 const bool           interpolate_to_faces,
                 const Number         factor_mass       = Number(1),
                 const Number         factor_laplace    = Number(1),
                 const unsigned int   n_cells_per_batch = numbers::invalid_unsigned_int,
                 const unsigned int   n_blocks          = numbers::invalid_unsigned_int,
                 const unsigned int   threads_per_block = numbers::invalid_unsigned_int)

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
          (shmemPerBlock / (n_components * (dim + 1) * n_q_total) / sizeof(Number)) :
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
                                  v0 += interpolate_quad_to_boundary(0, n, 0) * u;
                                  v1 += interpolate_quad_to_boundary(0, n, 1) * u;
                                  dn0 += interpolate_quad_to_boundary(1, n, 0) * u;
                                  dn1 += interpolate_quad_to_boundary(1, n, 1) * u;
                                }
                              face_values_at_quads(m, 2 * d + 0, comp, global_cell_id) = v0;
                              face_values_at_quads(m, 2 * d + 1, comp, global_cell_id) = v1;
                              face_normal_derivatives_at_quads(m, 2 * d + 0, comp, global_cell_id) =
                                dn0;
                              face_normal_derivatives_at_quads(m, 2 * d + 1, comp, global_cell_id) =
                                dn1;
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

    template <int dim, int n_t, int n_q, typename Number, bool compute_exterior>
    void
    compute_face(const Kokkos::Array<DeviceView<Number>, 2> shape_values_info,
                 const DeviceView<Number>                   shape_gradients_collocation,
                 const Kokkos::Array<Kokkos::Array<DeviceView<Number>, 2>, 2> shape_data_on_face,
                 const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
                                          interpolate_quad_to_boundary,
                 const DeviceView<Number> geometric_tensor_mass,
                 const DeviceView<Number> geometric_tensor_stiffness,
                 const Kokkos::View<Number **, MemorySpace::Default::kokkos_space> quad_values,
                 const DeviceView<Number>                                          vector_in,
                 DeviceView<Number>                                                vector_out,
                 const DoFIndicesView                                              dof_indices,
                 const DoFIndicesView                                              neighbor_cells,
                 const unsigned int                                                n_cells,
                 const Number       factor_mass       = Number(1),
                 const Number       factor_laplace    = Number(1),
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

      constexpr int n_q_total      = Utilities::pow(n_q, dim);
      constexpr int n_q_total_face = Utilities::pow(n_q, dim - 1);

      constexpr int n_dofs_per_face  = Utilities::pow(n_t, dim - 1);
      constexpr int n_dofs_per_plane = n_t * (n_t - 1);

      constexpr int n_cell_dofs_per_component = Utilities::pow(n_t, dim - 1) * (n_t - 1);

      constexpr int n_dofs_per_component = n_n * Utilities::pow(n_t, dim - 1);
      constexpr int n_faces              = 2 * dim;

      const int nelmt = n_cells;


      // const size_t shmemPerBlock =
      //   Kokkos::TeamPolicy<>::scratch_size_max(0); // maximum shared memory size per thread block

      int shmemPerBlock = 10800; // total shared memory used per block (KB)


      const int nelmtPerBatch =
        (n_cells_per_batch == numbers::invalid_unsigned_int) ?
          (shmemPerBlock / (n_components * (dim + 1) * n_q_total) / sizeof(Number)) :
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

          Number *quad_to_boundary_values_0 = shape_values_tangent + n_q * n_q;
          Number *quad_to_boundary_values_1 = quad_to_boundary_values_0 + n_q;
          Number *quad_to_boundary_grads_0  = quad_to_boundary_values_1 + n_q;
          Number *quad_to_boundary_grads_1  = quad_to_boundary_grads_0 + n_q;

          // x=0
          Number *normal_face_values_u0_face0 = quad_to_boundary_grads_1 + n_q;
          Number *normal_face_grads_u0_face0 =
            normal_face_values_u0_face0 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_values_u1_face0 =
            normal_face_grads_u0_face0 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_grads_u1_face0 =
            normal_face_values_u1_face0 + nelmtPerBatch * n_q_total_face;

          Number *tangential_face_grads_u0_face0 =
            normal_face_grads_u1_face0 + nelmtPerBatch * n_q_total_face;
          Number *tangential_face_grads_u1_face0 =
            tangential_face_grads_u0_face0 + nelmtPerBatch * n_q_total_face * (dim - 1);

          // x=1
          Number *normal_face_values_u0_face1 =
            tangential_face_grads_u1_face0 + nelmtPerBatch * n_q_total_face * (dim - 1);
          Number *normal_face_grads_u0_face1 =
            normal_face_values_u0_face1 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_values_u1_face1 =
            normal_face_grads_u0_face1 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_grads_u1_face1 =
            normal_face_values_u1_face1 + nelmtPerBatch * n_q_total_face;

          Number *tangential_face_grads_u0_face1 =
            normal_face_grads_u1_face1 + nelmtPerBatch * n_q_total_face;
          Number *tangential_face_grads_u1_face1 =
            tangential_face_grads_u0_face1 + nelmtPerBatch * n_q_total_face * (dim - 1);

          // y=0
          Number *normal_face_values_u0_face2 =
            tangential_face_grads_u1_face1 + nelmtPerBatch * n_q_total_face * (dim - 1);
          Number *normal_face_grads_u0_face2 =
            normal_face_values_u0_face2 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_values_u1_face2 =
            normal_face_grads_u0_face2 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_grads_u1_face2 =
            normal_face_grads_u1_face2 + nelmtPerBatch * n_q_total_face;

          Number *tangential_face_grads_u0_face2 =
            normal_face_grads_u1_face2 + nelmtPerBatch * n_q_total_face;
          Number *tangential_face_grads_u1_face2 =
            tangential_face_grads_u0_face2 + nelmtPerBatch * n_q_total_face * (dim - 1);

          // y=1
          Number *normal_face_values_u0_face3 =
            tangential_face_grads_u1_face2 + nelmtPerBatch * n_q_total_face * (dim - 1);
          Number *normal_face_grads_u0_face3 =
            normal_face_values_u0_face3 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_values_u1_face3 =
            normal_face_grads_u0_face3 + nelmtPerBatch * n_q_total_face;
          Number *normal_face_grads_u1_face3 =
            normal_face_values_u1_face3 + nelmtPerBatch * n_q_total_face;


          Number *tangential_face_grads_u0_face3 =
            normal_face_grads_u1_face3 + nelmtPerBatch * n_q_total_face;
          Number *tangential_face_grads_u1_face3 =
            tangential_face_grads_u0_face3 + nelmtPerBatch * n_q_total_face * (dim - 1);

          Number *quad_values_0 =
            tangential_face_grads_u1_face3 + nelmtPerBatch * n_q_total * (dim - 1);
          Number *quad_values_1 = quad_values_0 + nelmtPerBatch * n_q_total;
          Number *quad_values_2;

          Number *normal_face_values_u0_face4, normal_face_grads_u0_face4,
            normal_face_values_u1_face4, normal_face_grads_u1_face4, normal_face_values_u0_face5,
            normal_face_grads_u0_face5, normal_face_values_u1_face5, normal_face_grads_u1_face5;

          Number *normal_face_values_u2_face0, normal_face_grads_u2_face0;
          Number *normal_face_values_u2_face1, normal_face_grads_u2_face1;
          Number *normal_face_values_u2_face2, normal_face_grads_u2_face2;
          Number *normal_face_values_u2_face3, normal_face_grads_u2_face3;
          Number *normal_face_values_u2_face4, normal_face_grads_u2_face4;
          Number *normal_face_values_u2_face5, normal_face_grads_u2_face5;
          Number *tangential_face_grads_u0_face4, tangential_face_grads_u0_face5,
            tangential_face_grads_u1_face4, tangential_face_grads_u1_face5;
          Number *tangential_face_grads_u2_face0, tangential_face_grads_u2_face1,
            tangential_face_grads_u2_face2, tangential_face_grads_u2_face3,
            tangential_face_grads_u2_face4, tangential_face_grads_u2_face5;

          if (dim > 3)
            {
              quad_values_2 = normal_face_grads_u1_face3 + nelmtPerBatch * n_q_total_face;

              // x=0
              normal_face_values_u2_face0 = quad_values_2 + nelmtPerBatch * n_q_total;
              normal_face_grads_u2_face0 =
                normal_face_values_u2_face0 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u2_face0 =
                normal_face_grads_u2_face0 + nelmtPerBatch * n_q_total_face;


              // x=1
              normal_face_values_u2_face1 =
                tangential_face_grads_u2_face0 + nelmtPerBatch * n_q_total_face * (dim - 1);
              normal_face_grads_u2_face1 =
                normal_face_values_u2_face1 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u2_face1 =
                normal_face_grads_u2_face1 + nelmtPerBatch * n_q_total_face;


              // y=0
              normal_face_values_u2_face2 =
                tangential_face_grads_u2_face1 + nelmtPerBatch * n_q_total_face * (dim - 1);
              normal_face_grads_u2_face2 =
                normal_face_values_u2_face2 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u2_face2 =
                normal_face_grads_u2_face2 + nelmtPerBatch * n_q_total_face;

              // y=1
              normal_face_values_u2_face3 =
                tangential_face_grads_u2_face2 + nelmtPerBatch * n_q_total_face * (dim - 1);
              normal_face_grads_u2_face3 =
                normal_face_values_u2_face3 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u2_face3 =
                normal_face_grads_u2_face3 + nelmtPerBatch * n_q_total_face;

              // z=0
              normal_face_values_u0_face4 =
                tangential_face_grads_u2_face3 + nelmtPerBatch * n_q_total_face * (dim - 1);
              normal_face_grads_u0_face4 =
                normal_face_values_u0_face4 + nelmtPerBatch * n_q_total_face;
              normal_face_values_u1_face4 =
                normal_face_grads_u0_face4 + nelmtPerBatch * n_q_total_face;
              normal_face_grads_u1_face4 =
                normal_face_values_u1_face4 + nelmtPerBatch * n_q_total_face;
              normal_face_values_u2_face4 =
                normal_face_grads_u1_face4 + nelmtPerBatch * n_q_total_face;
              normal_face_grads_u2_face4 =
                normal_face_values_u2_face4 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u0_face4 =
                normal_face_grads_u2_face4 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u1_face4 =
                tangential_face_grads_u0_face4 + nelmtPerBatch * n_q_total_face * (dim - 1);
              tangential_face_grads_u2_face4 =
                tangential_face_grads_u1_face4 + nelmtPerBatch * n_q_total_face * (dim - 1);

              // z=1
              normal_face_values_u0_face5 =
                tangential_face_grads_u2_face4 + nelmtPerBatch * n_q_total_face * (dim - 1);
              normal_face_grads_u0_face5 =
                normal_face_values_u0_face5 + nelmtPerBatch * n_q_total_face;
              normal_face_values_u1_face5 =
                normal_face_grads_u0_face5 + nelmtPerBatch * n_q_total_face;
              normal_face_grads_u1_face5 =
                normal_face_values_u1_face5 + nelmtPerBatch * n_q_total_face;
              normal_face_values_u2_face5 =
                normal_face_grads_u1_face5 + nelmtPerBatch * n_q_total_face;
              normal_face_grads_u2_face5 =
                normal_face_values_u2_face5 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u0_face4 =
                normal_face_grads_u2_face4 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u0_face5 =
                normal_face_grads_u2_face5 + nelmtPerBatch * n_q_total_face;
              tangential_face_grads_u1_face5 =
                tangential_face_grads_u0_face5 + nelmtPerBatch * n_q_total_face * (dim - 1);
              tangential_face_grads_u2_face4 =
                tangential_face_grads_u1_face5 + nelmtPerBatch * n_q_total_face * (dim - 1);
            }

          // Number *s_uq_0  = co_shape_gradients + n_q * n_q;
          // Number *s_duq_0 = s_uq_0 + nelmtPerBatch * n_q_total;
          // Number *s_uq_1  = s_duq_0 + nelmtPerBatch * n_q_total * dim;
          // Number *s_duq_1 = s_uq_1 + nelmtPerBatch * n_q_total;


          // Number *s_uq_2, *s_duq_2;
          // if constexpr (dim > 2)
          //   {
          //     s_uq_2  = s_duq_1 + nelmtPerBatch * n_q_total * dim;
          //     s_duq_2 = s_uq_2 + nelmtPerBatch * n_q_total;
          //   }

          const int threadIdx = team_member.team_rank();
          const int blockSize = team_member.team_size();


          // copy to shared memory
          {
            for (int tid = threadIdx; tid < n_q; tid += blockSize)
              {
                quad_to_boundary_values_0[tid] = interpolate_quad_to_boundary(0, tid, 0);
                quad_to_boundary_grads_0[tid]  = interpolate_quad_to_boundary(1, tid, 0);

                quad_to_boundary_values_1[tid] = interpolate_quad_to_boundary(0, tid, 1);
                quad_to_boundary_grads_1[tid]  = interpolate_quad_to_boundary(1, tid, 1);
              }
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

              int neighbor_cell_ids[c_nelmtPerBatch * n_faces];

              for (int e = 0; e < c_nelmtPerBatch; ++e)
                for (int f = 0; f < n_faces; ++f)
                  neighbor_cell_ids[e * n_faces + f] = neighbor_cells(f, eb * nelmtPerBatch + e);


              // read quad values
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_q_total; tid += blockSize)
                  {
                    const int e       = tid / n_q_total;
                    const int q_index = tid % n_q_total;

                    const int global_cell_id = eb * nelmtPerBatch + e;

                    quad_values_0[tid] = quad_values(q_index, 0, global_cell_id);
                    quad_values_1[tid] = quad_values(q_index, 1, global_cell_id);

                    if (dim > 2)
                      quad_values_2[tid] = quad_values(q_index, 2, global_cell_id);
                  }
              }

              // intepolate normal values from quads
              {
                for (int tid = threadIdx; tid < c_nelmtPerBatch * n_q_total_face; tid += blockSize)
                  {
                    const int e = tid / n_q_total_face;
                    if (dim == 2)
                      {
                        {
                          const int q = tid % n_q_total_face;

                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0
                              r_p0[n] = quad_values_0[e * n_q * n_q + q * n_q + n];

                              // component 1
                              r_p1[n] = quad_values_1[e * n_q * n_q + q * n_q + n];
                            }

                          Number v0[2], d0[2], v1[2], d1[2];
                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0, x=0
                              v0[0] += quad_to_boundary_values_0[n] * r_p0[n];
                              d0[0] += quad_to_boundary_grads_0[n] * r_p0[n];

                              // component 0, x=1
                              v0[1] += quad_to_boundary_values_1[n] * r_p0[n];
                              d0[1] += quad_to_boundary_grads_1[n] * r_p0[n];

                              // component 1, x=0
                              v1[0] += quad_to_boundary_values_0[n] * r_p1[n];
                              d1[0] += quad_to_boundary_grads_0[n] * r_p1[n];

                              // component 1, x=1
                              v1[1] += quad_to_boundary_values_1[n] * r_p1[n];
                              d1[1] += quad_to_boundary_grads_1[n] * r_p1[n];
                            }

                          // x=0
                          normal_face_values_u0_face0[tid] = v0[0];
                          normal_face_grads_u0_face0[tid]  = d0[0];
                          normal_face_values_u1_face0[tid] = v1[0];
                          normal_face_grads_u1_face0[tid]  = d1[0];

                          // x=1
                          normal_face_values_u0_face1[tid] = v0[1];
                          normal_face_grads_u0_face1[tid]  = d0[1];
                          normal_face_values_u1_face1[tid] = v1[1];
                          normal_face_grads_u1_face1[tid]  = d1[1];
                        }

                        {
                          const int p = tid % n_q_total_face;

                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0
                              r_p0[n] = quad_values_0[e * n_q * n_q + n * n_q + p];

                              // component 1
                              r_p1[n] = quad_values_1[e * n_q * n_q + n * n_q + p];
                            }

                          Number v0[2], d0[2], v1[2], d1[2];

                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0, x=0
                              v0[0] += quad_to_boundary_values_0[n] * r_p0[n];
                              d0[0] += quad_to_boundary_grads_0[n] * r_p0[n];

                              // component 0, x=1
                              v0[1] += quad_to_boundary_values_1[n] * r_p0[n];
                              d0[1] += quad_to_boundary_grads_1[n] * r_p0[n];

                              // component 1, x=0
                              v1[0] += quad_to_boundary_values_0[n] * r_p1[n];
                              d1[0] += quad_to_boundary_grads_0[n] * r_p1[n];

                              // component 1, x=1
                              v1[1] += quad_to_boundary_values_1[n] * r_p1[n];
                              d1[1] += quad_to_boundary_grads_1[n] * r_p1[n];
                            }

                          // y=0
                          normal_face_values_u0_face0[tid] = v0[0];
                          normal_face_grads_u0_face0[tid]  = d0[0];
                          normal_face_values_u1_face0[tid] = v1[0];
                          normal_face_grads_u1_face0[tid]  = d1[0];

                          // y=1
                          normal_face_values_u0_face1[tid] = v0[1];
                          normal_face_grads_u0_face1[tid]  = d0[1];
                          normal_face_values_u1_face1[tid] = v1[1];
                          normal_face_grads_u1_face1[tid]  = d1[1];
                        }
                      }
                    else if constexpr (dim == 3)
                      {
                        {
                          const int q = (tid % n_q_total_face) / n_q;
                          const int r = tid % n_q;

                          const int quad_offset = e * n_q_total + r * n_q * n_q + q * n_q;

                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0
                              r_p0[n] = quad_values_0[quad_offset + n];

                              // component 1
                              r_p1[n] = quad_values_1[quad_offset + n];

                              // component 2
                              r_p2[n] = quad_values_2[quad_offset + n];
                            }

                          Number v0[2], d0[2], v1[2], d1[2], v2[2], d2[2];

                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0, x=0
                              v0[0] += quad_to_boundary_values_0[n] * r_p0[n];
                              d0[0] += quad_to_boundary_grads_0[n] * r_p0[n];

                              // component 0, x=1
                              v0[1] += quad_to_boundary_values_1[n] * r_p0[n];
                              d0[1] += quad_to_boundary_grads_1[n] * r_p0[n];

                              // component 1, x=0
                              v1[0] += quad_to_boundary_values_0[n] * r_p1[n];
                              d1[0] += quad_to_boundary_grads_0[n] * r_p1[n];

                              // component 1, x=1
                              v1[1] += quad_to_boundary_values_1[n] * r_p1[n];
                              d1[1] += quad_to_boundary_grads_1[n] * r_p1[n];

                              // component 2, x=0
                              v2[0] += quad_to_boundary_values_0[n] * r_p2[n];
                              d2[0] += quad_to_boundary_grads_0[n] * r_p2[n];

                              // component 2, x=1
                              v2[1] += quad_to_boundary_values_1[n] * r_p2[n];
                              d2[1] += quad_to_boundary_grads_1[n] * r_p2[n];
                            }

                          // x=0
                          normal_face_values_u0_face0[tid] = v0[0];
                          normal_face_grads_u0_face0[tid]  = d0[0];
                          normal_face_values_u1_face0[tid] = v1[0];
                          normal_face_grads_u1_face0[tid]  = d1[0];
                          normal_face_values_u2_face0[tid] = v2[0];
                          normal_face_grads_u2_face0[tid]  = d2[0];


                          // x=1
                          normal_face_values_u0_face1[tid] = v0[1];
                          normal_face_grads_u0_face1[tid]  = d0[1];
                          normal_face_values_u1_face1[tid] = v1[1];
                          normal_face_grads_u1_face1[tid]  = d1[1];
                          normal_face_values_u2_face1[tid] = v2[1];
                          normal_face_grads_u2_face1[tid]  = d2[1];
                        }

                        {
                          const int p = (tid % n_q_total_face) / n_q;
                          const int r = tid % n_q;

                          const int quad_offset = e * n_q_total + r * n_q * n_q + p;
                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0
                              r_p0[n] = quad_values_0[quad_offset + n * n_q];

                              // component 1
                              r_p1[n] = quad_values_1[quad_offset + n * n_q];

                              // component 2
                              r_p2[n] = quad_values_2[quad_offset + n * n_q];
                            }

                          Number v0[2], d0[2], v1[2], d1[2], v2[2], d2[2];
                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0, y=0
                              v0[0] += quad_to_boundary_values_0[n] * r_p0[n];
                              d0[0] += quad_to_boundary_grads_0[n] * r_p0[n];

                              // component 0, y=1
                              v0[1] += quad_to_boundary_values_1[n] * r_p0[n];
                              d0[1] += quad_to_boundary_grads_1[n] * r_p0[n];

                              // component 1, y=0
                              v1[0] += quad_to_boundary_values_0[n] * r_p1[n];
                              d1[0] += quad_to_boundary_grads_0[n] * r_p1[n];

                              // component 1, y=1
                              v1[1] += quad_to_boundary_values_1[n] * r_p1[n];
                              d1[1] += quad_to_boundary_grads_1[n] * r_p1[n];

                              // component 2, y=0
                              v2[0] += quad_to_boundary_values_0[n] * r_p2[n];
                              d2[0] += quad_to_boundary_grads_0[n] * r_p2[n];

                              // component 2, y=1
                              v2[1] += quad_to_boundary_values_1[n] * r_p2[n];
                              d2[1] += quad_to_boundary_grads_1[n] * r_p2[n];
                            }

                          // y=0
                          normal_face_values_u0_face2[tid] = v0[0];
                          normal_face_grads_u0_face2[tid]  = d0[0];
                          normal_face_values_u1_face2[tid] = v1[0];
                          normal_face_grads_u1_face2[tid]  = d1[0];
                          normal_face_values_u2_face2[tid] = v2[0];
                          normal_face_grads_u2_face2[tid]  = d2[0];


                          // y=1
                          normal_face_values_u0_face3[tid] = v0[1];
                          normal_face_grads_u0_face3[tid]  = d0[1];
                          normal_face_values_u1_face3[tid] = v1[1];
                          normal_face_grads_u1_face3[tid]  = d1[1];
                          normal_face_values_u2_face3[tid] = v2[1];
                          normal_face_grads_u2_face3[tid]  = d2[1];
                        }

                        {
                          const int p = (tid % n_q_total_face) / n_q;
                          const int q = tid % n_q;

                          const int quad_offset = e * n_q_total + q * n_q + p;
                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0
                              r_p0[n] = quad_values_0[quad_offset + n * n_q * n_q];

                              // component 1
                              r_p1[n] = quad_values_1[quad_offset + n * n_q * n_q];

                              // component 2
                              r_p2[n] = quad_values_2[quad_offset + n * n_q * n_q];
                            }

                          Number v0[2], d0[2], v1[2], d1[2], v2[2], d2[2];
                          for (int n = 0; n < n_q; ++n)
                            {
                              // component 0, z=0
                              v0[0] += quad_to_boundary_values_0[n] * r_p0[n];
                              d0[0] += quad_to_boundary_grads_0[n] * r_p0[n];

                              // component 0, z=1
                              v0[1] += quad_to_boundary_values_1[n] * r_p0[n];
                              d0[1] += quad_to_boundary_grads_1[n] * r_p0[n];

                              // component 1, z=0
                              v1[0] += quad_to_boundary_values_0[n] * r_p1[n];
                              d1[0] += quad_to_boundary_grads_0[n] * r_p1[n];

                              // component 1, z=1
                              v1[1] += quad_to_boundary_values_1[n] * r_p1[n];
                              d1[1] += quad_to_boundary_grads_1[n] * r_p1[n];

                              // component 2, z=0
                              v2[0] += quad_to_boundary_values_0[n] * r_p2[n];
                              d2[0] += quad_to_boundary_grads_0[n] * r_p2[n];

                              // component 2, z=1
                              v2[1] += quad_to_boundary_values_1[n] * r_p2[n];
                              d2[1] += quad_to_boundary_grads_1[n] * r_p2[n];
                            }

                          // z=0
                          normal_face_values_u0_face4[tid] = v0[0];
                          normal_face_grads_u0_face4[tid]  = d0[0];
                          normal_face_values_u1_face4[tid] = v1[0];
                          normal_face_grads_u1_face4[tid]  = d1[0];
                          normal_face_values_u2_face4[tid] = v2[0];
                          normal_face_grads_u2_face4[tid]  = d2[0];


                          // z=1
                          normal_face_values_u0_face5[tid] = v0[1];
                          normal_face_grads_u0_face5[tid]  = d0[1];
                          normal_face_values_u1_face5[tid] = v1[1];
                          normal_face_grads_u1_face5[tid]  = d1[1];
                          normal_face_values_u2_face5[tid] = v2[1];
                          normal_face_grads_u2_face5[tid]  = d2[1];
                        }
                      }
                  }
              }

              // compute tangential derivatives at quadrature points
              {
                constexpr int co_dimension_size = Utilities::pow(n_q, dim - 2);
                for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                     tid += blockSize)
                  {
                    const int e = tid / co_dimension_size;

                    if constexpr (dim == 2)
                      {
                        {
                          for (int q = 0; q < n_q; ++q)
                            {
                              Number qr0 = 0, qs0 = 0, qr1 = 0, qs1 = 0;

                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr0 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u0_face0[e * n_q + n];
                                  qs0 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u1_face0[e * n_q + n];

                                  qr1 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u0_face1[e * n_q + n];
                                  qs1 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u1_face1[e * n_q + n];
                                }

                              tangential_face_grads_u0_face0[e * n_q + q] = qr0;
                              tangential_face_grads_u1_face0[e * n_q + q] = qs0;

                              tangential_face_grads_u0_face1[e * n_q + q] = qr1;
                              tangential_face_grads_u1_face1[e * n_q + q] = qs1;
                            }
                        }
                        {
                          for (int q = 0; q < n_q; ++q)
                            {
                              Number qr0 = 0, qs0 = 0, qr1 = 0, qs1 = 0;

                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr0 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u0_face2[e * n_q + n];
                                  qs0 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u1_face2[e * n_q + n];

                                  qr1 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u0_face3[e * n_q + n];
                                  qs1 += co_shape_gradients[n * n_q + q] *
                                         normal_face_values_u1_face3[e * n_q + n];
                                }

                              tangential_face_grads_u0_face2[e * n_q + q] = qr0;
                              tangential_face_grads_u1_face2[e * n_q + q] = qs0;

                              tangential_face_grads_u0_face3[e * n_q + q] = qr1;
                              tangential_face_grads_u1_face3[e * n_q + q] = qs1;
                            }
                        }
                      }
                    else if constexpr (dim == 3)
                      {
                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face0[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face0[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face0[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim], qt[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face0[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face0[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face0[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face0[idx0] = qr[0];
                              tangential_face_grads_u0_face0[idx1] = qs[0];
                              tangential_face_grads_u1_face0[idx0] = qr[1];
                              tangential_face_grads_u1_face0[idx1] = qs[1];
                              tangential_face_grads_u2_face0[idx0] = qr[2];
                              tangential_face_grads_u2_face0[idx1] = qs[2];
                            }
                        }

                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face1[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face1[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face1[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face1[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face1[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face1[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face1[idx0] = qr[0];
                              tangential_face_grads_u0_face1[idx1] = qs[0];
                              tangential_face_grads_u1_face1[idx0] = qr[1];
                              tangential_face_grads_u1_face1[idx1] = qs[1];
                              tangential_face_grads_u2_face1[idx0] = qr[2];
                              tangential_face_grads_u2_face1[idx1] = qs[2];
                            }
                        }

                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face2[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face2[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face2[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face2[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face2[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face2[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face2[idx0] = qr[0];
                              tangential_face_grads_u0_face2[idx1] = qs[0];
                              tangential_face_grads_u1_face2[idx0] = qr[1];
                              tangential_face_grads_u1_face2[idx1] = qs[1];
                              tangential_face_grads_u2_face2[idx0] = qr[2];
                              tangential_face_grads_u2_face2[idx1] = qs[2];
                            }
                        }

                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face3[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face3[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face3[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face3[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face3[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face3[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face3[idx0] = qr[0];
                              tangential_face_grads_u0_face3[idx1] = qs[0];
                              tangential_face_grads_u1_face3[idx0] = qr[1];
                              tangential_face_grads_u1_face3[idx1] = qs[1];
                              tangential_face_grads_u2_face3[idx0] = qr[2];
                              tangential_face_grads_u2_face3[idx1] = qs[2];
                            }
                        }
                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face4[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face4[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face4[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face4[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face4[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face4[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face4[idx0] = qr[0];
                              tangential_face_grads_u0_face4[idx1] = qs[0];
                              tangential_face_grads_u1_face4[idx0] = qr[1];
                              tangential_face_grads_u1_face4[idx1] = qs[1];
                              tangential_face_grads_u2_face4[idx0] = qr[2];
                              tangential_face_grads_u2_face4[idx1] = qs[2];
                            }
                        }
                        {
                          const int q = tid % n_q;
                          for (int n = 0; n < n_q; ++n)
                            {
                              r_p0[n] = normal_face_values_u0_face5[e * n_q * n_q + q * n_q + n];
                              r_p1[n] = normal_face_values_u1_face5[e * n_q * n_q + q * n_q + n];
                              r_p2[n] = normal_face_values_u2_face5[e * n_q * n_q + q * n_q + n];

                              r_q[n] = co_shape_gradients[n * n_q + q];
                            }

                          Number qr[dim], qs[dim];
                          for (int p = 0; p < n_q; ++p)
                            {
                              for (int d = 0; d < dim; ++d)
                                {
                                  qr[d] = 0;
                                  qs[d] = 0;
                                }
                              for (int n = 0; n < n_q; ++n)
                                {
                                  qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
                                  qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
                                  qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

                                  qs[0] += r_q[n] *
                                           normal_face_values_u0_face5[e * n_q * n_q + n * n_q + p];
                                  qs[1] += r_q[n] *
                                           normal_face_values_u1_face5[e * n_q * n_q + n * n_q + p];
                                  qs[2] += r_q[n] *
                                           normal_face_values_u2_face5[e * n_q * n_q + n * n_q + p];
                                }

                              const int idx0 =
                                e * dim * n_q_total_face + 0 * n_q_total_face + q * n_q + p;
                              const int idx1 =
                                e * dim * n_q_total_face + 1 * n_q_total_face + q * n_q + p;

                              tangential_face_grads_u0_face5[idx0] = qr[0];
                              tangential_face_grads_u0_face5[idx1] = qs[0];
                              tangential_face_grads_u1_face5[idx0] = qr[1];
                              tangential_face_grads_u1_face5[idx1] = qs[1];
                              tangential_face_grads_u2_face5[idx0] = qr[2];
                              tangential_face_grads_u2_face5[idx1] = qs[2];
                            }
                        }
                      }
                  }
              }

              // // ====================================================
              // // PHASE 1: Read from global L vector per component
              // // ====================================================
              // {
              //   for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
              //        tid += blockSize)
              //     {
              //       const int e                  = tid / n_dofs_per_component;
              //       const int local_dof_index_1d = tid % n_dofs_per_component;

              //       const int global_cell_id = eb * nelmtPerBatch + e;

              //       {
              //         const unsigned int dof_x =
              //           dof_indices(0 * n_dofs_per_component + local_dof_index_1d,
              //           global_cell_id);
              //         if (dof_x != numbers::invalid_unsigned_int)
              //           s_uq_0[tid] = vector_in[dof_x];
              //         else
              //           s_uq_0[tid] = 0;
              //       }
              //       {
              //         const unsigned int dof_y =
              //           dof_indices(1 * n_dofs_per_component + local_dof_index_1d,
              //           global_cell_id);

              //         if (dof_y != numbers::invalid_unsigned_int)
              //           s_uq_1[tid] = vector_in[dof_y];
              //         else
              //           s_uq_1[tid] = 0;
              //       }

              //       if constexpr (dim > 2)
              //         {
              //           const unsigned int dof_z =
              //             dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
              //                         global_cell_id);

              //           if (dof_z != numbers::invalid_unsigned_int)
              //             s_uq_2[tid] = vector_in[dof_z];
              //           else
              //             s_uq_2[tid] = 0;
              //         }
              //     }
              //   team_member.team_barrier();
              // }

              // // ====================================================
              // // PHASE 2: Interpolate to quadrature nodes
              // // ====================================================
              // {
              //   // ------------------------ Component 0 (x-direction) ------------------------
              //   // x is normal (basis_n), y and z are tangent (basis_t)
              //   {
              //     // component 0 in x direction
              //     {
              //       constexpr int co_dimension_size = Utilities::pow(n_t, dim - 1);

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int j = tid % co_dimension_size;

              //               for (int i = 0; i < n_n; ++i)
              //                 r_p[i] = s_uq_0[e * n_n * n_t + j * n_n + i];

              //               for (int p = 0; p < n_q; ++p)
              //                 {
              //                   Number tmp = 0;
              //                   for (int i = 0; i < n_n; ++i)
              //                     tmp += shape_values_normal[i * n_q + p] * r_p[i];

              //                   s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int j = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int i = 0; i < n_n; ++i)
              //                 r_p[i] = s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i];

              //               for (int p = 0; p < n_q; ++p)
              //                 {
              //                   Number tmp = 0;
              //                   for (int i = 0; i < n_n; ++i)
              //                     tmp += shape_values_normal[i * n_q + p] * r_p[i];


              //                   s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 0 in y direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int p = tid % co_dimension_size;

              //               for (int j = 0; j < n_t; ++j)
              //                 r_p[j] = s_duq_1[e * n_q * n_t + j * n_q + p];

              //               for (int q = 0; q < n_q; ++q)
              //                 {
              //                   Number tmp = 0;
              //                   for (int j = 0; j < n_t; ++j)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[j];

              //                   s_uq_0[e * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int p = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int j = 0; j < n_t; ++j)
              //                 r_p[j] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q +
              //                 p];

              //               for (int q = 0; q < n_q; ++q)
              //                 {
              //                   Number tmp = 0;
              //                   for (int j = 0; j < n_t; ++j)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[j];

              //                   s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 0 in z direction
              //     {
              //       if constexpr (dim == 3)
              //         {
              //           constexpr int co_dimension_size = n_q * n_q;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int p = (tid % co_dimension_size) / n_q;
              //               const int q = tid % n_q;

              //               for (int k = 0; k < n_t; ++k)
              //                 r_p[k] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q +
              //                 p];

              //               for (int r = 0; r < n_q; ++r)
              //                 {
              //                   Number tmp = 0;
              //                   for (int k = 0; k < n_t; ++k)
              //                     tmp += shape_values_tangent[k * n_q + r] * r_p[k];

              //                   s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }
              //     }
              //   }

              //   // ------------------------ Component 1 (y-direction) ------------------------
              //   // y is normal (basis_n), x and z are tangent (basis_t)
              //   {
              //     // component 1 in x direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int j = tid % co_dimension_size;

              //               for (int i = 0; i < n_t; ++i)
              //                 r_p[i] = s_uq_1[e * n_t * n_n + j * n_t + i];

              //               for (int p = 0; p < n_q; ++p)
              //                 {
              //                   Number tmp = 0;
              //                   for (int i = 0; i < n_t; ++i)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[i];
              //                   s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int j = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int i = 0; i < n_t; ++i)
              //                 r_p[i] = s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i];

              //               for (int p = 0; p < n_q; ++p)
              //                 {
              //                   Number tmp = 0;
              //                   for (int i = 0; i < n_t; ++i)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[i];

              //                   s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 1 in y direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int p = tid % co_dimension_size;

              //               for (int j = 0; j < n_n; ++j)
              //                 r_p[j] = s_duq_1[e * n_q * n_n + j * n_q + p];

              //               for (int q = 0; q < n_q; ++q)
              //                 {
              //                   Number tmp = 0;
              //                   for (int j = 0; j < n_n; ++j)
              //                     tmp += shape_values_normal[j * n_q + q] * r_p[j];

              //                   s_uq_1[e * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int p = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int j = 0; j < n_n; ++j)
              //                 r_p[j] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q +
              //                 p];

              //               for (int q = 0; q < n_q; ++q)
              //                 {
              //                   Number tmp = 0;
              //                   for (int j = 0; j < n_n; ++j)
              //                     tmp += shape_values_normal[j * n_q + q] * r_p[j];

              //                   s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 1 in z direction
              //     {
              //       if constexpr (dim == 3)
              //         {
              //           constexpr int co_dimension_size = n_q * n_q;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int p = (tid % co_dimension_size) / n_q;
              //               const int q = tid % n_q;

              //               for (int k = 0; k < n_t; ++k)
              //                 r_p[k] =
              //                   s_duq_0[e * n_dofs_per_component + k * n_q * n_q + q * n_q + p];

              //               for (int r = 0; r < n_q; ++r)
              //                 {
              //                   Number tmp = 0;
              //                   for (int k = 0; k < n_t; ++k)
              //                     tmp += shape_values_tangent[k * n_q + r] * r_p[k];

              //                   s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }
              //     }
              //   }
              //   {
              //     // ------------------------ Component 2 (x-direction) ------------------------
              //     // z is normal (basis_n), x and y are tangent (basis_t)
              //     if constexpr (dim == 3)
              //       {
              //         // component 2 in x direction
              //         {
              //           constexpr int co_dimension_size = n_t * n_n;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int j = (tid % co_dimension_size) / n_n;
              //               const int k = tid % n_n;

              //               for (int i = 0; i < n_t; ++i)
              //                 r_p[i] = s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i];

              //               for (int p = 0; p < n_q; ++p)
              //                 {
              //                   Number tmp = 0;
              //                   for (int i = 0; i < n_t; ++i)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[i];

              //                   s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }

              //         // component 2 in y direction
              //         {
              //           constexpr int co_dimension_size = n_q * n_n;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int p = (tid % co_dimension_size) / n_n;
              //               const int k = tid % n_n;

              //               for (int j = 0; j < n_t; ++j)
              //                 r_p[j] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q +
              //                 p];

              //               for (int q = 0; q < n_q; ++q)
              //                 {
              //                   Number tmp = 0;
              //                   for (int j = 0; j < n_t; ++j)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[j];

              //                   s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }

              //         // component 2 in z direction
              //         {
              //           constexpr int co_dimension_size = n_q * n_q;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int p = (tid % co_dimension_size) / n_q;
              //               const int q = tid % n_q;

              //               for (int k = 0; k < n_n; ++k)
              //                 r_p[k] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q +
              //                 p];

              //               for (int r = 0; r < n_q; ++r)
              //                 {
              //                   Number tmp = 0;
              //                   for (int k = 0; k < n_n; ++k)
              //                     tmp += shape_values_normal[k * n_q + r] * r_p[k];

              //                   s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }
              //       }
              //   }
              // }

              // // ====================================================
              // // PHASE 3: Evaluate gradients at quadrature nodes
              // // ====================================================

              // {
              //   // 1. evaluate gradients in reference space and multiply by stiffness geometric
              //   // tensor
              //   {
              //     constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
              //     constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
              //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //          tid += blockSize)
              //       {
              //         const int e = tid / co_dimension_size;

              //         //  Base offset for the current element's geometric factors
              //         const int e_offset =
              //           eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
              //           e * symmetric_tensor_dimension * n_q_total;

              //         if constexpr (dim == 2)
              //           {
              //             const int q = tid % co_dimension_size;

              //             for (int n = 0; n < n_q; ++n)
              //               {
              //                 r_p0[n] = s_uq_0[e * n_q * n_q + q * n_q + n];
              //                 r_p1[n] = s_uq_1[e * n_q * n_q + q * n_q + n];

              //                 r_q[n] = co_shape_gradients[n * n_q + q];
              //               }

              //             Number d_G[dim][dim];
              //             Number qr[dim];
              //             Number qs[dim];

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 // Load stiffness geometric tensor
              //                 int index = 0;
              //                 for (int d1 = 0; d1 < dim; ++d1)
              //                   {
              //                     qr[d1] = 0;
              //                     qs[d1] = 0;
              //                     for (int d2 = d1; d2 < dim; ++d2)
              //                       {
              //                         d_G[d1][d2] =
              //                           geometric_tensor_stiffness[e_offset + index * n_q_total +
              //                                                      q * n_q + p];
              //                         if (d2 != d1)
              //                           d_G[d2][d1] = d_G[d1][d2]; // symmetric
              //                         ++index;
              //                       }
              //                   }

              //                 // Multiply by D
              //                 for (int n = 0; n < n_q; ++n)
              //                   {
              //                     qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
              //                     qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];

              //                     qs[0] += r_q[n] * s_uq_0[e * n_q * n_q + n * n_q + p];
              //                     qs[1] += r_q[n] * s_uq_1[e * n_q * n_q + n * n_q + p];
              //                   }

              //                 const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
              //                 const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

              //                 s_duq_0[idx0] = qr[0] * d_G[0][0] + qs[0] * d_G[1][0];
              //                 s_duq_0[idx1] = qr[0] * d_G[0][1] + qs[0] * d_G[1][1];

              //                 s_duq_1[idx0] = qr[1] * d_G[0][0] + qs[1] * d_G[1][0];
              //                 s_duq_1[idx1] = qr[1] * d_G[0][1] + qs[1] * d_G[1][1];
              //               }
              //           }
              //         else if constexpr (dim == 3)
              //           {
              //             const int q = (tid % co_dimension_size) / n_q;
              //             const int r = tid % n_q;

              //             for (int n = 0; n < n_q; ++n)

              //               {
              //                 r_p0[n] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q +
              //                 n]; r_p1[n] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q
              //                 + n]; r_p2[n] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q *
              //                 n_q + n];

              //                 r_q[n] = co_shape_gradients[n * n_q + q];
              //                 r_r[n] = co_shape_gradients[n * n_q + r];
              //               }

              //             Number d_G[dim][dim];
              //             Number qr[dim];
              //             Number qs[dim];
              //             Number qt[dim];

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 // Load stiffness geometric tensor
              //                 int index = 0;
              //                 for (int d1 = 0; d1 < dim; ++d1)
              //                   {
              //                     qr[d1] = 0;
              //                     qs[d1] = 0;
              //                     qt[d1] = 0;
              //                     for (int d2 = d1; d2 < dim; ++d2)
              //                       {
              //                         d_G[d1][d2] =
              //                           geometric_tensor_stiffness[e_offset + index * n_q_total +
              //                                                      r * n_q * n_q + q * n_q + p];
              //                         if (d2 != d1)
              //                           d_G[d2][d1] = d_G[d1][d2]; // symmetric
              //                         ++index;
              //                       }
              //                   }
              //                 // Multiply by D
              //                 for (int n = 0; n < n_q; ++n)
              //                   {
              //                     qr[0] += co_shape_gradients[n * n_q + p] * r_p0[n];
              //                     qr[1] += co_shape_gradients[n * n_q + p] * r_p1[n];
              //                     qr[2] += co_shape_gradients[n * n_q + p] * r_p2[n];

              //                     qs[0] +=
              //                       r_q[n] * s_uq_0[e * n_q_total + r * n_q * n_q + n * n_q + p];
              //                     qs[1] +=
              //                       r_q[n] * s_uq_1[e * n_q_total + r * n_q * n_q + n * n_q + p];
              //                     qs[2] +=
              //                       r_q[n] * s_uq_2[e * n_q_total + r * n_q * n_q + n * n_q + p];

              //                     qt[0] +=
              //                       r_r[n] * s_uq_0[e * n_q_total + n * n_q * n_q + q * n_q + p];
              //                     qt[1] +=
              //                       r_r[n] * s_uq_1[e * n_q_total + n * n_q * n_q + q * n_q + p];
              //                     qt[2] +=
              //                       r_r[n] * s_uq_2[e * n_q_total + n * n_q * n_q + q * n_q + p];
              //                   }

              //                 const int idx0 =
              //                   e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;
              //                 const int idx1 =
              //                   e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;
              //                 const int idx2 =
              //                   e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;

              //                 s_duq_0[idx0] =
              //                   qr[0] * d_G[0][0] + qs[0] * d_G[1][0] + qt[0] * d_G[2][0];
              //                 s_duq_0[idx1] =
              //                   qr[0] * d_G[0][1] + qs[0] * d_G[1][1] + qt[0] * d_G[2][1];
              //                 s_duq_0[idx2] =
              //                   qr[0] * d_G[0][2] + qs[0] * d_G[1][2] + qt[0] * d_G[2][2];

              //                 s_duq_1[idx0] =
              //                   qr[1] * d_G[0][0] + qs[1] * d_G[1][0] + qt[1] * d_G[2][0];
              //                 s_duq_1[idx1] =
              //                   qr[1] * d_G[0][1] + qs[1] * d_G[1][1] + qt[1] * d_G[2][1];
              //                 s_duq_1[idx2] =
              //                   qr[1] * d_G[0][2] + qs[1] * d_G[1][2] + qt[1] * d_G[2][2];

              //                 s_duq_2[idx0] =
              //                   qr[2] * d_G[0][0] + qs[2] * d_G[1][0] + qt[2] * d_G[2][0];
              //                 s_duq_2[idx1] =
              //                   qr[2] * d_G[0][1] + qs[2] * d_G[1][1] + qt[2] * d_G[2][1];
              //                 s_duq_2[idx2] =
              //                   qr[2] * d_G[0][2] + qs[2] * d_G[1][2] + qt[2] * d_G[2][2];
              //               }
              //           }
              //       }
              //     team_member.team_barrier();
              //   }

              //   // 2. multiply by the mass geometric tensor
              //   {
              //     constexpr int co_dimension_size          = Utilities::pow(n_q, dim - 1);
              //     constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

              //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //          tid += blockSize)
              //       {
              //         const int e = tid / co_dimension_size;

              //         //  Base offset for the current element's geometric factors
              //         const int e_offset =
              //           eb * nelmtPerBatch * symmetric_tensor_dimension * n_q_total +
              //           e * symmetric_tensor_dimension * n_q_total;

              //         Number d_G[dim][dim];
              //         Number qr[dim];
              //         Number qs[dim];

              //         Number u[dim];

              //         if constexpr (dim == 2)
              //           {
              //             const int q = tid % co_dimension_size;

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 int index = 0;
              //                 for (int d1 = 0; d1 < dim; ++d1)
              //                   {
              //                     for (int d2 = d1; d2 < dim; ++d2)
              //                       {
              //                         d_G[d1][d2] =
              //                           geometric_tensor_mass[e_offset + index * n_q_total +
              //                                                 q * n_q + p];
              //                         if (d2 != d1)
              //                           d_G[d2][d1] = d_G[d1][d2]; // symmetric
              //                         ++index;
              //                       }

              //                     qr[d1] =
              //                       factor_laplace *
              //                       s_duq_0[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
              //                     qs[d1] =
              //                       factor_laplace *
              //                       s_duq_1[e * dim * n_q_total + d1 * n_q_total + q * n_q + p];
              //                   }

              //                 u[0] = factor_mass * s_uq_0[e * n_q_total + q * n_q + p];
              //                 u[1] = factor_mass * s_uq_1[e * n_q_total + q * n_q + p];

              //                 const int idx0 = e * dim * n_q_total + 0 * n_q_total + q * n_q + p;
              //                 const int idx1 = e * dim * n_q_total + 1 * n_q_total + q * n_q + p;

              //                 s_duq_0[idx0] = d_G[0][0] * qr[0] + d_G[0][1] * qs[0];
              //                 s_duq_0[idx1] = d_G[0][0] * qr[1] + d_G[0][1] * qs[1];

              //                 s_duq_1[idx0] = d_G[1][0] * qr[0] + d_G[1][1] * qs[0];
              //                 s_duq_1[idx1] = d_G[1][0] * qr[1] + d_G[1][1] * qs[1];

              //                 // also apply mass tensor to the value itself
              //                 s_uq_0[e * n_q_total + q * n_q + p] =
              //                   d_G[0][0] * u[0] + d_G[0][1] * u[1];
              //                 s_uq_1[e * n_q_total + q * n_q + p] =
              //                   d_G[1][0] * u[0] + d_G[1][1] * u[1];
              //               }
              //           }

              //         else if constexpr (dim == 3)
              //           {
              //             Number qt[dim];

              //             const int q = (tid % co_dimension_size) / n_q;
              //             const int r = tid % n_q;

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 int index = 0;
              //                 for (int d1 = 0; d1 < dim; ++d1)
              //                   {
              //                     for (int d2 = d1; d2 < dim; ++d2)
              //                       {
              //                         d_G[d1][d2] =
              //                           geometric_tensor_mass[e_offset + index * n_q_total +
              //                                                 r * n_q * n_q + q * n_q + p];
              //                         if (d2 != d1)
              //                           d_G[d2][d1] = d_G[d1][d2]; // symmetric
              //                         ++index;
              //                       }
              //                     qr[d1] =
              //                       factor_laplace * s_duq_0[e * dim * n_q_total + d1 * n_q_total
              //                       +
              //                                                r * n_q * n_q + q * n_q + p];
              //                     qs[d1] =
              //                       factor_laplace * s_duq_1[e * dim * n_q_total + d1 * n_q_total
              //                       +
              //                                                r * n_q * n_q + q * n_q + p];
              //                     qt[d1] =
              //                       factor_laplace * s_duq_2[e * dim * n_q_total + d1 * n_q_total
              //                       +
              //                                                r * n_q * n_q + q * n_q + p];
              //                   }

              //                 u[0] =
              //                   factor_mass * s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q +
              //                   p];
              //                 u[1] =
              //                   factor_mass * s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q +
              //                   p];
              //                 u[2] =
              //                   factor_mass * s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q +
              //                   p];

              //                 const int idx0 =
              //                   e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;
              //                 const int idx1 =
              //                   e * dim * n_q_total + 1 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;
              //                 const int idx2 =
              //                   e * dim * n_q_total + 2 * n_q_total + r * n_q * n_q + q * n_q +
              //                   p;

              //                 s_duq_0[idx0] =
              //                   d_G[0][0] * qr[0] + d_G[0][1] * qs[0] + d_G[0][2] * qt[0];
              //                 s_duq_0[idx1] =
              //                   d_G[0][0] * qr[1] + d_G[0][1] * qs[1] + d_G[0][2] * qt[1];
              //                 s_duq_0[idx2] =
              //                   d_G[0][0] * qr[2] + d_G[0][1] * qs[2] + d_G[0][2] * qt[2];

              //                 s_duq_1[idx0] =
              //                   d_G[1][0] * qr[0] + d_G[1][1] * qs[0] + d_G[1][2] * qt[0];
              //                 s_duq_1[idx1] =
              //                   d_G[1][0] * qr[1] + d_G[1][1] * qs[1] + d_G[1][2] * qt[1];
              //                 s_duq_1[idx2] =
              //                   d_G[1][0] * qr[2] + d_G[1][1] * qs[2] + d_G[1][2] * qt[2];

              //                 s_duq_2[idx0] =
              //                   d_G[2][0] * qr[0] + d_G[2][1] * qs[0] + d_G[2][2] * qt[0];
              //                 s_duq_2[idx1] =
              //                   d_G[2][0] * qr[1] + d_G[2][1] * qs[1] + d_G[2][2] * qt[1];
              //                 s_duq_2[idx2] =
              //                   d_G[2][0] * qr[2] + d_G[2][1] * qs[2] + d_G[2][2] * qt[2];

              //                 // also apply mass tensor to the value itself
              //                 s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p] =
              //                   d_G[0][0] * u[0] + d_G[0][1] * u[1] + d_G[0][2] * u[2];
              //                 s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p] =
              //                   d_G[1][0] * u[0] + d_G[1][1] * u[1] + d_G[1][2] * u[2];
              //                 s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p] =
              //                   d_G[2][0] * u[0] + d_G[2][1] * u[1] + d_G[2][2] * u[2];
              //               }
              //           }
              //       }
              //     team_member.team_barrier();
              //   }

              //   // 3. integrate, i.e apply D^T
              //   {
              //     constexpr int co_dimension_size = Utilities::pow(n_q, dim - 1);

              //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //          tid += blockSize)
              //       {
              //         const int e = tid / co_dimension_size;

              //         if constexpr (dim == 2)
              //           {
              //             const int q = tid % co_dimension_size;

              //             // copy to register
              //             for (int n = 0; n < n_q; ++n)
              //               {
              //                 const int idx_0 = e * dim * n_q_total + 0 * n_q_total + q * n_q +
              //                 n;

              //                 r_p0[n] = s_duq_0[idx_0];
              //                 r_p1[n] = s_duq_1[idx_0];

              //                 r_q[n] = co_shape_gradients[q * n_q + n];
              //               }

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 Number tmp0 = 0, tmp1 = 0;

              //                 for (unsigned int n = 0; n < n_q; ++n)
              //                   {
              //                     tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
              //                     tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
              //                   }

              //                 for (unsigned int n = 0; n < n_q; ++n)
              //                   {
              //                     const int idx_1 =
              //                       e * dim * n_q_total + 1 * n_q_total + n * n_q + p;
              //                     tmp0 += s_duq_0[idx_1] * r_q[n];
              //                     tmp1 += s_duq_1[idx_1] * r_q[n];
              //                   }

              //                 s_uq_0[e * n_q_total + q * n_q + p] += tmp0;
              //                 s_uq_1[e * n_q_total + q * n_q + p] += tmp1;
              //               }
              //           }
              //         else if constexpr (dim == 3)
              //           {
              //             const int q = (tid % co_dimension_size) / n_q;
              //             const int r = tid % n_q;

              //             // copy to register
              //             for (int n = 0; n < n_q; ++n)
              //               {
              //                 const int idx_0 =
              //                   e * dim * n_q_total + 0 * n_q_total + r * n_q * n_q + q * n_q +
              //                   n;

              //                 r_p0[n] = s_duq_0[idx_0];
              //                 r_p1[n] = s_duq_1[idx_0];
              //                 r_p2[n] = s_duq_2[idx_0];

              //                 r_q[n] = co_shape_gradients[q * n_q + n];
              //                 r_r[n] = co_shape_gradients[r * n_q + n];
              //               }

              //             for (int p = 0; p < n_q; ++p)
              //               {
              //                 Number tmp0 = 0, tmp1 = 0, tmp2 = 0;

              //                 for (unsigned int n = 0; n < n_q; ++n)
              //                   {
              //                     tmp0 += r_p0[n] * co_shape_gradients[p * n_q + n];
              //                     tmp1 += r_p1[n] * co_shape_gradients[p * n_q + n];
              //                     tmp2 += r_p2[n] * co_shape_gradients[p * n_q + n];
              //                   }

              //                 for (unsigned int n = 0; n < n_q; ++n)
              //                   {
              //                     const int idx_1 = e * dim * n_q_total + 1 * n_q_total +
              //                                       r * n_q * n_q + n * n_q + p;

              //                     tmp0 += s_duq_0[idx_1] * r_q[n];
              //                     tmp1 += s_duq_1[idx_1] * r_q[n];
              //                     tmp2 += s_duq_2[idx_1] * r_q[n];
              //                   }

              //                 for (unsigned int n = 0; n < n_q; ++n)
              //                   {
              //                     const int idx_2 = e * dim * n_q_total + 2 * n_q_total +
              //                                       n * n_q * n_q + q * n_q + p;

              //                     tmp0 += s_duq_0[idx_2] * r_r[n];
              //                     tmp1 += s_duq_1[idx_2] * r_r[n];
              //                     tmp2 += s_duq_2[idx_2] * r_r[n];
              //                   }

              //                 s_uq_0[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp0;
              //                 s_uq_1[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp1;
              //                 s_uq_2[e * n_q_total + r * n_q * n_q + q * n_q + p] += tmp2;
              //               }
              //           }
              //       }
              //   }
              //   team_member.team_barrier();
              // }


              // // ====================================================
              // // PHASE 4: Project back to Nodes
              // // ====================================================
              // {
              //   // ------------------------ Component 0 (x-direction) ------------------------
              //   // x is normal (basis_n), y and z are tangent (basis_t)
              //   {
              //     // component 0 in z direction
              //     if constexpr (dim == 3)
              //       {
              //         constexpr int co_dimension_size = n_q * n_q;

              //         for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //              tid += blockSize)
              //           {
              //             const int e = tid / co_dimension_size;

              //             const int p = (tid % co_dimension_size) / n_q;
              //             const int q = tid % n_q;

              //             for (int r = 0; r < n_q; ++r)
              //               r_p[r] = s_uq_0[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

              //             for (int k = 0; k < n_t; ++k)
              //               {
              //                 Number tmp = 0;
              //                 for (int r = 0; r < n_q; ++r)
              //                   tmp += shape_values_tangent[k * n_q + r] * r_p[r];

              //                 s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
              //               }
              //           }
              //         team_member.team_barrier();
              //       }

              //     // component 0 in y direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int p = tid % co_dimension_size;

              //               for (int q = 0; q < n_q; ++q)
              //                 r_p[q] = s_uq_0[e * n_q * n_q + q * n_q + p];

              //               for (int j = 0; j < n_t; ++j)
              //                 {
              //                   Number tmp = 0;
              //                   for (int q = 0; q < n_q; ++q)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[q];

              //                   s_duq_1[e * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int p = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int q = 0; q < n_q; ++q)
              //                 r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q +
              //                 p];

              //               for (int j = 0; j < n_t; ++j)
              //                 {
              //                   Number tmp = 0;
              //                   for (int q = 0; q < n_q; ++q)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[q];

              //                   s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 0 in x direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_t : n_t * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int j = tid % co_dimension_size;

              //               for (int p = 0; p < n_q; ++p)
              //                 r_p[p] = s_duq_1[e * n_q * n_t + j * n_q + p];

              //               for (int i = 0; i < n_n; ++i)
              //                 {
              //                   Number tmp = 0;
              //                   for (int p = 0; p < n_q; ++p)
              //                     tmp += shape_values_normal[i * n_q + p] * r_p[p];

              //                   s_uq_0[e * n_n * n_t + j * n_n + i] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int j = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int p = 0; p < n_q; ++p)
              //                 r_p[p] = s_duq_1[e * n_q * n_t * n_t + k * n_q * n_t + j * n_q +
              //                 p];

              //               for (int i = 0; i < n_n; ++i)
              //                 {
              //                   Number tmp = 0;
              //                   for (int p = 0; p < n_q; ++p)
              //                     tmp += shape_values_normal[i * n_q + p] * r_p[p];

              //                   s_uq_0[e * n_n * n_t * n_t + k * n_n * n_t + j * n_n + i] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }
              //   }

              //   // ------------------------ Component 1 (y-direction) ------------------------
              //   // y is normal (basis_n), x and z are tangent (basis_t)
              //   {
              //     // component 1 in z direction
              //     if constexpr (dim == 3)
              //       {
              //         constexpr int co_dimension_size = n_q * n_q;

              //         for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //              tid += blockSize)
              //           {
              //             const int e = tid / co_dimension_size;

              //             const int p = (tid % co_dimension_size) / n_q;
              //             const int q = tid % n_q;

              //             for (int r = 0; r < n_q; ++r)
              //               r_p[r] = s_uq_1[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

              //             for (int k = 0; k < n_t; ++k)
              //               {
              //                 Number tmp = 0;
              //                 for (int r = 0; r < n_q; ++r)
              //                   tmp += shape_values_tangent[k * n_q + r] * r_p[r];

              //                 s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q + p] = tmp;
              //               }
              //           }
              //         team_member.team_barrier();
              //       }

              //     // component 1 in y direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_q : n_q * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int p = tid % co_dimension_size;

              //               for (int q = 0; q < n_q; ++q)
              //                 r_p[q] = s_uq_1[e * n_q * n_q + q * n_q + p];

              //               for (int j = 0; j < n_n; ++j)
              //                 {
              //                   Number tmp = 0;
              //                   for (int q = 0; q < n_q; ++q)
              //                     tmp += shape_values_normal[j * n_q + q] * r_p[q];

              //                   s_duq_1[e * n_q * n_n + j * n_q + p] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int p = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int q = 0; q < n_q; ++q)
              //                 r_p[q] = s_duq_0[e * n_q * n_q * n_t + k * n_q * n_q + q * n_q +
              //                 p];

              //               for (int j = 0; j < n_n; ++j)
              //                 {
              //                   Number tmp = 0;
              //                   for (int q = 0; q < n_q; ++q)
              //                     tmp += shape_values_normal[j * n_q + q] * r_p[q];

              //                   s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q + p] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }

              //     // component 1 in x direction
              //     {
              //       constexpr int co_dimension_size = (dim == 2) ? n_n : n_n * n_t;

              //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //            tid += blockSize)
              //         {
              //           const int e = tid / co_dimension_size;

              //           if constexpr (dim == 2)
              //             {
              //               const int j = tid % co_dimension_size;

              //               for (int p = 0; p < n_q; ++p)
              //                 r_p[p] = s_duq_1[e * n_q * n_n + j * n_q + p];

              //               for (int i = 0; i < n_t; ++i)
              //                 {
              //                   Number tmp = 0;
              //                   for (int p = 0; p < n_q; ++p)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[p];

              //                   s_uq_1[e * n_t * n_n + j * n_t + i] = tmp;
              //                 }
              //             }
              //           else if constexpr (dim == 3)
              //             {
              //               const int j = (tid % co_dimension_size) / n_t;
              //               const int k = tid % n_t;

              //               for (int p = 0; p < n_q; ++p)
              //                 r_p[p] = s_duq_1[e * n_q * n_n * n_t + k * n_q * n_n + j * n_q +
              //                 p];

              //               for (int i = 0; i < n_t; ++i)
              //                 {
              //                   Number tmp = 0;
              //                   for (int p = 0; p < n_q; ++p)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[p];

              //                   s_uq_1[e * n_t * n_n * n_t + k * n_t * n_n + j * n_t + i] = tmp;
              //                 }
              //             }
              //         }
              //       team_member.team_barrier();
              //     }
              //   }

              //   // ------------------------ Component 2 (z-direction) ------------------------
              //   // z is normal (basis_n), x and y are tangent (basis_t)
              //   if constexpr (dim == 3)
              //     {
              //       // component 2 in z direction
              //       {
              //         constexpr int co_dimension_size = n_q * n_q;

              //         for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //              tid += blockSize)
              //           {
              //             const int e = tid / co_dimension_size;

              //             const int p = (tid % co_dimension_size) / n_q;
              //             const int q = tid % n_q;

              //             for (int r = 0; r < n_q; ++r)
              //               r_p[r] = s_uq_2[e * n_q * n_q * n_q + r * n_q * n_q + q * n_q + p];

              //             for (int k = 0; k < n_n; ++k)
              //               {
              //                 Number tmp = 0;
              //                 for (int r = 0; r < n_q; ++r)
              //                   tmp += shape_values_normal[k * n_q + r] * r_p[r];

              //                 s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q + p] = tmp;
              //               }
              //           }
              //         team_member.team_barrier();
              //       }

              //       // component 2 in y direction
              //       {
              //         constexpr int co_dimension_size = n_q * n_n;

              //         for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //              tid += blockSize)
              //           {
              //             const int e = tid / co_dimension_size;

              //             {
              //               const int p = (tid % co_dimension_size) / n_n;
              //               const int k = tid % n_n;

              //               for (int q = 0; q < n_q; ++q)
              //                 r_p[q] = s_duq_0[e * n_q * n_q * n_n + k * n_q * n_q + q * n_q +
              //                 p];

              //               for (int j = 0; j < n_t; ++j)
              //                 {
              //                   Number tmp = 0;
              //                   for (int q = 0; q < n_q; ++q)
              //                     tmp += shape_values_tangent[j * n_q + q] * r_p[q];

              //                   s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q + p] = tmp;
              //                 }
              //             }
              //             team_member.team_barrier();
              //           }

              //         // component 2 in x direction
              //         {
              //           constexpr int co_dimension_size = n_t * n_n;

              //           for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
              //                tid += blockSize)
              //             {
              //               const int e = tid / co_dimension_size;

              //               const int j = (tid % co_dimension_size) / n_n;
              //               const int k = tid % n_n;

              //               for (int p = 0; p < n_q; ++p)
              //                 r_p[p] = s_duq_1[e * n_q * n_t * n_n + k * n_q * n_t + j * n_q +
              //                 p];

              //               for (int i = 0; i < n_t; ++i)
              //                 {
              //                   Number tmp = 0;
              //                   for (int p = 0; p < n_q; ++p)
              //                     tmp += shape_values_tangent[i * n_q + p] * r_p[p];

              //                   s_uq_2[e * n_t * n_t * n_n + k * n_t * n_t + j * n_t + i] = tmp;
              //                 }
              //             }
              //           team_member.team_barrier();
              //         }
              //       }
              //     }
              // }

              // // ====================================================
              // // PHASE 5: Write the results to the global L vector.
              // // ====================================================

              // {
              //   for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_per_component;
              //        tid += blockSize)
              //     {
              //       const int e                  = tid / n_dofs_per_component;
              //       const int local_dof_index_1d = tid % n_dofs_per_component;

              //       const int global_cell_id = eb * nelmtPerBatch + e;

              //       {
              //         const unsigned int dof_x =
              //           dof_indices(0 * n_dofs_per_component + local_dof_index_1d,
              //           global_cell_id);

              //         if (dof_x != numbers::invalid_unsigned_int)
              //           Kokkos::atomic_add(&vector_out[dof_x], s_uq_0[tid]);
              //       }
              //       {
              //         const unsigned int dof_y =
              //           dof_indices(1 * n_dofs_per_component + local_dof_index_1d,
              //           global_cell_id);

              //         if (dof_y != numbers::invalid_unsigned_int)
              //           Kokkos::atomic_add(&vector_out[dof_y], s_uq_1[tid]);
              //       }

              //       if constexpr (dim > 2)
              //         {
              //           const unsigned int dof_z =
              //             dof_indices(2 * n_dofs_per_component + local_dof_index_1d,
              //                         global_cell_id);

              //           if (dof_z != numbers::invalid_unsigned_int)
              //             Kokkos::atomic_add(&vector_out[dof_z], s_uq_2[tid]);
              //         }
              //     }
              //   team_member.team_barrier();
              // }

              eb += team_member.league_size();
            }
        });

      Kokkos::fence();
    }


  } // namespace RT
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif