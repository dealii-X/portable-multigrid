#ifndef laplace_dg_kokkos_kernels_h
#define laplace_dg_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace DG
  {

    constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename Number>
    void
    compute_cell(
      const DeviceView<Number> shape_values,
      const DeviceView<Number> co_shape_gradients,
      const DeviceView<Number> geometric_transformation_cell,
      const DeviceView<Number> vector_in,
      DeviceView<Number>       vector_out,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
                                                                   interpolate_quad_to_boundary,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_values_at_quads,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads,
      const DoFIndicesView                                         dof_indices,
      const unsigned int                                           n_cells,
      const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total = Utilities::pow(nq, dim);

      constexpr int nm_total = Utilities::pow(nm, dim);

      constexpr int nq_total_per_face = Utilities::pow(nq, dim - 1);

      constexpr int n_scratch_arrays = 1 + dim; // values and gradients in each direction

      const int nelmt = n_cells;

      int nelmtPerBatch = 1;

      if (n_cells_per_batch == numbers::invalid_unsigned_int)
        nelmtPerBatch = (shmemPerBlock / (n_scratch_arrays * nq_total * sizeof(Number)));
      else
        nelmtPerBatch = n_cells_per_batch;

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 1) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));

      // std::cout << "nelmt = " << nelmt << ", nelmtPerBatch = " << nelmtPerBatch
      //           << ", numBlocks = " << numBlocks << ", threadsPerBlock = " << threadsPerBlock
      //           << std::endl;

      {
        const int ssize = nm * nq + // shape values
                          nq * nq + // co-shape gradients
                          4 * nq +  // interpolate quad to boundary
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total; // working scratch arrays: scratch_values, scratch_grads_0,
                                      // scratch_grads_1, scratch_grads_2


        const int shmem_size = ssize * sizeof(Number);

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

            Number *s_quad_to_boundary_value_0 = s_co_shape_gradients + nq * nq;
            Number *s_quad_to_boundary_value_1 = s_quad_to_boundary_value_0 + nq;
            Number *s_quad_to_boundary_grad_0  = s_quad_to_boundary_value_1 + nq;
            Number *s_quad_to_boundary_grad_1  = s_quad_to_boundary_grad_0 + nq;

            Number *scratch_values  = s_quad_to_boundary_grad_1 + nq;
            Number *scratch_grads_0 = scratch_values + nelmtPerBatch * nq_total;
            Number *scratch_grads_1 = scratch_grads_0 + nelmtPerBatch * nq_total;

            Number *scratch_grads_2;
            if (dim == 3)
              scratch_grads_2 = scratch_grads_1 + nelmtPerBatch * nq_total;


            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = co_shape_gradients[tid];
              }

            for (int tid = threadIdx; tid < nq; tid += blockSize)
              {
                s_quad_to_boundary_value_0[tid] = interpolate_quad_to_boundary(0, tid, 0);
                s_quad_to_boundary_value_1[tid] = interpolate_quad_to_boundary(0, tid, 1);

                s_quad_to_boundary_grad_0[tid] = interpolate_quad_to_boundary(1, tid, 0);
                s_quad_to_boundary_grad_1[tid] = interpolate_quad_to_boundary(1, tid, 1);
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

                // step-1 : Copy from in to the scratch values
                {
                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int global_cell_index = eb * nelmtPerBatch + e;

                      if (dim == 2)
                        {
                          const int i = tid % nm;

                          for (int j = 0; j < nm; ++j)
                            {
                              const int local_idx = j * nm + i;

                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              if (dof_index == numbers::invalid_unsigned_int)
                                scratch_values[e * nm_total + local_idx] = 0;
                              else
                                scratch_values[e * nm_total + local_idx] = vector_in[dof_index];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int j = (tid % co_dimension_size) / nm;
                          const int i = tid % nm;

                          for (int k = 0; k < nm; ++k)
                            {
                              const int local_idx = k * nm * nm + j * nm + i;

                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              if (dof_index == numbers::invalid_unsigned_int)
                                scratch_values[e * nm_total + local_idx] = 0;
                              else
                                scratch_values[e * nm_total + local_idx] = vector_in[dof_index];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-2: interpolate dof values to quadrature points in each direction
                {
                  // direction 0
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
                                r_p[i] = scratch_values[e * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nm;
                            const int j = tid % nm;

                            for (int i = 0; i < nm; ++i)
                              {
                                r_p[i] =
                                  scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 1
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
                                r_p[j] = scratch_grads_0[e * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_p[j];
                                  }

                                scratch_values[e * nq * nq + q * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nq;
                            const int p = tid % nq;

                            for (int j = 0; j < nm; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[j];
                                  }

                                scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = nq * nq;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int k = 0; k < nm; ++k)
                            {
                              r_r[k] = scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p];
                            }
                          for (int r = 0; r < nq; ++r)
                            {
                              Number tmp = 0.0;

                              for (int k = 0; k < nm; ++k)
                                {
                                  tmp += s_shape_values[k * nq + r] * r_r[k];
                                }

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }
                }

                // step-3: interpolate values to faces and write them to global storage
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nq_total_per_face;
                       tid += blockSize)
                    {
                      const int e = tid / nq_total_per_face;

                      const int global_cell_id = eb * nelmtPerBatch + e;

                      if (dim == 2)
                        {
                          const int m = tid % nq_total_per_face;

                          const int offset_0 = e * nq * nq + m * nq;
                          const int offset_1 = e * nq * nq + m;

                          for (int n = 0; n < nq; ++n)
                            {
                              r_p[n] = scratch_values[offset_0 + n];      // x direction
                              r_q[n] = scratch_values[offset_1 + n * nq]; // y direction
                            }

                          Number v_x[2], d_x[2], v_y[2], d_y[2];
                          for (int i = 0; i < 2; ++i)
                            {
                              v_x[i] = 0;
                              d_x[i] = 0;

                              v_y[i] = 0;
                              d_y[i] = 0;
                            }

                          for (int n = 0; n < nq; ++n)
                            {
                              v_x[0] += s_quad_to_boundary_value_0[n] * r_p[n];
                              d_x[0] += s_quad_to_boundary_grad_0[n] * r_p[n];

                              v_x[1] += s_quad_to_boundary_value_1[n] * r_p[n];
                              d_x[1] += s_quad_to_boundary_grad_1[n] * r_p[n];

                              v_y[0] += s_quad_to_boundary_value_0[n] * r_q[n];
                              d_y[0] += s_quad_to_boundary_grad_0[n] * r_q[n];

                              v_y[1] += s_quad_to_boundary_value_1[n] * r_q[n];
                              d_y[1] += s_quad_to_boundary_grad_1[n] * r_q[n];
                            }


                          const int local_q_id = m;

                          face_values_at_quads(local_q_id, 0, global_cell_id)             = v_x[0];
                          face_normal_derivatives_at_quads(local_q_id, 0, global_cell_id) = d_x[0];

                          face_values_at_quads(local_q_id, 1, global_cell_id)             = v_x[1];
                          face_normal_derivatives_at_quads(local_q_id, 1, global_cell_id) = d_x[1];

                          face_values_at_quads(local_q_id, 2, global_cell_id)             = v_y[0];
                          face_normal_derivatives_at_quads(local_q_id, 2, global_cell_id) = d_y[0];

                          face_values_at_quads(local_q_id, 3, global_cell_id)             = v_y[1];
                          face_normal_derivatives_at_quads(local_q_id, 3, global_cell_id) = d_y[1];
                        }
                      else if (dim == 3)
                        {
                          const int m2 = (tid % nq_total_per_face) / nq;
                          const int m1 = tid % nq;

                          const int offset_0 = e * nq * nq * nq + m2 * nq * nq + m1 * nq;
                          const int offset_1 = e * nq * nq * nq + m2 * nq * nq + m1;
                          const int offset_2 = e * nq * nq * nq + m2 * nq + m1;

                          for (int n = 0; n < nq; ++n)
                            {
                              r_p[n] = scratch_values[offset_0 + n];           // x direction
                              r_q[n] = scratch_values[offset_1 + n * nq];      // y direction
                              r_r[n] = scratch_values[offset_2 + n * nq * nq]; // z direction
                            }

                          Number v_x[2], d_x[2], v_y[2], d_y[2], v_z[2], d_z[2];
                          for (int i = 0; i < 2; ++i)
                            {
                              v_x[i] = 0;
                              d_x[i] = 0;

                              v_y[i] = 0;
                              d_y[i] = 0;

                              v_z[i] = 0;
                              d_z[i] = 0;
                            }

                          for (int n = 0; n < nq; ++n)
                            {
                              v_x[0] += s_quad_to_boundary_value_0[n] * r_p[n];
                              d_x[0] += s_quad_to_boundary_grad_0[n] * r_p[n];

                              v_x[1] += s_quad_to_boundary_value_1[n] * r_p[n];
                              d_x[1] += s_quad_to_boundary_grad_1[n] * r_p[n];

                              v_y[0] += s_quad_to_boundary_value_0[n] * r_q[n];
                              d_y[0] += s_quad_to_boundary_grad_0[n] * r_q[n];

                              v_y[1] += s_quad_to_boundary_value_1[n] * r_q[n];
                              d_y[1] += s_quad_to_boundary_grad_1[n] * r_q[n];

                              v_z[0] += s_quad_to_boundary_value_0[n] * r_r[n];
                              d_z[0] += s_quad_to_boundary_grad_0[n] * r_r[n];

                              v_z[1] += s_quad_to_boundary_value_1[n] * r_r[n];
                              d_z[1] += s_quad_to_boundary_grad_1[n] * r_r[n];
                            }

                          const int local_q_id = m2 * nq + m1;

                          face_values_at_quads(local_q_id, 0, global_cell_id)             = v_x[0];
                          face_normal_derivatives_at_quads(local_q_id, 0, global_cell_id) = d_x[0];

                          face_values_at_quads(local_q_id, 1, global_cell_id)             = v_x[1];
                          face_normal_derivatives_at_quads(local_q_id, 1, global_cell_id) = d_x[1];

                          face_values_at_quads(local_q_id, 2, global_cell_id)             = v_y[0];
                          face_normal_derivatives_at_quads(local_q_id, 2, global_cell_id) = d_y[0];

                          face_values_at_quads(local_q_id, 3, global_cell_id)             = v_y[1];
                          face_normal_derivatives_at_quads(local_q_id, 3, global_cell_id) = d_y[1];

                          face_values_at_quads(local_q_id, 4, global_cell_id)             = v_z[0];
                          face_normal_derivatives_at_quads(local_q_id, 4, global_cell_id) = d_z[0];

                          face_values_at_quads(local_q_id, 5, global_cell_id)             = v_z[1];
                          face_normal_derivatives_at_quads(local_q_id, 5, global_cell_id) = d_z[1];
                        }
                    }
                  team_member.team_barrier();
                }

                // step-4: apply geometric factors and compute stiffness contributions at
                // quadrature points
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int global_cell_id = eb * nelmtPerBatch + e;

                      const int tensor_offset =
                        global_cell_id * symmetric_tensor_dimension * nq_total;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                            }

                          Number Grr, Grs, Gss;
                          Number qr, qs;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;


                              const int local_q_idx = q * nq + p;

                              // Load Geometric Factors, coalesced access
                              Grr = geometric_transformation_cell[tensor_offset + 0 * nq_total +
                                                                  local_q_idx];
                              Grs = geometric_transformation_cell[tensor_offset + 1 * nq_total +
                                                                  local_q_idx];
                              Gss = geometric_transformation_cell[tensor_offset + 2 * nq_total +
                                                                  local_q_idx];

                              // Multiply by D
                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] * scratch_values[e * nq * nq + n * nq + p];
                                }

                              const int idx = e * nq * nq + q * nq + p;

                              // Apply chain rule
                              scratch_grads_0[idx] = Grr * qr + Grs * qs;
                              scratch_grads_1[idx] = Grs * qr + Gss * qs;
                            }
                        }
                      else if (dim == 3)
                        {
                          int r = (tid % co_dimension_size) / nq;
                          int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                              r_r[n] = s_co_shape_gradients[n * nq + r];
                            }

                          Number Grr, Grs, Grt, Gss, Gst, Gtt;
                          Number qr, qs, qt;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;

                              const int local_q_idx = r * nq * nq + q * nq + p;

                              // Load Geometric Factors, coalesced access
                              Grr = geometric_transformation_cell[tensor_offset + 0 * nq_total +
                                                                  local_q_idx];

                              Grs = geometric_transformation_cell[tensor_offset + 1 * nq_total +
                                                                  local_q_idx];

                              Grt = geometric_transformation_cell[tensor_offset + 2 * nq_total +
                                                                  local_q_idx];

                              Gss = geometric_transformation_cell[tensor_offset + 3 * nq_total +
                                                                  local_q_idx];

                              Gst = geometric_transformation_cell[tensor_offset + 4 * nq_total +
                                                                  local_q_idx];

                              Gtt = geometric_transformation_cell[tensor_offset + 5 * nq_total +
                                                                  local_q_idx];

                              // Multiply by D
                              for (int n = 0; n < nq; n++)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] *
                                        scratch_values[e * nq * nq * nq + r * nq * nq + n * nq + p];
                                  qt += r_r[n] *
                                        scratch_values[e * nq * nq * nq + n * nq * nq + q * nq + p];
                                }


                              const int idx = e * nq * nq * nq + r * nq * nq + q * nq + p;

                              // Apply chain rule
                              scratch_grads_0[idx] = Grr * qr + Grs * qs + Grt * qt;

                              scratch_grads_1[idx] = Grs * qr + Gss * qs + Gst * qt;

                              scratch_grads_2[idx] = Grt * qr + Gst * qs + Gtt * qt;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-5: integrate gradient (apply D^T)
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp += s_co_shape_gradients[p * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp += scratch_grads_1[e * nq * nq + n * nq + p] * r_q[n];

                              scratch_values[e * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int r = (tid % co_dimension_size) / nq;
                          const int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                              r_r[n] = s_co_shape_gradients[r * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp += r_p[n] * s_co_shape_gradients[p * nq + n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_1[e * nq * nq * nq + r * nq * nq + n * nq + p] *
                                  r_q[n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_2[e * nq * nq * nq + n * nq * nq + q * nq + p] *
                                  r_r[n];

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-6: interpolate quad values back to cell dofs
                {
                  // direction 2
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
                              r_r[r] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p];
                            }

                          for (int k = 0; k < nm; ++k)
                            {
                              Number tmp = 0.0;

                              for (int r = 0; r < nq; ++r)
                                {
                                  tmp += s_shape_values[k * nq + r] * r_r[r];
                                }

                              scratch_grads_0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  //  direction 1
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
                                r_q[q] = scratch_values[e * nq * nq + q * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nq;
                            int p = tid % nq;

                            for (int j = 0; j < nq; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nq * nm + k * nq * nq + j * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 0
                  {
                    const int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % nm;

                            for (int p = 0; p < nq; ++p)
                              {
                                r_p[p] = scratch_grads_1[e * nq * nm + j * nq + p];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int p = 0; p < nq; ++p)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[p];
                                  }
                                scratch_values[e * nm * nm + j * nm + i] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nm;
                            int j = tid % nm;

                            for (int p = 0; p < nq; ++p)
                              {
                                r_p[p] =
                                  scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + p];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int p = 0; p < nq; ++p)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[p];
                                  }
                                scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // step-7 : Distribute cell contribution into the global vector
                {
                  constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int global_cell_index = eb * nelmtPerBatch + e;

                      if (dim == 2)
                        {
                          const int i = tid % nm;

                          for (int j = 0; j < nm; ++j)
                            {
                              const int local_idx = j * nm + i;

                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // here we have DG space so we don't need atomic add
                                  // Kokkos::atomic_store(&d_out[dof_index], scratch_values[tid]);
                                  vector_out[dof_index] = scratch_values[e * nm_total + local_idx];
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

                              const unsigned int dof_index =
                                dof_indices(local_idx, global_cell_index);

                              if (dof_index != numbers::invalid_unsigned_int)
                                {
                                  // here we have DG space so we don't need atomic add
                                  // Kokkos::atomic_store(&d_out[dof_index], scratch_values[tid]);
                                  vector_out[dof_index] = scratch_values[e * nm_total + local_idx];
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

    template <int dim, int nm, int nq, typename Number>
    void
    compute_inner_faces(
      const DeviceView<Number>                                            co_shape_gradients,
      const Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jacobians_times_normal,
      const Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jxw_values,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space>    penalty_parameters,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>        face_values_at_quads,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads,
      Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space> face_info,
      const unsigned int                                                  n_faces,
      const unsigned int n_faces_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total_per_face = Utilities::pow(nq, dim - 1);

      // finding the batch size
      // constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays =
        2 * (1 + dim); // values and gradients on both sides of the face

      const int nelmt = n_faces;

      int nelmtPerBatch = 1;

      if (n_faces_per_batch == numbers::invalid_unsigned_int)
        nelmtPerBatch = (shmemPerBlock / (n_scratch_arrays * nq_total_per_face * sizeof(Number)));
      else
        nelmtPerBatch = n_faces_per_batch;

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));


      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 2) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));

      // std::cout << "nelmt = " << nelmt << ", nelmtPerBatch = " << nelmtPerBatch
      //           << ", numBlocks = " << numBlocks << ", threadsPerBlock = " << threadsPerBlock
      //           << std::endl;

      {
        const int ssize = nq * nq + // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total_per_face; // working scratch arrays: scratch_values,
                                               // scratch_grads_0, scratch_grads_1

        // scratch data + size for face and cells id's storage
        const int shmem_size = ssize * sizeof(Number); //+ 4 * nelmtPerBatch * sizeof(int);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];
            Number r_r[nq];


            // Look up tangent directions based on normal_direction (0, 1, or 2)
            constexpr int lookup_tangents_3d[3][2] = {
              {1, 2}, // normal == 0
              {0, 2}, // normal == 1
              {0, 1}  // normal == 2
            };

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_co_shape_gradients = scratch;

            Number *scratch_values_minus = s_co_shape_gradients + nq * nq;
            Number *scratch_values_plus  = scratch_values_minus + nelmtPerBatch * nq_total_per_face;

            Number *scratch_gradients_minus =
              scratch_values_plus + nelmtPerBatch * nq_total_per_face;
            Number *scratch_gradients_plus =
              scratch_gradients_minus + nelmtPerBatch * dim * nq_total_per_face;

            // Number *end_number_type_scratch =
            //   scratch_gradients_plus + nelmtPerBatch * dim * nq_total_per_face;

            // int *cell_indices_minus = (int *)end_number_type_scratch;
            // int *cell_indices_plus  = cell_indices_minus + nelmtPerBatch;
            // int *face_indices_minus = cell_indices_plus + nelmtPerBatch;
            // int *face_indices_plus  = face_indices_minus + nelmtPerBatch;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = co_shape_gradients[tid];
              }
            team_member.team_barrier();

            // element batch iteration
            int eb = team_member.league_rank();
            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                              (nelmt - eb * nelmtPerBatch) :
                                              nelmtPerBatch;

                // step-0: copy cell and face indices on face's sides to local memory
                // for (int tid = threadIdx; tid < c_nelmtPerBatch; tid += blockSize)
                //   {
                //     const int e_global = eb * nelmtPerBatch + tid;

                //     cell_indices_minus[tid] = face_info(e_global, 0);
                //     cell_indices_plus[tid]  = face_info(e_global, 1);
                //     face_indices_minus[tid] = face_info(e_global, 2);
                //     face_indices_plus[tid]  = face_info(e_global, 3);
                //   }
                // step-1 : Copy quad values and normal derivatives on both face sides from global
                // storage
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;
                      const int q = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int cell_minus = face_info(e_global, 0);
                      const int cell_plus  = face_info(e_global, 1);
                      const int f_minus    = face_info(e_global, 2);
                      const int f_plus     = face_info(e_global, 3);

                      const int normal_direction = f_minus / 2;

                      for (int p = 0; p < nq; ++p)
                        {
                          const int q_idx = (dim == 2) ? p : q * nq + p;

                          scratch_values_minus[e * nq_total_per_face + q_idx] =
                            face_values_at_quads(q_idx, f_minus, cell_minus);

                          scratch_values_plus[e * nq_total_per_face + q_idx] =
                            face_values_at_quads(q_idx, f_plus, cell_plus);

                          Number jac_per_n[2];
                          jac_per_n[0] =
                            jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                     normal_direction * nq_total_per_face + q_idx,
                                                   0);
                          jac_per_n[1] =
                            jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                     normal_direction * nq_total_per_face + q_idx,
                                                   1);

                          scratch_gradients_minus[e * dim * nq_total_per_face +
                                                  normal_direction * nq_total_per_face + q_idx] =
                            face_normal_derivatives_at_quads(q_idx, f_minus, cell_minus) *
                            jac_per_n[0];

                          scratch_gradients_plus[e * dim * nq_total_per_face +
                                                 normal_direction * nq_total_per_face + q_idx] =
                            face_normal_derivatives_at_quads(q_idx, f_plus, cell_plus) *
                            jac_per_n[1];
                        }
                    }
                  team_member.team_barrier();
                }

                // step-2 : Evaluate tangential derivatives
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int f_minus = face_info(e_global, 2);

                      const int normal_direction = f_minus / 2;

                      if (dim == 2)
                        {
                          const int tangent_direction = 1 - normal_direction;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values_minus[e * nq + n];
                              r_q[n] = scratch_values_plus[e * nq + n];
                            }

                          Number qr, qs;
                          Number jac_per_n[2];
                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += s_co_shape_gradients[n * nq + p] * r_q[n];
                                }

                              jac_per_n[0] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction * nq_total_per_face + p,
                                                       0);
                              jac_per_n[1] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction * nq_total_per_face + p,
                                                       1);

                              scratch_gradients_minus[e * dim * nq_total_per_face +
                                                      normal_direction * nq_total_per_face + p] +=
                                qr * jac_per_n[0];

                              scratch_gradients_plus[e * dim * nq_total_per_face +
                                                     normal_direction * nq_total_per_face + p] +=
                                qs * jac_per_n[1];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int p = tid % co_dimension_size;

                          const int tangent_direction_0 = lookup_tangents_3d[normal_direction][0];
                          const int tangent_direction_1 = lookup_tangents_3d[normal_direction][1];

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values_minus[e * nq * nq + n * nq + p];
                              r_q[n] = scratch_values_plus[e * nq * nq + n * nq + p];
                              r_r[n] = s_co_shape_gradients[n * nq + p];
                            }

                          Number qr, qs, qt, qv;
                          Number jac_per_n[2][2];

                          for (int q = 0; q < nq; ++q)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;
                              qv = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + q] * r_p[n];
                                  qs += s_co_shape_gradients[n * nq + q] * r_q[n];

                                  qt += scratch_values_minus[e * nq * nq + q * nq + n] * r_r[n];
                                  qv += scratch_values_plus[e * nq * nq + q * nq + n] * r_r[n];
                                }

                              jac_per_n[0][0] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction_0 * nq_total_per_face +
                                                         q,
                                                       0);
                              jac_per_n[0][1] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction_0 * nq_total_per_face +
                                                         q,
                                                       1);
                              jac_per_n[1][0] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction_1 * nq_total_per_face +
                                                         q,
                                                       0);
                              jac_per_n[1][1] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         tangent_direction_1 * nq_total_per_face +
                                                         q,
                                                       1);

                              scratch_gradients_minus[e * dim * nq_total_per_face +
                                                      normal_direction * nq_total_per_face +
                                                      q * nq + p] +=
                                qr * jac_per_n[0][0] + qt * jac_per_n[1][0];

                              scratch_gradients_plus[e * dim * nq_total_per_face +
                                                     normal_direction * nq_total_per_face + q * nq +
                                                     p] +=
                                qs * jac_per_n[0][1] + qv * jac_per_n[1][1];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-3 : Submit values and normal derivatives
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int p = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int f_minus = face_info(e_global, 2);

                      const int normal_direction = f_minus / 2;

                      Number sigma = penalty_parameters(e_global);

                      // copy to register
                      for (int n = 0; n < nq; n++)
                        {
                          const int idx         = (dim == 2) ? n : n * nq + p;
                          const int idx_jump    = e * nq_total_per_face + idx;
                          const int idx_average = e * dim * nq_total_per_face +
                                                  normal_direction * nq_total_per_face + idx;

                          Number solution_jump =
                            scratch_values_minus[idx_jump] - scratch_values_plus[idx_jump];
                          Number average_normal_derivative =
                            0.5 * (scratch_gradients_minus[idx_average] +
                                   scratch_gradients_plus[idx_average]);

                          r_p[n] = solution_jump * sigma - average_normal_derivative;
                          r_q[n] = -0.5 * solution_jump;
                        }

                      Number jac_per_n[dim][2];
                      Number jxw[2];

                      for (int n = 0; n < nq; ++n)
                        {
                          const int q_idx = (dim == 2) ? n : n * nq + p;

                          // load jacobian times normal and jxw values
                          jxw[0] = jxw_values(e_global * nq_total_per_face + q_idx, 0);
                          jxw[1] = jxw_values(e_global * nq_total_per_face + q_idx, 1);

                          for (int d = 0; d < dim; ++d)
                            {
                              jac_per_n[d][0] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         d * nq_total_per_face + q_idx,
                                                       0);
                              jac_per_n[d][1] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                         d * nq_total_per_face + q_idx,
                                                       1);
                            }

                          // submit values
                          scratch_values_minus[e * nq_total_per_face + q_idx] = r_p[n] * jxw[0];
                          scratch_values_plus[e * nq_total_per_face + q_idx] = -1 * r_p[n] * jxw[1];

                          // submit normal derivatives
                          for (int d = 0; d < dim; ++d)
                            {
                              scratch_gradients_minus[e * dim * nq_total_per_face +
                                                      d * nq_total_per_face + q_idx] =
                                r_q[n] * jac_per_n[d][0] * jxw[0];
                              scratch_gradients_plus[e * dim * nq_total_per_face +
                                                     d * nq_total_per_face + q_idx] =
                                r_q[n] * jac_per_n[d][1] * jxw[1];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-4: integrate tangential derivatives +  write to global storage
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;
                      const int q = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int cell_minus = face_info(e_global, 0);
                      const int cell_plus  = face_info(e_global, 1);
                      const int f_minus    = face_info(e_global, 2);
                      const int f_plus     = face_info(e_global, 3);

                      const int normal_direction = f_minus / 2;

                      const int tangent_direction_0 = (dim == 2) ?
                                                        (1 - normal_direction) :
                                                        lookup_tangents_3d[normal_direction][0];

                      const int tangent_direction_1 = lookup_tangents_3d[normal_direction][1];

                      // copy to register
                      for (int n = 0; n < nq; ++n)
                        {
                          const int idx = (dim == 2) ? n : q * nq + n;

                          r_p[n] =
                            scratch_gradients_minus[e * nq_total_per_face * dim +
                                                    tangent_direction_0 * nq_total_per_face + idx];
                          r_q[n] =
                            scratch_gradients_plus[e * nq_total_per_face * dim +
                                                   tangent_direction_0 * nq_total_per_face + idx];

                          if (dim == 3)
                            r_r[n] = s_co_shape_gradients[q * nq + n];
                        }

                      for (int p = 0; p < nq; ++p)
                        {
                          const int idx = (dim == 2) ? p : q * nq + p;

                          // get quad values
                          Number tmp_minus = scratch_values_minus[e * nq_total_per_face + idx];
                          Number tmp_plus  = scratch_values_plus[e * nq_total_per_face + idx];

                          for (int n = 0; n < nq; ++n)
                            {
                              tmp_minus += s_co_shape_gradients[p * nq + n] * r_p[n];
                              tmp_plus += s_co_shape_gradients[p * nq + n] * r_q[n];
                            }

                          if (dim == 3)
                            for (int n = 0; n < nq; ++n)
                              {
                                tmp_minus +=
                                  scratch_gradients_minus[e * nq_total_per_face * dim +
                                                          tangent_direction_1 * nq_total_per_face +
                                                          n * nq + p] *
                                  r_r[n];

                                tmp_plus +=
                                  scratch_gradients_plus[e * nq_total_per_face * dim +
                                                         tangent_direction_1 * nq_total_per_face +
                                                         n * nq + p] *
                                  r_r[n];
                              }

                          // write into global storage
                          face_values_at_quads(idx, f_minus, cell_minus) = tmp_minus;
                          face_values_at_quads(idx, f_plus, cell_plus)   = tmp_plus;

                          face_normal_derivatives_at_quads(idx, f_minus, cell_minus) =
                            scratch_gradients_minus[e * nq_total_per_face * dim +
                                                    normal_direction * nq_total_per_face + idx];

                          face_normal_derivatives_at_quads(idx, f_plus, cell_plus) =
                            scratch_gradients_plus[e * nq_total_per_face * dim +
                                                   normal_direction * nq_total_per_face + idx];
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


    template <int dim, int nm, int nq, typename Number>
    void
    compute_boundary_faces(
      const DeviceView<Number>                                         co_shape_gradients,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> jacobians_times_normal,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> jxw_values,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> penalty_parameters,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>     face_values_at_quads,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads,
      Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space> face_info,
      const unsigned int                                                  n_faces,
      const unsigned int n_faces_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total_per_face = Utilities::pow(nq, dim - 1);

      // finding the batch size
      // constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = (1 + dim); // values and gradients the face

      const int nelmt = n_faces;

      int nelmtPerBatch = 1;

      if (n_faces_per_batch == numbers::invalid_unsigned_int)
        nelmtPerBatch = (shmemPerBlock / (n_scratch_arrays * nq_total_per_face * sizeof(Number)));
      else
        nelmtPerBatch = n_faces_per_batch;

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 2) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));

      // std::cout << "nelmt = " << nelmt << ", nelmtPerBatch = " << nelmtPerBatch
      //           << ", numBlocks = " << numBlocks << ", threadsPerBlock = " << threadsPerBlock
      //           << std::endl;

      {
        const int ssize = nq * nq + // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total_per_face; // working scratch arrays: scratch_values,
                                               // scratch_grads_0, scratch_grads_1

        // scratch data + size for face and cells id's storage
        const int shmem_size = ssize * sizeof(Number); //+ 2 * nelmtPerBatch * sizeof(int);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];

            // Look up tangent directions based on normal_direction (0, 1, or 2)
            constexpr int lookup_tangents_3d[3][2] = {
              {1, 2}, // normal == 0
              {0, 2}, // normal == 1
              {0, 1}  // normal == 2
            };

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_co_shape_gradients = scratch;

            Number *scratch_values = s_co_shape_gradients + nq * nq;

            Number *scratch_gradients = scratch_values + nelmtPerBatch * nq_total_per_face;

            // Number *end_number_type_scratch =
            //   scratch_gradients + nelmtPerBatch * dim * nq_total_per_face;

            // int *cell_indices = (int *)end_number_type_scratch;
            // int *face_indices = cell_indices + nelmtPerBatch;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = co_shape_gradients[tid];
              }
            team_member.team_barrier();

            // element batch iteration
            int eb = team_member.league_rank();
            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                              (nelmt - eb * nelmtPerBatch) :
                                              nelmtPerBatch;

                // step-0: copy cell and face indices on face's sides to local memory
                // for (int tid = threadIdx; tid < c_nelmtPerBatch; tid += blockSize)
                //   {
                //     const int e_global = eb * nelmtPerBatch + tid;

                //     cell_indices[tid] = face_info(e_global, 0);
                //     face_indices[tid] = face_info(e_global, 2);
                //   }

                // step-1 : Copy quad values and normal derivatives from global storage
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;
                      const int q = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int cell = face_info(e_global, 0);
                      const int face = face_info(e_global, 2);

                      const int normal_direction = face / 2;

                      for (int p = 0; p < nq; ++p)
                        {
                          const int q_idx = (dim == 2) ? p : (q * nq + p);

                          scratch_values[e * nq_total_per_face + q_idx] =
                            face_values_at_quads(q_idx, face, cell);

                          Number jac_per_n =
                            jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                   normal_direction * nq_total_per_face + q_idx);

                          scratch_gradients[e * dim * nq_total_per_face +
                                            normal_direction * nq_total_per_face + q_idx] =
                            face_normal_derivatives_at_quads(q_idx, face, cell) * jac_per_n;
                        }
                    }
                  team_member.team_barrier();
                }

                // step-2 : Evaluate tangential derivatives
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int face = face_info(e_global, 2);

                      const int normal_direction = face / 2;

                      if (dim == 2)
                        {
                          const int tangent_direction = 1 - normal_direction;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq + n];
                            }

                          Number qr;
                          Number jac_per_n;
                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                }

                              jac_per_n =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                       tangent_direction * nq_total_per_face + p);

                              scratch_gradients[e * dim * nq_total_per_face +
                                                normal_direction * nq_total_per_face + p] +=
                                qr * jac_per_n;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int p = tid % co_dimension_size;

                          const int tangent_direction_0 = lookup_tangents_3d[normal_direction][0];
                          const int tangent_direction_1 = lookup_tangents_3d[normal_direction][1];

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq + n * nq + p];
                              r_q[n] = s_co_shape_gradients[n * nq + p];
                            }

                          Number qr, qs;
                          Number jac_per_n[2];

                          for (int q = 0; q < nq; ++q)
                            {
                              qr = 0;
                              qs = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + q] * r_p[n];

                                  qs += scratch_values[e * nq * nq + q * nq + n] * r_q[n];
                                }

                              jac_per_n[0] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                       tangent_direction_0 * nq_total_per_face + q);
                              jac_per_n[1] =
                                jacobians_times_normal(e_global * dim * nq_total_per_face +
                                                       tangent_direction_1 * nq_total_per_face + q);

                              scratch_gradients[e * dim * nq_total_per_face +
                                                normal_direction * nq_total_per_face + q * nq +
                                                p] += qr * jac_per_n[0] + qs * jac_per_n[1];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-3 : Submit values and normal derivatives
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int p = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int face = face_info(e_global, 2);

                      const int normal_direction = face / 2;

                      const int boundary_id = face_info(e_global, 3);

                      Number sigma = penalty_parameters(e_global);

                      // copy to register
                      for (int n = 0; n < nq; n++)
                        {
                          const int idx = (dim == 2) ? n : n * nq + p;

                          const int idx_jump    = e * nq_total_per_face + idx;
                          const int idx_average = e * dim * nq_total_per_face +
                                                  normal_direction * nq_total_per_face + idx;

                          Number u_inner                 = 0;
                          Number u_outer                 = 0;
                          Number normal_derivative_inner = 0;
                          Number normal_derivative_outer = 0;

                          u_inner                 = scratch_values[idx_jump];
                          normal_derivative_inner = scratch_gradients[idx_average];

                          if (boundary_id == 0) // is dirichlet boundary
                            {
                              u_outer                 = -u_inner;
                              normal_derivative_outer = normal_derivative_inner;
                            }
                          else
                            {
                              u_outer                 = u_inner;
                              normal_derivative_outer = -normal_derivative_inner;
                            }

                          Number solution_jump = u_inner - u_outer;
                          Number average_normal_derivative =
                            0.5 * (normal_derivative_inner + normal_derivative_outer);

                          r_p[n] = solution_jump * sigma - average_normal_derivative;
                          r_q[n] = -0.5 * solution_jump;
                        }

                      Number jac_per_n[dim];
                      Number jxw;

                      for (int n = 0; n < nq; ++n)
                        {
                          const int q_idx = (dim == 2) ? n : n * nq + p;

                          // load jacobian times normal and jxw values
                          jxw = jxw_values(e_global * nq_total_per_face + q_idx);

                          for (int d = 0; d < dim; ++d)
                            {
                              jac_per_n[d] = jacobians_times_normal(
                                e_global * dim * nq_total_per_face + d * nq_total_per_face + q_idx);
                            }

                          // submit values
                          scratch_values[e * nq_total_per_face + q_idx] = r_p[n] * jxw;

                          // submit normal derivatives
                          for (int d = 0; d < dim; ++d)
                            {
                              scratch_gradients[e * dim * nq_total_per_face +
                                                d * nq_total_per_face + q_idx] =
                                r_q[n] * jac_per_n[d] * jxw;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // step-4: integrate tangential derivatives +  write to global storage
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;
                      const int q = tid % co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int cell = face_info(e_global, 0);
                      const int face = face_info(e_global, 2);

                      const int normal_direction = face / 2;

                      const int tangent_direction_0 = (dim == 2) ?
                                                        (1 - normal_direction) :
                                                        lookup_tangents_3d[normal_direction][0];

                      const int tangent_direction_1 = lookup_tangents_3d[normal_direction][1];

                      // copy to register
                      for (int n = 0; n < nq; ++n)
                        {
                          const int idx = (dim == 2) ? n : q * nq + n;

                          r_p[n] = scratch_gradients[e * nq_total_per_face * dim +
                                                     tangent_direction_0 * nq_total_per_face + idx];

                          if (dim == 3)
                            r_q[n] = s_co_shape_gradients[q * nq + n];
                        }

                      for (int p = 0; p < nq; ++p)
                        {
                          const int idx = (dim == 2) ? p : q * nq + p;

                          // get quad values
                          Number tmp = scratch_values[e * nq_total_per_face + idx];

                          for (int n = 0; n < nq; ++n)
                            {
                              tmp += s_co_shape_gradients[p * nq + n] * r_p[n];
                            }

                          if (dim == 3)
                            for (int n = 0; n < nq; ++n)
                              {
                                tmp += scratch_gradients[e * nq_total_per_face * dim +
                                                         tangent_direction_1 * nq_total_per_face +
                                                         n * nq + p] *
                                       r_q[n];
                              }


                          face_values_at_quads(idx, face, cell) = tmp;

                          face_normal_derivatives_at_quads(idx, face, cell) =
                            scratch_gradients[e * nq_total_per_face * dim +
                                              normal_direction * nq_total_per_face + idx];
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

    template <int dim, int nm, int nq, typename Number>
    void
    distribute_face_to_global(
      const DeviceView<Number> shape_values,
      DeviceView<Number>       vector_out,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
        interpolate_quad_to_boundary,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_values_at_quads,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
                           face_normal_derivatives_at_quads,
      const DoFIndicesView dof_indices,
      const unsigned int   n_cells,
      const unsigned int   n_cells_per_batch = numbers::invalid_unsigned_int,
      const unsigned int   n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int   threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total = Utilities::pow(nq, dim);

      constexpr int nm_total = Utilities::pow(nm, dim);

      // finding the batch size
      // constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = dim; // values and 1 temp in 2d / 2 temps in 3d

      const int nelmt = n_cells;

      int nelmtPerBatch = 1;

      if (n_cells_per_batch == numbers::invalid_unsigned_int)
        nelmtPerBatch = (shmemPerBlock / (n_scratch_arrays * nq_total * sizeof(Number)));
      else
        nelmtPerBatch = n_cells_per_batch;

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;
      else if (nelmtPerBatch > nelmt)
        nelmtPerBatch = nelmt;

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
          nm * nq + // shape values
          4 * nq +  // interpolate quad to boundary
          n_scratch_arrays * nq_total *
            nelmtPerBatch; // working scratch arrays: scratch_values and temporary vectors


        const int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_quad_to_boundary_value_0 = s_shape_values + nq * nm;
            Number *s_quad_to_boundary_value_1 = s_quad_to_boundary_value_0 + nq;
            Number *s_quad_to_boundary_grad_0  = s_quad_to_boundary_value_1 + nq;
            Number *s_quad_to_boundary_grad_1  = s_quad_to_boundary_grad_0 + nq;

            Number *scratch_values = s_quad_to_boundary_grad_1 + nq;
            Number *scratch_temp1  = scratch_values + nelmtPerBatch * nq_total;

            Number *scratch_temp2 = nullptr;
            if (dim == 3)
              scratch_temp2 = scratch_temp1 + nelmtPerBatch * nq_total;


            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq; tid += blockSize)
              {
                s_quad_to_boundary_value_0[tid] = interpolate_quad_to_boundary(0, tid, 0);
                s_quad_to_boundary_value_1[tid] = interpolate_quad_to_boundary(0, tid, 1);

                s_quad_to_boundary_grad_0[tid] = interpolate_quad_to_boundary(1, tid, 0);
                s_quad_to_boundary_grad_1[tid] = interpolate_quad_to_boundary(1, tid, 1);
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
                int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                        (nelmt - eb * nelmtPerBatch) :
                                        nelmtPerBatch;

                // step-1: read face values and normal derivatives from global storage and
                // interpolate them to cell quad values
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nq_total; tid += blockSize)
                    {
                      const int e          = tid / nq_total;
                      const int q_idx_cell = tid % nq_total;

                      const int r = (dim == 3) ? (q_idx_cell / (nq * nq)) : 0;
                      const int q =
                        (dim == 3) ? ((q_idx_cell % (nq * nq) / nq)) : (q_idx_cell / nq);
                      const int p = q_idx_cell % nq;

                      const int cell_id = eb * nelmtPerBatch + e;

                      Number cell_quad_value = 0.0;

                      // process faces per direction
                      for (int f = 0; f < dim; ++f)
                        {
                          int q_idx_face = 0;
                          int n_idx      = 0;

                          if (f == 0)
                            {
                              q_idx_face = (dim == 3) ? (r * nq + q) : q;
                              n_idx      = p;
                            }
                          else if (f == 1)
                            {
                              q_idx_face = (dim == 3) ? (r * nq + p) : p;
                              n_idx      = q;
                            }
                          else if (f == 2)
                            {
                              q_idx_face = q * nq + p;
                              n_idx      = r;
                            }

                          Number value_0 = face_values_at_quads(q_idx_face, 2 * f, cell_id);
                          Number value_1 = face_values_at_quads(q_idx_face, 2 * f + 1, cell_id);

                          Number normal_derivative_0 =
                            face_normal_derivatives_at_quads(q_idx_face, 2 * f, cell_id);
                          Number normal_derivative_1 =
                            face_normal_derivatives_at_quads(q_idx_face, 2 * f + 1, cell_id);

                          cell_quad_value +=
                            s_quad_to_boundary_value_0[n_idx] * value_0 +
                            s_quad_to_boundary_value_1[n_idx] * value_1 +
                            s_quad_to_boundary_grad_0[n_idx] * normal_derivative_0 +
                            s_quad_to_boundary_grad_1[n_idx] * normal_derivative_1;
                        }

                      scratch_values[tid] = cell_quad_value;
                    }
                  team_member.team_barrier();
                }

                // step-2: interpolate cell quad values to cell dof values
                {
                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int n = 0; n < nq; ++n)
                            {
                              r_p[n] = scratch_values[e * nq * nq * nq + n * nq * nq + q * nq + p];
                            }

                          for (int k = 0; k < nm; ++k)
                            {
                              Number tmp = 0.0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  tmp += s_shape_values[k * nq + n] * r_p[n];
                                }

                              scratch_temp2[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  //  direction 1
                  {
                    constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int n = 0; n < nq; ++n)
                              {
                                r_p[n] = scratch_values[e * nq * nq + n * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int n = 0; n < nq; ++n)
                                  {
                                    tmp += s_shape_values[j * nq + n] * r_p[n];
                                  }
                                scratch_temp1[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nq;
                            int p = tid % nq;

                            for (int n = 0; n < nq; ++n)
                              {
                                r_p[n] = scratch_temp2[e * nq * nq * nm + k * nq * nq + n * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int n = 0; n < nq; ++n)
                                  {
                                    tmp += s_shape_values[j * nq + n] * r_p[n];
                                  }
                                scratch_temp1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 0
                  {
                    const int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int n = 0; n < nq; ++n)
                              {
                                r_p[n] = scratch_temp1[e * nq * nm + j * nq + n];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int n = 0; n < nq; ++n)
                                  {
                                    tmp += s_shape_values[i * nq + n] * r_p[n];
                                  }
                                scratch_values[e * nm * nm + j * nm + i] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nm;
                            int j = tid % nm;

                            for (int n = 0; n < nq; ++n)
                              {
                                r_p[n] = scratch_temp1[e * nq * nm * nm + k * nq * nm + j * nq + n];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int n = 0; n < nq; ++n)
                                  {
                                    tmp += s_shape_values[i * nq + n] * r_p[n];
                                  }
                                scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + i] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // step-3 : distribute results into global vector
                for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                  {
                    const int e = tid / nm_total;

                    const int local_idx = tid % nm_total;

                    const int global_cell_index = eb * nelmtPerBatch + e;

                    // Find where this node lives in the global 'd_out'
                    // vector
                    const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        // here we have DG space so we don't need atomic add
                        // Kokkos::atomic_add(&d_out[dof_index], scratch_values[tid]);
                        // Kokkos::atomic_add(&d_out[dof_index],scratch_values[tid]);

                        vector_out[dof_index] += scratch_values[tid];
                      }
                  }

                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }
  } // namespace DG
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif