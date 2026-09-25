#ifndef bk4_cuda_plain_kernels_cuh
#define bk4_cuda_plain_kernels_cuh

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

#ifdef __CUDACC__

#  include <cuda_runtime.h>

#  include <algorithm>
#  include <string>

DEAL_II_NAMESPACE_OPEN

namespace BK4
{
  namespace Parallel
  {
    // Plain (non-tensor-core) fp64 GEMM-in-registers BK4 kernel, ported
    // from dealiiX_benchmarks/CEED_BK/include/kernels/BK4/templated_cuda_kernels.cuh
    // (BK4::Parallel::LaplaceOperator). The tensor-product-contraction math
    // is unchanged from that benchmark; only I/O is different here: the
    // benchmark reads/writes a dense per-element scratch array, this reads
    // the global dof vector via dof_indices_per_component and scatters back
    // with atomicAdd, the same way f64_m8n8k4_mma (bk4_cuda_kernels.cuh)
    // does for the tensor-core path.
    namespace Cuda
    {
      template <typename T,
                const unsigned int nq,
                const unsigned int nm,
                const unsigned int nelmtPerBatch,
                const unsigned int n_components>
      __global__ void
      gemm_laplace_operator(
        const unsigned int nelmt,
        const T *__restrict__ d_basis,
        const T *__restrict__ d_dbasis,
        const T *__restrict__ d_G,
        const T *__restrict__ d_in,
        T *__restrict__ d_out,
        const Kokkos::Array<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>,
                            n_components> dof_indices_per_component)
      {
        constexpr unsigned int ndof_1D = nm * nm * nm;

        T r_p[nq];
        T r_q[nq];
        T r_r[nq];

        extern __shared__ T shared[];
        T                  *s_basis  = shared;
        T                  *s_dbasis = s_basis + nq * nm;

        T *s_wsp0 = s_dbasis + nq * nq;
        T *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;

        T *s_rqr = s_wsp1 + nelmtPerBatch * nq * nq * nq;
        T *s_rqs = s_rqr + nelmtPerBatch * nq * nq * nq;
        T *s_rqt = s_wsp0;

        // copy basis to shared memory
        for (unsigned int tid = threadIdx.x; tid < nm * nq; tid += blockDim.x)
          s_basis[tid] = d_basis[tid];
        for (unsigned int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x)
          s_dbasis[tid] = d_dbasis[tid];
        __syncthreads();

        unsigned int eb = blockIdx.x;
        while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
          {
            const unsigned int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                                   (nelmt - eb * nelmtPerBatch) :
                                                   nelmtPerBatch;

            for (unsigned int c = 0; c < n_components; ++c)
              {
                const auto &dof_indices = dof_indices_per_component[c];

                // step-1: gather dof values from the global vector
                for (unsigned int tid = threadIdx.x; tid < nelmtPerBatch * ndof_1D;
                     tid += blockDim.x)
                  {
                    const unsigned int e_local = tid / ndof_1D;
                    const unsigned int n_local = tid % ndof_1D;

                    if (e_local < c_nelmtPerBatch)
                      {
                        const unsigned int global_cell_index = eb * nelmtPerBatch + e_local;
                        const unsigned int dof_index = dof_indices(n_local, global_cell_index);

                        s_wsp0[tid] =
                          (dof_index == numbers::invalid_unsigned_int) ? 0.0 : d_in[dof_index];
                      }
                    else
                      {
                        s_wsp0[tid] = 0.0;
                      }
                  }
                __syncthreads();

                // step-2: direction 0
                {
                  constexpr int co_dimension_size = nm * nm;
                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int k = (tid % co_dimension_size) / nm;
                      const unsigned int j = tid % nm;

                      for (unsigned int i = 0; i < nm; ++i)
                        r_p[i] = s_wsp0[e * nm * nm * nm + k * nm * nm + j * nm + i];

                      for (unsigned int p = 0; p < nq; ++p)
                        {
                          T tmp = 0.0;
                          for (unsigned int i = 0; i < nm; ++i)
                            tmp += s_basis[i * nq + p] * r_p[i];

                          s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                        }
                    }
                  __syncthreads();
                }

                // step-3: direction 1
                {
                  constexpr int co_dimension_size = nq * nm;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int k = (tid % co_dimension_size) / nq;
                      const unsigned int p = tid % nq;

                      for (unsigned int j = 0; j < nm; ++j)
                        r_q[j] = s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p];

                      for (unsigned int q = 0; q < nq; ++q)
                        {
                          T tmp = 0.0;
                          for (unsigned int j = 0; j < nm; ++j)
                            tmp += s_basis[j * nq + q] * r_q[j];

                          s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                        }
                    }
                  __syncthreads();
                }

                // step-4: direction 2
                {
                  constexpr int co_dimension_size = nq * nq;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int q = (tid % co_dimension_size) / nq;
                      const unsigned int p = tid % nq;

                      for (unsigned int k = 0; k < nm; ++k)
                        r_r[k] = s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p];

                      for (unsigned int r = 0; r < nq; ++r)
                        {
                          T tmp = 0.0;
                          for (unsigned int k = 0; k < nm; ++k)
                            tmp += s_basis[k * nq + r] * r_r[k];

                          s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                        }
                    }
                  __syncthreads();
                }

                // gradient on quad points + apply G + chain rule
                {
                  constexpr int co_dimension_size = nq * nq;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int q = (tid % co_dimension_size) / nq;
                      const unsigned int p = tid % nq;

                      for (unsigned int n = 0; n < nq; n++)
                        {
                          r_p[n] = s_dbasis[n * nq + p];
                          r_q[n] = s_dbasis[n * nq + q];
                          r_r[n] = s_wsp1[e * nq * nq * nq + n * nq * nq + q * nq + p];
                        }

                      T Grr, Grs, Grt, Gss, Gst, Gtt;
                      T qr, qs, qt;

                      for (unsigned int r = 0; r < nq; ++r)
                        {
                          qr = 0;
                          qs = 0;
                          qt = 0;

                          const size_t g_base = eb * nelmtPerBatch * 6 * nq * nq * nq +
                                                e * 6 * nq * nq * nq + r * nq * nq + q * nq + p;

                          Grr = d_G[g_base + 0 * nq * nq * nq];
                          Grs = d_G[g_base + 1 * nq * nq * nq];
                          Grt = d_G[g_base + 2 * nq * nq * nq];
                          Gss = d_G[g_base + 3 * nq * nq * nq];
                          Gst = d_G[g_base + 4 * nq * nq * nq];
                          Gtt = d_G[g_base + 5 * nq * nq * nq];

                          for (unsigned int n = 0; n < nq; n++)
                            {
                              qr += r_p[n] * s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              qs += r_q[n] * s_wsp1[e * nq * nq * nq + r * nq * nq + n * nq + p];
                              qt += s_dbasis[n * nq + r] * r_r[n];
                            }

                          const unsigned int out_idx = e * nq * nq * nq + r * nq * nq + q * nq + p;

                          s_rqr[out_idx] = Grr * qr + Grs * qs + Grt * qt;
                          s_rqs[out_idx] = Grs * qr + Gss * qs + Gst * qt;
                          s_rqt[out_idx] = Grt * qr + Gst * qs + Gtt * qt;
                        }
                    }
                  __syncthreads();
                }

                // divergence
                {
                  constexpr int co_dimension_size = nq * nq;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / (co_dimension_size);
                      const unsigned int q = (tid / nq) % nq;
                      const unsigned int p = tid % nq;

                      for (unsigned int n = 0; n < nq; n++)
                        {
                          r_p[n] = s_dbasis[p * nq + n];
                          r_q[n] = s_dbasis[q * nq + n];
                          r_r[n] = s_rqt[e * nq * nq * nq + n * nq * nq + q * nq + p];
                        }

                      for (unsigned int r = 0; r < nq; ++r)
                        {
                          T tmp0 = 0;
                          for (int n = 0; n < nq; ++n)
                            tmp0 += s_rqr[e * nq * nq * nq + r * nq * nq + q * nq + n] * r_p[n];

                          for (int n = 0; n < nq; ++n)
                            tmp0 += s_rqs[e * nq * nq * nq + r * nq * nq + n * nq + p] * r_q[n];

                          for (int n = 0; n < nq; ++n)
                            tmp0 += r_r[n] * s_dbasis[r * nq + n];

                          s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp0;
                        }
                    }
                  __syncthreads();
                }

                // interpolate back to GLL nodes -- direction 2
                {
                  constexpr int co_dimension_size = nq * nq;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int q = (tid % co_dimension_size) / nq;
                      const unsigned int p = tid % nq;


                      for (unsigned int r = 0; r < nq; ++r)
                        r_r[r] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p];

                      for (unsigned int k = 0; k < nm; ++k)
                        {
                          T tmp = 0.0;
                          for (unsigned int r = 0; r < nq; ++r)
                            tmp += s_basis[k * nq + r] * r_r[r];

                          s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                        }
                    }
                  __syncthreads();
                }

                // direction 1
                {
                  constexpr int co_dimension_size = nq * nm;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int k = (tid % co_dimension_size) / nq;
                      const unsigned int p = tid % nq;


                      for (unsigned int q = 0; q < nq; ++q)
                        r_q[q] = s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + p];

                      for (unsigned int j = 0; j < nm; ++j)
                        {
                          T tmp = 0.0;
                          for (unsigned int q = 0; q < nq; ++q)
                            tmp += s_basis[j * nq + q] * r_q[q];

                          s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                        }
                    }
                  __syncthreads();
                }

                // direction 0 + scatter-add to the global vector
                {
                  constexpr int co_dimension_size = nm * nm;

                  for (unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockDim.x)
                    {
                      const unsigned int e = tid / co_dimension_size;
                      const unsigned int k = (tid % co_dimension_size) / nm;
                      const unsigned int j = tid % nm;

                      for (unsigned int p = 0; p < nq; ++p)
                        r_p[p] = s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p];

                      const unsigned int global_cell_index = eb * nelmtPerBatch + e;

                      for (unsigned int i = 0; i < nm; ++i)
                        {
                          T tmp = 0.0;
                          for (unsigned int p = 0; p < nq; ++p)
                            tmp += s_basis[i * nq + p] * r_p[p];

                          const unsigned int n_local   = k * nm * nm + j * nm + i;
                          const unsigned int dof_index = dof_indices(n_local, global_cell_index);

                          if (dof_index != numbers::invalid_unsigned_int)
                            atomicAdd(&d_out[dof_index], tmp);
                        }
                    }
                  __syncthreads();
                }

              } // component loop

            eb += gridDim.x;
          }
      }



      // Host-side launcher: computes the dynamic shared-memory footprint
      // and grid/block dimensions, matching launch_f64_m8n8k4_mma's shared
      // memory formula (the two kernels use the same scratch layout).
      template <const unsigned int nq,
                const unsigned int nm,
                const unsigned int nelmtPerBatch,
                const unsigned int n_components>
      void
      launch_gemm_laplace_operator(
        const unsigned int                 nelmt,
        const double                      *d_basis,
        const double                      *d_dbasis,
        const double                      *d_G,
        const double                      *d_in,
        double                            *d_out,
        const Kokkos::Array<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>,
                            n_components> &dof_indices_per_component)
      {
        static_assert(nq >= nm,
                      "gemm_laplace_operator's shared-memory buffers are sized from nq (see "
                      "ssize below) and are too small for nm > nq (under-integration).");

        if (nelmt == 0)
          return;

        const unsigned int padded_nelmt =
          ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch) * nelmtPerBatch;

        const unsigned int numBlocks       = std::max(1U, padded_nelmt / nelmtPerBatch);
        const unsigned int threadsPerBlock = nq * nq * std::max(1U, nelmtPerBatch);

        const unsigned int ssize      = nq * nm + nq * nq + 4u * nelmtPerBatch * nq * nq * nq;
        const unsigned int shmem_size = ssize * sizeof(double);

        auto *kernel = &gemm_laplace_operator<double, nq, nm, nelmtPerBatch, n_components>;

        constexpr unsigned int shmem_ceiling = 225'000;
        const cudaError_t      attr_err =
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_ceiling);
        AssertThrow(attr_err == cudaSuccess,
                    ExcMessage("cudaFuncSetAttribute() failed for gemm_laplace_operator "
                               "(shmem_ceiling = " +
                               std::to_string(shmem_ceiling) +
                               " bytes, requested shmem_size = " + std::to_string(shmem_size) +
                               " bytes): " + cudaGetErrorString(attr_err)));

        kernel<<<numBlocks, threadsPerBlock, shmem_size>>>(
          nelmt, d_basis, d_dbasis, d_G, d_in, d_out, dof_indices_per_component);

        const cudaError_t launch_err = cudaGetLastError();
        AssertThrow(launch_err == cudaSuccess,
                    ExcMessage("gemm_laplace_operator launch failed: " +
                               std::string(cudaGetErrorString(launch_err))));
      }

    } // namespace Cuda
  } // namespace Parallel
} // namespace BK4

DEAL_II_NAMESPACE_CLOSE

#endif // __CUDACC__
#endif // bk4_cuda_plain_kernels_cuh
