// The Kokkos linear elasticity kernel of the dealii-X benchmarks
// (elasticity_kernels/include/kokkos_kernels/linear_elasticity.hpp, commit
// 3c09cef, LinearElasticity::Parallel::Kokkos_LinearElasticityOperator), as a
// baseline for the operators of this repository. The device buffers, the
// timing loop and the norm of the result moved to the caller. Changes to the
// kernel body: it is templated on both nm and nq (the benchmark fixes
// nm = nq - 1), and the load of the input and the store of the result go
// through an Access object, read(c, element, i, j, k) and
// write(c, element, i, j, k, value), for element-local arrays
// (ElementAccess, as in the benchmark) or a global vector with gather and
// scatter (ElasticityOperator::vmult_benchmark()). The elements of the last
// batch beyond the number of elements are passed to Access as well. On a
// host backend (DefaultExecutionSpace == DefaultHostExecutionSpace, as in the
// BK3 kernels) a team has one thread: the thread-indexed blocks are loops over
// tid = team_rank, team_rank + team_size, ..., which run once per thread on
// the device, and the fluxes of components 1 and 2, kept in registers between
// stages 2 and 3 on the device, go through shared memory.
//
// nm nodes (degree nm - 1) and nq quadrature points per direction, nm <= nq,
// elements padded to whole batches:
//   G        [element][entry][a][b][c], entry 0 = quadrature weight * det J,
//            entries 1 + 3 * d + e = d xi_d / d x_e
//   basis    [i][p] = phi_i(x_p), dbasis [n][p] = phi'_n(x_p) of the
//            collocation basis on the quadrature points
// Direction d = 0 of xi is the index i of the nodes and a of the points.

#ifndef kernels_elasticity_benchmark_kernel_h
#define kernels_elasticity_benchmark_kernel_h

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Benchmark
{
  // the element-local arrays of the benchmark, [component][element][i][j][k]
  // with component stride n_elements_padded * nm^3
  template <typename T, unsigned int nm>
  struct ElementAccess
  {
    Kokkos::View<const T *> in;
    Kokkos::View<T *>       out;
    std::size_t             n_elements_padded;

    KOKKOS_INLINE_FUNCTION std::size_t
    index(const int c, const std::size_t element, const int i, const int j, const int k) const
    {
      return ((c * n_elements_padded + element) * nm + i) * nm * nm + j * nm + k;
    }

    KOKKOS_INLINE_FUNCTION T
    read(const int c, const std::size_t element, const int i, const int j, const int k) const
    {
      return in[index(c, element, i, j, k)];
    }

    KOKKOS_INLINE_FUNCTION void
    write(const int         c,
          const std::size_t element,
          const int         i,
          const int         j,
          const int         k,
          const T           value) const
    {
      out[index(c, element, i, j, k)] = value;
    }
  };



  template <typename T, const unsigned int nm, const unsigned int nq>
  class LinearElasticityKernel
  {
  public:
    // a team has nelmtPerBatch * nq * nq threads, the sweeps on the nodes use
    // nelmtPerBatch * nm * nm and nelmtPerBatch * nm * nq of them
    static_assert(nm >= 1 && nm <= nq, "the kernel needs 1 <= nm <= nq");

    using View = Kokkos::View<T *>;

    static constexpr unsigned int ncomp = 3;

    // teams of one thread on a host backend
    static constexpr bool one_thread_per_team =
      std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

    // the batch size of the benchmark driver: a 10 kB budget for the
    // 7 * nq^3 shared values per element, between 1 and 32 elements
    static unsigned int
    default_elements_per_batch()
    {
      const unsigned int raw = 10000u / (7u * nq * nq * nq * sizeof(T));
      return std::clamp(raw, 1u, 32u);
    }

    LinearElasticityKernel(const std::size_t     nelmt,
                           const unsigned int    nelmtPerBatch,
                           const std::vector<T> &basis,
                           const std::vector<T> &dbasis,
                           const T               lambda,
                           const T               mu)
      : nelmt(nelmt)
      , nelmtPerBatch(nelmtPerBatch)
      , nbatches((nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
      , d_dbasis(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_dbasis"), nq * nq)
      , lambda(lambda)
      , mu(mu)
    {
      if (basis.size() != nm * nq || dbasis.size() != nq * nq)
        throw std::invalid_argument("LinearElasticityKernel: wrong size of the 1D shape data");

      for (unsigned int i = 0u; i < nm * nq; ++i)
        c_basis[i] = basis[i];
      for (unsigned int i = 0u; i < nq * nq; ++i)
        c_dbasis[i] = dbasis[i];

      const auto host = Kokkos::create_mirror_view(d_dbasis);
      for (unsigned int i = 0u; i < nq * nq; ++i)
        host(i) = dbasis[i];
      Kokkos::deep_copy(d_dbasis, host);
    }

    std::size_t
    n_elements_padded() const
    {
      return nbatches * nelmtPerBatch;
    }

    std::size_t
    n_batches() const
    {
      return nbatches;
    }

    // number of threads per team the kernel requires
    unsigned int
    team_size() const
    {
      return one_thread_per_team ? 1 : nelmtPerBatch * nq * nq;
    }

    // out = A in on the element-local arrays of ElementAccess
    void
    apply(const View &d_G, const View &d_in, const View &d_out, const unsigned int numBlocks) const
    {
      apply(d_G, ElementAccess<T, nm>{d_in, d_out, n_elements_padded()}, numBlocks);
    }

    // numBlocks teams, each one going through every numBlocks-th batch. Not
    // fenced.
    template <typename Access>
    void
    apply(const View &d_G, const Access &access, const unsigned int numBlocks) const
    {
      const std::size_t  nbatches      = this->nbatches;
      const unsigned int nelmtPerBatch = this->nelmtPerBatch;
      const auto         c_basis       = this->c_basis;
      const auto         c_dbasis      = this->c_dbasis;
      const auto         d_dbasis      = this->d_dbasis;
      const T            lambda        = this->lambda;
      const T            mu            = this->mu;

      const unsigned int threadsPerBlock = team_size();

      // Kokkos with shared memory
      unsigned int ssize = nq * nq + 7 * nelmtPerBatch * nq * nq * nq;
      if constexpr (one_thread_per_team)
        ssize += 6 * nelmtPerBatch * nq * nq * nq;

      const unsigned int shmem_size = ssize * sizeof(T);

      typedef Kokkos::TeamPolicy<>::member_type member_type;
      Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      Kokkos::parallel_for(
        "Benchmark::LinearElasticityKernel", policy, KOKKOS_LAMBDA(member_type team_member) {
          T r_p[nq];
          T r_q[nq];
          T r_r[nq];

          T r_p0[nq];
          T r_p1[nq];
          T r_p2[nq];

          // Quadrature fluxes of components 1 and 2, computed once and carried in registers
          T r_f1r[nq], r_f1s[nq], r_f1t[nq];
          T r_f2r[nq], r_f2s[nq], r_f2t[nq];

          // shared memory access
          T *scratch = (T *)team_member.team_shmem().get_shmem(shmem_size);

          T *s_dbasis = scratch;

          T *s_wsp0 = s_dbasis + nq * nq;
          T *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;

          T *s_rqr = s_wsp1 + nelmtPerBatch * nq * nq * nq;
          T *s_rqs = s_rqr + nelmtPerBatch * nq * nq * nq;
          T *s_rqt = s_wsp0;

          T *s_u0   = s_rqs + nelmtPerBatch * nq * nq * nq;
          T *s_u1   = s_u0 + nelmtPerBatch * nq * nq * nq;
          T *s_u2   = s_u1 + nelmtPerBatch * nq * nq * nq;
          T *s_u[3] = {s_u0, s_u1, s_u2};

          // one_thread_per_team: the fluxes of components 1 and 2 of all
          // pencils, [pencil][6][p]
          T *s_f = s_u2 + nelmtPerBatch * nq * nq * nq;

          const unsigned int threadIdx = team_member.team_rank();
          const unsigned int blockSize = team_member.team_size();

          // Copy thread-indexed dbasis to shared memory (basis and uniform dbasis lookups use cmem)
          for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
            {
              s_dbasis[tid] = d_dbasis[tid];
            }
          team_member.team_barrier();


          // element batch iteration
          size_t eb = team_member.league_rank();
          while (eb < nbatches)
            {
            /*
            Stage 1: Interpolate to GL nodes (all 3 components required before quadrature coupling)
            */
#pragma unroll
              for (int c = 0; c < 3; ++c)
                {
                  // step-1 + step-2 : direction 0
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nm;
                       tid += blockSize)
                    {
                      int e = tid / (nm * nm);
                      int j = tid % (nm * nm) / nm;
                      int k = tid % nm;

#pragma unroll
                      for (int i = 0; i < nm; ++i)
                        {
                          r_p[i] = access.read(c, eb * nelmtPerBatch + e, i, j, k);
                        }

#pragma unroll
                      for (int p = 0; p < nq; ++p)
                        {
                          T tmp = 0.0;

#pragma unroll
                          for (int i = 0; i < nm; ++i)
                            {
                              tmp += c_basis[i * nq + p] * r_p[i];
                            }

                          s_wsp1[e * nq * nm * nm + p * nm * nm + j * nm + k] = tmp;
                        }
                    }
                  team_member.team_barrier();

                  // step-3 : direction 1
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nq;
                       tid += blockSize)
                    {
                      int e = tid / (nq * nm);
                      int p = tid % (nq * nm) / nm;
                      int k = tid % nm;
#pragma unroll
                      for (int j = 0; j < nm; ++j)
                        {
                          r_q[j] = s_wsp1[e * nq * nm * nm + p * nm * nm + j * nm + k];
                        }

#pragma unroll
                      for (int q = 0; q < nq; ++q)
                        {
                          T tmp = 0.0;

#pragma unroll
                          for (int j = 0; j < nm; ++j)
                            {
                              tmp += c_basis[j * nq + q] * r_q[j];
                            }

                          s_wsp0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
                        }
                    }
                  team_member.team_barrier();

                  // step-4 : direction 2 (all blockSize == nelmtPerBatch * nq * nq threads active)
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nq * nq;
                       tid += blockSize)
                    {
                      int e = tid / (nq * nq);
                      int q = tid % (nq * nq) / nq;
                      int p = tid % nq;

#pragma unroll
                      for (int k = 0; k < nm; ++k)
                        {
                          r_r[k] = s_wsp0[e * nq * nq * nm + q * nq * nm + p * nm + k];
                        }
#pragma unroll
                      for (int r = 0; r < nq; ++r)
                        {
                          T tmp = 0.0;

#pragma unroll
                          for (int k = 0; k < nm; ++k)
                            {
                              tmp += c_basis[k * nq + r] * r_r[k];
                            }

                          s_u[c][e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                        }
                    }
                  if (c == 2)
                    team_member.team_barrier();
                } // interpolate component loop


              /*
              Stage 2: Quadrature derivatives, constitutive law, and reference fluxes (evaluated
              once for all 3 components)
              */
              for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nq * nq; tid += blockSize)
                {
                  const int e = tid / (nq * nq);
                  const int q = tid % (nq * nq) / nq;
                  const int r = tid % nq;

#pragma unroll
                  for (int n = 0; n < nq; n++)
                    {
                      r_p0[n] = s_u0[e * nq * nq * nq + r * nq * nq + q * nq + n];
                      r_p1[n] = s_u1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                      r_p2[n] = s_u2[e * nq * nq * nq + r * nq * nq + q * nq + n];
                      r_q[n]  = s_dbasis[n * nq + q];
                      r_r[n]  = s_dbasis[n * nq + r];
                    }

                  const size_t g_elem_base =
                    eb * nelmtPerBatch * 10 * nq * nq * nq + e * 10 * nq * nq * nq + q * nq + r;
                  const T two_mu = (T)2 * mu;
                  T       wdetJ, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz;
                  T       ur0, us0, ut0;
                  T       ur1, us1, ut1;
                  T       ur2, us2, ut2;

#pragma unroll
                  for (int p = 0; p < nq; ++p)
                    {
                      ur0 = 0;
                      us0 = 0;
                      ut0 = 0;
                      ur1 = 0;
                      us1 = 0;
                      ut1 = 0;
                      ur2 = 0;
                      us2 = 0;
                      ut2 = 0;

                      // Load Geometric Factors, coalesced access
                      const size_t g_base = g_elem_base + p * nq * nq;
                      wdetJ               = d_G[g_base + 0 * nq * nq * nq];
                      drdx                = d_G[g_base + 1 * nq * nq * nq];
                      drdy                = d_G[g_base + 2 * nq * nq * nq];
                      drdz                = d_G[g_base + 3 * nq * nq * nq];
                      dsdx                = d_G[g_base + 4 * nq * nq * nq];
                      dsdy                = d_G[g_base + 5 * nq * nq * nq];
                      dsdz                = d_G[g_base + 6 * nq * nq * nq];
                      dtdx                = d_G[g_base + 7 * nq * nq * nq];
                      dtdy                = d_G[g_base + 8 * nq * nq * nq];
                      dtdz                = d_G[g_base + 9 * nq * nq * nq];

// Multiply by D
#pragma unroll
                      for (int n = 0; n < nq; n++)
                        {
                          T dp = c_dbasis[n * nq + p];

                          ur0 += dp * r_p0[n];
                          us0 += r_q[n] * s_u0[e * nq * nq * nq + r * nq * nq + n * nq + p];
                          ut0 += r_r[n] * s_u0[e * nq * nq * nq + n * nq * nq + q * nq + p];

                          ur1 += dp * r_p1[n];
                          us1 += r_q[n] * s_u1[e * nq * nq * nq + r * nq * nq + n * nq + p];
                          ut1 += r_r[n] * s_u1[e * nq * nq * nq + n * nq * nq + q * nq + p];

                          ur2 += dp * r_p2[n];
                          us2 += r_q[n] * s_u2[e * nq * nq * nq + r * nq * nq + n * nq + p];
                          ut2 += r_r[n] * s_u2[e * nq * nq * nq + n * nq * nq + q * nq + p];
                        }

                      // Apply chain rule and constitutive law (eps : C : eps)
                      T dudx00 = ur0 * drdx + us0 * dsdx + ut0 * dtdx;
                      T dudx01 = ur0 * drdy + us0 * dsdy + ut0 * dtdy;
                      T dudx02 = ur0 * drdz + us0 * dsdz + ut0 * dtdz;

                      T dudx10 = ur1 * drdx + us1 * dsdx + ut1 * dtdx;
                      T dudx11 = ur1 * drdy + us1 * dsdy + ut1 * dtdy;
                      T dudx12 = ur1 * drdz + us1 * dsdz + ut1 * dtdz;

                      T dudx20 = ur2 * drdx + us2 * dsdx + ut2 * dtdx;
                      T dudx21 = ur2 * drdy + us2 * dsdy + ut2 * dtdy;
                      T dudx22 = ur2 * drdz + us2 * dsdz + ut2 * dtdz;

                      T div_u = dudx00 + dudx11 + dudx22;

                      const T sigma_xx = lambda * div_u + two_mu * dudx00;
                      const T sigma_yy = lambda * div_u + two_mu * dudx11;
                      const T sigma_zz = lambda * div_u + two_mu * dudx22;
                      const T sigma_xy = mu * (dudx01 + dudx10);
                      const T sigma_xz = mu * (dudx02 + dudx20);
                      const T sigma_yz = mu * (dudx12 + dudx21);

                      // Component 0 directly into shared memory for step-8
                      s_rqr[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                        wdetJ * (sigma_xx * drdx + sigma_xy * drdy + sigma_xz * drdz);
                      s_rqs[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                        wdetJ * (sigma_xx * dsdx + sigma_xy * dsdy + sigma_xz * dsdz);
                      s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                        wdetJ * (sigma_xx * dtdx + sigma_xy * dtdy + sigma_xz * dtdz);

                      // Components 1 and 2 kept in registers for subsequent passes
                      r_f1r[p] = wdetJ * (sigma_xy * drdx + sigma_yy * drdy + sigma_yz * drdz);
                      r_f1s[p] = wdetJ * (sigma_xy * dsdx + sigma_yy * dsdy + sigma_yz * dsdz);
                      r_f1t[p] = wdetJ * (sigma_xy * dtdx + sigma_yy * dtdy + sigma_yz * dtdz);

                      r_f2r[p] = wdetJ * (sigma_xz * drdx + sigma_yz * drdy + sigma_zz * drdz);
                      r_f2s[p] = wdetJ * (sigma_xz * dsdx + sigma_yz * dsdy + sigma_zz * dsdz);
                      r_f2t[p] = wdetJ * (sigma_xz * dtdx + sigma_yz * dtdy + sigma_zz * dtdz);
                    }

                  if (one_thread_per_team) // compile-time constant
                    {
                      for (int p = 0; p < nq; ++p)
                        {
                          s_f[(tid * 6 + 0) * nq + p] = r_f1r[p];
                          s_f[(tid * 6 + 1) * nq + p] = r_f1s[p];
                          s_f[(tid * 6 + 2) * nq + p] = r_f1t[p];
                          s_f[(tid * 6 + 3) * nq + p] = r_f2r[p];
                          s_f[(tid * 6 + 4) * nq + p] = r_f2s[p];
                          s_f[(tid * 6 + 5) * nq + p] = r_f2t[p];
                        }
                    }
                }
              team_member.team_barrier();


            /*
            Stage 3: Adjoint differentiation & projection to GLL nodes for each component
            */
#pragma unroll
              for (int c = 0; c < 3; ++c)
                {
                  // c == 0 fluxes are already in shared memory from Stage 2
                  if (c > 0)
                    {
                      for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nq * nq;
                           tid += blockSize)
                        {
                          const int e = tid / (nq * nq);
                          const int q = tid % (nq * nq) / nq;
                          const int r = tid % nq;

                          if (one_thread_per_team) // compile-time constant
                            {
                              for (int p = 0; p < nq; ++p)
                                {
                                  r_f1s[p] = s_f[(tid * 6 + 1) * nq + p];
                                  r_f1t[p] = s_f[(tid * 6 + 2) * nq + p];
                                  r_f2s[p] = s_f[(tid * 6 + 4) * nq + p];
                                  r_f2t[p] = s_f[(tid * 6 + 5) * nq + p];
                                }
                            }

#pragma unroll
                          for (int p = 0; p < nq; ++p)
                            {
                              // r-fluxes (r_f1r / r_f2r) stay in registers for step-8 directly
                              s_rqs[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                                (c == 1) ? r_f1s[p] : r_f2s[p];
                              s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                                (c == 1) ? r_f1t[p] : r_f2t[p];
                            }
                        }
                      team_member.team_barrier();
                    }

                  // step-8 : Compute out vector in GL nodes
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nq * nq;
                       tid += blockSize)
                    {
                      int e = tid / (nq * nq);
                      int q = tid % (nq * nq) / nq;
                      int r = tid % nq;

                      if (one_thread_per_team) // compile-time constant
                        {
                          for (int n = 0; n < nq; ++n)
                            {
                              r_f1r[n] = s_f[(tid * 6 + 0) * nq + n];
                              r_f2r[n] = s_f[(tid * 6 + 3) * nq + n];
                            }
                        }

// copy to register
#pragma unroll
                      for (int n = 0; n < nq; n++)
                        {
                          if (c == 0)
                            r_p[n] = s_rqr[e * nq * nq * nq + n * nq * nq + q * nq + r];
                          r_q[n] = s_dbasis[q * nq + n];
                          r_r[n] = s_dbasis[r * nq + n];
                        }

#pragma unroll
                      for (int p = 0; p < nq; ++p)
                        {
                          T tmp0 = 0;
#pragma unroll
                          for (int n = 0; n < nq; ++n)
                            {
                              const T rp_n = (c == 0) ? r_p[n] : ((c == 1) ? r_f1r[n] : r_f2r[n]);
                              tmp0 += rp_n * c_dbasis[p * nq + n];
                            }

#pragma unroll
                          for (int n = 0; n < nq; ++n)
                            tmp0 += s_rqs[e * nq * nq * nq + p * nq * nq + n * nq + r] * r_q[n];

#pragma unroll
                          for (int n = 0; n < nq; ++n)
                            tmp0 += s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + n] * r_r[n];

                          s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp0;
                        }
                    }
                  team_member.team_barrier();


                  /*
                  Interpolate to GLL nodes
                  */

                  // step-9 : direction 2
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nq * nq;
                       tid += blockSize)
                    {
                      int e = tid / (nq * nq);
                      int q = tid % (nq * nq) / nq;
                      int p = tid % nq;

#pragma unroll
                      for (int r = 0; r < nq; ++r)
                        {
                          r_r[r] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p];
                        }

#pragma unroll
                      for (int k = 0; k < nm; ++k)
                        {
                          T tmp = 0.0;

#pragma unroll
                          for (int r = 0; r < nq; ++r)
                            {
                              tmp += c_basis[k * nq + r] * r_r[r];
                            }

                          s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
                        }
                    }
                  team_member.team_barrier();

                  // step-10 : direction 1
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nq;
                       tid += blockSize)
                    {
                      int e = tid / (nm * nq);
                      int k = tid % (nm * nq) / nq;
                      int p = tid % nq;

#pragma unroll
                      for (int q = 0; q < nq; ++q)
                        {
                          r_q[q] = s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p];
                        }

#pragma unroll
                      for (int j = 0; j < nm; ++j)
                        {
                          T tmp = 0.0;

#pragma unroll
                          for (int q = 0; q < nq; ++q)
                            {
                              tmp += c_basis[j * nq + q] * r_q[q];
                            }
                          s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
                        }
                    }
                  team_member.team_barrier();

                  // step-11 + step 12 : direction 0 and write d_out
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nm;
                       tid += blockSize)
                    {
                      int e = tid / (nm * nm);
                      int j = tid % (nm * nm) / nm;
                      int k = tid % nm;

#pragma unroll
                      for (int p = 0; p < nq; ++p)
                        {
                          r_p[p] = s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p];
                        }

#pragma unroll
                      for (int i = 0; i < nm; ++i)
                        {
                          T tmp = 0.0;
#pragma unroll
                          for (int p = 0; p < nq; ++p)
                            {
                              tmp += c_basis[i * nq + p] * r_p[p];
                            }

                          access.write(c, eb * nelmtPerBatch + e, i, j, k, tmp);
                        }
                    }
                  if (c == 2 && eb + team_member.league_size() < nbatches)
                    team_member.team_barrier();
                } // component loop

              eb += team_member.league_size();
            }
        });
    }

  private:
    std::size_t               nelmt;
    unsigned int              nelmtPerBatch;
    std::size_t               nbatches;
    Kokkos::Array<T, nm * nq> c_basis;
    Kokkos::Array<T, nq * nq> c_dbasis;
    View                      d_dbasis;
    T                         lambda;
    T                         mu;
  };
} // namespace Benchmark

#endif
