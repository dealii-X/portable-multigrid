// A BK3-style counterpart of the elasticity kernel of the dealii-X benchmarks
// (kernels/elasticity_benchmark_kernel.h): the same stages, but as in the BK3
// kernels (bk3_kokkos_kernels.h)
//  - the 1D shape data in global memory, copied to shared memory by every
//    team (the benchmark keeps them in the kernel closure, constant memory),
//  - the DoF values of all components gathered to shared memory in a step of
//    their own, and the results scattered from shared memory in a step of
//    their own (the benchmark reads and writes them in its first and last
//    sweep),
//  - the fluxes of components 1 and 2 in shared memory between stages 2 and 3
//    (the benchmark keeps them in registers),
//  - BK3's batch size (10800 bytes) and persistent teams (default_n_teams()).
// Data layouts and Access as in elasticity_benchmark_kernel.h.

#ifndef kernels_bk3_elasticity_kokkos_kernel_h
#define kernels_bk3_elasticity_kokkos_kernel_h

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "kernels/elasticity_benchmark_kernel.h"

namespace BK3
{
  namespace Parallel
  {
    template <typename T, const unsigned int nm, const unsigned int nq>
    class ElasticityKernel
    {
    public:
      // a team has nelmtPerBatch * nq * nq threads, the sweeps on the nodes use
      // nelmtPerBatch * nm * nm and nelmtPerBatch * nm * nq of them
      static_assert(nm >= 1 && nm <= nq, "the kernel needs 1 <= nm <= nq");

      using View = Kokkos::View<T *>;

      static constexpr unsigned int ncomp = 3;

      // teams of one thread on a host backend
      static constexpr bool host_backend =
        std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

      // BK3's batch rule: a 10800-byte budget for the shared values of an
      // element, here 13 * nq^3, at least one element
      static unsigned int
      default_elements_per_batch()
      {
        return std::max(1u, static_cast<unsigned int>(10800u / (13u * nq * nq * nq * sizeof(T))));
      }

      // BK3's number of teams: persistent teams, each one for two batches
      unsigned int
      default_n_teams() const
      {
        return std::max<std::size_t>(1, nbatches / 2);
      }

      ElasticityKernel(const std::size_t     nelmt,
                       const unsigned int    nelmtPerBatch,
                       const std::vector<T> &basis,
                       const std::vector<T> &dbasis,
                       const T               lambda,
                       const T               mu)
        : nelmt(nelmt)
        , nelmtPerBatch(nelmtPerBatch)
        , nbatches((nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
        , d_basis(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_basis"), nm * nq)
        , d_dbasis(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_dbasis"), nq * nq)
        , lambda(lambda)
        , mu(mu)
      {
        if (basis.size() != nm * nq || dbasis.size() != nq * nq)
          throw std::invalid_argument(
            "BK3::Parallel::ElasticityKernel: wrong size of the 1D shape data");

        // the 1D shape data stay in global memory and are copied to shared
        // memory by every team, as in the BK3 kernels
        const auto host_basis = Kokkos::create_mirror_view(d_basis);
        for (unsigned int i = 0u; i < nm * nq; ++i)
          host_basis(i) = basis[i];
        Kokkos::deep_copy(d_basis, host_basis);

        const auto host_dbasis = Kokkos::create_mirror_view(d_dbasis);
        for (unsigned int i = 0u; i < nq * nq; ++i)
          host_dbasis(i) = dbasis[i];
        Kokkos::deep_copy(d_dbasis, host_dbasis);
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

      unsigned int
      elements_per_batch() const
      {
        return nelmtPerBatch;
      }

      // threads per team: one per pencil (nelmtPerBatch * nq^2) unless a
      // smaller team is requested; one on a host backend
      unsigned int
      team_size(const unsigned int requested = 0) const
      {
        if (host_backend)
          return 1;
        return requested > 0 ? requested : nelmtPerBatch * nq * nq;
      }

      // out = A in on the element-local arrays of ElementAccess
      void
      apply(const View        &d_G,
            const View        &d_in,
            const View        &d_out,
            const unsigned int numBlocks,
            const unsigned int requested_team_size = 0) const
      {
        apply(d_G,
              Benchmark::ElementAccess<T, nm>{d_in, d_out, n_elements_padded()},
              numBlocks,
              requested_team_size);
      }

      // numBlocks teams, each one going through every numBlocks-th batch. Not
      // fenced.
      template <typename Access>
      void
      apply(const View        &d_G,
            const Access      &access,
            const unsigned int numBlocks,
            const unsigned int requested_team_size = 0) const
      {
        const std::size_t  nbatches      = this->nbatches;
        const unsigned int nelmtPerBatch = this->nelmtPerBatch;
        const auto         d_basis       = this->d_basis;
        const auto         d_dbasis      = this->d_dbasis;
        const T            lambda        = this->lambda;
        const T            mu            = this->mu;

        const unsigned int threadsPerBlock = team_size(requested_team_size);

        // Kokkos with shared memory
        const unsigned int ssize      = nm * nq + nq * nq + 13 * nelmtPerBatch * nq * nq * nq;
        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          "BK3::Parallel::ElasticityKernel", policy, KOKKOS_LAMBDA(member_type team_member) {
            T r_p[nq];
            T r_q[nq];
            T r_r[nq];

            T r_p0[nq];
            T r_p1[nq];
            T r_p2[nq];

            // quadrature fluxes of components 1 and 2 of one pencil
            T r_f1r[nq], r_f1s[nq], r_f1t[nq];
            T r_f2r[nq], r_f2s[nq], r_f2t[nq];

            // shared memory access
            T *scratch = (T *)team_member.team_shmem().get_shmem(shmem_size);

            T *s_basis  = scratch;
            T *s_dbasis = s_basis + nm * nq;

            T *s_wsp0 = s_dbasis + nq * nq;
            T *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;

            T *s_rqr = s_wsp1 + nelmtPerBatch * nq * nq * nq;
            T *s_rqs = s_rqr + nelmtPerBatch * nq * nq * nq;
            T *s_rqt = s_wsp0;

            T *s_u0   = s_rqs + nelmtPerBatch * nq * nq * nq;
            T *s_u1   = s_u0 + nelmtPerBatch * nq * nq * nq;
            T *s_u2   = s_u1 + nelmtPerBatch * nq * nq * nq;
            T *s_u[3] = {s_u0, s_u1, s_u2};

            // the fluxes of components 1 and 2 of all pencils, [pencil][6][p]
            T *s_f = s_u2 + nelmtPerBatch * nq * nq * nq;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();

            // copy the 1D shape data to shared memory
            for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_basis[tid] = d_basis[tid];
              }
            for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_dbasis[tid] = d_dbasis[tid];
              }
            team_member.team_barrier();


            // element batch iteration
            size_t eb = team_member.league_rank();
            while (eb < nbatches)
              {
                // the DoF values of all components to shared memory
                for (int c = 0; c < 3; ++c)
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nm;
                       tid += blockSize)
                    {
                      const int e = tid / (nm * nm);
                      const int j = tid % (nm * nm) / nm;
                      const int k = tid % nm;
                      for (int i = 0; i < nm; ++i)
                        s_u[c][e * nq * nq * nq + (i * nm + j) * nm + k] =
                          access.read(c, eb * nelmtPerBatch + e, i, j, k);
                    }
                team_member.team_barrier();

              /*
              Stage 1: Interpolate to GL nodes (all 3 components required before quadrature
              coupling)
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
                            r_p[i] = s_u[c][e * nq * nq * nq + (i * nm + j) * nm + k];
                          }

#pragma unroll
                        for (int p = 0; p < nq; ++p)
                          {
                            T tmp = 0.0;

#pragma unroll
                            for (int i = 0; i < nm; ++i)
                              {
                                tmp += s_basis[i * nq + p] * r_p[i];
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
                                tmp += s_basis[j * nq + q] * r_q[j];
                              }

                            s_wsp0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
                          }
                      }
                    team_member.team_barrier();

                    // step-4 : direction 2 (all blockSize == nelmtPerBatch * nq * nq threads
                    // active)
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
                                tmp += s_basis[k * nq + r] * r_r[k];
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
                            T dp = s_dbasis[n * nq + p];

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

                        // components 1 and 2 to shared memory below
                        r_f1r[p] = wdetJ * (sigma_xy * drdx + sigma_yy * drdy + sigma_yz * drdz);
                        r_f1s[p] = wdetJ * (sigma_xy * dsdx + sigma_yy * dsdy + sigma_yz * dsdz);
                        r_f1t[p] = wdetJ * (sigma_xy * dtdx + sigma_yy * dtdy + sigma_yz * dtdz);

                        r_f2r[p] = wdetJ * (sigma_xz * drdx + sigma_yz * drdy + sigma_zz * drdz);
                        r_f2s[p] = wdetJ * (sigma_xz * dsdx + sigma_yz * dsdy + sigma_zz * dsdz);
                        r_f2t[p] = wdetJ * (sigma_xz * dtdx + sigma_yz * dtdy + sigma_zz * dtdz);
                      }

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

                            for (int p = 0; p < nq; ++p)
                              {
                                r_f1s[p] = s_f[(tid * 6 + 1) * nq + p];
                                r_f1t[p] = s_f[(tid * 6 + 2) * nq + p];
                                r_f2s[p] = s_f[(tid * 6 + 4) * nq + p];
                                r_f2t[p] = s_f[(tid * 6 + 5) * nq + p];
                              }

#pragma unroll
                            for (int p = 0; p < nq; ++p)
                              {
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

                        for (int n = 0; n < nq; ++n)
                          {
                            r_f1r[n] = s_f[(tid * 6 + 0) * nq + n];
                            r_f2r[n] = s_f[(tid * 6 + 3) * nq + n];
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
                                tmp0 += rp_n * s_dbasis[p * nq + n];
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
                                tmp += s_basis[k * nq + r] * r_r[r];
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
                                tmp += s_basis[j * nq + q] * r_q[q];
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
                                tmp += s_basis[i * nq + p] * r_p[p];
                              }

                            s_u[c][e * nq * nq * nq + (i * nm + j) * nm + k] = tmp;
                          }
                      }
                  } // component loop
                team_member.team_barrier();

                // the results of all components to the global vector
                for (int c = 0; c < 3; ++c)
                  for (unsigned int tid = threadIdx; tid < nelmtPerBatch * nm * nm;
                       tid += blockSize)
                    {
                      const int e = tid / (nm * nm);
                      const int j = tid % (nm * nm) / nm;
                      const int k = tid % nm;
                      for (int i = 0; i < nm; ++i)
                        access.write(c,
                                     eb * nelmtPerBatch + e,
                                     i,
                                     j,
                                     k,
                                     s_u[c][e * nq * nq * nq + (i * nm + j) * nm + k]);
                    }
                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
      }

    private:
      std::size_t  nelmt;
      unsigned int nelmtPerBatch;
      std::size_t  nbatches;
      View         d_basis;
      View         d_dbasis;
      T            lambda;
      T            mu;
    };
  } // namespace Parallel
} // namespace BK3

#endif
