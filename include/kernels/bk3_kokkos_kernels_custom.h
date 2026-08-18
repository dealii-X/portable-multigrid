#ifndef bk3_kokkos_kernels_custom_h
#define bk3_kokkos_kernels_custom_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

#include "kernels/portable_evaluation_kernels.h"
#include "kernels/portable_tensor_product_kernels.h"

// Structural twin of BK3::Parallel::KokkosKernel (bk3_kokkos_kernel.h),
// composed from the generic building blocks in kernels/portable_tensor_product_kernels.h instead of
// ten hand-written steps. Kept in its own file/namespace, alongside the
// original, so the two can be run side by side and diffed for correctness
// before anything switches over to this version -- see the discussion in
// the unify-kernels session about factoring the sum-factorization logic
// shared across include/kernels/ into a small number of generic,
// GPU-batched primitives.
//
// Shared memory is organized as two pools instead of BK3's four named,
// individually-aliased buffers: a single "values" slot (nelmtPerBatch *
// nq^dim elements) and a "gradients" pool of dim consecutive slots of the
// same size. The dof->quad interpolation sweep (steps 2-4) ping-pongs
// through the gradients pool's slots as scratch and lands its result in the
// values slot; the fused collocated-gradient step (5-6) reads/writes the
// values slot in place, using the gradients pool for its own (differently
// purposed) directional-derivative intermediates; the quad->dof integration
// sweep (7-9) ping-pongs back out through the gradients pool. Total
// footprint is (1 + dim) slots -- the same 4 slots BK3 uses for dim == 3,
// and one fewer (3) for dim == 2, since BK3's own 4-slot budget is sized
// for the dim == 3 case regardless of dim.
//
// Steps 1 (copy-in) and 10 (scatter) keep BK3::Parallel::KokkosKernel's
// logic verbatim, just reading/writing whichever pool slot the surrounding
// sequence puts the live data in.
DEAL_II_NAMESPACE_OPEN

namespace BK3Custom
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

      // 1 values slot + dim gradients-pool slots -- see the scratch-pointer
      // declarations below for how these replace BK3's original 4 fixed,
      // individually-named buffers.
      constexpr int n_scratch_arrays = 1 + dim;

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
        const int ssize =
          nm * nq +                                    // shape values
          nq * nq +                                    // co-shape gradients
          n_scratch_arrays * nelmtPerBatch * nq_total; // values slot + dim gradients-pool slots

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

          // Single "values" slot plus a "gradients" pool of dim
          // consecutive slots (see the file-level comment above),
          // replacing BK3's original 4 fixed, individually-named buffers
          // (s_wsp0, s_wsp1, s_rqr, s_rqs/s_rqt) with this (1 + dim)-slot
          // layout. Steps 2-4/7-9 ping-pong between s_values and
          // s_gradients's first slot exactly as they used to ping-pong
          // between s_wsp0/s_wsp1; steps 5-6 then read "the values" from
          // whichever of s_values/s_gradients[0] steps 2-4 left them in
          // (dim-dependently -- dim == 2 lands back in s_values, dim == 3
          // in s_gradients[0], same asymmetry the old s_wsp0/s_wsp1 code
          // already had) and write their directional-derivative
          // intermediates (formerly s_rqr/s_rqs/s_rqt) into whichever
          // slots are free: s_gradients[0]/[1] for dim == 2 (s_values
          // stays untouched, still holding the values being read), or
          // s_gradients[1]/[2] plus s_values itself for dim == 3 (s_values
          // is free at that point, exactly as s_wsp0 was when s_rqt
          // aliased onto it).
          Number *s_values    = s_co_shape_gradients + nq * nq;
          Number *s_gradients = s_values + nelmtPerBatch * nq_total;

          const int slot = nelmtPerBatch * nq_total;

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
                                s_values[shared_idx] = 0;
                              else
                                s_values[shared_idx] = d_in[dof_index];
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
                                s_values[shared_idx] = 0;
                              else
                                s_values[shared_idx] = d_in[dof_index];
                            }
                        }
                    }
                }
                team_member.team_barrier();
              }
              // steps 2-4: interpolate dof -> quad, one direction at a
              // time, via Custom::Parallel::
              // FEEvaluationImplTransformToCollocation::evaluate()
              // (kernels/portable_evaluation_kernels.h), BK3's counterpart
              // of deal.II's own struct of the same name. Replaces the
              // per-direction if constexpr (dim == 2)/(dim == 3) unroll
              // below (itself already verified bit-identical via
              // correctness_tests/check_correctness_common_kernels/) with a
              // single call; `in == out == s_values` here since steps
              // 2-4/7-9 always start and end at s_values regardless of dim
              // (see the scratch-pointer declarations above), and
              // `scratch == s_gradients` supplies the gradients pool's
              // first (dim == 2) or first two (dim == 3) slots as
              // intermediate storage. Original versions commented out below
              // for reference/diff, per request -- not deleted.
              {
                Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nm, nq, Number>::
                  evaluate(team_member,
                          s_shape_values,
                          s_values,
                          s_values,
                          s_gradients,
                          slot,
                          c_nelmtPerBatch,
                          threadIdx,
                          blockSize);
              }

              // Prior version, via Custom::Parallel::EvaluatorTensorProduct
              // directly (kernels/portable_tensor_product_kernels.h), one
              // if constexpr (dim == 2)/(dim == 3) unroll per call site:
              //
              // {
              //   const Custom::Parallel::EvaluatorTensorProduct<dim, nm, nq, Number> evaluator(
              //     team_member, s_shape_values, nullptr, c_nelmtPerBatch, threadIdx, blockSize);
              //
              //   // step-2 : direction 0
              //   if constexpr (dim == 2)
              //     {
              //       evaluator.template values<0, true, false>(s_values, s_gradients);
              //       evaluator.template values<1, true, false>(s_gradients, s_values);
              //     }
              //   else if constexpr (dim == 3)
              //     {
              //       evaluator.template values<0, true, false>(s_values, s_gradients);
              //
              //       // step-3 : direction 1
              //       evaluator.template values<1, true, false>(s_gradients, s_gradients + slot);
              //       // step-4 : direction 2
              //       evaluator.template values<2, true, false>(s_gradients + slot, s_values);
              //     }
              // }

              // Original hand-written steps 2-4 (pre-EvaluatorTensorProduct):
              //
                // {
                //   // step-2 : direction 0
                //   {
                //     constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);
                //
                //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //          tid += blockSize)
                //       {
                //         const int e = tid / co_dimension_size;
                //
                //         if (dim == 2)
                //           {
                //             const int j = tid % nm;
                //
                //             Custom::Parallel::
                //               apply_matrix_vector_product<nm, nq, true, false, 1, 1>(
                //                 s_shape_values,
                //                 s_wsp0 + e * nm * nm + j * nm,
                //                 s_wsp1 + e * nq * nm + j * nq);
                //           }
                //         else if (dim == 3)
                //           {
                //             const int k = (tid % co_dimension_size) / nm;
                //             const int j = tid % nm;
                //
                //             Custom::Parallel::
                //               apply_matrix_vector_product<nm, nq, true, false, 1, 1>(
                //                 s_shape_values,
                //                 s_wsp0 + e * nm * nm * nm + k * nm * nm + j * nm,
                //                 s_wsp1 + e * nq * nm * nm + k * nq * nm + j * nq);
                //           }
                //       }
                //     team_member.team_barrier();
                //   }
                //
                //   // step-3 : direction 1
                //   {
                //     constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);
                //
                //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //          tid += blockSize)
                //       {
                //         const int e = tid / co_dimension_size;
                //
                //         if (dim == 2)
                //           {
                //             const int p = tid % nq;
                //
                //             Custom::Parallel::
                //               apply_matrix_vector_product<nm, nq, true, false, nq, nq>(
                //                 s_shape_values, s_wsp1 + e * nq * nm + p, s_wsp0 + e * nq * nq +
                //                 p);
                //           }
                //         else if (dim == 3)
                //           {
                //             const int k = (tid % co_dimension_size) / nq;
                //             const int p = tid % nq;
                //
                //             Custom::Parallel::
                //               apply_matrix_vector_product<nm, nq, true, false, nq, nq>(
                //                 s_shape_values,
                //                 s_wsp1 + e * nq * nm * nm + k * nq * nm + p,
                //                 s_wsp0 + e * nq * nq * nm + k * nq * nq + p);
                //           }
                //       }
                //     team_member.team_barrier();
                //   }
                //
                //   // step-4 : direction 2
                //   if (dim == 3)
                //     {
                //       constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);
                //
                //       for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //            tid += blockSize)
                //         {
                //           int e = tid / co_dimension_size;
                //
                //           int q = (tid % co_dimension_size) / nq;
                //           int p = tid % nq;
                //
                //           Custom::Parallel::
                //             apply_matrix_vector_product<nm, nq, true, false, nq * nq, nq * nq>(
                //               s_shape_values,
                //               s_wsp0 + e * nq * nq * nm + q * nq + p,
                //               s_wsp1 + e * nq * nq * nq + q * nq + p);
                //         }
                //       team_member.team_barrier();
                //     }
                // }
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
                              r_q[n] = s_values[e * nq * nq + n * nq + p];
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
                                  qr += r_p[n] * s_values[e * nq * nq + q * nq + n];
                                  qs += s_co_shape_gradients[n * nq + q] * r_q[n];
                                }

                              // Apply chain rule -- writes into the
                              // gradients pool's first two slots (free at
                              // this point for dim == 2: s_gradients was
                              // only ping-pong scratch for steps 2-4, and
                              // "the values" landed back in s_values, not
                              // here).
                              s_gradients[e * nq * nq + q * nq + p] = Grr * qr + Grs * qs;

                              (s_gradients + slot)[e * nq * nq + q * nq + p] = Grs * qr + Gss * qs;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int q = tid % (co_dimension_size) / nq;
                          const int p = tid % nq;

                          // copy to register -- "the values" landed in
                          // s_gradients's first slot for dim == 3 (odd
                          // number of ping-pong hops in steps 2-4).
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = s_co_shape_gradients[n * nq + p];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                              r_r[n] = s_values[e * nq * nq * nq + n * nq * nq + q * nq + p];
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
                                    r_p[n] * s_values[e * nq * nq * nq + r * nq * nq + q * nq + n];
                                  qs +=
                                    r_q[n] * s_values[e * nq * nq * nq + r * nq * nq + n * nq + p];
                                  qt += s_co_shape_gradients[n * nq + r] * r_r[n];
                                }

                              // Apply chain rule -- writes into the
                              // gradients pool's remaining two slots
                              // (s_gradients's first slot still holds "the
                              // values", read above, so it's not
                              // overwritten here) plus s_values itself,
                              // free at this point for dim == 3 exactly as
                              // s_wsp0 was when s_rqt used to alias onto it.
                              s_gradients[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grr * qr + Grs * qs + Grt * qt;

                              (s_gradients + slot)[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grs * qr + Gss * qs + Gst * qt;

                              (s_gradients +
                               2 * slot)[e * nq * nq * nq + r * nq * nq + q * nq + p] =
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
                              r_q[n] = (s_gradients + slot)[e * nq * nq + n * nq + p];
                            }

                          for (int q = 0; q < nq; ++q)
                            {
                              Number tmp0 = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp0 += s_gradients[e * nq * nq + q * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += r_q[n] * s_co_shape_gradients[q * nq + n];

                              s_values[e * nq * nq + q * nq + p] = tmp0;
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

                              r_r[n] = (s_gradients +
                                        2 * slot)[e * nq * nq * nq + n * nq * nq + q * nq + p];
                            }

                          for (int r = 0; r < nq; ++r)
                            {
                              Number tmp0 = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp0 +=
                                  s_gradients[e * nq * nq * nq + r * nq * nq + q * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += (s_gradients +
                                         slot)[e * nq * nq * nq + r * nq * nq + n * nq + p] *
                                        r_q[n];

                              for (int n = 0; n < nq; ++n)
                                tmp0 += r_r[n] * s_co_shape_gradients[r * nq + n];

                              s_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp0;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                /*
                Interpolate to GLL nodes
                */

                // steps 7-9: integrate quad -> dof, one direction at a time
                // in reverse order, via Custom::Parallel::
                // FEEvaluationImplTransformToCollocation::integrate()
                // (kernels/portable_evaluation_kernels.h) -- mirror of
                // evaluate() above, same in/out/scratch routing (in == out
                // == s_values, scratch == s_gradients). Original versions
                // commented out below for reference/diff, per request --
                // not deleted.
                {
                  Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nm, nq, Number>::
                    integrate(team_member,
                             s_shape_values,
                             s_values,
                             s_values,
                             s_gradients,
                             slot,
                             c_nelmtPerBatch,
                             threadIdx,
                             blockSize);
                }

                // Prior version, via Custom::Parallel::EvaluatorTensorProduct
                // directly (kernels/portable_tensor_product_kernels.h):
                //
                // {
                //   const Custom::Parallel::EvaluatorTensorProduct<dim, nm, nq, Number> evaluator(
                //     team_member, s_shape_values, nullptr, c_nelmtPerBatch, threadIdx, blockSize);
                //
                //   if constexpr (dim == 2)
                //     {
                //       evaluator.template values<1, false, false>(s_values, s_gradients);
                //       evaluator.template values<0, false, false>(s_gradients, s_values);
                //     }
                //   // step-7 : direction 2
                //   if constexpr (dim == 3)
                //     {
                //       evaluator.template values<2, false, false>(s_values, s_gradients);
                //
                //       // step-8 : direction 1
                //       evaluator.template values<1, false, false>(s_gradients, s_gradients + slot);
                //
                //       // step-9 : direction 0
                //       evaluator.template values<0, false, false>(s_gradients + slot, s_values);
                //     }
                // }

                // Original hand-written steps 7-9 (pre-EvaluatorTensorProduct):
                //
                // // step-7 : direction 2
                // if (dim == 3)
                //   {
                //     constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);
                //
                //     for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //          tid += blockSize)
                //       {
                //         const int e = tid / co_dimension_size;
                //
                //         const int q = (tid % co_dimension_size) / nq;
                //         const int p = tid % nq;
                //
                //         Custom::Parallel::
                //           apply_matrix_vector_product<nm, nq, false, false, nq * nq, nq * nq>(
                //             s_shape_values,
                //             s_wsp1 + e * nq * nq * nq + q * nq + p,
                //             s_wsp0 + e * nq * nq * nm + q * nq + p);
                //       }
                //     team_member.team_barrier();
                //   }
                //
                // // step-8 : direction 1
                // {
                //   constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);
                //
                //   for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //        tid += blockSize)
                //     {
                //       const int e = tid / co_dimension_size;
                //       if (dim == 2)
                //         {
                //           const int p = tid % nq;
                //
                //           Custom::Parallel::
                //             apply_matrix_vector_product<nm, nq, false, false, nq, nq>(
                //               s_shape_values, s_wsp0 + e * nq * nq + p, s_wsp1 + e * nq * nm +
                //               p);
                //         }
                //       else if (dim == 3)
                //         {
                //           const int k = (tid % co_dimension_size) / nq;
                //           const int p = tid % nq;
                //
                //           Custom::Parallel::
                //             apply_matrix_vector_product<nm, nq, false, false, nq, nq>(
                //               s_shape_values,
                //               s_wsp0 + e * nq * nq * nm + k * nq * nq + p,
                //               s_wsp1 + e * nq * nm * nm + k * nq * nm + p);
                //         }
                //     }
                //   team_member.team_barrier();
                // }
                //
                // // step-9 : direction 0
                // {
                //   constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);
                //
                //   for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                //        tid += blockSize)
                //     {
                //       const int e = tid / co_dimension_size;
                //
                //       if (dim == 2)
                //         {
                //           const int j = tid % nm;
                //
                //           Custom::Parallel::apply_matrix_vector_product<nm, nq, false, false, 1,
                //           1>(
                //             s_shape_values,
                //             s_wsp1 + e * nq * nm + j * nq,
                //             s_wsp0 + e * nm * nm + j * nm);
                //         }
                //       else if (dim == 3)
                //         {
                //           const int k = (tid % co_dimension_size) / nm;
                //           const int j = tid % nm;
                //
                //           Custom::Parallel::apply_matrix_vector_product<nm, nq, false, false, 1,
                //           1>(
                //             s_shape_values,
                //             s_wsp1 + e * nq * nm * nm + k * nq * nm + j * nq,
                //             s_wsp0 + e * nm * nm * nm + k * nm * nm + j * nm);
                //         }
                //     }
                //   team_member.team_barrier();
                // }

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
                                  Kokkos::atomic_add(&d_out[dof_index], s_values[shared_idx]);
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
                                  Kokkos::atomic_add(&d_out[dof_index], s_values[shared_idx]);
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


      // template <int dim, int nm, int nq, typename Number>
      // void
      // KokkosKernel(const DeviceView<Number> d_shape_values,
      //              const DeviceView<Number> d_co_shape_gradients,
      //              const DeviceView<Number> d_G,
      //              const DeviceView<Number> d_in,
      //              DeviceView<Number>       d_out,
      //              const DoFIndicesView     dof_indices,
      //              const unsigned int       n_cells,
      //              const unsigned int       n_blocks          = numbers::invalid_unsigned_int,
      //              const unsigned int       threads_per_block = numbers::invalid_unsigned_int,
      //              const CellRangeIdView    cell_range_ids    = CellRangeIdView())
      // {
      //   if (n_cells == 0)
      //     return;

      //   static_assert(dim == 2 || dim == 3, "dim must be 2 or 3");

      //   constexpr int nq_total                   = Utilities::pow(nq, dim);
      //   constexpr int nm_total                   = Utilities::pow(nm, dim);
      //   constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

      //   // finding the batch size
      //   constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      //   // 1 values slot + dim gradients-pool slots (see file comment).
      //   constexpr int n_scratch_arrays = 1 + dim;

      //   if (cell_range_ids.size() > 0)
      //     AssertDimension(cell_range_ids.size(), n_cells);

      //   // evaluate_and_multiply_tensor() below indexes d_G by
      //   // eb * nelmtPerBatch + e directly, not remapped through
      //   // cell_range_ids the way dof_indices is in steps 1/10 -- see its doc
      //   // comment in kernels/portable_tensor_product_kernels.h. No current caller passes a
      //   non-empty
      //   // cell_range_ids to this kernel; this catches it loudly rather than
      //   // silently reading the wrong cell's geometric factors if that ever
      //   // changes without also revisiting this.
      //   AssertThrow(cell_range_ids.size() == 0,
      //              ExcMessage("BK3Custom::Parallel::KokkosKernel's collocated-gradient step does
      //              not "
      //                         "support a non-empty cell_range_ids."));

      //   const int nelmt = n_cells;

      //   const int nelmtPerBatch =
      //     std::max(1,
      //              static_cast<int>(shmemPerBlock / (n_scratch_arrays * nq_total) /
      //              sizeof(Number)));

      //   const int numBlocks = std::max(1,
      //                                  ((n_blocks == numbers::invalid_unsigned_int) ?
      //                                     ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
      //                                     static_cast<int>(n_blocks)));

      //   const int threadsPerBlock = std::max(1,
      //                                        ((threads_per_block ==
      //                                        numbers::invalid_unsigned_int)
      //                                        ?
      //                                           (Utilities::pow(nq, dim - 1) * nelmtPerBatch) :
      //                                           static_cast<int>(threads_per_block)));

      //   {
      //     const int ssize = nm * nq +                                    // shape values
      //                       nq * nq +                                    // co-shape gradients
      //                       n_scratch_arrays * nelmtPerBatch * nq_total; // values + gradients
      //                       pools

      //     const unsigned int shmem_size = ssize * sizeof(Number);

      //     typedef Kokkos::TeamPolicy<>::member_type member_type;
      //     Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
      //     policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

      //     Kokkos::parallel_for(
      //       policy, KOKKOS_LAMBDA(member_type team_member) {
      //         Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

      //         Number *s_shape_values       = scratch;
      //         Number *s_co_shape_gradients = s_shape_values + nq * nm;

      //         // Single work pool: offset 0 is the "values" slot, offsets
      //         // slot, 2*slot, (3*slot for dim == 3) are the "gradients" pool
      //         // (see file comment for how the two are used in sequence).
      //         Number *s_work = s_co_shape_gradients + nq * nq;

      //         const int slot = nelmtPerBatch * nq_total;

      //         const int off_values = 0;
      //         const int off_g0     = slot;
      //         const int off_g1     = 2 * slot;
      //         const int off_g2     = 3 * slot; // dim == 3 only

      //         const int threadIdx = team_member.team_rank();
      //         const int blockSize = team_member.team_size();

      //         // copy to shared memory
      //         for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
      //           {
      //             s_shape_values[tid] = d_shape_values[tid];
      //           }

      //         for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
      //           {
      //             s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
      //           }
      //         team_member.team_barrier();

      //         /*
      //         Interpolate to GL nodes
      //         */

      //         // element batch iteration
      //         int eb = team_member.league_rank();

      //         while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
      //           {
      //             // current nelmtPerBatch (edge case, last batch size can be
      //             // less)
      //             const int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
      //                                           (nelmt - eb * nelmtPerBatch) :
      //                                           nelmtPerBatch;

      //             // step-1 : Copy from in to the scratch values (lands in
      //             // the gradients pool's first slot -- see steps 2-4 below)
      //             {
      //               Number *s_wsp0 = s_work + off_g0;

      //               constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      //               for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
      //                    tid += blockSize)
      //                 {
      //                   const int e = tid / co_dimension_size;

      //                   unsigned int global_cell_index = eb * nelmtPerBatch + e;

      //                   if (cell_range_ids.size() > 0)
      //                     global_cell_index = cell_range_ids(global_cell_index);

      //                   if (dim == 2)
      //                     {
      //                       const int i = tid % nm;

      //                       for (int j = 0; j < nm; ++j)
      //                         {
      //                           const int          local_idx = j * nm + i;
      //                           const unsigned int dof_index =
      //                             dof_indices(local_idx, global_cell_index);
      //                           const int shared_idx = e * nm_total + local_idx;

      //                           if (dof_index == numbers::invalid_unsigned_int)
      //                             s_wsp0[shared_idx] = 0;
      //                           else
      //                             s_wsp0[shared_idx] = d_in[dof_index];
      //                         }
      //                     }
      //                   else if (dim == 3)
      //                     {
      //                       const int j = (tid % co_dimension_size) / nm;
      //                       const int i = tid % nm;

      //                       for (int k = 0; k < nm; ++k)
      //                         {
      //                           const int          local_idx = k * nm * nm + j * nm + i;
      //                           const unsigned int dof_index =
      //                             dof_indices(local_idx, global_cell_index);
      //                           const int shared_idx = e * nm_total + local_idx;

      //                           if (dof_index == numbers::invalid_unsigned_int)
      //                             s_wsp0[shared_idx] = 0;
      //                           else
      //                             s_wsp0[shared_idx] = d_in[dof_index];
      //                         }
      //                     }
      //                 }
      //               team_member.team_barrier();
      //             }

      //             // Single EvaluatorTensorProduct instance for both the
      //             // steps-2-4 interpolation sweep and the steps-7-9
      //             // integration sweep below -- matrix and the batching
      //             // parameters are the same throughout this element-batch
      //             // iteration, so one instance suffices (mirrors deal.II's
      //             // own EvaluatorTensorProduct usage: construct once per
      //             // team, call .values<direction, ...>(in, out) once per
      //             // direction, in/out taken as buffers directly -- see
      //             // kernels/portable_tensor_product_kernels.h).
      //             const Custom::Parallel::EvaluatorTensorProduct<dim, nm, nq, Number> evaluator(
      //               team_member, s_shape_values, nullptr, c_nelmtPerBatch, threadIdx, blockSize);

      //             // steps 2-4: interpolate dof -> quad, one direction at a
      //             // time, ping-ponging through the gradients pool's slots and
      //             // landing the result in the values slot. Each call ends
      //             // with its own team_barrier().
      //             if constexpr (dim == 2)
      //               {
      //                 evaluator.template values<0, true, false>(s_work + off_g0, s_work +
      //                 off_g1); evaluator.template values<1, true, false>(s_work + off_g1, s_work
      //                 + off_values);
      //               }
      //             else if constexpr (dim == 3)
      //               {
      //                 evaluator.template values<0, true, false>(s_work + off_g0, s_work +
      //                 off_g1); evaluator.template values<1, true, false>(s_work + off_g1, s_work
      //                 + off_g2); evaluator.template values<2, true, false>(s_work + off_g2,
      //                 s_work + off_values);
      //               }

      //             // steps 5-6: evaluate collocated gradients + apply the
      //             // geometric-factor tensor, then integrate back. Reads and
      //             // writes the values slot in place; uses the gradients pool
      //             // (now free) for its own directional-derivative
      //             // intermediates. Each call ends with its own
      //             // team_barrier().
      //             //
      //             // evaluate_and_multiply_tensor() takes a flat, already-
      //             // resolved offset into d_G rather than doing a
      //             // cell_range_ids lookup itself (see its doc comment in
      //             // kernels/portable_tensor_product_kernels.h) -- unlike dof_indices in steps
      //             1/10
      //             // above, d_G is NOT remapped through cell_range_ids here
      //             // (enforced by the AssertThrow near the top of this
      //             // function).
      //             {
      //               const int g_offset =
      //                 static_cast<int>(eb * nelmtPerBatch) * symmetric_tensor_dimension *
      //                 nq_total;
      //               const int gradient_slot_stride = slot;

      //               Custom::Parallel::evaluate_and_multiply_tensor<dim, nq>(team_member,
      //                                                                       s_co_shape_gradients,
      //                                                                       d_G,
      //                                                                       g_offset,
      //                                                                       s_work,
      //                                                                       off_values,
      //                                                                       off_g0,
      //                                                                       gradient_slot_stride,
      //                                                                       c_nelmtPerBatch,
      //                                                                       threadIdx,
      //                                                                       blockSize);

      //               Custom::Parallel::integrate<dim, nq>(team_member,
      //                                                    s_co_shape_gradients,
      //                                                    s_work,
      //                                                    off_g0,
      //                                                    gradient_slot_stride,
      //                                                    off_values,
      //                                                    c_nelmtPerBatch,
      //                                                    threadIdx,
      //                                                    blockSize);
      //             }

      //             // steps 7-9: integrate quad -> dof, one direction at a
      //             // time, in reverse order, ping-ponging back out through the
      //             // gradients pool. Final result lands in off_g2 (dim == 3)
      //             // or off_g1 (dim == 2), read by step-10 below.
      //             int result_offset;
      //             if constexpr (dim == 2)
      //               {
      //                 evaluator.template values<1, false, false>(s_work + off_values, s_work +
      //                 off_g0); evaluator.template values<0, false, false>(s_work + off_g0, s_work
      //                 + off_g1);

      //                 result_offset = off_g1;
      //               }
      //             else if constexpr (dim == 3)
      //               {
      //                 evaluator.template values<2, false, false>(s_work + off_values, s_work +
      //                 off_g0); evaluator.template values<1, false, false>(s_work + off_g0, s_work
      //                 + off_g1); evaluator.template values<0, false, false>(s_work + off_g1,
      //                 s_work + off_g2);

      //                 result_offset = off_g2;
      //               }

      //             // step-10 : Copy the final result back to global out vector
      //             {
      //               Number *s_result = s_work + result_offset;

      //               constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      //               for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
      //                    tid += blockSize)
      //                 {
      //                   const int e = tid / co_dimension_size;

      //                   unsigned int global_cell_index = eb * nelmtPerBatch + e;

      //                   if (cell_range_ids.size() > 0)
      //                     global_cell_index = cell_range_ids(global_cell_index);

      //                   if (dim == 2)
      //                     {
      //                       const int i = tid % nm;

      //                       for (int j = 0; j < nm; ++j)
      //                         {
      //                           const int          local_idx = j * nm + i;
      //                           const unsigned int dof_index =
      //                             dof_indices(local_idx, global_cell_index);
      //                           const int shared_idx = e * nm_total + local_idx;

      //                           if (dof_index != numbers::invalid_unsigned_int)
      //                             Kokkos::atomic_add(&d_out[dof_index], s_result[shared_idx]);
      //                         }
      //                     }
      //                   else if (dim == 3)
      //                     {
      //                       const int j = (tid % co_dimension_size) / nm;
      //                       const int i = tid % nm;

      //                       for (int k = 0; k < nm; ++k)
      //                         {
      //                           const int          local_idx = k * nm * nm + j * nm + i;
      //                           const unsigned int dof_index =
      //                             dof_indices(local_idx, global_cell_index);
      //                           const int shared_idx = e * nm_total + local_idx;

      //                           if (dof_index != numbers::invalid_unsigned_int)
      //                             Kokkos::atomic_add(&d_out[dof_index], s_result[shared_idx]);
      //                         }
      //                     }
      //                 }
      //               team_member.team_barrier();
      //             }

      //             eb += team_member.league_size();
      //           }
      //       });

      //     Kokkos::fence();
      //   }
      // }

    } // namespace Parallel
  } // namespace BK3Custom

  DEAL_II_NAMESPACE_CLOSE

#endif
