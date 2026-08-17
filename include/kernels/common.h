#ifndef kernels_common_h
#define kernels_common_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

// Generic, GPU-batched sum-factorization building blocks shared by the
// kernels in this directory (BK3 today, RT later -- see
// kernels/kokkos_kernels_rt.h, whose s_wsp0/s_wsp1/s_uq_0/s_uq_1/s_uq_2 are
// the same shape of same-sized, per-component/per-direction scratch slots
// this file is designed to address). Each primitive takes a single
// caller-owned shared-memory work array plus integer element offsets into
// it, rather than several separately-named buffers -- this is what lets a
// caller with more scratch slots in flight (e.g. RT's per-vector-component
// buffers, or BK3's own s_wsp0/s_wsp1 ping-pong) reuse the same primitive
// without the primitive needing to know how many named buffers exist. Each
// primitive also ends with its own team_barrier(), so downstream reads of
// its output are always safe without the caller having to remember to
// barrier -- mirroring how deal.II's own (non-Portable)
// internal::EvaluatorTensorProduct::apply_matrix_vector_product() and
// Portable::internal::EvaluatorTensorProduct are structured, generalized
// here to a batch of nelmtPerBatch cells sharing one team, to rectangular
// (n_rows != n_columns) shape matrices, and to a shared offset-addressed
// work array.
namespace Common
{
  namespace Parallel
  {
    using TeamHandle = Kokkos::TeamPolicy<>::member_type;

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    // Truly one-dimensional matrix-vector product along a single fiber:
    // contracts in[0], in[stride_in], ..., in[(mm-1)*stride_in] against
    // `matrix` (n_rows x n_columns, row-major; mm/nn below), writing
    // out[0], out[stride_out], ..., out[(nn-1)*stride_out]. Unlike the
    // batched overload below, this has no notion of dim, direction, a cell
    // batch, or any other tensor axis -- it's the single-fiber building
    // block that overload composes by looping over fibers, exposed
    // directly for callers that want to control the fiber layout/looping
    // themselves. Mirrors deal.II's own CPU
    // internal::apply_matrix_vector_product() (tensor_product_kernels.h),
    // which internal::EvaluatorTensorProduct::apply() loops over n_blocks1
    // * n_blocks2 times in exactly this role. No team_barrier(): this does
    // the work of exactly one fiber for whichever single thread calls it;
    // the caller parallelizes across fibers and handles barriers.
    template <int n_rows, int n_columns, bool contract_over_rows, bool add, typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const Number *matrix,
                                const Number *in,
                                Number       *out,
                                const int     stride_in,
                                const int     stride_out)
    {
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      // Cache the input fiber in registers once, rather than re-reading
      // shared memory for every output index below.
      Number r_in[mm];
      for (int k = 0; k < mm; ++k)
        r_in[k] = in[k * stride_in];

      for (int q = 0; q < nn; ++q)
        {
          Number sum = 0;
          for (int k = 0; k < mm; ++k)
            {
              const int row = contract_over_rows ? k : q;
              const int col = contract_over_rows ? q : k;
              sum += matrix[row * n_columns + col] * r_in[k];
            }

          if constexpr (add)
            out[q * stride_out] += sum;
          else
            out[q * stride_out] = sum;
        }
    }



    // Batched single-direction tensor contraction: for each of
    // c_nelmtPerBatch cells in the batch, contracts scratch[in_offset...]
    // along logical tensor direction `direction` against `matrix` (n_rows x
    // n_columns, row-major), writing the result to scratch[out_offset...].
    // Composed from the single-fiber overload above -- this just works out
    // which fiber (base pointer + stride) each thread owns and loops over
    // all of them.
    //
    // `contract_over_rows` selects which side of the (possibly rectangular)
    // matrix is summed over: true contracts over the n_rows index (the
    // "dof -> quad" / interpolation direction, e.g. BK3's steps 2-4), false
    // contracts over the n_columns index (the "quad -> dof" / integration
    // direction, e.g. BK3's steps 7-9), reusing the same matrix either way.
    //
    // The tensor axes are assumed laid out fastest-to-slowest in increasing
    // axis-role order, with axes of role < direction already transformed
    // (extent n_columns) and axes of role > direction not yet transformed
    // (extent n_rows) -- i.e. this call is meant to be issued in increasing
    // `direction` order (0, 1, ..., dim-1) for interpolation and decreasing
    // order (dim-1, ..., 1, 0) for integration, matching a standard
    // sum-factorization sweep. `in_offset`/`out_offset` must address
    // distinct, non-overlapping regions of `scratch` (no in-place support);
    // the caller manages ping-ponging between (at least) two regions across
    // calls. Ends with a team_barrier(), so the result is visible to
    // whatever the caller issues next -- no barrier is needed in between
    // calls to this function.
    template <int  dim,
              int  direction,
              int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const TeamHandle &team_member,
                                const Number     *matrix,
                                Number           *scratch,
                                const int         in_offset,
                                const int         out_offset,
                                const int         c_nelmtPerBatch,
                                const int         threadIdx,
                                const int         blockSize)
    {
      static_assert(direction >= 0 && direction < dim, "direction must be in [0, dim)");

      // mm: extent of the contracted axis in `in`; nn: extent in `out`.
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      // n_blocks1: combined extent of the already-transformed axes (role <
      // direction, extent n_columns each); n_blocks2: combined extent of
      // the not-yet-transformed axes (role > direction, extent n_rows
      // each). Their product is the per-cell thread count for this call,
      // and (since axes of role < direction are the fastest-varying, per
      // the layout convention above) also the stride between consecutive
      // entries of the fiber each thread owns.
      constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
      constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

      constexpr int n_in_per_elmt  = n_blocks1 * mm * n_blocks2;
      constexpr int n_out_per_elmt = n_blocks1 * nn * n_blocks2;

      const Number *in  = scratch + in_offset;
      Number       *out = scratch + out_offset;

      for (int tid = threadIdx; tid < c_nelmtPerBatch * n_blocks1 * n_blocks2; tid += blockSize)
        {
          const int e   = tid / (n_blocks1 * n_blocks2);
          const int rem = tid % (n_blocks1 * n_blocks2);
          const int i2  = rem / n_blocks1;
          const int i1  = rem % n_blocks1;

          const Number *in_e  = in + e * n_in_per_elmt + i2 * n_blocks1 * mm + i1;
          Number       *out_e = out + e * n_out_per_elmt + i2 * n_blocks1 * nn + i1;

          apply_matrix_vector_product<n_rows, n_columns, contract_over_rows, add>(
            matrix, in_e, out_e, n_blocks1, n_blocks1);
        }

      team_member.team_barrier();
    }



    // Linear index of the (d1, d2) entry (0 <= d1, d2 < dim) of a symmetric
    // dim x dim tensor stored as its dim*(dim+1)/2 upper-triangular
    // components in row-major (d1, d2 >= d1) order -- e.g. dim == 3 gives
    // the enumeration (Grr, Grs, Grt, Gss, Gst, Gtt) BK3's own G-tensor
    // construction (compute_G_tensors()) already uses; dim == 2 gives
    // (Grr, Grs, Gss). Shared by evaluate_and_multiply_tensor() below and
    // by whatever builds `d_G` in the first place.
    template <int dim>
    DEAL_II_HOST_DEVICE inline int
    symmetric_tensor_component_index(const int d1, const int d2)
    {
      const int a = (d1 < d2) ? d1 : d2;
      const int b = (d1 < d2) ? d2 : d1;
      return a * dim - (a * (a - 1)) / 2 + (b - a);
    }



    // evaluate()/evaluate_and_multiply_tensor()/integrate() below all share
    // the same loop shape. `tid % co_dimension_size` fixes one point in the
    // (dim - 1)-dimensional "co-dim plane" spanned by every axis but the
    // last; the thread then loops in registers over the last axis (extent
    // nq), so `point = rem + last * co_dimension_size` is the flat nq^dim
    // index of the current point (co_dimension_size == nq^(dim - 1), the
    // stride of the last axis).
    //
    // Per direction d, the contraction needed is:
    //   q[d](point) = sum_n co_shape_gradients[n * nq + idx_d] * u_in(point with axis d := n)
    // where idx_d is d's own position within `point`. Both quantities are
    // read directly off `point` with plain fixed-radix arithmetic
    // (Utilities::pow(nq, d) is axis d's stride): idx_d = (point /
    // Utilities::pow(nq, d)) % nq, and "point with axis d zeroed" is point -
    // idx_d * Utilities::pow(nq, d) -- so "point with axis d := n" is that
    // plus n * Utilities::pow(nq, d). This is the same formula for every d,
    // including d == dim - 1 (the register-loop axis itself, where idx_d ==
    // last by construction), so no special-casing by direction is needed
    // and the same body serves any dim. Utilities::pow(nq, d) is
    // recomputed at each use rather than cached in a per-thread array (d
    // is a small runtime loop variable, not a compile-time exponent, so
    // this isn't a constexpr computation) -- exponentiation by squaring on
    // d <= 2 is at most one multiply, cheaper than the register pressure
    // (and GPU spill risk) of holding a whole array live across the loops
    // below just to save that.



    // Evaluates the dim directional derivatives of values at
    // scratch[values_offset...] (nq^dim-per-cell, i.e. after the dof->quad
    // interpolation sweep has completed) via `co_shape_gradients` (an nq x
    // nq collocation differentiation matrix), with no further processing --
    // e.g. for an operator whose geometric-factor step isn't BK3's
    // isotropic-Laplace symmetric tensor. See evaluate_and_multiply_tensor()
    // below for the fused, BK3-specific counterpart (not implemented in
    // terms of this function, to keep the tensor multiply register-fused).
    //
    // `gradients_offset` addresses the base of a caller-owned region
    // holding dim consecutive nq^dim-per-cell slots (stride
    // `gradient_slot_stride` elements), one per direction. Ends with its
    // own team_barrier().
    template <int dim, int nq, typename Number>
    DEAL_II_HOST_DEVICE inline void
    evaluate(const TeamHandle &team_member,
            const Number     *s_co_shape_gradients,
            Number           *scratch,
            const int         values_offset,
            const int         gradients_offset,
            const int         gradient_slot_stride,
            const int         c_nelmtPerBatch,
            const int         threadIdx,
            const int         blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nq_total          = Utilities::pow(nq, dim);
      constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

      const Number *u_in = scratch + values_offset;

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          for (int last = 0; last < nq; ++last)
            {
              const int point = rem + last * co_dimension_size;

              for (int d = 0; d < dim; ++d)
                {
                  const int stride_d   = Utilities::pow(nq, d);
                  const int idx_d      = (point / stride_d) % nq;
                  const int point_base = point - idx_d * stride_d;

                  Number q_d = 0;
                  for (int n = 0; n < nq; ++n)
                    q_d += s_co_shape_gradients[n * nq + idx_d] *
                          u_in[e * nq_total + point_base + n * stride_d];

                  scratch[gradients_offset + d * gradient_slot_stride + e * nq_total + point] = q_d;
                }
            }
        }

      team_member.team_barrier();
    }



    // Same fan-out as evaluate() above, but fused with an immediate
    // pointwise multiply of the resulting dim-vector by the BK3 symmetric
    // geometric factor tensor `d_G` (chain rule for the reference-to-
    // physical mapping) before writing to the gradients pool -- this is
    // BK3::Parallel::KokkosKernel's step 5. Computing the dim directional
    // derivatives and applying the tensor together, all in registers per
    // thread, means one pass (one team_barrier() at the end) suffices,
    // rather than the barrier a plain evaluate() call followed by a
    // separate tensor-multiply pass would need in between; the tensor
    // multiply is hardcoded to the isotropic-Laplace symmetric tensor (not
    // a functor) for now. `G` is loaded into a small dim x dim register
    // array once per point (both triangular halves, via
    // symmetric_tensor_component_index()) rather than looked up per (d1,
    // d2) pair, so the multiply itself is a plain dense symmetric
    // matrix-vector product. Pair with integrate() below for the matching
    // fan-in half.
    //
    // `g_offset` is a raw element offset into `d_G`, already resolved by
    // the caller to point at cell e == 0 of this call's batch (each
    // thread's own cell e then adds e * (dim*(dim+1)/2) * nq^dim on top) --
    // deliberately not derived here from a batch/league index or a
    // cell_range_ids lookup, so this function doesn't need to know
    // BK3::Parallel::KokkosKernel's cell-batching conventions. Note this
    // means if a caller ever needs the cell_range_ids indirection
    // BK3::Parallel::KokkosKernel's steps 1/10 use for dof_indices, `d_G`
    // itself must already be laid out in that permuted order -- this
    // function cannot apply the permutation itself.
    template <int dim, int nq, typename Number>
    DEAL_II_HOST_DEVICE inline void
    evaluate_and_multiply_tensor(const TeamHandle         &team_member,
                                 const Number             *s_co_shape_gradients,
                                 const DeviceView<Number> &d_G,
                                 const int                 g_offset,
                                 Number                    *scratch,
                                 const int                 values_offset,
                                 const int                 gradients_offset,
                                 const int                 gradient_slot_stride,
                                 const int                 c_nelmtPerBatch,
                                 const int                 threadIdx,
                                 const int                 blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nq_total                   = Utilities::pow(nq, dim);
      constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
      constexpr int co_dimension_size          = Utilities::pow(nq, dim - 1);

      const Number *u_in = scratch + values_offset;

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          const int cell_g_offset = g_offset + e * symmetric_tensor_dimension * nq_total;

          for (int last = 0; last < nq; ++last)
            {
              const int point = rem + last * co_dimension_size;

              Number q[dim];
              for (int d = 0; d < dim; ++d)
                {
                  const int stride_d   = Utilities::pow(nq, d);
                  const int idx_d      = (point / stride_d) % nq;
                  const int point_base = point - idx_d * stride_d;

                  Number q_d = 0;
                  for (int n = 0; n < nq; ++n)
                    q_d += s_co_shape_gradients[n * nq + idx_d] *
                          u_in[e * nq_total + point_base + n * stride_d];
                  q[d] = q_d;
                }

              Number G[dim][dim];
              for (int d1 = 0; d1 < dim; ++d1)
                for (int d2 = d1; d2 < dim; ++d2)
                  {
                    const Number value =
                      d_G[cell_g_offset + symmetric_tensor_component_index<dim>(d1, d2) * nq_total +
                          point];
                    G[d1][d2] = value;
                    G[d2][d1] = value;
                  }

              for (int d1 = 0; d1 < dim; ++d1)
                {
                  Number out = 0;
                  for (int d2 = 0; d2 < dim; ++d2)
                    out += G[d1][d2] * q[d2];

                  scratch[gradients_offset + d1 * gradient_slot_stride + e * nq_total + point] = out;
                }
            }
        }

      team_member.team_barrier();
    }



    // Integrates the dim directional-derivative slots left by
    // evaluate() or evaluate_and_multiply_tensor() above back into a single
    // result at scratch[values_offset...] -- BK3::Parallel::KokkosKernel's
    // step 6. Each output entry sums contributions from all dim input
    // slots (the transpose-contracted mirror of evaluate()'s fan-out: the
    // co_shape_gradients row is read as [idx_d * nq + n] here rather than
    // [n * nq + idx_d]), so this is one pass, one team_barrier() at the end.
    //
    // `gradients_offset`/`gradient_slot_stride` address the same region
    // evaluate()/evaluate_and_multiply_tensor() wrote; `values_offset` may
    // address the same region they read their input from -- nothing here
    // reads scratch[values_offset...], so overwriting it in place is safe
    // without needing a barrier first.
    template <int dim, int nq, typename Number>
    DEAL_II_HOST_DEVICE inline void
    integrate(const TeamHandle &team_member,
             const Number     *s_co_shape_gradients,
             Number           *scratch,
             const int         gradients_offset,
             const int         gradient_slot_stride,
             const int         values_offset,
             const int         c_nelmtPerBatch,
             const int         threadIdx,
             const int         blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nq_total          = Utilities::pow(nq, dim);
      constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

      Number *u_out = scratch + values_offset;

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          for (int last = 0; last < nq; ++last)
            {
              const int point = rem + last * co_dimension_size;

              Number tmp0 = 0;
              for (int d = 0; d < dim; ++d)
                {
                  const int stride_d   = Utilities::pow(nq, d);
                  const int idx_d      = (point / stride_d) % nq;
                  const int point_base = point - idx_d * stride_d;

                  Number sum = 0;
                  for (int n = 0; n < nq; ++n)
                    sum += scratch[gradients_offset + d * gradient_slot_stride + e * nq_total +
                                   point_base + n * stride_d] *
                          s_co_shape_gradients[idx_d * nq + n];
                  tmp0 += sum;
                }

              u_out[e * nq_total + point] = tmp0;
            }
        }

      team_member.team_barrier();
    }
  } // namespace Parallel
} // namespace Common

DEAL_II_NAMESPACE_CLOSE

#endif
