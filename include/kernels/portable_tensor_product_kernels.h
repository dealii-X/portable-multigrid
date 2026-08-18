#ifndef kernels_portable_tensor_product_kernels_h
#define kernels_portable_tensor_product_kernels_h

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
//
// Filename/namespace deliberately mirror deal.II's own
// matrix_free/portable_tensor_product_kernels.h (Portable::internal) -- the
// long-term intent is upstreaming this generalization (batched, rectangular
// matrices, offset-addressed scratch) into deal.II itself, so this stays
// named/structured close to where it would eventually land. `Custom` is a
// placeholder namespace for as long as this lives outside deal.II proper.
//
// This file holds only the low-level tensor-product primitives
// (apply_matrix_vector_product(), EvaluatorTensorProduct); the higher-level,
// per-cell evaluate()/integrate()-style orchestration built on top of them
// lives in kernels/portable_evaluation_kernels.h instead, mirroring
// deal.II's own split between matrix_free/tensor_product_kernels.h and
// matrix_free/evaluation_kernels.h.
namespace Custom
{
  namespace Parallel
  {
    using TeamHandle = Kokkos::TeamPolicy<>::member_type;

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    // Per-cell local-to-global dof index table (local_idx, global_cell_index)
    // -> global dof index, or numbers::invalid_unsigned_int for a
    // constrained/absent dof -- shared shape used by read_dof_values()/
    // distribute_local_to_global() in kernels/portable_evaluation_kernels.h.
    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

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



    // Same single-fiber contraction as the overload above, but for callers
    // whose stride_in/stride_out happen to be compile-time constants (e.g.
    // the batched overload below, whose n_blocks1 already is one) --
    // template, rather than runtime, parameters let the compiler constant-
    // fold the fiber addressing instead of carrying stride_in/stride_out as
    // live runtime values. Mirrors deal.II's own CPU
    // internal::EvaluatorTensorProduct::apply()'s compile-time `stride`
    // template parameter (tensor_product_kernels.h) for the same reason.
    template <int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              int  stride_in,
              int  stride_out,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const Number *matrix, const Number *in, Number *out)
    {
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

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
    // c_nelmtPerBatch cells in the batch, contracts `in` along logical
    // tensor direction `direction` against `matrix` (n_rows x n_columns,
    // row-major), writing the result to `out`. Composed from the
    // single-fiber overload above -- this just works out which fiber (base
    // pointer + stride) each thread owns and loops over all of them.
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
    // sum-factorization sweep. `in` and `out` must be distinct,
    // non-overlapping buffers (no in-place support, matching deal.II's own
    // `apply()`'s in_place == false path -- see EvaluatorTensorProduct
    // below for the in_place == true path deal.II additionally supports via
    // a `temp` buffer, not needed by any caller here yet); the caller
    // manages ping-ponging between (at least) two buffers across calls.
    // Ends with a team_barrier(), so the result is visible to whatever the
    // caller issues next -- no barrier is needed in between calls to this
    // function.
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
                                const Number     *in,
                                Number           *out,
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

      for (int tid = threadIdx; tid < c_nelmtPerBatch * n_blocks1 * n_blocks2; tid += blockSize)
        {
          const int e   = tid / (n_blocks1 * n_blocks2);
          const int rem = tid % (n_blocks1 * n_blocks2);
          const int i2  = rem / n_blocks1;
          const int i1  = rem % n_blocks1;

          const Number *in_e  = in + e * n_in_per_elmt + i2 * n_blocks1 * mm + i1;
          Number       *out_e = out + e * n_out_per_elmt + i2 * n_blocks1 * nn + i1;

          apply_matrix_vector_product<n_rows, n_columns, contract_over_rows, add, n_blocks1, n_blocks1>(
            matrix, in_e, out_e);
        }

      team_member.team_barrier();
    }



    // Class wrapper around the batched dim/direction apply_matrix_vector_product()
    // above, in the same shape as deal.II's own
    // Portable::internal::EvaluatorTensorProduct
    // (matrix_free/portable_tensor_product_kernels.h): construct once per
    // team/matrix/batch, then issue one call per sum-factorization
    // direction via a templated `values<direction, dof_to_quad, add>(in,
    // out)` member function, `direction` itself a template (not runtime)
    // parameter -- mirroring deal.II's EvaluatorTensorProduct::values(),
    // whose signature this deliberately echoes (`dof_to_quad` there plays
    // the same role `contract_over_rows` does here, and `in`/`out` are
    // taken directly as the caller's buffers, exactly as in deal.II's
    // values(const ViewTypeIn in, ViewTypeOut out)).
    //
    // `temp` mirrors the constructor slot deal.II's EvaluatorTensorProduct
    // always takes (a SharedView, used only by its in_place == true path to
    // stage a result before copying it into `out` -- see
    // Portable::internal::populate_view()/apply()'s in_place branch). Every
    // caller here ping-pongs between two distinct buffers instead, i.e.
    // always takes deal.II's in_place == false path, so `temp` is stored
    // but never read by values() -- callers with no in-place use (all of
    // them, currently) pass nullptr for it, same as deal.II callers pass an
    // empty/unused View when they don't need in_place. Keeping the
    // parameter (rather than dropping it, as an earlier version of this
    // class did) keeps the constructor's shape aligned with deal.II's own,
    // in case an in_place values()/gradients() overload is added here
    // later.
    //
    // Named `values`, not `apply`, to match deal.II's convention for the
    // "shape_values"-matrix contraction -- deal.II's class also holds
    // shape_gradients/co_shape_gradients and exposes
    // `gradients()`/`co_gradients()` alongside `values()`; nothing in this
    // codebase needs those yet (BK3's own collocated-gradient step is
    // evaluate_and_multiply_tensor()/integrate() in
    // kernels/portable_evaluation_kernels.h, a different fused operation,
    // not a matrix-vector product against a second matrix), so only
    // `values()` is provided so far -- add the others here if/when a
    // caller needs them, rather than growing a second class.
    //
    // Takes `matrix` and the batching parameters (c_nelmtPerBatch,
    // threadIdx, blockSize) once at construction, exactly as deal.II's
    // EvaluatorTensorProduct takes its Views and TeamHandle once; only
    // `in`/`out` vary per values() call, exactly as in deal.II.
    template <int dim, int n_rows, int n_columns, typename Number>
    class EvaluatorTensorProduct
    {
    public:
      DEAL_II_HOST_DEVICE
      EvaluatorTensorProduct(const TeamHandle &team_member,
                             const Number     *matrix,
                             Number           *temp,
                             const int         c_nelmtPerBatch,
                             const int         threadIdx,
                             const int         blockSize)
        : team_member(team_member)
        , matrix(matrix)
        , temp(temp)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
      {}

      template <int direction, bool dof_to_quad, bool add>
      DEAL_II_HOST_DEVICE void
      values(const Number *in, Number *out) const
      {
        apply_matrix_vector_product<dim, direction, n_rows, n_columns, dof_to_quad, add>(
          team_member, matrix, in, out, c_nelmtPerBatch, threadIdx, blockSize);
      }

    private:
      const TeamHandle &team_member;
      const Number     *matrix;
      Number           *temp;
      const int         c_nelmtPerBatch;
      const int         threadIdx;
      const int         blockSize;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
