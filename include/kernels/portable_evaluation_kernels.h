#ifndef kernels_portable_evaluation_kernels_h
#define kernels_portable_evaluation_kernels_h

#include "kernels/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

// Higher-level, per-cell "evaluation" building blocks -- read_dof_values(),
// distribute_local_to_global(), and the FEEvaluationImplTransformToCollocation
// class below (evaluate_values()/integrate_values()/evaluate_gradients()/
// evaluate_gradients_and_multiply_tensor()/integrate_gradients()) -- built on
// top of the low-level tensor-product primitives in
// kernels/portable_tensor_product_kernels.h (apply_matrix_vector_product(),
// EvaluatorTensorProduct). Mirrors deal.II's own split between
// matrix_free/tensor_product_kernels.h and matrix_free/evaluation_kernels.h,
// and the naming of deal.II's FEEvaluation::read_dof_values()/
// distribute_local_to_global(); the values/gradients naming split below
// mirrors how deal.II's own FEEvaluation distinguishes "values" (the
// dof<->quad shape_values transform) from "gradients" (a derivative
// quantity) as separate named operations, rather than overloading a single
// evaluate()/integrate() name for both. Named _values()/_gradients() below
// being BK3::Parallel::KokkosKernel's steps 2-4/7-9 and 5-6 respectively;
// read_dof_values()/distribute_local_to_global() are its steps 1/10.
namespace Custom
{
  namespace Parallel
  {
    // Copies BK3's steps 1/10 gather/scatter loop shape: `co_dimension_size
    // = nm^(dim-1)` fixes one point in the (dim-1)-dimensional co-dim plane
    // spanned by every axis but the last (`rem`), and the thread loops over
    // the last axis in registers (`last`), so `local_idx = rem + last *
    // co_dimension_size` is the flat nm^dim local dof index -- the same
    // "rem + last * co_dimension_size" idiom
    // FEEvaluationImplTransformToCollocation's evaluate_gradients()/
    // evaluate_gradients_and_multiply_tensor()/integrate_gradients() below
    // share. `eb`/`nelmtPerBatch`/`cell_range_ids` resolve each thread's own
    // `global_cell_index`, remapped through `cell_range_ids` when non-empty.

    // Reads dof values for this batch out of the global vector `d_in`
    // (via `dof_indices`) into `values` -- BK3::Parallel::KokkosKernel's
    // step 1, and the counterpart of deal.II's own
    // FEEvaluation::read_dof_values(). A missing dof (numbers::
    // invalid_unsigned_int) reads as 0. Ends with its own team_barrier().
    template <int dim, int nm, typename Number>
    DEAL_II_HOST_DEVICE inline void
    read_dof_values(const TeamHandle         &team_member,
                    const DeviceView<Number> &d_in,
                    const DoFIndicesView     &dof_indices,
                    Number                    *values,
                    const int                 eb,
                    const int                 nelmtPerBatch,
                    const CellRangeIdView    &cell_range_ids,
                    const int                 c_nelmtPerBatch,
                    const int                 threadIdx,
                    const int                 blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nm_total          = Utilities::pow(nm, dim);
      constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          unsigned int global_cell_index = eb * nelmtPerBatch + e;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          for (int last = 0; last < nm; ++last)
            {
              const int local_idx = rem + last * co_dimension_size;

              const unsigned int dof_index = dof_indices(local_idx, global_cell_index);
              const int          shared_idx = e * nm_total + local_idx;

              if (dof_index == numbers::invalid_unsigned_int)
                values[shared_idx] = 0;
              else
                values[shared_idx] = d_in[dof_index];
            }
        }

      team_member.team_barrier();
    }



    // Scatter-adds `values` for this batch into the global vector `d_out`
    // (via `dof_indices`) -- BK3::Parallel::KokkosKernel's step 10, and the
    // counterpart of deal.II's own
    // FEEvaluation::distribute_local_to_global(). Uses Kokkos::atomic_add
    // since dofs are shared between elements. Ends with its own
    // team_barrier().
    template <int dim, int nm, typename Number>
    DEAL_II_HOST_DEVICE inline void
    distribute_local_to_global(const TeamHandle       &team_member,
                               const Number           *values,
                               DeviceView<Number>      d_out,
                               const DoFIndicesView   &dof_indices,
                               const int               eb,
                               const int               nelmtPerBatch,
                               const CellRangeIdView  &cell_range_ids,
                               const int               c_nelmtPerBatch,
                               const int               threadIdx,
                               const int               blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nm_total          = Utilities::pow(nm, dim);
      constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          unsigned int global_cell_index = eb * nelmtPerBatch + e;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          for (int last = 0; last < nm; ++last)
            {
              const int local_idx = rem + last * co_dimension_size;

              const unsigned int dof_index  = dof_indices(local_idx, global_cell_index);
              const int          shared_idx = e * nm_total + local_idx;

              if (dof_index != numbers::invalid_unsigned_int)
                Kokkos::atomic_add(&d_out[dof_index], values[shared_idx]);
            }
        }

      team_member.team_barrier();
    }



    // Groups the direction-by-direction Custom::Parallel::EvaluatorTensorProduct
    // ::values() calls (kernels/portable_tensor_product_kernels.h) into
    // per-cell evaluate_values()/integrate_values() orchestration, plus BK3's
    // own collocated-gradient step (evaluate_gradients()/
    // evaluate_gradients_and_multiply_tensor()/integrate_gradients()) --
    // grouped into one class, mirroring the organizing role deal.II's own
    // matrix_free/portable_evaluation_kernels.h plays for its
    // Portable::internal::EvaluatorTensorProduct: BK3 is exactly deal.II's
    // "collocation" case (fe_degree + 1 dof points, n_q_points_1d quadrature
    // points, transformed via `shape_values` to a space where nodes and
    // quadrature points coincide), so FEEvaluationImplTransformToCollocation
    // is the deal.II struct this one is the counterpart of.
    //
    // Constructed once per team/batch, storing `matrix` (shape_values, used
    // by evaluate_values()/integrate_values()) and
    // `shape_gradient_collocation` (co_shape_gradients, used by
    // evaluate_gradients()/evaluate_gradients_and_multiply_tensor()/
    // integrate_gradients()) alongside the batching parameters -- mirroring
    // deal.II's own EvaluatorTensorProduct (kernels/
    // portable_tensor_product_kernels.h), which likewise takes its shape
    // matrices and TeamHandle once at construction rather than per call.
    // Unlike an earlier version of this class, which had
    // evaluate_values()/integrate_values() build their own local
    // EvaluatorTensorProduct from parameters passed to each call, `matrix`
    // is now held as a member instead, since it's shared by (originally)
    // both methods and (now) potentially any method here that needs the
    // shape_values matrix.
    //
    // `evaluate_values()`/`integrate_values()` issue one
    // `evaluator.template values<direction, ...>` call per direction,
    // explicitly unrolled per `dim` via `if constexpr` (dim == 1/2/3) rather
    // than deal.II's in-place-buffer recursion or this file's own earlier
    // compile-time-recursive version -- SYCL (one of the Kokkos backends
    // this is meant to stay portable to) doesn't support recursive device
    // functions, so the per-dim unroll, not recursion, is the form to keep
    // here. Named `_values()`, matching deal.II's own values(in, out)
    // naming for the shape_values-matrix transform, and to distinguish this
    // pair from evaluate_gradients()/integrate_gradients() below, which
    // operate on the (differently purposed) collocated-gradient quantity
    // instead. `in`/`out` are taken as separate, explicit arguments
    // (matching deal.II's own values(in, out) naming) rather than a pair of
    // interchangeable ping-pong buffers with the result location reported
    // back via a return value -- the result always lands in `out`,
    // mirroring kernels/bk3_kokkos_kernels_custom.h's own routing, where
    // steps 2-4/7-9 always start and end at the same named buffer (s_values)
    // regardless of dim. A single extra `scratch` buffer supplies whatever
    // intermediate storage the sweep needs along the way -- one slot's
    // worth for dim <= 2 (no intermediate needed at all for dim == 1), and,
    // for dim == 3, `scratch` and `scratch + quad_size_per_batch` (its
    // second slot) -- `quad_size_per_batch` is the caller-owned stride
    // between those two, since it can't be derived from n_rows/n_columns
    // alone (it depends on the caller's batch size, e.g. BK3's
    // nelmtPerBatch * nq_total). `in`/`out`/`scratch` need not be three
    // distinct buffers -- BK3 passes the same pointer for `in` and `out`
    // (its single "values" slot) and a separate `scratch` (its "gradients"
    // pool's first two slots).
    //
    // `evaluate_gradients()`/`evaluate_gradients_and_multiply_tensor()`/
    // `integrate_gradients()` all share the same loop shape. `tid %
    // co_dimension_size` fixes one point in the (dim - 1)-dimensional
    // "co-dim plane" spanned by every axis but the last; the thread then
    // loops in registers over the last axis (extent n_columns), so `point =
    // rem + last * co_dimension_size` is the flat n_columns^dim index of the
    // current point (co_dimension_size == n_columns^(dim - 1), the stride of
    // the last axis). Per direction d, the contraction needed is:
    //   q[d](point) = sum_n shape_gradient_collocation[n * n_columns + idx_d] * u_in(point with axis d := n)
    // where idx_d is d's own position within `point`. Both quantities are
    // read directly off `point` with plain fixed-radix arithmetic
    // (Utilities::pow(n_columns, d) is axis d's stride): idx_d = (point /
    // Utilities::pow(n_columns, d)) % n_columns, and "point with axis d
    // zeroed" is point - idx_d * Utilities::pow(n_columns, d) -- so "point
    // with axis d := n" is that plus n * Utilities::pow(n_columns, d). This
    // is the same formula for every d, including d == dim - 1 (the
    // register-loop axis itself, where idx_d == last by construction), so
    // no special-casing by direction is needed and the same body serves any
    // dim.
    template <int dim, int n_rows, int n_columns, typename Number>
    class FEEvaluationImplTransformToCollocation
    {
    public:
      DEAL_II_HOST_DEVICE
      FEEvaluationImplTransformToCollocation(const TeamHandle &team_member,
                                             const Number     *matrix,
                                             const Number     *shape_gradient_collocation,
                                             const int         c_nelmtPerBatch,
                                             const int         threadIdx,
                                             const int         blockSize)
        : team_member(team_member)
        , matrix(matrix)
        , shape_gradient_collocation(shape_gradient_collocation)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
      {}



      // Transforms dof values to quadrature-point ("collocation") values,
      // one direction at a time in increasing order (BK3's steps 2-4).
      DEAL_II_HOST_DEVICE void
      evaluate_values(Number *in, Number *out, Number *scratch, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<dim, n_rows, n_columns, Number> evaluator(
          team_member, matrix, nullptr, c_nelmtPerBatch, threadIdx, blockSize);

        if constexpr (dim == 1)
          {
            evaluator.template values<0, true, false>(in, out);
          }
        else if constexpr (dim == 2)
          {
            evaluator.template values<0, true, false>(in, scratch);
            evaluator.template values<1, true, false>(scratch, out);
          }
        else // dim == 3
          {
            evaluator.template values<0, true, false>(in, scratch);
            evaluator.template values<1, true, false>(scratch, scratch + quad_size_per_batch);
            evaluator.template values<2, true, false>(scratch + quad_size_per_batch, out);
          }
      }



      // Transforms quadrature-point values back to dof values, one
      // direction at a time in decreasing order (BK3's steps 7-9).
      DEAL_II_HOST_DEVICE void
      integrate_values(Number *in, Number *out, Number *scratch, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<dim, n_rows, n_columns, Number> evaluator(
          team_member, matrix, nullptr, c_nelmtPerBatch, threadIdx, blockSize);

        if constexpr (dim == 1)
          {
            evaluator.template values<0, false, false>(in, out);
          }
        else if constexpr (dim == 2)
          {
            evaluator.template values<1, false, false>(in, scratch);
            evaluator.template values<0, false, false>(scratch, out);
          }
        else // dim == 3
          {
            evaluator.template values<2, false, false>(in, scratch);
            evaluator.template values<1, false, false>(scratch, scratch + quad_size_per_batch);
            evaluator.template values<0, false, false>(scratch + quad_size_per_batch, out);
          }
      }



      // Evaluates the dim directional derivatives of `values`
      // (n_columns^dim-per-cell, i.e. after evaluate_values() above has run)
      // via `shape_gradient_collocation` (an n_columns x n_columns
      // collocation differentiation matrix), with no further processing --
      // e.g. for an operator whose geometric-factor step isn't BK3's
      // isotropic-Laplace symmetric tensor. See
      // evaluate_gradients_and_multiply_tensor() below for the fused,
      // BK3-specific counterpart (not implemented in terms of this method,
      // to keep the tensor multiply register-fused).
      //
      // `gradients` addresses the base of a caller-owned region holding dim
      // consecutive n_columns^dim-per-cell slots (stride
      // `quad_size_per_batch` elements), one per direction -- may not
      // overlap `values` (this reads `values` throughout, never writing
      // it). Ends with its own team_barrier().
      DEAL_II_HOST_DEVICE void
      evaluate_gradients(const Number *values, Number *gradients, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                for (int d = 0; d < dim; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int idx_d      = (point / stride_d) % n_columns;
                    const int point_base = point - idx_d * stride_d;
                    // Base index for values, invariant across the n-loop
                    // below -- hoisted so it isn't recomputed n_columns
                    // times per direction.
                    const int in_base = e * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += shape_gradient_collocation[n * n_columns + idx_d] *
                             values[in_base + n * stride_d];

                    gradients[d * quad_size_per_batch + e * nq_total + point] = q_d;
                  }
              }
          }

        team_member.team_barrier();
      }



      // Same fan-out as evaluate_gradients() above, but fused with an
      // immediate pointwise multiply of the resulting dim-vector by the BK3
      // symmetric geometric factor tensor `d_G` (chain rule for the
      // reference-to-physical mapping) before writing to the gradients pool
      // -- this is BK3::Parallel::KokkosKernel's step 5. Computing the dim
      // directional derivatives and applying the tensor together, all in
      // registers per thread, means one pass (one team_barrier() at the
      // end) suffices, rather than the barrier a plain evaluate_gradients()
      // call followed by a separate tensor-multiply pass would need in
      // between; the tensor multiply is hardcoded to the isotropic-Laplace
      // symmetric tensor (not a functor) for now. `G` is loaded into a
      // small dim x dim register array once per point (both triangular
      // halves, via a running component-index counter -- see the comment at
      // its use below) rather than looked up per (d1, d2) pair, so the
      // multiply itself is a plain dense symmetric matrix-vector product.
      // Pair with integrate_gradients() below for the matching fan-in half.
      //
      // `eb`/`nelmtPerBatch`/`cell_range_ids` let this method compute each
      // thread's own `global_cell_index` itself -- remapped through
      // `cell_range_ids` when non-empty, exactly as read_dof_values()/
      // distribute_local_to_global() above do -- rather than requiring the
      // caller to pre-resolve a flat offset into `d_G`.
      //
      // Register-caches whichever factor is invariant across the `last`
      // loop below, mirroring BK3's own hand-written r_p/r_q/r_r caching
      // this method replaces: for directions d < dim - 1, idx_d = (rem /
      // stride_d) % n_columns depends only on `rem` (this thread's fixed
      // co-dimension index), not on `last` -- so the
      // shape_gradient_collocation row is cached once, and `values` is read
      // fresh inside the loop (its address does depend on `last` for these
      // directions). For the last direction (d == dim - 1, which coincides
      // with the `last` loop variable itself), it's the reverse: point_base
      // == rem regardless of `last`, so the `values` row is what's
      // cacheable, and the shape_gradient_collocation row (idx_{dim-1} ==
      // last) must be read fresh instead. Both caches are filled via one
      // shared n-loop rather than one loop per direction; the q[]
      // accumulation below intentionally keeps separate per-direction
      // n-loops instead (matching BK3's own qr/qs/qt structure here -- see
      // integrate_gradients() below for the case where BK3's own
      // hand-written code uses separate loops too, not a shared one).
      DEAL_II_HOST_DEVICE void
      evaluate_gradients_and_multiply_tensor(const DeviceView<Number> &d_G,
                                             const int                 eb,
                                             const int                 nelmtPerBatch,
                                             const CellRangeIdView    &cell_range_ids,
                                             const Number             *values,
                                             Number                    *gradients,
                                             const int                 quad_size_per_batch) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total                   = Utilities::pow(n_columns, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            unsigned int global_cell_index = eb * nelmtPerBatch + e;
            if (cell_range_ids.size() > 0)
              global_cell_index = cell_range_ids(global_cell_index);

            const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * nq_total;

            int    idx[dim - 1];
            Number reg[dim][n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (rem / stride_d) % n_columns;
              }

            for (int n = 0; n < n_columns; ++n)
              {
                for (int d = 0; d < dim - 1; ++d)
                  reg[d][n] = shape_gradient_collocation[n * n_columns + idx[d]];
                reg[dim - 1][n] = values[e * nq_total + rem + n * co_dimension_size];
              }

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                Number q[dim];
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    const int in_base    = e * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += reg[d][n] * values[in_base + n * stride_d];
                    q[d] = q_d;
                  }
                {
                  Number q_d = 0;
                  for (int n = 0; n < n_columns; ++n)
                    q_d += shape_gradient_collocation[n * n_columns + last] * reg[dim - 1][n];
                  q[dim - 1] = q_d;
                }

                // Mirrors LaplaceOperatorBK3::compute_G_tensors()'s own
                // indexing exactly: same (d1, d2) loop nest, same running
                // `component_index` counter (there named `idx`) -- correct
                // by construction, since it enumerates the packed
                // upper-triangular components in the same order they were
                // written in.
                Number G[dim][dim];
                int    component_index = 0;
                for (int d1 = 0; d1 < dim; ++d1)
                  for (int d2 = d1; d2 < dim; ++d2)
                    {
                      const Number value = d_G[cell_g_offset + component_index * nq_total + point];
                      G[d1][d2]          = value;
                      G[d2][d1]          = value;
                      ++component_index;
                    }

                for (int d1 = 0; d1 < dim; ++d1)
                  {
                    Number out = 0;
                    for (int d2 = 0; d2 < dim; ++d2)
                      out += G[d1][d2] * q[d2];

                    gradients[d1 * quad_size_per_batch + e * nq_total + point] = out;
                  }
              }
          }

        team_member.team_barrier();
      }



      // Integrates the dim directional-derivative slots left by
      // evaluate_gradients() or evaluate_gradients_and_multiply_tensor()
      // above back into a single result at `values` --
      // BK3::Parallel::KokkosKernel's step 6. Each output entry sums
      // contributions from all dim input slots (the transpose-contracted
      // mirror of evaluate_gradients()'s fan-out: the
      // shape_gradient_collocation row is read as [idx_d * n_columns + n]
      // here rather than [n * n_columns + idx_d]), so this is one pass, one
      // team_barrier() at the end.
      //
      // `gradients`/`quad_size_per_batch` address the same region
      // evaluate_gradients()/evaluate_gradients_and_multiply_tensor() wrote;
      // `values` may be the same buffer they read their input from --
      // nothing here reads `values`, so overwriting it in place is safe
      // without needing a barrier first.
      //
      // Register-caches whichever factor is invariant across the `last`
      // loop below, mirror image of
      // evaluate_gradients_and_multiply_tensor()'s own caching (see its doc
      // comment): the shape_gradient_collocation row here is transposed,
      // [idx_d * n_columns + n] rather than [n * n_columns + idx_d],
      // matching this method's own transpose-contracted relationship to
      // evaluate_gradients(). Unlike evaluate_gradients_and_multiply_tensor(),
      // BK3's own hand-written code uses separate n-loops per direction here
      // (not one shared loop), so the accumulation into `tmp0` below does
      // too -- directly, not via a per-direction partial sum added
      // afterwards, to preserve bit-identical output (floating-point
      // addition isn't associative).
      DEAL_II_HOST_DEVICE void
      integrate_gradients(const Number *gradients,
                         const int     quad_size_per_batch,
                         Number       *values) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            int    idx[dim - 1];
            Number r_shape[dim - 1][n_columns];
            Number r_grad[n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (rem / stride_d) % n_columns;
                for (int n = 0; n < n_columns; ++n)
                  r_shape[d][n] = shape_gradient_collocation[idx[d] * n_columns + n];
              }
            for (int n = 0; n < n_columns; ++n)
              r_grad[n] = gradients[(dim - 1) * quad_size_per_batch + e * nq_total + rem +
                                    n * co_dimension_size];

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                Number tmp0 = 0;
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    // Base index into the gradients pool, invariant across
                    // the n-loop below -- hoisted so it isn't recomputed
                    // n_columns times per direction.
                    const int grad_base = d * quad_size_per_batch + e * nq_total + point_base;

                    for (int n = 0; n < n_columns; ++n)
                      tmp0 += gradients[grad_base + n * stride_d] * r_shape[d][n];
                  }
                for (int n = 0; n < n_columns; ++n)
                  tmp0 += r_grad[n] * shape_gradient_collocation[last * n_columns + n];

                values[e * nq_total + point] = tmp0;
              }
          }

        team_member.team_barrier();
      }

    private:
      const TeamHandle &team_member;
      const Number     *matrix;
      const Number     *shape_gradient_collocation;
      const int         c_nelmtPerBatch;
      const int         threadIdx;
      const int         blockSize;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
