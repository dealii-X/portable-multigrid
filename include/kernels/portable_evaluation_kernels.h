#ifndef kernels_portable_evaluation_kernels_h
#define kernels_portable_evaluation_kernels_h

#include "kernels/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

// Groups the direction-by-direction Custom::Parallel::EvaluatorTensorProduct
// ::values() calls (kernels/portable_tensor_product_kernels.h) into per-cell
// evaluate()/integrate() orchestration, the same organizing role deal.II's
// own matrix_free/portable_evaluation_kernels.h plays for its
// Portable::internal::EvaluatorTensorProduct -- and, structurally, the same
// struct deal.II's file uses: BK3 is exactly deal.II's "collocation" case
// (fe_degree + 1 dof points, n_q_points_1d quadrature points, transformed
// via `shape_values` to a space where nodes and quadrature points
// coincide), so FEEvaluationImplTransformToCollocation is the deal.II struct
// this one is the counterpart of. `evaluate()`/`integrate()` are static
// member functions issuing one `evaluator.template values<direction, ...>`
// call per direction, explicitly unrolled per `dim` via `if constexpr`
// (dim == 1/2/3) rather than deal.II's in-place-buffer recursion or this
// file's own earlier compile-time-recursive version -- SYCL (one of the
// Kokkos backends this is meant to stay portable to) doesn't support
// recursive device functions, so the per-dim unroll, not recursion, is the
// form to keep here.
//
// Unlike deal.II's FEEvaluationImplTransformToCollocation, this only covers
// the `values<direction, ...>` transform-to/from-collocation sweep (BK3's
// steps 2-4/7-9); it doesn't loop over n_components, dispatch on
// EvaluationFlags, or apply the collocated-gradient/co_gradients step
// itself (BK3's steps 5-6, a fused geometric-tensor multiply specific to
// the isotropic-Laplace operator -- see evaluate_and_multiply_tensor()/
// integrate() in kernels/portable_tensor_product_kernels.h) -- those stay
// the caller's responsibility, same as they're already split out today.
//
// Just as deal.II's FEEvaluationImplTransformToCollocation::evaluate()/
// integrate() each build their own local `eval` (an EvaluatorTensorProduct)
// from `data->...` rather than taking one as a parameter, evaluate()/
// integrate() below construct their own EvaluatorTensorProduct from
// `team_member`/`matrix`/the batching parameters, rather than the caller
// constructing one and passing it in.
//
// `in`/`out` are taken as separate, explicit arguments (matching deal.II's
// own values(in, out) naming) rather than a pair of interchangeable
// ping-pong buffers with the result location reported back via a return
// value -- the result always lands in `out`, mirroring
// kernels/bk3_kokkos_kernels_custom.h's own routing, where steps 2-4/7-9
// always start and end at the same named buffer (s_values) regardless of
// dim. A single extra `scratch` buffer supplies whatever intermediate
// storage the sweep needs along the way -- one slot's worth for dim <= 2 (no
// intermediate needed at all for dim == 1), and, for dim == 3, `scratch` and
// `scratch + slot` (its second slot) -- `slot` is the caller-owned stride
// between those two, since it can't be derived from n_rows/n_columns alone
// (it depends on the caller's batch size, e.g. BK3's nelmtPerBatch *
// nq_total). `in`/`out`/`scratch` need not be three distinct buffers -- BK3
// passes the same pointer for `in` and `out` (its single "values" slot) and
// a separate `scratch` (its "gradients" pool's first two slots).
namespace Custom
{
  namespace Parallel
  {
    template <int dim, int n_rows, int n_columns, typename Number>
    struct FEEvaluationImplTransformToCollocation
    {
      // Transforms dof values to quadrature-point ("collocation") values,
      // one direction at a time in increasing order (BK3's steps 2-4).
      DEAL_II_HOST_DEVICE static void
      evaluate(const TeamHandle &team_member,
              const Number     *matrix,
              Number           *in,
              Number           *out,
              Number           *scratch,
              const int         slot,
              const int         c_nelmtPerBatch,
              const int         threadIdx,
              const int         blockSize)
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
            evaluator.template values<1, true, false>(scratch, scratch + slot);
            evaluator.template values<2, true, false>(scratch + slot, out);
          }
      }



      // Transforms quadrature-point values back to dof values, one
      // direction at a time in decreasing order (BK3's steps 7-9).
      DEAL_II_HOST_DEVICE static void
      integrate(const TeamHandle &team_member,
               const Number     *matrix,
               Number           *in,
               Number           *out,
               Number           *scratch,
               const int         slot,
               const int         c_nelmtPerBatch,
               const int         threadIdx,
               const int         blockSize)
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
            evaluator.template values<1, false, false>(scratch, scratch + slot);
            evaluator.template values<0, false, false>(scratch + slot, out);
          }
      }
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
