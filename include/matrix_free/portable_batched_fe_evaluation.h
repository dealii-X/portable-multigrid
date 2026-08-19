#ifndef kernels_portable_batched_fe_evaluation_h
#define kernels_portable_batched_fe_evaluation_h

#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include "matrix_free/portable_evaluation_kernels.h"

DEAL_II_NAMESPACE_OPEN

// Batched analog of deal.II's own Portable::FEEvaluation
// (matrix_free/portable_fe_evaluation.h), built on the abstractions in
// portable_evaluation_kernels.h/portable_local_laplace_operator_batched.h
// instead of deal.II's own one-cell-per-team internals. Named in
// Custom::Parallel (not dealii::Portable) purely to avoid colliding with
// deal.II's real Portable::FEEvaluation, which lives in the exact same
// (dealii::Portable) namespace and is commonly included in the same
// translation unit -- same reasoning as every other Custom::Parallel type
// in this codebase mirroring a real Portable:: counterpart under a
// different namespace.
//
// Deliberately mirrors deal.II's real read_dof_values()/evaluate()/
// get_value()/get_gradient()/submit_value()/submit_gradient()/integrate()/
// distribute_local_to_global() split so a per-quadrature-point Functor can
// be written exactly like step-64's HelmholtzOperatorQuad -- see
// correctness_tests/check_correctness_batched_fe_evaluation/program.cc for
// a worked Laplace example. The `q_point`/`point` argument these take is a
// *batch-flat* index over [0, c_nelmtPerBatch * n_q_points), not a
// per-cell index in [0, n_q_points) as in deal.II's unbatched class --
// Portable::LaplaceOperatorBatchData::for_each_quad_point() (kernels/
// portable_local_laplace_operator_batched.h) is the counterpart of
// deal.II's own Data::for_each_quad_point() that supplies exactly this
// index, spread across the whole team/batch in one call.
//
// get_gradient()/submit_gradient() apply the reference-to-physical mapping
// on the fly via inv_jacobian/JxW (two separate multiplies, mirroring
// deal.II's own get_gradient()/submit_gradient() and this project's
// unbatched LocalLaplaceOperator's step 5 exactly), not
// LaplaceOperatorBatchData::G_tensor's BK3-style precomputed symmetric
// tensor -- the on-the-fly geometric-factor path deferred when
// LaplaceOperatorBatchData was first designed, now built out for this more
// general, deal.II-faithful accessor API. LocalLaplaceOperatorBatched's own
// fused G_tensor path (kernels/portable_local_laplace_operator_batched.h)
// is untouched and still the one cell_loop_batched() uses.
namespace Custom
{
  namespace Parallel
  {
    template <int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
    class FEEvaluation
    {
      static_assert(n_components_ == 1,
                    "Custom::Parallel::FEEvaluation only supports scalar "
                    "(n_components == 1) problems for now -- the batched scratch "
                    "layout (values/gradients) has no component axis yet.");

    public:
      using value_type    = Number;
      using gradient_type = Tensor<1, dim, Number>;
      using data_type     = Custom::Parallel::BatchData<dim, Number>;

      static constexpr unsigned int n_local_dofs_1d = fe_degree + 1;
      static constexpr unsigned int n_q_points      = Utilities::pow(n_q_points_1d, dim);

      DEAL_II_HOST_DEVICE
      explicit FEEvaluation(const data_type *data)
        : data(data)
      {}

      // Return the global cell index the batch-flat index `point` belongs
      // to -- the batched counterpart of deal.II's get_current_cell_index(),
      // adapted since a batch spans many cells rather than the one cell a
      // real deal.II FEEvaluation is bound to.
      DEAL_II_HOST_DEVICE unsigned int
      get_global_cell_index(const int point) const
      {
        const int    nq_total          = data->quad_size_per_batch / data->nelmtPerBatch;
        const int    e                 = point / nq_total;
        unsigned int global_cell_index = data->batchIdx * data->nelmtPerBatch + e;
        if (data->cell_range_ids.size() > 0)
          global_cell_index = data->cell_range_ids(global_cell_index);
        return global_cell_index;
      }

      DEAL_II_HOST_DEVICE void
      read_dof_values(const Custom::Parallel::DeviceView<Number> &src) const
      {
        Custom::Parallel::read_dof_values<dim, n_local_dofs_1d>(data->team_member,
                                                                src,
                                                                data->dof_indices,
                                                                data->values,
                                                                data->batchIdx,
                                                                data->nelmtPerBatch,
                                                                data->cell_range_ids,
                                                                data->c_nelmtPerBatch,
                                                                data->threadIdx,
                                                                data->blockSize);
      }

      DEAL_II_HOST_DEVICE void
      distribute_local_to_global(Custom::Parallel::DeviceView<Number> &dst) const
      {
        Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(data->team_member,
                                                                           data->values,
                                                                           dst,
                                                                           data->dof_indices,
                                                                           data->batchIdx,
                                                                           data->nelmtPerBatch,
                                                                           data->cell_range_ids,
                                                                           data->c_nelmtPerBatch,
                                                                           data->threadIdx,
                                                                           data->blockSize);
      }

      // Deal.II-style evaluate(): a single EvaluatorTensorProduct instance
      // (`eval`, holding both shape_values and co_shape_gradients) serves
      // both the dof->quad values transform (BK3 steps 2-4, via values<>())
      // and the gradient step (BK3 steps 5-6, via co_gradients<>()) --
      // mirroring real deal.II's own
      // FEEvaluationImplTransformToCollocation::evaluate() exactly, which
      // likewise reuses one `eval` object for both, rather than going
      // through FEEvaluationImplTransformToCollocation::evaluate_values()'s
      // separate wrapper (still used, unchanged, by
      // LocalLaplaceOperatorBatched). The values<> transform always runs
      // (needed as co_gradients<>()'s own input regardless of which flags
      // were requested); the gradient step issues one separate
      // co_gradients<direction,...>() call per direction (not
      // LocalLaplaceOperatorBatched's fused, register-cached single pass),
      // so this class's performance can be measured against the fused path
      // directly: same math, same per-point work, but dim separate
      // passes/team_barrier()s instead of one. No early return for
      // EvaluationFlags::nothing, matching deal.II's own TransformToCollocation
      // evaluate().
      DEAL_II_HOST_DEVICE void
      evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag) const
      {
        const Custom::Parallel::EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                                       dim,
                                                       n_local_dofs_1d,
                                                       n_q_points_1d,
                                                       Number>
          eval(data->team_member,
               data->shape_values,
               nullptr,
               data->co_shape_gradients,
               nullptr,
               data->c_nelmtPerBatch,
               data->threadIdx,
               data->blockSize);

        if constexpr (dim == 1)
          {
            eval.template values<0, true, false>(data->values, data->values);
          }
        else if constexpr (dim == 2)
          {
            eval.template values<0, true, false>(data->values, data->scratch);
            eval.template values<1, true, false>(data->scratch, data->values);
          }
        else // dim == 3
          {
            eval.template values<0, true, false>(data->values, data->scratch);
            eval.template values<1, true, false>(data->scratch,
                                                 data->scratch + data->quad_size_per_batch);
            eval.template values<2, true, false>(data->scratch + data->quad_size_per_batch,
                                                 data->values);
          }

        if (evaluation_flag & EvaluationFlags::gradients)
          {
            eval.template co_gradients<0, true, false>(data->values, data->gradients);
            if constexpr (dim > 1)
              eval.template co_gradients<1, true, false>(data->values,
                                                         data->gradients +
                                                           data->quad_size_per_batch);
            if constexpr (dim > 2)
              eval.template co_gradients<2, true, false>(data->values,
                                                         data->gradients +
                                                           2 * data->quad_size_per_batch);
          }
      }

      // Deal.II-style integrate(): the same single `eval` instance handles
      // both the gradient step (BK3 steps 5-6 in reverse, via
      // co_gradients<>()) and the final quad->dof values transform (BK3
      // steps 7-9, via values<>()) -- mirroring real deal.II's own
      // FEEvaluationImplTransformToCollocation::integrate() exactly,
      // including its conditional `add` on the first co_gradients() call:
      // when values were *also* submitted (`values` already holds
      // submit_value()'s JxW-multiplied contribution), that first call
      // accumulates onto it instead of overwriting it, and every subsequent
      // direction's call always accumulates. The final values<> transform
      // always runs afterward, exactly as in deal.II's own integrate(),
      // which also has no early return for EvaluationFlags::nothing.
      DEAL_II_HOST_DEVICE void
      integrate(const EvaluationFlags::EvaluationFlags integration_flag) const
      {
        const Custom::Parallel::EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                                       dim,
                                                       n_local_dofs_1d,
                                                       n_q_points_1d,
                                                       Number>
          eval(data->team_member,
               data->shape_values,
               nullptr,
               data->co_shape_gradients,
               nullptr,
               data->c_nelmtPerBatch,
               data->threadIdx,
               data->blockSize);

        if (integration_flag & EvaluationFlags::gradients)
          {
            const bool add_values = static_cast<bool>(integration_flag & EvaluationFlags::values);

            if constexpr (dim == 1)
              {
                if (add_values)
                  eval.template co_gradients<0, false, true>(data->gradients, data->values);
                else
                  eval.template co_gradients<0, false, false>(data->gradients, data->values);
              }
            else if constexpr (dim == 2)
              {
                if (add_values)
                  eval.template co_gradients<1, false, true>(data->gradients +
                                                               data->quad_size_per_batch,
                                                             data->values);
                else
                  eval.template co_gradients<1, false, false>(data->gradients +
                                                                data->quad_size_per_batch,
                                                              data->values);
                eval.template co_gradients<0, false, true>(data->gradients, data->values);
              }
            else if constexpr (dim == 3)
              {
                if (add_values)
                  eval.template co_gradients<2, false, true>(data->gradients +
                                                               2 * data->quad_size_per_batch,
                                                             data->values);
                else
                  eval.template co_gradients<2, false, false>(data->gradients +
                                                                2 * data->quad_size_per_batch,
                                                              data->values);
                eval.template co_gradients<1, false, true>(data->gradients +
                                                             data->quad_size_per_batch,
                                                           data->values);
                eval.template co_gradients<0, false, true>(data->gradients, data->values);
              }
          }

        if constexpr (dim == 1)
          {
            eval.template values<0, false, false>(data->values, data->values);
          }
        else if constexpr (dim == 2)
          {
            eval.template values<1, false, false>(data->values, data->scratch);
            eval.template values<0, false, false>(data->scratch, data->values);
          }
        else // dim == 3
          {
            eval.template values<2, false, false>(data->values, data->scratch);
            eval.template values<1, false, false>(data->scratch,
                                                  data->scratch + data->quad_size_per_batch);
            eval.template values<0, false, false>(data->scratch + data->quad_size_per_batch,
                                                  data->values);
          }
      }

      DEAL_II_HOST_DEVICE Number
      get_value(const int point) const
      {
        return data->values[point];
      }

      DEAL_II_HOST_DEVICE void
      submit_value(const Number &value, const int point) const
      {
        const int          nq_total    = data->quad_size_per_batch / data->nelmtPerBatch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        data->values[point] = value * data->JxW(point_local, global_cell);
      }

      DEAL_II_HOST_DEVICE gradient_type
      get_gradient(const int point) const
      {
        const int          nq_total    = data->quad_size_per_batch / data->nelmtPerBatch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        gradient_type grad;
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->inv_jacobian(point_local, global_cell, d_2, d_1) *
                     data->gradients[d_2 * data->quad_size_per_batch + point];
            grad[d_1] = tmp;
          }
        return grad;
      }

      DEAL_II_HOST_DEVICE void
      submit_gradient(const gradient_type &gradient, const int point) const
      {
        const int          nq_total    = data->quad_size_per_batch / data->nelmtPerBatch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        const Number jxw = data->JxW(point_local, global_cell);

        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->inv_jacobian(point_local, global_cell, d_1, d_2) * gradient[d_2];
            data->gradients[d_1 * data->quad_size_per_batch + point] = tmp * jxw;
          }
      }

    private:
      const data_type *data;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
