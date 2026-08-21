#ifndef kernels_portable_batched_fe_evaluation_h
#define kernels_portable_batched_fe_evaluation_h

#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include "matrix_free/portable_evaluation_kernels.h"

DEAL_II_NAMESPACE_OPEN

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
        const int    nq_total          = data->n_q_points_per_batch / data->n_elements_per_batch;
        const int    e                 = point / nq_total;
        unsigned int global_cell_index = data->batch_index * data->n_elements_per_batch + e;
        if (data->cell_range_ids.size() > 0)
          global_cell_index = data->cell_range_ids(global_cell_index);
        return global_cell_index;
      }

      DEAL_II_HOST_DEVICE void
      read_dof_values(const Custom::Parallel::DeviceView<Number> &src) const
      {
        Custom::Parallel::read_dof_values<dim, n_local_dofs_1d>(data->team_member,
                                                                data->dof_indices,
                                                                data->cell_range_ids,
                                                                src,
                                                                data->values,
                                                                data->batch_index,
                                                                data->n_elements_per_batch,
                                                                data->n_elements_in_current_batch,
                                                                data->thread_id,
                                                                data->block_size);
      }

      DEAL_II_HOST_DEVICE void
      distribute_local_to_global(Custom::Parallel::DeviceView<Number> &dst) const
      {
        Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(
          data->team_member,
          data->dof_indices,
          data->cell_range_ids,
          data->values,
          dst,
          data->batch_index,
          data->n_elements_per_batch,
          data->n_elements_in_current_batch,
          data->thread_id,
          data->block_size);
      }

      DEAL_II_HOST_DEVICE void
      evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag) const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.evaluate(data->values, data->gradients, data->scratch, evaluation_flag);
      }

      DEAL_II_HOST_DEVICE void
      integrate(const EvaluationFlags::EvaluationFlags integration_flag) const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.integrate(data->values, data->gradients, data->scratch, integration_flag);
      }

      DEAL_II_HOST_DEVICE void
      evaluate_values() const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.evaluate_values(data->values, data->values, data->scratch);
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      evaluate_gradients() const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.template evaluate_gradients<add>(data->values, data->gradients);
      }

      DEAL_II_HOST_DEVICE void
      integrate_values() const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.integrate_values(data->values, data->values, data->scratch);
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      integrate_gradients() const
      {
        const Custom::Parallel::
          FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
            fe_eval(data->team_member,
                    data->shape_values,
                    data->co_shape_gradients,
                    data->n_elements_per_batch,
                    data->n_elements_in_current_batch,
                    data->batch_index,
                    data->thread_id,
                    data->block_size);

        fe_eval.template integrate_gradients<add>(data->gradients, data->values);
      }

      DEAL_II_HOST_DEVICE Number
      get_value(const int point) const
      {
        return data->values[point];
      }

      DEAL_II_HOST_DEVICE void
      submit_value(const Number &value, const int point) const
      {
        const int          nq_total    = data->n_q_points_per_batch / data->n_elements_per_batch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        data->values[point] = value * data->JxW(point_local, global_cell);
      }

      DEAL_II_HOST_DEVICE gradient_type
      get_gradient(const int point) const
      {
        const int          nq_total    = data->n_q_points_per_batch / data->n_elements_per_batch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        gradient_type grad;
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->inv_jacobian(point_local, global_cell, d_2, d_1) *
                     data->gradients[d_2 * data->n_q_points_per_batch + point];
            grad[d_1] = tmp;
          }
        return grad;
      }

      DEAL_II_HOST_DEVICE void
      submit_gradient(const gradient_type &gradient, const int point) const
      {
        const int          nq_total    = data->n_q_points_per_batch / data->n_elements_per_batch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        const Number jxw = data->JxW(point_local, global_cell);

        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->inv_jacobian(point_local, global_cell, d_1, d_2) * gradient[d_2];
            data->gradients[d_1 * data->n_q_points_per_batch + point] = tmp * jxw;
          }
      }

    private:
      const data_type *data;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
