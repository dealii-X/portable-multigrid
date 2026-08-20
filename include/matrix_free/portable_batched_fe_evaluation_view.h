#ifndef kernels_portable_batched_fe_evaluation_view_h
#define kernels_portable_batched_fe_evaluation_view_h

#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include "matrix_free/portable_evaluation_kernels_view.h"

DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {
    /**
     * View-based counterpart of FEEvaluation (portable_batched_fe_
     * evaluation.h) -- a separate class, not an overload, since its data
     * member's type (BatchDataView<dim, Number>* vs BatchData<dim,
     * Number>*) differs, same reasoning as EvaluatorTensorProductView vs
     * EvaluatorTensorProduct. All evaluate/integrate work is delegated
     * straight to FEEvaluationImplTransformToCollocationView's static
     * methods (portable_evaluation_kernels_view.h), so unlike the
     * pointer-based FEEvaluation there's no fe_eval instance to construct
     * per call -- data is threaded straight through. Geometric factors
     * (JxW/inv_jacobian) come from data->precomputed_data.data, real
     * deal.II's own Portable::MatrixFree<dim, Number>::PrecomputedData,
     * same fields BatchData held copies of by reference.
     */
    template <int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
    class FEEvaluationView
    {
      static_assert(n_components_ == 1,
                    "Custom::Parallel::FEEvaluationView only supports scalar "
                    "(n_components == 1) problems for now -- ShapeDataView's "
                    "values/gradients have no component axis yet.");

    public:
      using value_type    = Number;
      using gradient_type = Tensor<1, dim, Number>;
      using data_type     = Custom::Parallel::BatchDataView<dim, Number>;

      static constexpr unsigned int n_local_dofs_1d = fe_degree + 1;
      static constexpr unsigned int n_q_points      = Utilities::pow(n_q_points_1d, dim);

      using FEEvalImpl = Custom::Parallel::
        FEEvaluationImplTransformToCollocationView<dim, fe_degree, n_q_points_1d, Number>;

      DEAL_II_HOST_DEVICE
      explicit FEEvaluationView(const data_type *data)
        : data(data)
      {}

      // Same role as FEEvaluation::get_global_cell_index() -- see there.
      DEAL_II_HOST_DEVICE unsigned int
      get_global_cell_index(const int point) const
      {
        const int    nq_total          = data->quad_size_per_batch / data->nelmtPerBatch;
        const int    e                 = point / nq_total;
        unsigned int global_cell_index = data->batchIdx * data->nelmtPerBatch + e;
        if (data->precomputed_data.cell_range_ids.size() > 0)
          global_cell_index = data->precomputed_data.cell_range_ids(global_cell_index);
        return global_cell_index;
      }

      DEAL_II_HOST_DEVICE void
      read_dof_values(const Custom::Parallel::DeviceView<Number> &src) const
      {
        Custom::Parallel::read_dof_values<dim, n_local_dofs_1d>(
          data->team_member,
          data->precomputed_data.dof_indices,
          data->precomputed_data.cell_range_ids,
          src,
          data->shape_data.values,
          data->batchIdx,
          data->nelmtPerBatch,
          data->c_nelmtPerBatch,
          data->threadIdx,
          data->blockSize);
      }

      DEAL_II_HOST_DEVICE void
      distribute_local_to_global(Custom::Parallel::DeviceView<Number> &dst) const
      {
        Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(
          data->team_member,
          data->precomputed_data.dof_indices,
          data->precomputed_data.cell_range_ids,
          data->shape_data.values,
          dst,
          data->batchIdx,
          data->nelmtPerBatch,
          data->c_nelmtPerBatch,
          data->threadIdx,
          data->blockSize);
      }

      DEAL_II_HOST_DEVICE void
      evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag) const
      {
        FEEvalImpl::evaluate(data, evaluation_flag);
      }

      DEAL_II_HOST_DEVICE void
      integrate(const EvaluationFlags::EvaluationFlags integration_flag) const
      {
        FEEvalImpl::integrate(data, integration_flag);
      }

      DEAL_II_HOST_DEVICE void
      evaluate_values() const
      {
        FEEvalImpl::evaluate_values(data, data->shape_data.values, data->shape_data.values);
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      evaluate_gradients() const
      {
        FEEvalImpl::template evaluate_gradients<add>(data,
                                                      data->shape_data.values,
                                                      data->shape_data.gradients);
      }

      DEAL_II_HOST_DEVICE void
      integrate_values() const
      {
        FEEvalImpl::integrate_values(data, data->shape_data.values, data->shape_data.values);
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      integrate_gradients() const
      {
        FEEvalImpl::template integrate_gradients<add>(data,
                                                       data->shape_data.gradients,
                                                       data->shape_data.values);
      }

      DEAL_II_HOST_DEVICE Number
      get_value(const int point) const
      {
        return data->shape_data.values(point);
      }

      DEAL_II_HOST_DEVICE void
      submit_value(const Number &value, const int point) const
      {
        const int          nq_total    = data->quad_size_per_batch / data->nelmtPerBatch;
        const int          point_local = point % nq_total;
        const unsigned int global_cell = get_global_cell_index(point);

        data->shape_data.values(point) =
          value * data->precomputed_data.data.JxW(point_local, global_cell);
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
              tmp += data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_2, d_1) *
                     data->shape_data.gradients(point, d_2);
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

        const Number jxw = data->precomputed_data.data.JxW(point_local, global_cell);

        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, d_2) *
                     gradient[d_2];
            data->shape_data.gradients(point, d_1) = tmp * jxw;
          }
      }

    private:
      const data_type *data;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
