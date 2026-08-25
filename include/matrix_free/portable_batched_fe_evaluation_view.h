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
                    "values/gradients do have a component axis (matching real "
                    "deal.II's SharedData), but every access here hardcodes "
                    "component 0.");

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
        const int    e                 = point / data->n_q_points;
        unsigned int global_cell_index = data->batch_index * data->n_elements_per_batch + e;
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
          Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0),
          data->batch_index,
          data->n_elements_per_batch,
          data->n_elements_in_current_batch);
        if constexpr (running_in_debug_mode())
          dof_values_initialized = true;
      }

      DEAL_II_HOST_DEVICE void
      distribute_local_to_global(Custom::Parallel::DeviceView<Number> &dst) const
      {
        Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(
          data->team_member,
          data->precomputed_data.dof_indices,
          data->precomputed_data.cell_range_ids,
          Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0),
          dst,
          data->batch_index,
          data->n_elements_per_batch,
          data->n_elements_in_current_batch);
      }

      DEAL_II_HOST_DEVICE void
      evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag) const
      {
        if constexpr (running_in_debug_mode())
          Assert(dof_values_initialized,
                 ExcMessage("evaluate() was called without a prior read_dof_values()."));
        FEEvalImpl::evaluate(n_components_, evaluation_flag, data);
        if constexpr (running_in_debug_mode())
          {
            values_quad_initialized = static_cast<bool>(evaluation_flag & EvaluationFlags::values);
            gradients_quad_initialized =
              static_cast<bool>(evaluation_flag & EvaluationFlags::gradients);
          }
      }

      DEAL_II_HOST_DEVICE void
      integrate(const EvaluationFlags::EvaluationFlags integration_flag) const
      {
        if constexpr (running_in_debug_mode())
          {
            if (integration_flag & EvaluationFlags::values)
              Assert(values_quad_submitted,
                     ExcMessage("integrate() was asked for values, but submit_value() "
                                "was never called."));
            if (integration_flag & EvaluationFlags::gradients)
              Assert(gradients_quad_submitted,
                     ExcMessage("integrate() was asked for gradients, but submit_gradient() "
                                "was never called."));
          }
        FEEvalImpl::integrate(n_components_, integration_flag, data);
        if constexpr (running_in_debug_mode())
          values_quad_submitted = gradients_quad_submitted = false;
      }

      DEAL_II_HOST_DEVICE void
      evaluate_values() const
      {
        if constexpr (running_in_debug_mode())
          Assert(dof_values_initialized,
                 ExcMessage("evaluate_values() was called without a prior read_dof_values()."));
        const auto u = Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0);
        FEEvalImpl::evaluate_values(data, u, u);
        if constexpr (running_in_debug_mode())
          values_quad_initialized = true;
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      evaluate_gradients() const
      {
        FEEvalImpl::template evaluate_gradients<add>(
          data,
          Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0),
          Kokkos::subview(data->shape_data.gradients, Kokkos::ALL, Kokkos::ALL, 0));
        if constexpr (running_in_debug_mode())
          gradients_quad_initialized = true;
      }

      DEAL_II_HOST_DEVICE void
      integrate_values() const
      {
        if constexpr (running_in_debug_mode())
          Assert(values_quad_submitted,
                 ExcMessage("integrate_values() was called, but submit_value() was "
                            "never called."));
        const auto u = Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0);
        FEEvalImpl::integrate_values(data, u, u);
        if constexpr (running_in_debug_mode())
          values_quad_submitted = false;
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      integrate_gradients() const
      {
        if constexpr (running_in_debug_mode())
          Assert(gradients_quad_submitted,
                 ExcMessage("integrate_gradients() was called, but submit_gradient() "
                            "was never called."));
        FEEvalImpl::template integrate_gradients<add>(
          data,
          Kokkos::subview(data->shape_data.gradients, Kokkos::ALL, Kokkos::ALL, 0),
          Kokkos::subview(data->shape_data.values, Kokkos::ALL, 0));
        if constexpr (running_in_debug_mode())
          {
            gradients_quad_submitted = false;
            // integrate_gradients() writes its (quad->dof) result into
            // `values` -- in the gradients-only split call sequence this is
            // the only producer values ever gets, so the subsequent
            // integrate_values() call is legitimate even though
            // submit_value() itself was never called.
            values_quad_submitted = true;
          }
      }

      DEAL_II_HOST_DEVICE Number
      get_value(const int point) const
      {
        if constexpr (running_in_debug_mode())
          Assert(values_quad_initialized,
                 ExcMessage("get_value() was called without a prior evaluate()/"
                            "evaluate_values() that requested values."));
        return data->shape_data.values(point, 0);
      }

      DEAL_II_HOST_DEVICE void
      submit_value(const Number &value, const int point) const
      {
        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);

        data->shape_data.values(point, 0) =
          value * data->precomputed_data.data.JxW(point_local, global_cell);
        if constexpr (running_in_debug_mode())
          values_quad_submitted = true;
      }

      DEAL_II_HOST_DEVICE gradient_type
      get_gradient(const int point) const
      {
        if constexpr (running_in_debug_mode())
          Assert(gradients_quad_initialized,
                 ExcMessage("get_gradient() was called without a prior evaluate()/"
                            "evaluate_gradients() that requested gradients."));

        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);

        gradient_type grad;
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_2, d_1) *
                     data->shape_data.gradients(point, d_2, 0);
            grad[d_1] = tmp;
          }
        return grad;
      }

      DEAL_II_HOST_DEVICE void
      submit_gradient(const gradient_type &gradient, const int point) const
      {
        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);

        const Number jxw = data->precomputed_data.data.JxW(point_local, global_cell);

        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            Number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, d_2) *
                     gradient[d_2];
            data->shape_data.gradients(point, d_1, 0) = tmp * jxw;
          }
        if constexpr (running_in_debug_mode())
          gradients_quad_submitted = true;
      }

    private:
      const data_type *data;

      // Debug-only usage tracking, matching real deal.II's FEEvaluationBase
      // pattern -- compiled out entirely in Release. Catches: reading a
      // field evaluate() wasn't asked to produce (the actual hazard behind
      // the gradients-only scratch_pad/gradients buffer aliasing done by
      // cell_loop_batched_launch_view()), calling integrate() without
      // having submitted the corresponding field first, and calling
      // evaluate()/evaluate_values() before read_dof_values().
      mutable bool dof_values_initialized     = false;
      mutable bool values_quad_initialized    = false;
      mutable bool gradients_quad_initialized = false;
      mutable bool values_quad_submitted      = false;
      mutable bool gradients_quad_submitted   = false;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
