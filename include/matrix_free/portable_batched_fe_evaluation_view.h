#ifndef kernels_portable_batched_fe_evaluation_view_h
#define kernels_portable_batched_fe_evaluation_view_h

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/evaluation_flags.h>

#include "matrix_free/portable_evaluation_kernels_view.h"

DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {
    /**
     * This class provides all the functions necessary to evaluate functions at
     * quadrature points and cell integrations. In functionality, this class is
     * similar to FEValues<dim>.
     *
     * This class has five template arguments:
     *
     * @tparam dim Dimension in which this class is to be used
     *
     * @tparam fe_degree Degree of the tensor product finite element with fe_degree+1
     * degrees of freedom per coordinate direction
     *
     * @tparam n_q_points_1d Number of points in the quadrature formular in 1d,
     * defaults to fe_degree+1
     *
     * @tparam n_components Number of vector components when solving a system of
     * PDEs. If the same operation is applied to several components of a PDE (e.g.
     * a vector Laplace equation), they can be applied simultaneously with one
     * call (and often more efficiently). Defaults to 1
     *
     * @tparam Number Number format, @p double or @p float. Defaults to @p
     * double.
     */
    template <int dim,
              int fe_degree,
              int n_q_points_1d = fe_degree + 1,
              int n_components_ = 1,
              typename Number   = double>
    class FEEvaluationView
    {
    public:
      /**
       * An alias for the value type. This is @p Number for scalar problems
       * and Tensor<1, n_components> for vector-valued problems.
       */
      using value_type =
        std::conditional_t<(n_components_ == 1), Number, Tensor<1, n_components_, Number>>;

      /**
       * An alias for the gradient type.
       */
      using gradient_type =
        std::conditional_t<n_components_ == 1,
                           Tensor<1, dim, Number>,
                           std::conditional_t<n_components_ == dim,
                                              Tensor<2, dim, Number>,
                                              Tensor<1, n_components_, Tensor<1, dim, Number>>>>;

      /**
       * An alias to kernel specific information.
       */
      using data_type = Custom::Parallel::BatchDataView<dim, Number>;

      static constexpr unsigned int n_local_dofs_1d           = fe_degree + 1;
      static constexpr unsigned int n_q_points                = Utilities::pow(n_q_points_1d, dim);
      static constexpr unsigned int n_components              = n_components_;
      static constexpr unsigned int tensor_dofs_per_component = Utilities::pow(fe_degree + 1, dim);
      static constexpr unsigned int tensor_dofs_per_cell =
        tensor_dofs_per_component * n_components_;

      using FEEvalImpl = Custom::Parallel::
        FEEvaluationImplTransformToCollocationView<dim, fe_degree, n_q_points_1d, Number>;

      /**
       * Constructor. You will need to provide a pointer to the
       * BatchDataView object, which is typically provided to the functor
       * inside cell_loop_batched_launch_view(), and the index @p
       * dof_handler_index of the DoFHandler if more than one was provided
       * when the underlying MatrixFree object was initialized.
       *
       * @note dof_handler_index is accepted for signature parity with real
       * deal.II's Portable::FEEvaluation, but isn't functional yet --
       * BatchDataView::precomputed_data/shape_data are single references,
       * not arrays indexed by dof_handler_index (this project doesn't
       * support multiple DoFHandlers per MatrixFree). Must be 0 for now.
       */
      DEAL_II_HOST_DEVICE
      explicit FEEvaluationView(const data_type *data, const unsigned int dof_handler_index = 0)
        : data(data)
        , dof_handler_index(dof_handler_index)
      {
        Assert(dof_handler_index == 0,
               ExcMessage("Custom::Parallel::FEEvaluationView doesn't yet support "
                          "multiple DoFHandlers per MatrixFree -- dof_handler_index "
                          "must be 0."));
      }

      /**
       * Return a pointer to the BatchDataView<dim, Number> object that
       * contains necessary dof index and shape function information for
       * evaluation used in the matrix-free kernels.
       */
      DEAL_II_HOST_DEVICE const data_type *
      get_matrix_free_data() const
      {
        return data;
      }

      // Same role as FEEvaluation::get_current_cell_index() -- see the
      // class doc comment above for why this takes a quad-point index
      // instead: a team here processes a batch of cells, not exactly one.
      DEAL_II_HOST_DEVICE unsigned int
      get_global_cell_index(const int point) const
      {
        const int    e                 = point / data->n_q_points;
        unsigned int global_cell_index = data->batch_index * data->n_elements_per_batch + e;
        if (data->precomputed_data.cell_range_ids.size() > 0)
          global_cell_index = data->precomputed_data.cell_range_ids(global_cell_index);
        return global_cell_index;
      }

      /**
       * For the vector @p src, read out the values on the degrees of
       * freedom of the current cell, and store them internally. Similar
       * functionality as the function
       * DoFAccessor::get_interpolated_dof_values when no constraints are
       * present -- unlike real deal.II's Portable::FEEvaluation, this does
       * @b not yet resolve hanging-node constraints (no analog of
       * internal::resolve_hanging_nodes() ported here), so it's closer to
       * AffineConstraints::read_dof_values() without the hanging-node part.
       */
      DEAL_II_HOST_DEVICE void
      read_dof_values(const Custom::Parallel::DeviceView<Number> &src) const
      {
        Custom::Parallel::read_dof_values<dim, n_local_dofs_1d, n_components_>(
          data->team_member,
          data->precomputed_data.dof_indices,
          data->precomputed_data.cell_range_ids,
          src,
          data->shape_data.values,
          data->batch_index,
          data->n_elements_per_batch,
          data->n_elements_in_current_batch);
        if constexpr (running_in_debug_mode())
          dof_values_initialized = true;
      }

      /**
       * Take the value stored internally on dof values of the current
       * cell and sum them into the vector @p dst. Unlike real deal.II's
       * Portable::FEEvaluation, this does @b not yet apply hanging-node
       * constraints during the write operation (same caveat as
       * read_dof_values()), so it's closer to a plain scatter-add than
       * the full AffineConstraints::distribute_local_to_global().
       */
      DEAL_II_HOST_DEVICE void
      distribute_local_to_global(Custom::Parallel::DeviceView<Number> &dst) const
      {
        Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d, n_components_>(
          data->team_member,
          data->precomputed_data.dof_indices,
          data->precomputed_data.cell_range_ids,
          data->shape_data.values,
          dst,
          data->batch_index,
          data->n_elements_per_batch,
          data->n_elements_in_current_batch);
      }

      /**
       * Evaluate the function values and the gradients of the FE function
       * given at the DoF values in the input vector at the quadrature
       * points on the unit cell. The function argument @p evaluation_flag
       * specifies which parts shall actually be computed. This function
       * needs to be called before the functions @p get_value() or
       * @p get_gradient() give useful information.
       */
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

      /**
       * This function takes the values and/or gradients that are stored
       * on quadrature points, tests them by all the basis
       * functions/gradients on the cell and performs the cell integration
       * as specified by the @p integration_flag argument.
       */
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

      /**
       * Return the value of the finite element function at the quadrature
       * point with index @p point after a call to evaluate() with
       * EvaluationFlags::values set.
       */
      DEAL_II_HOST_DEVICE value_type
      get_value(const int point) const
      {
        if constexpr (running_in_debug_mode())
          Assert(values_quad_initialized,
                 ExcMessage("get_value() was called without a prior evaluate()/"
                            "evaluate_values() that requested values."));
        if constexpr (n_components_ == 1)
          return data->shape_data.values(point, 0);
        else
          {
            value_type result;
            for (unsigned int c = 0; c < n_components_; ++c)
              result[c] = data->shape_data.values(point, c);
            return result;
          }
      }

      /**
       * Return the value stored for the local degree of freedom with
       * index @p dof_index. This accesses the data loaded by
       * read_dof_values() -- same underlying buffer as get_value(), valid
       * before evaluate()/after integrate() instead of after
       * evaluate()/before integrate().
       */
      DEAL_II_HOST_DEVICE value_type
      get_dof_value(const int dof_index) const
      {
        if constexpr (running_in_debug_mode())
          Assert(dof_values_initialized,
                 ExcMessage("get_dof_value() was called without a prior "
                            "read_dof_values()/submit_dof_value()."));
        if constexpr (n_components_ == 1)
          return data->shape_data.values(dof_index, 0);
        else
          {
            value_type result;
            for (unsigned int c = 0; c < n_components_; ++c)
              result[c] = data->shape_data.values(dof_index, c);
            return result;
          }
      }

      /**
       * Submit the value @p value at quadrature point @p point for
       * subsequent integration via integrate() with
       * EvaluationFlags::values set.
       */
      DEAL_II_HOST_DEVICE void
      submit_value(const value_type &value, const int point) const
      {
        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);
        const Number       jxw         = data->precomputed_data.data.JxW(point_local, global_cell);

        if constexpr (n_components_ == 1)
          data->shape_data.values(point, 0) = value * jxw;
        else
          for (unsigned int c = 0; c < n_components_; ++c)
            data->shape_data.values(point, c) = value[c] * jxw;
        if constexpr (running_in_debug_mode())
          values_quad_submitted = true;
      }

      /**
       * Submit the value @p value for the local degree of freedom with
       * index @p dof_index, to be written out by a subsequent call to
       * distribute_local_to_global().
       */
      DEAL_II_HOST_DEVICE void
      submit_dof_value(const value_type &value, const int dof_index) const
      {
        if constexpr (n_components_ == 1)
          data->shape_data.values(dof_index, 0) = value;
        else
          for (unsigned int c = 0; c < n_components_; ++c)
            data->shape_data.values(dof_index, c) = value[c];
        if constexpr (running_in_debug_mode())
          dof_values_initialized = true;
      }

      /**
       * Return the gradient of the finite element function at the
       * quadrature point with index @p point after a call to evaluate()
       * with EvaluationFlags::gradients set.
       */
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
        if constexpr (n_components_ == 1)
          {
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp +=
                    data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_2, d_1) *
                    data->shape_data.gradients(point, d_2, 0);
                grad[d_1] = tmp;
              }
          }
        else
          {
            for (unsigned int c = 0; c < n_components_; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp +=
                      data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_2, d_1) *
                      data->shape_data.gradients(point, d_2, c);
                  grad[c][d_1] = tmp;
                }
          }
        return grad;
      }

      /**
       * Submit the gradient @p gradient at quadrature point @p point for
       * subsequent integration via integrate() with
       * EvaluationFlags::gradients set.
       */
      DEAL_II_HOST_DEVICE void
      submit_gradient(const gradient_type &gradient, const int point) const
      {
        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);

        const Number jxw = data->precomputed_data.data.JxW(point_local, global_cell);

        if constexpr (n_components_ == 1)
          {
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp +=
                    data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, d_2) *
                    gradient[d_2];
                data->shape_data.gradients(point, d_1, 0) = tmp * jxw;
              }
          }
        else
          {
            for (unsigned int c = 0; c < n_components_; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                    tmp +=
                      data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, d_2) *
                      gradient[c][d_2];
                  data->shape_data.gradients(point, d_1, c) = tmp * jxw;
                }
          }
        if constexpr (running_in_debug_mode())
          gradients_quad_submitted = true;
      }

      /**
       * Return the symmetric gradient of the finite element function at
       * quadrature point @p point after a call to evaluate() with
       * EvaluationFlags::gradients set. This function is only available
       * when the number of components equals the dimension
       * (n_components==dim) -- checked with a runtime Assert, not a
       * static_assert, matching real deal.II: the method body is only
       * instantiated if actually called, so a mismatched n_components_
       * that never calls this compiles fine.
       */
      DEAL_II_HOST_DEVICE SymmetricTensor<2, dim, Number>
                          get_symmetric_gradient(const int point) const
      {
        Assert(n_components_ == dim,
               ExcMessage("get_symmetric_gradient() only works when the number "
                          "of components equals the number of dimensions."));
        return symmetrize(get_gradient(point));
      }

      /**
       * Submit the symmetric gradient @p sym_grad at quadrature point
       * @p point for subsequent integration via integrate() with
       * EvaluationFlags::gradients set. This function is only available
       * when the number of components equals the dimension
       * (n_components_==dim).
       */
      DEAL_II_HOST_DEVICE void
      submit_symmetric_gradient(const SymmetricTensor<2, dim, Number> &sym_grad,
                                const int                              point) const
      {
        Assert(n_components_ == dim,
               ExcMessage("submit_symmetric_gradient() only works when the number "
                          "of components equals the number of dimensions."));

        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);
        const Number       jxw         = data->precomputed_data.data.JxW(point_local, global_cell);

        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              Number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp +=
                  data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, d_2) *
                  sym_grad[c][d_2];
              data->shape_data.gradients(point, d_1, c) = tmp * jxw;
            }
        if constexpr (running_in_debug_mode())
          gradients_quad_submitted = true;
      }

      /**
       * Return the divergence of the vector-valued finite element
       * function at quadrature point @p point after a call to evaluate()
       * with EvaluationFlags::gradients set. This function is only
       * available when the number of components equals the dimension
       * (n_components==dim).
       */
      DEAL_II_HOST_DEVICE Number
      get_divergence(const int point) const
      {
        Assert(n_components_ == dim,
               ExcMessage("get_divergence() only works when the number of "
                          "components equals the number of dimensions."));

        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);

        Number divergence = 0.;
        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d = 0; d < dim; ++d)
            divergence += data->precomputed_data.data.inv_jacobian(point_local, global_cell, d, c) *
                          data->shape_data.gradients(point, d, c);
        return divergence;
      }

      /**
       * Write a contribution that is multiplied by the divergence of the
       * test function to the field containing the gradients at
       * quadrature point @p point for subsequent integration via
       * integrate() with EvaluationFlags::gradients set. See
       * submit_gradient() for further information. This function is only
       * available when the number of components equals the dimension
       * (n_components==dim).
       *
       * @note This operation writes the data to the same field as
       * submit_gradient() and submit_symmetric_gradient(). As a
       * consequence, only one of these functions can be used. In case
       * several terms of this kind appear in a weak form, the
       * contribution of a potential call to this function must be added
       * into the diagonal of the rank-2 tensor contribution passed to
       * submit_gradient().
       */
      DEAL_II_HOST_DEVICE void
      submit_divergence(const Number &div_in, const int point) const
      {
        Assert(n_components_ == dim,
               ExcMessage("submit_divergence() only works when the number of "
                          "components equals the number of dimensions."));

        const int          point_local = point % data->n_q_points;
        const unsigned int global_cell = get_global_cell_index(point);
        const Number       jxw         = data->precomputed_data.data.JxW(point_local, global_cell);

        for (unsigned int c = 0; c < dim; ++c)
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            data->shape_data.gradients(point, d_1, c) =
              data->precomputed_data.data.inv_jacobian(point_local, global_cell, d_1, c) * div_in *
              jxw;
        if constexpr (running_in_debug_mode())
          gradients_quad_submitted = true;
      }

    private:
      const data_type *data;

      // Stored for parity with real deal.II's FEEvaluation, not otherwise
      // used yet -- see the constructor's doc comment above.
      const unsigned int dof_handler_index;

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
