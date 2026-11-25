// Copy of Portable::internal::batched::FEEvaluation from the deal.II branch
// portable-batched-cell-loop (include/deal.II/matrix_free/
// portable_fe_evaluation.h), in namespace Copy, so that it does not depend on
// the version of the installed deal.II. Keep in sync with the branch.

#ifndef matrix_free_dealii_portable_batched_fe_evaluation_h
#define matrix_free_dealii_portable_batched_fe_evaluation_h

#include <deal.II/base/config.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include "matrix_free_dealii/portable_batched_evaluation_kernels.h"
#include "matrix_free_dealii/portable_batched_hanging_nodes.h"

namespace Copy
{
  namespace Portable
  {
    namespace internal
    {
      using namespace dealii;

      namespace batched
      {
        /**
         * Same as Portable::FEEvaluation, but for all cells of
         * BatchedData::cell_range at once. The index @p q_point of the
         * functions below runs over the quadrature points of all cells of the
         * batch stored cell after cell, as passed to the functor of
         * BatchedData::for_each_quad_point(), and @p dof_index over their
         * degrees of freedom.
         */
        template <int dim,
                  int fe_degree,
                  int n_q_points_1d = fe_degree + 1,
                  int n_components_ = 1,
                  typename Number   = double>
        class FEEvaluation
        {
        public:
          /**
           * An alias for the value type. This is @p Number for scalar problems
           * and Tensor<1, n_components> for vector-valued problems.
           */
          using value_type = std::conditional_t<(n_components_ == 1),
                                                Number,
                                                Tensor<1, n_components_, Number>>;

          /**
           * An alias for the gradient type.
           */
          using gradient_type = std::conditional_t<
            n_components_ == 1,
            Tensor<1, dim, Number>,
            std::conditional_t<n_components_ == dim,
                               Tensor<2, dim, Number>,
                               Tensor<1, n_components_, Tensor<1, dim, Number>>>>;

          /**
           * An alias to kernel specific information.
           */
          using data_type = BatchedData<dim, Number>;

          /**
           * Dimension.
           */
          static constexpr unsigned int dimension = dim;

          /**
           * Number of components.
           */
          static constexpr unsigned int n_components = n_components_;

          /**
           * Number of quadrature points per cell.
           */
          static constexpr unsigned int n_q_points_per_cell =
            Utilities::pow(n_q_points_1d, dim);

          /**
           * Number of tensor degrees of freedom of a scalar component per cell.
           */
          static constexpr unsigned int tensor_dofs_per_component =
            Utilities::pow(fe_degree + 1, dim);

          /**
           * Number of tensor degrees of freedom of all components per cell.
           */
          static constexpr unsigned int tensor_dofs_per_cell =
            tensor_dofs_per_component * n_components;

          /**
           * Constructor, see Portable::FEEvaluation.
           */
          DEAL_II_HOST_DEVICE
          explicit FEEvaluation(const data_type   *data,
                                const unsigned int dof_handler_index = 0);

          /**
           * Return the index of the cell the quadrature point @p q_point
           * belongs to.
           */
          DEAL_II_HOST_DEVICE
          int
          get_current_cell_index(const int q_point) const;

          /**
           * Return a pointer to the BatchedData object.
           */
          DEAL_II_HOST_DEVICE
          const data_type *
          get_matrix_free_data();

          /**
           * Read the values of @p src on the degrees of freedom of the cells of
           * the batch.
           */
          DEAL_II_HOST_DEVICE void
          read_dof_values(const DeviceVector<Number> &src);

          /**
           * Sum the values stored on the degrees of freedom of the cells of the
           * batch into @p dst. The function ends with a team barrier, so that
           * the team can go on with the next batch.
           */
          DEAL_II_HOST_DEVICE void
          distribute_local_to_global(DeviceVector<Number> &dst) const;

          /**
           * Evaluate the values and gradients at the quadrature points as
           * specified by @p evaluate_flag.
           */
          DEAL_II_HOST_DEVICE void
          evaluate(const EvaluationFlags::EvaluationFlags evaluate_flag);

          /**
           * Test the submitted values and gradients by the basis functions as
           * specified by @p integration_flag.
           */
          DEAL_II_HOST_DEVICE void
          integrate(const EvaluationFlags::EvaluationFlags integration_flag);

          /**
           * The same as evaluate(EvaluationFlags::gradients) followed by
           * submit_gradient(q_op(get_gradient(q_point), q_point), q_point) for
           * every quadrature point, but with the collocation and the
           * transform-to-collocation kernels the operation is applied inside
           * the last pass of the evaluation, without storing the gradients.
           * q_op must only depend on the data of its quadrature point. Only
           * EvaluationFlags::gradients is supported.
           */
          template <typename QuadOperation>
          DEAL_II_HOST_DEVICE void
          evaluate_and_apply(
            const EvaluationFlags::EvaluationFlags evaluation_flag,
            const QuadOperation                   &q_op);

          /**
           * As evaluate_and_apply(), but on the reference cell: @p op gets the
           * gradient in reference coordinates, grad_ref[component][direction],
           * the cell index cell (PrecomputedData::first_cell plus the cell
           * within the color) and the quadrature point q within the cell, and
           * returns flux[component][direction], which is stored for
           * integrate() as it is, without inverse Jacobian or JxW. For
           * operators that bring their own geometry, e.g. the symmetric
           * G = JxW J^{-1} J^{-T} of the Laplacian.
           */
          template <typename ReferenceOperation>
          DEAL_II_HOST_DEVICE void
          evaluate_and_apply_in_reference(
            const EvaluationFlags::EvaluationFlags evaluation_flag,
            const ReferenceOperation              &op);

          /**
           * Same as Portable::FEEvaluation::get_value().
           */
          DEAL_II_HOST_DEVICE value_type
          get_value(const int q_point) const;

          /**
           * Same as Portable::FEEvaluation::get_dof_value().
           */
          DEAL_II_HOST_DEVICE value_type
          get_dof_value(const int dof_index) const;

          /**
           * Same as Portable::FEEvaluation::submit_value().
           */
          DEAL_II_HOST_DEVICE void
          submit_value(const value_type &val_in, const int q_point);

          /**
           * Same as Portable::FEEvaluation::submit_dof_value().
           */
          DEAL_II_HOST_DEVICE void
          submit_dof_value(const value_type &value, const int dof_index);

          /**
           * Same as Portable::FEEvaluation::get_gradient().
           */
          DEAL_II_HOST_DEVICE gradient_type
          get_gradient(const int q_point) const;

          /**
           * Same as Portable::FEEvaluation::submit_gradient().
           */
          DEAL_II_HOST_DEVICE void
          submit_gradient(const gradient_type &gradient, const int q_point);

          /**
           * Same as Portable::FEEvaluation::get_symmetric_gradient().
           */
          DEAL_II_HOST_DEVICE
          SymmetricTensor<2, dim, Number>
          get_symmetric_gradient(const int q_point) const;

          /**
           * Same as Portable::FEEvaluation::get_divergence().
           */
          DEAL_II_HOST_DEVICE Number
          get_divergence(const int q_point) const;

          /**
           * Same as Portable::FEEvaluation::submit_divergence().
           */
          DEAL_II_HOST_DEVICE void
          submit_divergence(const Number &div_in, const int q_point);

          /**
           * Same as Portable::FEEvaluation::submit_symmetric_gradient().
           */
          DEAL_II_HOST_DEVICE void
          submit_symmetric_gradient(
            const SymmetricTensor<2, dim, Number> &sym_grad,
            const int                              q_point);

        private:
          const unsigned int dof_handler_index;
          const data_type   *data;
          const typename MatrixFree<dim, Number>::PrecomputedData
                                  *precomputed_data;
          SharedData<dim, Number> *shared_data;

          /**
           * Index of the first cell of the batch, BatchedData::cell_range.first.
           */
          const int first_cell_id;

          /**
           * Number of cells in the batch.
           */
          const int batch_size;

          /**
           * Number of quadrature points of all cells in the batch.
           */
          const int n_q_points_per_cell_range;

          /**
           * Whether any cell of the batch has hanging node constraints.
           */
          const bool has_hanging_nodes;

          /**
           * Index within the current color of the cell the quadrature point
           * @p q_point belongs to.
           */
          DEAL_II_HOST_DEVICE int
          cell_id(const int q_point) const
          {
            return first_cell_id +
                   q_point / static_cast<int>(n_q_points_per_cell);
          }

          /**
           * Index of the quadrature point @p q_point within its cell.
           */
          DEAL_II_HOST_DEVICE int
          cell_q_point(const int q_point) const
          {
            return q_point % static_cast<int>(n_q_points_per_cell);
          }

          /**
           * Entry (@p c, @p d) of a gradient_type.
           */
          DEAL_II_HOST_DEVICE static Number &
          entry(gradient_type &g, const unsigned int c, const unsigned int d)
          {
            if constexpr (n_components_ == 1)
              {
                (void)c;
                return g[d];
              }
            else
              {
                return g[c][d];
              }
          }

          /**
           * The inverse Jacobian at quadrature point @p q_point. All entries
           * are loaded unconditionally and the off-diagonal ones of Cartesian
           * cells set to zero afterwards: with loads that do not depend on the
           * cell type, nvcc reuses them between get_gradient() and
           * submit_gradient() of the same point (with a Cartesian/general
           * branch it loads them twice). The general formula on the zeroed
           * entries gives the same results as the diagonal one.
           */
          DEAL_II_HOST_DEVICE void
          load_inverse_jacobian(const int q_point,
                                Number (&inv_jac)[dim][dim]) const
          {
            const int          cell = cell_id(q_point);
            const unsigned int index =
              precomputed_data->inv_jacobian_index(cell, cell_q_point(q_point));
            const bool is_cartesian =
              precomputed_data->cell_type(cell) ==
              ::dealii::internal::MatrixFreeFunctions::cartesian;
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                {
                  const Number value =
                    precomputed_data->inv_jacobian(index, d_1, d_2);
                  inv_jac[d_1][d_2] =
                    (d_1 == d_2 || !is_cartesian) ? value : Number(0);
                }
          }

          /**
           * submit_gradient(q_op(get_gradient(q_point), q_point), q_point)
           * with get_gradient() computed from the reference gradient
           * @p grad_ref ([component][direction]) instead of the one stored in
           * shared_data, and the inverse Jacobian loaded once for both.
           */
          template <typename QuadOperation>
          DEAL_II_HOST_DEVICE void
          apply_at_point(const Number (&grad_ref)[n_components_][dim],
                         const int            q_point,
                         const QuadOperation &q_op) const
          {
            const int          cell = cell_id(q_point);
            const int          q    = cell_q_point(q_point);
            const unsigned int inv_jac_idx =
              precomputed_data->inv_jacobian_index(cell, q);
            const bool is_cartesian =
              precomputed_data->cell_type(cell) ==
              ::dealii::internal::MatrixFreeFunctions::cartesian;

            Number inv_jac[dim][dim];
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                inv_jac[d_1][d_2] =
                  (d_1 == d_2 || !is_cartesian) ?
                    precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) :
                    Number(0);

            // as get_gradient()
            gradient_type grad;
            for (unsigned int c = 0; c < n_components_; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  Number tmp = 0.;
                  if (is_cartesian)
                    tmp = inv_jac[d_1][d_1] * grad_ref[c][d_1];
                  else
                    for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                      tmp += inv_jac[d_2][d_1] * grad_ref[c][d_2];
                  entry(grad, c, d_1) = tmp;
                }

            gradient_type flux = q_op(grad, q_point);

            // as submit_gradient()
            const Number JxW = precomputed_data->JxW_value(cell, q);
            for (unsigned int c = 0; c < n_components_; ++c)
              for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
                {
                  if (is_cartesian)
                    shared_data->gradients(q_point, d_1, c) =
                      inv_jac[d_1][d_1] * entry(flux, c, d_1) * JxW;
                  else
                    {
                      Number tmp = 0.;
                      for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                        tmp += inv_jac[d_1][d_2] * entry(flux, c, d_2);
                      shared_data->gradients(q_point, d_1, c) = tmp * JxW;
                    }
                }
          }

          /**
           * Check whether any cell of the batch has hanging node constraints.
           */
          DEAL_II_HOST_DEVICE bool
          batch_has_hanging_nodes() const
          {
            using ConstraintKinds =
              dealii::internal::MatrixFreeFunctions::ConstraintKinds;

            ConstraintKinds batch_mask = ConstraintKinds::unconstrained;
            for (int i = 0; i < batch_size * n_components_; ++i)
              batch_mask |= precomputed_data->constraint_mask(
                first_cell_id * n_components_ + i);
            return batch_mask != ConstraintKinds::unconstrained;
          }

          /**
           * Copy the hanging node constraint weights to the scratch pad, which
           * is only in use during evaluate() and integrate(), and return a view
           * of them. The caller has to add a team barrier.
           */
          DEAL_II_HOST_DEVICE
          typename SharedData<dim, Number>::SharedViewScratchPad
          stage_constraint_weights() const
          {
            const int n_weights = precomputed_data->constraint_weights.size();
            Assert(shared_data->scratch_pad.size() >=
                     static_cast<std::size_t>(n_weights),
                   ExcInternalError());

            const typename SharedData<dim, Number>::SharedViewScratchPad weights =
              Kokkos::subview(shared_data->scratch_pad,
                              Kokkos::make_pair(0, n_weights));
            Kokkos::parallel_for(Kokkos::TeamVectorRange(data->team_member,
                                                         n_weights),
                                 [&](const int &i) {
                                   weights(i) =
                                     precomputed_data->constraint_weights(i);
                                 });
            return weights;
          }

          /**
           * The hanging node constraint masks of component @p component of the
           * cells of the batch.
           */
          DEAL_II_HOST_DEVICE auto
          constraint_masks(const unsigned int component) const
          {
            // stored as [cell * n_components + component]
            const Kokkos::View<
              const dealii::internal::MatrixFreeFunctions::ConstraintKinds **,
              Kokkos::LayoutRight,
              MemorySpace::Default::kokkos_space,
              Kokkos::MemoryTraits<Kokkos::Unmanaged>>
              masks(precomputed_data->constraint_mask.data(),
                    precomputed_data->n_cells,
                    n_components_);
            return Kokkos::subview(masks,
                                   Kokkos::make_pair(first_cell_id,
                                                     first_cell_id + batch_size),
                                   component);
          }
        };



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          FEEvaluation(const data_type   *data,
                       const unsigned int dof_handler_index)
          : dof_handler_index(dof_handler_index)
          , data(data)
          , precomputed_data(&data->precomputed_data[dof_handler_index])
          , shared_data(&data->shared_data[dof_handler_index])
          , first_cell_id(data->cell_range.first)
          , batch_size(data->batch_size())
          , n_q_points_per_cell_range(batch_size * n_q_points_per_cell)
          , has_hanging_nodes(batch_has_hanging_nodes())
        {
          AssertIndexRange(dof_handler_index, data->n_dof_handler);
          Assert(n_components_ == precomputed_data->n_components,
                 ExcMessage("Wrong number of components."));
          Assert(first_cell_id >= 0, ExcInternalError());
          AssertIndexRange(first_cell_id + batch_size,
                           precomputed_data->n_cells + 1);
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE int
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_current_cell_index(const int q_point) const
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          return precomputed_data->first_cell + cell_id(q_point);
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE const typename FEEvaluation<dim,
                                                        fe_degree,
                                                        n_q_points_1d,
                                                        n_components_,
                                                        Number>::data_type *
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_matrix_free_data()
        {
          return data;
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          read_dof_values(const DeviceVector<Number> &src)
        {
          constexpr int n_dofs    = tensor_dofs_per_component;
          constexpr int n_dofs_1d = fe_degree + 1;
          constexpr int n_lines   = n_dofs / n_dofs_1d;

          // One thread per line of dofs along the last direction, with a
          // compile-time loop over the line: all its index and value loads
          // are in flight at once (as in the BK3 kernels), instead of one
          // index-then-value round trip per dof.
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(data->team_member, batch_size * n_lines),
            [&](const int &t) {
              const int b    = t / n_lines;
              const int line = t % n_lines;
              const int cell = first_cell_id + b;
              for (unsigned int c = 0; c < n_components_; ++c)
                {
                  types::global_dof_index index[n_dofs_1d];
                  for (int k = 0; k < n_dofs_1d; ++k)
                    index[k] = precomputed_data->local_to_global(
                      line + k * n_lines + n_dofs * c, cell);
                  Number value[n_dofs_1d];
                  for (int k = 0; k < n_dofs_1d; ++k)
                    // Dirichlet constrained DoFs read as zero
                    value[k] = (index[k] == numbers::invalid_unsigned_int) ?
                                 Number(0) :
                                 src[index[k]];
                  for (int k = 0; k < n_dofs_1d; ++k)
                    shared_data->values(b * n_dofs + line + k * n_lines, c) =
                      value[k];
                }
            });
          typename SharedData<dim, Number>::SharedViewScratchPad
            constraint_weights;
          if (has_hanging_nodes)
            constraint_weights = stage_constraint_weights();
          data->team_member.team_barrier();

          if (has_hanging_nodes)
            for (unsigned int c = 0; c < n_components_; ++c)
              resolve_hanging_nodes<dim, fe_degree, false>(
                data->team_member,
                constraint_weights,
                constraint_masks(c),
                Kokkos::subview(shared_data->values, Kokkos::ALL, c),
                batch_size);
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          distribute_local_to_global(DeviceVector<Number> &dst) const
        {
          constexpr int n_dofs = tensor_dofs_per_component;

          if (has_hanging_nodes)
            {
              const auto constraint_weights = stage_constraint_weights();
              data->team_member.team_barrier();

              for (unsigned int c = 0; c < n_components_; ++c)
                resolve_hanging_nodes<dim, fe_degree, true>(
                  data->team_member,
                  constraint_weights,
                  constraint_masks(c),
                  Kokkos::subview(shared_data->values, Kokkos::ALL, c),
                  batch_size);
            }

          if (precomputed_data->use_coloring)
            {
              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(data->team_member, batch_size * n_dofs),
                [&](const int &i) {
                  const int cell = first_cell_id + i / n_dofs;
                  const int dof  = i % n_dofs;
                  for (unsigned int c = 0; c < n_components_; ++c)
                    {
                      const types::global_dof_index index =
                        precomputed_data->local_to_global(dof + n_dofs * c, cell);
                      if (index != numbers::invalid_unsigned_int)
                        dst[index] += shared_data->values(i, c);
                    }
                });
            }
          else
            {
              Kokkos::parallel_for(
                Kokkos::TeamVectorRange(data->team_member, batch_size * n_dofs),
                [&](const int &i) {
                  const int cell = first_cell_id + i / n_dofs;
                  const int dof  = i % n_dofs;
                  for (unsigned int c = 0; c < n_components_; ++c)
                    {
                      const types::global_dof_index index =
                        precomputed_data->local_to_global(dof + n_dofs * c, cell);
                      if (index != numbers::invalid_unsigned_int)
                        Kokkos::atomic_add(&dst[index],
                                           shared_data->values(i, c));
                    }
                });
            }

          // the next batch of the team reuses the shared memory
          data->team_member.team_barrier();
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          evaluate(const EvaluationFlags::EvaluationFlags evaluation_flag)
        {
          using ElementType =
            ::dealii::internal::MatrixFreeFunctions::ElementType;

          if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
              precomputed_data->element_type ==
                ElementType::tensor_symmetric_collocation)
            {
              FEEvaluationImplCollocation<dim, fe_degree, Number>::evaluate(
                dof_handler_index, n_components, evaluation_flag, data);
            }
          // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
          // shape_info.h for more details
          else if (fe_degree >= 0 &&
                   use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric)
            {
              FEEvaluationImplTransformToCollocation<
                dim,
                fe_degree,
                n_q_points_1d,
                Number>::evaluate(dof_handler_index,
                                  n_components,
                                  evaluation_flag,
                                  data);
            }
          else if (fe_degree >= 0 &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric_no_collocation)
            {
              FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::evaluate(
                dof_handler_index, n_components, evaluation_flag, data);
            }
          else
            {
              Kokkos::abort(
                "The element type is not yet supported by the portable "
                "matrix-free module.");
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          integrate(const EvaluationFlags::EvaluationFlags integration_flag)
        {
          using ElementType =
            ::dealii::internal::MatrixFreeFunctions::ElementType;

          if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
              precomputed_data->element_type ==
                ElementType::tensor_symmetric_collocation)
            {
              FEEvaluationImplCollocation<dim, fe_degree, Number>::integrate(
                dof_handler_index, n_components, integration_flag, data);
            }
          // '<=' on type means tensor_symmetric or tensor_symmetric_hermite, see
          // shape_info.h for more details
          else if (fe_degree >= 0 &&
                   use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric)
            {
              FEEvaluationImplTransformToCollocation<
                dim,
                fe_degree,
                n_q_points_1d,
                Number>::integrate(dof_handler_index,
                                   n_components,
                                   integration_flag,
                                   data);
            }
          else if (fe_degree >= 0 &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric_no_collocation)
            {
              FEEvaluationImpl<dim, fe_degree, n_q_points_1d, Number>::integrate(
                dof_handler_index, n_components, integration_flag, data);
            }
          else
            {
              Kokkos::abort(
                "The element type is not yet supported by the portable "
                "matrix-free module.");
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        template <typename QuadOperation>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          evaluate_and_apply(
            const EvaluationFlags::EvaluationFlags evaluation_flag,
            const QuadOperation                   &q_op)
        {
          using ElementType =
            ::dealii::internal::MatrixFreeFunctions::ElementType;

          Assert(evaluation_flag == EvaluationFlags::gradients,
                 ExcNotImplemented());

          // writes shared_data->gradients, which the fused pass does not read
          const auto apply =
            [&](const int     q_point,
                const Number (&grad_ref)[n_components_][dim]) {
              apply_at_point(grad_ref, q_point, q_op);
            };

          // the dispatch of evaluate()
          if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
              precomputed_data->element_type ==
                ElementType::tensor_symmetric_collocation)
            {
              FEEvaluationImplCollocation<dim, fe_degree, Number>::
                template evaluate_gradients_and_apply<n_components_>(
                  dof_handler_index, data, apply);
            }
          else if (fe_degree >= 0 &&
                   use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric)
            {
              FEEvaluationImplTransformToCollocation<dim,
                                                     fe_degree,
                                                     n_q_points_1d,
                                                     Number>::
                template evaluate_gradients_and_apply<n_components_>(
                  dof_handler_index, data, apply);
            }
          else
            {
              evaluate(evaluation_flag);
              data->for_each_quad_point([&](const int &q_point) {
                submit_gradient(q_op(get_gradient(q_point), q_point), q_point);
              });
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        template <typename ReferenceOperation>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          evaluate_and_apply_in_reference(
            const EvaluationFlags::EvaluationFlags evaluation_flag,
            const ReferenceOperation              &op)
        {
          using ElementType =
            ::dealii::internal::MatrixFreeFunctions::ElementType;

          Assert(evaluation_flag == EvaluationFlags::gradients,
                 ExcNotImplemented());

          const auto apply =
            [&](const int     q_point,
                const Number (&grad_ref)[n_components_][dim]) {
              Number flux[n_components_][dim];
              op(grad_ref,
                 precomputed_data->first_cell + cell_id(q_point),
                 cell_q_point(q_point),
                 flux);
              for (unsigned int c = 0; c < n_components_; ++c)
                for (unsigned int d = 0; d < dim; ++d)
                  shared_data->gradients(q_point, d, c) = flux[c][d];
            };

          if (fe_degree >= 0 && fe_degree + 1 == n_q_points_1d &&
              precomputed_data->element_type ==
                ElementType::tensor_symmetric_collocation)
            {
              FEEvaluationImplCollocation<dim, fe_degree, Number>::
                template evaluate_gradients_and_apply<n_components_>(
                  dof_handler_index, data, apply);
            }
          else if (fe_degree >= 0 &&
                   use_collocation_evaluation(fe_degree, n_q_points_1d) &&
                   precomputed_data->element_type <=
                     ElementType::tensor_symmetric)
            {
              FEEvaluationImplTransformToCollocation<dim,
                                                     fe_degree,
                                                     n_q_points_1d,
                                                     Number>::
                template evaluate_gradients_and_apply<n_components_>(
                  dof_handler_index, data, apply);
            }
          else
            {
              // shared_data->gradients holds the reference gradients
              evaluate(evaluation_flag);
              data->for_each_quad_point([&](const int &q_point) {
                Number grad_ref[n_components_][dim];
                for (unsigned int c = 0; c < n_components_; ++c)
                  for (unsigned int d = 0; d < dim; ++d)
                    grad_ref[c][d] = shared_data->gradients(q_point, d, c);
                apply(q_point, grad_ref);
              });
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                  fe_degree,
                                                  n_q_points_1d,
                                                  n_components_,
                                                  Number>::value_type
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_value(const int q_point) const
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          if constexpr (n_components_ == 1)
            {
              return shared_data->values(q_point, 0);
            }
          else
            {
              value_type result;
              for (unsigned int c = 0; c < n_components; ++c)
                result[c] = shared_data->values(q_point, c);
              return result;
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                  fe_degree,
                                                  n_q_points_1d,
                                                  n_components_,
                                                  Number>::value_type
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_dof_value(const int dof_index) const
        {
          AssertIndexRange(dof_index, batch_size * tensor_dofs_per_component);
          if constexpr (n_components_ == 1)
            {
              return shared_data->values(dof_index, 0);
            }
          else
            {
              value_type result;
              for (unsigned int c = 0; c < n_components; ++c)
                result[c] = shared_data->values(dof_index, c);
              return result;
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          submit_value(const value_type &value, const int q_point)
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(precomputed_data->JxW.size() > 0,
                 ExcMessage("submit_value() requires precomputed JxW"));
          const Number JxW =
            precomputed_data->JxW_value(cell_id(q_point), cell_q_point(q_point));
          if constexpr (n_components_ == 1)
            {
              shared_data->values(q_point, 0) = value * JxW;
            }
          else
            {
              for (unsigned int c = 0; c < n_components; ++c)
                shared_data->values(q_point, c) = value[c] * JxW;
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          submit_dof_value(const value_type &value, const int dof_index)
        {
          AssertIndexRange(dof_index, batch_size * tensor_dofs_per_component);
          if constexpr (n_components_ == 1)
            {
              shared_data->values(dof_index, 0) = value;
            }
          else
            {
              for (unsigned int c = 0; c < n_components; ++c)
                shared_data->values(dof_index, c) = value[c];
            }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE typename FEEvaluation<dim,
                                                  fe_degree,
                                                  n_q_points_1d,
                                                  n_components_,
                                                  Number>::gradient_type
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_gradient(const int q_point) const
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(precomputed_data->inv_jacobian.size() > 0,
                 ExcMessage("get_gradient() requires precomputed inv_jacobian"));

          Number inv_jac[dim][dim];
          load_inverse_jacobian(q_point, inv_jac);

          gradient_type grad;
          for (unsigned int c = 0; c < n_components_; ++c)
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp += inv_jac[d_2][d_1] * shared_data->gradients(q_point, d_2, c);
                entry(grad, c, d_1) = tmp;
              }
          return grad;
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          submit_gradient(const gradient_type &gradient, const int q_point)
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(precomputed_data->inv_jacobian.size() > 0,
                 ExcMessage(
                   "submit_gradient() requires precomputed inv_jacobian"));
          Assert(precomputed_data->JxW.size() > 0,
                 ExcMessage("submit_gradient() requires precomputed JxW"));

          Number inv_jac[dim][dim];
          load_inverse_jacobian(q_point, inv_jac);
          const Number JxW =
            precomputed_data->JxW_value(cell_id(q_point), cell_q_point(q_point));

          gradient_type grad = gradient;
          for (unsigned int c = 0; c < n_components_; ++c)
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp += inv_jac[d_1][d_2] * entry(grad, c, d_2);
                shared_data->gradients(q_point, d_1, c) = tmp * JxW;
              }
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE SymmetricTensor<2, dim, Number>
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_symmetric_gradient(const int q_point) const
        {
          Assert(n_components_ == dim,
                 ExcMessage(
                   "Function get_symmetric_gradient() only works when the "
                   "number of components and the number of dimensions are "
                   "equal."));

          return symmetrize(get_gradient(q_point));
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE Number
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          get_divergence(const int q_point) const
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(n_components_ == dim,
                 ExcMessage("Function get_divergence() only works when the "
                            "number of components and the number of dimensions "
                            "are equal."));
          Assert(precomputed_data->inv_jacobian.size() > 0,
                 ExcMessage(
                   "get_divergence() requires precomputed inv_jacobian"));

          const unsigned int inv_jac_idx =
            precomputed_data->inv_jacobian_index(cell_id(q_point),
                                                 cell_q_point(q_point));
          Number divergence = 0.;
          for (unsigned int c = 0; c < dim; ++c)
            for (unsigned int d = 0; d < dim; ++d)
              divergence += precomputed_data->inv_jacobian(inv_jac_idx, d, c) *
                            shared_data->gradients(q_point, d, c);
          return divergence;
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          submit_divergence(const Number &div_in, const int q_point)
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(n_components_ == dim,
                 ExcMessage("Function submit_divergence() only works when the "
                            "number of components and the number of dimensions "
                            "are equal."));
          Assert(precomputed_data->inv_jacobian.size() > 0,
                 ExcMessage(
                   "submit_divergence() requires precomputed inv_jacobian"));
          Assert(precomputed_data->JxW.size() > 0,
                 ExcMessage("submit_divergence() requires precomputed JxW"));

          const int          cell = cell_id(q_point);
          const int          q    = cell_q_point(q_point);
          const unsigned int inv_jac_idx =
            precomputed_data->inv_jacobian_index(cell, q);
          const Number JxW = precomputed_data->JxW_value(cell, q);
          for (unsigned int c = 0; c < dim; ++c)
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              shared_data->gradients(q_point, d_1, c) =
                precomputed_data->inv_jacobian(inv_jac_idx, d_1, c) * div_in *
                JxW;
        }



        template <int dim,
                  int fe_degree,
                  int n_q_points_1d,
                  int n_components_,
                  typename Number>
        DEAL_II_HOST_DEVICE void
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
          submit_symmetric_gradient(
            const SymmetricTensor<2, dim, Number> &sym_grad,
            const int                              q_point)
        {
          AssertIndexRange(q_point, n_q_points_per_cell_range);
          Assert(n_components_ == dim,
                 ExcMessage(
                   "Function submit_symmetric_gradient() only works when "
                   "the number of components and the number of dimensions "
                   "are equal."));
          Assert(precomputed_data->inv_jacobian.size() > 0,
                 ExcMessage("submit_symmetric_gradient() requires precomputed "
                            "inv_jacobian"));
          Assert(precomputed_data->JxW.size() > 0,
                 ExcMessage(
                   "submit_symmetric_gradient() requires precomputed JxW"));

          const int          cell = cell_id(q_point);
          const int          q    = cell_q_point(q_point);
          const unsigned int inv_jac_idx =
            precomputed_data->inv_jacobian_index(cell, q);
          const Number JxW = precomputed_data->JxW_value(cell, q);
          for (unsigned int c = 0; c < dim; ++c)
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                Number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp += precomputed_data->inv_jacobian(inv_jac_idx, d_1, d_2) *
                         sym_grad[c][d_2];
                shared_data->gradients(q_point, d_1, c) = tmp * JxW;
              }
        }
      } // namespace batched
    } // namespace internal
  }   // namespace Portable
} // namespace Copy

#endif
