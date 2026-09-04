#ifndef kernels_portable_evaluation_kernels_view_h
#define kernels_portable_evaluation_kernels_view_h

#include "matrix_free/portable_evaluation_kernels.h"


DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {

    template <int dim, typename Number>
    struct PrecomputedData
    {
      const typename Portable::MatrixFree<dim, Number>::PrecomputedData &data;

      const Custom::Parallel::DoFIndicesView  &dof_indices;
      const Custom::Parallel::CellRangeIdView &cell_range_ids;
    };

    template <typename Number>
    struct SharedDataView
    {
      // 1D unmanaged scratch view -- used for shape_values/co_shape_gradients
      // (staged shared-memory copies of global shape data, no real deal.II
      // SharedData analog) and as SharedViewScratchPad below.
      using ScratchView =
        Kokkos::View<Number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      // Same field names/shapes as real deal.II's own
      // Portable::SharedData<dim, Number> (portable_matrix_free.h): values
      // is (n_q_points, n_components), gradients is (n_q_points, dim,
      // n_components). Only the storage shape is adopted for now -- every
      // caller still hardcodes component 0, n_components_ == 1 is enforced
      // in FEEvaluationView.
      using SharedViewValues =
        Kokkos::View<Number **,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      using SharedViewGradients =
        Kokkos::View<Number ***,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      using SharedViewScratchPad = ScratchView;

      ScratchView shape_values;
      ScratchView shape_gradients;
      ScratchView co_shape_gradients;


      SharedViewValues     values;
      SharedViewGradients  gradients;
      SharedViewScratchPad scratch_pad;
    };

    template <int dim, typename Number>
    struct BatchDataView
    {
      using TeamHandle = Custom::Parallel::TeamHandle;

      TeamHandle team_member;

      const Custom::Parallel::PrecomputedData<dim, Number> &precomputed_data;
      const Custom::Parallel::SharedDataView<Number>       &shared_data;

      const int batch_index;
      const int n_elements_per_batch;
      const int n_elements_in_current_batch;

      const int n_q_points;

      template <typename Functor>
      DEAL_II_HOST_DEVICE void
      for_each_quad_point(const Functor &func) const
      {
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member,
                                                     n_elements_in_current_batch * n_q_points),
                             [&](int &q) { func(q); });

        team_member.team_barrier();
      }

      DEAL_II_HOST_DEVICE unsigned int
      local_q_point_id(const unsigned int cell, const unsigned int q_point) const
      {
        AssertIndexRange(cell, precomputed_data.data.n_cells);
        AssertIndexRange(q_point, static_cast<unsigned int>(n_q_points));

        return (precomputed_data.data.row_start / precomputed_data.data.padding_length + cell) *
                 n_q_points +
               q_point;
      }

      DEAL_II_HOST_DEVICE
      typename Portable::MatrixFree<dim, Number>::point_type &
      get_quadrature_point(const unsigned int cell, const unsigned int q_point) const
      {
        AssertIndexRange(cell, precomputed_data.data.n_cells);
        AssertIndexRange(q_point, static_cast<unsigned int>(n_q_points));

        return precomputed_data.data.q_points(q_point, cell);
      }
    };

    // General-case evaluator, real deal.II's Portable::internal::FEEvaluationImpl
    // (portable_evaluation_kernels.h). Ported verbatim (same dim 1/2/3 branches,
    // same temp/temp1/temp2 subview sizes) -- the only structural difference from
    // FEEvaluationImplTransformToCollocationView is that this one genuinely needs
    // shape_gradients (the direct, non-square dof->quad gradient matrix), not just
    // co_shape_gradients.
    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImplView
    {
      // See FEEvaluationImplTransformToCollocationView::evaluate() for the
      // signature-order rationale.
      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                     n_components,
               const EvaluationFlags::EvaluationFlags evaluation_flag,
               const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        if (evaluation_flag == EvaluationFlags::nothing)
          return;

        const auto &shared_data = data->shared_data;

        // No in-place operation happens in this function
        const auto scratch_for_eval =
          Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, 0));

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shared_data.shape_values,
               shared_data.shape_gradients,
               shared_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            if constexpr (dim == 1)
              {
                const auto temp = Kokkos::subview(
                  shared_data.scratch_pad,
                  Kokkos::make_pair(0, data->n_elements_per_batch * n_q_points_1d));

                if (evaluation_flag & EvaluationFlags::gradients)
                  eval.template gradients<0, true, false, false>(
                    u, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                if (evaluation_flag & EvaluationFlags::values)
                  {
                    eval.template values<0, true, false, false>(u, temp);
                    populate_view<false>(data->team_member,
                                         u,
                                         temp,
                                         data->n_elements_in_current_batch * n_q_points_1d);
                  }
              }
            else if constexpr (dim == 2)
              {
                const int temp_size =
                  data->n_elements_per_batch * (fe_degree + 1) * n_q_points_1d;
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, temp_size));

                // grad x
                if (evaluation_flag & EvaluationFlags::gradients)
                  {
                    eval.template gradients<0, true, false, false>(u, temp);
                    eval.template values<1, true, false, false>(
                      temp, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                  }

                // grad y
                eval.template values<0, true, false, false>(u, temp);
                if (evaluation_flag & EvaluationFlags::gradients)
                  eval.template gradients<1, true, false, false>(
                    temp, Kokkos::subview(grad_u, Kokkos::ALL, 1));

                // val: can use values applied in x
                if (evaluation_flag & EvaluationFlags::values)
                  eval.template values<1, true, false, false>(temp, u);
              }
            else // dim == 3
              {
                const int temp1_size =
                  data->n_elements_per_batch * Utilities::pow(fe_degree + 1, 2) * n_q_points_1d;
                const int temp2_size = data->n_elements_per_batch *
                                       Utilities::pow(n_q_points_1d, 2) * (fe_degree + 1);

                const auto temp1 =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, temp1_size));
                const auto temp2 =
                  Kokkos::subview(shared_data.scratch_pad,
                                  Kokkos::make_pair(temp1_size, temp1_size + temp2_size));

                if (evaluation_flag & EvaluationFlags::gradients)
                  {
                    // grad x
                    eval.template gradients<0, true, false, false>(u, temp1);
                    eval.template values<1, true, false, false>(temp1, temp2);
                    eval.template values<2, true, false, false>(
                      temp2, Kokkos::subview(grad_u, Kokkos::ALL, 0));
                  }

                // grad y
                eval.template values<0, true, false, false>(u, temp1);
                if (evaluation_flag & EvaluationFlags::gradients)
                  {
                    eval.template gradients<1, true, false, false>(temp1, temp2);
                    eval.template values<2, true, false, false>(
                      temp2, Kokkos::subview(grad_u, Kokkos::ALL, 1));
                  }

                // grad z: can use the values applied in x direction stored in temp1
                eval.template values<1, true, false, false>(temp1, temp2);
                if (evaluation_flag & EvaluationFlags::gradients)
                  eval.template gradients<2, true, false, false>(
                    temp2, Kokkos::subview(grad_u, Kokkos::ALL, 2));

                // val: can use the values applied in x & y direction stored in temp2
                if (evaluation_flag & EvaluationFlags::values)
                  eval.template values<2, true, false, false>(temp2, u);
              }
          }
      }

      // See evaluate() above for the signature-order rationale.
      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                     n_components,
                const EvaluationFlags::EvaluationFlags integration_flag,
                const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        if (integration_flag == EvaluationFlags::nothing)
          return;

        const auto &shared_data = data->shared_data;

        const auto scratch_for_eval =
          Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, 0));

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shared_data.shape_values,
               shared_data.shape_gradients,
               shared_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            if constexpr (dim == 1)
              {
                const auto temp = Kokkos::subview(
                  shared_data.scratch_pad,
                  Kokkos::make_pair(0, data->n_elements_per_batch * (fe_degree + 1)));

                if ((integration_flag & EvaluationFlags::values) &&
                    !(integration_flag & EvaluationFlags::gradients))
                  {
                    eval.template values<0, false, false, false>(u, temp);
                    populate_view<false>(data->team_member,
                                         u,
                                         temp,
                                         data->n_elements_in_current_batch * (fe_degree + 1));
                  }
                if (integration_flag & EvaluationFlags::gradients)
                  {
                    if (integration_flag & EvaluationFlags::values)
                      {
                        eval.template values<0, false, false, false>(u, temp);
                        eval.template gradients<0, false, true, false>(
                          Kokkos::subview(grad_u, Kokkos::ALL, 0), temp);
                        populate_view<false>(data->team_member,
                                             u,
                                             temp,
                                             data->n_elements_in_current_batch * (fe_degree + 1));
                      }
                    else
                      eval.template gradients<0, false, false, false>(
                        Kokkos::subview(grad_u, Kokkos::ALL, 0), u);
                  }
              }
            else if constexpr (dim == 2)
              {
                const int temp_size =
                  data->n_elements_per_batch * (fe_degree + 1) * n_q_points_1d;
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, temp_size));

                if ((integration_flag & EvaluationFlags::values) &&
                    !(integration_flag & EvaluationFlags::gradients))
                  {
                    eval.template values<1, false, false, false>(u, temp);
                    eval.template values<0, false, false, false>(temp, u);
                  }
                if (integration_flag & EvaluationFlags::gradients)
                  {
                    eval.template gradients<1, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 1), temp);
                    if (integration_flag & EvaluationFlags::values)
                      eval.template values<1, false, true, false>(u, temp);
                    eval.template values<0, false, false, false>(temp, u);
                    eval.template values<1, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 0), temp);
                    eval.template gradients<0, false, true, false>(temp, u);
                  }
              }
            else // dim == 3
              {
                const int temp1_size =
                  data->n_elements_per_batch * Utilities::pow(n_q_points_1d, 2) * (fe_degree + 1);
                const int temp2_size =
                  data->n_elements_per_batch * Utilities::pow(fe_degree + 1, 2) * n_q_points_1d;

                const auto temp1 =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, temp1_size));
                const auto temp2 =
                  Kokkos::subview(shared_data.scratch_pad,
                                  Kokkos::make_pair(temp1_size, temp1_size + temp2_size));

                if ((integration_flag & EvaluationFlags::values) &&
                    !(integration_flag & EvaluationFlags::gradients))
                  {
                    eval.template values<2, false, false, false>(u, temp1);
                    eval.template values<1, false, false, false>(temp1, temp2);
                    eval.template values<0, false, false, false>(temp2, u);
                  }
                if (integration_flag & EvaluationFlags::gradients)
                  {
                    eval.template gradients<2, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 2), temp1);
                    if (integration_flag & EvaluationFlags::values)
                      eval.template values<2, false, true, false>(u, temp1);
                    eval.template values<1, false, false, false>(temp1, temp2);
                    eval.template values<2, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 1), temp1);
                    eval.template gradients<1, false, true, false>(temp1, temp2);
                    eval.template values<0, false, false, false>(temp2, u);
                    eval.template values<2, false, false, false>(
                      Kokkos::subview(grad_u, Kokkos::ALL, 0), temp1);
                    eval.template values<1, false, false, false>(temp1, temp2);
                    eval.template gradients<0, false, true, false>(temp2, u);
                  }
              }
          }
      }
    };


    // Collocation-space evaluator, real deal.II's own
    // Portable::internal::FEEvaluationImplCollocation -- specialized for
    // Gauss-Lobatto elements where nodal points coincide with quad points,
    // so the "values" operation is the identity and only co_shape_gradients
    // is needed (no shape_values/shape_gradients matrices at all).
    template <int dim, int fe_degree, typename Number>
    struct FEEvaluationImplCollocationView
    {
      // See FEEvaluationImplTransformToCollocationView::evaluate() for the
      // signature-order rationale.
      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                     n_components,
               const EvaluationFlags::EvaluationFlags evaluation_flag,
               const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        // Values are already sitting in shared_data.values (identity), so
        // there's nothing to do unless gradients are requested.
        if (!(evaluation_flag & EvaluationFlags::gradients))
          return;

        const auto &shared_data = data->shared_data;

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         fe_degree + 1,
                                         Number>
          eval(data->team_member,
               typename SharedDataView<Number>::ScratchView(), // no shape_values needed (identity)
               typename SharedDataView<Number>::ScratchView(), // no direct shape_gradients needed
               shared_data.co_shape_gradients,
               typename SharedDataView<Number>::ScratchView(), // no temp needed -- only the
                                                               // fused co_gradients<transpose,
                                                               // add>() below is ever called,
                                                               // and it never touches temp
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            // Broadcast: one scalar u feeds all dim independent gradient
            // components
            eval.template co_gradients<false, false>(u, grad_u);
          }
      }

      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                     n_components,
                const EvaluationFlags::EvaluationFlags integration_flag,
                const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        if (!(integration_flag & EvaluationFlags::gradients))
          return;

        const auto &shared_data = data->shared_data;

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         fe_degree + 1,
                                         Number>
          eval(data->team_member,
               typename SharedDataView<Number>::ScratchView(), // no shape_values needed (identity)
               typename SharedDataView<Number>::ScratchView(), // no direct shape_gradients needed
               shared_data.co_shape_gradients,
               typename SharedDataView<Number>::ScratchView(), // no temp needed
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            if (integration_flag & EvaluationFlags::values)
              eval.template co_gradients<true, true>(grad_u, u);
            else
              eval.template co_gradients<true, false>(grad_u, u);
          }
      }
    };



    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImplTransformToCollocationView
    {
      DEAL_II_HOST_DEVICE static void
      evaluate(const unsigned int                     n_components,
               const EvaluationFlags::EvaluationFlags evaluation_flag,
               const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shared_data = data->shared_data;

        const int batch_capacity = data->n_elements_per_batch * data->n_q_points;

        // No in-place operation happens in this function
        const auto scratch_for_eval =
          Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, 0));

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shared_data.shape_values,
               typename SharedDataView<Number>::ScratchView(), // no gradients
               shared_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            if constexpr (dim == 1)
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));

                eval.template values<0, true, false>(u, temp);

                populate_view<false>(data->team_member,
                                     u,
                                     temp,
                                     data->n_elements_in_current_batch * data->n_q_points);
              }
            else if constexpr (dim == 2)
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));

                eval.template values<0, true, false>(u, temp);
                eval.template values<1, true, false>(temp, u);
              }
            else // dim == 3
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));

                eval.template values<0, true, false>(u, temp);
                eval.template values<1, true, false>(temp, u);
                eval.template values<2, true, false>(u, temp);

                populate_view<false>(data->team_member,
                                     u,
                                     temp,
                                     data->n_elements_in_current_batch * data->n_q_points);
              }

            if (evaluation_flag & EvaluationFlags::gradients)
              eval.template co_gradients<false, false>(u, grad_u);
          }
      }

      DEAL_II_HOST_DEVICE static void
      integrate(const unsigned int                     n_components,
                const EvaluationFlags::EvaluationFlags integration_flag,
                const BatchDataView<dim, Number>      *data)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shared_data = data->shared_data;

        const int batch_capacity = data->n_elements_per_batch * data->n_q_points;

        // No in-place operation happens in this function
        const auto scratch_for_eval =
          Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, 0));


        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shared_data.shape_values,
               typename SharedDataView<Number>::ScratchView(), // no gradients
               shared_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch);

        for (unsigned int c = 0; c < n_components; ++c)
          {
            const auto u      = Kokkos::subview(shared_data.values, Kokkos::ALL, c);
            const auto grad_u = Kokkos::subview(shared_data.gradients, Kokkos::ALL, Kokkos::ALL, c);

            if (integration_flag & EvaluationFlags::gradients)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<true, true>(grad_u, u);
                else
                  eval.template co_gradients<true, false>(grad_u, u);
              }

            if constexpr (dim == 1)
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));

                eval.template values<0, false, false>(u, temp);

                populate_view<false>(data->team_member,
                                     u,
                                     temp,
                                     data->n_elements_in_current_batch * (fe_degree + 1));
              }
            else if constexpr (dim == 2)
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));
                eval.template values<1, false, false>(u, temp);
                eval.template values<0, false, false>(temp, u);
              }
            else // dim == 3
              {
                const auto temp =
                  Kokkos::subview(shared_data.scratch_pad, Kokkos::make_pair(0, batch_capacity));

                eval.template values<2, false, false>(u, temp);
                eval.template values<1, false, false>(temp, u);
                eval.template values<0, false, false>(u, temp);

                populate_view<false>(data->team_member,
                                     u,
                                     temp,
                                     data->n_elements_in_current_batch *
                                       Utilities::pow(fe_degree + 1, dim));
              }
          }
      }

      template <bool add = false, typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE static void
      evaluate_gradients_and_multiply_symmetric_tensor(const BatchDataView<dim, Number> *data,
                                                       const DeviceView<Number>         &d_G,
                                                       const ViewTypeIn                  in,
                                                       ViewTypeOut                       out)
      {
        static_assert(dim >= 1, "dim must be at least 1");

        const auto &co_shape_gradients = data->shared_data.co_shape_gradients;
        const auto &cell_range_ids     = data->precomputed_data.cell_range_ids;

        constexpr int n_q_points                 = Utilities::pow(n_q_points_1d, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_q_points_1d, dim - 1);

        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(data->team_member,
                                  data->n_elements_in_current_batch * co_dimension_size),
          [&](const int tid)
            {
              const int elmnt_idx = tid / co_dimension_size;
              const int reminder  = tid % co_dimension_size;

              unsigned int global_cell_index =
                data->batch_index * data->n_elements_per_batch + elmnt_idx;
              if (cell_range_ids.size() > 0)
                global_cell_index = cell_range_ids(global_cell_index);

              const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * n_q_points;

              Kokkos::Array<int, dim - 1> idx_d[dim], stride_d[dim];
              Number                      reg[dim][n_q_points_1d];

              for (int d = 0; d < dim - 1; ++d)
                {
                  stride_d[d] = Utilities::pow(n_q_points_1d, d);
                  idx_d[d]    = (reminder / stride_d[d]) % n_q_points_1d;
                }

              for (int n = 0; n < n_q_points_1d; ++n)
                {
                  for (int d = 0; d < dim - 1; ++d)
                    reg[d][n] = co_shape_gradients(n * n_q_points_1d + idx_d[d]);
                  reg[dim - 1][n] = in(elmnt_idx * n_q_points + reminder + n * co_dimension_size);
                }

              for (int last = 0; last < n_q_points_1d; ++last)
                {
                  const int q_point = reminder + last * co_dimension_size;

                  Number res[dim];
                  for (int d = 0; d < dim - 1; ++d)
                    {
                      const int q_point_base = q_point - idx_d[d] * stride_d[d];
                      const int in_base      = elmnt_idx * n_q_points + q_point_base;

                      Number res_d = 0;
                      for (int n = 0; n < n_q_points_1d; ++n)
                        res_d += reg[d][n] * in(in_base + n * stride_d[d]);
                      res[d] = res_d;
                    }
                  {
                    Number res_d = 0;
                    for (int n = 0; n < n_q_points_1d; ++n)
                      res_d += co_shape_gradients(n * n_q_points_1d + last) * reg[dim - 1][n];
                    res[dim - 1] = res_d;
                  }

                  Number G[dim][dim];
                  int    component_index = 0;
                  for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = d1; d2 < dim; ++d2)
                      {
                        const Number value =
                          d_G[cell_g_offset + component_index * n_q_points + q_point];
                        G[d1][d2] = value;
                        G[d2][d1] = value;
                        ++component_index;
                      }

                  for (int d1 = 0; d1 < dim; ++d1)
                    {
                      Number value_out = 0;
                      for (int d2 = 0; d2 < dim; ++d2)
                        value_out += G[d1][d2] * res[d2];

                      if constexpr (add)
                        out(elmnt_idx * n_q_points + q_point, d1) += value_out;
                      else
                        out(elmnt_idx * n_q_points + q_point, d1) = value_out;
                    }
                }
            });

        data->team_member.team_barrier();
      }
    };


    // n_components-aware, matching real deal.II's own Portable::FEEvaluation::
    // read_dof_values() component-blocked local numbering exactly: component c's
    // dofs occupy local indices [c * n_dofs_total, (c + 1) * n_dofs_total) --
    // "i + tensor_dofs_per_component * c" in deal.II's own naming. `values` is
    // the full 2D (batched-flat-index, component) SharedViewValues, not a
    // single-component subview.
    template <int dim,
              int n_dofs_1d,
              int n_components,
              typename Number,
              typename ViewTypeValues,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeValues>::value>>
    DEAL_II_HOST_DEVICE inline void
    read_dof_values(const TeamHandle         &team_member,
                    const DoFIndicesView     &dof_indices,
                    const CellRangeIdView    &cell_range_ids,
                    const DeviceView<Number> &d_in,
                    ViewTypeValues            values,
                    const int                 batch_index,
                    const int                 n_elements_per_batch,
                    const int                 n_elements_in_current_batch)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member,
                                                   n_elements_in_current_batch * n_dofs_total),
                           [&](const int tid)
                             {
                               const int elmnt_idx = tid / n_dofs_total;
                               const int local_idx = tid % n_dofs_total;

                               unsigned int global_cell_index =
                                 batch_index * n_elements_per_batch + elmnt_idx;
                               if (cell_range_ids.size() > 0)
                                 global_cell_index = cell_range_ids(global_cell_index);

                               for (int c = 0; c < n_components; ++c)
                                 {
                                   const unsigned int dof_index =
                                     dof_indices(local_idx + n_dofs_total * c, global_cell_index);

                                   if (dof_index == numbers::invalid_unsigned_int)
                                     values(tid, c) = 0;
                                   else
                                     values(tid, c) = d_in[dof_index];
                                 }
                             });

      team_member.team_barrier();
    }


    // See read_dof_values() above for the component-blocked local numbering.
    template <int dim,
              int n_dofs_1d,
              int n_components,
              typename Number,
              typename ViewTypeValues,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeValues>::value>>
    DEAL_II_HOST_DEVICE inline void
    distribute_local_to_global(const TeamHandle      &team_member,
                               const DoFIndicesView  &dof_indices,
                               const CellRangeIdView &cell_range_ids,
                               const ViewTypeValues   values,
                               DeviceView<Number>     d_out,
                               const int              batch_index,
                               const int              n_elements_per_batch,
                               const int              n_elements_in_current_batch)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member,
                                                   n_elements_in_current_batch * n_dofs_total),
                           [&](const int tid)
                             {
                               const int elmnt_idx = tid / n_dofs_total;
                               const int local_idx = tid % n_dofs_total;

                               unsigned int global_cell_index =
                                 batch_index * n_elements_per_batch + elmnt_idx;
                               if (cell_range_ids.size() > 0)
                                 global_cell_index = cell_range_ids(global_cell_index);

                               for (int c = 0; c < n_components; ++c)
                                 {
                                   const unsigned int dof_index =
                                     dof_indices(local_idx + n_dofs_total * c, global_cell_index);

                                   if (dof_index != numbers::invalid_unsigned_int)
                                     Kokkos::atomic_add(&d_out[dof_index], values(tid, c));
                                 }
                             });

      team_member.team_barrier();
    }

  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
