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
    struct ShapeDataView
    {
      using ScratchView =
        Kokkos::View<Number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      using GradientsView =
        Kokkos::View<Number **,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      ScratchView shape_values;
      ScratchView co_shape_gradients;

      ScratchView   values;
      GradientsView gradients;
      ScratchView   scratch_pad;
    };

    template <int dim, typename Number>
    struct BatchDataView
    {
      using TeamHandle = Custom::Parallel::TeamHandle;

      TeamHandle team_member;

      const Custom::Parallel::PrecomputedData<dim, Number> &precomputed_data;
      const Custom::Parallel::ShapeDataView<Number>        &shape_data;

      const int batch_index;
      const int n_elements_per_batch;
      const int n_elements_in_current_batch;
      const int thread_id;
      const int block_size;

      const int n_q_points_per_batch;

      const int n_q_points;

      template <typename Functor>
      DEAL_II_HOST_DEVICE void
      for_each_quad_point(const Functor &func) const
      {
        const int n_points = n_elements_in_current_batch * n_q_points;

        for (int tid = thread_id; tid < n_points; tid += block_size)
          func(tid);

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

    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImplTransformToCollocationView
    {
      DEAL_II_HOST_DEVICE static void
      evaluate(const BatchDataView<dim, Number>      *data,
               const EvaluationFlags::EvaluationFlags evaluation_flag)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shape_data = data->shape_data;

        const auto scratch_for_eval =
          Kokkos::subview(shape_data.scratch_pad, Kokkos::make_pair(0, data->n_q_points_per_batch));

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shape_data.shape_values,
               typename ShapeDataView<Number>::ScratchView(), // no gradients
               shape_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch,
               data->thread_id,
               data->block_size);

        eval.template values<0, true, false, true>(shape_data.values, shape_data.values);
        if constexpr (dim > 1)
          eval.template values<1, true, false, true>(shape_data.values, shape_data.values);
        if constexpr (dim > 2)
          eval.template values<2, true, false, true>(shape_data.values, shape_data.values);

        if (evaluation_flag & EvaluationFlags::gradients)
          {
            eval.template co_gradients<0, true, false, false>(
              shape_data.values, Kokkos::subview(shape_data.gradients, Kokkos::ALL, 0));
            if constexpr (dim > 1)
              eval.template co_gradients<1, true, false, false>(
                shape_data.values, Kokkos::subview(shape_data.gradients, Kokkos::ALL, 1));
            if constexpr (dim > 2)
              eval.template co_gradients<2, true, false, false>(
                shape_data.values, Kokkos::subview(shape_data.gradients, Kokkos::ALL, 2));
          }
      }

      DEAL_II_HOST_DEVICE static void
      integrate(const BatchDataView<dim, Number>      *data,
                const EvaluationFlags::EvaluationFlags integration_flag)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shape_data = data->shape_data;

        const auto scratch_for_eval =
          Kokkos::subview(shape_data.scratch_pad, Kokkos::make_pair(0, data->n_q_points_per_batch));

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          eval(data->team_member,
               shape_data.shape_values,
               typename ShapeDataView<Number>::ScratchView(), // no gradients
               shape_data.co_shape_gradients,
               scratch_for_eval,
               data->n_elements_in_current_batch,
               data->thread_id,
               data->block_size);

        if (integration_flag & EvaluationFlags::gradients)
          {
            if constexpr (dim == 1)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<0, false, true, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 0), shape_data.values);
                else
                  eval.template co_gradients<0, false, false, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 0), shape_data.values);
              }
            else if constexpr (dim == 2)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<1, false, true, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 1), shape_data.values);
                else
                  eval.template co_gradients<1, false, false, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 1), shape_data.values);
                eval.template co_gradients<0, false, true, false>(
                  Kokkos::subview(shape_data.gradients, Kokkos::ALL, 0), shape_data.values);
              }
            else if constexpr (dim == 3)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<2, false, true, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 2), shape_data.values);
                else
                  eval.template co_gradients<2, false, false, false>(
                    Kokkos::subview(shape_data.gradients, Kokkos::ALL, 2), shape_data.values);
                eval.template co_gradients<1, false, true, false>(
                  Kokkos::subview(shape_data.gradients, Kokkos::ALL, 1), shape_data.values);
                eval.template co_gradients<0, false, true, false>(
                  Kokkos::subview(shape_data.gradients, Kokkos::ALL, 0), shape_data.values);
              }
            else
              Assert(false, ExcMessage("dim must not exceed 3!"));
          }

        if constexpr (dim > 2)
          eval.template values<2, false, false, true>(shape_data.values, shape_data.values);
        if constexpr (dim > 1)
          eval.template values<1, false, false, true>(shape_data.values, shape_data.values);
        eval.template values<0, false, false, true>(shape_data.values, shape_data.values);
      }

      template <typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE static void
      evaluate_values(const BatchDataView<dim, Number> *data, const ViewTypeIn in, ViewTypeOut out)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shape_data = data->shape_data;

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          evaluator(data->team_member,
                    shape_data.shape_values,
                    typename ShapeDataView<Number>::ScratchView(), // no gradients
                    typename ShapeDataView<Number>::ScratchView(), // no co-gradients
                    typename ShapeDataView<Number>::ScratchView(), // temp unused (no in_place here)
                    data->n_elements_in_current_batch,
                    data->thread_id,
                    data->block_size);

        if constexpr (dim == 1)
          {
            evaluator.template values<0, true, false>(in, out);
          }
        else if constexpr (dim == 2)
          {
            const auto scratch = Kokkos::subview(shape_data.scratch_pad,
                                                 Kokkos::make_pair(0, data->n_q_points_per_batch));

            evaluator.template values<0, true, false>(in, scratch);
            evaluator.template values<1, true, false>(scratch, out);
          }
        else // dim == 3
          {
            const auto scratch0 = Kokkos::subview(shape_data.scratch_pad,
                                                  Kokkos::make_pair(0, data->n_q_points_per_batch));
            const auto scratch1 =
              Kokkos::subview(shape_data.scratch_pad,
                              Kokkos::make_pair(data->n_q_points_per_batch,
                                                2 * data->n_q_points_per_batch));

            evaluator.template values<0, true, false>(in, scratch0);
            evaluator.template values<1, true, false>(scratch0, scratch1);
            evaluator.template values<2, true, false>(scratch1, out);
          }
      }

      template <typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE static void
      integrate_values(const BatchDataView<dim, Number> *data, const ViewTypeIn in, ViewTypeOut out)
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const auto &shape_data = data->shape_data;

        const EvaluatorTensorProductView<EvaluatorVariant::evaluate_general,
                                         dim,
                                         fe_degree + 1,
                                         n_q_points_1d,
                                         Number>
          evaluator(data->team_member,
                    shape_data.shape_values,
                    typename ShapeDataView<Number>::ScratchView(), // no gradients
                    typename ShapeDataView<Number>::ScratchView(), // no co-gradients
                    typename ShapeDataView<Number>::ScratchView(), // temp unused (no in_place here)
                    data->n_elements_in_current_batch,
                    data->thread_id,
                    data->block_size);

        if constexpr (dim == 1)
          {
            evaluator.template values<0, false, false>(in, out);
          }
        else if constexpr (dim == 2)
          {
            const auto scratch = Kokkos::subview(shape_data.scratch_pad,
                                                 Kokkos::make_pair(0, data->n_q_points_per_batch));

            evaluator.template values<1, false, false>(in, scratch);
            evaluator.template values<0, false, false>(scratch, out);
          }
        else // dim == 3
          {
            const auto scratch0 = Kokkos::subview(shape_data.scratch_pad,
                                                  Kokkos::make_pair(0, data->n_q_points_per_batch));
            const auto scratch1 =
              Kokkos::subview(shape_data.scratch_pad,
                              Kokkos::make_pair(data->n_q_points_per_batch,
                                                2 * data->n_q_points_per_batch));

            evaluator.template values<2, false, false>(in, scratch0);
            evaluator.template values<1, false, false>(scratch0, scratch1);
            evaluator.template values<0, false, false>(scratch1, out);
          }
      }

      template <bool add = false, typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE static void
      evaluate_gradients(const BatchDataView<dim, Number> *data,
                         const ViewTypeIn                  in,
                         ViewTypeOut                       out)
      {
        static_assert(dim >= 1, "dim must be at least 1");

        const auto &co_shape_gradients = data->shape_data.co_shape_gradients;

        constexpr int n_q_points        = Utilities::pow(n_q_points_1d, dim);
        constexpr int co_dimension_size = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = data->thread_id; tid < data->n_elements_in_current_batch * co_dimension_size;
             tid += data->block_size)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            // Sized [dim], not [dim - 1], purely to keep the array valid at
            // dim == 1 (where the d-loops below never execute) without a
            // separate dim == 1 code path -- the extra slot is never touched.
            int    idx_d[dim], stride_d[dim];
            Number reg[dim][n_q_points_1d];

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

                Number result[dim];
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int q_point_base = q_point - idx_d[d] * stride_d[d];
                    const int in_base      = elmnt_idx * n_q_points + q_point_base;

                    Number res_d = 0;
                    for (int n = 0; n < n_q_points_1d; ++n)
                      res_d += reg[d][n] * in(in_base + n * stride_d[d]);
                    result[d] = res_d;
                  }
                {
                  Number res_d = 0;
                  for (int n = 0; n < n_q_points_1d; ++n)
                    res_d += co_shape_gradients(n * n_q_points_1d + last) * reg[dim - 1][n];
                  result[dim - 1] = res_d;
                }

                for (int d = 0; d < dim; ++d)
                  {
                    if constexpr (add)
                      out(elmnt_idx * n_q_points + q_point, d) += result[d];
                    else
                      out(elmnt_idx * n_q_points + q_point, d) = result[d];
                  }
              }
          }

        data->team_member.team_barrier();
      }

      template <bool add = false, typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE static void
      evaluate_gradients_and_multiply_symmetric_tensor(const BatchDataView<dim, Number> *data,
                                                       const DeviceView<Number>         &d_G,
                                                       const ViewTypeIn                  in,
                                                       ViewTypeOut                       out)
      {
        static_assert(dim >= 1, "dim must be at least 1");

        const auto &co_shape_gradients = data->shape_data.co_shape_gradients;
        const auto &cell_range_ids     = data->precomputed_data.cell_range_ids;

        constexpr int n_q_points                 = Utilities::pow(n_q_points_1d, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = data->thread_id; tid < data->n_elements_in_current_batch * co_dimension_size;
             tid += data->block_size)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            unsigned int global_cell_index =
              data->batch_index * data->n_elements_per_batch + elmnt_idx;
            if (cell_range_ids.size() > 0)
              global_cell_index = cell_range_ids(global_cell_index);

            const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * n_q_points;

            // Sized [dim], see evaluate_gradients() above.
            int    idx_d[dim], stride_d[dim];
            Number reg[dim][n_q_points_1d];

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
          }

        data->team_member.team_barrier();
      }

      template <bool add = false, typename ViewTypeGrad, typename ViewTypeVal>
      DEAL_II_HOST_DEVICE static void
      integrate_gradients(const BatchDataView<dim, Number> *data,
                          const ViewTypeGrad                gradients,
                          ViewTypeVal                       values)
      {
        static_assert(dim >= 1, "dim must be at least 1");

        const auto &co_shape_gradients = data->shape_data.co_shape_gradients;

        constexpr int n_q_points        = Utilities::pow(n_q_points_1d, dim);
        constexpr int co_dimension_size = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = data->thread_id; tid < data->n_elements_in_current_batch * co_dimension_size;
             tid += data->block_size)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            // Sized [dim], see evaluate_gradients() above.
            int    idx[dim], stride_d[dim];
            Number reg[dim][n_q_points_1d];

            for (int d = 0; d < dim - 1; ++d)
              {
                stride_d[d] = Utilities::pow(n_q_points_1d, d);
                idx[d]      = (reminder / stride_d[d]) % n_q_points_1d;
                for (int n = 0; n < n_q_points_1d; ++n)
                  reg[d][n] = co_shape_gradients(idx[d] * n_q_points_1d + n);
              }
            for (int n = 0; n < n_q_points_1d; ++n)
              reg[dim - 1][n] =
                gradients(elmnt_idx * n_q_points + reminder + n * co_dimension_size, dim - 1);

            for (int last = 0; last < n_q_points_1d; ++last)
              {
                const int q_point = reminder + last * co_dimension_size;

                Number result = 0;
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int point_base = q_point - idx[d] * stride_d[d];
                    const int grad_row   = elmnt_idx * n_q_points + point_base;

                    for (int n = 0; n < n_q_points_1d; ++n)
                      result += gradients(grad_row + n * stride_d[d], d) * reg[d][n];
                  }
                for (int n = 0; n < n_q_points_1d; ++n)
                  result += reg[dim - 1][n] * co_shape_gradients(last * n_q_points_1d + n);

                if constexpr (add)
                  values(elmnt_idx * n_q_points + q_point) += result;
                else
                  values(elmnt_idx * n_q_points + q_point) = result;
              }
          }

        data->team_member.team_barrier();
      }
    };


    template <int dim,
              int n_dofs_1d,
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
                    const int                 n_elements_in_current_batch,
                    const int                 thread_id,
                    const int                 block_size)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_dofs_total; tid += block_size)
        {
          const int elmnt_idx = tid / n_dofs_total;
          const int local_idx = tid % n_dofs_total;

          unsigned int global_cell_index = batch_index * n_elements_per_batch + elmnt_idx;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

          if (dof_index == numbers::invalid_unsigned_int)
            values(tid) = 0;
          else
            values(tid) = d_in[dof_index];
        }

      team_member.team_barrier();
    }


    template <int dim,
              int n_dofs_1d,
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
                               const int              n_elements_in_current_batch,
                               const int              thread_id,
                               const int              block_size)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_dofs_total; tid += block_size)
        {
          const int elmnt_idx = tid / n_dofs_total;
          const int local_idx = tid % n_dofs_total;

          unsigned int global_cell_index = batch_index * n_elements_per_batch + elmnt_idx;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

          if (dof_index != numbers::invalid_unsigned_int)
            Kokkos::atomic_add(&d_out[dof_index], values(tid));
        }

      team_member.team_barrier();
    }

  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
