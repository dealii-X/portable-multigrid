#ifndef kernels_portable_evaluation_kernels_h
#define kernels_portable_evaluation_kernels_h

#include <deal.II/matrix_free/evaluation_flags.h>

#include "matrix_free/portable_tensor_product_kernels.h"


DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {

    template <int dim, typename Number>
    struct BatchData
    {
      using TeamHandle = Custom::Parallel::TeamHandle;

      TeamHandle team_member;

      const Number *shape_values;
      const Number *co_shape_gradients;

      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space>            &G_tensor;
      const Kokkos::View<Number **[dim][dim], MemorySpace::Default::kokkos_space> &inv_jacobian;
      const Kokkos::View<Number **, MemorySpace::Default::kokkos_space>           &JxW;

      const Custom::Parallel::DoFIndicesView &dof_indices;

      const int batchIdx;
      const int nelmtPerBatch;
      const int c_nelmtPerBatch;
      const int threadIdx;
      const int blockSize;

      const Custom::Parallel::CellRangeIdView &cell_range_ids;

      Number *values;
      Number *gradients;
      Number *scratch;

      const int quad_size_per_batch;

      template <typename Functor>
      DEAL_II_HOST_DEVICE void
      for_each_quad_point(const Functor &func) const
      {
        const int nq_total = quad_size_per_batch / nelmtPerBatch;
        const int n_points = c_nelmtPerBatch * nq_total;

        for (int tid = threadIdx; tid < n_points; tid += blockSize)
          func(tid);

        team_member.team_barrier();
      }
    };


    template <int dim, int n_dofs_1d, typename Number>
    DEAL_II_HOST_DEVICE inline void
    read_dof_values(const TeamHandle         &team_member,
                    const DoFIndicesView     &dof_indices,
                    const CellRangeIdView    &cell_range_ids,
                    const DeviceView<Number> &d_in,
                    Number                   *values,
                    const int                 batchIdx,
                    const int                 nelmtPerBatch,
                    const int                 c_nelmtPerBatch,
                    const int                 threadIdx,
                    const int                 blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_total; tid += blockSize)
        {
          const int elmnt_idx = tid / n_dofs_total;
          const int local_idx = tid % n_dofs_total;

          unsigned int global_cell_index = batchIdx * nelmtPerBatch + elmnt_idx;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

          if (dof_index == numbers::invalid_unsigned_int)
            values[tid] = 0;
          else
            values[tid] = d_in[dof_index];
        }

      team_member.team_barrier();
    }


    template <int dim, int n_dofs_1d, typename Number>
    DEAL_II_HOST_DEVICE inline void
    distribute_local_to_global(const TeamHandle      &team_member,
                               const DoFIndicesView  &dof_indices,
                               const CellRangeIdView &cell_range_ids,
                               const Number          *values,
                               DeviceView<Number>     d_out,
                               const int              batchIdx,
                               const int              nelmtPerBatch,
                               const int              c_nelmtPerBatch,
                               const int              threadIdx,
                               const int              blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int n_dofs_total = Utilities::pow(n_dofs_1d, dim);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * n_dofs_total; tid += blockSize)
        {
          const int elmnt_idx = tid / n_dofs_total;
          const int local_idx = tid % n_dofs_total;

          unsigned int global_cell_index = batchIdx * nelmtPerBatch + elmnt_idx;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);


          const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

          if (dof_index != numbers::invalid_unsigned_int)
            Kokkos::atomic_add(&d_out[dof_index], values[tid]);
        }

      team_member.team_barrier();
    }


    template <int dim, int fe_degree, int n_q_points_1d, typename Number>
    struct FEEvaluationImplTransformToCollocation
    {
    public:
      DEAL_II_HOST_DEVICE
      FEEvaluationImplTransformToCollocation(const TeamHandle &team_member,
                                             const Number     *shape_values,
                                             const Number     *shape_gradient_collocation,
                                             const int         nelmtPerBatch,
                                             const int         c_nelmtPerBatch,
                                             const int         batchIdx,
                                             const int         threadIdx,
                                             const int         blockSize)
        : team_member(team_member)
        , shape_values(shape_values)
        , shape_gradient_collocation(shape_gradient_collocation)
        , nelmtPerBatch(nelmtPerBatch)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , batchIdx(batchIdx)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
        , quad_size_per_batch(Utilities::pow(n_q_points_1d, dim) * nelmtPerBatch)
      {}



      DEAL_II_HOST_DEVICE void
      evaluate(Number                                *values,
               Number                                *gradients,
               Number                                *scratch,
               const EvaluationFlags::EvaluationFlags evaluation_flag) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                     dim,
                                     fe_degree + 1,
                                     n_q_points_1d,
                                     Number>
          eval(team_member,
               shape_values,
               nullptr, // no gradients
               shape_gradient_collocation,
               scratch,
               c_nelmtPerBatch,
               threadIdx,
               blockSize);

        eval.template values<0, true, false, true>(values, values);
        if constexpr (dim > 1)
          eval.template values<1, true, false, true>(values, values);
        if constexpr (dim > 2)
          eval.template values<2, true, false, true>(values, values);

        if (evaluation_flag & EvaluationFlags::gradients)
          {
            eval.template co_gradients<0, true, false, false>(values, gradients);
            if constexpr (dim > 1)
              eval.template co_gradients<1, true, false, false>(values,
                                                                gradients + quad_size_per_batch);
            if constexpr (dim > 2)
              eval.template co_gradients<2, true, false, false>(values,
                                                                gradients +
                                                                  2 * quad_size_per_batch);
          }
      }

      DEAL_II_HOST_DEVICE void
      integrate(Number                                *values,
                Number                                *gradients,
                Number                                *scratch,
                const EvaluationFlags::EvaluationFlags integration_flag) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                     dim,
                                     fe_degree + 1,
                                     n_q_points_1d,
                                     Number>
          eval(team_member,
               shape_values,
               nullptr, // no gradients
               shape_gradient_collocation,
               scratch,
               c_nelmtPerBatch,
               threadIdx,
               blockSize);

        if (integration_flag & EvaluationFlags::gradients)
          {
            if constexpr (dim == 1)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<0, false, true, false>(gradients, values);
                else
                  eval.template co_gradients<0, false, false, false>(gradients, values);
              }
            else if constexpr (dim == 2)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<1, false, true, false>(gradients + quad_size_per_batch,
                                                                    values);
                else
                  eval.template co_gradients<1, false, false, false>(gradients +
                                                                       quad_size_per_batch,
                                                                     values);
                eval.template co_gradients<0, false, true, false>(gradients, values);
              }
            else if constexpr (dim == 3)
              {
                if (integration_flag & EvaluationFlags::values)
                  eval.template co_gradients<2, false, true, false>(gradients +
                                                                      2 * quad_size_per_batch,
                                                                    values);
                else
                  eval.template co_gradients<2, false, false, false>(gradients +
                                                                       2 * quad_size_per_batch,
                                                                     values);
                eval.template co_gradients<1, false, true, false>(gradients + quad_size_per_batch,
                                                                  values);
                eval.template co_gradients<0, false, true, false>(gradients, values);
              }
            else
              Assert(false, ExcMessage("dim must not exceed 3!"));
          }

        if constexpr (dim > 2)
          eval.template values<2, false, false, true>(values, values);
        if constexpr (dim > 1)
          eval.template values<1, false, false, true>(values, values);
        eval.template values<0, false, false, true>(values, values);
      }

      DEAL_II_HOST_DEVICE void
      evaluate_values(Number *in, Number *out, Number *scratch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                     dim,
                                     fe_degree + 1,
                                     n_q_points_1d,
                                     Number>
          evaluator(team_member,
                    shape_values,
                    nullptr, // no gradients
                    nullptr, // no co-gradients
                    nullptr, // no temp
                    c_nelmtPerBatch,
                    threadIdx,
                    blockSize);

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

      DEAL_II_HOST_DEVICE void
      integrate_values(Number *in, Number *out, Number *scratch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                     dim,
                                     fe_degree + 1,
                                     n_q_points_1d,
                                     Number>
          evaluator(team_member,
                    shape_values,
                    nullptr, // no gradients
                    nullptr, // no co-gradients
                    nullptr, // no temp
                    c_nelmtPerBatch,
                    threadIdx,
                    blockSize);

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

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      evaluate_gradients(const Number *in, Number *out) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int n_q_points        = Utilities::pow(n_q_points_1d, dim);
        constexpr int co_dimension_size = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            int    idx_d[dim - 1], stride_d[dim - 1];
            Number reg[dim][n_q_points_1d];

            for (int d = 0; d < dim - 1; ++d)
              {
                stride_d[d] = Utilities::pow(n_q_points_1d, d);
                idx_d[d]    = (reminder / stride_d[d]) % n_q_points_1d;
              }

            for (int n = 0; n < n_q_points_1d; ++n)
              {
                for (int d = 0; d < dim - 1; ++d)
                  reg[d][n] = shape_gradient_collocation[n * n_q_points_1d + idx_d[d]];

                reg[dim - 1][n] = in[elmnt_idx * n_q_points + reminder + n * co_dimension_size];
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
                      res_d += reg[d][n] * in[in_base + n * stride_d[d]];
                    result[d] = res_d;
                  }
                {
                  Number res_d = 0;
                  for (int n = 0; n < n_q_points_1d; ++n)
                    res_d += shape_gradient_collocation[n * n_q_points_1d + last] * reg[dim - 1][n];
                  result[dim - 1] = res_d;
                }

                for (int d = 0; d < dim; ++d)
                  {
                    if constexpr (add)
                      out[d * quad_size_per_batch + elmnt_idx * n_q_points + q_point] += result[d];
                    else
                      out[d * quad_size_per_batch + elmnt_idx * n_q_points + q_point] = result[d];
                  }
              }
          }

        team_member.team_barrier();
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      evaluate_gradients_and_multiply_symmetric_tensor(const DeviceView<Number> &d_G,
                                                       const CellRangeIdView    &cell_range_ids,
                                                       const Number             *in,
                                                       Number                   *out) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int n_q_points                 = Utilities::pow(n_q_points_1d, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            unsigned int global_cell_index = batchIdx * nelmtPerBatch + elmnt_idx;
            if (cell_range_ids.size() > 0)
              global_cell_index = cell_range_ids(global_cell_index);

            const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * n_q_points;

            int    idx_d[dim - 1], stride_d[dim - 1];
            Number reg[dim][n_q_points_1d];

            for (int d = 0; d < dim - 1; ++d)
              {
                stride_d[d] = Utilities::pow(n_q_points_1d, d);
                idx_d[d]    = (reminder / stride_d[d]) % n_q_points_1d;
              }

            for (int n = 0; n < n_q_points_1d; ++n)
              {
                for (int d = 0; d < dim - 1; ++d)
                  reg[d][n] = shape_gradient_collocation[n * n_q_points_1d + idx_d[d]];
                reg[dim - 1][n] = in[elmnt_idx * n_q_points + reminder + n * co_dimension_size];
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
                      res_d += reg[d][n] * in[in_base + n * stride_d[d]];
                    res[d] = res_d;
                  }
                {
                  Number res_d = 0;
                  for (int n = 0; n < n_q_points_1d; ++n)
                    res_d += shape_gradient_collocation[n * n_q_points_1d + last] * reg[dim - 1][n];
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
                      out[d1 * quad_size_per_batch + elmnt_idx * n_q_points + q_point] += value_out;
                    else
                      out[d1 * quad_size_per_batch + elmnt_idx * n_q_points + q_point] = value_out;
                  }
              }
          }

        team_member.team_barrier();
      }

      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      integrate_gradients(const Number *gradients, Number *values) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int n_q_points        = Utilities::pow(n_q_points_1d, dim);
        constexpr int co_dimension_size = Utilities::pow(n_q_points_1d, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            int    idx[dim - 1], stride_d[dim - 1];
            Number reg[dim][n_q_points_1d];

            for (int d = 0; d < dim - 1; ++d)
              {
                stride_d[d] = Utilities::pow(n_q_points_1d, d);
                idx[d]      = (reminder / stride_d[d]) % n_q_points_1d;
                for (int n = 0; n < n_q_points_1d; ++n)
                  reg[d][n] = shape_gradient_collocation[idx[d] * n_q_points_1d + n];
              }
            for (int n = 0; n < n_q_points_1d; ++n)
              reg[dim - 1][n] = gradients[(dim - 1) * quad_size_per_batch + elmnt_idx * n_q_points +
                                          reminder + n * co_dimension_size];

            for (int last = 0; last < n_q_points_1d; ++last)
              {
                const int q_point = reminder + last * co_dimension_size;

                Number result = 0;
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int point_base = q_point - idx[d] * stride_d[d];
                    const int grad_base =
                      d * quad_size_per_batch + elmnt_idx * n_q_points + point_base;

                    for (int n = 0; n < n_q_points_1d; ++n)
                      result += gradients[grad_base + n * stride_d[d]] * reg[d][n];
                  }
                for (int n = 0; n < n_q_points_1d; ++n)
                  result += reg[dim - 1][n] * shape_gradient_collocation[last * n_q_points_1d + n];

                if constexpr (add)
                  values[elmnt_idx * n_q_points + q_point] += result;
                else
                  values[elmnt_idx * n_q_points + q_point] = result;
              }
          }

        team_member.team_barrier();
      }

    private:
      const TeamHandle &team_member;
      const Number     *shape_values;
      const Number     *shape_gradient_collocation;
      const int         nelmtPerBatch;
      const int         c_nelmtPerBatch;
      const int         batchIdx;
      const int         threadIdx;
      const int         blockSize;
      const int         quad_size_per_batch;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
