#ifndef kernels_portable_evaluation_kernels_h
#define kernels_portable_evaluation_kernels_h

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
                    const DeviceView<Number> &d_in,
                    const DoFIndicesView     &dof_indices,
                    const CellRangeIdView    &cell_range_ids,
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


    template <int dim, int n_rows, int n_columns, typename Number>
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
        , quad_size_per_batch(Utilities::pow(n_columns, dim) * nelmtPerBatch)
      {}



      DEAL_II_HOST_DEVICE void
      evaluate_values(Number *in, Number *out, Number *scratch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                     dim,
                                     n_rows,
                                     n_columns,
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
                                     n_rows,
                                     n_columns,
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

      DEAL_II_HOST_DEVICE void
      evaluate_gradients(const Number *in, Number *out) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = reminder + last * co_dimension_size;

                for (int d = 0; d < dim; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int idx_d      = (point / stride_d) % n_columns;
                    const int point_base = point - idx_d * stride_d;
                    // Base index for values, invariant across the n-loop
                    // below -- hoisted so it isn't recomputed n_columns
                    // times per direction.
                    const int in_base = elmnt_idx * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += shape_gradient_collocation[n * n_columns + idx_d] *
                             in[in_base + n * stride_d];

                    out[d * quad_size_per_batch + elmnt_idx * nq_total + point] = q_d;
                  }
              }
          }

        team_member.team_barrier();
      }

      DEAL_II_HOST_DEVICE void
      evaluate_gradients_and_multiply_symmetric_tensor(const DeviceView<Number> &d_G,
                                                       const CellRangeIdView    &cell_range_ids,
                                                       const Number             *in,
                                                       Number                   *out) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total                   = Utilities::pow(n_columns, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            unsigned int global_cell_index = batchIdx * nelmtPerBatch + elmnt_idx;
            if (cell_range_ids.size() > 0)
              global_cell_index = cell_range_ids(global_cell_index);

            const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * nq_total;

            int    idx[dim - 1];
            Number reg[dim][n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (reminder / stride_d) % n_columns;
              }

            for (int n = 0; n < n_columns; ++n)
              {
                for (int d = 0; d < dim - 1; ++d)
                  reg[d][n] = shape_gradient_collocation[n * n_columns + idx[d]];
                reg[dim - 1][n] = in[elmnt_idx * nq_total + reminder + n * co_dimension_size];
              }

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = reminder + last * co_dimension_size;

                Number q[dim];
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    const int in_base    = elmnt_idx * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += reg[d][n] * in[in_base + n * stride_d];
                    q[d] = q_d;
                  }
                {
                  Number q_d = 0;
                  for (int n = 0; n < n_columns; ++n)
                    q_d += shape_gradient_collocation[n * n_columns + last] * reg[dim - 1][n];
                  q[dim - 1] = q_d;
                }

                Number G[dim][dim];
                int    component_index = 0;
                for (int d1 = 0; d1 < dim; ++d1)
                  for (int d2 = d1; d2 < dim; ++d2)
                    {
                      const Number value = d_G[cell_g_offset + component_index * nq_total + point];
                      G[d1][d2]          = value;
                      G[d2][d1]          = value;
                      ++component_index;
                    }

                for (int d1 = 0; d1 < dim; ++d1)
                  {
                    Number value_out = 0;
                    for (int d2 = 0; d2 < dim; ++d2)
                      value_out += G[d1][d2] * q[d2];

                    out[d1 * quad_size_per_batch + elmnt_idx * nq_total + point] = value_out;
                  }
              }
          }

        team_member.team_barrier();
      }


      // `add` mirrors real deal.II's own FEEvaluationImplTransformToCollocation
      // ::integrate()'s conditional `add` on its first co_gradients() call:
      // when the caller also submitted values (so `values` already holds
      // submit_value()'s JxW-multiplied contribution), add<true> accumulates
      // this method's gradient-integration result on top instead of
      // overwriting it, fusing the two contributions before
      // integrate_values() (BK3's steps 7-9) transforms the combined
      // collocation-space result back to dof space -- see
      // kernels/portable_batched_fe_evaluation.h's FEEvaluation::integrate().
      // Defaults to false so every existing call site (this project's own
      // gradients-only fused Laplace path) is unaffected.
      template <bool add = false>
      DEAL_II_HOST_DEVICE void
      integrate_gradients(const Number *gradients, Number *values) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int elmnt_idx = tid / co_dimension_size;
            const int reminder  = tid % co_dimension_size;

            int    idx[dim - 1];
            Number r_shape[dim - 1][n_columns];
            Number r_grad[n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (reminder / stride_d) % n_columns;
                for (int n = 0; n < n_columns; ++n)
                  r_shape[d][n] = shape_gradient_collocation[idx[d] * n_columns + n];
              }
            for (int n = 0; n < n_columns; ++n)
              r_grad[n] = gradients[(dim - 1) * quad_size_per_batch + elmnt_idx * nq_total +
                                    reminder + n * co_dimension_size];

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = reminder + last * co_dimension_size;

                Number tmp0 = 0;
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    // Base index into the gradients pool, invariant across
                    // the n-loop below -- hoisted so it isn't recomputed
                    // n_columns times per direction.
                    const int grad_base =
                      d * quad_size_per_batch + elmnt_idx * nq_total + point_base;

                    for (int n = 0; n < n_columns; ++n)
                      tmp0 += gradients[grad_base + n * stride_d] * r_shape[d][n];
                  }
                for (int n = 0; n < n_columns; ++n)
                  tmp0 += r_grad[n] * shape_gradient_collocation[last * n_columns + n];

                if constexpr (add)
                  values[elmnt_idx * nq_total + point] += tmp0;
                else
                  values[elmnt_idx * nq_total + point] = tmp0;
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
