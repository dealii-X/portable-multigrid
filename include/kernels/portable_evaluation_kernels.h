#ifndef kernels_portable_evaluation_kernels_h
#define kernels_portable_evaluation_kernels_h

#include "kernels/portable_tensor_product_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {
    template <int dim, int nm, typename Number>
    DEAL_II_HOST_DEVICE inline void
    read_dof_values(const TeamHandle         &team_member,
                    const DeviceView<Number> &d_in,
                    const DoFIndicesView     &dof_indices,
                    Number                    *values,
                    const int                 eb,
                    const int                 nelmtPerBatch,
                    const CellRangeIdView    &cell_range_ids,
                    const int                 c_nelmtPerBatch,
                    const int                 threadIdx,
                    const int                 blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nm_total          = Utilities::pow(nm, dim);
      constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          unsigned int global_cell_index = eb * nelmtPerBatch + e;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          for (int last = 0; last < nm; ++last)
            {
              const int local_idx = rem + last * co_dimension_size;

              const unsigned int dof_index = dof_indices(local_idx, global_cell_index);
              const int          shared_idx = e * nm_total + local_idx;

              if (dof_index == numbers::invalid_unsigned_int)
                values[shared_idx] = 0;
              else
                values[shared_idx] = d_in[dof_index];
            }
        }

      team_member.team_barrier();
    }


    template <int dim, int nm, typename Number>
    DEAL_II_HOST_DEVICE inline void
    distribute_local_to_global(const TeamHandle       &team_member,
                               const Number           *values,
                               DeviceView<Number>      d_out,
                               const DoFIndicesView   &dof_indices,
                               const int               eb,
                               const int               nelmtPerBatch,
                               const CellRangeIdView  &cell_range_ids,
                               const int               c_nelmtPerBatch,
                               const int               threadIdx,
                               const int               blockSize)
    {
      static_assert(dim >= 1, "dim must be at least 1");

      constexpr int nm_total          = Utilities::pow(nm, dim);
      constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
        {
          const int e   = tid / co_dimension_size;
          const int rem = tid % co_dimension_size;

          unsigned int global_cell_index = eb * nelmtPerBatch + e;
          if (cell_range_ids.size() > 0)
            global_cell_index = cell_range_ids(global_cell_index);

          for (int last = 0; last < nm; ++last)
            {
              const int local_idx = rem + last * co_dimension_size;

              const unsigned int dof_index  = dof_indices(local_idx, global_cell_index);
              const int          shared_idx = e * nm_total + local_idx;

              if (dof_index != numbers::invalid_unsigned_int)
                Kokkos::atomic_add(&d_out[dof_index], values[shared_idx]);
            }
        }

      team_member.team_barrier();
    }


    template <int dim, int n_rows, int n_columns, typename Number>
    class FEEvaluationImplTransformToCollocation
    {
    public:
      DEAL_II_HOST_DEVICE
      FEEvaluationImplTransformToCollocation(const TeamHandle &team_member,
                                             const Number     *matrix,
                                             const Number     *shape_gradient_collocation,
                                             const int         c_nelmtPerBatch,
                                             const int         threadIdx,
                                             const int         blockSize)
        : team_member(team_member)
        , matrix(matrix)
        , shape_gradient_collocation(shape_gradient_collocation)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
      {}



      DEAL_II_HOST_DEVICE void
      evaluate_values(Number *in, Number *out, Number *scratch, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<dim, n_rows, n_columns, Number> evaluator(
          team_member, matrix, nullptr, c_nelmtPerBatch, threadIdx, blockSize);

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
      integrate_values(Number *in, Number *out, Number *scratch, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1 && dim <= 3, "dim must be 1, 2, or 3");

        const EvaluatorTensorProduct<dim, n_rows, n_columns, Number> evaluator(
          team_member, matrix, nullptr, c_nelmtPerBatch, threadIdx, blockSize);

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
      evaluate_gradients(const Number *values, Number *gradients, const int quad_size_per_batch) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                for (int d = 0; d < dim; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int idx_d      = (point / stride_d) % n_columns;
                    const int point_base = point - idx_d * stride_d;
                    // Base index for values, invariant across the n-loop
                    // below -- hoisted so it isn't recomputed n_columns
                    // times per direction.
                    const int in_base = e * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += shape_gradient_collocation[n * n_columns + idx_d] *
                             values[in_base + n * stride_d];

                    gradients[d * quad_size_per_batch + e * nq_total + point] = q_d;
                  }
              }
          }

        team_member.team_barrier();
      }

      DEAL_II_HOST_DEVICE void
      evaluate_gradients_and_multiply_symmetric_tensor(const DeviceView<Number> &d_G,
                                             const int                 eb,
                                             const int                 nelmtPerBatch,
                                             const CellRangeIdView    &cell_range_ids,
                                             const Number             *values,
                                             Number                    *gradients,
                                             const int                 quad_size_per_batch) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total                   = Utilities::pow(n_columns, dim);
        constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;
        constexpr int co_dimension_size          = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            unsigned int global_cell_index = eb * nelmtPerBatch + e;
            if (cell_range_ids.size() > 0)
              global_cell_index = cell_range_ids(global_cell_index);

            const int cell_g_offset = global_cell_index * symmetric_tensor_dimension * nq_total;

            int    idx[dim - 1];
            Number reg[dim][n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (rem / stride_d) % n_columns;
              }

            for (int n = 0; n < n_columns; ++n)
              {
                for (int d = 0; d < dim - 1; ++d)
                  reg[d][n] = shape_gradient_collocation[n * n_columns + idx[d]];
                reg[dim - 1][n] = values[e * nq_total + rem + n * co_dimension_size];
              }

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                Number q[dim];
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    const int in_base    = e * nq_total + point_base;

                    Number q_d = 0;
                    for (int n = 0; n < n_columns; ++n)
                      q_d += reg[d][n] * values[in_base + n * stride_d];
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
                    Number out = 0;
                    for (int d2 = 0; d2 < dim; ++d2)
                      out += G[d1][d2] * q[d2];

                    gradients[d1 * quad_size_per_batch + e * nq_total + point] = out;
                  }
              }
          }

        team_member.team_barrier();
      }


      DEAL_II_HOST_DEVICE void
      integrate_gradients(const Number *gradients,
                         const int     quad_size_per_batch,
                         Number       *values) const
      {
        static_assert(dim >= 1, "dim must be at least 1");

        constexpr int nq_total          = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size; tid += blockSize)
          {
            const int e   = tid / co_dimension_size;
            const int rem = tid % co_dimension_size;

            int    idx[dim - 1];
            Number r_shape[dim - 1][n_columns];
            Number r_grad[n_columns];

            for (int d = 0; d < dim - 1; ++d)
              {
                const int stride_d = Utilities::pow(n_columns, d);
                idx[d]             = (rem / stride_d) % n_columns;
                for (int n = 0; n < n_columns; ++n)
                  r_shape[d][n] = shape_gradient_collocation[idx[d] * n_columns + n];
              }
            for (int n = 0; n < n_columns; ++n)
              r_grad[n] = gradients[(dim - 1) * quad_size_per_batch + e * nq_total + rem +
                                    n * co_dimension_size];

            for (int last = 0; last < n_columns; ++last)
              {
                const int point = rem + last * co_dimension_size;

                Number tmp0 = 0;
                for (int d = 0; d < dim - 1; ++d)
                  {
                    const int stride_d   = Utilities::pow(n_columns, d);
                    const int point_base = point - idx[d] * stride_d;
                    // Base index into the gradients pool, invariant across
                    // the n-loop below -- hoisted so it isn't recomputed
                    // n_columns times per direction.
                    const int grad_base = d * quad_size_per_batch + e * nq_total + point_base;

                    for (int n = 0; n < n_columns; ++n)
                      tmp0 += gradients[grad_base + n * stride_d] * r_shape[d][n];
                  }
                for (int n = 0; n < n_columns; ++n)
                  tmp0 += r_grad[n] * shape_gradient_collocation[last * n_columns + n];

                values[e * nq_total + point] = tmp0;
              }
          }

        team_member.team_barrier();
      }

    private:
      const TeamHandle &team_member;
      const Number     *matrix;
      const Number     *shape_gradient_collocation;
      const int         c_nelmtPerBatch;
      const int         threadIdx;
      const int         blockSize;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
