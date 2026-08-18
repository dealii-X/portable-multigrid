#ifndef kernels_portable_tensor_product_kernels_h
#define kernels_portable_tensor_product_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {
    using TeamHandle = Kokkos::TeamPolicy<>::member_type;

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int n_rows, int n_columns, bool contract_over_rows, bool add, typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const Number *matrix,
                                const Number *in,
                                Number       *out,
                                const int     stride_in,
                                const int     stride_out)
    {
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      // Cache the input fiber in registers once, rather than re-reading
      // shared memory for every output index below.
      Number r_in[mm];
      for (int k = 0; k < mm; ++k)
        r_in[k] = in[k * stride_in];

      for (int q = 0; q < nn; ++q)
        {
          Number sum = 0;
          for (int k = 0; k < mm; ++k)
            {
              const int row = contract_over_rows ? k : q;
              const int col = contract_over_rows ? q : k;
              sum += matrix[row * n_columns + col] * r_in[k];
            }

          if constexpr (add)
            out[q * stride_out] += sum;
          else
            out[q * stride_out] = sum;
        }
    }


    template <int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              int  stride_in,
              int  stride_out,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const Number *matrix, const Number *in, Number *out)
    {
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      Number r_in[mm];
      for (int k = 0; k < mm; ++k)
        r_in[k] = in[k * stride_in];

      for (int q = 0; q < nn; ++q)
        {
          Number sum = 0;
          for (int k = 0; k < mm; ++k)
            {
              const int row = contract_over_rows ? k : q;
              const int col = contract_over_rows ? q : k;
              sum += matrix[row * n_columns + col] * r_in[k];
            }

          if constexpr (add)
            out[q * stride_out] += sum;
          else
            out[q * stride_out] = sum;
        }
    }


    template <int  dim,
              int  direction,
              int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const TeamHandle &team_member,
                                const Number     *matrix,
                                const Number     *in,
                                Number           *out,
                                const int         c_nelmtPerBatch,
                                const int         threadIdx,
                                const int         blockSize)
    {
      static_assert(direction >= 0 && direction < dim, "direction must be in [0, dim)");

      // mm: extent of the contracted axis in `in`; nn: extent in `out`.
      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      // n_blocks1: combined extent of the already-transformed axes (role <
      // direction, extent n_columns each); n_blocks2: combined extent of
      // the not-yet-transformed axes (role > direction, extent n_rows
      // each). Their product is the per-cell thread count for this call,
      // and (since axes of role < direction are the fastest-varying, per
      // the layout convention above) also the stride between consecutive
      // entries of the fiber each thread owns.
      constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
      constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

      constexpr int n_in_per_elmt  = n_blocks1 * mm * n_blocks2;
      constexpr int n_out_per_elmt = n_blocks1 * nn * n_blocks2;

      for (int tid = threadIdx; tid < c_nelmtPerBatch * n_blocks1 * n_blocks2; tid += blockSize)
        {
          const int e   = tid / (n_blocks1 * n_blocks2);
          const int rem = tid % (n_blocks1 * n_blocks2);
          const int i2  = rem / n_blocks1;
          const int i1  = rem % n_blocks1;

          const Number *in_e  = in + e * n_in_per_elmt + i2 * n_blocks1 * mm + i1;
          Number       *out_e = out + e * n_out_per_elmt + i2 * n_blocks1 * nn + i1;

          apply_matrix_vector_product<n_rows, n_columns, contract_over_rows, add, n_blocks1, n_blocks1>(
            matrix, in_e, out_e);
        }

      team_member.team_barrier();
    }

    template <int dim, int n_rows, int n_columns, typename Number>
    class EvaluatorTensorProduct
    {
    public:
      DEAL_II_HOST_DEVICE
      EvaluatorTensorProduct(const TeamHandle &team_member,
                             const Number     *matrix,
                             Number           *temp,
                             const int         c_nelmtPerBatch,
                             const int         threadIdx,
                             const int         blockSize)
        : team_member(team_member)
        , matrix(matrix)
        , temp(temp)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
      {}

      template <int direction, bool dof_to_quad, bool add>
      DEAL_II_HOST_DEVICE void
      values(const Number *in, Number *out) const
      {
        apply_matrix_vector_product<dim, direction, n_rows, n_columns, dof_to_quad, add>(
          team_member, matrix, in, out, c_nelmtPerBatch, threadIdx, blockSize);
      }

    private:
      const TeamHandle &team_member;
      const Number     *matrix;
      Number           *temp;
      const int         c_nelmtPerBatch;
      const int         threadIdx;
      const int         blockSize;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
