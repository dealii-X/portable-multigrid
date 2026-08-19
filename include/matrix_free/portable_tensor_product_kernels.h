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

    /**
     * In this namespace, the evaluator routines that evaluate the tensor
     * products are implemented.
     */
    // TODO: for now only the general variant is implemented
    enum EvaluatorVariant
    {
      evaluate_general,
      evaluate_symmetric,
      evaluate_evenodd
    };


    using TeamHandle = Kokkos::TeamPolicy<>::member_type;

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using CellRangeIdView = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;



    /**
     * One-dimensional kernel for use by the generic tensor product
     * interpolation as provided by the class EvaluatorTensorProduct,
     * implementing a matrix-vector product along this dimension, controlled by
     * the number of rows and columns and the stride in the input and output
     * arrays, which are embedded into some lexicographic ordering of unknowns
     * in a tensor-product arrangement.
     */
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

    /**
     * Specialized version of apply_matrix_vector_product() that takes the strides as arguments,
     * rather than as template parameters.
     */
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


    /**
     * Helper function that applies sum factorization in a specified direction using batched kernel
     * and apply_matrix_vector_product().
     *
     * Sizes of the input and output vectors in 2D:
     * -----------------------------------------------------------
     * direction|  contract_over_rows  |  !contract_over_rows
     * ----------------------------------------------------------
     *      0   |  mm x mm -> nn x mm  |  nn x mm -> mm x mm
     * ----------------------------------------------------------
     *      1   |  mm x nn -> mm x nn  |  mm x nn -> mm x nn
     * ----------------------------------------------------------
     *
     * Sizes of the input and output vectors in 3D:
     * -----------------------------------------------------------------------
     * direction|     contract_over_rows       |      !contract_over_rows
     * -----------------------------------------------------------------------
     *     0    | mm x mm x mm -> nn x mm x mm |  nn x mm x mm -> mm x mm x mm
     * -----------------------------------------------------------------------
     *     1    | nn x mm x mm -> nn x nn x mm |  nn x nn x mm -> nn x mm x mm
     * -----------------------------------------------------------------------
     *     2    | nn x nn x mm -> nn x nn x nn |  nn x nn x nn -> nn x nn x mm
     * -----------------------------------------------------------------------
     */
    template <int  dim,
              int  direction,
              int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply(const TeamHandle &team_member,
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

          apply_matrix_vector_product<n_rows,
                                      n_columns,
                                      contract_over_rows,
                                      add,
                                      n_blocks1,
                                      n_blocks1>(matrix, in_e, out_e);
        }

      team_member.team_barrier();
    }

    /**
     * Helper function that copies or adds the first N entries of src to
     * dst, depending on the template argument "add".
     */
    template <bool add, typename Number>
    DEAL_II_HOST_DEVICE inline void
    populate_view(const TeamHandle &team_member,
                  Number           *dst,
                  const Number     *src,
                  const int         N,
                  const int         threadIdx,
                  const int         blockSize)
    {
      for (int tid = threadIdx; tid < N; tid += blockSize)
        {
          if constexpr (add)
            Kokkos::atomic_add(&dst[tid], src[tid]);
          else
            dst[tid] = src[tid];
        }

      team_member.team_barrier();
    }



    /**
     * Generic evaluator framework.
     */
    template <EvaluatorVariant variant, int dim, int n_rows, int n_columns, typename Number>
    struct EvaluatorTensorProduct
    {};

    /**
     * Internal evaluator for 1d-3d shape function using the tensor product form
     * of the basis functions.
     */
    template <int dim, int n_rows, int n_columns, typename Number>
    struct EvaluatorTensorProduct<evaluate_general, dim, n_rows, n_columns, Number>
    {
    public:
      DEAL_II_HOST_DEVICE
      EvaluatorTensorProduct(const TeamHandle &team_member,
                             const Number     *shape_values,
                             const Number     *shape_gradients,
                             const Number     *co_shape_gradients,
                             Number           *temp,
                             const int         c_nelmtPerBatch,
                             const int         threadIdx,
                             const int         blockSize)
        : team_member(team_member)
        , shape_values(shape_values)
        , shape_gradients(shape_gradients)
        , co_shape_gradients(co_shape_gradients)
        , temp(temp)
        , c_nelmtPerBatch(c_nelmtPerBatch)
        , threadIdx(threadIdx)
        , blockSize(blockSize)
      {}

      /**
       * Evaluate/integrate the values of a finite element function at the
       * quadrature points for a given @p direction.
       */
      template <int direction, bool dof_to_quad, bool add, bool in_place = false>
      DEAL_II_HOST_DEVICE void
      values(const Number *in, Number *out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, false>(
              team_member, shape_values, in, temp, c_nelmtPerBatch, threadIdx, blockSize);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               c_nelmtPerBatch * n_blocks1 * nn * n_blocks2,
                               threadIdx,
                               blockSize);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(
              team_member, shape_values, in, out, c_nelmtPerBatch, threadIdx, blockSize);
          }
      }

      /**
       * Evaluate/integrate the gradient of a finite element function at the
       * quadrature points for a given @p direction.
       */
      template <int direction, bool dof_to_quad, bool add, bool in_place = false>
      DEAL_II_HOST_DEVICE void
      gradients(const Number *in, Number *out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, false>(
              team_member, shape_gradients, in, temp, c_nelmtPerBatch, threadIdx, blockSize);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               c_nelmtPerBatch * n_blocks1 * nn * n_blocks2,
                               threadIdx,
                               blockSize);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(
              team_member, shape_gradients, in, out, c_nelmtPerBatch, threadIdx, blockSize);
          }
      }

      /**
       * Evaluate the gradient of a finite element function at the quadrature
       * points for a given @p direction for collocation methods.
       */
      template <int direction, bool dof_to_quad, bool add, bool in_place = false>
      DEAL_II_HOST_DEVICE void
      co_gradients(const Number *in, Number *out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_columns, n_columns, dof_to_quad, false>(
              team_member, co_shape_gradients, in, temp, c_nelmtPerBatch, threadIdx, blockSize);

            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_columns, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               c_nelmtPerBatch * n_blocks1 * n_columns * n_blocks2,
                               threadIdx,
                               blockSize);
          }
        else
          {
            apply<dim, direction, n_columns, n_columns, dof_to_quad, add>(
              team_member, co_shape_gradients, in, out, c_nelmtPerBatch, threadIdx, blockSize);
          }
      }

    private:
      const TeamHandle &team_member;
      const Number     *shape_values;
      const Number     *shape_gradients;
      const Number     *co_shape_gradients;
      Number           *temp;
      const int         c_nelmtPerBatch;
      const int         threadIdx;
      const int         blockSize;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
