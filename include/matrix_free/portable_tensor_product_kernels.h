#ifndef kernels_portable_tensor_product_kernels_h
#define kernels_portable_tensor_product_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <type_traits>

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
     * Kokkos::View-based overload of apply_matrix_vector_product()
     * above with `matrix`/`in`/`out` passed as Kokkos::Views.
     */
    template <int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              int  stride_in,
              int  stride_out,
              typename ViewTypeMatrix,
              typename ViewTypeIn,
              typename ViewTypeOut,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeOut>::value>>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const ViewTypeMatrix matrix, const ViewTypeIn in, ViewTypeOut out)
    {
      using Number = typename ViewTypeOut::non_const_value_type;

      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      Number r_in[mm];
      for (int k = 0; k < mm; ++k)
        r_in[k] = in(k * stride_in);

      for (int q = 0; q < nn; ++q)
        {
          Number sum = 0;
          for (int k = 0; k < mm; ++k)
            {
              const int row = contract_over_rows ? k : q;
              const int col = contract_over_rows ? q : k;
              sum += matrix(row * n_columns + col) * r_in[k];
            }

          if constexpr (add)
            out(q * stride_out) += sum;
          else
            out(q * stride_out) = sum;
        }
    }

    /**
     * View-based overload of the runtime-stride apply_matrix_vector_product()
     * above.
     */
    template <int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              typename ViewTypeMatrix,
              typename ViewTypeIn,
              typename ViewTypeOut,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeOut>::value>>
    DEAL_II_HOST_DEVICE inline void
    apply_matrix_vector_product(const ViewTypeMatrix matrix,
                                const ViewTypeIn     in,
                                ViewTypeOut          out,
                                const int            stride_in,
                                const int            stride_out)
    {
      using Number = typename ViewTypeOut::non_const_value_type;

      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      Number r_in[mm];
      for (int k = 0; k < mm; ++k)
        r_in[k] = in(k * stride_in);

      for (int q = 0; q < nn; ++q)
        {
          Number sum = 0;
          for (int k = 0; k < mm; ++k)
            {
              const int row = contract_over_rows ? k : q;
              const int col = contract_over_rows ? q : k;
              sum += matrix(row * n_columns + col) * r_in[k];
            }

          if constexpr (add)
            out(q * stride_out) += sum;
          else
            out(q * stride_out) = sum;
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
          const int         n_elements_in_current_batch,
          const int         thread_id,
          const int         block_size)
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

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_blocks1 * n_blocks2;
           tid += block_size)
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
     * Anisotropic variant of apply(): the combined extent of the
     * not-yet-transformed axes perpendicular to `direction` (`n_blocks2`) is
     * supplied explicitly instead of being assumed to be
     * pow(n_rows, dim - direction - 1). This is needed for tensor-product
     * elements whose 1D size differs between axes, e.g. Raviart-Thomas, where
     * one axis carries the normal component (n_rows here) and the others the
     * tangential one. `n_blocks1` stays pow(n_columns, direction), which is
     * unaffected since every already-transformed axis has extent n_columns.
     */
    template <int  dim,
              int  direction,
              int  n_rows,
              int  n_columns,
              int  n_blocks2,
              bool contract_over_rows,
              bool add,
              typename Number>
    DEAL_II_HOST_DEVICE inline void
    apply_anisotropic(const TeamHandle &team_member,
                      const Number     *matrix,
                      const Number     *in,
                      Number           *out,
                      const int         n_elements_in_current_batch,
                      const int         thread_id,
                      const int         block_size)
    {
      static_assert(direction >= 0 && direction < dim, "direction must be in [0, dim)");

      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      constexpr int n_blocks1      = Utilities::pow(n_columns, direction);
      constexpr int n_in_per_elmt  = n_blocks1 * mm * n_blocks2;
      constexpr int n_out_per_elmt = n_blocks1 * nn * n_blocks2;

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_blocks1 * n_blocks2;
           tid += block_size)
        {
          const int e   = tid / (n_blocks1 * n_blocks2);
          const int rem = tid % (n_blocks1 * n_blocks2);
          const int i2  = rem / n_blocks1;
          const int i1  = rem % n_blocks1;

          apply_matrix_vector_product<n_rows,
                                      n_columns,
                                      contract_over_rows,
                                      add,
                                      n_blocks1,
                                      n_blocks1>(matrix,
                                                 in + e * n_in_per_elmt + i2 * n_blocks1 * mm + i1,
                                                 out + e * n_out_per_elmt + i2 * n_blocks1 * nn + i1);
        }

      team_member.team_barrier();
    }


    /**
     * Collocation gradient of `n_components` scalar fields sampled at the
     * n_q^dim quadrature points of a cell (batch), followed by a pointwise
     * symmetric-tensor multiply that maps the `dim` reference-space derivatives
     * to `dim` output directions:
     *
     *   gradients[c][d1][q] = sum_{d0} ( d/dxi_{d0} values[c] )(q) * G(q)[d0][d1]
     *
     * The 1D collocation-derivative matrix `co_shape_gradients` is n_q x n_q,
     * row-major (`[row * n_q + col]`, contracting over the row). `G` is loaded
     * per quadrature point from `symmetric_tensor`, which stores the
     * (dim*(dim+1))/2 independent entries element-major as
     * `[elem][sym_index][q]`, sym_index enumerating (d0,d1) with d1 >= d0 in
     * row-major order. All of `symmetric_tensor`, `values[c]` (length n_q^dim
     * per element) and `gradients[c]` (length dim*n_q^dim per element, layout
     * [dir][q]) are indexed by the *local* batch element, so the caller passes
     * pointers already offset to the start of the current batch.
     */
    template <int dim, int n_components, int n_q, typename Number>
    DEAL_II_HOST_DEVICE inline void
    evaluate_vector_gradients_and_multiply_symmetric_tensor(
      const TeamHandle    &team_member,
      const Number        *co_shape_gradients,
      const Number        *symmetric_tensor,
      const Number *const *values,
      Number *const       *gradients,
      const int            n_elements_in_current_batch,
      const int            thread_id,
      const int            block_size)
    {
      constexpr int n_q_total = Utilities::pow(n_q, dim);
      constexpr int n_sym     = (dim * (dim + 1)) / 2;
      constexpr int n_planes  = Utilities::pow(n_q, dim - 1);

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_planes; tid += block_size)
        {
          const int e     = tid / n_planes;
          const int plane = tid % n_planes;
          const int a1    = plane % n_q;                    // fixed index along axis 1
          const int a2    = (dim == 3) ? (plane / n_q) : 0; // fixed index along axis 2

          const int e_val  = e * n_q_total;
          const int e_grad = e * dim * n_q_total;
          const int e_ten  = e * n_sym * n_q_total;

          // Derivative rows that stay fixed as this thread sweeps axis 0, plus
          // the axis-0 fiber of every component (contiguous, reused for all a0).
          Number d_row_1[n_q];
          Number d_row_2[n_q];
          Number fiber_0[n_components][n_q];
          for (int n = 0; n < n_q; ++n)
            {
              d_row_1[n] = co_shape_gradients[n * n_q + a1];
              if constexpr (dim == 3)
                d_row_2[n] = co_shape_gradients[n * n_q + a2];
              for (int c = 0; c < n_components; ++c)
                fiber_0[c][n] = values[c][e_val + (n + a1 * n_q + a2 * n_q * n_q)];
            }

          for (int a0 = 0; a0 < n_q; ++a0) // output quadrature point along axis 0
            {
              const int q = a0 + a1 * n_q + a2 * n_q * n_q;

              Number G[dim][dim];
              for (int d0 = 0, s = 0; d0 < dim; ++d0)
                for (int d1 = d0; d1 < dim; ++d1, ++s)
                  {
                    const Number g   = symmetric_tensor[e_ten + s * n_q_total + q];
                    G[d0][d1]        = g;
                    G[d1][d0]        = g;
                  }

              for (int c = 0; c < n_components; ++c)
                {
                  Number grad_ref[dim];
                  for (int d = 0; d < dim; ++d)
                    grad_ref[d] = 0;

                  for (int n = 0; n < n_q; ++n)
                    {
                      grad_ref[0] += co_shape_gradients[n * n_q + a0] * fiber_0[c][n];
                      grad_ref[1] += d_row_1[n] * values[c][e_val + (a0 + n * n_q + a2 * n_q * n_q)];
                      if constexpr (dim == 3)
                        grad_ref[2] +=
                          d_row_2[n] * values[c][e_val + (a0 + a1 * n_q + n * n_q * n_q)];
                    }

                  for (int d1 = 0; d1 < dim; ++d1)
                    {
                      Number acc = 0;
                      for (int d0 = 0; d0 < dim; ++d0)
                        acc += grad_ref[d0] * G[d0][d1];
                      gradients[c][e_grad + d1 * n_q_total + q] = acc;
                    }
                }
            }
        }

      team_member.team_barrier();
    }


    /**
     * Transpose of the collocation-gradient part of
     * evaluate_vector_gradients_and_multiply_symmetric_tensor(): given the
     * `dim` direction components of the gradient of each of `n_components`
     * fields at the quadrature points, apply the transpose 1D
     * collocation-derivative along each axis and accumulate into `values[c]`
     * (which is *added to*, not overwritten):
     *
     *   values[c][q] += sum_{d} ( D_d^T gradients[c][d] )(q)
     *
     * Layout and batch-local indexing conventions match
     * evaluate_vector_gradients_and_multiply_symmetric_tensor().
     */
    template <int dim, int n_components, int n_q, typename Number>
    DEAL_II_HOST_DEVICE inline void
    integrate_vector_gradients(const TeamHandle    &team_member,
                               const Number        *co_shape_gradients,
                               const Number *const *gradients,
                               Number *const       *values,
                               const int            n_elements_in_current_batch,
                               const int            thread_id,
                               const int            block_size)
    {
      constexpr int n_q_total = Utilities::pow(n_q, dim);
      constexpr int n_planes  = Utilities::pow(n_q, dim - 1);

      for (int tid = thread_id; tid < n_elements_in_current_batch * n_planes; tid += block_size)
        {
          const int e     = tid / n_planes;
          const int plane = tid % n_planes;
          const int a1    = plane % n_q;
          const int a2    = (dim == 3) ? (plane / n_q) : 0;

          const int e_val  = e * n_q_total;
          const int e_grad = e * dim * n_q_total;

          Number d_row_1[n_q];
          Number d_row_2[n_q];
          Number fiber_0[n_components][n_q];
          for (int n = 0; n < n_q; ++n)
            {
              d_row_1[n] = co_shape_gradients[a1 * n_q + n];
              if constexpr (dim == 3)
                d_row_2[n] = co_shape_gradients[a2 * n_q + n];
              for (int c = 0; c < n_components; ++c)
                fiber_0[c][n] =
                  gradients[c][e_grad + 0 * n_q_total + (n + a1 * n_q + a2 * n_q * n_q)];
            }

          for (int a0 = 0; a0 < n_q; ++a0)
            {
              const int q = a0 + a1 * n_q + a2 * n_q * n_q;

              for (int c = 0; c < n_components; ++c)
                {
                  Number acc = 0;
                  for (int n = 0; n < n_q; ++n)
                    {
                      acc += fiber_0[c][n] * co_shape_gradients[a0 * n_q + n];
                      acc += gradients[c][e_grad + 1 * n_q_total + (a0 + n * n_q + a2 * n_q * n_q)] *
                             d_row_1[n];
                      if constexpr (dim == 3)
                        acc +=
                          gradients[c][e_grad + 2 * n_q_total + (a0 + a1 * n_q + n * n_q * n_q)] *
                          d_row_2[n];
                    }
                  values[c][e_val + q] += acc;
                }
            }
        }

      team_member.team_barrier();
    }


    /**
     * View-based overload of apply() above.
     */
    template <int  dim,
              int  direction,
              int  n_rows,
              int  n_columns,
              bool contract_over_rows,
              bool add,
              typename ViewTypeMatrix,
              typename ViewTypeIn,
              typename ViewTypeOut,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeOut>::value>>
    DEAL_II_HOST_DEVICE inline void
    apply(const TeamHandle    &team_member,
          const ViewTypeMatrix matrix,
          const ViewTypeIn     in,
          ViewTypeOut          out,
          const int            n_elements_in_current_batch)
    {
      static_assert(direction >= 0 && direction < dim, "direction must be in [0, dim)");

      constexpr int mm = contract_over_rows ? n_rows : n_columns;
      constexpr int nn = contract_over_rows ? n_columns : n_rows;

      constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
      constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

      constexpr int n_in_per_elmt  = n_blocks1 * mm * n_blocks2;
      constexpr int n_out_per_elmt = n_blocks1 * nn * n_blocks2;

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team_member, n_elements_in_current_batch * n_blocks1 * n_blocks2),
        [&](const int tid)
          {
            const int e   = tid / (n_blocks1 * n_blocks2);
            const int rem = tid % (n_blocks1 * n_blocks2);
            const int i2  = rem / n_blocks1;
            const int i1  = rem % n_blocks1;

            const int in_offset  = e * n_in_per_elmt + i2 * n_blocks1 * mm + i1;
            const int out_offset = e * n_out_per_elmt + i2 * n_blocks1 * nn + i1;

            apply_matrix_vector_product<n_rows,
                                        n_columns,
                                        contract_over_rows,
                                        add,
                                        n_blocks1,
                                        n_blocks1>(
              matrix,
              Kokkos::subview(in, Kokkos::make_pair(in_offset, static_cast<int>(in.extent(0)))),
              Kokkos::subview(out, Kokkos::make_pair(out_offset, static_cast<int>(out.extent(0)))));
          });

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
                  const int         thread_id,
                  const int         block_size)
    {
      for (int tid = thread_id; tid < N; tid += block_size)
        {
          if constexpr (add)
            dst[tid] += src[tid];
          else
            dst[tid] = src[tid];
        }

      team_member.team_barrier();
    }

    /**
     * View-based overload of populate_view() above.
     */
    template <bool add,
              typename ViewTypeOut,
              typename ViewTypeIn,
              typename = std::enable_if_t<Kokkos::is_view<ViewTypeOut>::value>>
    DEAL_II_HOST_DEVICE inline void
    populate_view(const TeamHandle &team_member, ViewTypeOut dst, const ViewTypeIn src, const int N)
    {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, N),
                           [&](const int tid)
                             {
                               if constexpr (add)
                                 dst(tid) += src(tid);
                               else
                                 dst(tid) = src(tid);
                             });

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
                             const int         n_elements_in_current_batch,
                             const int         thread_id,
                             const int         block_size)
        : team_member(team_member)
        , shape_values(shape_values)
        , shape_gradients(shape_gradients)
        , co_shape_gradients(co_shape_gradients)
        , temp(temp)
        , n_elements_in_current_batch(n_elements_in_current_batch)
        , thread_id(thread_id)
        , block_size(block_size)
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
              team_member,
              shape_values,
              in,
              temp,
              n_elements_in_current_batch,
              thread_id,
              block_size);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * nn * n_blocks2,
                               thread_id,
                               block_size);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(team_member,
                                                                       shape_values,
                                                                       in,
                                                                       out,
                                                                       n_elements_in_current_batch,
                                                                       thread_id,
                                                                       block_size);
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
              team_member,
              shape_gradients,
              in,
              temp,
              n_elements_in_current_batch,
              thread_id,
              block_size);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * nn * n_blocks2,
                               thread_id,
                               block_size);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(team_member,
                                                                       shape_gradients,
                                                                       in,
                                                                       out,
                                                                       n_elements_in_current_batch,
                                                                       thread_id,
                                                                       block_size);
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
              team_member,
              co_shape_gradients,
              in,
              temp,
              n_elements_in_current_batch,
              thread_id,
              block_size);

            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_columns, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * n_columns * n_blocks2,
                               thread_id,
                               block_size);
          }
        else
          {
            apply<dim, direction, n_columns, n_columns, dof_to_quad, add>(
              team_member,
              co_shape_gradients,
              in,
              out,
              n_elements_in_current_batch,
              thread_id,
              block_size);
          }
      }

    private:
      const TeamHandle &team_member;
      const Number     *shape_values;
      const Number     *shape_gradients;
      const Number     *co_shape_gradients;
      Number           *temp;
      const int         n_elements_in_current_batch;
      const int         thread_id;
      const int         block_size;
    };



    /**
     * Kokkos::View-based counterpart of EvaluatorTensorProduct with
     * shape_values/shape_gradients/co_shape_gradients/temp, and
     * the per-call in/out buffers, are all Kokkos::Views.
     */
    template <EvaluatorVariant variant,
              int              dim,
              int              n_rows,
              int              n_columns,
              typename Number,
              typename ShapeDataType = Kokkos::View<
                Number *,
                MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>>
    struct EvaluatorTensorProductView
    {};

    template <int dim, int n_rows, int n_columns, typename Number, typename ShapeDataType>
    struct EvaluatorTensorProductView<evaluate_general,
                                      dim,
                                      n_rows,
                                      n_columns,
                                      Number,
                                      ShapeDataType>
    {
    public:
      using SharedView =
        Kokkos::View<Number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      DEAL_II_HOST_DEVICE
      EvaluatorTensorProductView(const TeamHandle &team_member,
                                 ShapeDataType     shape_values,
                                 ShapeDataType     shape_gradients,
                                 ShapeDataType     co_shape_gradients,
                                 SharedView        temp,
                                 const int         n_elements_in_current_batch)
        : team_member(team_member)
        , shape_values(shape_values)
        , shape_gradients(shape_gradients)
        , co_shape_gradients(co_shape_gradients)
        , temp(temp)
        , n_elements_in_current_batch(n_elements_in_current_batch)
      {}

      /**
       * Evaluate/integrate the values of a finite element function at the
       * quadrature points for a given @p direction.
       */
      template <int  direction,
                bool dof_to_quad,
                bool add,
                bool in_place = false,
                typename ViewTypeIn,
                typename ViewTypeOut>
      DEAL_II_HOST_DEVICE void
      values(const ViewTypeIn in, ViewTypeOut out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, false>(
              team_member, shape_values, in, temp, n_elements_in_current_batch);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * nn * n_blocks2);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(
              team_member, shape_values, in, out, n_elements_in_current_batch);
          }
      }

      /**
       * Evaluate/integrate the gradient of a finite element function at the
       * quadrature points for a given @p direction.
       */
      template <int  direction,
                bool dof_to_quad,
                bool add,
                bool in_place = false,
                typename ViewTypeIn,
                typename ViewTypeOut>
      DEAL_II_HOST_DEVICE void
      gradients(const ViewTypeIn in, ViewTypeOut out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, false>(
              team_member, shape_gradients, in, temp, n_elements_in_current_batch);

            constexpr int nn        = dof_to_quad ? n_columns : n_rows;
            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_rows, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * nn * n_blocks2);
          }
        else
          {
            apply<dim, direction, n_rows, n_columns, dof_to_quad, add>(
              team_member, shape_gradients, in, out, n_elements_in_current_batch);
          }
      }

      /**
       * Evaluate the gradient of a finite element function at the quadrature
       * points for a given @p direction for collocation methods.
       */
      template <int  direction,
                bool dof_to_quad,
                bool add,
                bool in_place = false,
                typename ViewTypeIn,
                typename ViewTypeOut>
      DEAL_II_HOST_DEVICE void
      co_gradients(const ViewTypeIn in, ViewTypeOut out) const
      {
        if constexpr (in_place)
          {
            apply<dim, direction, n_columns, n_columns, dof_to_quad, false>(
              team_member, co_shape_gradients, in, temp, n_elements_in_current_batch);

            constexpr int n_blocks1 = Utilities::pow(n_columns, direction);
            constexpr int n_blocks2 = Utilities::pow(n_columns, dim - direction - 1);

            populate_view<add>(team_member,
                               out,
                               temp,
                               n_elements_in_current_batch * n_blocks1 * n_columns * n_blocks2);
          }
        else
          {
            apply<dim, direction, n_columns, n_columns, dof_to_quad, add>(
              team_member, co_shape_gradients, in, out, n_elements_in_current_batch);
          }
      }

      /**
       * Evaluate (transpose = false) or integrate (transpose = true) the
       * full gradient (all `dim` components at once) of a
       * collocation-space finite element function, fusing what would
       * otherwise be `dim` separate direction-by-direction co_gradients()
       * calls into a single pass with one register-cached read of `in`
       * per output quadrature point. transpose = false broadcasts one
       * scalar field into `dim` gradient components (the forward,
       * evaluate_gradients() case); transpose = true is the adjoint -- it
       * reduces `dim` gradient components back into one scalar field (the
       * integrate_gradients() case).
       */
      template <bool transpose, bool add = false, typename ViewTypeIn, typename ViewTypeOut>
      DEAL_II_HOST_DEVICE void
      co_gradients(const ViewTypeIn in, ViewTypeOut out) const
      {
        static_assert(dim >= 1, "dim must be at least 1");
        static_assert(ViewTypeIn::rank == (transpose ? 2 : 1),
                      "in must be the values (1D) for evaluate_gradients (transpose = "
                      "false), or the gradients (2D) for integrate_gradients (transpose "
                      "= true).");
        static_assert(ViewTypeOut::rank == (transpose ? 1 : 2),
                      "out must be the gradients (2D) for evaluate_gradients (transpose "
                      "= false), or the values (1D) for integrate_gradients (transpose "
                      "= true).");

        constexpr int n_q_points        = Utilities::pow(n_columns, dim);
        constexpr int co_dimension_size = Utilities::pow(n_columns, dim - 1);

        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team_member, n_elements_in_current_batch * co_dimension_size),
          [&](const int tid)
            {
              const int elmnt_idx = tid / co_dimension_size;
              const int reminder  = tid % co_dimension_size;

              Kokkos::Array<int, dim - 1> idx_d, stride_d;
              Number                      reg[dim][n_columns];

              for (int d = 0; d < dim - 1; ++d)
                {
                  stride_d[d] = Utilities::pow(n_columns, d);
                  idx_d[d]    = (reminder / stride_d[d]) % n_columns;
                  for (int n = 0; n < n_columns; ++n)
                    {
                      if constexpr (!transpose)
                        reg[d][n] = co_shape_gradients(n * n_columns + idx_d[d]);
                      else
                        reg[d][n] = co_shape_gradients(idx_d[d] * n_columns + n);
                    }
                }

              for (int n = 0; n < n_columns; ++n)
                {
                  if constexpr (!transpose)
                    reg[dim - 1][n] = in(elmnt_idx * n_q_points + reminder + n * co_dimension_size);
                  else
                    reg[dim - 1][n] =
                      in(elmnt_idx * n_q_points + reminder + n * co_dimension_size, dim - 1);
                }

              for (int last = 0; last < n_columns; ++last)
                {
                  const int q_point = reminder + last * co_dimension_size;

                  if constexpr (!transpose)
                    {
                      Number result[dim];
                      for (int d = 0; d < dim - 1; ++d)
                        {
                          const int q_point_base = q_point - idx_d[d] * stride_d[d];
                          const int in_base      = elmnt_idx * n_q_points + q_point_base;

                          Number res_d = 0;
                          for (int n = 0; n < n_columns; ++n)
                            res_d += reg[d][n] * in(in_base + n * stride_d[d]);
                          result[d] = res_d;
                        }
                      {
                        Number res_d = 0;
                        for (int n = 0; n < n_columns; ++n)
                          res_d += co_shape_gradients(n * n_columns + last) * reg[dim - 1][n];
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
                  else
                    {
                      Number result = 0;
                      for (int d = 0; d < dim - 1; ++d)
                        {
                          const int point_base = q_point - idx_d[d] * stride_d[d];
                          const int grad_row   = elmnt_idx * n_q_points + point_base;

                          for (int n = 0; n < n_columns; ++n)
                            result += in(grad_row + n * stride_d[d], d) * reg[d][n];
                        }
                      for (int n = 0; n < n_columns; ++n)
                        result += reg[dim - 1][n] * co_shape_gradients(last * n_columns + n);

                      if constexpr (add)
                        out(elmnt_idx * n_q_points + q_point) += result;
                      else
                        out(elmnt_idx * n_q_points + q_point) = result;
                    }
                }
            });

        team_member.team_barrier();
      }

    private:
      const TeamHandle &team_member;
      ShapeDataType     shape_values;
      ShapeDataType     shape_gradients;
      ShapeDataType     co_shape_gradients;
      SharedView        temp;
      const int         n_elements_in_current_batch;
    };
  } // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
