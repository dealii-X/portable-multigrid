#ifndef portable_local_laplace_operator_h
#define portable_local_laplace_operator_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  struct CellData
  {
    using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients = Kokkos::View<
      number **,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    TeamHandle team_member;

    const unsigned int n_q_points;
    const unsigned int cell_index;

    const typename MatrixFree<dim, number>::PrecomputedData &precomputed_data;

    const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
      &dof_indices;

    /**
     * Memory for dof and quad values.
     */
    ViewValues &values;

    /**
     * Memory for computed gradients in reference coordinate system.
     */
    ViewGradients &gradients;

    /**
     * Memory for temporary arrays required by evaluation and integration.
     */
    ViewValues &scratch_pad;
  };

  namespace internal
  {
    template <int dim, typename number, typename Functor>
    struct ApplyCellKernel
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;
      using SharedViewValues =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using SharedViewGradients =
        Kokkos::View<number **,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using SharedViewScratchPad =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      ApplyCellKernel(
        Functor func,
        const typename MatrixFree<dim, number>::PrecomputedData
          precomputed_data,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
          dof_indices,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                         &src,
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
        : func(func)
        , precomputed_data(precomputed_data)
        , dof_indices(dof_indices)
        , src(src.get_values(), src.locally_owned_size())
        , dst(dst.get_values(), dst.locally_owned_size())
      {}

      Functor func;

      const typename MatrixFree<dim, number>::PrecomputedData precomputed_data;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices;

      const DeviceVector<number> src;
      DeviceVector<number>       dst;

      // Provide the shared memory capacity. This function takes the team_size
      // as an argument, which allows team_size dependent allocations.
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        std::size_t result =
          SharedViewValues::shmem_size(Functor::n_q_points) +
          SharedViewGradients::shmem_size(Functor::n_q_points, dim) +
          SharedViewScratchPad::shmem_size(precomputed_data.scratch_pad_size);

        return result;
      }


      DEAL_II_HOST_DEVICE
      void
      operator()(const TeamHandle &team_member) const
      {
        const unsigned int cell_index = team_member.league_rank();

        SharedViewValues values(team_member.team_shmem(), Functor::n_q_points);
        SharedViewGradients  gradients(team_member.team_shmem(),
                                      Functor::n_q_points,
                                      dim);
        SharedViewScratchPad scratch_pad(team_member.team_shmem(),
                                         precomputed_data.scratch_pad_size);

        CellData<dim, number> data{team_member,
                                   Functor::n_q_points,
                                   cell_index,
                                   precomputed_data,
                                   dof_indices,
                                   values,
                                   gradients,
                                   scratch_pad};

        DeviceVector<number> nonconstdst = dst;
        func(&data, src, nonconstdst);
      }
    };

    template <int dim, typename number, typename Functor>
    struct ApplyCellKernelRange
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;
      using SharedViewValues =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using SharedViewGradients =
        Kokkos::View<number **,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using SharedViewScratchPad =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      ApplyCellKernelRange(
        Functor func,
        const typename MatrixFree<dim, number>::PrecomputedData
          precomputed_data,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
          dof_indices,
        const Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
          cell_range_ids,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                         &src,
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
        : func(func)
        , precomputed_data(precomputed_data)
        , dof_indices(dof_indices)
        , cell_range_ids(cell_range_ids)
        , src(src.get_values(), src.locally_owned_size())
        , dst(dst.get_values(), dst.locally_owned_size())
      {}

      Functor func;

      const typename MatrixFree<dim, number>::PrecomputedData precomputed_data;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices;

      const Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
        cell_range_ids;

      const DeviceVector<number> src;
      DeviceVector<number>       dst;

      // Provide the shared memory capacity. This function takes the team_size
      // as an argument, which allows team_size dependent allocations.
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        std::size_t result =
          SharedViewValues::shmem_size(Functor::n_q_points) +
          SharedViewGradients::shmem_size(Functor::n_q_points, dim) +
          SharedViewScratchPad::shmem_size(precomputed_data.scratch_pad_size);

        return result;
      }


      DEAL_II_HOST_DEVICE
      void
      operator()(const TeamHandle &team_member) const
      {
        const unsigned int thread_id = team_member.league_rank();

        const unsigned int cell_index = cell_range_ids[thread_id];

        SharedViewValues values(team_member.team_shmem(), Functor::n_q_points);
        SharedViewGradients  gradients(team_member.team_shmem(),
                                      Functor::n_q_points,
                                      dim);
        SharedViewScratchPad scratch_pad(team_member.team_shmem(),
                                         precomputed_data.scratch_pad_size);

        CellData<dim, number> data{team_member,
                                   Functor::n_q_points,
                                   cell_index,
                                   precomputed_data,
                                   dof_indices,
                                   values,
                                   gradients,
                                   scratch_pad};

        DeviceVector<number> nonconstdst = dst;
        func(&data, src, nonconstdst);
      }
    };

  } // namespace internal

  template <int dim, int fe_degree, typename number>
  class LocalLaplaceOperator
  {
  public:
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim);

    LocalLaplaceOperator()
    {}

    DEAL_II_HOST_DEVICE void
    operator()(const CellData<dim, number> *data,
               const DeviceVector<number>  &src,
               DeviceVector<number>        &dst) const;
  };

  template <int dim, int fe_degree, typename number>
  DEAL_II_HOST_DEVICE void
  LocalLaplaceOperator<dim, fe_degree, number>::operator()(
    const CellData<dim, number> *data,
    const DeviceVector<number>  &src,
    DeviceVector<number>        &dst) const
  {
    const auto &precomputed_data = data->precomputed_data;
    const int   cell_id          = data->cell_index;
    const auto &team_member      = data->team_member;

    auto &values      = data->values;
    auto &gradients   = data->gradients;
    auto &scratch_pad = data->scratch_pad;

    // 1. read dof values
    {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(data->team_member,
                                                   n_local_dofs),
                           [&](const int &i) {
                             if (data->dof_indices(i, cell_id) ==
                                 numbers::invalid_unsigned_int)
                               values(i) = 0.;
                             else
                               values(i) = src[data->dof_indices(i, cell_id)];
                           });

      data->team_member.team_barrier();
    }

    // 2. define scratch pad for the evaluation
    constexpr int scratch_size = Utilities::pow(fe_degree + 1, dim);
    auto          scratch_for_eval =
      Kokkos::subview(scratch_pad, Kokkos::make_pair(0, scratch_size));

    // 3. initialize tensor-product kernel
    internal::EvaluatorTensorProduct<
      internal::EvaluatorVariant::evaluate_general,
      dim,
      fe_degree + 1,
      fe_degree + 1,
      number>
      eval(team_member,
           precomputed_data.shape_values,
           precomputed_data.shape_gradients,
           precomputed_data.co_shape_gradients,
           scratch_for_eval);


    // 4.evaluate the kernel using sum factorization
    {
      // 4a.transform to the collocation space
      eval.template values<0, true, false, true>(values, values);
      if constexpr (dim > 1)
        eval.template values<1, true, false, true>(values, values);
      if constexpr (dim > 2)
        eval.template values<2, true, false, true>(values, values);


      // 4b. evaluate gradients in the colloction space
      eval.template co_gradients<0, true, false, false>(
        values, Kokkos::subview(gradients, Kokkos::ALL, 0));
      if constexpr (dim > 1)
        eval.template co_gradients<1, true, false, false>(
          values, Kokkos::subview(gradients, Kokkos::ALL, 1));
      if constexpr (dim > 2)
        eval.template co_gradients<2, true, false, false>(
          values, Kokkos::subview(gradients, Kokkos::ALL, 2));

      team_member.team_barrier();
    }

    // 5.compute Laplace kernel at each quadrature point
    {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_q_points),
        [&](const int &q_point) {
          // 5a. get gradient
          Tensor<1, dim, number> grad;
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp +=
                  precomputed_data.inv_jacobian(q_point, cell_id, d_2, d_1) *
                  gradients(q_point, d_2);
              grad[d_1] = tmp;
            }

          // 5b. submit gradient
          for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
            {
              number tmp = 0.;
              for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                tmp +=
                  precomputed_data.inv_jacobian(q_point, cell_id, d_1, d_2) *
                  grad[d_2];
              gradients(q_point, d_1) =
                tmp * precomputed_data.JxW(q_point, cell_id);
            }
        });

      team_member.team_barrier();
    }

    // 6. integrate using time factorization
    {
      // 6a. apply derivatives in collocation space
      if constexpr (dim == 1)
        eval.template co_gradients<0, false, false, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 2), values);
      else if constexpr (dim == 2)
        {
          eval.template co_gradients<1, false, false, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 1), values);
          eval.template co_gradients<0, false, true, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 0), values);
        }
      else if constexpr (dim == 3)
        {
          eval.template co_gradients<2, false, false, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 2), values);
          eval.template co_gradients<1, false, true, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 1), values);
          eval.template co_gradients<0, false, true, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 0), values);
        }

      // 6b. transform back to the original space
      if constexpr (dim > 2)
        eval.template values<2, false, false, true>(values, values);
      if constexpr (dim > 1)
        eval.template values<1, false, false, true>(values, values);
      eval.template values<0, false, false, true>(values, values);

      team_member.team_barrier();
    }

    // 7.distribute dofs
    {
      if (precomputed_data.use_coloring)
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_local_dofs),
                             [&](const int &i) {
                               if (data->dof_indices(i, cell_id) !=
                                   numbers::invalid_unsigned_int)
                                 dst[data->dof_indices(i, cell_id)] +=
                                   values(i);
                             });
      else
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs),
          [&](const int &i) {
            if (data->dof_indices(i, cell_id) != numbers::invalid_unsigned_int)
              Kokkos::atomic_add(&dst[data->dof_indices(i, cell_id)],
                                 values(i));
          });
    }
  }

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif