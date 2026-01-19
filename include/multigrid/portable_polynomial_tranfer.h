#ifndef portable_polynomial_transfer_h
#define portable_polynomial_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include "base/portable_mg_transfer_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace p_mg_transfer
  {
    template <int dim, typename number>
    struct TransferData
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      TeamHandle team_member;

      // const unsigned int n_q_points;
      const int cell_index;

      const SharedView &prolongation_matrix;

      const typename MatrixFree<dim, number>::PrecomputedData
        precomputed_data_coarse;
      const typename MatrixFree<dim, number>::PrecomputedData
        precomputed_data_fine;

      const Kokkos::View<number **, MemorySpace::Default::kokkos_space>
        &weights;

      // const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
      //   &prolongation_matrix;


      // const Kokkos::View<int *, MemorySpace::Default::kokkos_space>
      //   &cell_lists_fine_to_coarse;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        &dirichlet_boundary_dofs_mask_coarse;

      // const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
      //   &boundary_dofs_mask_fine;

      /**
       * Memory for dof values.
       */
      SharedView &values_coarse;

      SharedView &values_fine;

      /**
       * Memory for temporary arrays required by kernel evaluation.
       */
      SharedView &scratch_pad;
    };


    template <int dim,
              int p_coarse,
              int p_fine,
              typename number,
              typename Functor>
    struct ApplyCellKernel
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;
      using SharedViewValues =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::
                       scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      ApplyCellKernel(
        Functor func,
        const typename MatrixFree<dim, number>::PrecomputedData
          precomputed_data_coarse,
        const typename MatrixFree<dim, number>::PrecomputedData
          precomputed_data_fine,
        const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
          prolongation_matrix_shared_memory,
        const Kokkos::View<number **, MemorySpace::Default::kokkos_space>
          weights,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
          dirichlet_boundary_dofs_mask_coarse,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                         &src,
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
        : func(func)
        , precomputed_data_coarse(precomputed_data_coarse)
        , precomputed_data_fine(precomputed_data_fine)
        , prolongation_matrix_shared_memory(prolongation_matrix_shared_memory)
        , weights(weights)
        , dirichlet_boundary_dofs_mask_coarse(
            dirichlet_boundary_dofs_mask_coarse)
        , src(src.get_values(), src.locally_owned_size())
        , dst(dst.get_values(), dst.locally_owned_size())
      {}

      Functor func;

      typename MatrixFree<dim, number>::PrecomputedData precomputed_data_coarse;
      typename MatrixFree<dim, number>::PrecomputedData precomputed_data_fine;


      const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
        prolongation_matrix_shared_memory;

      const Kokkos::View<number **, MemorySpace::Default::kokkos_space> weights;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dirichlet_boundary_dofs_mask_coarse;

      const DeviceVector<number> src;
      DeviceVector<number>       dst;


      // Provide the shared memory capacity. This function takes the team_size
      // as an argument, which allows team_size dependent allocations.
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        // std::size_t result =
        //   SharedViewValues::shmem_size(Functor::n_q_points) +
        //   //  precomputed_data.n_components) +
        //   SharedViewGradients::shmem_size(Functor::n_q_points, dim) +
        //   // precomputed_data.n_components) +
        //   SharedViewScratchPad::shmem_size(precomputed_data.scratch_pad_size);
        //   return result;

        return SharedViewValues::shmem_size(
          Functor::n_local_dofs_coarse +  // coarse dof values
          Functor::n_local_dofs_fine +    // fine dof values
          2 * Functor::n_local_dofs_fine  // at most two tmp vectors of at most
                                          // n_local_dofs_fine size
          + (p_fine + 1) * (p_coarse + 1) // prolongation matrix
        );
      }


      DEAL_II_HOST_DEVICE
      void
      operator()(const TeamHandle &team_member) const
      {
        const int cell_index = team_member.league_rank();

        SharedViewValues values_coarse(team_member.team_shmem(),
                                       Functor::n_local_dofs_coarse);

        SharedViewValues values_fine(team_member.team_shmem(),
                                     Functor::n_local_dofs_fine);

        SharedViewValues prolongation_matrix_device(team_member.team_shmem(),
                                                    (p_coarse + 1) *
                                                      (p_fine + 1));

        SharedViewValues scratch_pad(team_member.team_shmem(),
                                     Functor::n_local_dofs_fine);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, (p_coarse + 1) * (p_fine + 1)),
          [&](const int &i) {
            prolongation_matrix_device(i) =
              prolongation_matrix_shared_memory(i);
          });
        team_member.team_barrier();


        TransferData<dim, number> data{team_member,
                                       //  Functor::n_q_points,
                                       cell_index,
                                       prolongation_matrix_device,
                                       precomputed_data_coarse,
                                       precomputed_data_fine,
                                       weights,
                                       dirichlet_boundary_dofs_mask_coarse,
                                       values_coarse,
                                       values_fine,
                                       scratch_pad};

        DeviceVector<number> nonconstdst = dst;
        func(&data, src, nonconstdst);
      }
    };

    template <int dim, int p_coarse, int p_fine, typename number>
    class CellProlongationKernel : public EnableObserverPointer
    {
    public:
      // using DistributedVectorType =
      //   LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellProlongationKernel();


      // std::size_t
      // team_shmem_size(int team_size) const;

      DEAL_II_HOST_DEVICE void
      operator()(const TransferData<dim, number> *transfer_data,
                 const DeviceVector<number>      &src,
                 DeviceVector<number>            &dst) const;

      static const unsigned int n_local_dofs_coarse =
        Utilities::pow(p_coarse + 1, dim);
      static const unsigned int n_local_dofs_fine =
        Utilities::pow(p_fine + 1, dim);
    };

    template <int dim, int p_coarse, int p_fine, typename number>
    CellProlongationKernel<dim, p_coarse, p_fine, number>::
      CellProlongationKernel()
    {}
    //   TransferData<dim, number>   *transfer_data,
    //                          const DistributedVectorType &src,
    //                          DistributedVectorType       &dst)
    //   : transfer_data(transfer_data)
    //   , src(src.get_values(), src.locally_owned_size())
    //   , dst(dst.get_values(), dst.locally_owned_size())
    // {}

    // template <int dim, int p_coarse, int p_fine, typename number>
    // std::size_t
    // CellProlongationKernel<dim, p_coarse, p_fine, number>::team_shmem_size(
    //   int /*team_size*/) const
    // {
    //   return SharedView::shmem_size(
    //     n_local_dofs_coarse +           // coarse dof values
    //     n_local_dofs_fine +             // fine dof values
    //     2 * n_local_dofs_fine           // at most two tmp vectors of at most
    //                                     // n_local_dofs_fine size
    //     + (p_fine + 1) * (p_coarse + 1) // prolongation matrix
    //   );
    // }

    template <int dim, int p_coarse, int p_fine, typename number>
    DEAL_II_HOST_DEVICE void
    CellProlongationKernel<dim, p_coarse, p_fine, number>::operator()(
      const TransferData<dim, number> *transfer_data,
      const DeviceVector<number>      &src,
      DeviceVector<number>            &dst) const
    {
      const int   cell_id     = transfer_data->cell_index;
      const auto &team_member = transfer_data->team_member;

      const auto &precomputed_data_coarse =
        transfer_data->precomputed_data_coarse;
      const auto &precomputed_data_fine = transfer_data->precomputed_data_fine;

      const auto &prolongation_matrix = transfer_data->prolongation_matrix;

      auto &values_coarse = transfer_data->values_coarse;
      auto &values_fine   = transfer_data->values_fine;
      auto &scratch_pad   = transfer_data->scratch_pad;

      // auto scratch_for_eval =
      //   Kokkos::subview(scratch_pad,
      //                   Kokkos::make_pair(0, 2 * n_local_dofs_fine))


      // const int cell_index   = team_member.league_rank();
      // const int cell_index_coarse = cell_index_fine;
      // transfer_data.cell_lists_fine_to_coarse[cell_index_fine];

      // SharedView values_coarse(team_member.team_shmem(),
      // n_local_dofs_coarse); SharedView
      // values_fine(team_member.team_shmem(), n_local_dofs_fine);

      // read coarse dof values
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
        [&](const int &i) {
          if (transfer_data->dirichlet_boundary_dofs_mask_coarse(i, cell_id) ==
              numbers::invalid_unsigned_int)
            values_coarse(i) = 0.;
          else
            values_coarse(i) =
              src[precomputed_data_coarse.local_to_global(i, cell_id)];
        });
      team_member.team_barrier();


      // SharedView prolongation_matrix(team_member.team_shmem(),
      //                                (p_coarse + 1) * (p_fine + 1));



      // Kokkos::parallel_for(
      //   Kokkos::TeamThreadRange(team_member, (p_coarse + 1) * (p_fine + 1)),
      //   [&](const int &i) {
      //     prolongation_matrix(i) = transfer_data.prolongation_matrix(i);
      //   });
      // team_member.team_barrier();



      // Kokkos::parallel_for(
      //   Kokkos::TeamThreadRange(team_member, (p_coarse + 1) * (p_fine + 1)),
      //   [&](const int &i) {
      //     Kokkos::printf("%f, ", transfer_data.prolongation_matrix(i));
      //   });
      // team_member.team_barrier();


      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          // auto tmp =
          //   SharedView(team_member.team_shmem(), (p_coarse + 1) * (p_fine +
          //   1));

          constexpr int temp_size = (p_coarse + 1) * (p_fine + 1);

          auto tmp =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp_size));

          {
            constexpr int Ni = p_coarse + 1;
            constexpr int Nj = p_fine + 1;
            constexpr int Nk = p_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = p_fine + 1;

              const int base_coarse   = i * Nk;
              const int stride_coarse = 1;

              number sum =
                prolongation_matrix(base_kernel) * values_coarse(base_coarse);

              for (int k = 1; k < Nk; ++k)
                sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                       values_coarse(base_coarse + k * stride_coarse);

              const int index_tmp = i * Nj + j;

              tmp(index_tmp) = sum;
            });
          }
          team_member.team_barrier();

          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_fine + 1;
            constexpr int Nk = p_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = p_fine + 1;

              const int base_tmp   = i;
              const int stride_tmp = p_fine + 1;

              number sum = prolongation_matrix(base_kernel) * tmp(base_tmp);

              for (int k = 1; k < Nk; ++k)
                sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                       tmp(base_tmp + k * stride_tmp);

              const int index_fine    = i + j * Ni;
              values_fine(index_fine) = sum;
            });
          }

          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          // auto tmp1 =
          //   SharedView(team_member.team_shmem(),
          //              Utilities::pow(p_coarse + 1, 2) * (p_fine + 1));
          // auto tmp2 =
          //   SharedView(team_member.team_shmem(),
          //              Utilities::pow(p_fine + 1, 2) * (p_coarse + 1));
          constexpr int temp1_size =
                          Utilities::pow(p_coarse + 1, 2) * (p_fine + 1),
                        temp2_size =
                          Utilities::pow(p_fine + 1, 2) * (p_coarse + 1);

          auto tmp1 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp1_size));


          auto tmp2 =
            Kokkos::subview(scratch_pad,
                            Kokkos::make_pair(temp1_size,
                                              temp1_size + temp2_size));

          {
            constexpr int Ni = p_coarse + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nm = p_fine + 1;
            constexpr int Nk = p_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = p_fine + 1;

                const int base_coarse   = (i * Nj + j) * Nk;
                const int stride_coarse = 1;

                number sum =
                  prolongation_matrix(base_kernel) * values_coarse(base_coarse);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         values_coarse(base_coarse + k * stride_coarse);

                const int index_tmp1 = (i * Nj + j) * Nm + m;
                tmp1(index_tmp1)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nm = p_fine + 1;
            constexpr int Nk = p_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = p_fine + 1;

                const int base_tmp1   = i + j * Ni * Nk;
                const int stride_tmp1 = p_fine + 1;

                number sum = prolongation_matrix(base_kernel) * tmp1(base_tmp1);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         tmp1(base_tmp1 + k * stride_tmp1);

                const int index_tmp2 = i + (j * Nm + m) * Ni;
                tmp2(index_tmp2)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_fine + 1;
            constexpr int Nm = p_fine + 1;
            constexpr int Nk = p_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = p_fine + 1;

                const int base_tmp2   = i * Nj + j;
                const int stride_tmp2 = Utilities::pow(p_fine + 1, 2);

                number sum = prolongation_matrix(base_kernel) * tmp2(base_tmp2);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         tmp2(base_tmp2 + k * stride_tmp2);

                const int index_fine    = (i + m * Ni) * Nj + j;
                values_fine(index_fine) = sum;
              });
          }
          team_member.team_barrier();
        }

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_local_dofs_fine),
                           [&](const int &i) {
                             values_fine(i) *=
                               transfer_data->weights(i, cell_id);
                           });
      team_member.team_barrier();

      // distribute fine dofs values
      if (precomputed_data_fine.use_coloring)
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
          [&](const int &i) {
            // if (transfer_data.boundary_dofs_mask_fine(i,
            // cell_index_fine) !=
            //     numbers::invalid_unsigned_int)
            dst[precomputed_data_fine.local_to_global(i, cell_id)] +=
              values_fine(i);
          });
      else
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
          [&](const int &i) {
            // if (transfer_data.boundary_dofs_mask_fine(i, cell_index_fine) !=
            //     numbers::invalid_unsigned_int)
            Kokkos::atomic_add(
              &dst[precomputed_data_fine.local_to_global(i, cell_id)],
              values_fine(i));
          });
      team_member.team_barrier();
    }

    template <int dim, int p_coarse, int p_fine, typename number>
    class CellRestrictionKernel : public EnableObserverPointer
    {
    public:
      // using DistributedVectorType =
      //   LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


      CellRestrictionKernel();

      // std::size_t
      // team_shmem_size(int team_size) const;

      DEAL_II_HOST_DEVICE void
      operator()(const TransferData<dim, number> *transfer_data,
                 const DeviceVector<number>      &src,
                 DeviceVector<number>            &dst) const;

      static const unsigned int n_local_dofs_coarse =
        Utilities::pow(p_coarse + 1, dim);
      static const unsigned int n_local_dofs_fine =
        Utilities::pow(p_fine + 1, dim);
    };

    template <int dim, int p_coarse, int p_fine, typename number>
    CellRestrictionKernel<dim, p_coarse, p_fine, number>::
      CellRestrictionKernel()
    {}

    // template <int dim, int p_coarse, int p_fine, typename number>
    // std::size_t
    // CellRestrictionKernel<dim, p_coarse, p_fine, number>::team_shmem_size(
    //   int /*team_size*/) const
    // {
    //   return SharedView::shmem_size(
    //     n_local_dofs_coarse +           // coarse dof values
    //     n_local_dofs_fine +             // fine dof values
    //     2 * n_local_dofs_fine           // at most two tmp vectors of at most
    //                                     // n_local_dofs_fine size
    //     + (p_fine + 1) * (p_coarse + 1) // prolongation matrix
    //   );
    // }

    template <int dim, int p_coarse, int p_fine, typename number>
    DEAL_II_HOST_DEVICE void
    CellRestrictionKernel<dim, p_coarse, p_fine, number>::operator()(
      const TransferData<dim, number> *transfer_data,
      const DeviceVector<number>      &src,
      DeviceVector<number>            &dst) const
    {
      const int   cell_id     = transfer_data->cell_index;
      const auto &team_member = transfer_data->team_member;

      const auto &precomputed_data_coarse =
        transfer_data->precomputed_data_coarse;
      const auto &precomputed_data_fine = transfer_data->precomputed_data_fine;

      const auto &prolongation_matrix = transfer_data->prolongation_matrix;

      auto &values_coarse = transfer_data->values_coarse;
      auto &values_fine   = transfer_data->values_fine;
      auto &scratch_pad   = transfer_data->scratch_pad;

      // read fine dof values
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
        [&](const int &i) {
          values_fine(i) =
            src[precomputed_data_fine.local_to_global(i, cell_id)];
        });
      team_member.team_barrier();

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_local_dofs_fine),
                           [&](const int &i) {
                             values_fine(i) *=
                               transfer_data->weights(i, cell_id);
                           });
      team_member.team_barrier();

      // SharedView prolongation_matrix(team_member.team_shmem(),
      //                                (p_coarse + 1) * (p_fine + 1));

      // Kokkos::parallel_for(
      //   Kokkos::TeamThreadRange(team_member, (p_coarse + 1) * (p_fine + 1)),
      //   [&](const int &i) {
      //     prolongation_matrix(i) = transfer_data.prolongation_matrix(i);
      //   });
      // team_member.team_barrier();

      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          // auto tmp =
          //   SharedView(team_member.team_shmem(), (p_fine + 1) * (p_coarse +
          //   1));

          constexpr int temp_size = (p_coarse + 1) * (p_fine + 1);

          auto tmp =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp_size));

          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nk = p_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j * (p_fine + 1);
              const int stride_kernel = 1;

              const int base_fine   = i;
              const int stride_fine = p_fine + 1;

              number sum =
                prolongation_matrix(base_kernel) * values_fine(base_fine);

              for (int k = 1; k < Nk; ++k)
                sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                       values_fine(base_fine + k * stride_fine);

              const int index_tmp = i + j * Ni;

              tmp(index_tmp) = sum;
            });
          }
          team_member.team_barrier();

          {
            constexpr int Ni = p_coarse + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nk = p_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j * (p_fine + 1);
              const int stride_kernel = 1;

              const int base_tmp   = i * Nk;
              const int stride_tmp = 1;

              number sum = prolongation_matrix(base_kernel) * tmp(base_tmp);

              for (int k = 1; k < Nk; ++k)
                sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                       tmp(base_tmp + k * stride_tmp);

              const int index_coarse = i * Nj + j;

              values_coarse(index_coarse) = sum;
            });
          }
          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          // auto tmp1 =
          //   SharedView(team_member.team_shmem(),
          //              Utilities::pow(p_fine + 1, 2) * (p_coarse + 1));
          // auto tmp2 =
          //   SharedView(team_member.team_shmem(),
          //              Utilities::pow(p_coarse + 1, 2) * (p_fine + 1));

          constexpr int temp1_size =
                          Utilities::pow(p_fine + 1, 2) * (p_coarse + 1),
                        temp2_size =
                          Utilities::pow(p_coarse + 1, 2) * (p_fine + 1);

          auto tmp1 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp1_size));


          auto tmp2 =
            Kokkos::subview(scratch_pad,
                            Kokkos::make_pair(temp1_size,
                                              temp1_size + temp2_size));
          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_fine + 1;
            constexpr int Nm = p_coarse + 1;
            constexpr int Nk = p_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (p_fine + 1);
                const int stride_kernel = 1;

                const int base_fine   = i * Nj + j;
                const int stride_fine = Utilities::pow(p_fine + 1, 2);

                number sum =
                  prolongation_matrix(base_kernel) * values_fine(base_fine);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         values_fine(base_fine + k * stride_fine);

                const int index_tmp1 = (i + m * Ni) * Nj + j;
                tmp1(index_tmp1)     = sum;
              });
          }
          team_member.team_barrier();

          {
            constexpr int Ni = p_fine + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nm = p_coarse + 1;
            constexpr int Nk = p_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (p_fine + 1);
                const int stride_kernel = 1;

                const int base_tmp1   = i + j * Ni * Nk;
                const int stride_tmp1 = p_fine + 1;

                number sum = prolongation_matrix(base_kernel) * tmp1(base_tmp1);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         tmp1(base_tmp1 + k * stride_tmp1);

                const int index_tmp2 = i + (j * Nm + m) * Ni;
                tmp2(index_tmp2)     = sum;
              });
          }
          team_member.team_barrier();

          {
            constexpr int Ni = p_coarse + 1;
            constexpr int Nj = p_coarse + 1;
            constexpr int Nm = p_coarse + 1;
            constexpr int Nk = p_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);

            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (p_fine + 1);
                const int stride_kernel = 1;

                const int base_tmp2   = (i * Nj + j) * Nk;
                const int stride_tmp2 = 1;

                number sum = prolongation_matrix(base_kernel) * tmp2(base_tmp2);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix(base_kernel + k * stride_kernel) *
                         tmp2(base_tmp2 + k * stride_tmp2);

                const int index_coarse      = (i * Nj + j) * Nm + m;
                values_coarse(index_coarse) = sum;
              });
          }

          team_member.team_barrier();
        }

      // distribute coarse dofs values
      if (precomputed_data_coarse.use_coloring)
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
          [&](const int &i) {
            if (transfer_data->dirichlet_boundary_dofs_mask_coarse(i,
                                                                   cell_id) !=
                numbers::invalid_unsigned_int)
              dst[precomputed_data_coarse.local_to_global(i, cell_id)] +=
                values_coarse(i);
          });
      else
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
          [&](const int &i) {
            if (transfer_data->dirichlet_boundary_dofs_mask_coarse(i,
                                                                   cell_id) !=
                numbers::invalid_unsigned_int)
              Kokkos::atomic_add(
                &dst[precomputed_data_coarse.local_to_global(i, cell_id)],
                values_coarse(i));
          });
      team_member.team_barrier();
    }
  } // namespace p_mg_transfer

  template <int dim, int p_coarse, int p_fine, typename number>
  class PolynomialTransfer : public MGTransferBase<dim, number>
  {
  public:
    PolynomialTransfer();

    void
    prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const override;

    void
    restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const override;

    void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) override;

  private:
    void
    prolongate_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    void
    restrict_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;
    void
    setup_weights_and_boundary_dofs_masks();

    ObserverPointer<const MatrixFree<dim, number>> matrix_free_coarse;
    ObserverPointer<const MatrixFree<dim, number>> matrix_free_fine;

    ObserverPointer<const AffineConstraints<number>> constraints_fine;
    ObserverPointer<const AffineConstraints<number>> constraints_coarse;

    Kokkos::View<number *, MemorySpace::Default::kokkos_space>
      prolongation_matrix_1d;

    std::vector<Kokkos::View<int *, MemorySpace::Default::kokkos_space>>
      cell_lists_fine_to_coarse;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      boundary_dofs_mask_coarse;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      boundary_dofs_mask_fine;

    std::vector<Kokkos::View<number **, MemorySpace::Default::kokkos_space>>
      weights_view_kokkos;
  };

  template <int dim, int p_coarse, int p_fine, typename number>
  PolynomialTransfer<dim, p_coarse, p_fine, number>::PolynomialTransfer()
  {}

  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::prolongate_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not initialized correctly."));
    Assert(src.get_partitioner() ==
             matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not initialized correctly."));


    this->prolongate_and_add_internal(dst, src);

    // MemorySpace::Default::kokkos_space::execution_space exec;

    // const auto        &colored_graph = matrix_free_fine->get_colored_graph();
    // const unsigned int n_colors      = colored_graph.size();

    // if (matrix_free_fine->use_overlap_communication_computation())
    //   {
    //     auto do_color = [&](const unsigned int color) {
    //       const auto &gpu_data_coarse = matrix_free_coarse->get_data(color);
    //       const auto &gpu_data_fine   = matrix_free_fine->get_data(color);

    //       const auto n_cells = gpu_data_fine.n_cells;

    //       Kokkos::TeamPolicy<
    //         MemorySpace::Default::kokkos_space::execution_space>
    //         team_policy(exec, n_cells, Kokkos::AUTO);

    //       p_mg_transfer::TransferData<dim, number> transfer_data{
    //         gpu_data_coarse,
    //         gpu_data_fine,
    //         weights_view_kokkos[color],
    //         prolongation_matrix_1d,
    //         cell_lists_fine_to_coarse[color],
    //         boundary_dofs_mask_coarse[color],
    //         boundary_dofs_mask_fine[color]};

    //       p_mg_transfer::CellProlongationKernel<dim, p_coarse, p_fine,
    //       number>
    //         prolongator(transfer_data, src, dst);

    //       Kokkos::parallel_for("prolongate_" + std::to_string(color),
    //                            team_policy,
    //                            prolongator);
    //     };

    //     src.update_ghost_values_start(0);

    //     if (n_colors > 0 && colored_graph[0].size() > 0)
    //       do_color(0);

    //     src.update_ghost_values_finish();

    //     if (n_colors > 1 && colored_graph[1].size() > 0)
    //       {
    //         do_color(1);

    //         // We need a synchronization point because we don't want
    //         // device-aware MPI to start the MPI communication until the
    //         // kernel is done.
    //         Kokkos::fence();
    //       }
    //     dst.compress_start(0, VectorOperation::add);

    //     if (n_colors > 2 && colored_graph[2].size() > 0)
    //       do_color(2);

    //     dst.compress_finish(VectorOperation::add);
    //   }
    // else
    //   {
    //     src.update_ghost_values();

    //     for (unsigned int color = 0; color < n_colors; ++color)
    //       {
    //         const auto &gpu_data_coarse =
    //         matrix_free_coarse->get_data(color); const auto &gpu_data_fine =
    //         matrix_free_fine->get_data(color);

    //         const auto n_cells = gpu_data_fine.n_cells;

    //         p_mg_transfer::TransferData<dim, number> transfer_data{
    //           gpu_data_coarse,
    //           gpu_data_fine,
    //           weights_view_kokkos[color],
    //           prolongation_matrix_1d,
    //           cell_lists_fine_to_coarse[color],
    //           boundary_dofs_mask_coarse[color],
    //           boundary_dofs_mask_fine[color]};

    //         Kokkos::TeamPolicy<
    //           MemorySpace::Default::kokkos_space::execution_space>
    //           team_policy(exec, n_cells, Kokkos::AUTO);

    //         p_mg_transfer::CellProlongationKernel<dim, p_coarse, p_fine,
    //         number>
    //           prolongator(transfer_data, src, dst);

    //         Kokkos::parallel_for("prolongate_" + std::to_string(color),
    //                              team_policy,
    //                              prolongator);
    //       }
    //     dst.compress(VectorOperation::add);
    //   }
    // src.zero_out_ghost_values();

    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage(
             "Fine vector is not handled correclty after prolongation."));

    Assert(
      src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Coarse vector is not handled correclty after prolongation."));
  }

  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::restrict_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(dst.get_partitioner() ==
             matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not initialized correctly."));

    Assert(src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not initialized correctly."));

    this->restrict_and_add_internal(dst, src);

    // MemorySpace::Default::kokkos_space::execution_space exec;

    // const auto        &colored_graph = matrix_free_fine->get_colored_graph();
    // const unsigned int n_colors      = colored_graph.size();

    // if (matrix_free_fine->use_overlap_communication_computation())
    //   {
    //     auto do_color = [&](const unsigned int color) {
    //       const auto &gpu_data_coarse = matrix_free_coarse->get_data(color);
    //       const auto &gpu_data_fine   = matrix_free_fine->get_data(color);

    //       const auto n_cells = gpu_data_fine.n_cells;

    //       Kokkos::TeamPolicy<
    //         MemorySpace::Default::kokkos_space::execution_space>
    //         team_policy(exec, n_cells, Kokkos::AUTO);

    //       p_mg_transfer::TransferData<dim, number> transfer_data{
    //         gpu_data_coarse,
    //         gpu_data_fine,
    //         weights_view_kokkos[color],
    //         prolongation_matrix_1d,
    //         cell_lists_fine_to_coarse[color],
    //         boundary_dofs_mask_coarse[color],
    //         boundary_dofs_mask_fine[color]};

    //       p_mg_transfer::CellRestrictionKernel<dim, p_coarse, p_fine, number>
    //         restrictor(transfer_data, src, dst);

    //       Kokkos::parallel_for("restrict_" + std::to_string(color),
    //                            team_policy,
    //                            restrictor);
    //     };

    //     src.update_ghost_values_start(0);

    //     if (n_colors > 0 && colored_graph[0].size() > 0)
    //       do_color(0);

    //     src.update_ghost_values_finish();

    //     if (n_colors > 1 && colored_graph[1].size() > 0)
    //       {
    //         do_color(1);

    //         // We need a synchronization point because we don't want
    //         // device-aware MPI to start the MPI communication until the
    //         // kernel is done.
    //         Kokkos::fence();
    //       }
    //     dst.compress_start(0, VectorOperation::add);

    //     if (n_colors > 2 && colored_graph[2].size() > 0)
    //       do_color(2);

    //     dst.compress_finish(VectorOperation::add);
    //   }
    // else
    //   {
    //     src.update_ghost_values();

    //     for (unsigned int color = 0; color < n_colors; ++color)
    //       {
    //         const auto &gpu_data_coarse =
    //         matrix_free_coarse->get_data(color); const auto &gpu_data_fine =
    //         matrix_free_fine->get_data(color);

    //         const auto n_cells = gpu_data_fine.n_cells;

    //         p_mg_transfer::TransferData<dim, number> transfer_data{
    //           gpu_data_coarse,
    //           gpu_data_fine,
    //           weights_view_kokkos[color],
    //           prolongation_matrix_1d,
    //           cell_lists_fine_to_coarse[color],
    //           boundary_dofs_mask_coarse[color],
    //           boundary_dofs_mask_fine[color]};

    //         Kokkos::TeamPolicy<
    //           MemorySpace::Default::kokkos_space::execution_space>
    //           team_policy(exec, n_cells, Kokkos::AUTO);

    //         p_mg_transfer::CellRestrictionKernel<dim, p_coarse, p_fine,
    //         number>
    //           restrictor(transfer_data, src, dst);

    //         Kokkos::parallel_for("restrict_" + std::to_string(color),
    //                              team_policy,
    //                              restrictor);
    //       }
    //     dst.compress(VectorOperation::add);
    //   }
    // src.zero_out_ghost_values();

    Assert(
      dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Coarse vector is not handled correclty after restrtiction."));

    Assert(src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage(
             "Fine vector is not handled correclty after restrtiction."));
  }


  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::
    prolongate_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor =
      p_mg_transfer::CellProlongationKernel<dim, p_coarse, p_fine, number>;

    const auto &colored_graph = matrix_free_fine->get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free_fine->use_overlap_communication_computation())
      {
        // helper to process one color
        auto do_color = [&](const unsigned int color) {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;


          const auto &precomputed_data_coarse =
            matrix_free_coarse->get_data(color, 0);
          const auto &precomputed_data_fine =
            matrix_free_fine->get_data(color, 0);

          const auto n_cells = precomputed_data_fine.n_cells;

          Functor cell_prolongator;

          auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

          p_mg_transfer::ApplyCellKernel<dim, p_coarse, p_fine, number, Functor>
            apply_kernel(cell_prolongator,
                         precomputed_data_coarse,
                         precomputed_data_fine,
                         this->prolongation_matrix_1d,
                         this->weights_view_kokkos[color],
                         this->boundary_dofs_mask_coarse[color],
                         src,
                         dst);

          Kokkos::parallel_for("dealii::MatrixFree::prolongate_and_add_color_" +
                                 std::to_string(color),
                               team_policy,
                               apply_kernel);
        };

        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 &&
            matrix_free_fine->get_data(0, 0).n_cells > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 &&
            matrix_free_fine->get_data(1, 0).n_cells > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        dst.compress_start(0, VectorOperation::add);
        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 &&
            matrix_free_fine->get_data(2, 0).n_cells > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &precomputed_data_coarse =
              matrix_free_coarse->get_data(color, 0);
            const auto &precomputed_data_fine =
              matrix_free_fine->get_data(color, 0);

            if (precomputed_data_fine.n_cells > 0)
              {
                using TeamPolicy = Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>;


                const auto n_cells = precomputed_data_fine.n_cells;

                Functor cell_prolongator;

                auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

                p_mg_transfer::
                  ApplyCellKernel<dim, p_coarse, p_fine, number, Functor>
                    apply_kernel(cell_prolongator,
                                 precomputed_data_coarse,
                                 precomputed_data_fine,
                                 this->prolongation_matrix_1d,
                                 this->weights_view_kokkos[color],
                                 this->boundary_dofs_mask_coarse[color],
                                 src,
                                 dst);

                Kokkos::parallel_for(
                  "dealii::MatrixFree::prolongate_and_add_color " +
                    std::to_string(color),
                  team_policy,
                  apply_kernel);
              }
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor =
      p_mg_transfer::CellRestrictionKernel<dim, p_coarse, p_fine, number>;

    const auto &colored_graph = matrix_free_fine->get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free_fine->use_overlap_communication_computation())
      {
        // helper to process one color
        auto do_color = [&](const unsigned int color) {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;


          const auto &precomputed_data_coarse =
            matrix_free_coarse->get_data(color, 0);
          const auto &precomputed_data_fine =
            matrix_free_fine->get_data(color, 0);

          const auto n_cells = precomputed_data_fine.n_cells;

          Functor cell_restrictor;

          auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

          p_mg_transfer::ApplyCellKernel<dim, p_coarse, p_fine, number, Functor>
            apply_kernel(cell_restrictor,
                         precomputed_data_coarse,
                         precomputed_data_fine,
                         this->prolongation_matrix_1d,
                         this->weights_view_kokkos[color],
                         this->boundary_dofs_mask_coarse[color],
                         src,
                         dst);

          Kokkos::parallel_for("dealii::MatrixFree::restrict_and_add_color_" +
                                 std::to_string(color),
                               team_policy,
                               apply_kernel);
        };

        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 &&
            matrix_free_fine->get_data(0, 0).n_cells > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 &&
            matrix_free_fine->get_data(1, 0).n_cells > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        dst.compress_start(0, VectorOperation::add);
        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 &&
            matrix_free_fine->get_data(2, 0).n_cells > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &precomputed_data_coarse =
              matrix_free_coarse->get_data(color, 0);
            const auto &precomputed_data_fine =
              matrix_free_fine->get_data(color, 0);

            if (precomputed_data_fine.n_cells > 0)
              {
                using TeamPolicy = Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>;

                const auto n_cells = precomputed_data_fine.n_cells;

                Functor cell_restrictor;

                auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

                p_mg_transfer::
                  ApplyCellKernel<dim, p_coarse, p_fine, number, Functor>
                    apply_kernel(cell_restrictor,
                                 precomputed_data_coarse,
                                 precomputed_data_fine,
                                 this->prolongation_matrix_1d,
                                 this->weights_view_kokkos[color],
                                 this->boundary_dofs_mask_coarse[color],
                                 src,
                                 dst);

                Kokkos::parallel_for(
                  "dealii::MatrixFree::restrict_and_add_color " +
                    std::to_string(color),
                  team_policy,
                  apply_kernel);
              }
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::reinit(
    const MatrixFree<dim, number>   &mf_coarse,
    const MatrixFree<dim, number>   &mf_fine,
    const AffineConstraints<number> &constraints_coarse,
    const AffineConstraints<number> &constraints_fine)
  {
    // std::cout << "  P-transfer reinit start" << std::endl;
    this->matrix_free_coarse = &mf_coarse;
    this->matrix_free_fine   = &mf_fine;

    this->constraints_coarse = &constraints_coarse;
    this->constraints_fine   = &constraints_fine;

    auto &colored_graph_coarse = this->matrix_free_coarse->get_colored_graph();

    const auto &colored_graph_fine =
      this->matrix_free_fine->get_colored_graph();

    const unsigned int n_colors = colored_graph_fine.size();

    Assert(n_colors == colored_graph_coarse.size(),
           ExcMessage(
             "Coarse and fine levels must have the same number of colors"));

    FE_Q<1> fe_coarse_1d(p_coarse);
    FE_Q<1> fe_fine_1d(p_fine);

    std::vector<unsigned int> renumbering_fine(fe_fine_1d.n_dofs_per_cell());

    renumbering_fine[0] = 0;
    for (unsigned int i = 0; i < fe_fine_1d.dofs_per_line; ++i)
      renumbering_fine[i + fe_fine_1d.n_dofs_per_vertex()] =
        GeometryInfo<1>::vertices_per_cell * fe_fine_1d.n_dofs_per_vertex() + i;

    if (fe_fine_1d.n_dofs_per_vertex() > 0)
      renumbering_fine[fe_fine_1d.n_dofs_per_cell() -
                       fe_fine_1d.n_dofs_per_vertex()] =
        fe_fine_1d.n_dofs_per_vertex();

    std::vector<unsigned int> renumbering_coarse(
      fe_coarse_1d.n_dofs_per_cell());

    renumbering_coarse[0] = 0;
    for (unsigned int i = 0; i < fe_coarse_1d.dofs_per_line; ++i)
      renumbering_coarse[i + fe_coarse_1d.n_dofs_per_vertex()] =
        GeometryInfo<1>::vertices_per_cell * fe_coarse_1d.n_dofs_per_vertex() +
        i;

    if (fe_coarse_1d.n_dofs_per_vertex() > 0)
      renumbering_coarse[fe_coarse_1d.n_dofs_per_cell() -
                         fe_coarse_1d.n_dofs_per_vertex()] =
        fe_coarse_1d.n_dofs_per_vertex();

    FullMatrix<number> matrix(fe_fine_1d.n_dofs_per_cell(),
                              fe_coarse_1d.n_dofs_per_cell());

    FETools::get_projection_matrix(fe_coarse_1d, fe_fine_1d, matrix);

    // std::cout << "     get prolongation matrix" << std::endl;


    this->prolongation_matrix_1d =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("prolongation_matrix_1d_" +
                             std::to_string(p_coarse) + "_to_" +
                             std::to_string(p_fine),
                           Kokkos::WithoutInitializing),
        fe_coarse_1d.n_dofs_per_cell() * fe_fine_1d.n_dofs_per_cell());

    auto prolongation_matrix_1d_view =
      Kokkos::create_mirror_view(this->prolongation_matrix_1d);

    for (unsigned int i = 0, k = 0; i < fe_coarse_1d.n_dofs_per_cell(); ++i)
      for (unsigned int j = 0; j < fe_fine_1d.n_dofs_per_cell(); ++j, ++k)
        prolongation_matrix_1d_view[k] =
          matrix(renumbering_fine[j], renumbering_coarse[i]);

    // std::cout << "     prolongation matrix computed" << std::endl;

    Kokkos::deep_copy(this->prolongation_matrix_1d,
                      prolongation_matrix_1d_view);
    Kokkos::fence();

    // std::cout << "     prolongation matrix deep_copy()" << std::endl;

    // const auto &tria =
    //   this->matrix_free_coarse->get_dof_handler().get_triangulation();
    // std::vector<std::vector<unsigned int>> coarse_cell_ids(n_colors);

    // for (unsigned int color = 0; color < n_colors; ++color)
    //   {
    //     coarse_cell_ids[color].resize(tria.n_active_cells());

    //     const auto &graph = colored_graph_coarse[color];

    //     auto cell = graph.cbegin(), cell_end = graph.cend();

    //     for (int cell_id = 0; cell != cell_end; ++cell, ++cell_id)
    //       coarse_cell_ids[color][(*cell)->active_cell_index()] = cell_id;
    //   }

    // this->cell_lists_fine_to_coarse.clear();
    // this->cell_lists_fine_to_coarse.resize(n_colors);

    // for (unsigned int color = 0; color < n_colors; ++color)
    //   {
    //     const auto &graph = colored_graph_fine[color];

    //     this->cell_lists_fine_to_coarse[color] =
    //       Kokkos::View<int *, MemorySpace::Default::kokkos_space>(
    //         Kokkos::view_alloc("cell_lists_fine_to_coarse_" +
    //                              std::to_string(p_coarse) + "_to_" +
    //                              std::to_string(p_fine) + "_color_" +
    //                              std::to_string(color),
    //                            Kokkos::WithoutInitializing),
    //         graph.size());

    //     auto cell_list_host_view =
    //       Kokkos::create_mirror_view(this->cell_lists_fine_to_coarse[color]);

    //     auto cell = graph.cbegin(), cell_end = graph.cend();

    //     for (int cell_id = 0; cell != cell_end; ++cell, ++cell_id)
    //       cell_list_host_view[cell_id] =
    //         coarse_cell_ids[color][(*cell)->active_cell_index()];

    //     Kokkos::deep_copy(this->cell_lists_fine_to_coarse[color],
    //                       cell_list_host_view);
    //     Kokkos::fence();
    //   }


    // std::cout << "     cell_list_fine_to_coarse\n";


    setup_weights_and_boundary_dofs_masks();
  }

  template <int dim, int p_coarse, int p_fine, typename number>
  void
  PolynomialTransfer<dim, p_coarse, p_fine, number>::
    setup_weights_and_boundary_dofs_masks()
  {
    const auto &dof_handler_fine   = matrix_free_fine->get_dof_handler();
    const auto &dof_handler_coarse = matrix_free_coarse->get_dof_handler();
    const auto &fe_fine            = dof_handler_fine.get_fe();
    const auto &fe_coarse          = dof_handler_coarse.get_fe();

    const auto &colored_graph_fine   = matrix_free_fine->get_colored_graph();
    const auto &colored_graph_coarse = matrix_free_coarse->get_colored_graph();

    const unsigned int n_colors = colored_graph_fine.size();

    Assert(
      n_colors == colored_graph_coarse.size(),
      ExcMessage(
        "Portable matrix free objects must have the same number of colors"));

    const unsigned int n_dofs_per_cell_fine   = fe_fine.n_dofs_per_cell();
    const unsigned int n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();

    std::vector<unsigned int> lex_numbering_fine(n_dofs_per_cell_fine);
    std::vector<unsigned int> lex_numbering_coarse(n_dofs_per_cell_coarse);

    {
      const Quadrature<1> dummy_quadrature(
        std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

      shape_info.reinit(dummy_quadrature, fe_fine, 0);
      lex_numbering_fine = shape_info.lexicographic_numbering;
    }

    {
      const Quadrature<1> dummy_quadrature(
        std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

      shape_info.reinit(dummy_quadrature, fe_coarse, 0);
      lex_numbering_coarse = shape_info.lexicographic_numbering;
    }

    unsigned int n_cells_fine = 0;
    for (const auto &cell : dof_handler_fine.active_cell_iterators())
      if (cell->is_locally_owned())
        ++n_cells_fine;

    dealii::internal::MatrixFreeFunctions::
      ConstraintInfo<dim, VectorizedArray<number>, types::global_dof_index>
        constraint_info_fine;

    constraint_info_fine.reinit(dof_handler_fine, n_cells_fine);

    constraint_info_fine.set_locally_owned_indices(
      dof_handler_fine.locally_owned_dofs());

    std::vector<types::global_dof_index> local_dof_indices_fine(
      n_dofs_per_cell_fine);
    std::vector<types::global_dof_index> local_dof_indices_lex_fine(
      n_dofs_per_cell_fine);

    int cell_counter = 0;

    for (unsigned int color = 0; color < n_colors; ++color)
      for (const auto &cell : colored_graph_fine[color])
        {
          cell->get_dof_indices(local_dof_indices_fine);

          for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i)
            local_dof_indices_lex_fine[i] =
              local_dof_indices_fine[lex_numbering_fine[i]];

          constraint_info_fine.read_dof_indices(cell_counter,
                                                local_dof_indices_lex_fine,
                                                {});
          ++cell_counter;
        }

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine =
      constraint_info_fine.finalize(dof_handler_fine.get_mpi_communicator());

    // std::cout << "     constraint_info_finalized\n";


    LinearAlgebra::distributed::Vector<number> weight_vector;
    weight_vector.reinit(partitioner_fine);

    for (const auto i : constraint_info_fine.dof_indices)
      weight_vector.local_element(i) += 1.0;

    weight_vector.compress(VectorOperation::add);

    for (unsigned int i = 0; i < weight_vector.locally_owned_size(); ++i)
      if (weight_vector.local_element(i) > 0)
        weight_vector.local_element(i) = 1.0 / weight_vector.local_element(i);

    // ... clear constrained indices
    for (const auto &constrained_dofs : constraints_fine->get_lines())
      if (weight_vector.locally_owned_elements().is_element(
            constrained_dofs.index))
        weight_vector[constrained_dofs.index] = 0.0;

    weight_vector.update_ghost_values();

    // std::cout << "     compute weights\n";

    weights_view_kokkos.clear();
    weights_view_kokkos.resize(n_colors);



    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph_fine[color].size() > 0)
          {
            const auto &mf_data_fine = matrix_free_fine->get_data(color);
            const auto &graph        = colored_graph_fine[color];

            weights_view_kokkos[color] =
              Kokkos::View<number **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("weights_" + std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_dofs_per_cell_fine,
                mf_data_fine.n_cells);

            auto weights_view_host =
              Kokkos::create_mirror_view(weights_view_kokkos[color]);

            auto cell = graph.cbegin(), end_cell = graph.cend();

            for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id)
              {
                (*cell)->get_dof_indices(local_dof_indices_fine);

                for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i)
                  {
                    types::global_dof_index dof_index_lex =
                      local_dof_indices_fine[lex_numbering_fine[i]];
                    weights_view_host(i, cell_id) =
                      weight_vector[dof_index_lex];
                  }
              }
            Kokkos::deep_copy(weights_view_kokkos[color], weights_view_host);
            Kokkos::fence();
          }

        // std::cout << "     deep_copy weights\n";
      }

    // setup boundary dofs masks
    std::vector<types::global_dof_index> local_dof_indices_coarse(
      n_dofs_per_cell_coarse);

    this->boundary_dofs_mask_coarse.clear();
    this->boundary_dofs_mask_coarse.resize(n_colors);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph_fine[color].size() > 0)
          {
            const auto &mf_data_coarse = matrix_free_coarse->get_data(color);
            ;
            const auto &graph = colored_graph_coarse[color];

            this->boundary_dofs_mask_coarse[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("boundary_dofs_mask_coarse_" +
                                     std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_dofs_per_cell_coarse,
                mf_data_coarse.n_cells);

            auto dofs_mask_host = Kokkos::create_mirror_view(
              this->boundary_dofs_mask_coarse[color]);

            auto cell = graph.cbegin(), end_cell = graph.cend();

            for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id)
              {
                (*cell)->get_dof_indices(local_dof_indices_coarse);

                for (unsigned int i = 0; i < n_dofs_per_cell_coarse; ++i)
                  {
                    const auto global_dof =
                      local_dof_indices_coarse[lex_numbering_coarse[i]];
                    if (constraints_coarse->is_constrained(global_dof))
                      dofs_mask_host(i, cell_id) =
                        numbers::invalid_unsigned_int;
                    else
                      dofs_mask_host(i, cell_id) = global_dof;
                  }
              }
            Kokkos::deep_copy(this->boundary_dofs_mask_coarse[color],
                              dofs_mask_host);
            Kokkos::fence();
          }
      }

    // std::cout << "     boundary_masks_coarse\n";


    this->boundary_dofs_mask_fine.clear();
    this->boundary_dofs_mask_fine.resize(n_colors);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph_fine[color].size() > 0)
          {
            const auto &mf_data_fine = matrix_free_fine->get_data(color);
            
            const auto &graph = colored_graph_fine[color];

            this->boundary_dofs_mask_fine[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("boundary_dofs_mask_fine_" +
                                     std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_dofs_per_cell_fine,
                mf_data_fine.n_cells);

            auto dofs_mask_host =
              Kokkos::create_mirror_view(this->boundary_dofs_mask_fine[color]);

            auto cell = graph.cbegin(), end_cell = graph.cend();

            for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id)
              {
                (*cell)->get_dof_indices(local_dof_indices_fine);

                for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i)
                  {
                    const auto global_dof =
                      local_dof_indices_fine[lex_numbering_fine[i]];
                    if (constraints_fine->is_constrained(global_dof))
                      dofs_mask_host(i, cell_id) =
                        numbers::invalid_unsigned_int;
                    else
                      dofs_mask_host(i, cell_id) = global_dof;
                  }
              }
            Kokkos::deep_copy(this->boundary_dofs_mask_fine[color],
                              dofs_mask_host);
            Kokkos::fence();
          }
      }
    // std::cout << "     boundary_masks_fine\n";
  }

  class PolynomialTransferDispatchFactory
  {
  public:
    static constexpr unsigned int max_degree = 9;

    template <typename Runner>
    static bool
    dispatch(const int runtime_p_coarse,
             const int runtime_p_fine,
             Runner   &runner)
    {
      return recursive_dispatch<Runner, max_degree, max_degree>(
        runtime_p_coarse, runtime_p_fine, runner);
    }

  private:
    template <typename Runner,
              unsigned int degree_coarse,
              unsigned int degree_fine>
    static bool
    recursive_dispatch(const int runtime_p_coarse,
                       const int runtime_p_fine,
                       Runner   &runner)
    {
      if (runtime_p_fine == degree_fine)
        {
          if (runtime_p_coarse == degree_coarse)
            {
              runner.template run<degree_coarse, degree_fine>();
              return true;
            }
          else if constexpr (degree_coarse > 1)
            {
              return recursive_dispatch<Runner, degree_coarse - 1, degree_fine>(
                runtime_p_coarse, runtime_p_fine, runner);
            }
          else
            {
              return false;
            }
        }
      else if constexpr (degree_fine > 1)
        {
          return recursive_dispatch<Runner, degree_fine - 2, degree_fine - 1>(
            runtime_p_coarse, runtime_p_fine, runner);
        }

      else
        {
          return false;
        }
    }
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif