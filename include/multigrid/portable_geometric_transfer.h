#ifndef portable_geometric_transfer_h
#define portable_geometric_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include "base/portable_mg_transfer_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  namespace h_mg_transfer
  {

    /**  FIXME: MGTransferScheme for the moment assumes only global refinement,
     * not local. */
    /**
     * A multigrid transfer scheme. A multrigrid transfer class can have
     * different transfer transfer_schemes to enable p-adaptivity (one transfer
     * scheme per polynomial degree pair) and to enable global coarsening (one
     * transfer scheme for transfer between children and parent cells, as well
     * as, one transfer scheme for cells that are not refined).
     */
    template <int dim, int fe_degree, typename number>
    struct MGTransferScheme
    {
      /**
       * Number of coarse cells.
       */
      unsigned int n_coarse_cells;

      /**
       * Polynomial degree of the finite element of a coarse cell.
       */
      static const int degree_coarse = fe_degree;

      /**
       * "Polynomial degree" of the finite element of the union of all children
       * of a coarse cell, i.e., actually `degree_fine * 2 + 1` if a cell is
       * refined.
       */
      static const int degree_fine = 2 * fe_degree;

      /**
       * Number of degrees of freedom of a coarse cell.
       *
       * @note For tensor-product elements, the value equals
       *   `n_components * (degree_coarse + 1)^dim`.
       */
      static const unsigned int n_dofs_per_cell_coarse =
        Utilities::pow(fe_degree + 1, dim);

      /**
       * Number of degrees of freedom of fine cell.
       *
       * @note For tensor-product elements, the value equals
       *   `n_components * (n_dofs_per_cell_fine + 1)^dim`.
       */
      static const unsigned int n_dofs_per_cell_fine =
        Utilities::pow(2 * fe_degree + 1, dim);

      /**
       * Prolongation matrix used for the prolongate_and_add() and
       * restrict_and_add() functions.
       */

      Kokkos::View<number *, MemorySpace::Default::kokkos_space>
        prolongation_matrix_shared_memory;

      Kokkos::View<number **, MemorySpace::Default::kokkos_space> weights;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_coarse;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_fine;
    };

    template <int dim, typename number>
    struct TransferCellData
    {
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      TeamHandle team_member;

      const int cell_index;

      const SharedView &prolongation_matrix;

      const Kokkos::View<number **, MemorySpace::Default::kokkos_space>
        &weights;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        &dof_indices_coarse;


      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        &dof_indices_fine;

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

    template <int dim, int fe_degree, typename number, typename Functor>
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
        const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
          prolongation_matrix_shared_memory,
        const Kokkos::View<number **, MemorySpace::Default::kokkos_space>
          weights,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
          dof_indices_coarse,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
          dof_indices_fine,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                         &src,
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
        : func(func)
        , prolongation_matrix_shared_memory(prolongation_matrix_shared_memory)
        , weights(weights)
        , dof_indices_coarse(dof_indices_coarse)
        , dof_indices_fine(dof_indices_fine)
        , src(src.get_values(), src.locally_owned_size())
        , dst(dst.get_values(), dst.locally_owned_size())
      {}

      Functor func;

      const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
        prolongation_matrix_shared_memory;

      const Kokkos::View<number **, MemorySpace::Default::kokkos_space> weights;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_coarse;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_fine;

      const DeviceVector<number> src;
      DeviceVector<number>       dst;

      // Provide the shared memory capacity. This function takes the team_size
      // as an argument, which allows team_size dependent allocations.
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        return SharedViewValues::shmem_size(
          Functor::n_dofs_per_cell_coarse + // coarse dof values
          Functor::n_dofs_per_cell_fine +   // fine dof values
          2 * Functor::n_dofs_per_cell_fine // at most two tmp vectors of at
                                            // most n_dofs_per_cell_fine size
          + (Functor::degree_coarse + 1) *
              (Functor::degree_fine + 1) // prolongation matrix
        );
      }


      DEAL_II_HOST_DEVICE
      void
      operator()(const TeamHandle &team_member) const
      {
        const int cell_index = team_member.league_rank();

        SharedViewValues values_coarse(team_member.team_shmem(),
                                       Functor::n_dofs_per_cell_coarse);

        SharedViewValues values_fine(team_member.team_shmem(),
                                     Functor::n_dofs_per_cell_fine);

        SharedViewValues prolongation_matrix_device(
          team_member.team_shmem(),
          (Functor::degree_coarse + 1) * (Functor::degree_fine + 1));

        SharedViewValues scratch_pad(team_member.team_shmem(),
                                     Functor::n_dofs_per_cell_fine * 2);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member,
                                  (Functor::degree_coarse + 1) *
                                    (Functor::degree_fine + 1)),
          [&](const int &i) {
            prolongation_matrix_device(i) =
              prolongation_matrix_shared_memory(i);
          });
        team_member.team_barrier();


        TransferCellData<dim, number> data{team_member,
                                           cell_index,
                                           prolongation_matrix_device,
                                           weights,
                                           dof_indices_coarse,
                                           dof_indices_fine,
                                           values_coarse,
                                           values_fine,
                                           scratch_pad};

        DeviceVector<number> nonconstdst = dst;
        func(&data, src, nonconstdst);
      }
    };

    template <int dim, int fe_degree, typename number>
    class CellProlongationKernel : public EnableObserverPointer
    {
    public:
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellProlongationKernel();


      DEAL_II_HOST_DEVICE void
      operator()(const TransferCellData<dim, number> *cell_data,
                 const DeviceVector<number>          &src,
                 DeviceVector<number>                &dst) const;

      static const unsigned int degree_coarse =
        MGTransferScheme<dim, fe_degree, number>::degree_coarse;
      static const unsigned int degree_fine =
        MGTransferScheme<dim, fe_degree, number>::degree_fine;

      static const unsigned int n_dofs_per_cell_coarse =
        MGTransferScheme<dim, fe_degree, number>::n_dofs_per_cell_coarse;

      static const unsigned int n_dofs_per_cell_fine =
        MGTransferScheme<dim, fe_degree, number>::n_dofs_per_cell_fine;
    };

    template <int dim, int fe_degree, typename number>
    CellProlongationKernel<dim, fe_degree, number>::CellProlongationKernel()
    {}


    template <int dim, int fe_degree, typename number>
    DEAL_II_HOST_DEVICE void
    CellProlongationKernel<dim, fe_degree, number>::operator()(
      const TransferCellData<dim, number> *cell_data,
      const DeviceVector<number>          &src,
      DeviceVector<number>                &dst) const
    {
      const int   cell_index  = cell_data->cell_index;
      const auto &team_member = cell_data->team_member;

      const auto &prolongation_matrix_scratch = cell_data->prolongation_matrix;

      const auto &dof_indices_coarse = cell_data->dof_indices_coarse;
      const auto &dof_indices_fine   = cell_data->dof_indices_fine;

      auto &values_coarse = cell_data->values_coarse;
      auto &values_fine   = cell_data->values_fine;
      auto &scratch_pad   = cell_data->scratch_pad;

      // read coarse dof values
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_dofs_per_cell_coarse),
                           [&](const int &i) {
                             const unsigned int dof_index =
                               dof_indices_coarse(i, cell_index);
                             if (dof_index != numbers::invalid_unsigned_int)
                               values_coarse(i) = src[dof_index];
                             else
                               values_coarse(i) = 0.;
                           });
      team_member.team_barrier();

      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          constexpr int temp_size = (degree_coarse + 1) * (degree_fine + 1);
          auto          tmp =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp_size));

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = degree_fine + 1;

              const int base_coarse   = i * Nk;
              const int stride_coarse = 1;

              number sum = prolongation_matrix_scratch(base_kernel) *
                           values_coarse(base_coarse);

              for (int k = 1; k < Nk; ++k)
                sum +=
                  prolongation_matrix_scratch(base_kernel + k * stride_kernel) *
                  values_coarse(base_coarse + k * stride_coarse);

              const int index_tmp = i * Nj + j;

              tmp(index_tmp) = sum;
            });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = degree_fine + 1;

              const int base_tmp   = i;
              const int stride_tmp = degree_fine + 1;

              number sum =
                prolongation_matrix_scratch(base_kernel) * tmp(base_tmp);

              for (int k = 1; k < Nk; ++k)
                sum +=
                  prolongation_matrix_scratch(base_kernel + k * stride_kernel) *
                  tmp(base_tmp + k * stride_tmp);

              const int index_fine    = i + j * Ni;
              values_fine(index_fine) = sum;
            });
          }

          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          constexpr int tmp1_size =
            Utilities::pow(degree_coarse + 1, 2) * (degree_fine + 1);
          constexpr int tmp2_size =
            Utilities::pow(degree_fine + 1, 2) * (degree_coarse + 1);
          auto tmp1 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp1_size));
          auto tmp2 = Kokkos::subview(scratch_pad,
                                      Kokkos::make_pair(tmp1_size,
                                                        tmp1_size + tmp2_size));

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nm = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = degree_fine + 1;

                const int base_coarse   = (i * Nj + j) * Nk;
                const int stride_coarse = 1;

                number sum = prolongation_matrix_scratch(base_kernel) *
                             values_coarse(base_coarse);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix_scratch(base_kernel +
                                                     k * stride_kernel) *
                         values_coarse(base_coarse + k * stride_coarse);

                const int index_tmp1 = (i * Nj + j) * Nm + m;
                tmp1(index_tmp1)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nm = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m) {
                                   const int base_kernel   = m;
                                   const int stride_kernel = degree_fine + 1;

                                   const int base_tmp1   = i + j * Ni * Nk;
                                   const int stride_tmp1 = degree_fine + 1;

                                   number sum =
                                     prolongation_matrix_scratch(base_kernel) *
                                     tmp1(base_tmp1);

                                   for (int k = 1; k < Nk; ++k)
                                     sum += prolongation_matrix_scratch(
                                              base_kernel + k * stride_kernel) *
                                            tmp1(base_tmp1 + k * stride_tmp1);

                                   const int index_tmp2 = i + (j * Nm + m) * Ni;
                                   tmp2(index_tmp2)     = sum;
                                 });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nm = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = degree_fine + 1;

                const int base_tmp2   = i * Nj + j;
                const int stride_tmp2 = Utilities::pow(degree_fine + 1, 2);
                number    sum =
                  prolongation_matrix_scratch(base_kernel) * tmp2(base_tmp2);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix_scratch(base_kernel +
                                                     k * stride_kernel) *
                         tmp2(base_tmp2 + k * stride_tmp2);

                const int index_fine    = (i + m * Ni) * Nj + j;
                values_fine(index_fine) = sum;
              });
          }
          team_member.team_barrier();
        }

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_dofs_per_cell_fine),
                           [&](const int &i) {
                             values_fine(i) *=
                               cell_data->weights(i, cell_index);
                           });
      team_member.team_barrier();


      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_fine),
        [&](const int &i) {
          const unsigned int dof_index = dof_indices_fine(i, cell_index);
          Kokkos::atomic_add(&dst[dof_index], values_fine(i));
        });
      team_member.team_barrier();
    }

    template <int dim, int fe_degree, typename number>
    class CellRestrictionKernel : public EnableObserverPointer
    {
    public:
      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellRestrictionKernel();



      DEAL_II_HOST_DEVICE void
      operator()(const TransferCellData<dim, number> *cell_data,
                 const DeviceVector<number>          &src,
                 DeviceVector<number>                &dst) const;

      static const unsigned int degree_coarse =
        MGTransferScheme<dim, fe_degree, number>::degree_coarse;
      static const unsigned int degree_fine =
        MGTransferScheme<dim, fe_degree, number>::degree_fine;

      static const unsigned int n_dofs_per_cell_coarse =
        MGTransferScheme<dim, fe_degree, number>::n_dofs_per_cell_coarse;
      static const unsigned int n_dofs_per_cell_fine =
        MGTransferScheme<dim, fe_degree, number>::n_dofs_per_cell_fine;
    };

    template <int dim, int fe_degree, typename number>
    CellRestrictionKernel<dim, fe_degree, number>::CellRestrictionKernel()
    {}

    template <int dim, int fe_degree, typename number>
    DEAL_II_HOST_DEVICE void
    CellRestrictionKernel<dim, fe_degree, number>::operator()(
      const TransferCellData<dim, number> *cell_data,
      const DeviceVector<number>          &src,
      DeviceVector<number>                &dst) const
    {
      const int   cell_index  = cell_data->cell_index;
      const auto &team_member = cell_data->team_member;

      const auto &prolongation_matrix_scratch = cell_data->prolongation_matrix;

      const auto &dof_indices_coarse = cell_data->dof_indices_coarse;
      const auto &dof_indices_fine   = cell_data->dof_indices_fine;

      auto &values_coarse = cell_data->values_coarse;
      auto &values_fine   = cell_data->values_fine;
      auto &scratch_pad   = cell_data->scratch_pad;

      // read fine dof values
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_dofs_per_cell_fine),
                           [&](const int &i) {
                             values_fine(i) =
                               src[dof_indices_fine(i, cell_index)];
                           });
      team_member.team_barrier();

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                   n_dofs_per_cell_fine),
                           [&](const int &i) {
                             values_fine(i) *=
                               cell_data->weights(i, cell_index);
                           });
      team_member.team_barrier();

      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          constexpr int tmp_size = (degree_coarse + 1) * (degree_fine + 1);

          auto tmp =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp_size));
          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j * (degree_fine + 1);
              const int stride_kernel = 1;

              const int base_fine   = i;
              const int stride_fine = degree_fine + 1;

              number sum = prolongation_matrix_scratch(base_kernel) *
                           values_fine(base_fine);

              for (int k = 1; k < Nk; ++k)
                sum +=
                  prolongation_matrix_scratch(base_kernel + k * stride_kernel) *
                  values_fine(base_fine + k * stride_fine);

              const int index_tmp = i + j * Ni;

              tmp(index_tmp) = sum;
            });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j * (degree_fine + 1);
              const int stride_kernel = 1;

              const int base_tmp   = i * Nk;
              const int stride_tmp = 1;

              number sum =
                prolongation_matrix_scratch(base_kernel) * tmp(base_tmp);

              for (int k = 1; k < Nk; ++k)
                sum +=
                  prolongation_matrix_scratch(base_kernel + k * stride_kernel) *
                  tmp(base_tmp + k * stride_tmp);

              const int index_coarse = i * Nj + j;

              values_coarse(index_coarse) = sum;
            });
          }

          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          constexpr int tmp1_size =
            Utilities::pow(degree_fine + 1, 2) * (degree_coarse + 1);
          constexpr int tmp2_size =
            Utilities::pow(degree_coarse + 1, 2) * (degree_fine + 1);

          auto tmp1 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp1_size));
          auto tmp2 = Kokkos::subview(scratch_pad,
                                      Kokkos::make_pair(tmp1_size,
                                                        tmp1_size + tmp2_size));
          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nm = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (degree_fine + 1);
                const int stride_kernel = 1;

                const int base_fine   = i * Nj + j;
                const int stride_fine = Utilities::pow(degree_fine + 1, 2);

                number sum = prolongation_matrix_scratch(base_kernel) *
                             values_fine(base_fine);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix_scratch(base_kernel +
                                                     k * stride_kernel) *
                         values_fine(base_fine + k * stride_fine);

                const int index_tmp1 = (i + m * Ni) * Nj + j;
                tmp1(index_tmp1)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nm = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (degree_fine + 1);
                const int stride_kernel = 1;

                const int base_tmp1   = i + j * Ni * Nk;
                const int stride_tmp1 = degree_fine + 1;
                number    sum =
                  prolongation_matrix_scratch(base_kernel) * tmp1(base_tmp1);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix_scratch(base_kernel +
                                                     k * stride_kernel) *
                         tmp1(base_tmp1 + k * stride_tmp1);

                const int index_tmp2 = i + (j * Nm + m) * Ni;
                tmp2(index_tmp2)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nm = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);

            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m * (degree_fine + 1);
                const int stride_kernel = 1;

                const int base_tmp2   = (i * Nj + j) * Nk;
                const int stride_tmp2 = 1;

                number sum =
                  prolongation_matrix_scratch(base_kernel) * tmp2(base_tmp2);

                for (int k = 1; k < Nk; ++k)
                  sum += prolongation_matrix_scratch(base_kernel +
                                                     k * stride_kernel) *
                         tmp2(base_tmp2 + k * stride_tmp2);

                const int index_coarse      = (i * Nj + j) * Nm + m;
                values_coarse(index_coarse) = sum;
              });
          }

          team_member.team_barrier();
        }

      // distribute coarse dofs values
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_coarse),
        [&](const int &i) {
          const unsigned int dof_index = dof_indices_coarse(i, cell_index);
          if (dof_index != numbers::invalid_unsigned_int)
            Kokkos::atomic_add(&dst[dof_index], values_coarse(i));
        });
      team_member.team_barrier();
    }
  } // namespace h_mg_transfer

  template <int dim, int fe_degree, typename number>
  class GeometricTransfer : public MGTransferBase<dim, number>
  {
  public:
    GeometricTransfer();

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
    setup_weights();

    void
    setup_dof_indices();


    std::vector<h_mg_transfer::MGTransferScheme<dim, fe_degree, number>>
      transfer_schemes;

    ObserverPointer<const MatrixFree<dim, number>> matrix_free_coarse;
    ObserverPointer<const MatrixFree<dim, number>> matrix_free_fine;


    ObserverPointer<const AffineConstraints<number>> constraints_fine;
    ObserverPointer<const AffineConstraints<number>> constraints_coarse;

    ObserverPointer<const DoFHandler<dim>> dof_handler_fine;
    ObserverPointer<const DoFHandler<dim>> dof_handler_coarse;

    const unsigned int mg_level_coarse = numbers::invalid_unsigned_int;
    const unsigned int mg_level_fine   = numbers::invalid_unsigned_int;

    /**
     * Partitioner needed by the intermediate vector.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

    /**
     * Partitioner needed by the intermediate vector.
     */
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;


    dealii::internal::MatrixFreeFunctions::
      ConstraintInfo<dim, VectorizedArray<number, 1>, types::global_dof_index>
        constraint_info_fine;

    dealii::internal::MatrixFreeFunctions::
      ConstraintInfo<dim, VectorizedArray<number, 1>, types::global_dof_index>
        constraint_info_coarse;
  };


  template <int dim, int fe_degree, typename number>
  GeometricTransfer<dim, fe_degree, number>::GeometricTransfer()
  {}


  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::prolongate_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;

    Assert(src.get_partitioner() ==
               matrix_free_coarse->get_vector_partitioner() ||
             src.get_partitioner() == this->partitioner_coarse,
           ExcMessage("Coarse vector is not initialized correctly."));

    Assert(dst.get_partitioner() ==
               matrix_free_fine->get_vector_partitioner() ||
             dst.get_partitioner() == this->partitioner_fine,
           ExcMessage("Fine vector is not initialized correctly."));

    /** FIXME:: "smarter" copy */
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> vec_coarse;
    vec_coarse.reinit(this->partitioner_coarse);
    vec_coarse = src;
    vec_coarse.update_ghost_values();

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> vec_fine;
    vec_fine.reinit(this->partitioner_fine);

    this->prolongate_and_add_internal(vec_fine, vec_coarse);

    vec_fine.compress(VectorOperation::add);

    /** FIXME: "smarter" copy */
    dst += vec_fine;
    dst.compress(VectorOperation::insert);

    src.zero_out_ghost_values();

    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage(
             "Fine vector is not handled correctly after during prolongation"));
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::restrict_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;

    Assert(dst.get_partitioner() ==
               matrix_free_coarse->get_vector_partitioner() ||
             dst.get_partitioner() == this->partitioner_coarse,
           ExcMessage("Fine vector is not initialized correctly."));
    Assert(src.get_partitioner() ==
               matrix_free_fine->get_vector_partitioner() ||
             src.get_partitioner() == this->partitioner_fine,
           ExcMessage("Coarse vector is not initialized correctly."));

    src.update_ghost_values();

    /** FIXME:: "smarter" copy */
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> vec_fine;
    vec_fine.reinit(this->partitioner_fine);
    vec_fine = src;
    vec_fine.update_ghost_values();

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> vec_coarse;
    vec_coarse.reinit(this->partitioner_coarse);

    this->restrict_and_add_internal(vec_coarse, vec_fine);

    vec_coarse.compress(VectorOperation::add);

    /** FIXME: "smarter" copy */
    dst += vec_coarse;

    dst.compress(VectorOperation::insert);

    src.zero_out_ghost_values();


    Assert(
      dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Fine vector is not handled correctly during restriction."));
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    using TeamPolicy =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

    using Functor =
      h_mg_transfer::CellProlongationKernel<dim, fe_degree, number>;

    MemorySpace::Default::kokkos_space::execution_space exec;

    unsigned int scheme_index = 0;
    for (auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;

        h_mg_transfer::CellProlongationKernel<dim, fe_degree, number>
          prolongator;

        auto team_policy =
          TeamPolicy(exec, scheme.n_coarse_cells, Kokkos::AUTO);

        h_mg_transfer::ApplyCellKernel<dim, fe_degree, number, Functor>
          apply_prolongation(prolongator,
                             scheme.prolongation_matrix_shared_memory,
                             scheme.weights,
                             scheme.dof_indices_coarse,
                             scheme.dof_indices_fine,
                             src,
                             dst);

        Kokkos::parallel_for("prolongate_and_add_h_transfer_scheme_" +
                               std::to_string(scheme_index),
                             team_policy,
                             apply_prolongation);
        ++scheme_index;
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    using TeamPolicy =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;
    using Functor =
      h_mg_transfer::CellRestrictionKernel<dim, fe_degree, number>;

    MemorySpace::Default::kokkos_space::execution_space exec;

    unsigned int scheme_index = 0;
    for (auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;

        h_mg_transfer::CellRestrictionKernel<dim, fe_degree, number> restrictor;

        auto team_policy =
          TeamPolicy(exec, scheme.n_coarse_cells, Kokkos::AUTO);

        h_mg_transfer::ApplyCellKernel<dim, fe_degree, number, Functor>
          apply_restriction(restrictor,
                            scheme.prolongation_matrix_shared_memory,
                            scheme.weights,
                            scheme.dof_indices_coarse,
                            scheme.dof_indices_fine,
                            src,
                            dst);

        Kokkos::parallel_for("restrict_and_add_h_transfer_scheme_" +
                               std::to_string(scheme_index),
                             team_policy,
                             apply_restriction);
        ++scheme_index;
      }
  }


  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::reinit(
    const MatrixFree<dim, number>   &mf_coarse,
    const MatrixFree<dim, number>   &mf_fine,
    const AffineConstraints<number> &constraints_coarse,
    const AffineConstraints<number> &constraints_fine)
  {
    Assert((this->mg_level_fine == numbers::invalid_unsigned_int &&
            this->mg_level_coarse == numbers::invalid_unsigned_int) ||
             (this->mg_level_coarse + 1 == this->mg_level_fine),
           ExcNotImplemented());

    this->matrix_free_coarse = &mf_coarse;
    this->matrix_free_fine   = &mf_fine;

    this->constraints_coarse = &constraints_coarse;
    this->constraints_fine   = &constraints_fine;

    this->dof_handler_coarse = &matrix_free_coarse->get_dof_handler();
    this->dof_handler_fine   = &matrix_free_fine->get_dof_handler();

    Assert(
      this->dof_handler_coarse->get_mpi_communicator() ==
        this->dof_handler_fine->get_mpi_communicator(),
      ExcMessage(
        "Coarse and fine DoFHandler must have the same MPI communicator."));

    std::unique_ptr<dealii::internal::FineDoFHandlerViewBase<dim>>
      dof_handler_fine_view = std::make_unique<
        dealii::internal::GlobalCoarseningFineDoFHandlerView<dim>>(
        *this->dof_handler_fine,
        *this->dof_handler_coarse,
        this->mg_level_fine,
        this->mg_level_coarse);

    const auto reference_cell =
      this->dof_handler_fine->get_fe().reference_cell();

    // set up mg-transfer_schemes
    //   (0) no refinement -> identity
    //   (1) h-refinement
    //   (2) - other
    transfer_schemes.resize(
      1 + reference_cell.n_isotropic_refinement_choices()); // size=2

    const auto &fe_fine   = this->dof_handler_fine->get_fe();
    const auto &fe_coarse = this->dof_handler_coarse->get_fe();

    // helper function: to process the fine level cells; function
    // fu_non_refined is performed on cells that are not refined and
    // fu_refined is performed on children of cells that are refined
    const auto process_cells = [&](const auto &fu_non_refined,
                                   const auto &fu_refined) {
      dealii::internal::loop_over_active_or_level_cells(
        *this->dof_handler_coarse,
        this->mg_level_coarse,
        [&](const auto &cell_coarse) {
          if (this->mg_level_coarse == numbers::invalid_unsigned_int)
            {
              // get a reference to the equivalent cell on the fine
              // triangulation
              const auto cell_coarse_on_fine_mesh =
                dof_handler_fine_view->get_cell_view(cell_coarse);

              // check if cell has children
              if (cell_coarse_on_fine_mesh.has_children())
                // ... cell has children -> process children
                for (unsigned int c = 0;
                     c < GeometryInfo<dim>::max_children_per_cell;
                     c++)
                  fu_refined(cell_coarse,
                             dof_handler_fine_view->get_cell_view(cell_coarse,
                                                                  c),
                             c);
              else // ... cell has no children -> process cell
                fu_non_refined(cell_coarse, cell_coarse_on_fine_mesh);
            }
          else
            {
              // check if cell has children
              if (cell_coarse->has_children())
                // ... cell has children -> process children
                for (unsigned int c = 0;
                     c < GeometryInfo<dim>::max_children_per_cell;
                     c++)
                  fu_refined(cell_coarse,
                             dof_handler_fine_view->get_cell_view(cell_coarse,
                                                                  c),
                             c);
            }
        });
    };

    // check if FE is the same
    AssertDimension(fe_coarse.n_dofs_per_cell(), fe_fine.n_dofs_per_cell());

    for (auto &scheme : transfer_schemes)
      {
        /**  FIXME: MGTransferScheme for the moment assumes only global
         * refinement, not local. */
        // number of dofs on coarse and fine cells
        // scheme.n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();
        // scheme.n_dofs_per_cell_fine =
        //   Utilities::pow(2 * fe_fine.degree + 1, dim);

        // degree of FE on coarse and fine cell
        // scheme.degree_coarse = fe_coarse.degree;
        // scheme.degree_fine   = fe_coarse.degree * 2;

        Assert(scheme.degree_coarse == fe_coarse.degree,
               ExcMessage("Scheme coarse degree is not set correctly."));
        Assert(scheme.degree_fine == fe_coarse.degree * 2,
               ExcMessage("Scheme fine degree is not set correctly."));

        Assert(scheme.n_dofs_per_cell_coarse == fe_coarse.n_dofs_per_cell(),
               ExcMessage(
                 "Scheme n_dofs_per_cell_coarse is not set correctly."));

        Assert(scheme.n_dofs_per_cell_fine ==
                 Utilities::pow(2 * fe_fine.degree + 1, dim),
               ExcMessage("Scheme n_dofs_per_cell_fine is not set correctly."));

        // reset number of coarse cells
        scheme.n_coarse_cells = 0;
      }

    /**  FIXME: MGTransferScheme for the moment assumes only global refinement,
     * not local. */
    // correct for first scheme
    // transfer_schemes[0].n_dofs_per_cell_fine = fe_coarse.n_dofs_per_cell();
    // transfer_schemes[0].degree_fine          = fe_coarse.degree;

    std::uint8_t current_refinement_case = static_cast<std::uint8_t>(-1);

    // count coarse cells for each scheme (0, 1, ...)
    {
      // count by looping over all coarse cells
      process_cells([&](const auto &,
                        const auto &) { transfer_schemes[0].n_coarse_cells++; },
                    [&](const auto &, const auto &cell_fine, const auto c) {
                      std::uint8_t refinement_case =
                        cell_fine.refinement_case();

                      // Assert triggers if cell has no children
                      Assert(RefinementCase<dim>(refinement_case) ==
                               RefinementCase<dim>::isotropic_refinement,
                             ExcNotImplemented());

                      refinement_case = 1;

                      if (c == 0)
                        {
                          transfer_schemes[refinement_case].n_coarse_cells++;

                          current_refinement_case = refinement_case;
                        }
                      else
                        // Check that all children have the same refinement case
                        AssertThrow(current_refinement_case == refinement_case,
                                    ExcNotImplemented());
                    });
    }

    const auto cell_local_children_indices =
      dealii::internal::get_child_offsets<dim>(
        transfer_schemes[0].n_dofs_per_cell_coarse,
        fe_fine.degree,
        fe_fine.degree);

    std::vector<unsigned int> n_dof_indices_fine(transfer_schemes.size() + 1);
    std::vector<unsigned int> n_dof_indices_coarse(transfer_schemes.size() + 1);

    for (unsigned int i = 0; i < transfer_schemes.size(); ++i)
      {
        n_dof_indices_fine[i + 1] = transfer_schemes[i].n_dofs_per_cell_fine *
                                    transfer_schemes[i].n_coarse_cells;
        n_dof_indices_coarse[i + 1] =
          transfer_schemes[i].n_dofs_per_cell_coarse *
          transfer_schemes[i].n_coarse_cells;
      }

    for (unsigned int i = 0; i < transfer_schemes.size(); ++i)
      {
        n_dof_indices_fine[i + 1] += n_dof_indices_fine[i];
        n_dof_indices_coarse[i + 1] += n_dof_indices_coarse[i];
      }

    // indices

    {
      std::vector<types::global_dof_index> local_dof_indices(
        transfer_schemes[0].n_dofs_per_cell_coarse);

      // ---------------------- lexicographic_numbering ----------------------
      std::vector<unsigned int> lexicographic_numbering_fine;
      std::vector<unsigned int> lexicographic_numbering_coarse;
      {
        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));

        dealii::internal::MatrixFreeFunctions::ShapeInfo<number> shape_info;

        shape_info.reinit(dummy_quadrature, fe_fine, 0);
        lexicographic_numbering_fine = shape_info.lexicographic_numbering;

        shape_info.reinit(dummy_quadrature, fe_coarse, 0);
        lexicographic_numbering_coarse = shape_info.lexicographic_numbering;
      }

      // ------------------------------ indices ------------------------------
      std::vector<types::global_dof_index> level_dof_indices_coarse(
        transfer_schemes[0].n_dofs_per_cell_fine);

      std::vector<types::global_dof_index> level_dof_indices_fine(
        transfer_schemes[1].n_dofs_per_cell_fine);

      unsigned int n_coarse_cells_total = 0;

      for (const auto &scheme : transfer_schemes)
        n_coarse_cells_total += scheme.n_coarse_cells;

      this->constraint_info_coarse.reinit(*this->dof_handler_coarse,
                                          n_coarse_cells_total,
                                          constraints_coarse.n_constraints() >
                                            0);

      this->constraint_info_coarse.set_locally_owned_indices(
        this->dof_handler_coarse->locally_owned_dofs());

      this->constraint_info_fine.reinit(n_coarse_cells_total);

      this->constraint_info_fine.set_locally_owned_indices(
        this->dof_handler_fine->locally_owned_dofs());


      std::vector<unsigned int> cell_no(transfer_schemes.size(), 0);
      for (unsigned int i = 1; i < transfer_schemes.size(); ++i)
        cell_no[i] = cell_no[i - 1] + transfer_schemes[i - 1].n_coarse_cells;

      process_cells(
        [&](const auto &cell_coarse, const auto &cell_fine) {
          // first process cells with scheme 0
          // parent
          {
            this->constraint_info_coarse.read_dof_indices(cell_no[0],
                                                          this->mg_level_coarse,
                                                          cell_coarse,
                                                          constraints_coarse,
                                                          {});
          }

          // child
          {
            cell_fine.get_dof_indices(local_dof_indices);
            for (unsigned int i = 0;
                 i < transfer_schemes[0].n_dofs_per_cell_coarse;
                 i++)
              level_dof_indices_coarse[i] =
                local_dof_indices[lexicographic_numbering_fine[i]];

            this->constraint_info_fine.read_dof_indices(
              cell_no[0], level_dof_indices_coarse, {});
          }

          // move pointers
          {
            ++cell_no[0];
          }
        },
        [&](const auto &cell_coarse, const auto &cell_fine, const auto c) {
          // process rest of cells
          const std::uint8_t refinement_case = 1;
          // parent (only once at the beginning)
          if (c == 0)
            {
              this->constraint_info_coarse.read_dof_indices(
                cell_no[refinement_case],
                this->mg_level_coarse,
                cell_coarse,
                constraints_coarse,
                {});

              level_dof_indices_fine.assign(level_dof_indices_fine.size(),
                                            numbers::invalid_dof_index);
            }

          // child
          {
            cell_fine.get_dof_indices(local_dof_indices);
            for (unsigned int i = 0;
                 i < transfer_schemes[refinement_case].n_dofs_per_cell_coarse;
                 ++i)
              {
                const auto index =
                  local_dof_indices[lexicographic_numbering_fine[i]];
                Assert(
                  level_dof_indices_fine[cell_local_children_indices[c][i]] ==
                      numbers::invalid_dof_index ||
                    level_dof_indices_fine[cell_local_children_indices[c][i]] ==
                      index,
                  ExcInternalError());

                level_dof_indices_fine[cell_local_children_indices[c][i]] =
                  index;
              }
          }

          // move pointers (only once at the end)
          if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
            {
              this->constraint_info_fine.read_dof_indices(
                cell_no[refinement_case], level_dof_indices_fine, {});

              ++cell_no[refinement_case];
            }
        });
    }

    {
      this->partitioner_coarse = this->constraint_info_coarse.finalize(
        this->dof_handler_coarse->get_mpi_communicator());

      this->partitioner_fine = this->constraint_info_fine.finalize(
        this->dof_handler_fine->get_mpi_communicator());

      // if constexpr (running_in_debug_mode())
      //   {
      //     // We would like to assert that no strange indices were added in
      //     // the transfer. Unfortunately, we can only do this if we're
      //     // working with the multigrid indices within the DoFHandler, not
      //     // when the transfer comes from different DoFHandler object, as
      //     // the latter might have unrelated parallel partitions.
      //     if (mg_level_fine != numbers::invalid_unsigned_int)
      //       {
      //         Utilities::MPI::Partitioner part_check(
      //           dof_handler_fine.locally_owned_mg_dofs(mg_level_fine),
      //           DoFTools::extract_locally_relevant_level_dofs(dof_handler_fine,
      //                                                         mg_level_fine),
      //           dof_handler_fine.get_mpi_communicator());
      //         Assert(partitioner_fine->ghost_indices().is_subset_of(
      //                  part_check.ghost_indices()),
      //                ExcMessage(
      //                  "The setup of ghost indices failed, because the set
      //                  " "of ghost indices identified for the transfer is "
      //                  "not a subset of the locally relevant dofs on level
      //                  "
      //                  + std::to_string(mg_level_fine) + " with " +
      //                  std::to_string(dof_handler_fine.n_dofs(mg_level_fine))
      //                  + " dofs in total, which means we do not understand
      //                  " "the indices that were collected. This is very "
      //                  "likely a bug in deal.II, and could e.g. be caused "
      //                  "by some integer type narrowing between 64 bit and "
      //                  "32 bit integers."));
      //       }
      //   }
    }

    // ------------- prolongation matrix (0) -> identity matrix --------------

    // nothing to do since for identity prolongation matrices a short-cut
    // code path is used during prolongation/restriction

    // -------------------prolongation matrix (i = 1 ... n)-------------------
    {
      AssertDimension(fe_fine.n_base_elements(), 1);



      for (unsigned int transfer_scheme_index = 1;
           transfer_scheme_index < transfer_schemes.size();
           ++transfer_scheme_index)
        {
          // const auto fe = create_1D_fe(fe_fine.base_element(0));
          const auto fe = FE_Q<1>(fe_fine.degree);

          std::vector<unsigned int> renumbering(fe.n_dofs_per_cell());
          {
            AssertIndexRange(fe.n_dofs_per_vertex(), 2);
            renumbering[0] = 0;
            for (unsigned int i = 0; i < fe.dofs_per_line; ++i)
              renumbering[i + fe.n_dofs_per_vertex()] =
                GeometryInfo<1>::vertices_per_cell * fe.n_dofs_per_vertex() + i;
            if (fe.n_dofs_per_vertex() > 0)
              renumbering[fe.n_dofs_per_cell() - fe.n_dofs_per_vertex()] =
                fe.n_dofs_per_vertex();
          }

          const unsigned int shift =
            fe.n_dofs_per_cell() - fe.n_dofs_per_vertex();
          const unsigned int n_child_dofs_1d =
            fe.n_dofs_per_cell() * 2 - fe.n_dofs_per_vertex();

          {
            // transfer_schemes[scheme.prolongation_matrix]
            //   .prolongation_matrix.resize(fe.n_dofs_per_cell() *
            //                               n_child_dofs_1d);

            transfer_schemes[transfer_scheme_index]
              .prolongation_matrix_shared_memory =
              Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("prolongation_matrix_h_transfer_scheme_" +
                                     std::to_string(transfer_scheme_index),
                                   Kokkos::WithoutInitializing),
                fe.n_dofs_per_cell() * n_child_dofs_1d);

            auto prolongation_matrix_host =
              Kokkos::create_mirror_view(transfer_schemes[transfer_scheme_index]
                                           .prolongation_matrix_shared_memory);

            for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell;
                 ++c)
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                  prolongation_matrix_host[i * n_child_dofs_1d + j +
                                           c * shift] =
                    fe.get_prolongation_matrix(c)(renumbering[j],
                                                  renumbering[i]);

            Kokkos::deep_copy(transfer_schemes[transfer_scheme_index]
                                .prolongation_matrix_shared_memory,
                              prolongation_matrix_host);
            Kokkos::fence();
          }
        }
    }

    setup_dof_indices();

    setup_weights();
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::setup_weights()
  {
    LinearAlgebra::distributed::Vector<number> weight_vector;
    weight_vector.reinit(this->partitioner_fine);

    for (const auto i : constraint_info_fine.dof_indices)
      weight_vector.local_element(i) += 1.0;

    weight_vector.compress(VectorOperation::add);

    for (unsigned int i = 0; i < weight_vector.locally_owned_size(); ++i)
      if (weight_vector.local_element(i) > 0)
        weight_vector.local_element(i) = 1.0 / weight_vector.local_element(i);

    // ... clear constrained indices
    for (const auto &constrained_dofs : this->constraints_fine->get_lines())
      if (weight_vector.locally_owned_elements().is_element(
            constrained_dofs.index))
        weight_vector[constrained_dofs.index] = 0.0;

    weight_vector.update_ghost_values();

    unsigned int cell_counter = 0;
    unsigned int scheme_index = 0;
    for (auto &scheme : transfer_schemes)
      {
        scheme.weights =
          Kokkos::View<number **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("weights_h_transer_scheme_" +
                                 std::to_string(scheme_index),
                               Kokkos::WithoutInitializing),
            scheme.n_dofs_per_cell_fine,
            scheme.n_coarse_cells);

        auto weights_view_host = Kokkos::create_mirror_view(scheme.weights);

        const unsigned int first_cell = cell_counter;
        for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
          {
            const unsigned int  cell_index = first_cell + cell;
            const unsigned int *dof_indices_fine =
              this->constraint_info_fine.dof_indices.data() +
              this->constraint_info_fine.row_starts[cell_index].first;

            for (unsigned int i = 0; i < scheme.n_dofs_per_cell_fine;
                 ++dof_indices_fine, ++i)
              {
                weights_view_host(i, cell) =
                  weight_vector.local_element(*dof_indices_fine);
              }
            ++cell_counter;
          }

        Kokkos::deep_copy(scheme.weights, weights_view_host);
        Kokkos::fence();
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::setup_dof_indices()
  {
    unsigned int scheme_counter = 0;
    unsigned int cell_counter   = 0;
    for (auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;

        scheme.dof_indices_coarse =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("h_transfer_dof_indices_coarse_scheme_" +
                                 std::to_string(scheme_counter),
                               Kokkos::WithoutInitializing),
            scheme.n_dofs_per_cell_coarse,
            scheme.n_coarse_cells);

        scheme.dof_indices_fine =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("h_transfer_dof_indices_fine_scheme_" +
                                 std::to_string(scheme_counter),
                               Kokkos::WithoutInitializing),
            scheme.n_dofs_per_cell_fine,
            scheme.n_coarse_cells);

        auto dof_indices_coarse_host =
          Kokkos::create_mirror_view(scheme.dof_indices_coarse);

        auto dof_indices_fine_host =
          Kokkos::create_mirror_view(scheme.dof_indices_fine);

        const unsigned int firts_cell = cell_counter;
        for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
          {
            const unsigned int cell_index = firts_cell + cell;

            // fill coarse indices
            {
              const unsigned int *dof_indices_coarse =
                this->constraint_info_coarse.dof_indices.data() +
                this->constraint_info_coarse.row_starts[cell_index].first;
              unsigned int index_indicators =
                this->constraint_info_coarse.row_starts[cell_index].second;
              unsigned int next_index_indicators =
                this->constraint_info_coarse.row_starts[cell_index + 1].second;

              unsigned int ind_local = 0;
              for (; index_indicators != next_index_indicators;
                   ++index_indicators)
                {
                  const std::pair<unsigned short, unsigned short> indicator =
                    this->constraint_info_coarse
                      .constraint_indicator[index_indicators];

                  for (unsigned int j = 0; j < indicator.first; ++j)
                    {
                      dof_indices_coarse_host(ind_local + j, cell) =
                        dof_indices_coarse[j];
                    }
                  ind_local += indicator.first;
                  dof_indices_coarse += indicator.first;

                  dof_indices_coarse_host(ind_local, cell) =
                    numbers::invalid_unsigned_int;

                  ++ind_local;
                }

              for (; ind_local < scheme.n_dofs_per_cell_coarse;
                   ++dof_indices_coarse, ++ind_local)
                dof_indices_coarse_host(ind_local, cell) = *dof_indices_coarse;
            }

            // fill fine indices
            {
              const unsigned int *dof_indices_fine =
                this->constraint_info_fine.dof_indices.data() +
                this->constraint_info_fine.row_starts[cell_index].first;

              for (unsigned int j = 0; j < scheme.n_dofs_per_cell_fine;
                   ++dof_indices_fine, ++j)
                dof_indices_fine_host(j, cell) = *dof_indices_fine;
            }

            ++cell_counter;
          }

        Kokkos::deep_copy(scheme.dof_indices_coarse, dof_indices_coarse_host);
        Kokkos::fence();

        Kokkos::deep_copy(scheme.dof_indices_fine, dof_indices_fine_host);
        Kokkos::fence();

        ++scheme_counter;
      }
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif

