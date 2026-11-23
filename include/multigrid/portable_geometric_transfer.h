#ifndef portable_geometric_transfer_h
#define portable_geometric_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include "kernels/bk1_kokkos_kernels.h"
#include "multigrid/portable_geometric_transfer_core.h"
#include "multigrid_dealii/mg_transfer_private_access.h"


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
      static const unsigned int n_dofs_per_cell_coarse = Utilities::pow(fe_degree + 1, dim);

      /**
       * Number of degrees of freedom of fine cell.
       *
       * @note For tensor-product elements, the value equals
       *   `n_components * (n_dofs_per_cell_fine + 1)^dim`.
       */
      static const unsigned int n_dofs_per_cell_fine = Utilities::pow(2 * fe_degree + 1, dim);

      /**
       * Prolongation matrix used for the prolongate_and_add() and
       * restrict_and_add() functions.
       */

      Kokkos::View<number *, MemorySpace::Default::kokkos_space> prolongation_matrix_shared_memory;

      Kokkos::View<number **, MemorySpace::Default::kokkos_space> weights;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_coarse;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_fine;
    };

    template <int dim, typename number>
    struct TransferCellData
    {
      using TeamHandle =
        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      TeamHandle team_member;

      const int cell_index;

      const SharedView &prolongation_matrix;

      const Kokkos::View<number **, MemorySpace::Default::kokkos_space> &weights;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> &dof_indices_coarse;


      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> &dof_indices_fine;

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
      using TeamHandle =
        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;
      using SharedViewValues =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      ApplyCellKernel(
        Functor func,
        const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
          prolongation_matrix_shared_memory,
        const Kokkos::View<number **, MemorySpace::Default::kokkos_space>       weights,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_coarse,
        const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_fine,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst)
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

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_coarse;

      const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_fine;

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
          + (Functor::degree_coarse + 1) * (Functor::degree_fine + 1) // prolongation matrix
        );
      }


      DEAL_II_HOST_DEVICE
      void
      operator()(const TeamHandle &team_member) const
      {
        const int cell_index = team_member.league_rank();

        SharedViewValues values_coarse(team_member.team_shmem(), Functor::n_dofs_per_cell_coarse);

        SharedViewValues values_fine(team_member.team_shmem(), Functor::n_dofs_per_cell_fine);

        SharedViewValues prolongation_matrix_device(team_member.team_shmem(),
                                                    (Functor::degree_coarse + 1) *
                                                      (Functor::degree_fine + 1));

        SharedViewValues scratch_pad(team_member.team_shmem(), Functor::n_dofs_per_cell_fine * 2);

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member,
                                  (Functor::degree_coarse + 1) * (Functor::degree_fine + 1)),
          [&](const int &i)
            { prolongation_matrix_device(i) = prolongation_matrix_shared_memory(i); });
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
      using TeamHandle =
        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellProlongationKernel();


      DEAL_II_HOST_DEVICE void
      operator()(const TransferCellData<dim, number> *cell_data,
                 const DeviceVector<number>          &src,
                 DeviceVector<number>                &dst) const;

      static const unsigned int degree_coarse =
        MGTransferScheme<dim, fe_degree, number>::degree_coarse;
      static const unsigned int degree_fine = MGTransferScheme<dim, fe_degree, number>::degree_fine;

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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_coarse),
                           [&](const int &i)
                             {
                               const unsigned int dof_index = dof_indices_coarse(i, cell_index);
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
          auto          tmp       = Kokkos::subview(scratch_pad, Kokkos::make_pair(0, temp_size));

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j)
                                   {
                                     const int base_kernel   = j;
                                     const int stride_kernel = degree_fine + 1;

                                     const int base_coarse   = i * Nk;
                                     const int stride_coarse = 1;

                                     number sum = prolongation_matrix_scratch(base_kernel) *
                                                  values_coarse(base_coarse);

                                     for (int k = 1; k < Nk; ++k)
                                       sum += prolongation_matrix_scratch(base_kernel +
                                                                          k * stride_kernel) *
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
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j)
                                   {
                                     const int base_kernel   = j;
                                     const int stride_kernel = degree_fine + 1;

                                     const int base_tmp   = i;
                                     const int stride_tmp = degree_fine + 1;

                                     number sum =
                                       prolongation_matrix_scratch(base_kernel) * tmp(base_tmp);

                                     for (int k = 1; k < Nk; ++k)
                                       sum += prolongation_matrix_scratch(base_kernel +
                                                                          k * stride_kernel) *
                                              tmp(base_tmp + k * stride_tmp);

                                     const int index_fine    = i + j * Ni;
                                     values_fine(index_fine) = sum;
                                   });
          }

          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          constexpr int tmp1_size = Utilities::pow(degree_coarse + 1, 2) * (degree_fine + 1);
          constexpr int tmp2_size = Utilities::pow(degree_fine + 1, 2) * (degree_coarse + 1);
          auto          tmp1      = Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp1_size));
          auto          tmp2 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(tmp1_size, tmp1_size + tmp2_size));

          {
            constexpr int Ni = degree_coarse + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nm = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
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
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
                                     const int base_kernel   = m;
                                     const int stride_kernel = degree_fine + 1;

                                     const int base_tmp1   = i + j * Ni * Nk;
                                     const int stride_tmp1 = degree_fine + 1;

                                     number sum =
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
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nm = degree_fine + 1;
            constexpr int Nk = degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_fine),
                           [&](const int &i)
                             { values_fine(i) *= cell_data->weights(i, cell_index); });
      team_member.team_barrier();


      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_fine),
                           [&](const int &i)
                             {
                               const unsigned int dof_index = dof_indices_fine(i, cell_index);
                               Kokkos::atomic_add(&dst[dof_index], values_fine(i));
                             });
      team_member.team_barrier();
    }

    template <int dim, int fe_degree, typename number>
    class CellRestrictionKernel : public EnableObserverPointer
    {
    public:
      using TeamHandle =
        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView =
        Kokkos::View<number *,
                     MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellRestrictionKernel();



      DEAL_II_HOST_DEVICE void
      operator()(const TransferCellData<dim, number> *cell_data,
                 const DeviceVector<number>          &src,
                 DeviceVector<number>                &dst) const;

      static const unsigned int degree_coarse =
        MGTransferScheme<dim, fe_degree, number>::degree_coarse;
      static const unsigned int degree_fine = MGTransferScheme<dim, fe_degree, number>::degree_fine;

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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_fine),
                           [&](const int &i)
                             { values_fine(i) = src[dof_indices_fine(i, cell_index)]; });
      team_member.team_barrier();

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_fine),
                           [&](const int &i)
                             { values_fine(i) *= cell_data->weights(i, cell_index); });
      team_member.team_barrier();

      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          constexpr int tmp_size = (degree_coarse + 1) * (degree_fine + 1);

          auto tmp = Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp_size));
          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j)
                                   {
                                     const int base_kernel   = j * (degree_fine + 1);
                                     const int stride_kernel = 1;

                                     const int base_fine   = i;
                                     const int stride_fine = degree_fine + 1;

                                     number sum = prolongation_matrix_scratch(base_kernel) *
                                                  values_fine(base_fine);

                                     for (int k = 1; k < Nk; ++k)
                                       sum += prolongation_matrix_scratch(base_kernel +
                                                                          k * stride_kernel) *
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
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j)
                                   {
                                     const int base_kernel   = j * (degree_fine + 1);
                                     const int stride_kernel = 1;

                                     const int base_tmp   = i * Nk;
                                     const int stride_tmp = 1;

                                     number sum =
                                       prolongation_matrix_scratch(base_kernel) * tmp(base_tmp);

                                     for (int k = 1; k < Nk; ++k)
                                       sum += prolongation_matrix_scratch(base_kernel +
                                                                          k * stride_kernel) *
                                              tmp(base_tmp + k * stride_tmp);

                                     const int index_coarse = i * Nj + j;

                                     values_coarse(index_coarse) = sum;
                                   });
          }

          team_member.team_barrier();
        }
      else if constexpr (dim == 3)
        {
          constexpr int tmp1_size = Utilities::pow(degree_fine + 1, 2) * (degree_coarse + 1);
          constexpr int tmp2_size = Utilities::pow(degree_coarse + 1, 2) * (degree_fine + 1);

          auto tmp1 = Kokkos::subview(scratch_pad, Kokkos::make_pair(0, tmp1_size));
          auto tmp2 =
            Kokkos::subview(scratch_pad, Kokkos::make_pair(tmp1_size, tmp1_size + tmp2_size));
          {
            constexpr int Ni = degree_fine + 1;
            constexpr int Nj = degree_fine + 1;
            constexpr int Nm = degree_coarse + 1;
            constexpr int Nk = degree_fine + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
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
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
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
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(team_member, Ni, Nj, Nm);

            Kokkos::parallel_for(thread_policy,
                                 [&](const int i, const int j, const int m)
                                   {
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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs_per_cell_coarse),
                           [&](const int &i)
                             {
                               const unsigned int dof_index = dof_indices_coarse(i, cell_index);
                               if (dof_index != numbers::invalid_unsigned_int)
                                 Kokkos::atomic_add(&dst[dof_index], values_coarse(i));
                             });
      team_member.team_barrier();
    }
  } // namespace h_mg_transfer

  template <int dim, int fe_degree, typename number>
  class GeometricTransfer : public internal::GeometricTransferCore<dim, number>
  {
  public:
    GeometricTransfer();

    using MGTransferBase<dim, number>::reinit;

    void
    reinit(const OperatorInfoBase<dim>     &op_coarse,
           const OperatorInfoBase<dim>     &op_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) override;

  private:
    void
    prolongate_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    void
    restrict_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    std::vector<h_mg_transfer::MGTransferScheme<dim, fe_degree, number>> transfer_schemes;


    ObserverPointer<const AffineConstraints<number>> constraints_fine;
    ObserverPointer<const AffineConstraints<number>> constraints_coarse;

    ObserverPointer<const DoFHandler<dim>> dof_handler_fine;
    ObserverPointer<const DoFHandler<dim>> dof_handler_coarse;

    const unsigned int mg_level_coarse = numbers::invalid_unsigned_int;
    const unsigned int mg_level_fine   = numbers::invalid_unsigned_int;
  };


  template <int dim, int fe_degree, typename number>
  GeometricTransfer<dim, fe_degree, number>::GeometricTransfer()
  {}

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    using TeamPolicy = Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

    using Functor = h_mg_transfer::CellProlongationKernel<dim, fe_degree, number>;

    MemorySpace::Default::kokkos_space::execution_space exec;

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    unsigned int scheme_index = 0;
    for (auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;

        constexpr bool is_serial =
          std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

        unsigned int numBlocks       = numbers::invalid_unsigned_int;
        unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
        if (is_serial)
          {
            numBlocks       = 1u;
            threadsPerBlock = 1u;
          }

        BK1::Parallel::
          KokkosProlongationBatchedKernelAbstracted<dim, fe_degree + 1, 2 * fe_degree + 1, number>(
            scheme.prolongation_matrix_shared_memory,
            src_device,
            dst_device,
            scheme.dof_indices_coarse,
            scheme.dof_indices_fine,
            scheme.weights,
            scheme.n_coarse_cells,
            numBlocks,
            threadsPerBlock);

        ++scheme_index;
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    using TeamPolicy = Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;
    using Functor    = h_mg_transfer::CellRestrictionKernel<dim, fe_degree, number>;

    MemorySpace::Default::kokkos_space::execution_space exec;

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());


    unsigned int scheme_index = 0;
    for (auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;


        constexpr bool is_serial =
          std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

        unsigned int numBlocks       = numbers::invalid_unsigned_int;
        unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
        if (is_serial)
          {
            numBlocks       = 1u;
            threadsPerBlock = 1u;
          }

        BK1::Parallel::
          KokkosRestrictionBatchedKernelAbstracted<dim, fe_degree + 1, 2 * fe_degree + 1, number>(
            scheme.prolongation_matrix_shared_memory,
            src_device,
            dst_device,
            scheme.dof_indices_coarse,
            scheme.dof_indices_fine,
            scheme.weights,
            scheme.n_coarse_cells,
            numBlocks,
            threadsPerBlock);

        ++scheme_index;
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::reinit(
    const OperatorInfoBase<dim>     &op_coarse,
    const OperatorInfoBase<dim>     &op_fine,
    const AffineConstraints<number> &constraints_coarse,
    const AffineConstraints<number> &constraints_fine)
  {
    namespace host_access = Copy::Portable::internal;

    this->constraints_coarse = &constraints_coarse;
    this->constraints_fine   = &constraints_fine;

    this->dof_handler_coarse = &op_coarse.get_dof_handler();
    this->dof_handler_fine   = &op_fine.get_dof_handler();

    // deal.II's host transfer does the setup; its data is copied to the
    // device below and the host transfer is dropped at the end of this function
    host_access::CpuTransfer<dim, number> host_transfer;
    host_transfer.reinit_geometric_transfer(*this->dof_handler_fine,
                                            *this->dof_handler_coarse,
                                            constraints_fine,
                                            constraints_coarse,
                                            this->mg_level_fine,
                                            this->mg_level_coarse);

    const auto &host_schemes           = host_access::cpu_schemes(host_transfer);
    const auto &constraint_info_coarse = host_access::cpu_constraint_info_coarse(host_transfer);
    const auto &constraint_info_fine   = host_access::cpu_constraint_info_fine(host_transfer);
    const auto &host_weights           = host_access::cpu_weights(host_transfer);
    const auto &host_weights_start     = host_access::cpu_weights_start(host_transfer);
    const auto &host_weights_are_compressed =
      host_access::cpu_weights_are_compressed(host_transfer);

    AssertThrow(host_access::cpu_fine_element_is_continuous(host_transfer), ExcNotImplemented());
    AssertThrow(host_access::cpu_n_components(host_transfer) == 1, ExcNotImplemented());

    // schemes: (0) not refined, (1) refined. The kernels handle refined cells only.
    AssertThrow(host_schemes.size() == 2 && host_schemes[0].n_coarse_cells == 0,
                ExcNotImplemented());

    this->partitioner_coarse = host_access::cpu_partitioner_coarse(host_transfer);
    this->partitioner_fine   = host_access::cpu_partitioner_fine(host_transfer);

    this->vec_coarse.reinit(this->partitioner_coarse);
    this->vec_fine.reinit(this->partitioner_fine);

    transfer_schemes.resize(host_schemes.size());

    // the host transfer stores the weights per batch of n_lanes cells
    const unsigned int n_lanes      = VectorizedArray<number>::size();
    std::size_t        batch_offset = 0;
    unsigned int       cell_offset  = 0;

    for (unsigned int scheme_index = 0; scheme_index < host_schemes.size(); ++scheme_index)
      {
        auto       &scheme      = transfer_schemes[scheme_index];
        const auto &host_scheme = host_schemes[scheme_index];

        scheme.n_coarse_cells = host_scheme.n_coarse_cells;

        if (scheme_index > 0)
          {
            AssertThrow(host_scheme.degree_coarse == scheme.degree_coarse &&
                          host_scheme.degree_fine == scheme.degree_fine &&
                          host_scheme.n_dofs_per_cell_coarse == scheme.n_dofs_per_cell_coarse &&
                          host_scheme.n_dofs_per_cell_fine == scheme.n_dofs_per_cell_fine,
                        ExcInternalError());

            // prolongation matrix
            scheme.prolongation_matrix_shared_memory =
              Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("prolongation_matrix_h_transfer_scheme_" +
                                     std::to_string(scheme_index),
                                   Kokkos::WithoutInitializing),
                host_scheme.prolongation_matrix.size());

            auto prolongation_matrix_host =
              Kokkos::create_mirror_view(scheme.prolongation_matrix_shared_memory);

            for (unsigned int i = 0; i < host_scheme.prolongation_matrix.size(); ++i)
              prolongation_matrix_host[i] = host_scheme.prolongation_matrix[i];

            Kokkos::deep_copy(scheme.prolongation_matrix_shared_memory, prolongation_matrix_host);
            Kokkos::fence();
          }

        // weights
        {
          scheme.weights = Kokkos::View<number **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("weights_h_transer_scheme_" + std::to_string(scheme_index),
                               Kokkos::WithoutInitializing),
            scheme.n_dofs_per_cell_fine,
            scheme.n_coarse_cells);

          auto weights_view_host = Kokkos::create_mirror_view(scheme.weights);

          // a batch holds either all weights or 3^dim values by entity
          // (first / inner / last point per direction)
          const int  n_points_1d = scheme.degree_fine + 1;
          const auto entity      = [n_points_1d](const int i)
            {
              return i == 0 ? 0 : (i == n_points_1d - 1 ? 2 : 1);
            };

          for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
            {
              const std::size_t  batch         = batch_offset + cell / n_lanes;
              const unsigned int lane          = cell % n_lanes;
              const auto        *batch_weights = host_weights.data() + host_weights_start[batch];

              if (host_weights_are_compressed[batch])
                {
                  unsigned int index = 0;
                  for (int k = 0; k < (dim > 2 ? n_points_1d : 1); ++k)
                    for (int j = 0; j < (dim > 1 ? n_points_1d : 1); ++j)
                      for (int i = 0; i < n_points_1d; ++i, ++index)
                        weights_view_host(index, cell) =
                          batch_weights[9 * (dim > 2 ? entity(k) : 0) +
                                        3 * (dim > 1 ? entity(j) : 0) + entity(i)][lane];
                }
              else
                for (unsigned int i = 0; i < scheme.n_dofs_per_cell_fine; ++i)
                  weights_view_host(i, cell) = batch_weights[i][lane];
            }

          Kokkos::deep_copy(scheme.weights, weights_view_host);
          Kokkos::fence();
        }

        // DoF indices
        if (scheme.n_coarse_cells > 0)
          {
            scheme.dof_indices_coarse =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("h_transfer_dof_indices_coarse_scheme_" +
                                     std::to_string(scheme_index),
                                   Kokkos::WithoutInitializing),
                scheme.n_dofs_per_cell_coarse,
                scheme.n_coarse_cells);

            scheme.dof_indices_fine =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("h_transfer_dof_indices_fine_scheme_" +
                                     std::to_string(scheme_index),
                                   Kokkos::WithoutInitializing),
                scheme.n_dofs_per_cell_fine,
                scheme.n_coarse_cells);

            auto dof_indices_coarse_host = Kokkos::create_mirror_view(scheme.dof_indices_coarse);
            auto dof_indices_fine_host   = Kokkos::create_mirror_view(scheme.dof_indices_fine);

            for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
              {
                const unsigned int cell_index = cell_offset + cell;

                // coarse indices, constrained DoFs marked invalid
                {
                  const unsigned int *dof_indices_coarse =
                    constraint_info_coarse.dof_indices.data() +
                    constraint_info_coarse.row_starts[cell_index].first;
                  unsigned int index_indicators =
                    constraint_info_coarse.row_starts[cell_index].second;
                  const unsigned int next_index_indicators =
                    constraint_info_coarse.row_starts[cell_index + 1].second;

                  unsigned int ind_local = 0;
                  for (; index_indicators != next_index_indicators; ++index_indicators)
                    {
                      const std::pair<unsigned short, unsigned short> indicator =
                        constraint_info_coarse.constraint_indicator[index_indicators];

                      for (unsigned int j = 0; j < indicator.first; ++j)
                        dof_indices_coarse_host(ind_local + j, cell) = dof_indices_coarse[j];

                      ind_local += indicator.first;
                      dof_indices_coarse += indicator.first;

                      dof_indices_coarse_host(ind_local, cell) = numbers::invalid_unsigned_int;

                      ++ind_local;
                    }

                  for (; ind_local < scheme.n_dofs_per_cell_coarse;
                       ++dof_indices_coarse, ++ind_local)
                    dof_indices_coarse_host(ind_local, cell) = *dof_indices_coarse;
                }

                // fine indices
                {
                  const unsigned int *dof_indices_fine =
                    constraint_info_fine.dof_indices.data() +
                    constraint_info_fine.row_starts[cell_index].first;

                  for (unsigned int j = 0; j < scheme.n_dofs_per_cell_fine;
                       ++dof_indices_fine, ++j)
                    dof_indices_fine_host(j, cell) = *dof_indices_fine;
                }
              }

            Kokkos::deep_copy(scheme.dof_indices_coarse, dof_indices_coarse_host);
            Kokkos::fence();

            Kokkos::deep_copy(scheme.dof_indices_fine, dof_indices_fine_host);
            Kokkos::fence();
          }

        batch_offset += (scheme.n_coarse_cells + n_lanes - 1) / n_lanes;
        cell_offset += scheme.n_coarse_cells;
      }
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
