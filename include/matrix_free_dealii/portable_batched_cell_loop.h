// Copy of Portable::internal::batched::cell_loop() and its launch rules from the deal.II branch
// portable-batched-cell-loop (include/deal.II/matrix_free/
// portable_matrix_free.templates.h), in namespace Copy, so that it does not depend on
// the version of the installed deal.II. Keep in sync with the branch.

#ifndef matrix_free_dealii_portable_batched_cell_loop_h
#define matrix_free_dealii_portable_batched_cell_loop_h

#include <deal.II/base/config.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <string>

#include "matrix_free_dealii/portable_batched_evaluation_kernels.h"

namespace Copy
{
  namespace Portable
  {
    namespace internal
    {
      using namespace dealii;

      namespace batched
      {
        /**
         * Team scratch memory in bytes that the cells of one batch may use
         * for values, gradients and scratch pads when the batch size is chosen
         * automatically. The shape data of a team comes on top.
         */
        inline constexpr std::size_t cell_data_bytes_per_batch = 10800;



        /**
         * Number of units that run teams concurrently: streaming
         * multiprocessors on CUDA and HIP, hardware threads otherwise.
         */
        inline int
        n_compute_units(
          const MemorySpace::Default::kokkos_space::execution_space &exec)
        {
          using ExecutionSpace =
            MemorySpace::Default::kokkos_space::execution_space;
  #if defined(KOKKOS_ENABLE_CUDA)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            return exec.cuda_device_prop().multiProcessorCount;
  #endif
  #if defined(KOKKOS_ENABLE_HIP)
          if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>)
            return exec.hip_device_prop().multiProcessorCount;
  #endif
          return std::max(1, static_cast<int>(exec.concurrency()));
        }



        /**
         * Team scratch memory in bytes needed by SharedData::reinit() of all
         * dof handlers for batches of @p batch_size cells.
         */
        template <int dim, typename Number>
        std::size_t
        shmem_size(
          const unsigned int                       n_q_points_per_cell,
          const int                                batch_size,
          const unsigned int                       n_dof_handler,
          const Kokkos::Array<typename MatrixFree<dim, Number>::PrecomputedData,
                              n_max_dof_handlers> &precomputed_data)
        {
          std::size_t result = 0;
          for (unsigned int d = 0; d < n_dof_handler; ++d)
            result += SharedData<dim, Number>::shmem_size(n_q_points_per_cell,
                                                          batch_size,
                                                          precomputed_data[d]);
          return result;
        }



        /**
         * Number of cells per batch. If @p requested_batch_size is not given,
         * as many cells as fit into cell_data_bytes_per_batch are taken. In
         * both cases the batch size is reduced until the team scratch memory
         * fits the limit of the backend.
         */
        template <int dim, typename Number>
        int
        compute_batch_size(
          const unsigned int                       n_q_points_per_cell,
          const unsigned int                       n_dof_handler,
          const Kokkos::Array<typename MatrixFree<dim, Number>::PrecomputedData,
                              n_max_dof_handlers> &precomputed_data,
          const unsigned int requested_batch_size = numbers::invalid_unsigned_int)
        {
          std::size_t cell_data_bytes = 0;
          for (unsigned int d = 0; d < n_dof_handler; ++d)
            cell_data_bytes += (precomputed_data[d].n_components * (1 + dim) *
                                  n_q_points_per_cell +
                                SharedData<dim, Number>::scratch_pad_size(
                                  n_q_points_per_cell, precomputed_data[d])) *
                               sizeof(Number);

          int batch_size =
            (requested_batch_size == numbers::invalid_unsigned_int) ?
              static_cast<int>(cell_data_bytes_per_batch / cell_data_bytes) :
              static_cast<int>(requested_batch_size);
          batch_size = std::max(1, batch_size);

          const std::size_t scratch_size_max =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::
                                 execution_space>::scratch_size_max(0);
          while (batch_size > 1 &&
                 shmem_size<dim, Number>(n_q_points_per_cell,
                                         batch_size,
                                         n_dof_handler,
                                         precomputed_data) > scratch_size_max)
            --batch_size;

          return batch_size;
        }



        /**
         * Team policy for a batched loop over the cells of one color, in which
         * each team handles the batches league_rank(), league_rank() +
         * league_size(), ... The team size is one thread per line of
         * quadrature points of a batch unless @p requested_team_size is given,
         * limited by what the backend allows. There is one team per batch as
         * long as these are at most two teams per compute unit, and half as
         * many teams as batches otherwise.
         */
        template <int dim, typename Number, typename Functor>
        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
        make_team_policy(
          const MemorySpace::Default::kokkos_space::execution_space &exec,
          const Functor                                             &functor,
          const unsigned int                       n_q_points_per_cell,
          const int                                batch_size,
          const unsigned int                       n_dof_handler,
          const Kokkos::Array<typename MatrixFree<dim, Number>::PrecomputedData,
                              n_max_dof_handlers> &precomputed_data,
          const unsigned int requested_team_size = numbers::invalid_unsigned_int)
        {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;

          const std::size_t scratch_size = shmem_size<dim, Number>(
            n_q_points_per_cell, batch_size, n_dof_handler, precomputed_data);

          unsigned int n_q_points_1d = 1;
          while (Utilities::pow(n_q_points_1d, dim) < n_q_points_per_cell)
            ++n_q_points_1d;

          TeamPolicy probe(exec, 1, Kokkos::AUTO);
          probe.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
          const int team_size_max =
            probe.team_size_max(functor, Kokkos::ParallelForTag());
          const int team_size = std::max(
            1,
            std::min((requested_team_size == numbers::invalid_unsigned_int) ?
                       static_cast<int>(n_q_points_per_cell / n_q_points_1d) *
                         batch_size :
                       static_cast<int>(requested_team_size),
                     team_size_max));

          const int n_batches =
            (precomputed_data[0].n_cells + batch_size - 1) / batch_size;
          const int league_size =
            std::max(1,
                     (n_batches <= 2 * n_compute_units(exec)) ? n_batches :
                                                                n_batches / 2);

          TeamPolicy team_policy(exec, league_size, team_size);
          team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
          return team_policy;
        }



        /**
         * Functor of the team loop of a batched cell loop over the cells of
         * one color. A team sets up its shared memory once and then calls
         * @p func for the batches league_rank(), league_rank() +
         * league_size(), ... of @p batch_size cells, the last batch of the
         * color possibly being shorter.
         *
         * @p func needs to define
         * \code
         * DEAL_II_HOST_DEVICE void
         * operator()(const BatchedData<dim, Number> *data,
         *            const DeviceVector<Number>     &src,
         *            DeviceVector<Number>           &dst) const;
         * static const unsigned int n_q_points;
         * \endcode
         * and must return with the team synchronized, as it is after
         * FEEvaluation::distribute_local_to_global().
         */
        template <int dim, typename Number, typename Functor>
        struct ApplyKernel
        {
          using TeamHandle = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>::member_type;

          DEAL_II_HOST_DEVICE void
          operator()(const TeamHandle &team_member) const
          {
            Kokkos::Array<SharedData<dim, Number>, n_max_dof_handlers>
              shared_data;
            for (unsigned int d = 0; d < n_dof_handler; ++d)
              shared_data[d].reinit(team_member,
                                    Functor::n_q_points,
                                    batch_size,
                                    precomputed_data[d]);
            // reinit() stages the shape data without a trailing barrier
            team_member.team_barrier();

            const int n_cells = precomputed_data[0].n_cells;
            for (int batch = team_member.league_rank();
                 batch * batch_size < n_cells;
                 batch += team_member.league_size())
              {
                // The cells of the batch, numbered within the current color
                const int                    first = batch * batch_size;
                const Kokkos::pair<int, int> cell_range(
                  first,
                  (first + batch_size < n_cells) ? first + batch_size : n_cells);

                const BatchedData<dim, Number> data{team_member,
                                                    Functor::n_q_points,
                                                    n_dof_handler,
                                                    cell_range,
                                                    precomputed_data,
                                                    shared_data};

                DeviceVector<Number> nonconstdst = dst;
                func(&data, src, nonconstdst);
              }
          }

          Functor            func;
          const unsigned int n_dof_handler;
          const Kokkos::Array<typename MatrixFree<dim, Number>::PrecomputedData,
                              n_max_dof_handlers>
                                     precomputed_data;
          const int                  batch_size;
          const DeviceVector<Number> src;
          const DeviceVector<Number> dst;
        };



        /**
         * Batched counterpart of MatrixFree::cell_loop() for the first
         * DoFHandler of @p matrix_free: apply @p func, as described in
         * ApplyKernel, to batches of cells with the launch parameters of
         * compute_batch_size() and make_team_policy(). Only the locally owned
         * entries of the vectors are accessed and no ghost values are
         * exchanged.
         */
        template <int dim, typename Number, typename Functor>
        void
        cell_loop(
          const MatrixFree<dim, Number> &matrix_free,
          const Functor                 &func,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
                                                                           &src,
          LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &dst,
          const unsigned int requested_batch_size = numbers::invalid_unsigned_int,
          const unsigned int requested_team_size  = numbers::invalid_unsigned_int)
        {
          const unsigned int n_dof_handler = 1;
          const unsigned int n_colors = matrix_free.get_colored_graph().size();

          MemorySpace::Default::kokkos_space::execution_space exec;

          Kokkos::Array<typename MatrixFree<dim, Number>::PrecomputedData,
                        n_max_dof_handlers>
            colored_data;

          for (unsigned int color = 0; color < n_colors; ++color)
            {
              colored_data[0] = matrix_free.get_data(color);
              if (colored_data[0].n_cells == 0)
                continue;

              const int batch_size =
                compute_batch_size<dim, Number>(Functor::n_q_points,
                                                n_dof_handler,
                                                colored_data,
                                                requested_batch_size);

              const ApplyKernel<dim, Number, Functor> apply_kernel{
                func,
                n_dof_handler,
                colored_data,
                batch_size,
                DeviceVector<Number>(src.get_values(), src.locally_owned_size()),
                DeviceVector<Number>(dst.get_values(), dst.locally_owned_size())};

              Kokkos::parallel_for(
                "Copy::MatrixFree::batched_cell_loop color " +
                  std::to_string(color),
                make_team_policy<dim, Number>(exec,
                                              apply_kernel,
                                              Functor::n_q_points,
                                              batch_size,
                                              n_dof_handler,
                                              colored_data,
                                              requested_team_size),
                apply_kernel);
            }
          Kokkos::fence();
        }
      } // namespace batched
    } // namespace internal
  }   // namespace Portable
} // namespace Copy

#endif
