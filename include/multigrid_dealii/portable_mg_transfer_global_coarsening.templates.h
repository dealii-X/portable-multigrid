// -----------------------------------------------------------------------------
//
// Copy of deal.II's include/deal.II/multigrid/portable_mg_transfer_global_coarsening.templates.h
// (branch batched-portable-evaluation-kernels, commit d45a6d985a), placed in
// the namespace Copy so that it can be compiled and tested against an
// installed deal.II without rebuilding the library. Apart from the namespace,
// the include paths and the accessors for private members of the host
// transfer in mg_transfer_private_access.h, the setup code is unchanged; the
// cell kernels are the subject of this branch.
//
// -----------------------------------------------------------------------------

#ifndef multigrid_dealii_portable_mg_transfer_global_coarsening_templates_h
#define multigrid_dealii_portable_mg_transfer_global_coarsening_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/portable_hanging_nodes_internal.h>
#include <deal.II/matrix_free/portable_tensor_product_kernels.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>

#include "multigrid_dealii/mg_transfer_private_access.h"
#include "multigrid_dealii/portable_mg_transfer_global_coarsening.h"

DEAL_II_NAMESPACE_OPEN

namespace Copy
{

  namespace Portable
  {
    namespace internal
    {
      // The kernels below use the batched tensor-product evaluator and the
      // hanging-node kernels of the installed deal.II (identical to the deal.II
      // branch this file was copied from), so that the kernel code reads
      // exactly as in deal.II's Portable::internal namespace.
      namespace batched = dealii::Portable::internal::batched;
      using dealii::Portable::internal::EvaluatorVariant;
      using dealii::Portable::internal::resolve_hanging_nodes;


      /**
       * Helper class to select the right templated implementation.
       *
       * @note This class is similar to internal::FEEvaluationFactory
       */
      class CellTransferFactory
      {
      public:
        static const unsigned int max_degree = 4;

        CellTransferFactory(const unsigned int degree_fine, const unsigned int degree_coarse)
          : degree_fine(degree_fine)
          , degree_coarse(degree_coarse)
        {}

        template <typename KernelType>
        DEAL_II_HOST_DEVICE bool
        run(KernelType &kernel) const
        {
          // h-MG can use a faster path
          if (degree_fine == 2 && degree_coarse == 1)
            {
              kernel.template run<2, 1>();
              return true; // fast path taken
            }
          if (degree_fine == 4 && degree_coarse == 2)
            {
              kernel.template run<4, 2>();
              return true; // fast path taken
            }
          if (degree_fine == 6 && degree_coarse == 3)
            {
              kernel.template run<6, 3>();
              return true; // fast path taken
            }
          if (degree_fine == 8 && degree_coarse == 4)
            {
              kernel.template run<8, 4>();
              return true; // fast path taken
            }
          if (degree_fine == 10 && degree_coarse == 5)
            {
              kernel.template run<10, 5>();
              return true; // fast path taken
            }
          if (degree_fine == 12 && degree_coarse == 6)
            {
              kernel.template run<12, 6>();
              return true; // fast path taken
            }

          // due to limited functionality (e.g., FE_Q only) at the moment this
          // should be unreachable due to the checks in the initialization phase
          return false; // indicate that slow path has been taken
        }

      private:
        const unsigned int degree_fine;
        const unsigned int degree_coarse;
      };



      /**
       * Dispatch resolve_hanging_nodes for a run-time coarse degree.
       */
      class HangingNodeDegreeFactory
      {
      public:
        static const unsigned int max_degree = 9;

        explicit HangingNodeDegreeFactory(const unsigned int degree)
          : degree(degree)
        {}

        template <int dim, bool transpose, typename Number, typename ViewType>
        DEAL_II_HOST_DEVICE bool
        run(
          const Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type
                                                                      &team_member,
          Kokkos::View<Number *, MemorySpace::Default::kokkos_space>   constraint_weights,
          const dealii::internal::MatrixFreeFunctions::ConstraintKinds constraint_mask,
          ViewType                                                     values) const
        {
          // resolve_hanging_nodes() only implements dim == 2 and dim == 3.
          if constexpr (dim < 2)
            {
              (void)team_member;
              (void)constraint_weights;
              (void)constraint_mask;
              (void)values;
              return false;
            }
          else
            {
#define DEAL_II_HANGING_NODE_DEGREE_CASE(degree_value)                                \
  case degree_value:                                                                  \
    {                                                                                 \
      resolve_hanging_nodes<dim, degree_value, transpose, Number>(team_member,        \
                                                                  constraint_weights, \
                                                                  constraint_mask,    \
                                                                  values);            \
      return true;                                                                    \
    }

              switch (degree)
                {
                  DEAL_II_HANGING_NODE_DEGREE_CASE(1)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(2)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(3)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(4)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(5)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(6)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(7)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(8)
                  DEAL_II_HANGING_NODE_DEGREE_CASE(9)
                  default:
                    return false;
                }
#undef DEAL_II_HANGING_NODE_DEGREE_CASE
            }
        }

      private:
        const unsigned int degree;
      };



      /**
       * Helper class containing the cell-wise prolongation operation.
       *
       * The prolongation from the coarse cell to its children is a tensor
       * product of the 1d prolongation matrix (degree_coarse + 1 rows,
       * degree_fine + 1 columns), applied direction by direction with the
       * batched sum-factorization kernels of batched::EvaluatorTensorProduct.
       * The arrays in TransferCellData hold the values of a single component
       * of the TransferCellData::batch_size cells of the batch, cell after
       * cell; for vector-valued elements the caller loops over the components.
       *
       * There is no separate scratch pad: the directions alternate between
       * the two arrays, each direction reading from one and writing into the
       * other, and both arrays are allocated with the size of the fine values
       * so that every intermediate fits. In 3d this gives coarse -> fine ->
       * coarse -> fine; in 2d the result of the second direction ends up in
       * the coarse array and is copied into the fine array as a last step.
       */
      template <int dim, typename VectorType>
      class CellProlongator
      {
      public:
        using Number = typename VectorType::value_type;

        using TeamHandle =
          Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

        using SharedViewScratchPad =
          Kokkos::View<Number *,
                       MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        DEAL_II_HOST_DEVICE
        CellProlongator(
          const typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData *cell_data)
          : cell_data(cell_data)
        {}

        template <int degree_fine, int degree_coarse>
        DEAL_II_HOST_DEVICE void
        run() const
        {
          constexpr int n_coarse_1d = degree_coarse + 1;
          constexpr int n_fine_1d   = degree_fine + 1;

          const auto &team_member = cell_data->team_member;

          auto &values_coarse = cell_data->values_coarse;
          auto &values_fine   = cell_data->values_fine;

          // The prolongation matrix takes the role of the 1d shape values.
          // All directions are applied out of place for a single component of
          // the cells of the batch, so the evaluator needs neither gradient
          // data nor temporary storage.
          internal::batched::EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                                    dim,
                                                    n_coarse_1d,
                                                    n_fine_1d,
                                                    Number>
            eval(team_member,
                 cell_data->prolongation_matrix_device,
                 SharedViewScratchPad(),
                 SharedViewScratchPad(),
                 SharedViewScratchPad(),
                 cell_data->batch_size);

          if constexpr (dim == 1)
            {
              eval.template value<0, true, false, false>(values_coarse, values_fine);
            }
          else if constexpr (dim == 2)
            {
              // Nc x Nc -> Nf x Nc (in values_fine) -> Nf x Nf (in
              // values_coarse) -> copy to values_fine
              constexpr int n_dofs_fine = n_fine_1d * n_fine_1d;

              eval.template value<0, true, false, false>(values_coarse, values_fine);
              eval.template value<1, true, false, false>(values_fine, values_coarse);
              batched::populate_view<false>(team_member,
                                            values_fine,
                                            values_coarse,
                                            cell_data->batch_size * n_dofs_fine);
            }
          else if constexpr (dim == 3)
            {
              // Nc^3 -> Nf Nc^2 (in values_fine) -> Nf^2 Nc (in values_coarse)
              // -> Nf^3 (in values_fine)
              eval.template value<0, true, false, false>(values_coarse, values_fine);
              eval.template value<1, true, false, false>(values_fine, values_coarse);
              eval.template value<2, true, false, false>(values_coarse, values_fine);
            }
        }

      private:
        const typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData *cell_data;
      };

      /**
       * Helper class containing the cell-wise restriction operation, the
       * transpose of CellProlongator with the directions applied in reverse
       * order. The same arrays are used for the intermediate results, see
       * there.
       */
      template <int dim, typename VectorType>
      class CellRestrictor
      {
      public:
        using Number = typename VectorType::value_type;

        using TeamHandle =
          Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

        using SharedViewScratchPad =
          Kokkos::View<Number *,
                       MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        DEAL_II_HOST_DEVICE
        CellRestrictor(
          const typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData *cell_data)
          : cell_data(cell_data)
        {}

        template <int degree_fine, int degree_coarse>
        DEAL_II_HOST_DEVICE void
        run() const
        {
          constexpr int n_coarse_1d = degree_coarse + 1;
          constexpr int n_fine_1d   = degree_fine + 1;

          const auto &team_member = cell_data->team_member;

          auto &values_coarse = cell_data->values_coarse;
          auto &values_fine   = cell_data->values_fine;

          internal::batched::EvaluatorTensorProduct<EvaluatorVariant::evaluate_general,
                                                    dim,
                                                    n_coarse_1d,
                                                    n_fine_1d,
                                                    Number>
            eval(team_member,
                 cell_data->prolongation_matrix_device,
                 SharedViewScratchPad(),
                 SharedViewScratchPad(),
                 SharedViewScratchPad(),
                 cell_data->batch_size);

          if constexpr (dim == 1)
            {
              eval.template value<0, false, false, false>(values_fine, values_coarse);
            }
          else if constexpr (dim == 2)
            {
              // Nf x Nf -> Nf x Nc (in values_coarse) -> Nc x Nc (in
              // values_fine, which is free by now) -> copy to values_coarse
              constexpr int n_dofs_coarse = n_coarse_1d * n_coarse_1d;

              eval.template value<1, false, false, false>(values_fine, values_coarse);
              eval.template value<0, false, false, false>(values_coarse, values_fine);
              batched::populate_view<false>(team_member,
                                            values_coarse,
                                            values_fine,
                                            cell_data->batch_size * n_dofs_coarse);
            }
          else if constexpr (dim == 3)
            {
              // Nf^3 -> Nf^2 Nc (in values_coarse) -> Nf Nc^2 (in values_fine)
              // -> Nc^3 (in values_coarse)
              eval.template value<2, false, false, false>(values_fine, values_coarse);
              eval.template value<1, false, false, false>(values_coarse, values_fine);
              eval.template value<0, false, false, false>(values_fine, values_coarse);
            }
        }

      private:
        const typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData *cell_data;
      };

      class MGTwoLevelTransferImplementation
      {
        /**
         * Compute weights.
         */
        template <int dim, typename Number>
        static void
        setup_weights(
          const dealii::AffineConstraints<Number> &constraints_fine,
          MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>
                    &transfer,
          const bool is_feq)
        {
          if (transfer.fine_element_is_continuous == false)
            return; // nothing to do

          Assert(is_feq, ExcNotImplemented());

          // compute weights globally
          LinearAlgebra::distributed::Vector<Number> weight_vector;
          weight_vector.reinit(internal::cpu_partitioner_fine(transfer.transfer_cpu));

          // ... compute valence of DoFs
          for (const auto i : internal::cpu_constraint_info_fine(transfer.transfer_cpu).dof_indices)
            weight_vector.local_element(i) += 1.0;
          weight_vector.compress(VectorOperation::add);

          // ... invert valence
          for (unsigned int i = 0; i < weight_vector.locally_owned_size(); ++i)
            if (weight_vector.local_element(i) > 0)
              weight_vector.local_element(i) = 1.0 / weight_vector.local_element(i);

          // ... clear constrained indices
          for (const auto &constrained_dofs : constraints_fine.get_lines())
            if (weight_vector.locally_owned_elements().is_element(constrained_dofs.index))
              weight_vector[constrained_dofs.index] = 0.0;

          weight_vector.update_ghost_values();

          unsigned int cell_counter = 0;
          unsigned int scheme_index = 0;
          for (auto &scheme : transfer.schemes)
            {
              scheme.weights = Kokkos::View<Number **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("weights_h_transfer_scheme_" + std::to_string(scheme_index),
                                   Kokkos::WithoutInitializing),
                scheme.n_dofs_per_cell_fine,
                scheme.n_coarse_cells);

              auto weights_view_host = Kokkos::create_mirror_view(scheme.weights);

              const unsigned int first_cell = cell_counter;
              for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
                {
                  const unsigned int  coarse_cell_index = first_cell + cell;
                  const unsigned int *dof_indices_fine =
                    internal::cpu_constraint_info_fine(transfer.transfer_cpu).dof_indices.data() +
                    internal::cpu_constraint_info_fine(transfer.transfer_cpu)
                      .row_starts[coarse_cell_index]
                      .first;

                  for (unsigned int i = 0; i < scheme.n_dofs_per_cell_fine; ++dof_indices_fine, ++i)
                    {
                      weights_view_host(i, cell) = weight_vector.local_element(*dof_indices_fine);
                    }
                  ++cell_counter;
                }

              Kokkos::deep_copy(scheme.weights, weights_view_host);
              Kokkos::fence();
            }
        }

      public:
        template <int dim, typename Number>
        static void
        reinit_geometric_transfer(
          const DoFHandler<dim>                   &dof_handler_fine,
          const DoFHandler<dim>                   &dof_handler_coarse,
          const dealii::AffineConstraints<Number> &constraints_fine,
          const dealii::AffineConstraints<Number> &constraints_coarse,
          const unsigned int                       mg_level_fine,
          const unsigned int                       mg_level_coarse,
          MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>
            &transfer)
        {
          AssertDimension(constraints_fine.n_inhomogeneities(), 0);
          AssertDimension(constraints_coarse.n_inhomogeneities(), 0);

          transfer.transfer_cpu.reinit_geometric_transfer(dof_handler_fine,
                                                          dof_handler_coarse,
                                                          constraints_fine,
                                                          constraints_coarse,
                                                          mg_level_fine,
                                                          mg_level_coarse);

          const auto &transfer_cpu = transfer.transfer_cpu;

          transfer.dof_handler_fine = &dof_handler_fine;
          transfer.mg_level_fine    = mg_level_fine;

          transfer.schemes.resize(internal::cpu_schemes(transfer_cpu).size());

          transfer.n_components = internal::cpu_n_components(transfer_cpu);

          const auto &fe_fine = dof_handler_fine.get_fe();

          const bool is_feq =
            fe_fine.n_base_elements() == 1 &&
            ((dynamic_cast<const FE_Q<dim> *>(&fe_fine.base_element(0)) != nullptr));

          Assert(is_feq, ExcNotImplemented());

          transfer.fine_element_is_continuous =
            internal::cpu_fine_element_is_continuous(transfer_cpu);

          Assert(transfer.fine_element_is_continuous, ExcNotImplemented());

          for (unsigned int scheme_index = 0; scheme_index < transfer.schemes.size();
               ++scheme_index)
            {
              auto       &scheme     = transfer.schemes[scheme_index];
              const auto &scheme_cpu = internal::cpu_schemes(transfer_cpu)[scheme_index];

              scheme.n_dofs_per_cell_coarse = scheme_cpu.n_dofs_per_cell_coarse;
              scheme.n_dofs_per_cell_fine   = scheme_cpu.n_dofs_per_cell_fine;

              scheme.degree_coarse = scheme_cpu.degree_coarse;
              scheme.degree_fine   = scheme_cpu.degree_fine;

              scheme.n_coarse_cells = scheme_cpu.n_coarse_cells;
            }

          {
            transfer.partitioner_coarse = internal::cpu_partitioner_coarse(transfer_cpu);
            transfer.vec_coarse.reinit(transfer.partitioner_coarse);

            transfer.partitioner_fine = internal::cpu_partitioner_fine(transfer_cpu);
            transfer.vec_fine.reinit(transfer.partitioner_fine);
          }

          // -------- prolongation matrix (0) -> identity matrix --------
          //
          // nothing to do since for identity prolongation matrices a short-cut
          // code path is used during prolongation/restriction

          // --------------prolongation matrix (i = 1 ... n)--------------
          {
            for (unsigned int transfer_scheme_index = 1;
                 transfer_scheme_index < transfer.schemes.size();
                 ++transfer_scheme_index)
              {
                const auto prolongation_matrix_host =
                  internal::cpu_schemes(transfer_cpu)[transfer_scheme_index].prolongation_matrix;

                if (prolongation_matrix_host.size() > 0)
                  {
                    transfer.schemes[transfer_scheme_index].prolongation_matrix =
                      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                        Kokkos::view_alloc("prolongation_matrix_h_transfer_scheme_" +
                                             std::to_string(transfer_scheme_index),
                                           Kokkos::WithoutInitializing),
                        prolongation_matrix_host.size());

                    auto prolongation_matrix_mirror_host = Kokkos::create_mirror_view(
                      transfer.schemes[transfer_scheme_index].prolongation_matrix);

                    for (unsigned int i = 0; i < prolongation_matrix_host.size(); ++i)
                      prolongation_matrix_mirror_host[i] = prolongation_matrix_host[i];

                    Kokkos::deep_copy(transfer.schemes[transfer_scheme_index].prolongation_matrix,
                                      prolongation_matrix_mirror_host);
                    Kokkos::fence();
                  }
              }
          }

          // ------------------- hanging-node weights -------------------
          const auto &constraint_info_coarse = internal::cpu_constraint_info_coarse(transfer_cpu);
          const bool  has_hanging_nodes =
            constraint_info_coarse.hanging_node_constraint_masks.empty() == false;

          if (has_hanging_nodes)
            {
              Assert(constraint_info_coarse.shape_infos.empty() == false, ExcInternalError());
              const auto &subface_matrix = constraint_info_coarse.shape_infos.front()
                                             .data.front()
                                             .subface_interpolation_matrices[0];

              transfer.hanging_node_constraint_weights =
                Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("hanging_node_constraint_weights",
                                     Kokkos::WithoutInitializing),
                  subface_matrix.size());

              using UnmanagedViewType = Kokkos::
                View<const Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
              UnmanagedViewType view(subface_matrix.data(), subface_matrix.size());
              Kokkos::deep_copy(transfer.hanging_node_constraint_weights, view);
            }

          // ---------- coarse constraint CRS + fine dof indices ----------
          {
            MemorySpace::Default::kokkos_space::execution_space exec;

            unsigned int scheme_counter = 0;
            unsigned int cell_counter   = 0;
            for (auto &scheme : transfer.schemes)
              {
                if (scheme.n_coarse_cells == 0)
                  {
                    ++scheme_counter;
                    continue;
                  }

                const unsigned int n_rows = scheme.n_coarse_cells * scheme.n_dofs_per_cell_coarse;

                // CRS is only needed for non-Dirichlet AffineConstraints
                // (multi-entry / weighted). Homogeneous Dirichlet uses
                // invalid_unsigned_int on the identity path.
                bool use_constraint_crs = false;
                {
                  for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
                    {
                      const unsigned int coarse_cell_index = cell_counter + cell;
                      unsigned int       index_indicators =
                        constraint_info_coarse.row_starts[coarse_cell_index].second;
                      const unsigned int next_index_indicators =
                        constraint_info_coarse.row_starts[coarse_cell_index + 1].second;

                      for (; index_indicators != next_index_indicators; ++index_indicators)
                        {
                          const unsigned int pool_row =
                            constraint_info_coarse.constraint_indicator[index_indicators].second;
                          if (constraint_info_coarse.constraint_pool_data.empty() == false)
                            {
                              const unsigned int begin =
                                constraint_info_coarse.constraint_pool_row_index[pool_row];
                              const unsigned int end =
                                constraint_info_coarse.constraint_pool_row_index[pool_row + 1];
                              if (begin != end)
                                {
                                  use_constraint_crs = true;
                                  break;
                                }
                            }
                        }
                      if (use_constraint_crs)
                        break;
                    }
                }

                std::vector<unsigned int> constraint_offsets;
                std::vector<unsigned int> constraint_indices;
                std::vector<Number>       constraint_weights;
                if (use_constraint_crs)
                  {
                    constraint_offsets.resize(n_rows + 1);
                    constraint_indices.reserve(n_rows);
                    constraint_weights.reserve(n_rows);
                  }
                else
                  {
                    constraint_indices.resize(n_rows);
                  }

                scheme.dof_indices_fine =
                  Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                    Kokkos::view_alloc("h_transfer_dof_indices_fine_scheme_" +
                                         std::to_string(scheme_counter),
                                       Kokkos::WithoutInitializing),
                    scheme.n_dofs_per_cell_fine,
                    scheme.n_coarse_cells);

                auto dof_indices_fine_host = Kokkos::create_mirror_view(scheme.dof_indices_fine);

                typename std::remove_reference_t<
                  decltype(scheme.hanging_node_masks)>::host_mirror_type hanging_node_masks_host;
                if (has_hanging_nodes)
                  {
                    scheme.hanging_node_masks =
                      Kokkos::View<dealii::internal::MatrixFreeFunctions::ConstraintKinds *,
                                   MemorySpace::Default::kokkos_space>(
                        "h_transfer_hanging_node_masks_scheme_" + std::to_string(scheme_counter),
                        scheme.n_coarse_cells);
                    hanging_node_masks_host = Kokkos::create_mirror_view(scheme.hanging_node_masks);
                  }

                const unsigned int first_cell = cell_counter;
                for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
                  {
                    const unsigned int coarse_cell_index = first_cell + cell;

                    // expand ConstraintInfo into identity indices (with
                    // Dirichlet as invalid) or a CRS gather/scatter map
                    {
                      const unsigned int *dof_indices_coarse =
                        constraint_info_coarse.dof_indices.data() +
                        constraint_info_coarse.row_starts[coarse_cell_index].first;
                      unsigned int index_indicators =
                        constraint_info_coarse.row_starts[coarse_cell_index].second;
                      const unsigned int next_index_indicators =
                        constraint_info_coarse.row_starts[coarse_cell_index + 1].second;

                      unsigned int ind_local = 0;
                      for (; index_indicators != next_index_indicators; ++index_indicators)
                        {
                          const std::pair<unsigned short, unsigned short> indicator =
                            constraint_info_coarse.constraint_indicator[index_indicators];

                          for (unsigned int j = 0; j < indicator.first; ++j)
                            {
                              const unsigned int row =
                                cell * scheme.n_dofs_per_cell_coarse + ind_local + j;
                              if (use_constraint_crs)
                                {
                                  constraint_offsets[row] = constraint_indices.size();
                                  constraint_indices.push_back(dof_indices_coarse[j]);
                                  constraint_weights.push_back(Number(1.));
                                }
                              else
                                constraint_indices[row] = dof_indices_coarse[j];
                            }
                          ind_local += indicator.first;
                          dof_indices_coarse += indicator.first;

                          const unsigned int row = cell * scheme.n_dofs_per_cell_coarse + ind_local;

                          if (use_constraint_crs)
                            {
                              constraint_offsets[row] = constraint_indices.size();

                              const unsigned int pool_row = indicator.second;
                              const Number      *data_val =
                                constraint_info_coarse.constraint_pool_data.empty() ?
                                       nullptr :
                                       constraint_info_coarse.constraint_pool_data.data() +
                                    constraint_info_coarse.constraint_pool_row_index[pool_row];
                              const Number *end_pool =
                                constraint_info_coarse.constraint_pool_data.empty() ?
                                  nullptr :
                                  constraint_info_coarse.constraint_pool_data.data() +
                                    constraint_info_coarse.constraint_pool_row_index[pool_row + 1];

                              for (; data_val != end_pool; ++data_val, ++dof_indices_coarse)
                                {
                                  constraint_indices.push_back(*dof_indices_coarse);
                                  constraint_weights.push_back(*data_val);
                                }
                            }
                          else
                            {
                              // Homogeneous Dirichlet: empty constraint pool.
                              constraint_indices[row] = numbers::invalid_unsigned_int;
                            }

                          ++ind_local;
                        }

                      for (; ind_local < scheme.n_dofs_per_cell_coarse;
                           ++dof_indices_coarse, ++ind_local)
                        {
                          const unsigned int row = cell * scheme.n_dofs_per_cell_coarse + ind_local;
                          if (use_constraint_crs)
                            {
                              constraint_offsets[row] = constraint_indices.size();
                              constraint_indices.push_back(*dof_indices_coarse);
                              constraint_weights.push_back(Number(1.));
                            }
                          else
                            constraint_indices[row] = *dof_indices_coarse;
                        }
                    }

                    // fill fine indices
                    {
                      const unsigned int *dof_indices_fine =
                        internal::cpu_constraint_info_fine(transfer_cpu).dof_indices.data() +
                        internal::cpu_constraint_info_fine(transfer_cpu)
                          .row_starts[coarse_cell_index]
                          .first;

                      for (unsigned int j = 0; j < scheme.n_dofs_per_cell_fine;
                           ++dof_indices_fine, ++j)
                        dof_indices_fine_host(j, cell) = *dof_indices_fine;
                    }

                    if (has_hanging_nodes)
                      hanging_node_masks_host(cell) =
                        dealii::internal::MatrixFreeFunctions::decompress(
                          constraint_info_coarse.hanging_node_constraint_masks[coarse_cell_index],
                          dim);

                    ++cell_counter;
                  }

                if (use_constraint_crs)
                  constraint_offsets[n_rows] = constraint_indices.size();

                scheme.use_constraint_crs = use_constraint_crs;
                Assert(use_constraint_crs || constraint_indices.size() == n_rows,
                       ExcInternalError());

                scheme.coarse_constraint_indices =
                  Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                    Kokkos::view_alloc("h_transfer_coarse_constraint_indices_" +
                                         std::to_string(scheme_counter),
                                       Kokkos::WithoutInitializing),
                    constraint_indices.size());

                if (use_constraint_crs)
                  {
                    scheme.coarse_constraint_offsets =
                      Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                        Kokkos::view_alloc("h_transfer_coarse_constraint_offsets_" +
                                             std::to_string(scheme_counter),
                                           Kokkos::WithoutInitializing),
                        constraint_offsets.size());
                    scheme.coarse_constraint_weights =
                      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
                        Kokkos::view_alloc("h_transfer_coarse_constraint_weights_" +
                                             std::to_string(scheme_counter),
                                           Kokkos::WithoutInitializing),
                        constraint_weights.size());
                  }
                else
                  {
                    scheme.coarse_constraint_offsets =
                      Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>();
                    scheme.coarse_constraint_weights =
                      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>();
                  }

                {
                  using UnmanagedUIntViewType =
                    Kokkos::View<const unsigned int *,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                  using UnmanagedNumberViewType =
                    Kokkos::View<const Number *,
                                 Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

                  UnmanagedUIntViewType indices_view(constraint_indices.data(),
                                                     constraint_indices.size());

                  Kokkos::deep_copy(exec, scheme.coarse_constraint_indices, indices_view);
                  if (use_constraint_crs)
                    {
                      UnmanagedUIntViewType   offsets_view(constraint_offsets.data(),
                                                         constraint_offsets.size());
                      UnmanagedNumberViewType weights_view(constraint_weights.data(),
                                                           constraint_weights.size());

                      Kokkos::deep_copy(exec, scheme.coarse_constraint_offsets, offsets_view);
                      Kokkos::deep_copy(exec, scheme.coarse_constraint_weights, weights_view);
                    }
                  Kokkos::deep_copy(exec, scheme.dof_indices_fine, dof_indices_fine_host);
                  if (has_hanging_nodes)
                    Kokkos::deep_copy(exec, scheme.hanging_node_masks, hanging_node_masks_host);
                }

                ++scheme_counter;
              }

            exec.fence();
          }

          // ------------------- weights-------------------
          if (transfer.fine_element_is_continuous)
            setup_weights(constraints_fine, transfer, is_feq);
        }

        /**
         * Shared memory, in bytes, that the two value arrays of a team should
         * not exceed when the number of coarse cells per batch is chosen
         * automatically; see n_cells_per_batch(). 48 KB is available per
         * block on every CUDA device without opting in to larger dynamic
         * shared memory, and below the 64 KB of local data share of AMD GPUs.
         * For FE_Q in 3d this gives 111, 48, 8, 3, 2 and 1 cells per team for
         * degrees 1 to 6 in double precision.
         */
        static constexpr std::size_t shared_memory_budget = 48 * 1024;

        /**
         * Return the number of coarse cells processed by one team for the
         * given scheme: the requested number if it is not
         * numbers::invalid_unsigned_int, otherwise as many cells as fit into
         * shared_memory_budget with the two value arrays of
         * team_shmem_size(). The result is at least one and at most the
         * number of coarse cells of the scheme.
         */
        template <typename Number, typename SchemeType>
        static unsigned int
        n_cells_per_batch(const SchemeType  &scheme,
                          const unsigned int n_components,
                          const unsigned int requested)
        {
          unsigned int result = requested;
          if (result == numbers::invalid_unsigned_int)
            {
              const unsigned int n_scalar_dofs_per_cell_fine =
                scheme.n_dofs_per_cell_fine / n_components;
              result = shared_memory_budget / (2 * n_scalar_dofs_per_cell_fine * sizeof(Number));
            }
          return std::max(1u, std::min(result, scheme.n_coarse_cells));
        }

        /**
         * Return the amount of shared memory per team for the given scheme
         * and number of cells per batch, and the sizes of the arrays it
         * consists of: one coarse and one fine array for a single component
         * of all cells of the batch, both of the size of the fine values since
         * both hold intermediate results of the sum factorization (see
         * CellProlongator), and the prolongation matrix.
         */
        /**
         * Return the team size for a kernel: one thread per line of the
         * widest direction of the sum factorization of a batch, rounded up
         * to a multiple of 32, and at most what Kokkos allows for the kernel
         * with the given scratch size. With Kokkos::AUTO the team size is
         * chosen from occupancy alone and does not match the available
         * parallelism, which leaves part of each team idle.
         */
        template <typename Functor>
        static int
        team_size(const MemorySpace::Default::kokkos_space::execution_space &exec,
                  const Functor                                             &kernel,
                  const bool                                                 use_vector_lanes,
                  const unsigned int                                         n_batches,
                  const std::size_t                                          shmem_size,
                  const unsigned int                                         n_lines)
        {
          using TeamPolicy =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

          TeamPolicy probe = use_vector_lanes ?
                               TeamPolicy(exec, n_batches, Kokkos::AUTO, Kokkos::AUTO) :
                               TeamPolicy(exec, n_batches, Kokkos::AUTO);
          probe.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
          const int team_size_max = probe.team_size_max(kernel, Kokkos::ParallelForTag());

          const int rounded = ((static_cast<int>(n_lines) + 31) / 32) * 32;
          return std::max(1, std::min(rounded, team_size_max));
        }

        template <int dim, typename Number, typename SchemeType>
        static std::size_t
        team_shmem_size(const SchemeType  &scheme,
                        const unsigned int n_components,
                        const unsigned int cells_per_batch,
                        unsigned int      &values_coarse_size,
                        unsigned int      &values_fine_size,
                        unsigned int      &prolongation_matrix_size)
        {
          using SharedViewScratchPad =
            Kokkos::View<Number *,
                         MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

          values_coarse_size = cells_per_batch * (scheme.n_dofs_per_cell_coarse / n_components);
          values_fine_size   = cells_per_batch * (scheme.n_dofs_per_cell_fine / n_components);
          prolongation_matrix_size = 0;

          const bool needs_interpolation = scheme.prolongation_matrix.extent(0) > 0;
          if (needs_interpolation)
            {
              prolongation_matrix_size = (scheme.degree_coarse + 1) * (scheme.degree_fine + 1);

              // the coarse array also holds intermediate results
              values_coarse_size = values_fine_size;
            }

          return SharedViewScratchPad::shmem_size(values_coarse_size) +
                 SharedViewScratchPad::shmem_size(values_fine_size) +
                 SharedViewScratchPad::shmem_size(prolongation_matrix_size);
        }

        template <int dim, typename VectorType>
        static void
        prolongate_and_add_internal(const MGTwoLevelTransfer<dim, VectorType> &transfer,
                                    VectorType                                &dst,
                                    const VectorType                          &src)
        {
          using TeamPolicy =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

          using TeamHandle =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

          using Number = typename VectorType::value_type;

          using SharedViewScratchPad =
            Kokkos::View<Number *,
                         MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

          MemorySpace::Default::kokkos_space::execution_space exec;

          dealii::Portable::DeviceVector<Number> dst_device(dst.get_values(),
                                                            dst.locally_owned_size());
          dealii::Portable::DeviceVector<Number> src_device(src.get_values(),
                                                            src.locally_owned_size());

          unsigned int scheme_index = 0;

          const unsigned int n_components         = transfer.n_components;
          const auto         hanging_node_weights = transfer.hanging_node_constraint_weights;

          for (const auto &scheme : transfer.schemes)
            {
              // Per scheme we select:
              // - whether a prolongation matrix is needed, or the levels share
              //   the same cells (identity);
              // - how coarse constraints are applied: Dirichlet via
              //   invalid_unsigned_int indices, hanging nodes via masks, or
              //   general AffineConstraints via CRS.
              if (scheme.n_coarse_cells == 0)
                continue;

              const bool needs_interpolation = scheme.prolongation_matrix.extent(0) > 0;
              const bool has_hanging_nodes   = scheme.hanging_node_masks.extent(0) > 0;

              CellTransferFactory      cell_transfer(scheme.degree_fine, scheme.degree_coarse);
              HangingNodeDegreeFactory hanging_node_factory(scheme.degree_coarse);

              // One team processes a batch of coarse cells; the last batch may
              // be shorter.
              const unsigned int cells_per_batch =
                n_cells_per_batch<Number>(scheme, n_components, transfer.n_cells_per_batch);
              const unsigned int n_batches =
                (scheme.n_coarse_cells + cells_per_batch - 1) / cells_per_batch;

              unsigned int      values_coarse_size, values_fine_size, prolongation_matrix_size;
              const std::size_t shmem_size = team_shmem_size<dim, Number>(scheme,
                                                                          n_components,
                                                                          cells_per_batch,
                                                                          values_coarse_size,
                                                                          values_fine_size,
                                                                          prolongation_matrix_size);

              const auto         coarse_constraint_offsets = scheme.coarse_constraint_offsets;
              const auto         coarse_constraint_indices = scheme.coarse_constraint_indices;
              const auto         coarse_constraint_weights = scheme.coarse_constraint_weights;
              const auto         hanging_node_masks        = scheme.hanging_node_masks;
              const auto         dof_indices_fine          = scheme.dof_indices_fine;
              const auto         weights                   = scheme.weights;
              const auto         prolongation_matrix       = scheme.prolongation_matrix;
              const int          n_coarse_cells            = scheme.n_coarse_cells;
              const unsigned int n_dofs_coarse             = scheme.n_dofs_per_cell_coarse;
              const int          n_scalar_dofs_per_cell_coarse =
                scheme.n_dofs_per_cell_coarse / n_components;
              const int n_scalar_dofs_per_cell_fine = scheme.n_dofs_per_cell_fine / n_components;

              const bool use_crs = scheme.use_constraint_crs;

              auto kernel = KOKKOS_LAMBDA(const TeamHandle &team_member)
              {
                const int first_cell = team_member.league_rank() * cells_per_batch;
                const int n_cells_in_batch =
                  (n_coarse_cells - first_cell < static_cast<int>(cells_per_batch)) ?
                    (n_coarse_cells - first_cell) :
                    static_cast<int>(cells_per_batch);

                SharedViewScratchPad prolongation_matrix_device(team_member.team_shmem(),
                                                                prolongation_matrix_size);
                SharedViewScratchPad values_coarse(team_member.team_shmem(), values_coarse_size);
                SharedViewScratchPad values_fine(team_member.team_shmem(), values_fine_size);

                if (needs_interpolation)
                  {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                                 prolongation_matrix_size),
                                         [&](const int &i) {
                                           prolongation_matrix_device(i) = prolongation_matrix(i);
                                         });
                    team_member.team_barrier();
                  }

                typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData cell_data{
                  team_member,
                  prolongation_matrix_device,
                  values_coarse,
                  values_fine,
                  n_cells_in_batch};
                CellProlongator<dim, VectorType> cell_prolongator(&cell_data);

                // In the identity scheme (the cell exists on both levels) the
                // coarse values are scattered directly. The view handle is
                // copied: a reference to one of two local views would force
                // both into local memory.
                const SharedViewScratchPad values_fine_result =
                  needs_interpolation ? values_fine : values_coarse;

                // The shared memory holds the values of a single component
                // of the cells of the batch, cell after cell, so the
                // components are processed one after the other.
                for (unsigned int component = 0; component < n_components; ++component)
                  {
                    // read coarse dof values of this component
                    Kokkos::parallel_for(
                      Kokkos::TeamThreadRange(team_member,
                                              n_cells_in_batch * n_scalar_dofs_per_cell_coarse),
                      [&](const int &tid)
                      {
                        const int cell_in_batch     = tid / n_scalar_dofs_per_cell_coarse;
                        const int local_dof         = tid % n_scalar_dofs_per_cell_coarse;
                        const int coarse_cell_index = first_cell + cell_in_batch;

                        const unsigned int row = coarse_cell_index * n_dofs_coarse +
                                                 component * n_scalar_dofs_per_cell_coarse +
                                                 local_dof;

                        if (use_crs)
                          {
                            Number     result = 0.;
                            const auto begin  = coarse_constraint_offsets(row);
                            const auto end    = coarse_constraint_offsets(row + 1);

                            Kokkos::parallel_reduce(
                              Kokkos::ThreadVectorRange(team_member, begin, end),
                              [&](const int &k, Number &value) {
                                value += coarse_constraint_weights(k) *
                                         src_device[coarse_constraint_indices(k)];
                              },
                              Kokkos::Sum<Number>(result));

                            values_coarse(tid) = result;
                          }
                        else
                          {
                            const unsigned int global_dof = coarse_constraint_indices(row);
                            values_coarse(tid) = (global_dof != numbers::invalid_unsigned_int) ?
                                                   src_device[global_dof] :
                                                   Number(0.);
                          }
                      });
                    team_member.team_barrier();

                    // fast hanging-node interpolation on the coarse cells
                    if (has_hanging_nodes)
                      for (int cell_in_batch = 0; cell_in_batch < n_cells_in_batch; ++cell_in_batch)
                        {
                          const auto mask = hanging_node_masks(first_cell + cell_in_batch);
                          if (mask !=
                              dealii::internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                            hanging_node_factory.template run<dim, false, Number>(
                              team_member,
                              hanging_node_weights,
                              mask,
                              Kokkos::subview(values_coarse,
                                              Kokkos::make_pair(cell_in_batch *
                                                                  n_scalar_dofs_per_cell_coarse,
                                                                (cell_in_batch + 1) *
                                                                  n_scalar_dofs_per_cell_coarse)));
                        }

                    // interpolate to the children
                    if (needs_interpolation)
                      cell_transfer.run(cell_prolongator);

                    // apply weights and add the fine values of this component
                    // into the result vector
                    Kokkos::parallel_for(
                      Kokkos::TeamThreadRange(team_member,
                                              n_cells_in_batch * n_scalar_dofs_per_cell_fine),
                      [&](const int &tid)
                      {
                        const int cell_in_batch     = tid / n_scalar_dofs_per_cell_fine;
                        const int local_dof         = tid % n_scalar_dofs_per_cell_fine;
                        const int coarse_cell_index = first_cell + cell_in_batch;

                        const unsigned int index =
                          component * n_scalar_dofs_per_cell_fine + local_dof;

                        const unsigned int global_dof = dof_indices_fine(index, coarse_cell_index);

                        const Number weight = weights(index, coarse_cell_index);

                        Kokkos::atomic_add(&dst_device[global_dof],
                                           weight * values_fine_result(tid));
                      });

                    // the next component overwrites the shared arrays
                    team_member.team_barrier();
                  }
              };

              // one thread per line of the widest direction (Nf^(dim-1) lines per
              // cell), or per fine dof for the identity scheme
              const unsigned int n_lines =
                needs_interpolation ?
                  cells_per_batch * Utilities::pow(scheme.degree_fine + 1, dim - 1) :
                  cells_per_batch * n_scalar_dofs_per_cell_fine;
              const int size_of_team =
                team_size(exec, kernel, use_crs, n_batches, shmem_size, n_lines);
              TeamPolicy team_policy = use_crs ?
                                         TeamPolicy(exec, n_batches, size_of_team, Kokkos::AUTO) :
                                         TeamPolicy(exec, n_batches, size_of_team);
              team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

              Kokkos::parallel_for("prolongate_and_add_geometric_transfer_scheme_" +
                                     std::to_string(scheme_index),
                                   team_policy,
                                   kernel);

              ++scheme_index;
            }
        }

        template <int dim, typename VectorType>
        static void
        restrict_and_add_internal(const MGTwoLevelTransfer<dim, VectorType> &transfer,
                                  VectorType                                &dst,
                                  const VectorType                          &src)
        {
          using TeamPolicy =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

          using TeamHandle =
            Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;

          using Number = typename VectorType::value_type;

          using SharedViewScratchPad =
            Kokkos::View<Number *,
                         MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

          MemorySpace::Default::kokkos_space::execution_space exec;

          dealii::Portable::DeviceVector<Number> dst_device(dst.get_values(),
                                                            dst.locally_owned_size());
          dealii::Portable::DeviceVector<Number> src_device(src.get_values(),
                                                            src.locally_owned_size());

          const unsigned int n_components         = transfer.n_components;
          const auto         hanging_node_weights = transfer.hanging_node_constraint_weights;

          unsigned int scheme_index = 0;

          for (const auto &scheme : transfer.schemes)
            {
              // Per scheme we select:
              // - whether a prolongation matrix is needed, or the levels share
              //   the same cells (identity);
              // - how coarse constraints are applied: Dirichlet via
              //   invalid_unsigned_int indices, hanging nodes via masks, or
              //   general AffineConstraints via CRS.

              if (scheme.n_coarse_cells == 0)
                continue;

              const bool needs_interpolation = scheme.prolongation_matrix.extent(0) > 0;
              const bool has_hanging_nodes   = scheme.hanging_node_masks.extent(0) > 0;

              CellTransferFactory      cell_transfer(scheme.degree_fine, scheme.degree_coarse);
              HangingNodeDegreeFactory hanging_node_factory(scheme.degree_coarse);

              // One team processes a batch of coarse cells; the last batch may
              // be shorter.
              const unsigned int cells_per_batch =
                n_cells_per_batch<Number>(scheme, n_components, transfer.n_cells_per_batch);
              const unsigned int n_batches =
                (scheme.n_coarse_cells + cells_per_batch - 1) / cells_per_batch;

              unsigned int      values_coarse_size, values_fine_size, prolongation_matrix_size;
              const std::size_t shmem_size = team_shmem_size<dim, Number>(scheme,
                                                                          n_components,
                                                                          cells_per_batch,
                                                                          values_coarse_size,
                                                                          values_fine_size,
                                                                          prolongation_matrix_size);

              const auto         coarse_constraint_offsets = scheme.coarse_constraint_offsets;
              const auto         coarse_constraint_indices = scheme.coarse_constraint_indices;
              const auto         coarse_constraint_weights = scheme.coarse_constraint_weights;
              const auto         hanging_node_masks        = scheme.hanging_node_masks;
              const auto         dof_indices_fine          = scheme.dof_indices_fine;
              const auto         weights                   = scheme.weights;
              const auto         prolongation_matrix       = scheme.prolongation_matrix;
              const int          n_coarse_cells            = scheme.n_coarse_cells;
              const unsigned int n_dofs_coarse             = scheme.n_dofs_per_cell_coarse;
              const int          n_scalar_dofs_per_cell_coarse =
                scheme.n_dofs_per_cell_coarse / n_components;
              const int n_scalar_dofs_per_cell_fine = scheme.n_dofs_per_cell_fine / n_components;

              const bool use_crs = scheme.use_constraint_crs;

              auto kernel = KOKKOS_LAMBDA(const TeamHandle &team_member)
              {
                const int first_cell = team_member.league_rank() * cells_per_batch;
                const int n_cells_in_batch =
                  (n_coarse_cells - first_cell < static_cast<int>(cells_per_batch)) ?
                    (n_coarse_cells - first_cell) :
                    static_cast<int>(cells_per_batch);

                SharedViewScratchPad prolongation_matrix_device(team_member.team_shmem(),
                                                                prolongation_matrix_size);
                SharedViewScratchPad values_coarse(team_member.team_shmem(), values_coarse_size);
                SharedViewScratchPad values_fine(team_member.team_shmem(), values_fine_size);

                if (needs_interpolation)
                  {
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member,
                                                                 prolongation_matrix_size),
                                         [&](const int &i) {
                                           prolongation_matrix_device(i) = prolongation_matrix(i);
                                         });
                    team_member.team_barrier();
                  }

                typename MGTwoLevelTransfer<dim, VectorType>::TransferCellData cell_data{
                  team_member,
                  prolongation_matrix_device,
                  values_coarse,
                  values_fine,
                  n_cells_in_batch};
                CellRestrictor<dim, VectorType> cell_restrictor(&cell_data);

                // In the identity scheme (the cell exists on both levels) the
                // fine values are the coarse values. The view handle is
                // copied, see prolongate_and_add_internal().
                const SharedViewScratchPad values_coarse_result =
                  needs_interpolation ? values_coarse : values_fine;

                // The shared memory holds the values of a single component
                // of the cells of the batch, cell after cell, so the
                // components are processed one after the other.
                for (unsigned int component = 0; component < n_components; ++component)
                  {
                    // read fine dof values of this component and apply weights
                    Kokkos::parallel_for(
                      Kokkos::TeamThreadRange(team_member,
                                              n_cells_in_batch * n_scalar_dofs_per_cell_fine),
                      [&](const int &tid)
                      {
                        const int cell_in_batch     = tid / n_scalar_dofs_per_cell_fine;
                        const int local_dof         = tid % n_scalar_dofs_per_cell_fine;
                        const int coarse_cell_index = first_cell + cell_in_batch;

                        const unsigned int index =
                          component * n_scalar_dofs_per_cell_fine + local_dof;

                        const Number weight = weights(index, coarse_cell_index);

                        const unsigned int global_dof = dof_indices_fine(index, coarse_cell_index);

                        values_fine(tid) = weight * src_device[global_dof];
                      });
                    team_member.team_barrier();

                    // restrict from the children
                    if (needs_interpolation)
                      cell_transfer.run(cell_restrictor);

                    // transpose hanging-node interpolation before scattering
                    if (has_hanging_nodes)
                      for (int cell_in_batch = 0; cell_in_batch < n_cells_in_batch; ++cell_in_batch)
                        {
                          const auto mask = hanging_node_masks(first_cell + cell_in_batch);
                          if (mask !=
                              dealii::internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                            hanging_node_factory.template run<dim, true, Number>(
                              team_member,
                              hanging_node_weights,
                              mask,
                              Kokkos::subview(values_coarse_result,
                                              Kokkos::make_pair(cell_in_batch *
                                                                  n_scalar_dofs_per_cell_coarse,
                                                                (cell_in_batch + 1) *
                                                                  n_scalar_dofs_per_cell_coarse)));
                        }

                    // distribute coarse dof values of this component
                    Kokkos::parallel_for(
                      Kokkos::TeamThreadRange(team_member,
                                              n_cells_in_batch * n_scalar_dofs_per_cell_coarse),
                      [&](const int &tid)
                      {
                        const int cell_in_batch     = tid / n_scalar_dofs_per_cell_coarse;
                        const int local_dof         = tid % n_scalar_dofs_per_cell_coarse;
                        const int coarse_cell_index = first_cell + cell_in_batch;

                        const Number       value = values_coarse_result(tid);
                        const unsigned int row   = coarse_cell_index * n_dofs_coarse +
                                                 component * n_scalar_dofs_per_cell_coarse +
                                                 local_dof;

                        if (use_crs)
                          {
                            const auto begin = coarse_constraint_offsets(row);
                            const auto end   = coarse_constraint_offsets(row + 1);

                            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, begin, end),
                                                 [&](const int &k)
                                                 {
                                                   Kokkos::atomic_add(
                                                     &dst_device[coarse_constraint_indices(k)],
                                                     coarse_constraint_weights(k) * value);
                                                 });
                          }
                        else
                          {
                            const unsigned int global_dof = coarse_constraint_indices(row);
                            if (global_dof != numbers::invalid_unsigned_int)
                              Kokkos::atomic_add(&dst_device[global_dof], value);
                          }
                      });

                    // the next component overwrites the shared arrays
                    team_member.team_barrier();
                  }
              };

              // one thread per line of the widest direction (Nf^(dim-1) lines per
              // cell), or per fine dof for the identity scheme
              const unsigned int n_lines =
                needs_interpolation ?
                  cells_per_batch * Utilities::pow(scheme.degree_fine + 1, dim - 1) :
                  cells_per_batch * n_scalar_dofs_per_cell_fine;
              const int size_of_team =
                team_size(exec, kernel, use_crs, n_batches, shmem_size, n_lines);
              TeamPolicy team_policy = use_crs ?
                                         TeamPolicy(exec, n_batches, size_of_team, Kokkos::AUTO) :
                                         TeamPolicy(exec, n_batches, size_of_team);
              team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

              Kokkos::parallel_for("restrict_and_add_geometric_transfer_scheme_" +
                                     std::to_string(scheme_index),
                                   team_policy,
                                   kernel);

              ++scheme_index;
            }
        }
      };
    } // namespace internal

    template <int dim, typename VectorType>
    void
    MGTwoLevelTransfer<dim, VectorType>::prolongate_and_add_internal(VectorType       &dst,
                                                                     const VectorType &src) const
    {
      internal::MGTwoLevelTransferImplementation::prolongate_and_add_internal(*this, dst, src);
    }

    template <int dim, typename VectorType>
    void
    MGTwoLevelTransfer<dim, VectorType>::restrict_and_add_internal(VectorType       &dst,
                                                                   const VectorType &src) const
    {
      internal::MGTwoLevelTransferImplementation::restrict_and_add_internal(*this, dst, src);
    }

    template <int dim, typename VectorType>
    std::pair<const DoFHandler<dim> *, unsigned int>
    MGTwoLevelTransfer<dim, VectorType>::get_dof_handler_fine() const
    {
      return {this->dof_handler_fine, this->mg_level_fine};
    }

    template <int dim, typename VectorType>
    void
    MGTwoLevelTransfer<dim, VectorType>::set_cells_per_batch(const unsigned int n_cells)
    {
      n_cells_per_batch = n_cells;
    }

    template <int dim, typename VectorType>
    std::size_t
    MGTwoLevelTransfer<dim, VectorType>::memory_consumption() const
    {
      Assert(false, ExcNotImplemented());

      return 0;
    }

    template <int dim, typename VectorType>
    void
    MGTwoLevelTransfer<dim, VectorType>::interpolate(VectorType &dst, const VectorType &src) const
    {
      (void)dst;
      (void)src;
      Assert(false, ExcNotImplemented());
    }

    template <int dim, typename VectorType>
    std::pair<bool, bool>
    MGTwoLevelTransfer<dim, VectorType>::enable_inplace_operations_if_possible(
      const std::shared_ptr<const Utilities::MPI::Partitioner> &external_partitioner_coarse,
      const std::shared_ptr<const Utilities::MPI::Partitioner> &external_partitioner_fine)
    {
      return this->internal_enable_inplace_operations_if_possible(
        external_partitioner_coarse,
        external_partitioner_fine,
        this->vec_fine_needs_ghost_update,
        internal::cpu_constraint_info_coarse(transfer_cpu),
        internal::cpu_constraint_info_fine(transfer_cpu).dof_indices);
    }

    template <int dim, typename VectorType>
    void
    MGTwoLevelTransfer<dim, VectorType>::reinit_geometric_transfer(
      const DoFHandler<dim>           &dof_handler_fine,
      const DoFHandler<dim>           &dof_handler_coarse,
      const AffineConstraints<Number> &constraints_fine,
      const AffineConstraints<Number> &constraints_coarse,
      const unsigned int               mg_level_fine,
      const unsigned int               mg_level_coarse)
    {
      internal::MGTwoLevelTransferImplementation::reinit_geometric_transfer(dof_handler_fine,
                                                                            dof_handler_coarse,
                                                                            constraints_fine,
                                                                            constraints_coarse,
                                                                            mg_level_fine,
                                                                            mg_level_coarse,
                                                                            *this);
    }
  } // namespace Portable

} // namespace Copy

DEAL_II_NAMESPACE_CLOSE

#endif
