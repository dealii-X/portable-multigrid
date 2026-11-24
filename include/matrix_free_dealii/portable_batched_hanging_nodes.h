// Copy of Portable::internal::batched::resolve_hanging_nodes() from the deal.II branch
// portable-batched-cell-loop (include/deal.II/matrix_free/
// portable_hanging_nodes_internal.h), in namespace Copy, so that it does not depend on
// the version of the installed deal.II.
// index2(), index3(), is_constrained_dof_2d() and is_constrained_dof_3d() are
// the installed ones. Keep in sync with the branch.

#ifndef matrix_free_dealii_portable_batched_hanging_nodes_h
#define matrix_free_dealii_portable_batched_hanging_nodes_h

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/hanging_nodes_internal.h>
#include <deal.II/matrix_free/portable_hanging_nodes_internal.h>

#include <Kokkos_Core.hpp>

namespace Copy
{
  namespace Portable
  {
    namespace internal
    {
      using namespace dealii;

      using dealii::Portable::internal::index2;
      using dealii::Portable::internal::index3;
      using dealii::Portable::internal::is_constrained_dof_2d;
      using dealii::Portable::internal::is_constrained_dof_3d;

      namespace batched
      {
        /**
         * View of the `(fe_degree + 1)^2` 1d interpolation weights in team
         * scratch memory.
         */
        template <typename Number>
        using ConstraintWeightsView =
          Kokkos::View<Number *,
                       MemorySpace::Default::kokkos_space::execution_space::
                         scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;



        /**
         * Batched variant of Portable::internal::interpolate_boundary(): apply
         * the hanging-node interpolation along one @p direction to the cells
         * of a batch, whose values are stored cell after cell in @p values
         * with `(fe_degree + 1)^dim` entries each. @p constraint_masks is a
         * one-dimensional view of the ConstraintKinds of the cells of the
         * batch, of which there are @p batch_size, and @p constraint_weights
         * holds the 1d interpolation weights in team scratch memory.
         *
         * As in the cell-wise function, each thread handles one constrained
         * line of one cell, now with the lines of all cells of the batch in a
         * single parallel loop, followed by a single team barrier.
         */
        template <int  dim,
                  int  fe_degree,
                  int  direction,
                  bool transpose,
                  typename Number,
                  typename MaskViewType,
                  typename ViewType>
        DEAL_II_HOST_DEVICE inline void
        interpolate_boundary(
          const Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>::member_type
                                             &team_member,
          const ConstraintWeightsView<Number> constraint_weights,
          const MaskViewType                  constraint_masks,
          ViewType                            values,
          const int                           batch_size = 1)
        {
          static_assert(dim == 2 || dim == 3, "Only implemented for 2d and 3d");
          static_assert(direction >= 0 && direction < dim,
                        "direction must be in [0, dim)");

          using ConstraintKinds =
            dealii::internal::MatrixFreeFunctions::ConstraintKinds;

          constexpr unsigned int n_dofs_1d = fe_degree + 1;
          constexpr unsigned int n_dofs    = Utilities::pow(n_dofs_1d, dim);

          // lines along the direction per cell, and the stride along a line
          constexpr unsigned int n_lines = Utilities::pow(n_dofs_1d, dim - 1);
          constexpr unsigned int stride  = Utilities::pow(n_dofs_1d, direction);

          Assert(batch_size >= 0, ExcInternalError());
          Assert(constraint_masks.extent(0) >=
                   static_cast<std::size_t>(batch_size),
                 ExcInternalError());
          Assert(values.size() >= static_cast<std::size_t>(batch_size) * n_dofs,
                 ExcInternalError());

          // see interpolate_boundary() for the meaning of these flags
          const auto this_type = (direction == 0) ? ConstraintKinds::subcell_x :
                                 (direction == 1) ? ConstraintKinds::subcell_y :
                                                    ConstraintKinds::subcell_z;
          [[maybe_unused]] const auto face1_type =
            (direction == 0) ? ConstraintKinds::subcell_y :
            (direction == 1) ? ConstraintKinds::subcell_z :
                               ConstraintKinds::subcell_x;
          [[maybe_unused]] const auto face2_type =
            (direction == 0) ? ConstraintKinds::subcell_z :
            (direction == 1) ? ConstraintKinds::subcell_x :
                               ConstraintKinds::subcell_y;

          const auto face1 = (direction == 0) ? ConstraintKinds::face_y :
                             (direction == 1) ? ConstraintKinds::face_z :
                                                ConstraintKinds::face_x;
          const auto face2 = (direction == 0) ? ConstraintKinds::face_z :
                             (direction == 1) ? ConstraintKinds::face_x :
                                                ConstraintKinds::face_y;
          const auto edge  = (direction == 0) ? ConstraintKinds::edge_x :
                             (direction == 1) ? ConstraintKinds::edge_y :
                                                ConstraintKinds::edge_z;
          // in 2d the lines along x lie on the y faces and vice versa
          const auto constrained_faces =
            (dim == 2) ? ((direction == 0) ? ConstraintKinds::face_y :
                                             ConstraintKinds::face_x) :
                         (face1 | face2 | edge);

          // the same for all threads of the team, so they all skip the barrier
          ConstraintKinds batch_mask = ConstraintKinds::unconstrained;
          for (int cell = 0; cell < batch_size; ++cell)
            batch_mask |= constraint_masks(cell);
          if ((batch_mask & constrained_faces) == ConstraintKinds::unconstrained)
            return;

          Assert(constraint_weights.size() >= n_dofs_1d * n_dofs_1d,
                 ExcInternalError());

          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, batch_size * n_lines),
            [&](const int tid) {
              const int          cell = tid / n_lines;
              const unsigned int line = tid % n_lines;

              const ConstraintKinds constraint_mask = constraint_masks(cell);
              if ((constraint_mask & constrained_faces) ==
                  ConstraintKinds::unconstrained)
                return;

              // indices of the line: zero along the direction, the others
              // follow from the line index in increasing coordinate order
              unsigned int x_idx = 0, y_idx = 0, z_idx = 0;
              if constexpr (dim == 2)
                {
                  if (direction == 0)
                    y_idx = line;
                  else
                    x_idx = line;
                }
              else
                {
                  const unsigned int i1 = line % n_dofs_1d;
                  const unsigned int i2 = line / n_dofs_1d;
                  if (direction == 0)
                    {
                      y_idx = i1;
                      z_idx = i2;
                    }
                  else if (direction == 1)
                    {
                      x_idx = i1;
                      z_idx = i2;
                    }
                  else
                    {
                      x_idx = i1;
                      y_idx = i2;
                    }
                }

              bool constrained_line;
              if constexpr (dim == 2)
                constrained_line =
                  is_constrained_dof_2d<fe_degree, direction>(constraint_mask,
                                                              x_idx,
                                                              y_idx);
              else
                constrained_line =
                  is_constrained_dof_3d<fe_degree, direction>(constraint_mask,
                                                              x_idx,
                                                              y_idx,
                                                              z_idx,
                                                              face1_type,
                                                              face2_type,
                                                              face1,
                                                              face2,
                                                              edge);
              if (!constrained_line)
                return;

              const unsigned int offset =
                cell * n_dofs + ((dim == 2) ?
                                   index2<n_dofs_1d>(x_idx, y_idx) :
                                   index3<n_dofs_1d>(x_idx, y_idx, z_idx));

              // the line is read completely before it is written
              Number line_values[n_dofs_1d];
              for (unsigned int i = 0; i < n_dofs_1d; ++i)
                line_values[i] = values(offset + i * stride);

              const bool type =
                (constraint_mask & this_type) != ConstraintKinds::unconstrained;

              for (unsigned int j = 0; j < n_dofs_1d; ++j)
                {
                  Number sum = 0.;
                  if (type)
                    for (unsigned int i = 0; i < n_dofs_1d; ++i)
                      {
                        const Number w = transpose ?
                                           constraint_weights[i * n_dofs_1d + j] :
                                           constraint_weights[j * n_dofs_1d + i];
                        sum += w * line_values[i];
                      }
                  else
                    for (unsigned int i = 0; i < n_dofs_1d; ++i)
                      {
                        const Number w =
                          transpose ?
                            constraint_weights[(fe_degree - i) * n_dofs_1d +
                                               fe_degree - j] :
                            constraint_weights[(fe_degree - j) * n_dofs_1d +
                                               fe_degree - i];
                        sum += w * line_values[i];
                      }
                  values(offset + j * stride) = sum;
                }
            });

          team_member.team_barrier();
        }



        /**
         * Batched variant of Portable::internal::resolve_hanging_nodes():
         * resolve the hanging-node constraints of the cells of a batch, whose
         * values are stored cell after cell in @p values with
         * `(fe_degree + 1)^dim` entries each, by applying
         * batched::interpolate_boundary() in all directions one after the
         * other. @p constraint_masks holds the ConstraintKinds of the cells of
         * the batch, of which there are @p batch_size, and
         * @p constraint_weights the 1d interpolation weights in team scratch
         * memory.
         */
        template <int  dim,
                  int  fe_degree,
                  bool transpose,
                  typename Number,
                  typename MaskViewType,
                  typename ViewType>
        DEAL_II_HOST_DEVICE void
        resolve_hanging_nodes(
          const Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>::member_type
                                             &team_member,
          const ConstraintWeightsView<Number> constraint_weights,
          const MaskViewType                  constraint_masks,
          ViewType                            values,
          const int                           batch_size = 1)
        {
          if constexpr (dim == 2)
            {
              interpolate_boundary<2, fe_degree, 0, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               values,
                                                               batch_size);
              interpolate_boundary<2, fe_degree, 1, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               values,
                                                               batch_size);
            }
          else if constexpr (dim == 3)
            {
              // Interpolate y and z faces (x-direction)
              interpolate_boundary<3, fe_degree, 0, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               values,
                                                               batch_size);
              // Interpolate x and z faces (y-direction)
              interpolate_boundary<3, fe_degree, 1, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               values,
                                                               batch_size);
              // Interpolate x and y faces (z-direction)
              interpolate_boundary<3, fe_degree, 2, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               values,
                                                               batch_size);
            }
        }
      } // namespace batched
    } // namespace internal
  }   // namespace Portable
} // namespace Copy

#endif
