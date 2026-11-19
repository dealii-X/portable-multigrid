// Copy of Portable::internal::batched::resolve_hanging_nodes() from the
// deal.II branch batched-hanging-nodes (include/deal.II/matrix_free/
// portable_hanging_nodes_internal.h), in namespace Copy so that it can be
// compared with the cell-wise dealii::Portable::internal::
// resolve_hanging_nodes() of an installed deal.II that does not have it.
// The helpers index2(), index3(), is_constrained_dof_2d() and
// is_constrained_dof_3d() are the installed ones. Keep the body in sync with
// the branch.

#ifndef multigrid_dealii_portable_hanging_nodes_batched_h
#define multigrid_dealii_portable_hanging_nodes_batched_h

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
         * Apply the hanging-node interpolation along one @p direction to the
         * cells of a batch, whose values are stored cell after cell in @p values
         * with `(fe_degree + 1)^dim` entries each. @p constraint_masks is a
         * callable that returns the ConstraintKinds of the cell with the given
         * index within the batch, and @p batch_size the number of cells.
         *
         * For a given direction, the constrained degrees of freedom of a cell
         * are complete lines along that direction: whether a degree of freedom
         * is constrained depends only on its indices orthogonal to the
         * direction (see is_constrained_dof_2d() and is_constrained_dof_3d()),
         * and the interpolation replaces the values on a line by a linear
         * combination of the values on the same line. Each thread therefore
         * handles one line of one cell: it loads the line into registers,
         * applies the 1d weights and writes the line back. Since the lines are
         * disjoint, no temporary storage and only a single team barrier per
         * direction are needed.
         */
        template <int  dim,
                  int  fe_degree,
                  int  direction,
                  bool transpose,
                  typename ViewTypeWeights,
                  typename MaskAccessor,
                  typename ViewTypeValues>
        DEAL_II_HOST_DEVICE inline void
        interpolate_boundary(
          const Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>::member_type
                               &team_member,
          const ViewTypeWeights constraint_weights,
          const MaskAccessor    constraint_masks,
          const int             batch_size,
          ViewTypeValues        values)
        {
          static_assert(dim == 2 || dim == 3, "Only implemented for 2d and 3d");
          static_assert(direction >= 0 && direction < dim,
                        "direction must be in [0, dim)");

          using Number = typename ViewTypeValues::non_const_value_type;
          using ConstraintKinds =
            dealii::internal::MatrixFreeFunctions::ConstraintKinds;

          constexpr unsigned int n_dofs_1d = fe_degree + 1;
          constexpr unsigned int n_dofs    = Utilities::pow(n_dofs_1d, dim);
          // lines along the direction per cell, and the stride along a line
          constexpr unsigned int n_lines = Utilities::pow(n_dofs_1d, dim - 1);
          constexpr unsigned int stride  = Utilities::pow(n_dofs_1d, direction);

          // The subcell bit of this direction selects between the weights and
          // their point-reflected variant, the face (and edge) bits of the other
          // directions whether the lines on those faces are constrained at all.
          // face1_type and face2_type are only needed in 3d.
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

          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team_member, batch_size * n_lines),
            [&](const int tid) {
              const int          cell = tid / n_lines;
              const unsigned int line = tid % n_lines;

              const ConstraintKinds constraint_mask = constraint_masks(cell);
              if (!((constraint_mask & constrained_faces) !=
                    ConstraintKinds::unconstrained))
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
         * Resolve the hanging-node constraints of a batch of cells using the
         * tensor product structure, i.e., apply interpolate_boundary() in all
         * directions one after the other. The values of the cells are stored
         * cell after cell in @p values with `(fe_degree + 1)^dim` entries each,
         * @p constraint_masks is a callable returning the ConstraintKinds of the
         * cell with the given index within the batch and @p batch_size the
         * number of cells. Nothing is done if no cell of the batch is
         * constrained.
         *
         * The implementation of this function is explained in Section 3 of
         * @cite ljungkvist2017matrix and in Section 3.4 of
         * @cite kronbichler2019multigrid.
         */
        template <int  dim,
                  int  fe_degree,
                  bool transpose,
                  typename ViewTypeWeights,
                  typename MaskAccessor,
                  typename ViewTypeValues>
        DEAL_II_HOST_DEVICE void
        resolve_hanging_nodes(
          const Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>::member_type
                               &team_member,
          const ViewTypeWeights constraint_weights,
          const MaskAccessor    constraint_masks,
          const int             batch_size,
          ViewTypeValues        values)
        {
          // team-uniform early exit
          bool constrained = false;
          for (int cell = 0; cell < batch_size; ++cell)
            if (constraint_masks(cell) != dealii::internal::MatrixFreeFunctions::
                                            ConstraintKinds::unconstrained)
              constrained = true;
          if (!constrained)
            return;

          if constexpr (dim == 2)
            {
              interpolate_boundary<2, fe_degree, 0, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               batch_size,
                                                               values);
              interpolate_boundary<2, fe_degree, 1, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               batch_size,
                                                               values);
            }
          else if constexpr (dim == 3)
            {
              // Interpolate y and z faces (x-direction)
              interpolate_boundary<3, fe_degree, 0, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               batch_size,
                                                               values);
              // Interpolate x and z faces (y-direction)
              interpolate_boundary<3, fe_degree, 1, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               batch_size,
                                                               values);
              // Interpolate x and y faces (z-direction)
              interpolate_boundary<3, fe_degree, 2, transpose>(team_member,
                                                               constraint_weights,
                                                               constraint_masks,
                                                               batch_size,
                                                               values);
            }
        }
      } // namespace batched
    } // namespace internal
  } // namespace Portable
} // namespace Copy

#endif
