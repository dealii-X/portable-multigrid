#ifndef portable_masked_dof_indices_h
#define portable_masked_dof_indices_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <functional>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {
    /**
     * Builds a per-color dof-index array with the same layout
     * SubdomainLaplaceOperator's plain_dof_indices_per_color/
     * interior_dof_indices_per_color use (n_local_dofs x n_cells,
     * numbers::invalid_unsigned_int marking an excluded dof), for an
     * arbitrary "is this subdomain-local dof excluded" predicate --
     * rather than one baked in via an AffineConstraints object built
     * against this specific matrix_free/dof_handler.
     *
     * Doesn't need fe_degree as a template parameter: n_local_dofs and the
     * lexicographic renumbering both come from the DoFHandler's own
     * (runtime) FiniteElement, matching how
     * SubdomainLaplaceOperator::setup_dof_indices_per_color() already
     * computes lex_numbering. Only actually launching the tensor-product
     * kernel against the resulting array needs a compile-time fe_degree
     * (see SubdomainLaplaceOperatorBase::vmult_masked()), which stays on
     * the concrete SubdomainLaplaceOperator via virtual dispatch.
     */
    template <int dim, typename Number>
    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
    build_masked_dof_indices_per_color(
      const MatrixFree<dim, Number>                &matrix_free,
      const DoFHandler<dim>                         &dof_handler,
      const std::function<bool(unsigned int)>       &is_excluded)
    {
      dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

      const auto        &colored_graph = matrix_free.get_colored_graph();
      const unsigned int n_colors      = colored_graph.size();
      const auto        &partitioner   = matrix_free.get_vector_partitioner();

      const unsigned int n_local_dofs = dof_handler.get_fe().n_dofs_per_cell();

      std::vector<unsigned int> lex_numbering(n_local_dofs);
      {
        const Quadrature<1>                                       dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
        lex_numbering = shape_info.lexicographic_numbering;
      }

      std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
        dof_indices_per_color(n_colors);

      std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
      std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          if (colored_graph[color].size() == 0)
            continue;

          const auto &mf_data = matrix_free.get_data(color);
          const auto &graph   = colored_graph[color];

          dof_indices_per_color[color] =
            Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("masked_dof_indices_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_local_dofs,
              mf_data.n_cells);

          auto dof_indices_host = Kokkos::create_mirror_view(dof_indices_per_color[color]);

          for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
            {
              auto triacell = graph[cell_id];

              typename DoFHandler<dim>::cell_iterator cell =
                triacell->as_dof_handler_iterator(dof_handler);

              cell->get_dof_indices(local_dof_indices);
              triacell->get_dof_indices(subdomain_local_dof_indices);

              if (partitioner)
                for (auto &index : local_dof_indices)
                  index = partitioner->global_to_local(index);

              for (unsigned int i = 0; i < n_local_dofs; ++i)
                {
                  const auto global_dof = local_dof_indices[lex_numbering[i]];
                  const unsigned int subdomain_local_dof =
                    subdomain_local_dof_indices[lex_numbering[i]];

                  dof_indices_host(i, cell_id) =
                    is_excluded(subdomain_local_dof) ? numbers::invalid_unsigned_int : global_dof;
                }
            }

          Kokkos::deep_copy(exec_space, dof_indices_per_color[color], dof_indices_host);
        }

      return dof_indices_per_color;
    }

    /**
     * A copy of base_constraints with additional_pinned_dofs also added as
     * trivial (homogeneous, no-dependency) constraint lines -- the transfer
     * classes' own reinit() needs a real AffineConstraints object (their
     * dof-index/weight setup goes through dealii's ConstraintInfo, which is
     * built around that type), but building one from a flat list of extra
     * pinned dofs on top of an already-built AffineConstraints (e.g.
     * physical-boundary-only constraints) is cheap -- no
     * DoFTools::make_hanging_node_constraints/VectorTools::
     * interpolate_boundary_values re-derivation needed, since
     * additional_pinned_dofs (primal-constraint dofs) are ordinary,
     * non-hanging mesh dofs that just also want to be pinned to zero.
     * left_object_wins in case a primal-constraint dof happens to already be
     * physically constrained (a subdomain touching the true domain
     * boundary) -- both sides pin to the same value (zero) either way.
     */
    template <typename Number>
    AffineConstraints<Number>
    build_pinned_constraints(const AffineConstraints<Number> &base_constraints,
                             const std::vector<unsigned int> &additional_pinned_dofs)
    {
      AffineConstraints<Number> extra;
      for (const unsigned int dof : additional_pinned_dofs)
        if (!base_constraints.is_constrained(dof))
          extra.add_line(dof);
      extra.close();

      AffineConstraints<Number> merged;
      merged.copy_from(base_constraints);
      merged.merge(extra, AffineConstraints<Number>::MergeConflictBehavior::left_object_wins);
      return merged;
    }

  } // namespace internal
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
