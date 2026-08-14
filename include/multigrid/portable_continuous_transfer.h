#ifndef portable_continuous_transfer_h
#define portable_continuous_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include "base/portable_mg_transfer_base.h"
#include "kernels/bk1_kokkos_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, int fe_degree, typename number>
  class ContinuousTransfer : public MGTransferBase<dim, number>
  {
  public:
    ContinuousTransfer() = default;

    void
    prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    void
    restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    void
    prolongate_and_add_block(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      const unsigned int n_rhs) const override
    {
      (void)dst;
      (void)src;
      (void)n_rhs;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    restrict_and_add_block(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      const unsigned int n_rhs) const override
    {
      (void)dst;
      (void)src;
      (void)n_rhs;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) override;

    void
    prolongate_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    restrict_and_add_internal(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

  private:
    void
    setup_dof_indices();

    ObserverPointer<const MatrixFree<dim, number>> matrix_free_coarse;
    ObserverPointer<const MatrixFree<dim, number>> matrix_free_fine;

    ObserverPointer<const AffineConstraints<number>> constraints_fine;
    ObserverPointer<const AffineConstraints<number>> constraints_coarse;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dof_indices_coarse_cg;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dof_indices_fine_dg;
  };

  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::prolongate_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not initialized correctly."));
    Assert(src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not initialized correctly."));


    this->prolongate_and_add_internal(dst, src);

    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not handled correclty after prolongation."));

    Assert(src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not handled correclty after prolongation."));
  }

  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::restrict_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not initialized correctly."));

    Assert(src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not initialized correctly."));

    this->restrict_and_add_internal(dst, src);

    Assert(dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not handled correclty after restrtiction."));

    Assert(src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not handled correclty after restrtiction."));
  }


  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;

    DeviceVector<number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto &colored_graph = matrix_free_fine->get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const auto n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;


            auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

            using MemberType = typename decltype(team_policy)::member_type;

            const auto dof_indices_coarse = this->dof_indices_coarse_cg[color];
            const auto dof_indices_fine   = this->dof_indices_fine_dg[color];

            constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

            Kokkos::parallel_for(
              "Portable::ContinuousTransfer::prolongate_and_add_color_ " + std::to_string(color),
              team_policy,
              KOKKOS_LAMBDA(const MemberType &team_member) {
                const int cell_id = team_member.league_rank();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs),
                                     [=](const int i)
                                       {
                                         const unsigned int dof_coarse =
                                           dof_indices_coarse(i, cell_id);
                                         const unsigned int dof_fine = dof_indices_fine(i, cell_id);

                                         if (dof_fine != numbers::invalid_unsigned_int &&
                                             dof_coarse != numbers::invalid_unsigned_int)
                                           dst_device[dof_fine] += src_device[dof_coarse];
                                       });
                team_member.team_barrier();
              });
            Kokkos::fence();
          }
      };

    if (matrix_free_fine->use_overlap_communication_computation())
      {
        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && colored_graph[0].size() > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && colored_graph[1].size() > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          do_color(2);

      }
    else
      {
        src.update_ghost_values();

        DeviceVector<number> src_device(src.get_values(), src.size()),
          dst_device(dst.get_values(), dst.locally_owned_size());

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (colored_graph[color].size() > 0)
              do_color(color);
          }
      }
    Kokkos::fence();

    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    MemorySpace::Default::kokkos_space::execution_space exec;

    const auto &colored_graph = matrix_free_fine->get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    DeviceVector<number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    auto do_color = [&](const unsigned int color)
      {
        const auto n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

            auto team_policy = TeamPolicy(exec, n_cells, Kokkos::AUTO);

            using MemberType = typename decltype(team_policy)::member_type;

            const auto dof_indices_coarse = this->dof_indices_coarse_cg[color];
            const auto dof_indices_fine   = this->dof_indices_fine_dg[color];

            constexpr int n_dofs = Utilities::pow(fe_degree + 1, dim);

            Kokkos::parallel_for(
              "Portable::ContinuousTransfer::restrict_and_add_color_ " + std::to_string(color),
              team_policy,
              KOKKOS_LAMBDA(const MemberType &team_member) {
                const int cell_id = team_member.league_rank();
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_dofs),
                                     [=](const int i)
                                       {
                                         const unsigned int dof_coarse =
                                           dof_indices_coarse(i, cell_id);
                                         const unsigned int dof_fine = dof_indices_fine(i, cell_id);

                                         if (dof_fine != numbers::invalid_unsigned_int &&
                                             dof_coarse != numbers::invalid_unsigned_int)
                                           Kokkos::atomic_add(&dst_device[dof_coarse],
                                                              src_device[dof_fine]);
                                       });
                team_member.team_barrier();
              });
            Kokkos::fence();
          }
      };

    if (matrix_free_fine->use_overlap_communication_computation())
      {
        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && colored_graph[0].size() > 0)
          do_color(0);

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && colored_graph[1].size() > 0)
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
        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          do_color(2);

        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (colored_graph[color].size() > 0)
              do_color(color);
          }

        dst.compress(VectorOperation::add);
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::reinit(
    const MatrixFree<dim, number>   &mf_coarse,
    const MatrixFree<dim, number>   &mf_fine,
    const AffineConstraints<number> &constraints_coarse,
    const AffineConstraints<number> &constraints_fine)
  {
    this->matrix_free_coarse = &mf_coarse;
    this->matrix_free_fine   = &mf_fine;

    this->constraints_coarse = &constraints_coarse;
    this->constraints_fine   = &constraints_fine;

    setup_dof_indices();
  }

  template <int dim, int fe_degree, typename number>
  void
  ContinuousTransfer<dim, fe_degree, number>::setup_dof_indices()
  {
    const auto &dof_handler_fine   = matrix_free_fine->get_dof_handler();
    const auto &dof_handler_coarse = matrix_free_coarse->get_dof_handler();
    const auto &fe_fine            = dof_handler_fine.get_fe();
    const auto &fe_coarse          = dof_handler_coarse.get_fe();

    const auto &colored_graph_fine   = matrix_free_fine->get_colored_graph();
    const auto &colored_graph_coarse = matrix_free_coarse->get_colored_graph();

    const unsigned int n_colors = colored_graph_fine.size();

    Assert(n_colors == colored_graph_coarse.size(),
           ExcMessage("Portable matrix free objects must have the same number of colors"));

    const unsigned int n_dofs_per_cell_fine   = fe_fine.n_dofs_per_cell();
    const unsigned int n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();

    Assert(n_dofs_per_cell_fine == n_dofs_per_cell_coarse,
           ExcMessage("Continuous transfer works on the spaces of the same degree."));

    std::vector<unsigned int> lex_numbering_fine(n_dofs_per_cell_fine);
    std::vector<unsigned int> lex_numbering_coarse(n_dofs_per_cell_coarse);

    {
      const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

      shape_info.reinit(dummy_quadrature, fe_fine, 0);
      lex_numbering_fine = shape_info.lexicographic_numbering;
    }

    {
      const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

      shape_info.reinit(dummy_quadrature, fe_coarse, 0);
      lex_numbering_coarse = shape_info.lexicographic_numbering;
    }

    // setup coarse dof indices
    {
      std::vector<types::global_dof_index> local_dof_indices_coarse(n_dofs_per_cell_coarse);
      std::vector<types::global_dof_index> subdomain_dof_indices_coarse(n_dofs_per_cell_coarse);
      std::vector<types::global_dof_index> local_dof_indices_fine(n_dofs_per_cell_fine);
      std::vector<types::global_dof_index> subdomain_dof_indices_fine(n_dofs_per_cell_fine);

      this->dof_indices_coarse_cg.clear();
      this->dof_indices_coarse_cg.resize(n_colors);

      this->dof_indices_fine_dg.clear();
      this->dof_indices_fine_dg.resize(n_colors);


      const auto &partitioner_coarse = matrix_free_coarse->get_vector_partitioner();
      const auto &partitioner_fine   = matrix_free_fine->get_vector_partitioner();

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          if (colored_graph_coarse[color].size() > 0)
            {
              const auto &mf_data_coarse = matrix_free_coarse->get_data(color);
              const auto &mf_data_fine   = matrix_free_coarse->get_data(color);

              const auto &graph_coarse = colored_graph_coarse[color];
              const auto &graph_fine   = colored_graph_fine[color];

              AssertDimension(mf_data_coarse.n_cells, mf_data_fine.n_cells);

              this->dof_indices_coarse_cg[color] =
                Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("dof_indices_coarse_cg_" + std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  n_dofs_per_cell_coarse,
                  mf_data_coarse.n_cells);

              this->dof_indices_fine_dg[color] =
                Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("dof_indices_fine_dg_" + std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  n_dofs_per_cell_fine,
                  mf_data_fine.n_cells);

              auto dof_indices_coarse_host =
                Kokkos::create_mirror_view(this->dof_indices_coarse_cg[color]);

              auto dof_indices_fine_host =
                Kokkos::create_mirror_view(this->dof_indices_fine_dg[color]);

              for (unsigned int cell_id = 0; cell_id < mf_data_coarse.n_cells; ++cell_id)
                {
                  const auto triacell_coarse = graph_coarse[cell_id];
                  const auto triacell_fine   = graph_fine[cell_id];

                  const typename DoFHandler<dim>::cell_iterator cell_coarse =
                    triacell_coarse->as_dof_handler_iterator(dof_handler_coarse);

                  const typename DoFHandler<dim>::cell_iterator cell_fine =
                    triacell_fine->as_dof_handler_iterator(dof_handler_fine);

                  cell_coarse->get_dof_indices(local_dof_indices_coarse);

                  cell_fine->get_dof_indices(local_dof_indices_fine);

                  triacell_coarse->get_dof_indices(subdomain_dof_indices_coarse);
                  triacell_fine->get_dof_indices(subdomain_dof_indices_fine);

                  if (partitioner_coarse)
                    for (auto &index : local_dof_indices_coarse)
                      index = partitioner_coarse->global_to_local(index);

                  if (partitioner_fine)
                    for (auto &index : local_dof_indices_fine)
                      index = partitioner_fine->global_to_local(index);

                  for (unsigned int i = 0; i < n_dofs_per_cell_coarse; ++i)
                    {
                      const auto global_dof_coarse =
                        local_dof_indices_coarse[lex_numbering_coarse[i]];

                      const auto global_dof_fine = local_dof_indices_fine[lex_numbering_fine[i]];

                      const auto subdomain_local_dof_coarse =
                        subdomain_dof_indices_coarse[lex_numbering_coarse[i]];

                      const auto subdomain_local_dof_fine =
                        subdomain_dof_indices_fine[lex_numbering_fine[i]];

                      if (constraints_coarse->is_constrained(subdomain_local_dof_coarse))
                        dof_indices_coarse_host(i, cell_id) = numbers::invalid_unsigned_int;
                      else
                        dof_indices_coarse_host(i, cell_id) = global_dof_coarse;

                      if (constraints_fine->is_constrained(subdomain_local_dof_fine))
                        dof_indices_fine_host(i, cell_id) = numbers::invalid_unsigned_int;
                      else
                        dof_indices_fine_host(i, cell_id) = global_dof_fine;
                    }
                }
              Kokkos::deep_copy(this->dof_indices_coarse_cg[color], dof_indices_coarse_host);
              Kokkos::deep_copy(this->dof_indices_fine_dg[color], dof_indices_fine_host);
              Kokkos::fence();
            }
        }
    }
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
