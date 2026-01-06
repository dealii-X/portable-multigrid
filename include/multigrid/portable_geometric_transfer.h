#ifndef portable_geometric_transfer_h
#define portable_geometric_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include "base/portable_mg_transfer_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename number>
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
    setup_weights_and_boundary_dofs_masks();

    /**
     * A multigrid transfer scheme. A multrigrid transfer class can have
     * different transfer transfer_schemes to enable p-adaptivity (one transfer
     * scheme per polynomial degree pair) and to enable global coarsening (one
     * transfer scheme for transfer between children and parent cells, as well
     * as, one transfer scheme for cells that are not refined).
     */
    struct MGTransferScheme
    {
      /**
       * Number of coarse cells.
       */
      unsigned int n_coarse_cells;

      /**
       * Number of degrees of freedom of a coarse cell.
       *
       * @note For tensor-product elements, the value equals
       *   `n_components * (degree_coarse + 1)^dim`.
       */
      unsigned int n_dofs_per_cell_coarse;

      /**
       * Number of degrees of freedom of fine cell.
       *
       * @note For tensor-product elements, the value equals
       *   `n_components * (n_dofs_per_cell_fine + 1)^dim`.
       */
      unsigned int n_dofs_per_cell_fine;

      /**
       * Polynomial degree of the finite element of a coarse cell.
       */
      unsigned int degree_coarse;

      /**
       * "Polynomial degree" of the finite element of the union of all children
       * of a coarse cell, i.e., actually `degree_fine * 2 + 1` if a cell is
       * refined.
       */
      unsigned int degree_fine;

      /**
       * Prolongation matrix used for the prolongate_and_add() and
       * restrict_and_add() functions.
       */
      AlignedVector<double> prolongation_matrix;

      /**
       * Restriction matrix used for the interpolate() function.
       */
      AlignedVector<double> restriction_matrix;

      /**
       * ShapeInfo description of the coarse cell. Needed during the
       * fast application of hanging-node constraints.
       */
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double>
        shape_info_coarse;
    };

    std::vector<MGTransferScheme> transfer_schemes;

    ObserverPointer<const MatrixFree<dim, number>> matrix_free_coarse;
    ObserverPointer<const MatrixFree<dim, number>> matrix_free_fine;


    ObserverPointer<const AffineConstraints<number>> constraints_fine;
    ObserverPointer<const AffineConstraints<number>> constraints_coarse;

    dealii::internal::MatrixFreeFunctions::
      ConstraintInfo<dim, VectorizedArray<number>, types::global_dof_index>
        constraint_info_fine;

    dealii::internal::MatrixFreeFunctions::
      ConstraintInfo<dim, VectorizedArray<number>, types::global_dof_index>
        constraint_info_coarse;

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

  template <int dim, typename number>
  GeometricTransfer<dim, number>::GeometricTransfer()
  {}

  template <int dim, typename number>
  void
  GeometricTransfer<dim, number>::prolongate_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    
    return;
  }

  template <int dim, typename number>
  void
  GeometricTransfer<dim, number>::restrict_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    return;
  }

  template <int dim, typename number>
  void
  GeometricTransfer<dim, number>::reinit(
    const MatrixFree<dim, number>   &mf_coarse,
    const MatrixFree<dim, number>   &mf_fine,
    const AffineConstraints<number> &constraints_coarse,
    const AffineConstraints<number> &constraints_fine)
  {
    const unsigned int mg_level_coarse = mf_coarse.get_mg_level();
    const unsigned int mg_level_fine   = mf_fine.get_mg_level();


    Assert((mg_level_fine == numbers::invalid_unsigned_int &&
            mg_level_coarse == numbers::invalid_unsigned_int) ||
             (mg_level_coarse + 1 == mg_level_fine),
           ExcNotImplemented());

    this->matrix_free_coarse = &mf_coarse;
    this->matrix_free_fine   = &mf_fine;

    this->constraints_coarse = &constraints_coarse;
    this->constraints_fine   = &constraints_fine;



    const auto &dof_handler_coarse = mf_coarse.get_dof_handler();
    const auto &dof_handler_fine   = mf_fine.get_dof_handler();

    Assert(
      dof_handler_coarse.get_mpi_communicator() ==
        dof_handler_fine.get_mpi_communicator(),
      ExcMessage(
        "Coarse and fine DoFHandler must have the same MPI communicator."));


    std::unique_ptr<dealii::internal::FineDoFHandlerViewBase<dim>>
      dof_handler_fine_view = std::make_unique<
        dealii::internal::GlobalCoarseningFineDoFHandlerView<dim>>(
        dof_handler_fine, dof_handler_coarse, mg_level_fine, mg_level_coarse);

    const auto reference_cell = dof_handler_fine.get_fe().reference_cell();

    // set up mg-transfer_schemes
    //   (0) no refinement -> identity
    //   (1) h-refinement
    //   (2) - other
    transfer_schemes.resize(
      1 + reference_cell.n_isotropic_refinement_choices()); // size=2


    const auto &fe_fine   = dof_handler_fine.get_fe();
    const auto &fe_coarse = dof_handler_coarse.get_fe();

    // helper function: to process the fine level cells; function fu_non_refined
    // is performed on cells that are not refined and fu_refined is performed
    // on children of cells that are refined
    const auto process_cells = [&](const auto &fu_non_refined,
                                   const auto &fu_refined) {
      dealii::internal::loop_over_active_or_level_cells(
        dof_handler_coarse, mg_level_coarse, [&](const auto &cell_coarse) {
          if (mg_level_coarse == numbers::invalid_unsigned_int)
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
        // number of dofs on coarse and fine cells
        scheme.n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();
        scheme.n_dofs_per_cell_fine =
          Utilities::pow(2 * fe_fine.degree + 1, dim);

        // degree of FE on coarse and fine cell
        scheme.degree_coarse = fe_coarse.degree;
        scheme.degree_fine   = fe_coarse.degree * 2;

        // reset number of coarse cells
        scheme.n_coarse_cells = 0;
      }

    // correct for first scheme
    transfer_schemes[0].n_dofs_per_cell_fine = fe_coarse.n_dofs_per_cell();
    transfer_schemes[0].degree_fine          = fe_coarse.degree;

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

      this->constraint_info_coarse.reinit(dof_handler_coarse,
                                          n_coarse_cells_total,
                                          constraints_coarse.n_constraints() >
                                            0);

      this->constraint_info_coarse.set_locally_owned_indices(
        (mg_level_coarse == numbers::invalid_unsigned_int) ?
          dof_handler_coarse.locally_owned_dofs() :
          dof_handler_coarse.locally_owned_mg_dofs(mg_level_coarse));

      this->constraint_info_fine.reinit(n_coarse_cells_total);

      this->constraint_info_fine.set_locally_owned_indices(
        (mg_level_fine == numbers::invalid_unsigned_int) ?
          dof_handler_fine.locally_owned_dofs() :
          dof_handler_fine.locally_owned_mg_dofs(mg_level_fine));


      std::vector<unsigned int> cell_no(transfer_schemes.size(), 0);
      for (unsigned int i = 1; i < transfer_schemes.size(); ++i)
        cell_no[i] = cell_no[i - 1] + transfer_schemes[i - 1].n_coarse_cells;

      process_cells(
        [&](const auto &cell_coarse, const auto &cell_fine) {
          // first process cells with scheme 0
          // parent
          {
            this->constraint_info_coarse.read_dof_indices(
              cell_no[0], mg_level_coarse, cell_coarse, constraints_coarse, {});
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
                mg_level_coarse,
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
      auto partitioner_coarse = this->constraint_info_coarse.finalize(
        dof_handler_coarse.get_mpi_communicator());

      auto partitioner_fine = this->constraint_info_fine.finalize(
        dof_handler_fine.get_mpi_communicator());

      if constexpr (running_in_debug_mode())
        {
          // We would like to assert that no strange indices were added in
          // the transfer. Unfortunately, we can only do this if we're
          // working with the multigrid indices within the DoFHandler, not
          // when the transfer comes from different DoFHandler object, as
          // the latter might have unrelated parallel partitions.
          if (mg_level_fine != numbers::invalid_unsigned_int)
            {
              Utilities::MPI::Partitioner part_check(
                dof_handler_fine.locally_owned_mg_dofs(mg_level_fine),
                DoFTools::extract_locally_relevant_level_dofs(dof_handler_fine,
                                                              mg_level_fine),
                dof_handler_fine.get_mpi_communicator());
              Assert(partitioner_fine->ghost_indices().is_subset_of(
                       part_check.ghost_indices()),
                     ExcMessage(
                       "The setup of ghost indices failed, because the set "
                       "of ghost indices identified for the transfer is "
                       "not a subset of the locally relevant dofs on level " +
                       std::to_string(mg_level_fine) + " with " +
                       std::to_string(dof_handler_fine.n_dofs(mg_level_fine)) +
                       " dofs in total, which means we do not understand "
                       "the indices that were collected. This is very "
                       "likely a bug in deal.II, and could e.g. be caused "
                       "by some integer type narrowing between 64 bit and "
                       "32 bit integers."));
            }
        }
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

          // TODO: data structures are saved in form of DG data structures
          // here
          const unsigned int shift =
            fe.n_dofs_per_cell() - fe.n_dofs_per_vertex();
          const unsigned int n_child_dofs_1d =
            fe.n_dofs_per_cell() * 2 - fe.n_dofs_per_vertex();

          {
            transfer_schemes[transfer_scheme_index].prolongation_matrix.resize(
              fe.n_dofs_per_cell() * n_child_dofs_1d);
            for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell;
                 ++c)
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                  transfer_schemes[transfer_scheme_index]
                    .prolongation_matrix[i * n_child_dofs_1d + j + c * shift] =
                    fe.get_prolongation_matrix(c)(renumbering[j],
                                                  renumbering[i]);
          }
        }
    }



    for (int count = 1; count < transfer_schemes.size(); ++count)
      {
        std::cout << "Prolongation matrix scheme " << count << " : \n";
        for (unsigned int i = 0;
             i < transfer_schemes[count].prolongation_matrix.size();
             ++i)
          {
            std::cout << transfer_schemes[count].prolongation_matrix[i] << " ";
            std::cout << "\n";
          }
      }


    // auto &colored_graph_coarse = mf_coarse.get_colored_graph();

    // const auto &colored_graph_fine = mf_fine.get_colored_graph();

    // const unsigned int n_colors = colored_graph_fine.size();

    // Assert(n_colors == colored_graph_coarse.size(),
    //        ExcMessage(
    //          "Coarse and fine levels must have the same number of
    //          colors"));

    // FE_Q<1> fe_coarse_1d(p_coarse);
    // FE_Q<1> fe_fine_1d(p_fine);

    // std::vector<unsigned int>
    // renumbering_fine(fe_fine_1d.n_dofs_per_cell());

    // renumbering_fine[0] = 0;
    // for (unsigned int i = 0; i < fe_fine_1d.dofs_per_line; ++i)
    //   renumbering_fine[i + fe_fine_1d.n_dofs_per_vertex()] =
    //     GeometryInfo<1>::vertices_per_cell *
    //     fe_fine_1d.n_dofs_per_vertex() + i;

    // if (fe_fine_1d.n_dofs_per_vertex() > 0)
    //   renumbering_fine[fe_fine_1d.n_dofs_per_cell() -
    //                    fe_fine_1d.n_dofs_per_vertex()] =
    //     fe_fine_1d.n_dofs_per_vertex();

    // std::vector<unsigned int> renumbering_coarse(
    //   fe_coarse_1d.n_dofs_per_cell());

    // renumbering_coarse[0] = 0;
    // for (unsigned int i = 0; i < fe_coarse_1d.dofs_per_line; ++i)
    //   renumbering_coarse[i + fe_coarse_1d.n_dofs_per_vertex()] =
    //     GeometryInfo<1>::vertices_per_cell *
    //     fe_coarse_1d.n_dofs_per_vertex()
    //     + i;

    // if (fe_coarse_1d.n_dofs_per_vertex() > 0)
    //   renumbering_coarse[fe_coarse_1d.n_dofs_per_cell() -
    //                      fe_coarse_1d.n_dofs_per_vertex()] =
    //     fe_coarse_1d.n_dofs_per_vertex();

    // FullMatrix<number> matrix(fe_fine_1d.n_dofs_per_cell(),
    //                           fe_coarse_1d.n_dofs_per_cell());

    // FETools::get_projection_matrix(fe_coarse_1d, fe_fine_1d, matrix);

    // this->prolongation_matrix_1d =
    //   Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
    //     Kokkos::view_alloc("prolongation_matrix_1d_" +
    //                          std::to_string(p_coarse) + "_to_" +
    //                          std::to_string(p_fine),
    //                        Kokkos::WithoutInitializing),
    //     fe_coarse_1d.n_dofs_per_cell() * fe_fine_1d.n_dofs_per_cell());

    // auto prolongation_matrix_1d_view =
    //   Kokkos::create_mirror_view(this->prolongation_matrix_1d);

    // for (unsigned int i = 0, k = 0; i < fe_coarse_1d.n_dofs_per_cell();
    // ++i)
    //   for (unsigned int j = 0; j < fe_fine_1d.n_dofs_per_cell(); ++j,
    //   ++k)
    //     prolongation_matrix_1d_view[k] =
    //       matrix(renumbering_fine[j], renumbering_coarse[i]);

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

    // setup_weights_and_boundary_dofs_masks();
  }

  template <int dim, typename number>
  void
  GeometricTransfer<dim, number>::setup_weights_and_boundary_dofs_masks()
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



    // int cell_counter = 0;

    // for (unsigned int color = 0; color < n_colors; ++color)
    //   for (const auto &cell : colored_graph_fine[color])
    //     {
    //       cell->get_dof_indices(local_dof_indices_fine);

    //       for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i)
    //         local_dof_indices_lex_fine[i] =
    //           local_dof_indices_fine[lex_numbering_fine[i]];

    //       constraint_info_fine.read_dof_indices(cell_counter,
    //                                             local_dof_indices_lex_fine,
    //                                             {});
    //       ++cell_counter;
    //     }

    std::vector<types::global_dof_index> local_dof_indices_fine(
      n_dofs_per_cell_fine);


    LinearAlgebra::distributed::Vector<number> weight_vector;
    weight_vector.reinit(this->matrix_free_fine->get_vector_partitioner());

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

    this->boundary_dofs_mask_fine.clear();
    this->boundary_dofs_mask_fine.resize(n_colors);



    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph_fine[color].size() > 0)
          {
            const auto &mf_data_fine = matrix_free_fine->get_data(color);
            ;
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
  }

  // class PolynomialTransferDispatchFactory
  // {
  // public:
  //   static constexpr unsigned int max_degree = 9;

  //   template <typename Runner>
  //   static bool
  //   dispatch(const int runtime_p_coarse, const int runtime_p_fine, Runner
  //   &runner)
  //   {
  //     return recursive_dispatch<Runner, max_degree,
  //     max_degree>(runtime_p_coarse,
  //                                                               runtime_p_fine,
  //                                                               runner);
  //   }

  // private:
  //   template <typename Runner,
  //             unsigned int degree_coarse,
  //             unsigned int degree_fine>
  //   static bool
  //   recursive_dispatch(const int runtime_p_coarse,
  //                      const int runtime_p_fine,
  //                      Runner   &runner)
  //   {
  //     if (runtime_p_fine == degree_fine)
  //       {
  //         if (runtime_p_coarse == degree_coarse)
  //           {
  //             runner.template run<degree_coarse, degree_fine>();
  //             return true;
  //           }
  //         else if constexpr (degree_coarse > 1)
  //           {
  //             return recursive_dispatch<Runner, degree_coarse - 1,
  //             degree_fine>(
  //               runtime_p_coarse, runtime_p_fine, runner);
  //           }
  //         else
  //           {
  //             return false;
  //           }
  //       }
  //     else if constexpr (degree_fine > 1)
  //       {
  //         return recursive_dispatch<Runner, degree_fine - 2, degree_fine -
  //         1>(
  //           runtime_p_coarse, runtime_p_fine, runner);
  //       }

  //     else
  //       {
  //         return false;
  //       }
  //   }
  // };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif