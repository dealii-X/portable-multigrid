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

  namespace h_mg_transfer
  {

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
        prolongation_matrix;

      Kokkos::View<number **, MemorySpace::Default::kokkos_space> weights;

      // TODO: ADAPT FOR OVERLAP_COMM_COMP

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_coarse;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
        dof_indices_fine;
    };



    template <int dim, int fe_degree, typename number>
    class CellProlongationKernel : public EnableObserverPointer
    {
    public:
      using DistributedVectorType =
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

      using TeamHandle = Kokkos::TeamPolicy<
        MemorySpace::Default::kokkos_space::execution_space>::member_type;

      using SharedView = Kokkos::View<number *,
                                      MemorySpace::Default::kokkos_space::
                                        execution_space::scratch_memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      CellProlongationKernel(
        const MGTransferScheme<dim, fe_degree, number> &transfer_scheme,
        unsigned int                                    first_cell,
        const DistributedVectorType                    &src,
        DistributedVectorType                          &dst);


      std::size_t
      team_shmem_size(int team_size) const;

      DEAL_II_HOST_DEVICE void
      operator()(const TeamHandle &team_member) const;

    private:
      const MGTransferScheme<dim, fe_degree, number> &transfer_scheme;

      unsigned int first_cell;

      const DeviceVector<number> src;
      DeviceVector<number>       dst;
    };

    template <int dim, int fe_degree, typename number>
    CellProlongationKernel<dim, fe_degree, number>::CellProlongationKernel(
      const MGTransferScheme<dim, fe_degree, number> &transfer_scheme,
      unsigned int                                    first_cell,
      const DistributedVectorType                    &src,
      DistributedVectorType                          &dst)
      : transfer_scheme(transfer_scheme)
      , first_cell(first_cell)
      , src(src.get_values(), src.locally_owned_size())
      , dst(dst.get_values(), dst.locally_owned_size())
    {}

    template <int dim, int fe_degree, typename number>
    std::size_t
    CellProlongationKernel<dim, fe_degree, number>::team_shmem_size(
      int /*team_size*/) const
    {
      return SharedView::shmem_size(
        5 *
        transfer_scheme.n_dofs_per_cell_fine // +           // coarse dof values
        // n_local_dofs_fine +             // fine dof values
        // 2 * n_local_dofs_fine           // at most two tmp vectors of at most
        // n_local_dofs_fine size
        // + (p_fine + 1) * (p_coarse + 1) // prolongation matrix
      );
    }

    template <int dim, int fe_degree, typename number>
    DEAL_II_HOST_DEVICE void
    CellProlongationKernel<dim, fe_degree, number>::operator()(
      const TeamHandle &team_member) const
    {
      const int cell_index = first_cell + team_member.league_rank();


      SharedView values_coarse(team_member.team_shmem(),
                               transfer_scheme.n_dofs_per_cell_coarse);
      SharedView values_fine(team_member.team_shmem(),
                             transfer_scheme.n_dofs_per_cell_fine);

      // read coarse dof values
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member,
                                transfer_scheme.n_dofs_per_cell_coarse),
        [&](const int &i) {
          const unsigned int dof_index =
            transfer_scheme.dof_indices_coarse(i, cell_index);
          if (dof_index != numbers::invalid_unsigned_int)
            values_coarse(i) = src[dof_index];
          else
            values_coarse(i) = 0;

          std::cout << values_coarse(i) << "   ";
        });

      std::cout << std::endl;
      team_member.team_barrier();


      // interpolation tensor-product prolongation kernel
      // internal::EvaluatorTensorProduct<
      //   internal::EvaluatorVariant::evaluate_general,
      //   dim,
      //   transfer_scheme.degree_coarse + 1,
      //   transfer_scheme.degree_fine + 1,
      //   number>
      //   prolongation_kernel(
      //     team_member,
      //     /*shape_values=*/transfer_scheme.prolongation_matrix,
      //     /*shape_gradients=*/
      //     Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
      //     /*co_shape_gradients=*/
      //     Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
      //     SharedView() // the evaluator does not need temporary
      //                  // storage since no in-place operation takes
      //                  // place in this function
      //   );

      // // apply kernel in each direction
      // if constexpr (dim == 2)
      //   {
      //     auto tmp = SharedView(team_member.team_shmem(),
      //                           (transfer_scheme.degree_coarse + 1) *
      //                             (transfer_scheme.degree_fine + 1));

      //     // <direction, dof_to_quad, add, in_place>
      //     // dof_to_quad == contract_over_rows
      //     prolongation_kernel.template values<0, true, false, false>(
      //       values_coarse, tmp);

      //     prolongation_kernel.template values<1, true, false, false>(
      //       tmp, values_fine);
      //   }
      // else if constexpr (dim == 3)
      //   {
      //     auto tmp1 =
      //       SharedView(team_member.team_shmem(),
      //                  Utilities::pow(transfer_scheme.degree_coarse + 1, 2) *
      //                    (transfer_scheme.degree_fine + 1));

      //     auto tmp2 =
      //       SharedView(team_member.team_shmem(),
      //                  Utilities::pow(transfer_scheme.degree_fine + 1, 2) *
      //                    (transfer_scheme.degree_coarse + 1));

      //     prolongation_kernel.template values<0, true, false, false>(
      //       values_coarse, tmp1);
      //     prolongation_kernel.template values<1, true, false, false>(tmp1,
      //                                                                tmp2);
      //     prolongation_kernel.template values<2, true, false, false>(
      //       tmp2, values_fine);
      //   }

      // SharedView prolongation_matrix(team_member.team_shmem(),
      //                                (p_coarse + 1) * (p_fine + 1));

      // Kokkos::parallel_for(
      //   Kokkos::TeamThreadRange(team_member, (p_coarse + 1) * (p_fine +
      //   1)),
      //   [&](const int &i) {
      //     prolongation_matrix(i) = transfer_data.prolongation_matrix(i);
      //   });
      // team_member.team_barrier();

      // apply kernel in each direction
      if constexpr (dim == 2)
        {
          auto tmp = SharedView(team_member.team_shmem(),
                                (transfer_scheme.degree_coarse + 1) *
                                  (transfer_scheme.degree_fine + 1));

          {
            constexpr int Ni = transfer_scheme.degree_coarse + 1;
            constexpr int Nj = transfer_scheme.degree_fine + 1;
            constexpr int Nk = transfer_scheme.degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = transfer_scheme.degree_fine + 1;

              const int base_coarse   = i * Nk;
              const int stride_coarse = 1;

              number sum = transfer_scheme.prolongation_matrix(base_kernel) *
                           values_coarse(base_coarse);

              for (int k = 1; k < Nk; ++k)
                sum += transfer_scheme.prolongation_matrix(base_kernel +
                                                           k * stride_kernel) *
                       values_coarse(base_coarse + k * stride_coarse);

              const int index_tmp = i * Nj + j;

              tmp(index_tmp) = sum;
            });
          }
          team_member.team_barrier();

          {
            constexpr int Ni = transfer_scheme.degree_fine + 1;
            constexpr int Nj = transfer_scheme.degree_fine + 1;
            constexpr int Nk = transfer_scheme.degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>(
                team_member, Ni, Nj);
            Kokkos::parallel_for(thread_policy, [&](const int i, const int j) {
              const int base_kernel   = j;
              const int stride_kernel = transfer_scheme.degree_fine + 1;

              const int base_tmp   = i;
              const int stride_tmp = transfer_scheme.degree_fine + 1;

              number sum = transfer_scheme.prolongation_matrix(base_kernel) *
                           tmp(base_tmp);

              for (int k = 1; k < Nk; ++k)
                sum += transfer_scheme.prolongation_matrix(base_kernel +
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
          auto tmp1 =
            SharedView(team_member.team_shmem(),
                       Utilities::pow(transfer_scheme.degree_coarse + 1, 2) *
                         (transfer_scheme.degree_fine + 1));
          auto tmp2 =
            SharedView(team_member.team_shmem(),
                       Utilities::pow(transfer_scheme.degree_fine + 1, 2) *
                         (transfer_scheme.degree_coarse + 1));
          {
            constexpr int Ni = transfer_scheme.degree_coarse + 1;
            constexpr int Nj = transfer_scheme.degree_coarse + 1;
            constexpr int Nm = transfer_scheme.degree_fine + 1;
            constexpr int Nk = transfer_scheme.degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = transfer_scheme.degree_fine + 1;

                const int base_coarse   = (i * Nj + j) * Nk;
                const int stride_coarse = 1;

                number sum = transfer_scheme.prolongation_matrix(base_kernel) *
                             values_coarse(base_coarse);

                for (int k = 1; k < Nk; ++k)
                  sum += transfer_scheme.prolongation_matrix(
                           base_kernel + k * stride_kernel) *
                         values_coarse(base_coarse + k * stride_coarse);

                const int index_tmp1 = (i * Nj + j) * Nm + m;
                tmp1(index_tmp1)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = transfer_scheme.degree_fine + 1;
            constexpr int Nj = transfer_scheme.degree_coarse + 1;
            constexpr int Nm = transfer_scheme.degree_fine + 1;
            constexpr int Nk = transfer_scheme.degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = transfer_scheme.degree_fine + 1;

                const int base_tmp1   = i + j * Ni * Nk;
                const int stride_tmp1 = transfer_scheme.degree_fine + 1;

                number sum = transfer_scheme.prolongation_matrix(base_kernel) *
                             tmp1(base_tmp1);

                for (int k = 1; k < Nk; ++k)
                  sum += transfer_scheme.prolongation_matrix(
                           base_kernel + k * stride_kernel) *
                         tmp1(base_tmp1 + k * stride_tmp1);

                const int index_tmp2 = i + (j * Nm + m) * Ni;
                tmp2(index_tmp2)     = sum;
              });
          }

          team_member.team_barrier();

          {
            constexpr int Ni = transfer_scheme.degree_fine + 1;
            constexpr int Nj = transfer_scheme.degree_fine + 1;
            constexpr int Nm = transfer_scheme.degree_fine + 1;
            constexpr int Nk = transfer_scheme.degree_coarse + 1;

            auto thread_policy =
              Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>(
                team_member, Ni, Nj, Nm);
            Kokkos::parallel_for(
              thread_policy, [&](const int i, const int j, const int m) {
                const int base_kernel   = m;
                const int stride_kernel = transfer_scheme.degree_fine + 1;

                const int base_tmp2 = i * Nj + j;
                const int stride_tmp2 =
                  Utilities::pow(transfer_scheme.degree_fine + 1, 2);

                number sum = transfer_scheme.prolongation_matrix(base_kernel) *
                             tmp2(base_tmp2);

                for (int k = 1; k < Nk; ++k)
                  sum += transfer_scheme.prolongation_matrix(
                           base_kernel + k * stride_kernel) *
                         tmp2(base_tmp2 + k * stride_tmp2);

                const int index_fine    = (i + m * Ni) * Nj + j;
                values_fine(index_fine) = sum;
              });
          }
          team_member.team_barrier();
        }

      // apply weights
      Kokkos::parallel_for(Kokkos::TeamThreadRange(
                             team_member, transfer_scheme.n_dofs_per_cell_fine),
                           [&](const int &i) {
                             values_fine(i) *=
                               transfer_scheme.weights(i, cell_index);
                           });
      team_member.team_barrier();


      Kokkos::parallel_for(Kokkos::TeamThreadRange(
                             team_member, transfer_scheme.n_dofs_per_cell_fine),
                           [&](const int &i) {
                             const unsigned int dof_index =
                               transfer_scheme.dof_indices_fine(i, cell_index);
                             if (dof_index != numbers::invalid_unsigned_int)
                               Kokkos::atomic_add(&dst[dof_index],
                                                  values_fine(i));
                             std::cout << values_coarse(i) << "   ";
                           });
      team_member.team_barrier();
      std::cout << std::endl;
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

    void
    test();

  private:
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
  GeometricTransfer<dim, fe_degree, number>::test()
  {
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> src;
    matrix_free_coarse->initialize_dof_vector(src);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> dst;
    matrix_free_fine->initialize_dof_vector(dst);


    for (unsigned int i = 0; i < src.locally_owned_size(); ++i)
      src = 1.0;


    src.update_ghost_values();
    matrix_free_coarse->set_constrained_values(0., src);

    std::cout << "src.l2_norm() = " << src.l2_norm() << std::endl;
    // for (unsigned int i = 0; i < src.locally_owned_size(); ++i)
    //   std::cout << src.local_element(i) << ",  ";
    // std::cout << std::endl;

    prolongate_and_add(dst, src);

    std::cout << "dst.l2_norm() = " << dst.l2_norm() << std::endl;
    // for (unsigned int i = 0; i < dst.locally_owned_size(); ++i)
    //   std::cout << dst.local_element(i) << ",  ";
    // std::cout << std::endl;
  }



  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::prolongate_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage("Fine vector is not initialized correctly."));
    Assert(src.get_partitioner() ==
             matrix_free_coarse->get_vector_partitioner(),
           ExcMessage("Coarse vector is not initialized correctly."));

    MemorySpace::Default::kokkos_space::execution_space exec;

    src.update_ghost_values();

    unsigned int cell_counter = 0;
    unsigned int scheme_index = 0;
    for (const auto &scheme : transfer_schemes)
      {
        if (scheme.n_coarse_cells == 0)
          continue;

        Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
          team_policy(exec, scheme.n_coarse_cells, Kokkos::AUTO);

        h_mg_transfer::CellProlongationKernel<dim, fe_degree, number>
          prolongator(scheme, cell_counter, src, dst);

        Kokkos::parallel_for("prolongate_h_transfer_scheme_" +
                               std::to_string(scheme_index),
                             team_policy,
                             prolongator);


        ++cell_counter;
        ++scheme_index;
      }

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
           ExcMessage(
             "Fine vector is not handled correclty after prolongation."));
    Assert(
      src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Coarse vector is not handled correclty after prolongation."));
  }

  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::restrict_and_add(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    (void)dst;
    (void)src;
    return;
  }



  template <int dim, int fe_degree, typename number>
  void
  GeometricTransfer<dim, fe_degree, number>::reinit(
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

    // helper function: to process the fine level cells; function
    // fu_non_refined is performed on cells that are not refined and
    // fu_refined is performed on children of cells that are refined
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
        // scheme.n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();
        // scheme.n_dofs_per_cell_fine =
        //   Utilities::pow(2 * fe_fine.degree + 1, dim);

        // degree of FE on coarse and fine cell
        // scheme.degree_coarse = fe_coarse.degree;
        // scheme.degree_fine   = fe_coarse.degree * 2;

        // reset number of coarse cells
        scheme.n_coarse_cells = 0;
      }

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


      this->constraint_info_coarse.reinit(dof_handler_coarse,
                                          n_coarse_cells_total,
                                          constraints_coarse.n_constraints() >
                                            0);

      // this->constraint_info_coarse.set_locally_owned_indices(
      //   (mg_level_coarse == numbers::invalid_unsigned_int) ?
      //     dof_handler_coarse.locally_owned_dofs() :
      //     dof_handler_coarse.locally_owned_mg_dofs(mg_level_coarse));

      this->constraint_info_coarse.set_locally_owned_indices(
        dof_handler_coarse.locally_owned_dofs());

      this->constraint_info_fine.reinit(n_coarse_cells_total);

      // this->constraint_info_fine.set_locally_owned_indices(
      //   (mg_level_fine == numbers::invalid_unsigned_int) ?
      //     dof_handler_fine.locally_owned_dofs() :
      //     dof_handler_fine.locally_owned_mg_dofs(mg_level_fine));


      this->constraint_info_fine.set_locally_owned_indices(
        dof_handler_fine.locally_owned_dofs());


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
      this->partitioner_coarse = this->constraint_info_coarse.finalize(
        dof_handler_coarse.get_mpi_communicator());

      this->partitioner_fine = this->constraint_info_fine.finalize(
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

          const unsigned int shift =
            fe.n_dofs_per_cell() - fe.n_dofs_per_vertex();
          const unsigned int n_child_dofs_1d =
            fe.n_dofs_per_cell() * 2 - fe.n_dofs_per_vertex();

          {
            // transfer_schemes[scheme.prolongation_matrix]
            //   .prolongation_matrix.resize(fe.n_dofs_per_cell() *
            //                               n_child_dofs_1d);

            transfer_schemes[transfer_scheme_index].prolongation_matrix =
              Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("prolongation_matrix_h_transfer_scheme_" +
                                     std::to_string(transfer_scheme_index),
                                   Kokkos::WithoutInitializing),
                fe.n_dofs_per_cell() * n_child_dofs_1d);

            auto prolongation_matrix_host = Kokkos::create_mirror_view(
              transfer_schemes[transfer_scheme_index].prolongation_matrix);

            for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell;
                 ++c)
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
                  prolongation_matrix_host[i * n_child_dofs_1d + j +
                                           c * shift] =
                    fe.get_prolongation_matrix(c)(renumbering[j],
                                                  renumbering[i]);

            Kokkos::deep_copy(
              transfer_schemes[transfer_scheme_index].prolongation_matrix,
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
  GeometricTransfer<dim, fe_degree, number>::setup_dof_indices()
  {
    unsigned int scheme_counter = 0;
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
            Kokkos::view_alloc("h_tranasfer_dof_indices_fine_scheme_" +
                                 std::to_string(scheme_counter),
                               Kokkos::WithoutInitializing),
            scheme.n_dofs_per_cell_fine,
            scheme.n_coarse_cells);

        auto dofs_indices_coarse_host =
          Kokkos::create_mirror_view(scheme.dof_indices_coarse);



        auto dofs_indices_fine_host =
          Kokkos::create_mirror_view(scheme.dof_indices_fine);

        for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
          {
            const unsigned int *dof_indices_coarse =
              this->constraint_info_coarse.plain_dof_indices.data() +
              this->constraint_info_coarse.row_starts_plain_indices[cell];

            for (unsigned int j = 0; j < scheme.n_dofs_per_cell_coarse;
                 ++dof_indices_coarse, ++j)
              {
                if (this->constraints_coarse->is_constrained(
                      *dof_indices_coarse))
                  dofs_indices_coarse_host(j, cell) =
                    numbers::invalid_unsigned_int;
                else
                  dofs_indices_coarse_host(j, cell) = *dof_indices_coarse;
              }
            const unsigned int *dof_indices_fine =
              this->constraint_info_fine.dof_indices.data() +
              this->constraint_info_fine.row_starts[cell].first;


            for (unsigned int j = 0; j < scheme.n_dofs_per_cell_fine;
                 ++dof_indices_fine, ++j)
              {
                if (this->constraints_fine->is_constrained(*dof_indices_fine))
                  dofs_indices_fine_host(j, cell) =
                    numbers::invalid_unsigned_int;
                else
                  dofs_indices_fine_host(j, cell) = *dof_indices_fine;
              }
            std::cout << std::endl;
          }
        Kokkos::deep_copy(scheme.dof_indices_coarse, dofs_indices_coarse_host);
        Kokkos::fence();

        Kokkos::deep_copy(scheme.dof_indices_fine, dofs_indices_fine_host);
        Kokkos::fence();

        ++scheme_counter;



        // std::cout << "Coarse dofs:\n";
        // for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
        //   {
        //     std::cout << "  On cell " << cell << ": ";
        //     for (unsigned j = 0; j < scheme.n_dofs_per_cell_coarse; ++j)
        //       std::cout << dofs_indices_coarse_host(j, cell) << " ";
        //     std::cout << std::endl;
        //   }
        // std::cout << std::endl;

        // std::cout << "fine dofs:\n";
        // for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
        //   {
        //     std::cout << "  On cell " << cell << ": ";
        //     for (unsigned j = 0; j < scheme.n_dofs_per_cell_fine; ++j)
        //       std::cout << dofs_indices_fine_host(j, cell) << " ";
        //     std::cout << std::endl;
        //   }
        // std::cout << std::endl;
      }
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

        for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
          {
            const unsigned int *dof_indices_fine =
              this->constraint_info_fine.dof_indices.data() +
              this->constraint_info_fine.row_starts[cell].first;

            for (unsigned int i = 0; i < scheme.n_dofs_per_cell_fine;
                 ++dof_indices_fine, ++i)
              {
                weights_view_host(i, cell) =
                  weight_vector.local_element(*dof_indices_fine);
              }
          }

        Kokkos::deep_copy(scheme.weights, weights_view_host);
        Kokkos::fence();

        // std::cout << "fine  weights:\n";
        // for (unsigned int cell = 0; cell < scheme.n_coarse_cells; ++cell)
        //   {
        //     std::cout << "  On cell " << cell << ": ";
        //     for (unsigned j = 0; j < scheme.n_dofs_per_cell_fine; ++j)
        //       std::cout << weights_view_host(j, cell) << "    ";
        //     std::cout << std::endl;
        //   }
        // std::cout << std::endl;
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