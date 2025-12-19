#ifndef portable_laplace_operator_h
#define portable_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  struct CellData
  {
    using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;

    using ViewValues = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients = Kokkos::View<
      number **,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    TeamHandle team_member;

    const unsigned int n_q_points;
    const int          cell_index;

    const typename MatrixFree<dim, number>::PrecomputedData &precomputed_data;

    const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
      &dirichlet_boundary_dofs_mask;

    /**
     * Memory for dof and quad values.
     */
    ViewValues &values;

    /**
     * Memory for computed gradients in reference coordinate system.
     */
    ViewGradients &gradients;

    /**
     * Memory for temporary arrays required by evaluation and integration.
     */
    ViewValues &scratch_pad;
  };

  // template <int dim, int fe_degree, typename number>
  // class LaplaceOperatorQuad
  // {
  // public:
  //   DEAL_II_HOST_DEVICE void
  //   operator()(
  //     Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>
  //     *fe_eval, const int q_point) const;

  //   static const unsigned int n_q_points =
  //     dealii::Utilities::pow(fe_degree + 1, dim);
  // };

  // template <int dim, int fe_degree, typename number>
  // DEAL_II_HOST_DEVICE void
  // LaplaceOperatorQuad<dim, fe_degree, number>::operator()(
  //   Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>
  //   *fe_eval, const int q_point) const
  // {
  //   fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
  // }

  template <int dim, int fe_degree, typename number>
  class LaplaceDiagonalOperator
  {
  public:
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const CellData<dim, number> *data,
               DeviceVector<number>        &diagonal) const;
  };

  template <int dim, int fe_degree, typename number>
  DEAL_II_HOST_DEVICE void
  LaplaceDiagonalOperator<dim, fe_degree, number>::operator()(
    const CellData<dim, number> *data,
    DeviceVector<number>        &diagonal) const
  {
    const auto &precomputed_data = data->precomputed_data;
    const int   cell_id          = data->cell_index;
    const auto &team_member      = data->team_member;

    auto &values      = data->values;
    auto &gradients   = data->gradients;
    auto &scratch_pad = data->scratch_pad;


    // define scratch pad for the evaluation
    constexpr int scratch_size = Utilities::pow(fe_degree + 1, dim);
    auto          scratch_for_eval =
      Kokkos::subview(scratch_pad, Kokkos::make_pair(0, scratch_size));

    // initialize tensor-product kernel
    internal::EvaluatorTensorProduct<
      internal::EvaluatorVariant::evaluate_general,
      dim,
      fe_degree + 1,
      fe_degree + 1,
      number>
      eval(team_member,
           precomputed_data.shape_values,
           precomputed_data.shape_gradients,
           precomputed_data.co_shape_gradients,
           scratch_for_eval);

    for (unsigned int i = 0; i < n_local_dofs; ++i)
      {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_local_dofs),
                             [&](const unsigned int &j) {
                               values(j) = (i == j) ? 1.0 : 0.0;
                             });
        team_member.team_barrier();

        // evaluate the kernel using sum factorization

        // 1.transform to the collocation space
        eval.template values<0, true, false, true>(values, values);
        if constexpr (dim > 1)
          eval.template values<1, true, false, true>(values, values);
        if constexpr (dim > 2)
          eval.template values<2, true, false, true>(values, values);

        // 2. evaluate gradients in the colloction space
        eval.template co_gradients<0, true, false, false>(
          values, Kokkos::subview(gradients, Kokkos::ALL, 0));
        if constexpr (dim > 1)
          eval.template co_gradients<1, true, false, false>(
            values, Kokkos::subview(gradients, Kokkos::ALL, 1));
        if constexpr (dim > 2)
          eval.template co_gradients<2, true, false, false>(
            values, Kokkos::subview(gradients, Kokkos::ALL, 2));

        team_member.team_barrier();

        // 3.compute Laplace kernel at each quadrature point
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_q_points),
          [&](const int &q_point) {
            // 3a. get gradient
            Tensor<1, dim, number> grad;
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp +=
                    precomputed_data.inv_jacobian(q_point, cell_id, d_2, d_1) *
                    gradients(q_point, d_2);
                grad[d_1] = tmp;
              }

            // 3b.submit gradient
            for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
              {
                number tmp = 0.;
                for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
                  tmp +=
                    precomputed_data.inv_jacobian(q_point, cell_id, d_1, d_2) *
                    grad[d_2];
                gradients(q_point, d_1) =
                  tmp * precomputed_data.JxW(q_point, cell_id);
              }
          });

        team_member.team_barrier();

        // integrate using time factorization

        // 4. apply derivatives in collocation space
        if constexpr (dim == 1)
          eval.template co_gradients<0, false, false, false>(
            Kokkos::subview(gradients, Kokkos::ALL, 2), values);
        else if constexpr (dim == 2)
          {
            eval.template co_gradients<1, false, false, false>(
              Kokkos::subview(gradients, Kokkos::ALL, 1), values);
            eval.template co_gradients<0, false, true, false>(
              Kokkos::subview(gradients, Kokkos::ALL, 0), values);
          }
        else if constexpr (dim == 3)
          {
            eval.template co_gradients<2, false, false, false>(
              Kokkos::subview(gradients, Kokkos::ALL, 2), values);
            eval.template co_gradients<1, false, true, false>(
              Kokkos::subview(gradients, Kokkos::ALL, 1), values);
            eval.template co_gradients<0, false, true, false>(
              Kokkos::subview(gradients, Kokkos::ALL, 0), values);
          }

        // 5. transform back to the original space
        if constexpr (dim > 2)
          eval.template values<2, false, false, true>(values, values);
        if constexpr (dim > 1)
          eval.template values<1, false, false, true>(values, values);
        eval.template values<0, false, false, true>(values, values);

        team_member.team_barrier();

        // distribute diagonal dof
        {
          if (team_member.team_rank() == 0)
            {
              if (precomputed_data.use_coloring)
                diagonal[precomputed_data.local_to_global(i, cell_id)] +=
                  values(i);
              else
                Kokkos::atomic_add(
                  &diagonal[precomputed_data.local_to_global(i, cell_id)],
                  values(i));
            }
        }
      }
  }

  template <int dim, int fe_degree, typename number>
  class LocalLaplaceOperator
  {
  public:
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const CellData<dim, number> *data,
               const DeviceVector<number>  &src,
               DeviceVector<number>        &dst) const;
  };

  template <int dim, int fe_degree, typename number>
  DEAL_II_HOST_DEVICE void
  LocalLaplaceOperator<dim, fe_degree, number>::operator()(
    const CellData<dim, number> *data,
    const DeviceVector<number>  &src,
    DeviceVector<number>        &dst) const
  {
    const auto &precomputed_data = data->precomputed_data;
    const int   cell_id          = data->cell_index;
    const auto &team_member      = data->team_member;

    const auto &dirichlet_boundary_dofs_mask =
      data->dirichlet_boundary_dofs_mask;

    auto &values      = data->values;
    auto &gradients   = data->gradients;
    auto &scratch_pad = data->scratch_pad;

    // read dof values
    {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(data->team_member, n_local_dofs),
        [&](const int &i) {
          if (dirichlet_boundary_dofs_mask(i, cell_id) ==
              numbers::invalid_unsigned_int)
            values(i) = 0.;
          else
            values(i) = src[precomputed_data.local_to_global(i, cell_id)];
        });

      data->team_member.team_barrier();
    }


    // define scratch pad for the evaluation
    constexpr int scratch_size = Utilities::pow(fe_degree + 1, dim);
    auto          scratch_for_eval =
      Kokkos::subview(scratch_pad, Kokkos::make_pair(0, scratch_size));

    // initialize tensor-product kernel
    internal::EvaluatorTensorProduct<
      internal::EvaluatorVariant::evaluate_general,
      dim,
      fe_degree + 1,
      fe_degree + 1,
      number>
      eval(team_member,
           precomputed_data.shape_values,
           precomputed_data.shape_gradients,
           precomputed_data.co_shape_gradients,
           scratch_for_eval);

    // evaluate the kernel using sum factorization

    // 1.transform to the collocation space
    eval.template values<0, true, false, true>(values, values);
    if constexpr (dim > 1)
      eval.template values<1, true, false, true>(values, values);
    if constexpr (dim > 2)
      eval.template values<2, true, false, true>(values, values);

    // 2. evaluate gradients in the colloction space
    eval.template co_gradients<0, true, false, false>(
      values, Kokkos::subview(gradients, Kokkos::ALL, 0));
    if constexpr (dim > 1)
      eval.template co_gradients<1, true, false, false>(
        values, Kokkos::subview(gradients, Kokkos::ALL, 1));
    if constexpr (dim > 2)
      eval.template co_gradients<2, true, false, false>(
        values, Kokkos::subview(gradients, Kokkos::ALL, 2));

    team_member.team_barrier();

    // 3.compute Laplace kernel at each quadrature point
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, n_q_points),
      [&](const int &q_point) {
        // 3a. get gradient
        Tensor<1, dim, number> grad;
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += precomputed_data.inv_jacobian(q_point, cell_id, d_2, d_1) *
                     gradients(q_point, d_2);
            grad[d_1] = tmp;
          }

        // 3b.submit gradient
        for (unsigned int d_1 = 0; d_1 < dim; ++d_1)
          {
            number tmp = 0.;
            for (unsigned int d_2 = 0; d_2 < dim; ++d_2)
              tmp += precomputed_data.inv_jacobian(q_point, cell_id, d_1, d_2) *
                     grad[d_2];
            gradients(q_point, d_1) =
              tmp * precomputed_data.JxW(q_point, cell_id);
          }
      });

    team_member.team_barrier();

    // integrate using time factorization

    // 4. apply derivatives in collocation space
    if constexpr (dim == 1)
      eval.template co_gradients<0, false, false, false>(
        Kokkos::subview(gradients, Kokkos::ALL, 2), values);
    else if constexpr (dim == 2)
      {
        eval.template co_gradients<1, false, false, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 1), values);
        eval.template co_gradients<0, false, true, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 0), values);
      }
    else if constexpr (dim == 3)
      {
        eval.template co_gradients<2, false, false, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 2), values);
        eval.template co_gradients<1, false, true, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 1), values);
        eval.template co_gradients<0, false, true, false>(
          Kokkos::subview(gradients, Kokkos::ALL, 0), values);
      }

    // 5. transform back to the original space
    if constexpr (dim > 2)
      eval.template values<2, false, false, true>(values, values);
    if constexpr (dim > 1)
      eval.template values<1, false, false, true>(values, values);
    eval.template values<0, false, false, true>(values, values);

    team_member.team_barrier();

    // distribute dofs
    {
      if (precomputed_data.use_coloring)
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs),
          [&](const int &i) {
            if (dirichlet_boundary_dofs_mask(i, cell_id) !=
                numbers::invalid_unsigned_int)
              dst[precomputed_data.local_to_global(i, cell_id)] += values(i);
          });
      else
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, n_local_dofs),
          [&](const int &i) {
            if (dirichlet_boundary_dofs_mask(i, cell_id) !=
                numbers::invalid_unsigned_int)
              Kokkos::atomic_add(
                &dst[precomputed_data.local_to_global(i, cell_id)], values(i));
          });
    }
  }

  template <int dim, int fe_degree, typename number>
  class LaplaceOperator : public LaplaceOperatorBase<dim, number>
  {
  public:
    LaplaceOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<number> &constraints,
                    bool overlap_communication_computation);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const override;

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const override;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
      const override;

    void
    compute_diagonal() override;

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    number
    el(const types::global_dof_index row,
       const types::global_dof_index col) const override;

    const MatrixFree<dim, number> &
    get_matrix_free() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

  private:
    using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients = Kokkos::View<
      number **,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);

    MatrixFree<dim, number>                           matrix_free;
    typename MatrixFree<dim, number>::PrecomputedData gpu_data;

    static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dirichlet_boundary_dofs_mask_fine;
  };

  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim, fe_degree, number>::LaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
  {
    const MappingQ<dim>                              mapping(fe_degree);
    typename MatrixFree<dim, number>::AdditionalData additional_data;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation =
      overlap_communication_computation;

    const QGauss<1> quadrature_1d(fe_degree + 1);
    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature_1d, additional_data);

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();


    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(
        std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }

    dirichlet_boundary_dofs_mask_fine.clear();
    dirichlet_boundary_dofs_mask_fine.resize(n_colors);


    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto &mf_data = matrix_free.get_data(0, color);
            ;
            const auto &graph = colored_graph[color];

            this->dirichlet_boundary_dofs_mask_fine[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("dirichlet_boundary_dofs_" +
                                     std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_local_dofs,
                mf_data.n_cells);

            auto dofs_mask_host = Kokkos::create_mirror_view(
              this->dirichlet_boundary_dofs_mask_fine[color]);

            auto cell = graph.cbegin(), end_cell = graph.cend();

            std::vector<types::global_dof_index> local_dof_indices(
              n_local_dofs);
            for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id)
              {
                (*cell)->get_dof_indices(local_dof_indices);

                for (unsigned int i = 0; i < n_local_dofs; ++i)
                  {
                    const auto global_dof = local_dof_indices[lex_numbering[i]];
                    if (constraints.is_constrained(global_dof))
                      dofs_mask_host(i, cell_id) =
                        numbers::invalid_unsigned_int;
                    else
                      dofs_mask_host(i, cell_id) = global_dof;
                  }
              }
            Kokkos::deep_copy(this->dirichlet_boundary_dofs_mask_fine[color],
                              dofs_mask_host);
            Kokkos::fence();
          }
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    dst = 0.;
    LocalLaplaceOperator<dim, fe_degree, number> local_operator;

    // copy vectors to the kokkos views
    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size());
    DeviceVector<number> dst_device(dst.get_values(), dst.locally_owned_size());

    MemorySpace::Default::kokkos_space::execution_space exec;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        auto do_color = [&](const unsigned int color) {
          const auto &gpu_data = matrix_free.get_data(0, color);

          const auto n_cells = gpu_data.n_cells;

          Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>
            team_policy(exec, n_cells, Kokkos::AUTO);

          // ssize: shape values (n_components x n_q_points)
          // + shape_gtradients (n_components x dim x n_q_points)
          // +  scratch_pad
          std::size_t shmem_size =
            ViewValues::shmem_size(n_q_points) +
            ViewGradients::shmem_size(n_q_points, dim) +
            ViewValues::shmem_size(gpu_data.scratch_pad_size);

          team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

          Kokkos::parallel_for(
            "laplace_operator_cell_loop_" + std::to_string(color),
            team_policy,
            KOKKOS_LAMBDA(TeamHandle team_member) {
              // get the cell index from the block id
              const int cell_index = team_member.league_rank();

              // Allocate the scratch memory
              ViewValues    values(team_member.team_shmem(), n_q_points);
              ViewGradients gradients(team_member.team_shmem(),
                                      n_q_points,
                                      dim);
              ViewValues    scratch_pad(team_member.team_shmem(),
                                     gpu_data.scratch_pad_size);

              // prepare CellData for the local operator evaluation
              CellData<dim, number> cell_data{
                team_member,
                n_q_points,
                cell_index,
                gpu_data,
                dirichlet_boundary_dofs_mask_fine[color],
                values,
                gradients,
                scratch_pad};

              // evaluate local quad operator on the cell
              DeviceVector<number> nonconst_dst = dst_device;
              local_operator(&cell_data, src_device, nonconst_dst);
            });
        };

        src.update_ghost_values_start(0);

        if (n_colors > 0 && colored_graph[0].size() > 0)
          do_color(0);

        src.update_ghost_values_finish();

        if (n_colors > 1 && colored_graph[1].size() > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        dst.compress_start(0, VectorOperation::add);

        if (n_colors > 2 && colored_graph[2].size() > 0)
          do_color(2);

        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &gpu_data = matrix_free.get_data(0, color);
            const auto  n_cells  = gpu_data.n_cells;

            if (n_cells > 0)
              {
                Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>
                  team_policy(exec, n_cells, Kokkos::AUTO);

                // ssize: shape values (n_components x n_q_points)
                // + shape_gtradients (n_components x dim x n_q_points)
                // +  scratch_pad
                std::size_t shmem_size =
                  ViewValues::shmem_size(n_q_points) +
                  ViewGradients::shmem_size(n_q_points, dim) +
                  ViewValues::shmem_size(gpu_data.scratch_pad_size);
                team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

                Kokkos::parallel_for(
                  "laplace_operator_cell_loop_" + std::to_string(color),
                  team_policy,
                  KOKKOS_LAMBDA(TeamHandle team_member) {
                    const int cell_index = team_member.league_rank();

                    // Allocate the scratch memory
                    ViewValues    values(team_member.team_shmem(), n_q_points);
                    ViewGradients gradients(team_member.team_shmem(),
                                            n_q_points,
                                            dim);
                    ViewValues    scratch_pad(team_member.team_shmem(),
                                           gpu_data.scratch_pad_size);

                    CellData<dim, number> cell_data{
                      team_member,
                      n_q_points,
                      cell_index,
                      gpu_data,
                      dirichlet_boundary_dofs_mask_fine[color],
                      values,
                      gradients,
                      scratch_pad};

                    // evaluate quad operator on the cell
                    DeviceVector<number> nonconst_dst = dst_device;
                    local_operator(&cell_data, src_device, nonconst_dst);
                  });
              }
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();

    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::Tvmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, typename number>
  const MatrixFree<dim, number> &
  LaplaceOperator<dim, fe_degree, number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>());

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    // LaplaceOperatorQuad<dim, fe_degree, number> operator_quad;

    // MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1,
    // number>(
    //   matrix_free,
    //   inverse_diagonal,
    //   operator_quad,
    //   EvaluationFlags::gradients,
    //   EvaluationFlags::gradients);

    LaplaceDiagonalOperator<dim, fe_degree, number> diagonal_operator;

    MemorySpace::Default::kokkos_space::execution_space exec;

    DeviceVector<number> inverse_diagonal_device(
      inverse_diagonal.get_values(), inverse_diagonal.locally_owned_size());


    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        auto do_color = [&](const unsigned int color) {
          const auto &gpu_data = matrix_free.get_data(0, color);

          const auto n_cells = gpu_data.n_cells;

          Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>
            team_policy(exec, n_cells, Kokkos::AUTO);

          // ssize: shape values (n_components x n_q_points)
          // + shape_gtradients (n_components x dim x n_q_points)
          // +  scratch_pad
          std::size_t shmem_size =
            ViewValues::shmem_size(n_q_points) +
            ViewGradients::shmem_size(n_q_points, dim) +
            ViewValues::shmem_size(gpu_data.scratch_pad_size);

          team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

          Kokkos::parallel_for(
            "compute_diagonal_cell_loop_" + std::to_string(color),
            team_policy,
            KOKKOS_LAMBDA(TeamHandle team_member) {
              // get the cell index from the block id
              const int cell_index = team_member.league_rank();

              // Allocate the scratch memory
              ViewValues    values(team_member.team_shmem(), n_q_points);
              ViewGradients gradients(team_member.team_shmem(),
                                      n_q_points,
                                      dim);
              ViewValues    scratch_pad(team_member.team_shmem(),
                                     gpu_data.scratch_pad_size);

              // prepare CellData for the local operator evaluation
              CellData<dim, number> cell_data{
                team_member,
                n_q_points,
                cell_index,
                gpu_data,
                dirichlet_boundary_dofs_mask_fine[color],
                values,
                gradients,
                scratch_pad};

              // evaluate local quad operator on the cell
              DeviceVector<number> nonconst_inverse_diagonal_device =
                inverse_diagonal_device;
              diagonal_operator(&cell_data, nonconst_inverse_diagonal_device);
            });
        };

        if (n_colors > 0 && colored_graph[0].size() > 0)
          do_color(0);

        if (n_colors > 1 && colored_graph[1].size() > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        inverse_diagonal.compress_start(0, VectorOperation::add);

        if (n_colors > 2 && colored_graph[2].size() > 0)
          do_color(2);

        inverse_diagonal.compress_finish(VectorOperation::add);
      }
    else
      {
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &gpu_data = matrix_free.get_data(0, color);
            const auto  n_cells  = gpu_data.n_cells;

            if (n_cells > 0)
              {
                Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>
                  team_policy(exec, n_cells, Kokkos::AUTO);

                // ssize: shape values (n_components x n_q_points)
                // + shape_gtradients (n_components x dim x n_q_points)
                // +  scratch_pad
                std::size_t shmem_size =
                  ViewValues::shmem_size(n_q_points) +
                  ViewGradients::shmem_size(n_q_points, dim) +
                  ViewValues::shmem_size(gpu_data.scratch_pad_size);
                team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

                Kokkos::parallel_for(
                  "compute_diagonal_cell_loop_" + std::to_string(color),
                  team_policy,
                  KOKKOS_LAMBDA(TeamHandle team_member) {
                    const int cell_index = team_member.league_rank();

                    // Allocate the scratch memory
                    ViewValues    values(team_member.team_shmem(), n_q_points);
                    ViewGradients gradients(team_member.team_shmem(),
                                            n_q_points,
                                            dim);
                    ViewValues    scratch_pad(team_member.team_shmem(),
                                           gpu_data.scratch_pad_size);

                    CellData<dim, number> cell_data{
                      team_member,
                      n_q_points,
                      cell_index,
                      gpu_data,
                      dirichlet_boundary_dofs_mask_fine[color],
                      values,
                      gradients,
                      scratch_pad};

                    // evaluate quad operator on the cell
                    DeviceVector<number> nonconst_inverse_diagonal_device =
                      inverse_diagonal_device;
                    diagonal_operator(&cell_data,
                                      nonconst_inverse_diagonal_device);
                  });
              }
          }
        inverse_diagonal.compress(VectorOperation::add);
      }

    matrix_free.set_constrained_values(1.0, inverse_diagonal);

    number *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

  template <int dim, int fe_degree, typename number>
  std::shared_ptr<DiagonalMatrix<
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
  LaplaceOperator<dim, fe_degree, number>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, number>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, number>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number>
  number
  LaplaceOperator<dim, fe_degree, number>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr &&
             inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  template <int dim, int fe_degree, typename number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperator<dim, fe_degree, number>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
