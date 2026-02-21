#ifndef portable_subdomain_laplace_operator_h
#define portable_subdomain_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename number>
  class SubdomainLaplaceOperator : public LaplaceOperatorBase<dim, number>
  {
  public:
    SubdomainLaplaceOperator(
      const SubdomainDoFHandler<dim>  &dof_handler,
      const AffineConstraints<number> &constraints,
      const AffineConstraints<number> &constraints_physical,
      bool overlap_communication_computation = false);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const override;

    void
    vmult_dummy(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                &src,
      const bool ghost_exchange_on,
      const bool computation_on) const override;

    void
    dirichlet_solve_local(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    void
    vmult_schur(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

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

    void
    setup_dof_indices_per_color();

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

    void
    vmult_range(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const;

    void
    cell_loop(
      const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                       &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
      const;

    void
    cell_range_loop(
      const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                       &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
      const;

    void
    cell_loop_dummy(
      const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                                                                       &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const bool ghost_exchange_on,
      const bool computation_on) const;

    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);

    MatrixFree<dim, number> matrix_free;

    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;

    ObserverPointer<const AffineConstraints<number>> constraints;
    ObserverPointer<const AffineConstraints<number>> constraints_physical;

    static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      interior_dof_indices_per_color;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      interface_cell_dof_indices_per_color;


    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      interior_interface_cell_dof_indices;

    std::vector<
      Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>>
      interface_cell_ids_per_color;

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      temp_vector_src, temp_vector_dst, temp_vector_work;
  };

  template <int dim, int fe_degree, typename number>
  SubdomainLaplaceOperator<dim, fe_degree, number>::SubdomainLaplaceOperator(
    const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
    const AffineConstraints<number> &constraints,
    const AffineConstraints<number> &constraints_physical,
    bool                             overlap_communication_computation)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;

    this->constraints           = &constraints;
    this->constraints_physical  = &constraints_physical;
    this->subdomain_dof_handler = &subdomain_dof_handler;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation =
      overlap_communication_computation;

    const QGauss<1> quadrature_1d(fe_degree + 1);
    matrix_free.reinit(mapping,
                       subdomain_dof_handler.get_dof_handler(),
                       constraints,
                       quadrature_1d,
                       additional_data);

    matrix_free.initialize_dof_vector(temp_vector_src);
    matrix_free.initialize_dof_vector(temp_vector_dst);
    matrix_free.initialize_dof_vector(temp_vector_work);

    setup_dof_indices_per_color();
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::dirichlet_solve_local(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    dst = 0.;
    SolverControl solver_control(src.size(), 1e-12 * src.l2_norm());

    SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
      cg(solver_control);

    cg.solve(*this, dst, src, PreconditionIdentity());
  }


  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::vmult_schur(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    Assert(
      dst.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
         interface vector partitioner."));
    Assert(
      src.get_partitioner() ==
        this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's \
        interface vector partitioner."));

    MemorySpace::Default::kokkos_space::execution_space exec;

    dst = 0.;

    temp_vector_src  = 0.;
    temp_vector_work = 0.;

    src.update_ghost_values();

    DeviceVector<number> src_view(src.get_values(), src.locally_owned_size());
    DeviceVector<number> dst_view(dst.get_values(), src.locally_owned_size());
    DeviceVector<number> t_src(temp_vector_src.get_values(),
                               temp_vector_src.locally_owned_size());
    DeviceVector<number> t_dst(temp_vector_dst.get_values(),
                               temp_vector_dst.locally_owned_size());
    DeviceVector<number> t_work(temp_vector_work.get_values(),
                                temp_vector_work.locally_owned_size());


    Kokkos::parallel_for(
      "read_src_interface",
      interface_dof_indices.size(),
      KOKKOS_LAMBDA(const int i) {
        t_src[interface_dof_indices[i]] = src_view[i];
      });

    vmult_range(temp_vector_dst, temp_vector_src);

    Kokkos::parallel_for(
      "work",
      interior_interface_cell_dof_indices.size(),
      KOKKOS_LAMBDA(const int i) {
        const auto idx = interior_interface_cell_dof_indices(i);
        t_work[idx]    = t_dst[idx];
      });

    dirichlet_solve_local(temp_vector_src, temp_vector_work);

    vmult_range(temp_vector_work, temp_vector_src);

    Kokkos::parallel_for(
      "distribute_interface_dofs",
      interface_dof_indices.size(),
      KOKKOS_LAMBDA(const int i) {
        dst_view[i] =
          t_dst[interface_dof_indices[i]] - t_work[interface_dof_indices[i]];
      });

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    dst = 0.;

    LocalLaplaceOperator<dim, fe_degree, number> cell_operator;

    this->cell_loop(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::vmult_range(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    dst = 0.;

    LocalLaplaceOperator<dim, fe_degree, number> cell_operator;

    this->cell_loop_range(cell_operator, src, dst);
  }



  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::cell_loop(
    const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one colorportable_laplace_operator_quad
        auto do_color = [&](const unsigned int color) {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;


          const auto &gpu_data = matrix_free.get_data(color, 0);

          auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

          internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
            cell_operator,
            gpu_data,
            this->interior_dof_indices_per_color[color],
            src,
            dst);

          Kokkos::parallel_for(
            "dealii::MatrixFree::distributed_cell_loop color " +
              std::to_string(color),
            team_policy,
            apply_kernel);
        };

        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && matrix_free.get_data(0, 0).n_cells > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && matrix_free.get_data(1, 0).n_cells > 0)
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
        if (colored_graph.size() > 2 && matrix_free.get_data(2, 0).n_cells > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &gpu_data = matrix_free.get_data(color, 0);
            if (gpu_data.n_cells > 0)
              {
                using TeamPolicy = Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>;

                auto team_policy =
                  TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

                internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
                  cell_operator,
                  gpu_data,
                  this->interior_dof_indices_per_color[color],
                  src,
                  dst);

                Kokkos::parallel_for(
                  "dealii::MatrixFree::distributed_cell_loop color " +
                    std::to_string(color),
                  team_policy,
                  apply_kernel);
              }
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::cell_range_loop(
    const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one colorportable_laplace_operator_quad
        auto do_color = [&](const unsigned int color) {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;


          const auto &gpu_data = matrix_free.get_data(color, 0);

          auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

          internal::ApplyCellKernelRange<dim, number, Functor> apply_kernel(
            cell_operator,
            gpu_data,
            this->interface_cell_dof_indices_per_color[color],
            this->interface_cell_ids_per_color[color],
            src,
            dst);

          Kokkos::parallel_for(
            "dealii::MatrixFree::distributed_cell_loop color " +
              std::to_string(color),
            team_policy,
            apply_kernel);
        };

        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && matrix_free.get_data(0, 0).n_cells > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && matrix_free.get_data(1, 0).n_cells > 0)
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
        if (colored_graph.size() > 2 && matrix_free.get_data(2, 0).n_cells > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const auto &gpu_data = matrix_free.get_data(color, 0);
            if (gpu_data.n_cells > 0)
              {
                using TeamPolicy = Kokkos::TeamPolicy<
                  MemorySpace::Default::kokkos_space::execution_space>;

                auto team_policy =
                  TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

                internal::ApplyCellKernelRange<dim, number, Functor>
                  apply_kernel(
                    cell_operator,
                    gpu_data,
                    this->interface_cell_dof_indices_per_color[color],
                    this->interface_cell_ids_per_color[color],
                    src,
                    dst);
                Kokkos::parallel_for(
                  "dealii::MatrixFree::distributed_cell_loop color " +
                    std::to_string(color),
                  team_policy,
                  apply_kernel);
              }
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool ghost_exchange_on,
    const bool computation_on) const
  {
    dst = 0.;

    LocalLaplaceOperator<dim, fe_degree, number> cell_operator;

    this->cell_loop_dummy(
      cell_operator, src, dst, ghost_exchange_on, computation_on);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::cell_loop_dummy(
    const LocalLaplaceOperator<dim, fe_degree, number> &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const bool ghost_exchange_on,
    const bool computation_on) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one color
        auto do_color = [&](const unsigned int color) {
          using TeamPolicy = Kokkos::TeamPolicy<
            MemorySpace::Default::kokkos_space::execution_space>;


          const auto &gpu_data = matrix_free.get_data(color, 0);

          auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

          internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
            cell_operator,
            gpu_data,
            this->interior_dof_indices_per_color[color],
            src,
            dst);

          Kokkos::parallel_for(
            "dealii::MatrixFree::distributed_cell_loop color " +
              std::to_string(color),
            team_policy,
            apply_kernel);
        };

        if (ghost_exchange_on)
          src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && matrix_free.get_data(0, 0).n_cells > 0)
          if (computation_on)
            do_color(0);

        if (ghost_exchange_on)
          src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && matrix_free.get_data(1, 0).n_cells > 0)
          {
            if (computation_on)
              do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }
        if (ghost_exchange_on)
          dst.compress_start(0, VectorOperation::add);

        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 && matrix_free.get_data(2, 0).n_cells > 0)
          if (computation_on)

            do_color(2);

        if (ghost_exchange_on)
          dst.compress_finish(VectorOperation::add);
      }
    else
      {
        if (ghost_exchange_on)
          src.update_ghost_values();

        // Execute the loop on the cells
        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (computation_on)
              {
                const auto &gpu_data = matrix_free.get_data(color, 0);
                if (gpu_data.n_cells > 0)
                  {
                    using TeamPolicy = Kokkos::TeamPolicy<
                      MemorySpace::Default::kokkos_space::execution_space>;

                    auto team_policy =
                      TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);


                    internal::ApplyCellKernel<dim, number, Functor>
                      apply_kernel(cell_operator,
                                   gpu_data,
                                   this->interior_dof_indices_per_color[color],
                                   src,
                                   dst);

                    Kokkos::parallel_for(
                      "dealii::MatrixFree::distributed_cell_loop color " +
                        std::to_string(color),
                      team_policy,
                      apply_kernel);
                  }
              }
          }
        if (ghost_exchange_on)
          dst.compress(VectorOperation::add);
      }

    if (ghost_exchange_on)
      src.zero_out_ghost_values();
  }



  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::
    setup_dof_indices_per_color()
  {
    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();
    const auto        &partitioner   = matrix_free.get_vector_partitioner();



    const auto &dof_handler = matrix_free.get_dof_handler();

    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(
        std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }
    {
      this->interior_dof_indices_per_color.clear();
      this->interior_dof_indices_per_color.resize(n_colors);

      std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
      std::vector<types::global_dof_index> subdomain_local_dof_indices(
        n_local_dofs);


      for (unsigned int color = 0; color < n_colors; ++color)
        {
          if (colored_graph[color].size() > 0)
            {
              const auto &mf_data = matrix_free.get_data(color);

              const auto &graph = colored_graph[color];

              this->interior_dof_indices_per_color[color] =
                Kokkos::View<unsigned int **,
                             MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("interior_dof_indices_" +
                                       std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  n_local_dofs,
                  mf_data.n_cells);

              auto dof_indices_host = Kokkos::create_mirror_view(
                this->interior_dof_indices_per_color[color]);


              for (unsigned int cell_id = 0; cell_id < mf_data.n_cells;
                   ++cell_id)
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
                      const auto global_dof =
                        local_dof_indices[lex_numbering[i]];
                      const auto subdomain_local_dof =
                        subdomain_local_dof_indices[lex_numbering[i]];

                      if (constraints->is_constrained(subdomain_local_dof))
                        dof_indices_host(i, cell_id) =
                          numbers::invalid_unsigned_int;
                      else
                        dof_indices_host(i, cell_id) = global_dof;
                    }
                }

              Kokkos::deep_copy(exec_space,
                                this->interior_dof_indices_per_color[color],
                                dof_indices_host);
              Kokkos::fence();
            }
        }
    }
    {
      const auto &interface_dofs =
        this->subdomain_dof_handler->get_dof_info().subdomain_interface_dofs;

      this->interface_dof_indices =
        Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("interface_dof_indices",
                             Kokkos::WithoutInitializing),
          interface_dofs.size());

      auto interface_dof_indices_host =
        Kokkos::create_mirror_view(this->interface_dof_indices);

      for (unsigned int i = 0; i < interface_dofs.size(); ++i)
        interface_dof_indices_host(i) = interface_dofs[i];

      Kokkos::deep_copy(exec_space,
                        this->interface_dof_indices,
                        interface_dof_indices_host);
    }

    {
      this->interface_cell_ids_per_color.clear();
      this->interface_cell_ids_per_color.resize(n_colors);
      this->interface_cell_dof_indices_per_color.clear();
      this->interface_cell_dof_indices_per_color.resize(n_colors);

      const auto &interior_dofs_set =
        this->subdomain_dof_handler->get_dof_info().subdomain_interior_dofs;

      std::vector<unsigned int> interior_dofs_interface;

      std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          if (colored_graph[color].size() > 0)
            {
              const auto &graph   = colored_graph[color];
              const auto  n_cells = graph.size();
              const auto &mf_data = matrix_free.get_data(color);

              std::vector<unsigned int> interface_cell_ids;
              std::vector<unsigned int> interface_cell_dof_indices;

              for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
                {
                  typename DoFHandler<dim>::cell_iterator cell =
                    graph[cell_id]->as_dof_handler_iterator(dof_handler);


                  cell->get_dof_indices(local_dof_indices);
                  if (partitioner)
                    for (auto &index : local_dof_indices)
                      index = partitioner->global_to_local(index);

                  if (cell->at_boundary())
                    for (unsigned int f = 0;
                         f < GeometryInfo<dim>::faces_per_cell;
                         ++f)
                      if (cell->at_boundary(f) &&
                          cell->face(f)->boundary_id() ==
                            subdomain_dof_handler->get_interface_id())
                        {
                          interface_cell_ids.push_back(cell_id);

                          for (unsigned int i = 0; i < n_local_dofs; ++i)
                            {
                              const unsigned int global_dof =
                                local_dof_indices[lex_numbering[i]];
                              interface_cell_dof_indices.push_back(global_dof);

                              if (interior_dofs_set.is_element(global_dof))
                                interior_dofs_interface.push_back(global_dof);
                            }

                          break;
                        }
                }

              this->interface_cell_ids_per_color[color] =
                Kokkos::View<unsigned int *,
                             MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("interface_cell_ids_" +
                                       std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  interface_cell_ids.size());

              {
                Kokkos::View<unsigned int *,
                             Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                  host_view(interface_cell_ids.data(),
                            interface_cell_ids.size());

                Kokkos::deep_copy(exec_space,
                                  this->interface_cell_ids_per_color[color],
                                  host_view);
                Kokkos::fence();
              }
              {
                this->interface_cell_dof_indices_per_color[color] =
                  Kokkos::View<unsigned int **,
                               MemorySpace::Default::kokkos_space>(
                    Kokkos::view_alloc("interface_cell_dofs_" +
                                         std::to_string(color),
                                       Kokkos::WithoutInitializing),
                    n_local_dofs,
                    interface_cell_dof_indices.size() / n_local_dofs);

                Kokkos::View<unsigned int **,
                             Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                  host_view(interface_cell_dof_indices.data(),
                            n_local_dofs,
                            interface_cell_dof_indices.size() / n_local_dofs);

                Kokkos::deep_copy(
                  exec_space,
                  this->interface_cell_dof_indices_per_color[color],
                  host_view);
                Kokkos::fence();
              }
            }
        }
      {
        this->interior_interface_cell_dof_indices =
          Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("interior_interface_cell_dof_indices",
                               Kokkos::WithoutInitializing),
            interior_dofs_interface.size());

        Kokkos::View<unsigned int *,
                     Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          host_view(interior_dofs_interface.data(),
                    interior_dofs_interface.size());

        Kokkos::deep_copy(exec_space,
                          this->interior_interface_cell_dof_indices,
                          host_view);
        Kokkos::fence();
      }
    }
  }


  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::Tvmult(
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
  SubdomainLaplaceOperator<dim, fe_degree, number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, typename number>
  const MatrixFree<dim, number> &
  SubdomainLaplaceOperator<dim, fe_degree, number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, typename number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    internal::LaplaceOperatorQuad<dim, fe_degree, number> operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1, number>(
      matrix_free,
      inverse_diagonal,
      operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    double *raw_diagonal = inverse_diagonal.get_values();

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
  SubdomainLaplaceOperator<dim, fe_degree, number>::
    get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  SubdomainLaplaceOperator<dim, fe_degree, number>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number>
  types::global_dof_index
  SubdomainLaplaceOperator<dim, fe_degree, number>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number>
  number
  SubdomainLaplaceOperator<dim, fe_degree, number>::el(
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
  SubdomainLaplaceOperator<dim, fe_degree, number>::get_vector_partitioner()
    const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
