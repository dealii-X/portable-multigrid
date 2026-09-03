#ifndef portable_laplace_operator_h
#define portable_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "kernels/bk3_kokkos_kernels.h"
#include "kernels/portable_local_laplace_operator.h"
#include "kernels/portable_local_laplace_operator_batched.h"
#include "kernels/portable_local_laplace_operator_batched_view.h"
#include "matrix_free_dealii/portable_fe_evaluation.h"
#include "matrix_free_dealii/portable_matrix_free.h"
#include "matrix_free_dealii/portable_matrix_free.templates.h"
#include "operators/portable_laplace_operator_quad.h"



DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Same as LocalLaplaceOperatorStep64 (portable_local_laplace_operator.h),
  // but built on Copy::Portable::MatrixFree/FEEvaluation instead of real
  // deal.II's own -- exercises the local matrix_free_dealii port, in
  // particular FEEvaluationImpl's evaluate()/integrate() now reading shape
  // data from SharedData's shared-memory-staged copies rather than
  // precomputed_data (global memory).
  template <int dim, int fe_degree, int n_q_points_1d_, typename number>
  class LocalLaplaceOperatorNew
  {
  public:
    static constexpr unsigned int n_local_dofs  = Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points    = Utilities::pow(n_q_points_1d_, dim);
    static constexpr unsigned int n_q_points_1d = n_q_points_1d_;

    LocalLaplaceOperatorNew() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const typename Copy::Portable::MatrixFree<dim, number>::Data *data,
               const Copy::Portable::DeviceVector<number>                   &src,
               Copy::Portable::DeviceVector<number>                         &dst) const
    {
      Copy::Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> fe_eval(
        data, data->cell_index);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      data->for_each_quad_point(
        [&](const int q_point)
          { fe_eval.submit_gradient(fe_eval.get_gradient(q_point), q_point); });

      fe_eval.integrate(EvaluationFlags::gradients);

      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename number>
  class LaplaceOperator : public LaplaceOperatorBase<dim, number>
  {
  public:
    LaplaceOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<number> &constraints,
                    bool                             overlap_communication_computation);

    void
    vmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    void
    vmult_dealii(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
                 const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_bk3(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
              const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_bk3_abstracted(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_dealii_batched(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_dealii_batched_fused(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_dealii_batched_view(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_dealii_new(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override;

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const override;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const override;

    // vmult_dealii_new() needs both its src and dst vectors initialized
    // against matrix_free_new specifically: Copy::Portable::MatrixFree::
    // distributed_cell_loop() picks its dof_handler_index by comparing
    // dst.get_partitioner().get() against each registered DoFHandler's own
    // partitioner pointer by identity, so a vector initialized against the
    // real (separate) matrix_free -- as initialize_dof_vector() above does
    // -- never matches, regardless of whether the underlying dof layout is
    // equivalent.
    void
    initialize_dof_vector_new(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const;

    void
    compute_diagonal() override;

    void
    setup_dof_indices_per_color();

    void
    compute_G_tensors();

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    number
    el(const types::global_dof_index row, const types::global_dof_index col) const override;

    const MatrixFree<dim, number> &
    get_matrix_free() const override;

    // matrix_free_new's own accessor -- see initialize_dof_vector_new()'s
    // comment for why vmult_dealii_new()'s vectors need matrix_free_new
    // specifically, not the real matrix_free above.
    const Copy::Portable::MatrixFree<dim, number> &
    get_matrix_free_new() const;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

  private:
    using TeamHandle =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues =
      Kokkos::View<number *,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients =
      Kokkos::View<number **,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    void
    cell_loop(const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>      &cell_operator,
              const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
              LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst) const;

    void
    cell_loop_dummy(
      const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>      &cell_operator,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const bool                                                              ghost_exchange_on,
      const bool                                                              computation_on) const;

    template <typename Functor>
    void
    cell_loop_batched(const Functor &cell_operator,
                      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
                      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst) const;

    template <typename Functor>
    void
    cell_loop_batched_view(
      const Functor                                                          &cell_operator,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst) const;

    static constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);

    MatrixFree<dim, number> matrix_free;

    // Local matrix_free_dealii port of the above, filled identically -- see
    // vmult_dealii_new()/LocalLaplaceOperatorNew.
    Copy::Portable::MatrixFree<dim, number> matrix_free_new;

    ObserverPointer<const AffineConstraints<number>> constraints;

    static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dof_indices_per_color;

    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>> G_tensors;
  };

  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim, fe_degree, number>::LaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;

    this->constraints = &constraints;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    const QGauss<1> quadrature_1d(fe_degree + 1);
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

    typename Copy::Portable::MatrixFree<dim, number>::AdditionalData additional_data_new;
    additional_data_new.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data_new.overlap_communication_computation = overlap_communication_computation;
    matrix_free_new.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data_new);

    setup_dof_indices_per_color();

    compute_G_tensors();
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    this->vmult_dealii(dst, src);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dealii(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    // LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number> cell_operator;
    LocalLaplaceOperatorStep64<dim, fe_degree, fe_degree + 1, number> cell_operator;

    matrix_free.cell_loop(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dealii_new(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    LocalLaplaceOperatorNew<dim, fe_degree, fe_degree + 1, number> cell_operator;

    matrix_free_new.cell_loop(cell_operator, src, dst);

    matrix_free_new.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_bk3(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks         = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock   = numbers::invalid_unsigned_int;
    // unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_cells_per_batch = 1;


    if (is_serial)
      {
        threadsPerBlock = 1u;
      }

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock,
              n_cells_per_batch);
          }
      };

    if (matrix_free.use_overlap_communication_computation())
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

        dst.compress_start(0, VectorOperation::add);
        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (colored_graph[color].size() > color)
              do_color(color);
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_bk3_abstracted(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks         = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock   = numbers::invalid_unsigned_int;
    // unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_cells_per_batch = 1;


    if (is_serial)
      {
        threadsPerBlock = 1u;
      }

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK3::Parallel::KokkosKernelAbstracted<dim, fe_degree, fe_degree + 1, number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock,
              n_cells_per_batch);
          }
      };

    if (matrix_free.use_overlap_communication_computation())
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

        dst.compress_start(0, VectorOperation::add);
        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (colored_graph[color].size() > color)
              do_color(color);
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dealii_batched(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    LocalLaplaceOperatorGeneric<dim, fe_degree, fe_degree + 1, number> cell_operator;

    this->cell_loop_batched(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dealii_batched_fused(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    LocalLaplaceOperatorGenericSplit<dim, fe_degree, fe_degree + 1, number> cell_operator;

    this->cell_loop_batched(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dealii_batched_view(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    LocalLaplaceOperatorGenericView<dim, fe_degree, fe_degree + 1, number> cell_operator;

    this->cell_loop_batched_view(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::cell_loop(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one color
        auto do_color = [&](const unsigned int color)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;


            const auto &gpu_data = matrix_free.get_data(color, 0);

            auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

            Portable::internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
              cell_operator, gpu_data, this->dof_indices_per_color[color], src, dst);

            Kokkos::parallel_for("dealii::MatrixFree::distributed_cell_loop color " +
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
                using TeamPolicy =
                  Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

                auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

                internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
                  cell_operator, gpu_data, this->dof_indices_per_color[color], src, dst);

                Kokkos::parallel_for("dealii::MatrixFree::distributed_cell_loop color " +
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
  template <typename Functor>
  void
  LaplaceOperator<dim, fe_degree, number>::cell_loop_batched(
    const Functor                                                          &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst) const
  {
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    // The batch-size/team-size heuristic in cell_loop_batched_launch()
    // assumes a GPU backend; on a serial Kokkos host backend it must be
    // overridden to team_size == 1, exactly as
    // LaplaceOperatorBK3::vmult() does for BK3Custom::Parallel::KokkosKernel.
    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks         = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock   = numbers::invalid_unsigned_int;
    // unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_cells_per_batch = 1;


    if (is_serial)
      {
        threadsPerBlock = 1u;
      }

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const auto &gpu_data = matrix_free.get_data(color, 0);

        if (gpu_data.n_cells > 0)
          cell_loop_batched_launch<dim, fe_degree, fe_degree + 1, number>(
            cell_operator,
            gpu_data,
            this->dof_indices_per_color[color],
            this->G_tensors[color],
            src,
            dst,
            numBlocks,
            threadsPerBlock,
            n_cells_per_batch);
      };

    if (matrix_free.use_overlap_communication_computation())
      {
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

        for (unsigned int color = 0; color < n_colors; ++color)
          do_color(color);

        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  template <typename Functor>
  void
  LaplaceOperator<dim, fe_degree, number>::cell_loop_batched_view(
    const Functor                                                          &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst) const
  {
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks         = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock   = numbers::invalid_unsigned_int;
    // unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_cells_per_batch = 1;

    if (is_serial)
      {
        threadsPerBlock = 1u;
      }

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const auto &gpu_data = matrix_free.get_data(color, 0);

        if (gpu_data.n_cells > 0)
          cell_loop_batched_launch_view<dim, fe_degree, fe_degree + 1, number>(
            cell_operator,
            gpu_data,
            this->dof_indices_per_color[color],
            src,
            dst,
            numBlocks,
            threadsPerBlock,
            n_cells_per_batch);
      };

    if (matrix_free.use_overlap_communication_computation())
      {
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

        for (unsigned int color = 0; color < n_colors; ++color)
          do_color(color);

        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const
  {
    dst = 0.;

    LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number> cell_operator;

    this->cell_loop_dummy(cell_operator, src, dst, ghost_exchange_on, computation_on);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::cell_loop_dummy(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one color
        auto do_color = [&](const unsigned int color)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;


            const auto &gpu_data = matrix_free.get_data(color, 0);

            auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

            internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
              cell_operator, gpu_data, this->dof_indices_per_color[color], src, dst);

            Kokkos::parallel_for("dealii::MatrixFree::distributed_cell_loop color " +
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
                    using TeamPolicy =
                      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

                    auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);


                    internal::ApplyCellKernel<dim, number, Functor> apply_kernel(
                      cell_operator, gpu_data, this->dof_indices_per_color[color], src, dst);

                    Kokkos::parallel_for("dealii::MatrixFree::distributed_cell_loop color " +
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
  LaplaceOperator<dim, fe_degree, number>::setup_dof_indices_per_color()
  {
    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    const auto &dof_handler = matrix_free.get_dof_handler();

    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }

    this->dof_indices_per_color.clear();
    this->dof_indices_per_color.resize(n_colors);

    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
    std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto &mf_data = matrix_free.get_data(color);

            const auto &graph = colored_graph[color];

            this->dof_indices_per_color[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("dof_indices_" + std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_local_dofs,
                mf_data.n_cells);

            auto dof_indices_host = Kokkos::create_mirror_view(this->dof_indices_per_color[color]);


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
                    const auto global_dof          = local_dof_indices[lex_numbering[i]];
                    const auto subdomain_local_dof = subdomain_local_dof_indices[lex_numbering[i]];

                    if (constraints->is_constrained(subdomain_local_dof))
                      dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                    else
                      dof_indices_host(i, cell_id) = global_dof;
                  }
              }

            Kokkos::deep_copy(exec_space, this->dof_indices_per_color[color], dof_indices_host);
            Kokkos::fence();
          }
      }
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::compute_G_tensors()
  {
    constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    G_tensors.resize(n_colors);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto        &precomputed_data = matrix_free.get_data(color);
            const unsigned int n_cells          = precomputed_data.n_cells;

            const auto &inv_jacobian = precomputed_data.inv_jacobian;
            const auto &JxW          = precomputed_data.JxW;

            G_tensors[color] = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("G_tensor_color_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              symmetric_tensor_dim * n_cells * n_q_points);

            auto G = G_tensors[color];

            Kokkos::parallel_for(
              "Fill_G_tensor_color" + std::to_string(color),
              Kokkos::RangePolicy<dealii::MemorySpace::Default::kokkos_space::execution_space>(
                0, n_cells),
              KOKKOS_LAMBDA(const int cell_id) {
                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    number components[symmetric_tensor_dim];

                    int idx = 0;
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = d1; d2 < dim; ++d2)
                        {
                          number sum = 0;
                          for (int k = 0; k < dim; ++k)
                            sum += inv_jacobian(q_point, cell_id, d1, k) *
                                   inv_jacobian(q_point, cell_id, d2, k);
                          components[idx] = JxW(q_point, cell_id) * sum;
                          ++idx;
                        }

                    for (int c = 0; c < symmetric_tensor_dim; ++c)
                      {
                        G[cell_id * symmetric_tensor_dim * n_q_points + c * n_q_points + q_point] =
                          components[c];
                      }
                  }
              });
            Kokkos::fence();
          }
      }
  }


  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::Tvmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
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
  void
  LaplaceOperator<dim, fe_degree, number>::initialize_dof_vector_new(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const
  {
    matrix_free_new.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, typename number>
  const MatrixFree<dim, number> &
  LaplaceOperator<dim, fe_degree, number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, typename number>
  const Copy::Portable::MatrixFree<dim, number> &
  LaplaceOperator<dim, fe_degree, number>::get_matrix_free_new() const
  {
    return matrix_free_new;
  }

  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &inverse_diagonal =
      inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    internal::LaplaceOperatorQuad<dim, fe_degree, fe_degree + 1, number> operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1, number>(
      matrix_free,
      inverse_diagonal,
      operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

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
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
  LaplaceOperator<dim, fe_degree, number>::get_matrix_diagonal_inverse() const
  {
    // std::cout << "Diagonal entries: ";
    // for (const number value : inverse_diagonal_entries->get_vector())
    //   std::cout << value << " ";
    // std::cout << std::endl;
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
  LaplaceOperator<dim, fe_degree, number>::el(const types::global_dof_index row,
                                              const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
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