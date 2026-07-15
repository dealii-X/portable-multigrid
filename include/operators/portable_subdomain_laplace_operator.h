#ifndef portable_subdomain_laplace_operator_h
#define portable_subdomain_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "kernels/bk3_kokkos_kernel.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, typename Number>
  class SubdomainLaplaceOperator : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    SubdomainLaplaceOperator(const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
                             const AffineConstraints<Number> &constraints,
                             const AffineConstraints<Number> &constraints_physical,
                             const bool overlap_communication_computation = false);


    SubdomainLaplaceOperator(const DoFHandler<dim>           &dof_handler,
                             const AffineConstraints<Number> &constraints,
                             const AffineConstraints<Number> &constraints_physical,
                             const bool overlap_communication_computation = false);
    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_plain(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override;

    void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override;

    void
    compute_diagonal() override;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override;


    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override;

    const MatrixFree<dim, Number> &
    get_matrix_free() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override;

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override;

    void
    compute_G_tensors();

  private:
    using TeamHandle =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues =
      Kokkos::View<Number *,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients =
      Kokkos::View<Number **,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    void
    setup_dof_indices_per_color();

    void
    cell_loop(const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
              const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
              LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const;
    void
    cell_range_loop(
      const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const;

    void
    cell_loop_neumann(
      const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const;

    void
    cell_loop_dummy(
      const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const bool                                                              ghost_exchange_on,
      const bool                                                              computation_on) const;

    static constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    MatrixFree<dim, Number> matrix_free;

    ObserverPointer<const SubdomainDoFHandler<dim>>  subdomain_dof_handler;
    ObserverPointer<const DoFHandler<dim>>           dof_handler;
    ObserverPointer<const AffineConstraints<Number>> constraints;
    ObserverPointer<const AffineConstraints<Number>> constraints_physical;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries_dirichlet, inverse_diagonal_entries_neumann;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      plain_dof_indices_per_color;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      interior_dof_indices_per_color;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_partitioner;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> physical_boundary_dof_indices;

    std::vector<Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>>
      interface_cell_ids_per_color;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      interface_cell_interior_dof_indices;

    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> temp_vector_src,
      temp_vector_dst, temp_vector_work;

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> interface_weights;

    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> G_tensors;
  };

  template <int dim, int fe_degree, typename Number>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::SubdomainLaplaceOperator(
    const SubdomainDoFHandler<dim>  &subdomain_dof_handler,
    const AffineConstraints<Number> &constraints,
    const AffineConstraints<Number> &constraints_physical,
    const bool                       overlap_communication_computation)
    : subdomain_dof_handler(&subdomain_dof_handler)
    , dof_handler(&subdomain_dof_handler.get_dof_handler())
    , constraints(&constraints)
    , constraints_physical(&constraints_physical)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;

    additional_data.overlap_communication_computation = overlap_communication_computation;

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

    compute_G_tensors();
  }

  template <int dim, int fe_degree, typename Number>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::SubdomainLaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints,
    const AffineConstraints<Number> &constraints_physical,
    const bool                       overlap_communication_computation)
    : subdomain_dof_handler(nullptr)
    , dof_handler(&dof_handler)
    , constraints(&constraints)
    , constraints_physical(&constraints_physical)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;

    additional_data.overlap_communication_computation = overlap_communication_computation;

    const QGauss<1> quadrature_1d(fe_degree + 1);

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

    matrix_free.initialize_dof_vector(temp_vector_src);
    matrix_free.initialize_dof_vector(temp_vector_dst);
    matrix_free.initialize_dof_vector(temp_vector_work);

    setup_dof_indices_per_color();

    compute_G_tensors();
  }

  template <int dim, int fe_degree, typename Number>
  const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_interface_dof_indices_subdomain() const
  {
    return interface_dof_indices_subdomain;
  }

  template <int dim, int fe_degree, typename Number>
  const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_physical_boundary_dof_indices_subdomain()
    const
  {
    return physical_boundary_dof_indices;
  }


  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    DeviceVector<Number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.size());


    src.update_ghost_values();

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        // std::cout << "color: " << color << ", n_cells: " << n_cells
        //           << std::endl;

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            Kokkos::fence();


            constexpr bool is_serial =
              std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
            if (is_serial)
              {
                numBlocks       = 1u;
                threadsPerBlock = 1u;
              }

            // BK3::Parallel::
            //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
            //   Number>(
            //     precomputed_data.shape_values,
            //     precomputed_data.co_shape_gradients,
            //     G_tensors[color],
            //     src_device,
            //     dst_device,
            //     interior_dof_indices_per_color[color],
            //     n_cells,
            //     numBlocks,
            //     threadsPerBlock);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              interior_dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock);

            Kokkos::fence();
          }
      }


    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult_bk3(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    DeviceVector<Number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.size());

    dst = 0.;

    src.update_ghost_values();

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            Kokkos::fence();

            constexpr bool is_serial =
              std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

            if (is_serial)
              {
                numBlocks       = 1u;
                threadsPerBlock = 1u;
              }

            // BK3::Parallel::
            //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
            //   Number>(
            //     precomputed_data.shape_values,
            //     precomputed_data.co_shape_gradients,
            //     G_tensors[color],
            //     src_device,
            //     dst_device,
            //     interior_dof_indices_per_color[color],
            //     n_cells,
            //     numBlocks,
            //     threadsPerBlock);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              interior_dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock);

            Kokkos::fence();
          }
      }

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult_plain(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    DeviceVector<Number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.size());

    dst = 0.;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            Kokkos::fence();

            constexpr bool is_serial =
              std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

            if (is_serial)
              {
                numBlocks       = 1u;
                threadsPerBlock = 1u;
              }

            // BK3::Parallel::
            //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
            //   Number>(
            //     precomputed_data.shape_values,
            //     precomputed_data.co_shape_gradients,
            //     G_tensors[color],
            //     src_device,
            //     dst_device,
            //     plain_dof_indices_per_color[color],
            //     n_cells,
            //     numBlocks,
            //     threadsPerBlock);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              plain_dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock);

            Kokkos::fence();
          }
      }

    // copy physical constrained values
    const auto boundary_dofs = this->physical_boundary_dof_indices;

    if (boundary_dofs.size() > 0)
      Kokkos::parallel_for(
        "work", boundary_dofs.size(), KOKKOS_LAMBDA(const int i) {
          const auto idx  = boundary_dofs(i);
          dst_device(idx) = src_device(idx);
        });
  }


  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult_neumann(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    DeviceVector<Number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.size());

    dst = 0.;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            Kokkos::fence();

            constexpr bool is_serial =
              std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

            if (is_serial)
              {
                numBlocks       = 1u;
                threadsPerBlock = 1u;
              }

            // BK3::Parallel::
            //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
            //   Number>(
            //     precomputed_data.shape_values,
            //     precomputed_data.co_shape_gradients,
            //     G_tensors[color],
            //     src_device,
            //     dst_device,
            //     plain_dof_indices_per_color[color],
            //     n_cells,
            //     numBlocks,
            //     threadsPerBlock);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              plain_dof_indices_per_color[color],
              n_cells,
              numBlocks,
              threadsPerBlock);

            Kokkos::fence();
          }
      }

    // copy physical constrained values
    const auto boundary_dofs = this->physical_boundary_dof_indices;

    if (boundary_dofs.size() > 0)
      Kokkos::parallel_for(
        "work", boundary_dofs.size(), KOKKOS_LAMBDA(const int i) {
          const auto idx  = boundary_dofs(i);
          dst_device(idx) = src_device(idx);
        });
  }

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult_interface_cell_range(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    Assert(
      subdomain_dof_handler != nullptr,
      ExcMessage(
        "Interface cell loop is not available, as this onject was not instantiated with SubdomainDoFHadler."));

    DeviceVector<Number> src_device(src.get_values(), src.size()),
      dst_device(dst.get_values(), dst.size());

    dst                              = 0.;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_interface_cells = interface_cell_ids_per_color[color].size();

        if (n_interface_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            Kokkos::fence();

            constexpr bool is_serial =
              std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

            if (is_serial)
              {
                numBlocks       = 1u;
                threadsPerBlock = 1u;
              }

            // BK3::Parallel::
            //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
            //   Number>(
            //     precomputed_data.shape_values,
            //     precomputed_data.co_shape_gradients,
            //     G_tensors[color],
            //     src_device,
            //     dst_device,
            //     plain_dof_indices_per_color[color],
            //     n_interface_cells,
            //     numBlocks,
            //     threadsPerBlock,
            //     interface_cell_ids_per_color[color]);

            BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              plain_dof_indices_per_color[color],
              n_interface_cells,
              numBlocks,
              threadsPerBlock,
              interface_cell_ids_per_color[color]);

            Kokkos::fence();
          }
      }

    // copy constrained values only for physical boundary dofs
    const auto boundary_dofs = this->physical_boundary_dof_indices;

    if (boundary_dofs.size() > 0)
      Kokkos::parallel_for(
        "work", boundary_dofs.size(), KOKKOS_LAMBDA(const int i) {
          const auto idx  = boundary_dofs(i);
          dst_device(idx) = src_device(idx);
        });
  }

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::cell_loop(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one colorportable_laplace_operator_quad
        auto do_color = [&](const unsigned int color)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;


            const auto &gpu_data = matrix_free.get_data(color, 0);

            auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

            internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
              cell_operator, gpu_data, this->interior_dof_indices_per_color[color], src, dst);

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

                internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
                  cell_operator, gpu_data, this->interior_dof_indices_per_color[color], src, dst);

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

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::cell_loop_neumann(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one colorportable_laplace_operator_quad
        auto do_color = [&](const unsigned int color)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;


            const auto &gpu_data = matrix_free.get_data(color, 0);

            auto team_policy = TeamPolicy(exec, gpu_data.n_cells, Kokkos::AUTO);

            internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
              cell_operator, gpu_data, this->plain_dof_indices_per_color[color], src, dst);

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

                internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
                  cell_operator, gpu_data, this->plain_dof_indices_per_color[color], src, dst);

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
  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::cell_range_loop(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>;

    const auto &colored_graph = matrix_free.get_colored_graph();

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        // helper to process one colorportable_laplace_operator_quad
        auto do_color = [&](const unsigned int color)
          {
            using TeamPolicy =
              Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

            const auto &gpu_data = matrix_free.get_data(color, 0);

            const unsigned int n_interface_cells = this->interface_cell_ids_per_color[color].size();

            auto team_policy = TeamPolicy(exec, n_interface_cells, Kokkos::AUTO);

            internal::ApplyCellKernelRange<dim, Number, Functor> apply_kernel(
              cell_operator,
              gpu_data,
              this->plain_dof_indices_per_color[color],
              this->interface_cell_ids_per_color[color],
              src,
              dst);

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

            const unsigned int n_interface_cells = this->interface_cell_ids_per_color[color].size();

            if (n_interface_cells > 0)
              {
                using TeamPolicy =
                  Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;

                auto team_policy = TeamPolicy(exec, n_interface_cells, Kokkos::AUTO);

                internal::ApplyCellKernelRange<dim, Number, Functor> apply_kernel(
                  cell_operator,
                  gpu_data,
                  this->plain_dof_indices_per_color[color],
                  this->interface_cell_ids_per_color[color],
                  src,
                  dst);
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

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const
  {
    dst = 0.;

    LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number> cell_operator;

    this->cell_loop_dummy(cell_operator, src, dst, ghost_exchange_on, computation_on);

    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::cell_loop_dummy(
    const LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>      &cell_operator,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const

  {
    MemorySpace::Default::kokkos_space::execution_space exec;
    using Functor = LocalLaplaceOperator<dim, fe_degree, fe_degree + 1, Number>;

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

            internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
              cell_operator, gpu_data, this->interior_dof_indices_per_color[color], src, dst);

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


                    internal::ApplyCellKernel<dim, Number, Functor> apply_kernel(
                      cell_operator,
                      gpu_data,
                      this->interior_dof_indices_per_color[color],
                      src,
                      dst);

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

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::setup_dof_indices_per_color()
  {
    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();
    const auto        &partitioner   = matrix_free.get_vector_partitioner();

    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler->get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }
    {
      this->interior_dof_indices_per_color.clear();
      this->interior_dof_indices_per_color.resize(n_colors);

      this->plain_dof_indices_per_color.clear();
      this->plain_dof_indices_per_color.resize(n_colors);

      // this->dof_indices_minus_corners_per_color.clear();
      // this->dof_indices_minus_corners_per_color.resize(n_colors);

      std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
      std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          if (colored_graph[color].size() > 0)
            {
              const auto &mf_data = matrix_free.get_data(color);

              const auto &graph = colored_graph[color];

              this->interior_dof_indices_per_color[color] =
                Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("interior_dof_indices_" + std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  n_local_dofs,
                  mf_data.n_cells);

              auto dof_indices_host =
                Kokkos::create_mirror_view(this->interior_dof_indices_per_color[color]);


              this->plain_dof_indices_per_color[color] =
                Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                  Kokkos::view_alloc("plain_dof_indices_" + std::to_string(color),
                                     Kokkos::WithoutInitializing),
                  n_local_dofs,
                  mf_data.n_cells);

              auto plain_dof_indices_host =
                Kokkos::create_mirror_view(this->plain_dof_indices_per_color[color]);

              // if (subdomain_dof_handler != nullptr)
              //   {
              //     this->dof_indices_minus_corners_per_color[color] =
              //       Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
              //         Kokkos::view_alloc("interior_dof_indices_" + std::to_string(color),
              //                            Kokkos::WithoutInitializing),
              //         n_local_dofs,
              //         mf_data.n_cells);
              //   }
              // auto dof_indices_minus_corners_host =
              //   Kokkos::create_mirror_view(this->dof_indices_minus_corners_per_color[color]);

              for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
                {
                  auto triacell = graph[cell_id];

                  typename DoFHandler<dim>::cell_iterator cell =
                    triacell->as_dof_handler_iterator(*dof_handler);

                  cell->get_dof_indices(local_dof_indices);

                  triacell->get_dof_indices(subdomain_local_dof_indices);

                  if (partitioner)
                    for (auto &index : local_dof_indices)
                      index = partitioner->global_to_local(index);

                  for (unsigned int i = 0; i < n_local_dofs; ++i)
                    {
                      const auto global_dof = local_dof_indices[lex_numbering[i]];
                      const auto subdomain_local_dof =
                        subdomain_local_dof_indices[lex_numbering[i]];

                      if (constraints->is_constrained(subdomain_local_dof))
                        dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                      else
                        dof_indices_host(i, cell_id) = global_dof;

                      if (constraints_physical->is_constrained(subdomain_local_dof))
                        plain_dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                      else
                        plain_dof_indices_host(i, cell_id) = global_dof;

                      // if (subdomain_dof_handler != nullptr)
                      //   {
                      //     if (subdomain_dof_handler->get_dof_info()
                      //           .interface_vertex_dofs_subdomain.is_element(subdomain_local_dof)
                      //           ||
                      //         constraints_physical->is_constrained(subdomain_local_dof))
                      //       dof_indices_minus_corners_host(i, cell_id) =
                      //         numbers::invalid_unsigned_int;
                      //     else
                      //       dof_indices_minus_corners_host(i, cell_id) = global_dof;
                      //   }
                    }
                }

              Kokkos::deep_copy(exec_space,
                                this->interior_dof_indices_per_color[color],
                                dof_indices_host);

              Kokkos::deep_copy(exec_space,
                                this->plain_dof_indices_per_color[color],
                                plain_dof_indices_host);

              // if (subdomain_dof_handler != nullptr)
              //   Kokkos::deep_copy(exec_space,
              //                     this->dof_indices_minus_corners_per_color[color],
              //                     dof_indices_minus_corners_host);
            }
        }


      if (subdomain_dof_handler == nullptr && (constraints_physical->n_constraints() > 0))
        {
          this->physical_boundary_dof_indices =
            Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("physical_boundary_dof_indices", Kokkos::WithoutInitializing),
              constraints_physical->n_constraints());

          auto physical_boundary_dof_indices_host =
            Kokkos::create_mirror_view(physical_boundary_dof_indices);

          unsigned int counter = 0;
          for (const auto &line : constraints_physical->get_lines())
            physical_boundary_dof_indices_host(counter++) = line.index;

          AssertDimension(counter, constraints_physical->n_constraints());
        }
    }

    if (subdomain_dof_handler != nullptr)
      {
        const auto &interface_partitioner =
          this->subdomain_dof_handler->get_interface_vector_partitioner();

        if (interface_partitioner)
          {
            const unsigned int n_locally_relevant_interface_indices =
              this->subdomain_dof_handler->n_locally_relevant_interface_indices();

            this->interface_dof_indices_partitioner =
              Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("interface_dof_indices_partitioner",
                                   Kokkos::WithoutInitializing),
                n_locally_relevant_interface_indices);


            this->interface_dof_indices_subdomain =
              Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("interface_dof_indices_subdomain", Kokkos::WithoutInitializing),
                n_locally_relevant_interface_indices);

            auto interface_dof_indices_partitioner_host =
              Kokkos::create_mirror_view(this->interface_dof_indices_partitioner);


            auto interface_dof_indices_subdomain_host =
              Kokkos::create_mirror_view(this->interface_dof_indices_subdomain);

            for (unsigned int i = 0; i < n_locally_relevant_interface_indices; ++i)
              {
                unsigned int local_index =
                  this->subdomain_dof_handler->local_to_global_interface_partitioner(i);
                unsigned int subdomain_index =
                  this->subdomain_dof_handler->local_interface_to_subdomain(i);

                interface_dof_indices_partitioner_host(i) = local_index;

                interface_dof_indices_subdomain_host(i) = subdomain_index;
              }

            Kokkos::deep_copy(exec_space,
                              this->interface_dof_indices_partitioner,
                              interface_dof_indices_partitioner_host);


            Kokkos::deep_copy(exec_space,
                              this->interface_dof_indices_subdomain,
                              interface_dof_indices_subdomain_host);
          }

        {
          const auto &physical_boundary_dofs =
            this->subdomain_dof_handler->get_dof_info().subdomain_physical_boundary_dofs;

          this->physical_boundary_dof_indices =
            Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("physical_boundary_dof_indices", Kokkos::WithoutInitializing),
              physical_boundary_dofs.n_elements());

          auto boundary_dof_indices_host =
            Kokkos::create_mirror_view(this->physical_boundary_dof_indices);

          unsigned int counter = 0;
          for (const auto &index : physical_boundary_dofs)
            {
              boundary_dof_indices_host(counter) = index;
              ++counter;
            }
          Kokkos::deep_copy(exec_space,
                            this->physical_boundary_dof_indices,
                            boundary_dof_indices_host);
        }

        if (interface_partitioner)
          {
            this->interface_cell_ids_per_color.clear();
            this->interface_cell_ids_per_color.resize(n_colors);

            const auto &interior_dofs_set =
              this->subdomain_dof_handler->get_dof_info().subdomain_interior_dofs;

            IndexSet interior_dofs_interface(
              this->subdomain_dof_handler->get_dof_handler().n_dofs());

            const auto &physical_boundary_dofs =
              this->subdomain_dof_handler->get_dof_info().subdomain_physical_boundary_dofs;

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
                          graph[cell_id]->as_dof_handler_iterator(*dof_handler);

                        cell->get_dof_indices(local_dof_indices);
                        if (partitioner)
                          for (auto &index : local_dof_indices)
                            index = partitioner->global_to_local(index);

                        if (cell->at_boundary())
                          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                            if (cell->at_boundary(f) && cell->face(f)->boundary_id() ==
                                                          subdomain_dof_handler->get_interface_id())
                              {
                                interface_cell_ids.push_back(cell_id);

                                for (unsigned int i = 0; i < n_local_dofs; ++i)
                                  {
                                    const unsigned int global_dof =
                                      local_dof_indices[lex_numbering[i]];

                                    if (physical_boundary_dofs.is_element(global_dof))
                                      interface_cell_dof_indices.push_back(
                                        numbers::invalid_unsigned_int);
                                    else
                                      interface_cell_dof_indices.push_back(global_dof);

                                    if (interior_dofs_set.is_element(global_dof))
                                      interior_dofs_interface.add_index(global_dof);
                                  }

                                break;
                              }
                      }

                    this->interface_cell_ids_per_color[color] =
                      Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                        Kokkos::view_alloc("interface_cell_ids_" + std::to_string(color),
                                           Kokkos::WithoutInitializing),
                        interface_cell_ids.size());

                    {
                      Kokkos::View<unsigned int *,
                                   Kokkos::LayoutLeft,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                        host_view(interface_cell_ids.data(), interface_cell_ids.size());

                      Kokkos::deep_copy(exec_space,
                                        this->interface_cell_ids_per_color[color],
                                        host_view);
                    }
                  }

                this->interface_cell_interior_dof_indices =
                  Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
                    Kokkos::view_alloc("interface_cell_interior_dof_indices",
                                       Kokkos::WithoutInitializing),
                    interior_dofs_interface.n_elements());

                std::vector<unsigned int> interior_dofs_interface_vec;

                for (auto index : interior_dofs_interface)
                  {
                    interior_dofs_interface_vec.push_back(index);
                  }

                Kokkos::View<unsigned int *,
                             Kokkos::LayoutLeft,
                             Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                  host_view(interior_dofs_interface_vec.data(), interior_dofs_interface_vec.size());

                Kokkos::deep_copy(exec_space, this->interface_cell_interior_dof_indices, host_view);
              }

            // {
            //   const auto &sb_dof_info = subdomain_dof_handler->get_dof_info();

            //   interface_dof_indices_corner_subdomain =
            //     Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
            //       Kokkos::view_alloc("interface_dof_indices_corner_subdomain",
            //                          Kokkos::WithoutInitializing),
            //       sb_dof_info.subdomain_interface_vertex_dofs.size());


            //   interface_dof_indices_edge_subdomain =
            //     Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
            //       Kokkos::view_alloc("interface_dof_indices_edge_subdomain",
            //                          Kokkos::WithoutInitializing),
            //       sb_dof_info.subdomain_interface_edge_dofs.size());


            //   interface_dof_indices_face_subdomain =
            //     Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
            //       Kokkos::view_alloc("interface_dof_indices_face_subdomain",
            //                          Kokkos::WithoutInitializing),
            //       sb_dof_info.subdomain_interface_face_dofs.size());


            //   interface_dof_indices_minus_corners_subdomain =
            //     Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
            //       Kokkos::view_alloc("interface_dof_indices_minus_corners_subdomain",
            //                          Kokkos::WithoutInitializing),
            //       sb_dof_info.subdomain_interface_edge_dofs.size() +
            //         sb_dof_info.subdomain_interface_face_dofs.size());

            //   auto corner_dofs_host =
            //     Kokkos::create_mirror_view(interface_dof_indices_corner_subdomain);
            //   auto edge_dofs_host =
            //     Kokkos::create_mirror_view(interface_dof_indices_edge_subdomain);
            //   auto face_dofs_host =
            //     Kokkos::create_mirror_view(interface_dof_indices_face_subdomain);
            //   auto dofs_minus_corners_host =
            //     Kokkos::create_mirror_view(interface_dof_indices_minus_corners_subdomain);

            //   for (unsigned int i = 0; i < sb_dof_info.subdomain_interface_vertex_dofs.size();
            //   ++i)
            //     corner_dofs_host(i) =
            //     std::get<1>(sb_dof_info.subdomain_interface_vertex_dofs[i]);

            //   for (unsigned int i = 0; i < sb_dof_info.subdomain_interface_edge_dofs.size(); ++i)
            //     {
            //       edge_dofs_host(i) = std::get<1>(sb_dof_info.subdomain_interface_edge_dofs[i]);
            //       dofs_minus_corners_host(i) =
            //         std::get<1>(sb_dof_info.subdomain_interface_edge_dofs[i]);
            //     }

            //   for (unsigned int i = 0; i < sb_dof_info.subdomain_interface_face_dofs.size(); ++i)
            //     {
            //       face_dofs_host(i) = std::get<1>(sb_dof_info.subdomain_interface_face_dofs[i]);
            //       dofs_minus_corners_host(sb_dof_info.subdomain_interface_edge_dofs.size() + i) =
            //         std::get<1>(sb_dof_info.subdomain_interface_face_dofs[i]);
            //     }


            //   Kokkos::deep_copy(exec_space,
            //                     interface_dof_indices_corner_subdomain,
            //                     corner_dofs_host);

            //   Kokkos::deep_copy(exec_space, interface_dof_indices_edge_subdomain,
            //   edge_dofs_host);

            //   Kokkos::deep_copy(exec_space, interface_dof_indices_face_subdomain,
            //   face_dofs_host);

            //   Kokkos::deep_copy(exec_space,
            //                     interface_dof_indices_minus_corners_subdomain,
            //                     dofs_minus_corners_host);
            // }
          }
      }
    exec_space.fence();
  }

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::compute_G_tensors()
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

            G_tensors[color] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
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
                    Number components[symmetric_tensor_dim];

                    int idx = 0;
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = d1; d2 < dim; ++d2)
                        {
                          Number sum = 0;
                          for (int k = 0; k < dim; ++k)
                            sum += inv_jacobian(q_point, cell_id, k, d1) *
                                   inv_jacobian(q_point, cell_id, k, d2);
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



  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::Tvmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }



  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, typename Number>
  const MatrixFree<dim, Number> &
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, typename Number>
  void
  SubdomainLaplaceOperator<dim, fe_degree, Number>::compute_diagonal()
  {
    this->inverse_diagonal_entries_dirichlet.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>());

    this->inverse_diagonal_entries_neumann.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>());

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &inverse_diagonal_dirichlet =
      inverse_diagonal_entries_dirichlet->get_vector();
    initialize_dof_vector(inverse_diagonal_dirichlet);

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &inverse_diagonal_neumann =
      inverse_diagonal_entries_neumann->get_vector();
    initialize_dof_vector(inverse_diagonal_neumann);

    using QuadOperatorType = internal::LaplaceOperatorQuad<dim, fe_degree, fe_degree + 1, Number>;
    QuadOperatorType operator_quad;

    MatrixFreeTools::internal::
      ComputeDiagonalCellAction<dim, fe_degree, fe_degree + 1, 1, Number, QuadOperatorType>
        cell_action(0, operator_quad, EvaluationFlags::gradients, EvaluationFlags::gradients);

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> dummy;

    matrix_free.cell_loop(cell_action, dummy, inverse_diagonal_dirichlet);

    inverse_diagonal_neumann = inverse_diagonal_dirichlet;

    matrix_free.set_constrained_values(Number(1.), inverse_diagonal_dirichlet, 0);

    Number *raw_diagonal_dirichlet = inverse_diagonal_dirichlet.get_values();

    Kokkos::parallel_for(
      inverse_diagonal_dirichlet.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal_dirichlet[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal_dirichlet[i] = 1. / raw_diagonal_dirichlet[i];
      });

    const auto physical_boundary_dofs = this->physical_boundary_dof_indices;

    Number *raw_diagonal_neumann = inverse_diagonal_neumann.get_values();

    Kokkos::parallel_for(
      physical_boundary_dofs.size(),
      KOKKOS_LAMBDA(int i) { raw_diagonal_neumann[physical_boundary_dofs[i]] = Number(1.); });


    Kokkos::parallel_for(
      inverse_diagonal_neumann.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal_neumann[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal_neumann[i] = 1. / raw_diagonal_neumann[i];
      });
  }

  template <int dim, int fe_degree, typename Number>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries_dirichlet;
  }

  template <int dim, int fe_degree, typename Number>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_matrix_diagonal_inverse_neumann() const
  {
    return inverse_diagonal_entries_neumann;
  }

  template <int dim, int fe_degree, typename Number>
  types::global_dof_index
  SubdomainLaplaceOperator<dim, fe_degree, Number>::m() const
  {
    if (matrix_free.get_vector_partitioner())
      return matrix_free.get_vector_partitioner()->size();

    return dof_handler->n_dofs();
  }

  template <int dim, int fe_degree, typename Number>
  types::global_dof_index
  SubdomainLaplaceOperator<dim, fe_degree, Number>::n() const
  {
    if (matrix_free.get_vector_partitioner())
      return matrix_free.get_vector_partitioner()->size();

    return dof_handler->n_dofs();
  }

  template <int dim, int fe_degree, typename Number>
  Number
  SubdomainLaplaceOperator<dim, fe_degree, Number>::el(const types::global_dof_index row,
                                                       const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries_dirichlet.get() != nullptr &&
             inverse_diagonal_entries_dirichlet->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries_dirichlet)(row, row);
  }

  template <int dim, int fe_degree, typename Number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }


  template <int dim, int fe_degree, typename Number>
  inline const SubdomainDoFHandler<dim> &
  SubdomainLaplaceOperator<dim, fe_degree, Number>::get_subdomain_dof_handler() const
  {
    return *subdomain_dof_handler;
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
