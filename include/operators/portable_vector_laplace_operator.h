#ifndef portable_vector_laplace_operator_h
#define portable_vector_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "base/portable_laplace_operator_base.h"
#include "kernels/bk4_cuda_kernels.cuh"
#include "kernels/bk4_cuda_plain_kernels.cuh"
#include "kernels/bk4_kokkos_kernels.h"
#include "operators/portable_vector_laplace_operator_quad.h"



DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename number>
  class LocalVectorLaplaceOperatorStep64
  {
  public:
    static constexpr unsigned int n_local_dofs = n_components * Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);

    LocalVectorLaplaceOperatorStep64() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const typename MatrixFree<dim, number>::Data *data,
               const DeviceVector<number>                   &src,
               DeviceVector<number>                         &dst) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, number> fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      data->for_each_quad_point(
        [&](const int q_point)
          { fe_eval.submit_gradient(fe_eval.get_gradient(q_point), q_point); });

      fe_eval.integrate(EvaluationFlags::gradients);

      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim,
            int fe_degree,
            int n_components,
            typename number,
            int n_q_points_1d = fe_degree + 1>
  class VectorLaplaceOperator : public LaplaceOperatorBase<dim, number>
  {
  public:
    VectorLaplaceOperator(const DoFHandler<dim>           &dof_handler,
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
    vmult_bk4(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
              const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    compute_rhs_bk4(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &rhs) const;

    void
    compute_rhs_cuda(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &rhs) const;

    void
    vmult_tensor_core(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_cuda(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
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

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

  private:
    static constexpr unsigned int n_dofs_per_component = Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_local_dofs         = n_components * n_dofs_per_component;
    static constexpr unsigned int n_q_points           = Utilities::pow(n_q_points_1d, dim);

    static constexpr std::size_t  tensor_core_shmem_per_block = 10'000;
    static constexpr unsigned int tensor_core_nelmt_per_batch =
      std::max<std::size_t>(1, tensor_core_shmem_per_block / (4 * n_q_points) / sizeof(number));

    MatrixFree<dim, number> matrix_free;

    ObserverPointer<const AffineConstraints<number>> constraints;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    std::vector<Kokkos::Array<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>,
                              n_components>>
      dof_indices_per_color;

    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>> G_tensors;
  };



  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::VectorLaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
  {
    AssertDimension(dof_handler.get_fe().n_components(), n_components);

    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;

    this->constraints = &constraints;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    const QGauss<1> quadrature_1d(n_q_points_1d);
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

    setup_dof_indices_per_color();

    compute_G_tensors();
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    this->vmult_dealii(dst, src);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult_dealii(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    LocalVectorLaplaceOperatorStep64<dim, fe_degree, n_q_points_1d, n_components, number>
      cell_operator;

    matrix_free.cell_loop(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult_bk4(
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

    unsigned int numBlocks       = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

    if (is_serial)
      threadsPerBlock = 1u;

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK4::Parallel::
              KokkosKernelAbstracted<dim, fe_degree, n_q_points_1d, n_components, number>(
                precomputed_data.shape_values,
                precomputed_data.co_shape_gradients,
                G_tensors[color],
                src_device,
                dst_device,
                dof_indices_per_color[color],
                n_cells,
                numBlocks,
                threadsPerBlock);
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

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::compute_rhs_bk4(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &rhs) const
  {
    rhs = 0.;

    DeviceVector<number> rhs_device(rhs.get_values(), rhs.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
    if (is_serial)
      threadsPerBlock = 1u;

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK4::Parallel::KokkosRHSAbstracted<dim, fe_degree, n_q_points_1d, n_components, number>(
              precomputed_data.shape_values,
              precomputed_data.JxW,
              rhs_device,
              dof_indices_per_color[color],
              n_cells,
              numbers::invalid_unsigned_int,
              threadsPerBlock);
          }
      }

    rhs.compress(VectorOperation::add);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::compute_rhs_cuda(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &rhs) const
  {
    static_assert(dim == 3,
                  "compute_rhs_cuda() only implements the dim == 3 kernel "
                  "(bk4_cuda_plain_kernels.cuh).");
    static_assert(std::is_same_v<number, double>,
                  "gemm_rhs_operator's implementation assumes double.");

#ifndef __CUDACC__
    (void)rhs;
    Assert(false,
           ExcMessage("compute_rhs_cuda() requires this translation unit to be compiled "
                      "by nvcc (a CUDA-enabled Kokkos build) -- it was not."));
#else
    rhs = 0.;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK4::Parallel::Cuda::launch_gemm_rhs_operator<n_q_points_1d,
                                                          fe_degree + 1,
                                                          tensor_core_nelmt_per_batch,
                                                          n_components>(
              n_cells,
              precomputed_data.shape_values.data(),
              precomputed_data.JxW,
              rhs.get_values(),
              dof_indices_per_color[color]);
          }
      }

    Kokkos::fence();

    rhs.compress(VectorOperation::add);
#endif
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult_tensor_core(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    static_assert(dim == 3,
                  "vmult_tensor_core() only implements the dim == 3 tensor-core kernel "
                  "(bk4_cuda_kernels.cuh).");
    static_assert(std::is_same_v<number, double>,
                  "mma.sync.aligned.m8n8k4.f64 is a double-precision-only instruction.");

#ifndef __CUDACC__
    (void)dst;
    (void)src;
    Assert(false,
           ExcMessage("vmult_tensor_core() requires this translation unit to be compiled "
                      "by nvcc (a CUDA-enabled Kokkos build) -- it was not."));
#else
    dst = 0.;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            if (fe_degree == 1)
              BK4::Parallel::TensorCore::
                launch_f64_m8n8k4_mma<n_q_points_1d, fe_degree + 1, 16, n_components>(
                  n_cells,
                  precomputed_data.shape_values.data(),
                  precomputed_data.co_shape_gradients.data(),
                  G_tensors[color].data(),
                  src.get_values(),
                  dst.get_values(),
                  dof_indices_per_color[color]);
            else if (fe_degree <= 3)
              BK4::Parallel::TensorCore::
                launch_f64_m8n8k4_mma<n_q_points_1d, fe_degree + 1, 4, n_components>(
                  n_cells,
                  precomputed_data.shape_values.data(),
                  precomputed_data.co_shape_gradients.data(),
                  G_tensors[color].data(),
                  src.get_values(),
                  dst.get_values(),
                  dof_indices_per_color[color]);
            else if (fe_degree == 4)
              BK4::Parallel::TensorCore::
                launch_f64_m8n8k4_mma<n_q_points_1d, fe_degree + 1, 2, n_components>(
                  n_cells,
                  precomputed_data.shape_values.data(),
                  precomputed_data.co_shape_gradients.data(),
                  G_tensors[color].data(),
                  src.get_values(),
                  dst.get_values(),
                  dof_indices_per_color[color]);
            else
              BK4::Parallel::TensorCore::
                launch_f64_m8n8k4_mma<n_q_points_1d, fe_degree + 1, 1, n_components>(
                  n_cells,
                  precomputed_data.shape_values.data(),
                  precomputed_data.co_shape_gradients.data(),
                  G_tensors[color].data(),
                  src.get_values(),
                  dst.get_values(),
                  dof_indices_per_color[color]);
          }
      };

    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color)
      if (colored_graph[color].size() > color)
        do_color(color);

    Kokkos::fence();

    dst.compress(VectorOperation::add);

    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
#endif
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult_cuda(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    static_assert(dim == 3,
                  "vmult_cuda() only implements the dim == 3 kernel (bk4_cuda_plain_kernels.cuh).");
    static_assert(std::is_same_v<number, double>,
                  "gemm_laplace_operator's G tensor layout assumes double.");

#ifndef __CUDACC__
    (void)dst;
    (void)src;
    Assert(false,
           ExcMessage("vmult_cuda() requires this translation unit to be compiled "
                      "by nvcc (a CUDA-enabled Kokkos build) -- it was not."));
#else
    dst = 0.;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK4::Parallel::Cuda::launch_gemm_laplace_operator<n_q_points_1d,
                                                              fe_degree + 1,
                                                              tensor_core_nelmt_per_batch,
                                                              n_components>(
              n_cells,
              precomputed_data.shape_values.data(),
              precomputed_data.co_shape_gradients.data(),
              G_tensors[color].data(),
              src.get_values(),
              dst.get_values(),
              dof_indices_per_color[color]);
          }
      };

    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color)
      if (colored_graph[color].size() > color)
        do_color(color);

    Kokkos::fence();

    dst.compress(VectorOperation::add);

    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
#endif
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::vmult_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const
  {
    dst = 0.;

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int numBlocks       = numbers::invalid_unsigned_int;
    unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

    if (is_serial)
      threadsPerBlock = 1u;

    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK4::Parallel::
              KokkosKernelAbstracted<dim, fe_degree, n_q_points_1d, n_components, number>(
                precomputed_data.shape_values,
                precomputed_data.co_shape_gradients,
                G_tensors[color],
                src_device,
                dst_device,
                dof_indices_per_color[color],
                n_cells,
                numBlocks,
                threadsPerBlock);
          }
      };

    if (matrix_free.use_overlap_communication_computation())
      {
        if (ghost_exchange_on)
          src.update_ghost_values_start(0);

        if (colored_graph.size() > 0 && colored_graph[0].size() > 0)
          if (computation_on)
            do_color(0);

        if (ghost_exchange_on)
          src.update_ghost_values_finish();

        if (colored_graph.size() > 1 && colored_graph[1].size() > 0)
          {
            if (computation_on)
              do_color(1);

            Kokkos::fence();
          }

        if (ghost_exchange_on)
          dst.compress_start(0, VectorOperation::add);

        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          if (computation_on)
            do_color(2);

        if (ghost_exchange_on)
          dst.compress_finish(VectorOperation::add);
      }
    else
      {
        if (ghost_exchange_on)
          src.update_ghost_values();

        if (computation_on)
          for (unsigned int color = 0; color < n_colors; ++color)
            {
              if (colored_graph[color].size() > color)
                do_color(color);
            }

        if (ghost_exchange_on)
          dst.compress(VectorOperation::add);
      }

    if (ghost_exchange_on)
      src.zero_out_ghost_values();

    matrix_free.copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::Tvmult(
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

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  const MatrixFree<dim, number> &
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::get_matrix_free()
    const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &inverse_diagonal =
      inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    internal::VectorLaplaceOperatorQuad<dim, fe_degree, n_q_points_1d, n_components, number>
      operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, n_components, number>(
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

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::
    get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  types::global_dof_index
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  types::global_dof_index
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  number
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::
    get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::
    setup_dof_indices_per_color()
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

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto &mf_data = matrix_free.get_data(color);
            const auto &graph   = colored_graph[color];

            for (unsigned int c = 0; c < n_components; ++c)
              this->dof_indices_per_color[color][c] =
                DoFIndicesView(Kokkos::view_alloc("dof_indices_" + std::to_string(color) + "_" +
                                                    std::to_string(c),
                                                  Kokkos::WithoutInitializing),
                               n_dofs_per_component,
                               mf_data.n_cells);

            using DoFIndicesHostView =
              decltype(Kokkos::create_mirror_view(std::declval<DoFIndicesView>()));
            std::array<DoFIndicesHostView, n_components> dof_indices_host;
            for (unsigned int c = 0; c < n_components; ++c)
              dof_indices_host[c] =
                Kokkos::create_mirror_view(this->dof_indices_per_color[color][c]);

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
                    const unsigned int c       = i / n_dofs_per_component;
                    const unsigned int i_local = i % n_dofs_per_component;

                    const auto global_dof          = local_dof_indices[lex_numbering[i]];
                    const auto subdomain_local_dof = subdomain_local_dof_indices[lex_numbering[i]];

                    if (constraints->is_constrained(subdomain_local_dof))
                      dof_indices_host[c](i_local, cell_id) = numbers::invalid_unsigned_int;
                    else
                      dof_indices_host[c](i_local, cell_id) = global_dof;
                  }
              }

            for (unsigned int c = 0; c < n_components; ++c)
              Kokkos::deep_copy(exec_space,
                                this->dof_indices_per_color[color][c],
                                dof_indices_host[c]);
            Kokkos::fence();
          }
      }
  }

  template <int dim, int fe_degree, int n_components, typename number, int n_q_points_1d>
  void
  VectorLaplaceOperator<dim, fe_degree, n_components, number, n_q_points_1d>::compute_G_tensors()
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

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
