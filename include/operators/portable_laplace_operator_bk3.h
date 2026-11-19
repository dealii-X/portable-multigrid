#ifndef portable_laplace_operator_bk3_h
#define portable_laplace_operator_bk3_h

#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/shape_info.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/portable_laplace_operator_base.h"
#include "kernels/bk3_kokkos_kernels.h"
#include "kernels/bk4_cuda_kernels.cuh"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Matrix-free Laplace operator on the BK3 Kokkos kernels. Independent of
  // deal.II's portable matrix-free class: it builds its own partitioner, cell coloring,
  // 1-D shape tables, Dirichlet masks and per-cell G tensors
  // (G = JxW * J^{-1} J^{-T}, symmetric part only) from the DoFHandler.
  template <int dim, int fe_degree, typename number, int n_q_points_1d = fe_degree + 1>
  class LaplaceOperatorBK3 : public LaplaceOperatorBase<dim, number>
  {
  public:
    using VectorType   = LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;
    using ColoredGraph = typename OperatorInfoBase<dim>::ColoredGraph;
    using ExecutionSpace = MemorySpace::Default::kokkos_space::execution_space;

    LaplaceOperatorBK3(const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<number> &constraints,
                       bool                             overlap_communication_computation);

    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    vmult_tensor_core(VectorType &dst, const VectorType &src) const;

    void
    compute_rhs(VectorType &rhs) const;

    void
    vmult_dummy(VectorType       &dst,
                const VectorType &src,
                const bool        ghost_exchange_on,
                const bool        computation_on) const override;

    void
    Tvmult(VectorType &dst, const VectorType &src) const override;

    void
    initialize_dof_vector(VectorType &vec) const override;

    // vec[i] = value on every locally owned constrained DoF
    void
    set_constrained_values(const number value, VectorType &vec) const;

    // dst[c] = src[c] on the owned constrained DoFs; public because nvcc
    // rejects a KOKKOS_LAMBDA inside a private member function
    void
    copy_constrained_values(const VectorType     &src,
                            VectorType           &dst,
                            const ExecutionSpace &exec) const;

    void
    compute_diagonal() override;

    std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    number
    el(const types::global_dof_index row, const types::global_dof_index col) const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    const DoFHandler<dim> &
    get_dof_handler() const override;

    const ColoredGraph &
    get_colored_graph() const override;

    bool
    use_overlap_communication_computation() const override;

  private:
    static constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);
    static constexpr unsigned int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

    // "how many cells' worth of quadrature-point scratch fits in one CUDA
    // team's shared-memory budget" for the tensor-core kernel
    // (bk4_cuda_kernels.cuh), evaluated at compile time.
    static constexpr std::size_t  tensor_core_shmem_per_block = 10'000;
    static constexpr unsigned int tensor_core_nelmt_per_batch =
      std::max<std::size_t>(1, tensor_core_shmem_per_block / (4 * n_q_points) / sizeof(number));

    void
    setup_partitioner_and_colored_graph();

    void
    setup_shape_tables();

    // Dirichlet bookkeeping: is_constrained_local (host, MPI-local index) and
    // owned_constrained_dofs (device). Only homogeneous Dirichlet constraints
    // are supported.
    void
    setup_constrained_dofs();

    void
    setup_dof_indices_per_color();

    void
    compute_G_tensors();

    void
    launch_color(const ExecutionSpace       &exec,
                 const unsigned int          color,
                 const DeviceVector<number> &src_device,
                 DeviceVector<number>       &dst_device) const;

    // owned + ghost entries, what a DeviceVector over vec.get_values() spans
    static unsigned int
    mpi_local_size(const VectorType &vec);

    ObserverPointer<const DoFHandler<dim>>           dof_handler;
    ObserverPointer<const AffineConstraints<number>> constraints;

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    unsigned int                                       n_mpi_local_dofs = 0;
    bool                                               overlap_communication_computation;

    ColoredGraph colored_graph;

    std::vector<std::uint8_t> is_constrained_local;

    // 1-D tables: [dof * n_q_points_1d + q] and [q_dof * n_q_points_1d + q]
    Kokkos::View<number *, MemorySpace::Default::kokkos_space> shape_values;
    Kokkos::View<number *, MemorySpace::Default::kokkos_space> co_shape_gradients;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> owned_constrained_dofs;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dof_indices_per_color;

    // per color: G[(cell * symmetric_tensor_dim + c) * n_q_points + q]
    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>> G_tensors;

    // per color: JxW[cell * n_q_points + q], used by compute_rhs() only
    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>> JxW_per_color;

    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  };



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::LaplaceOperatorBK3(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
    : dof_handler(&dof_handler)
    , constraints(&constraints)
    , overlap_communication_computation(overlap_communication_computation)
  {
    AssertDimension(dof_handler.get_fe().n_dofs_per_cell(), n_local_dofs);
    AssertDimension(dof_handler.get_fe().n_components(), 1);

#ifndef DEAL_II_MPI_WITH_DEVICE_SUPPORT
    // The 3-color scheme posts MPI operations on device vectors between
    // kernels. Without device-aware MPI, only host "device" memory is
    // accepted (as in Portable::MatrixFree); see buffered_ghost_exchange.
    constexpr bool default_memory_is_host =
      std::is_same_v<MemorySpace::Default::kokkos_space, Kokkos::HostSpace>;
    AssertThrow(!overlap_communication_computation || default_memory_is_host,
                ExcMessage(
                  "Overlapping communication and computation requires device-aware MPI."));
#endif

    setup_partitioner_and_colored_graph();
    setup_shape_tables();
    setup_constrained_dofs();
    setup_dof_indices_per_color();
    compute_G_tensors();

    is_constrained_local.clear();
    is_constrained_local.shrink_to_fit();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::setup_partitioner_and_colored_graph()
  {
    const auto &triangulation = dof_handler->get_triangulation();

    const auto *parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(&triangulation);

    if (parallel_triangulation != nullptr)
      partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        dof_handler->locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(*dof_handler),
        parallel_triangulation->get_mpi_communicator());

    const std::size_t n_local = partitioner ? partitioner->locally_owned_size() +
                                                partitioner->n_ghost_indices() :
                                              dof_handler->n_dofs();
    AssertThrow(n_local < static_cast<std::size_t>(numbers::invalid_unsigned_int),
                ExcMessage("The MPI-local number of DoFs exceeds the 32-bit index range."));
    n_mpi_local_dofs = n_local;

    // Same cell order and coloring as deal.II's portable matrix-free class without graph
    // coloring: one color with all locally owned cells, or, for the
    // communication/computation overlap, color 1 with the cells touching a
    // ghost cell and colors 0 and 2 with the first and second half of the
    // interior cells.
    using CellIterator = typename OperatorInfoBase<dim>::CellIterator;

    std::vector<CellIterator> owned_cells;
    owned_cells.reserve(parallel_triangulation ?
                          parallel_triangulation->n_locally_owned_active_cells() :
                          triangulation.n_active_cells());
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        owned_cells.push_back(cell);

    colored_graph.clear();
    if (overlap_communication_computation)
      {
        colored_graph.resize(3);

        std::vector<bool> ghost_vertices(triangulation.n_vertices(), false);
        for (const auto &cell : triangulation.active_cell_iterators())
          if (cell->is_ghost())
            for (const unsigned int v : cell->vertex_indices())
              ghost_vertices[cell->vertex_index(v)] = true;

        std::vector<CellIterator> inner_cells;
        inner_cells.reserve(owned_cells.size());
        for (const auto &cell : owned_cells)
          {
            bool touches_ghost = false;
            for (const unsigned int v : cell->vertex_indices())
              if (ghost_vertices[cell->vertex_index(v)])
                {
                  touches_ghost = true;
                  break;
                }

            if (touches_ghost)
              colored_graph[1].push_back(cell);
            else
              inner_cells.push_back(cell);
          }

        const std::size_t half = inner_cells.size() / 2;
        colored_graph[0].assign(inner_cells.begin(), inner_cells.begin() + half);
        colored_graph[2].assign(inner_cells.begin() + half, inner_cells.end());
      }
    else
      {
        colored_graph.resize(1);
        colored_graph[0] = std::move(owned_cells);
      }
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::setup_shape_tables()
  {
    ExecutionSpace exec;

    const QGauss<1> quadrature_1d(n_q_points_1d);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<number> shape_info(
      quadrature_1d, dof_handler->get_fe());
    const auto &univariate = shape_info.data.front();

    constexpr unsigned int n_dofs_1d = fe_degree + 1;
    AssertDimension(univariate.shape_values.size(), n_dofs_1d * n_q_points_1d);
    AssertDimension(univariate.shape_gradients_collocation.size(), n_q_points_1d * n_q_points_1d);

    shape_values = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
      Kokkos::view_alloc("bk3_shape_values", Kokkos::WithoutInitializing),
      n_dofs_1d * n_q_points_1d);
    co_shape_gradients = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
      Kokkos::view_alloc("bk3_co_shape_gradients", Kokkos::WithoutInitializing),
      n_q_points_1d * n_q_points_1d);

    auto shape_values_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, shape_values);
    auto co_shape_gradients_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, co_shape_gradients);

    for (unsigned int i = 0; i < n_dofs_1d * n_q_points_1d; ++i)
      shape_values_host(i) = univariate.shape_values[i];
    for (unsigned int i = 0; i < n_q_points_1d * n_q_points_1d; ++i)
      co_shape_gradients_host(i) = univariate.shape_gradients_collocation[i];

    Kokkos::deep_copy(exec, shape_values, shape_values_host);
    Kokkos::deep_copy(exec, co_shape_gradients, co_shape_gradients_host);
    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::setup_constrained_dofs()
  {
    ExecutionSpace exec;

    const unsigned int locally_owned_size =
      partitioner ? partitioner->locally_owned_size() : n_mpi_local_dofs;

    is_constrained_local.assign(n_mpi_local_dofs, 0);

    std::vector<unsigned int> owned_constrained_host;
    owned_constrained_host.reserve(constraints->n_constraints());

    for (const auto &line : constraints->get_lines())
      {
        AssertThrow(line.entries.empty() && line.inhomogeneity == number(0),
                    ExcMessage("LaplaceOperatorBK3 only supports homogeneous Dirichlet "
                               "constraints; a hanging-node or inhomogeneous constraint "
                               "was found."));

        unsigned int local_dof = numbers::invalid_unsigned_int;
        if (partitioner)
          {
            if (partitioner->in_local_range(line.index) ||
                partitioner->is_ghost_entry(line.index))
              local_dof = partitioner->global_to_local(line.index);
          }
        else if (line.index < n_mpi_local_dofs)
          local_dof = static_cast<unsigned int>(line.index);

        if (local_dof != numbers::invalid_unsigned_int)
          {
            is_constrained_local[local_dof] = 1;
            if (local_dof < locally_owned_size)
              owned_constrained_host.push_back(local_dof);
          }
      }

    owned_constrained_dofs = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
      Kokkos::view_alloc("bk3_owned_constrained_dofs", Kokkos::WithoutInitializing),
      owned_constrained_host.size());

    if (!owned_constrained_host.empty())
      {
        const Kokkos::View<const unsigned int *,
                           Kokkos::HostSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          owned_constrained_host_view(owned_constrained_host.data(),
                                      owned_constrained_host.size());
        Kokkos::deep_copy(exec, owned_constrained_dofs, owned_constrained_host_view);
        exec.fence();
      }
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::setup_dof_indices_per_color()
  {
    ExecutionSpace exec;

    const unsigned int n_colors = colored_graph.size();

    // lexicographic numbering of the FE_Q basis, independent of the quadrature
    const std::vector<unsigned int> lex_numbering =
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double>(QGauss<1>(1),
                                                               dof_handler->get_fe())
        .lexicographic_numbering;
    AssertDimension(lex_numbering.size(), n_local_dofs);

    dof_indices_per_color.clear();
    dof_indices_per_color.resize(n_colors);

    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const auto        &graph   = colored_graph[color];
        const unsigned int n_cells = graph.size();
        if (n_cells == 0)
          continue;

        dof_indices_per_color[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("bk3_dof_indices_" + std::to_string(color),
                               Kokkos::WithoutInitializing),
            n_local_dofs,
            n_cells);

        auto dof_indices_host =
          Kokkos::create_mirror_view(Kokkos::WithoutInitializing, dof_indices_per_color[color]);

        for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
          {
            graph[cell_id]->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                const types::global_dof_index global_dof = local_dof_indices[lex_numbering[i]];
                const unsigned int            local_dof =
                  partitioner ? partitioner->global_to_local(global_dof) :
                                static_cast<unsigned int>(global_dof);

                dof_indices_host(i, cell_id) =
                  is_constrained_local[local_dof] ? numbers::invalid_unsigned_int : local_dof;
              }
          }

        Kokkos::deep_copy(exec, dof_indices_per_color[color], dof_indices_host);
      }

    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::compute_G_tensors()
  {
    ExecutionSpace exec;

    const unsigned int n_colors = colored_graph.size();

    G_tensors.clear();
    G_tensors.resize(n_colors);
    JxW_per_color.clear();
    JxW_per_color.resize(n_colors);

    // Same mapping the matrix-free object was given before: the geometry is
    // represented with the polynomial degree of the finite element.
    const MappingQ<dim> mapping(fe_degree);
    const QGauss<dim>   quadrature(n_q_points_1d);
    FEValues<dim>       fe_values(mapping,
                            dof_handler->get_fe(),
                            quadrature,
                            update_inverse_jacobians | update_JxW_values);
    fe_values.always_allow_check_for_cell_similarity(true);

    // The geometry is streamed in tiles so that the host staging buffer and
    // the device transfer never hold more than max_tile_cells cells at once.
    constexpr unsigned int max_tile_cells = 4096;

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const auto        &graph   = colored_graph[color];
        const unsigned int n_cells = graph.size();
        if (n_cells == 0)
          continue;

        const std::size_t g_per_cell = static_cast<std::size_t>(symmetric_tensor_dim) * n_q_points;

        G_tensors[color] = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("bk3_G_tensor_color_" + std::to_string(color),
                             Kokkos::WithoutInitializing),
          g_per_cell * n_cells);
        JxW_per_color[color] = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("bk3_JxW_color_" + std::to_string(color),
                             Kokkos::WithoutInitializing),
          static_cast<std::size_t>(n_q_points) * n_cells);

        const unsigned int tile_cells = std::min(n_cells, max_tile_cells);

        Kokkos::View<number *, Kokkos::HostSpace> G_tile_host(
          Kokkos::view_alloc("bk3_G_tile_host", Kokkos::WithoutInitializing),
          g_per_cell * tile_cells);
        Kokkos::View<number *, Kokkos::HostSpace> JxW_tile_host(
          Kokkos::view_alloc("bk3_JxW_tile_host", Kokkos::WithoutInitializing),
          static_cast<std::size_t>(n_q_points) * tile_cells);

        for (unsigned int cell_offset = 0; cell_offset < n_cells; cell_offset += tile_cells)
          {
            const unsigned int current_tile = std::min(tile_cells, n_cells - cell_offset);

            for (unsigned int t = 0; t < current_tile; ++t)
              {
                fe_values.reinit(graph[cell_offset + t]);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    // Cast first and contract in `number`, exactly like the
                    // former device kernel did on the matrix-free data.
                    const number JxW = static_cast<number>(fe_values.JxW(q));

                    number inverse_jacobian[dim][dim];
                    const auto &ij = fe_values.inverse_jacobian(q);
                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                      for (unsigned int d2 = 0; d2 < dim; ++d2)
                        inverse_jacobian[d1][d2] = static_cast<number>(ij[d1][d2]);

                    JxW_tile_host(static_cast<std::size_t>(t) * n_q_points + q) = JxW;

                    // G_{d1 d2} = JxW * sum_k (J^{-1})_{d1 k} (J^{-1})_{d2 k}:
                    // contraction over the physical (last) index.
                    unsigned int c = 0;
                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                      for (unsigned int d2 = d1; d2 < dim; ++d2)
                        {
                          number sum = 0;
                          for (unsigned int k = 0; k < dim; ++k)
                            sum += inverse_jacobian[d1][k] * inverse_jacobian[d2][k];
                          G_tile_host(static_cast<std::size_t>(t) * g_per_cell +
                                      static_cast<std::size_t>(c) * n_q_points + q) = JxW * sum;
                          ++c;
                        }
                  }
              }

            const auto G_device_range =
              Kokkos::subview(G_tensors[color],
                              std::make_pair(g_per_cell * cell_offset,
                                             g_per_cell * (cell_offset + current_tile)));
            const auto G_host_range =
              Kokkos::subview(G_tile_host, std::make_pair(std::size_t(0), g_per_cell * current_tile));
            Kokkos::deep_copy(exec, G_device_range, G_host_range);

            const auto JxW_device_range = Kokkos::subview(
              JxW_per_color[color],
              std::make_pair(static_cast<std::size_t>(n_q_points) * cell_offset,
                             static_cast<std::size_t>(n_q_points) * (cell_offset + current_tile)));
            const auto JxW_host_range = Kokkos::subview(
              JxW_tile_host,
              std::make_pair(std::size_t(0), static_cast<std::size_t>(n_q_points) * current_tile));
            Kokkos::deep_copy(exec, JxW_device_range, JxW_host_range);

            // the host tiles are reused by the next iteration
            exec.fence();
          }
      }
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  unsigned int
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::mpi_local_size(const VectorType &vec)
  {
    return vec.locally_owned_size() +
           (vec.get_partitioner() ? vec.get_partitioner()->n_ghost_indices() : 0u);
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::copy_constrained_values(
    const VectorType     &src,
    VectorType           &dst,
    const ExecutionSpace &exec) const
  {
    const unsigned int n_constrained = owned_constrained_dofs.extent(0);
    if (n_constrained == 0)
      return;

    const auto    constrained_dofs = owned_constrained_dofs;
    const number *src_ptr          = src.get_values();
    number       *dst_ptr          = dst.get_values();

    Kokkos::parallel_for(
      "bk3_copy_constrained_values",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_constrained),
      KOKKOS_LAMBDA(const unsigned int i) {
        const unsigned int dof = constrained_dofs(i);
        dst_ptr[dof]           = src_ptr[dof];
      });
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::launch_color(
    const ExecutionSpace       &exec,
    const unsigned int          color,
    const DeviceVector<number> &src_device,
    DeviceVector<number>       &dst_device) const
  {
    const unsigned int n_cells = colored_graph[color].size();
    if (n_cells == 0)
      return;

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    const unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
    const unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

    BK3::Parallel::KokkosKernelAbstracted<dim, fe_degree, n_q_points_1d, number>(
      exec,
      shape_values,
      co_shape_gradients,
      G_tensors[color],
      src_device,
      dst_device,
      dof_indices_per_color[color],
      n_cells,
      numBlocks,
      threadsPerBlock);
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::vmult(VectorType       &dst,
                                                                    const VectorType &src) const
  {
    ExecutionSpace exec;

    if (dst.has_ghost_elements())
      dst.zero_out_ghost_values();

    DeviceVector<number> src_device(src.get_values(), mpi_local_size(src)),
      dst_device(dst.get_values(), mpi_local_size(dst));

    Kokkos::deep_copy(exec, dst_device, number(0));

    const unsigned int n_colors = colored_graph.size();

    const bool needs_ghost_exchange =
      partitioner && (partitioner->n_ghost_indices() > 0 || partitioner->n_import_indices() > 0);

    if (!needs_ghost_exchange)
      {
        // single rank, or no DoF shared with another rank: no MPI at all
        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color(exec, color, src_device, dst_device);

        copy_constrained_values(src, dst, exec);
        exec.fence();
        return;
      }

    if (overlap_communication_computation)
      {
        src.update_ghost_values_start(0);

        // colors 0 and 2 touch no ghost DoFs; in parallel, some processors
        // may not own any cells
        if (n_colors > 0)
          launch_color(exec, 0, src_device, dst_device);
        if (buffered_ghost_exchange && n_colors > 2)
          launch_color(exec, 2, src_device, dst_device);

        src.update_ghost_values_finish();

        if (n_colors > 1 && colored_graph[1].size() > 0)
          {
            launch_color(exec, 1, src_device, dst_device);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            exec.fence();
          }

        dst.compress_start(0, VectorOperation::add);
        if (!buffered_ghost_exchange && n_colors > 2)
          launch_color(exec, 2, src_device, dst_device);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color(exec, color, src_device, dst_device);

        exec.fence();
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
    copy_constrained_values(src, dst, exec);
    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::vmult_tensor_core(
    VectorType       &dst,
    const VectorType &src) const
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
    ExecutionSpace exec;

    dst = 0.;

    const unsigned int n_colors = colored_graph.size();

    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();
        if (n_cells == 0)
          continue;

        // launch_f64_m8n8k4_mma() takes dof_indices per component
        // (Kokkos::Array<..., n_components>) -- wrap this class's own
        // single (n_components == 1) dof_indices_per_color[color] view.
        const Kokkos::Array<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>, 1>
          dof_indices_single_component{dof_indices_per_color[color]};

        BK4::Parallel::TensorCore::
          launch_f64_m8n8k4_mma<n_q_points_1d, fe_degree + 1, tensor_core_nelmt_per_batch, 1>(
            n_cells,
            shape_values.data(),
            co_shape_gradients.data(),
            G_tensors[color].data(),
            src.get_values(),
            dst.get_values(),
            dof_indices_single_component);
      }

    Kokkos::fence();

    dst.compress(VectorOperation::add);

    src.zero_out_ghost_values();
    copy_constrained_values(src, dst, exec);
    exec.fence();
#endif
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::compute_rhs(VectorType &rhs) const
  {
    ExecutionSpace exec;

    if (rhs.has_ghost_elements())
      rhs.zero_out_ghost_values();

    DeviceVector<number> rhs_device(rhs.get_values(), mpi_local_size(rhs));

    Kokkos::deep_copy(exec, rhs_device, number(0));

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    const unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
    const unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

    const unsigned int n_colors = colored_graph.size();
    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();
        if (n_cells == 0)
          continue;

        BK3::Parallel::KokkosRHSAbstracted<dim, fe_degree, n_q_points_1d, number>(
          exec,
          shape_values,
          JxW_per_color[color],
          rhs_device,
          dof_indices_per_color[color],
          n_cells,
          numBlocks,
          threadsPerBlock);
      }

    exec.fence();
    rhs.compress(VectorOperation::add);
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::vmult_dummy(
    VectorType       &dst,
    const VectorType &src,
    const bool        ghost_exchange_on,
    const bool        computation_on) const
  {
    ExecutionSpace exec;

    DeviceVector<number> src_device(src.get_values(), mpi_local_size(src)),
      dst_device(dst.get_values(), mpi_local_size(dst));

    if (computation_on)
      {
        if (dst.has_ghost_elements())
          dst.zero_out_ghost_values();
        Kokkos::deep_copy(exec, dst_device, number(0));
      }

    const unsigned int n_colors = colored_graph.size();

    if (overlap_communication_computation)
      {
        if (ghost_exchange_on)
          src.update_ghost_values_start(0);

        if (computation_on && n_colors > 0)
          launch_color(exec, 0, src_device, dst_device);
        if (computation_on && buffered_ghost_exchange && n_colors > 2)
          launch_color(exec, 2, src_device, dst_device);

        if (ghost_exchange_on)
          src.update_ghost_values_finish();

        if (n_colors > 1 && colored_graph[1].size() > 0)
          {
            if (computation_on)
              launch_color(exec, 1, src_device, dst_device);

            // see vmult()
            exec.fence();
          }

        if (ghost_exchange_on)
          dst.compress_start(0, VectorOperation::add);

        if (computation_on && !buffered_ghost_exchange && n_colors > 2)
          launch_color(exec, 2, src_device, dst_device);

        if (ghost_exchange_on)
          dst.compress_finish(VectorOperation::add);
      }
    else
      {
        if (ghost_exchange_on)
          src.update_ghost_values();

        if (computation_on)
          for (unsigned int color = 0; color < n_colors; ++color)
            launch_color(exec, color, src_device, dst_device);

        if (ghost_exchange_on)
          {
            exec.fence();
            dst.compress(VectorOperation::add);
          }
      }

    if (ghost_exchange_on)
      src.zero_out_ghost_values();

    if (computation_on)
      copy_constrained_values(src, dst, exec);

    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::Tvmult(VectorType       &dst,
                                                                     const VectorType &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(partitioners_are_compatible(dst.get_partitioner(), partitioner),
           ExcMessage("Vector is not correctly initialized."));
    Assert(partitioners_are_compatible(src.get_partitioner(), partitioner),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::initialize_dof_vector(
    VectorType &vec) const
  {
    if (partitioner)
      vec.reinit(partitioner);
    else
      vec.reinit(dof_handler->n_dofs());
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::set_constrained_values(
    const number value,
    VectorType  &vec) const
  {
    const unsigned int n_constrained = owned_constrained_dofs.extent(0);
    if (n_constrained == 0)
      return;

    ExecutionSpace exec;

    const auto constrained_dofs = owned_constrained_dofs;
    number    *vec_ptr          = vec.get_values();

    Kokkos::parallel_for(
      "bk3_set_constrained_values",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_constrained),
      KOKKOS_LAMBDA(const unsigned int i) { vec_ptr[constrained_dofs(i)] = value; });
    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::compute_diagonal()
  {
    ExecutionSpace exec;

    inverse_diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    // Accumulate diag(A) over the owned and ghost entries, then compress.
    DeviceVector<number> diagonal_device(inverse_diagonal.get_values(),
                                         mpi_local_size(inverse_diagonal));
    Kokkos::deep_copy(exec, diagonal_device, number(0));

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    const unsigned int numBlocks       = is_serial ? 1u : numbers::invalid_unsigned_int;
    const unsigned int threadsPerBlock = is_serial ? 1u : numbers::invalid_unsigned_int;

    const unsigned int n_colors = colored_graph.size();
    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();
        if (n_cells == 0)
          continue;

        BK3::Parallel::KokkosDiagonalAbstracted<dim, fe_degree, n_q_points_1d, number>(
          exec,
          shape_values,
          co_shape_gradients,
          G_tensors[color],
          diagonal_device,
          dof_indices_per_color[color],
          n_cells,
          numBlocks,
          threadsPerBlock);
      }

    exec.fence();
    inverse_diagonal.compress(VectorOperation::add);

    // constrained rows are identity rows
    set_constrained_values(number(1), inverse_diagonal);

    number *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      "bk3_invert_diagonal",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, inverse_diagonal.locally_owned_size()),
      KOKKOS_LAMBDA(const unsigned int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
    exec.fence();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  types::global_dof_index
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::m() const
  {
    return partitioner ? partitioner->size() : dof_handler->n_dofs();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  types::global_dof_index
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::n() const
  {
    return m();
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  number
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::get_vector_partitioner() const
  {
    return partitioner;
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  const DoFHandler<dim> &
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::get_dof_handler() const
  {
    return *dof_handler;
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  const typename LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::ColoredGraph &
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::get_colored_graph() const
  {
    return colored_graph;
  }



  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  bool
  LaplaceOperatorBK3<dim, fe_degree, number, n_q_points_1d>::use_overlap_communication_computation()
    const
  {
    return overlap_communication_computation;
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
