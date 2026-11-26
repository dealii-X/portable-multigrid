#ifndef portable_elasticity_operator_h
#define portable_elasticity_operator_h

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tools.h>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "base/portable_laplace_operator_base.h"
#include "base/portable_operator_info_base.h"
#include "kernels/elasticity_benchmark_kernel.h"
#include "kernels/portable_local_elasticity_operator.h"
#include "matrix_free_dealii/portable_batched_cell_loop.h"
#include "matrix_free_dealii/portable_batched_fe_evaluation.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {
    // Access of Benchmark::LinearElasticityKernel to a global vector through
    // the local_to_global map of one color: node (i, j, k) is the
    // lexicographic DoF i + nm (j + nm k) of the component, Dirichlet DoFs
    // read as zero and are not written, the elements of the padded last
    // batch are skipped.
    template <typename number, unsigned int nm>
    struct BenchmarkGlobalAccess
    {
      Kokkos::View<types::global_dof_index **, MemorySpace::Default::kokkos_space> local_to_global;
      const number                                                                *src;
      number                                                                      *dst;
      std::size_t                                                                  n_cells;
      bool                                                                         use_coloring;

      KOKKOS_INLINE_FUNCTION types::global_dof_index
      index(const int c, const std::size_t cell, const int i, const int j, const int k) const
      {
        return local_to_global(i + nm * (j + nm * k) + nm * nm * nm * c, cell);
      }

      KOKKOS_INLINE_FUNCTION number
      read(const int c, const std::size_t cell, const int i, const int j, const int k) const
      {
        if (cell >= n_cells)
          return 0;
        const types::global_dof_index dof = index(c, cell, i, j, k);
        return (dof == numbers::invalid_unsigned_int) ? number(0) : src[dof];
      }

      KOKKOS_INLINE_FUNCTION void
      write(const int         c,
            const std::size_t cell,
            const int         i,
            const int         j,
            const int         k,
            const number      value) const
      {
        if (cell >= n_cells)
          return;
        const types::global_dof_index dof = index(c, cell, i, j, k);
        if (dof == numbers::invalid_unsigned_int)
          return;
        if (use_coloring)
          dst[dof] += value;
        else
          Kokkos::atomic_add(&dst[dof], value);
      }
    };



    // The geometry of the cells of one color in the layout of
    // Benchmark::LinearElasticityKernel: [cell][entry][a][b][c] with entry 0 =
    // JxW and 1 + 3 d + e = d xi_d / d x_e, x along a (the point
    // a + nq (b + nq c) of deal.II), zero for the padding.
    template <int dim, typename number, int nq>
    Kokkos::View<number *, MemorySpace::Default::kokkos_space>
    benchmark_geometry(const typename MatrixFree<dim, number>::PrecomputedData &data,
                       const std::size_t                                        n_cells_padded)
    {
      constexpr int n_points = nq * nq * nq;

      Kokkos::View<number *, MemorySpace::Default::kokkos_space> G("benchmark_geometry",
                                                                   n_cells_padded * 10 * n_points);
      Kokkos::parallel_for(
        "ElasticityOperator benchmark geometry",
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0,
                                                                                 data.n_cells *
                                                                                   n_points),
        KOKKOS_LAMBDA(const int index) {
          const int cell = index / n_points;
          const int q    = index % n_points;
          const int a    = q % nq;
          const int b    = (q / nq) % nq;
          const int c    = q / (nq * nq);

          const std::size_t base =
            static_cast<std::size_t>(cell) * 10 * n_points + (a * nq + b) * nq + c;
          G(base) = data.JxW_value(cell, q);

          const unsigned int inv_jac_idx = data.inv_jacobian_index(cell, q);
          const bool         cartesian =
            data.cell_type(cell) == ::dealii::internal::MatrixFreeFunctions::cartesian;
          for (int d = 0; d < dim; ++d)
            for (int e = 0; e < dim; ++e)
              G(base + (1 + 3 * d + e) * n_points) =
                (cartesian && d != e) ? number(0) : data.inv_jacobian(inv_jac_idx, d, e);
        });
      return G;
    }
  } // namespace internal



  // (grad v, material.stress(grad u)) for a vector-valued FE_Q system with
  // dim components. vmult() uses the batched FEEvaluation, vmult_dealii()
  // the cell-wise one of deal.II; both run the same LocalElasticityOperator.
  template <int dim,
            int fe_degree,
            typename number,
            typename Material = LinearElasticMaterial<dim, number>,
            int n_q_points_1d = fe_degree + 1>
  class ElasticityOperator
    : public LaplaceOperatorBase<dim,
                                 LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

    ElasticityOperator(const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<number> &constraints,
                       const Material                  &material,
                       const bool                       overlap_communication_computation);

    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    vmult_dealii(VectorType &dst, const VectorType &src) const;

    void
    vmult_batched(VectorType &dst, const VectorType &src) const;

    // The kernel of the dealii-X benchmarks (kernels/elasticity_benchmark_kernel.h)
    // on the geometry of matrix_free, with gather and scatter on the global
    // vectors: dim = 3, LinearElasticMaterial, no hanging nodes.
    void
    vmult_benchmark(VectorType &dst, const VectorType &src) const;

    // vmult_batched() with the stress computed inside evaluate_and_apply()
    void
    vmult_fused(VectorType &dst, const VectorType &src) const;

    void
    vmult_dummy(VectorType       &dst,
                const VectorType &src,
                const bool        ghost_exchange_on,
                const bool        computation_on) const override;

    void
    Tvmult(VectorType &dst, const VectorType &src) const override;

    void
    initialize_dof_vector(VectorType &vec) const override;

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

    const MatrixFree<dim, number> &
    get_matrix_free() const;

    const Material &
    get_material() const;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    const DoFHandler<dim> &
    get_dof_handler() const override;

    const typename OperatorInfoBase<dim>::ColoredGraph &
    get_colored_graph() const override;

    bool
    use_overlap_communication_computation() const override;

  private:
    using BatchedFEEvaluation =
      Copy::Portable::internal::batched::FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number>;
    using BatchedLocalOperator =
      LocalElasticityOperator<dim, fe_degree, n_q_points_1d, number, Material, BatchedFEEvaluation>;
    using FusedLocalOperator = LocalElasticityOperator<dim,
                                                       fe_degree,
                                                       n_q_points_1d,
                                                       number,
                                                       Material,
                                                       BatchedFEEvaluation,
                                                       true>;
    using CellLocalOperator =
      LocalElasticityOperator<dim,
                              fe_degree,
                              n_q_points_1d,
                              number,
                              Material,
                              FEEvaluation<dim, fe_degree, n_q_points_1d, dim, number>>;

    // owned + ghost entries, what a DeviceVector over vec.get_values() spans
    static unsigned int
    mpi_local_size(const VectorType &vec)
    {
      return vec.locally_owned_size() +
             (vec.get_partitioner() ? vec.get_partitioner()->n_ghost_indices() : 0);
    }

    // batched cell loop over one color for vmult_dummy(), no fence
    void
    launch_color(const unsigned int          color,
                 const DeviceVector<number> &src,
                 const DeviceVector<number> &dst) const;

    using BenchmarkKernel = Benchmark::LinearElasticityKernel<number, fe_degree + 1, n_q_points_1d>;

    static constexpr bool has_benchmark_kernel =
      dim == 3 && std::is_same_v<Material, LinearElasticMaterial<dim, number>>;

    MatrixFree<dim, number> matrix_free;

    // per color, for vmult_benchmark()
    std::vector<BenchmarkKernel>                                            benchmark_kernels;
    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>> benchmark_geometry;

    // matrix_free's colored graph without the FilteredIterator wrapper
    typename OperatorInfoBase<dim>::ColoredGraph colored_graph;

    Material material;

    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  };



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::ElasticityOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const Material                  &material,
    const bool                       overlap_communication_computation)
    : material(material)
  {
    AssertDimension(dof_handler.get_fe().n_components(), dim);

    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;
    additional_data.mapping_update_flags              = update_gradients | update_JxW_values;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    matrix_free.reinit(
      mapping, dof_handler, constraints, QGauss<1>(n_q_points_1d), additional_data);
    colored_graph = make_colored_graph<dim>(matrix_free.get_colored_graph());

    if constexpr (has_benchmark_kernel)
      {
        // 1D shape values of FE_Q and collocation derivatives on the Gauss points
        const dealii::internal::MatrixFreeFunctions::ShapeInfo<number> shape_info(
          QGauss<1>{n_q_points_1d}, dof_handler.get_fe().base_element(0));
        const auto               &univariate = shape_info.data.front();
        const std::vector<number> basis(univariate.shape_values.begin(),
                                        univariate.shape_values.end());
        const std::vector<number> dbasis(univariate.shape_gradients_collocation.begin(),
                                         univariate.shape_gradients_collocation.end());

        bool has_hanging_nodes = false;
        for (unsigned int color = 0; color < colored_graph.size(); ++color)
          {
            const auto data = matrix_free.get_data(color);

            benchmark_kernels.emplace_back(data.n_cells,
                                           BenchmarkKernel::default_elements_per_batch(),
                                           basis,
                                           dbasis,
                                           material.lambda,
                                           material.mu);
            benchmark_geometry.push_back(internal::benchmark_geometry<dim, number, n_q_points_1d>(
              data, benchmark_kernels.back().n_elements_padded()));

            const auto masks =
              Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.constraint_mask);
            for (unsigned int i = 0; i < masks.size(); ++i)
              if (masks(i) != dealii::internal::MatrixFreeFunctions::ConstraintKinds::unconstrained)
                has_hanging_nodes = true;
          }
        AssertThrow(!has_hanging_nodes,
                    ExcMessage("vmult_benchmark() does not support hanging nodes"));
      }
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    vmult_batched(dst, src);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult_dealii(
    VectorType       &dst,
    const VectorType &src) const
  {
    dst = 0.;

    const CellLocalOperator cell_operator(material);
    matrix_free.cell_loop(cell_operator, src, dst);

    matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::launch_color(
    const unsigned int          color,
    const DeviceVector<number> &src,
    const DeviceVector<number> &dst) const
  {
    namespace batched = Copy::Portable::internal::batched;

    if (colored_graph[color].size() == 0)
      return;

    MemorySpace::Default::kokkos_space::execution_space exec;

    const unsigned int n_dof_handler = 1;
    Kokkos::Array<typename MatrixFree<dim, number>::PrecomputedData, n_max_dof_handlers>
      colored_data;
    colored_data[0] = matrix_free.get_data(color);

    const int batch_size =
      batched::compute_batch_size<dim, number>(BatchedLocalOperator::n_q_points,
                                               n_dof_handler,
                                               colored_data);

    const batched::ApplyKernel<dim, number, BatchedLocalOperator> apply_kernel{
      BatchedLocalOperator(material), n_dof_handler, colored_data, batch_size, src, dst};

    Kokkos::parallel_for("ElasticityOperator color " + std::to_string(color),
                         batched::make_team_policy<dim, number>(exec,
                                                                apply_kernel,
                                                                BatchedLocalOperator::n_q_points,
                                                                batch_size,
                                                                n_dof_handler,
                                                                colored_data),
                         apply_kernel);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult_batched(
    VectorType       &dst,
    const VectorType &src) const
  {
    dst = 0.;

    Copy::Portable::internal::batched::cell_loop(matrix_free,
                                                 BatchedLocalOperator(material),
                                                 src,
                                                 dst);

    matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult_benchmark(
    VectorType       &dst,
    const VectorType &src) const
  {
    static_assert(has_benchmark_kernel,
                  "vmult_benchmark() needs dim == 3 and LinearElasticMaterial");

    dst = 0.;
    src.update_ghost_values();

    for (unsigned int color = 0; color < benchmark_kernels.size(); ++color)
      {
        const auto data = matrix_free.get_data(color);
        if (data.n_cells == 0)
          continue;

        const internal::BenchmarkGlobalAccess<number, fe_degree + 1> access{data.local_to_global,
                                                                            src.get_values(),
                                                                            dst.get_values(),
                                                                            data.n_cells,
                                                                            data.use_coloring};
        benchmark_kernels[color].apply(benchmark_geometry[color],
                                       access,
                                       benchmark_kernels[color].n_batches());
      }
    Kokkos::fence();

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult_fused(
    VectorType       &dst,
    const VectorType &src) const
  {
    dst = 0.;

    Copy::Portable::internal::batched::cell_loop(matrix_free,
                                                 FusedLocalOperator(material),
                                                 src,
                                                 dst);

    matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::vmult_dummy(
    VectorType       &dst,
    const VectorType &src,
    const bool        ghost_exchange_on,
    const bool        computation_on) const
  {
    dst = 0.;

    const DeviceVector<number> src_device(src.get_values(), mpi_local_size(src));
    const DeviceVector<number> dst_device(dst.get_values(), mpi_local_size(dst));

    const unsigned int n_colors = colored_graph.size();

    if (matrix_free.use_overlap_communication_computation())
      {
        if (ghost_exchange_on)
          src.update_ghost_values_start(0);

        // colors 0 and 2 touch no ghost DoFs
        if (computation_on && n_colors > 0)
          launch_color(0, src_device, dst_device);

        if (ghost_exchange_on)
          src.update_ghost_values_finish();

        if (computation_on && n_colors > 1)
          launch_color(1, src_device, dst_device);

        // no MPI communication of dst before the kernels are done
        Kokkos::fence();

        if (ghost_exchange_on)
          dst.compress_start(0, VectorOperation::add);
        if (computation_on && n_colors > 2)
          launch_color(2, src_device, dst_device);
        if (ghost_exchange_on)
          dst.compress_finish(VectorOperation::add);
      }
    else
      {
        if (ghost_exchange_on)
          src.update_ghost_values();

        if (computation_on)
          for (unsigned int color = 0; color < n_colors; ++color)
            launch_color(color, src_device, dst_device);

        Kokkos::fence();

        if (ghost_exchange_on)
          dst.compress(VectorOperation::add);
      }

    Kokkos::fence();

    if (ghost_exchange_on)
      src.zero_out_ghost_values();

    matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::Tvmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    // symmetric for a hyperelastic material
    vmult(dst, src);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::initialize_dof_vector(
    VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  void
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::compute_diagonal()
  {
    inverse_diagonal_entries     = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    const ElasticityOperatorQuad<dim, fe_degree, n_q_points_1d, number, Material> operator_quad(
      material);

    MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, dim, number>(
      matrix_free,
      inverse_diagonal,
      operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    number *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      "ElasticityOperator invert diagonal",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, inverse_diagonal.locally_owned_size()),
      KOKKOS_LAMBDA(const int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
    Kokkos::fence();
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  std::shared_ptr<DiagonalMatrix<
    typename ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::VectorType>>
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_matrix_diagonal_inverse()
    const
  {
    return inverse_diagonal_entries;
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  types::global_dof_index
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::m() const
  {
    const auto &partitioner = matrix_free.get_vector_partitioner();
    return partitioner ? partitioner->size() : matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  types::global_dof_index
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::n() const
  {
    return m();
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  number
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  const MatrixFree<dim, number> &
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_matrix_free() const
  {
    return matrix_free;
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  const Material &
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_material() const
  {
    return material;
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_vector_partitioner()
    const
  {
    return matrix_free.get_vector_partitioner();
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  const DoFHandler<dim> &
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_dof_handler() const
  {
    return matrix_free.get_dof_handler();
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  const typename OperatorInfoBase<dim>::ColoredGraph &
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::get_colored_graph() const
  {
    return colored_graph;
  }



  template <int dim, int fe_degree, typename number, typename Material, int n_q_points_1d>
  bool
  ElasticityOperator<dim, fe_degree, number, Material, n_q_points_1d>::
    use_overlap_communication_computation() const
  {
    return matrix_free.use_overlap_communication_computation();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
