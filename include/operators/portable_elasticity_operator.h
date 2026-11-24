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
#include <deal.II/matrix_free/tools.h>

#include <memory>
#include <string>

#include "base/portable_laplace_operator_base.h"
#include "base/portable_operator_info_base.h"
#include "kernels/portable_local_elasticity_operator.h"
#include "matrix_free_dealii/portable_batched_cell_loop.h"
#include "matrix_free_dealii/portable_batched_fe_evaluation.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
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

    MatrixFree<dim, number> matrix_free;

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
