#ifndef cpu_laplace_operator_h
#define cpu_laplace_operator_h

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <memory>

#include "cpu_laplace_operator_base.h"

DEAL_II_NAMESPACE_OPEN

namespace CPU
{
  // Matrix-free Laplace operator on deal.II's CPU MatrixFree/FEEvaluation (step-37 style).
  template <int dim, int fe_degree, typename number, int n_q_points_1d = fe_degree + 1>
  class LaplaceOperator
    : public Portable::
        LaplaceOperatorBase<dim, LinearAlgebra::distributed::Vector<number, MemorySpace::Host>>
  {
  public:
    using VectorType       = LinearAlgebra::distributed::Vector<number, MemorySpace::Host>;
    using FECellIntegrator = FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number>;

    LaplaceOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<number> &constraints,
                    bool                             overlap_communication_computation);

    void
    vmult(VectorType &dst, const VectorType &src) const override;

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
    get_matrix_free() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

  private:
    void
    local_apply(const MatrixFree<dim, number>               &data,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void
    do_cell_integral(FECellIntegrator &phi) const;

    void
    copy_constrained_values(const VectorType &src, VectorType &dst) const;

    MatrixFree<dim, number> matrix_free;

    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  };

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::LaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;

    additional_data.tasks_parallel_scheme = MatrixFree<dim, number>::AdditionalData::none;
    additional_data.mapping_update_flags  = update_gradients | update_JxW_values;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    const QGauss<1> quadrature_1d(n_q_points_1d);
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::do_cell_integral(
    FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::gradients);
    for (const unsigned int q : phi.quadrature_point_indices())
      phi.submit_gradient(phi.get_gradient(q), q);
    phi.integrate(EvaluationFlags::gradients);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::local_apply(
    const MatrixFree<dim, number>               &data,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        do_cell_integral(phi);
        phi.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::copy_constrained_values(
    const VectorType &src,
    VectorType       &dst) const
  {
    for (const unsigned int i : matrix_free.get_constrained_dofs())
      dst.local_element(i) = src.local_element(i);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::vmult(VectorType       &dst,
                                                                const VectorType &src) const
  {
    // ghost exchange (overlapped if enabled) and zeroing dst are done by cell_loop
    matrix_free.cell_loop(&LaplaceOperator::local_apply, this, dst, src, true);

    copy_constrained_values(src, dst);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::vmult_dummy(
    VectorType       &dst,
    const VectorType &src,
    const bool        ghost_exchange_on,
    const bool        computation_on) const
  {
    // without exchange, ghosts are only marked valid so stale values can be read
    if (ghost_exchange_on)
      src.update_ghost_values();
    else
      src.set_ghost_state(true);

    if (computation_on)
      {
        dst = 0.;
        local_apply(matrix_free, dst, src, {0, matrix_free.n_cell_batches()});
      }

    if (ghost_exchange_on)
      {
        dst.compress(VectorOperation::add);
        src.zero_out_ghost_values();
        copy_constrained_values(src, dst);
      }
    else
      src.set_ghost_state(false);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::Tvmult(VectorType       &dst,
                                                                 const VectorType &src) const
  {
    vmult(dst, src);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::initialize_dof_vector(
    VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  void
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::compute_diagonal()
  {
    inverse_diagonal_entries     = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    MatrixFreeTools::compute_diagonal(matrix_free,
                                      inverse_diagonal,
                                      &LaplaceOperator::do_cell_integral,
                                      this);

    for (const unsigned int i : matrix_free.get_constrained_dofs())
      inverse_diagonal.local_element(i) = 1.;

    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
      }
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<number, MemorySpace::Host>>>
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  number
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::el(
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
  const MatrixFree<dim, number> &
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, typename number, int n_q_points_1d>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperator<dim, fe_degree, number, n_q_points_1d>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace CPU

DEAL_II_NAMESPACE_CLOSE

#endif
