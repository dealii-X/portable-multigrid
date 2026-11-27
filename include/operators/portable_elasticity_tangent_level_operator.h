#ifndef portable_elasticity_tangent_level_operator_h
#define portable_elasticity_tangent_level_operator_h

#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "operators/portable_elasticity_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // The operator of ElasticityOperator linearized at the u of the last
  // linearize() call, as a multigrid level operator: with prescribed DoFs
  // (indicator S, D = 1 - S), which the constraints of the operator do not
  // contain, D A D + S, and the diagonal D diag(A) + S.
  template <int dim, int fe_degree, typename number, typename Material>
  class ElasticityTangentLevelOperator
    : public LaplaceOperatorBase<dim,
                                 LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>
  {
  public:
    using VectorType   = LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;
    using Operator     = ElasticityOperator<dim, fe_degree, number, Material>;
    using ColoredGraph = typename OperatorInfoBase<dim>::ColoredGraph;

    // prescribed: locally owned DoFs with values set by the caller
    ElasticityTangentLevelOperator(const DoFHandler<dim>           &dof_handler,
                                   const AffineConstraints<number> &constraints,
                                   const Material                  &material,
                                   const IndexSet                  &prescribed);

    // stores grad u for the linearized operator; residual is R(u)
    void
    linearize(VectorType &residual, const VectorType &u);

    void
    linearize(const VectorType &u);

    Operator &
    get_operator();

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

    const DoFHandler<dim> &
    get_dof_handler() const override;

    const ColoredGraph &
    get_colored_graph() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    bool
    use_overlap_communication_computation() const override;

  private:
    Operator           op;
    bool               has_prescribed;
    VectorType         prescribed_mask, free_mask;
    mutable VectorType tmp;

    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
  };



  // u_coarse = the nodal interpolation of the finite element function u_fine
  // into the coarse space (the state the coarse level is linearized at)
  template <int dim, typename number>
  void
  interpolate_state(const DoFHandler<dim> &dof_handler_fine,
                    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &u_fine,
                    const DoFHandler<dim>           &dof_handler_coarse,
                    const AffineConstraints<number> &constraints_coarse,
                    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &u_coarse)
  {
    using HostVector = LinearAlgebra::distributed::Vector<number, MemorySpace::Host>;

    const MPI_Comm comm = dof_handler_fine.get_mpi_communicator();

    HostVector                             fine_host(dof_handler_fine.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler_fine),
                         comm);
    LinearAlgebra::ReadWriteVector<number> rw_fine(dof_handler_fine.locally_owned_dofs());
    rw_fine.import_elements(u_fine, VectorOperation::insert);
    fine_host.import_elements(rw_fine, VectorOperation::insert);
    fine_host.update_ghost_values();

    HostVector coarse_host(dof_handler_coarse.locally_owned_dofs(),
                           DoFTools::extract_locally_relevant_dofs(dof_handler_coarse),
                           comm);
    FETools::interpolate(
      dof_handler_fine, fine_host, dof_handler_coarse, constraints_coarse, coarse_host);

    LinearAlgebra::ReadWriteVector<number> rw_coarse(dof_handler_coarse.locally_owned_dofs());
    rw_coarse.import_elements(coarse_host, VectorOperation::insert);
    u_coarse.import_elements(rw_coarse, VectorOperation::insert);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::ElasticityTangentLevelOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const Material                  &material,
    const IndexSet                  &prescribed)
    : op(dof_handler, constraints, material, false)
    , has_prescribed(
        Utilities::MPI::max(prescribed.n_elements(), dof_handler.get_mpi_communicator()) > 0)
  {
    op.initialize_dof_vector(tmp);
    if (has_prescribed)
      {
        op.initialize_dof_vector(prescribed_mask);
        op.initialize_dof_vector(free_mask);
        const IndexSet                        &owned = dof_handler.locally_owned_dofs();
        LinearAlgebra::ReadWriteVector<number> rw_prescribed(owned), rw_free(owned);
        for (const auto i : owned)
          {
            rw_prescribed(i) = prescribed.is_element(i) ? 1. : 0.;
            rw_free(i)       = 1. - rw_prescribed(i);
          }
        prescribed_mask.import_elements(rw_prescribed, VectorOperation::insert);
        free_mask.import_elements(rw_free, VectorOperation::insert);
      }
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::linearize(VectorType &residual,
                                                                              const VectorType &u)
  {
    op.evaluate_residual(residual, u);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::linearize(const VectorType &u)
  {
    op.evaluate_residual(tmp, u);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  typename ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::Operator &
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::get_operator()
  {
    return op;
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    if (!has_prescribed)
      {
        op.vmult_linearized(dst, src);
        return;
      }
    tmp = src;
    tmp.scale(free_mask);
    op.vmult_linearized(dst, tmp);
    dst.scale(free_mask);
    tmp = src;
    tmp.scale(prescribed_mask);
    dst += tmp;
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::vmult_dummy(
    VectorType       &dst,
    const VectorType &src,
    const bool,
    const bool) const
  {
    vmult(dst, src);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::Tvmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    vmult(dst, src);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::initialize_dof_vector(
    VectorType &vec) const
  {
    op.initialize_dof_vector(vec);
  }



  template <int dim, int fe_degree, typename number, typename Material>
  void
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::compute_diagonal()
  {
    inverse_diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
    VectorType &diagonal     = inverse_diagonal_entries->get_vector();
    op.compute_linearized_diagonal(diagonal);
    if (has_prescribed)
      {
        diagonal.scale(free_mask);
        diagonal += prescribed_mask;
      }

    number *raw_diagonal = diagonal.get_values();
    Kokkos::parallel_for(
      "ElasticityTangentLevelOperator invert diagonal",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, diagonal.locally_owned_size()),
      KOKKOS_LAMBDA(const int i) { raw_diagonal[i] = 1. / raw_diagonal[i]; });
    Kokkos::fence();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  std::shared_ptr<DiagonalMatrix<
    typename ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::VectorType>>
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::get_matrix_diagonal_inverse()
    const
  {
    return inverse_diagonal_entries;
  }



  template <int dim, int fe_degree, typename number, typename Material>
  types::global_dof_index
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::m() const
  {
    return op.m();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  types::global_dof_index
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::n() const
  {
    return op.n();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  number
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::el(
    const types::global_dof_index,
    const types::global_dof_index) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0.;
  }



  template <int dim, int fe_degree, typename number, typename Material>
  const DoFHandler<dim> &
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::get_dof_handler() const
  {
    return op.get_dof_handler();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  const typename ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::ColoredGraph &
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::get_colored_graph() const
  {
    return op.get_colored_graph();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::get_vector_partitioner() const
  {
    return op.get_vector_partitioner();
  }



  template <int dim, int fe_degree, typename number, typename Material>
  bool
  ElasticityTangentLevelOperator<dim, fe_degree, number, Material>::
    use_overlap_communication_computation() const
  {
    return op.use_overlap_communication_computation();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
