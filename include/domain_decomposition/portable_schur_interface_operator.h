#ifndef portable_schur_interface_operator_h
#define portable_schur_interface_operator_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename Number>
  class SchurInterfaceOperator : public EnableObserverPointer
  {
  public:
    using SubdomainMatrixType     = SubdomainLaplaceOperatorBase<dim, Number>;
    using SubdomainPreconditioner = VCycleMultigridBase<dim, Number>;

    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    SchurInterfaceOperator(const SubdomainMatrixType     &subdomain_operator,
                           const SubdomainPreconditioner &dirichlet_preconditioner);

    void
    vmult(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

    void
    Tvmult(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

    types::global_dof_index
    m() const;
    types::global_dof_index
    n() const;

    void
    assemble_rhs_schur(InterfaceVectorType       &rhs_schur,
                       const SubdomainVectorType &rhs_subdomain) const;

    bool
    enable_printing() const;

    void
    compute_interface_weights();

    void
    dirichlet_solve_subdomain(SubdomainVectorType &dst, const SubdomainVectorType &src) const;

    void
    reconstruct_subdomain_solution_from_interface(SubdomainVectorType       &subdomain_solution,
                                                  const InterfaceVectorType &interface_solution,
                                                  const SubdomainVectorType &rhs_subdomain) const;

    DeviceVector<const Number>
    get_interface_weights() const;

    unsigned int
    get_maximum_subdomain_mg_iterations() const;

    struct DirichletSubdomainOperator
    {
      DirichletSubdomainOperator(const SubdomainLaplaceOperatorBase<dim, Number> &op)
        : op(op)
      {}

      void
      vmult(SubdomainVectorType &dst, const SubdomainVectorType &src) const
      {
        op.vmult(dst, src);
      }


      const SubdomainLaplaceOperatorBase<dim, Number> &op;
    };

    void
    reset_maximum_subdomain_mg_iterations()
    {
      max_subdomain_mg_iterations = 0;
    }

  private:
    ObserverPointer<const SubdomainMatrixType> subdomain_operator;

    ObserverPointer<const SubdomainDoFHandler<dim>> subdomain_dof_handler;

    ObserverPointer<const SubdomainPreconditioner> dirichlet_preconditioner;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    InterfaceVectorType interface_weights;

    mutable SubdomainVectorType temp_subdomain_vector_src, temp_subdomain_vector_dst,
      temp_subdomain_vector_work;

    DirichletSubdomainOperator subdomain_dirichlet_operator;

    mutable unsigned int max_subdomain_mg_iterations;
  };

  template <int dim, typename Number>
  SchurInterfaceOperator<dim, Number>::SchurInterfaceOperator(
    const SubdomainMatrixType     &subdomain_operator,
    const SubdomainPreconditioner &dirichlet_preconditioner)
    : subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , dirichlet_preconditioner(&dirichlet_preconditioner)
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
    , subdomain_dirichlet_operator(subdomain_operator)
  {
    Assert(this->subdomain_operator->get_subdomain_dof_handler()
               .get_interface_vector_partitioner() != nullptr,
           ExcMessage("The subdomain dof handler does not have an interface vector partitioner."));

    this->subdomain_operator->initialize_dof_vector(this->temp_subdomain_vector_src);
    this->subdomain_operator->initialize_dof_vector(this->temp_subdomain_vector_dst);
    this->subdomain_operator->initialize_dof_vector(this->temp_subdomain_vector_work);

    compute_interface_weights();

    max_subdomain_mg_iterations = 0;
  }

  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::compute_interface_weights()
  {
    this->interface_weights.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    const unsigned int n_locally_relevant =
      this->subdomain_dof_handler->n_locally_relevant_interface_indices();
    DeviceVector<Number> weights_view(interface_weights.get_values(), n_locally_relevant);

    Kokkos::parallel_for(
      "interface_weights_fill",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0,
                                                                               n_locally_relevant),
      KOKKOS_LAMBDA(const unsigned int i) { weights_view(i) = Number(1.0); });
    Kokkos::fence();

    this->interface_weights.compress(VectorOperation::add);

    const unsigned int n_locally_owned = interface_weights.locally_owned_size();

    Kokkos::parallel_for(
      "interface_weights_invert",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0, n_locally_owned),
      KOKKOS_LAMBDA(const unsigned int i) { weights_view(i) = Number(1.0) / weights_view(i); });
    Kokkos::fence();

    this->interface_weights.update_ghost_values();
  }

  template <int dim, typename Number>
  DeviceVector<const Number>
  SchurInterfaceOperator<dim, Number>::get_interface_weights() const
  {
    return DeviceVector<const Number>(
      interface_weights.get_values(),
      this->subdomain_dof_handler->n_locally_relevant_interface_indices());
  }

  template <int dim, typename Number>
  unsigned int
  SchurInterfaceOperator<dim, Number>::get_maximum_subdomain_mg_iterations() const
  {
    return max_subdomain_mg_iterations;
  }


  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::dirichlet_solve_subdomain(
    SubdomainVectorType       &dst,
    const SubdomainVectorType &src) const
  {
    dst = 0.;
    // SolverControl solver_control(src.size(), 1e-12 * src.l2_norm());
    ReductionControl solver_control(100, 1e-15, 1e-12);

    SolverCG<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>> cg(solver_control);

    cg.solve(this->subdomain_dirichlet_operator, dst, src, *dirichlet_preconditioner);

    // std::cout << "    Dirichlet solver on subdomain "
    //           << this->subdomain_dof_handler->get_subdomain_id()
    //           << " converged in " << solver_control.last_step()
    //           << " iterations " << std::endl;

    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));
  }

  /**
   * Schur action: y = [A_GG - A_GI*A_II^{-1}*A_IG]*x
   * 1. w = A_GG*x, v = A_IG*x
   * 2. z = A_II^{-1}*v
   * 3. vv =  A_GI*z
   * Result y = w - vv
   */
  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::vmult(InterfaceVectorType       &dst,
                                             const InterfaceVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    dst = 0.;

    src.update_ghost_values();

    DeviceVector<Number> src_view(src.get_values(), interface_dof_indices_subdomain.size()),
      dst_view(dst.get_values(), interface_dof_indices_subdomain.size());

    DeviceVector<Number> t_subdomain_src_view(temp_subdomain_vector_src.get_values(),
                                              temp_subdomain_vector_src.size()),
      t_subdomain_dst_view(temp_subdomain_vector_dst.get_values(),
                           temp_subdomain_vector_dst.size()),
      t_subdomain_work_view(temp_subdomain_vector_work.get_values(),
                            temp_subdomain_vector_work.size());


    const auto interface_dofs = this->interface_dof_indices_subdomain;

    // read interface values into the subdomain vector
    temp_subdomain_vector_src = 0.;
    Kokkos::parallel_for(
      "read_src_interface", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        t_subdomain_src_view(interface_dofs(i)) = src_view(i);
      });

    // Apply Schur complement operators w = A_GG*x and v = A_IG *x
    this->subdomain_operator->vmult_interface_cell_range(temp_subdomain_vector_dst,
                                                         temp_subdomain_vector_src);

    this->dirichlet_solve_subdomain(temp_subdomain_vector_work, temp_subdomain_vector_dst);

    // zero out entries of z corresponding to interface dofs
    Kokkos::parallel_for(
      "zero_out_interface_work", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        t_subdomain_work_view(interface_dofs(i)) = 0.;
      });

    // apply vv = A_GI * z
    this->subdomain_operator->vmult_interface_cell_range(temp_subdomain_vector_src,
                                                         temp_subdomain_vector_work);

    // write result y = w-vv
    Kokkos::parallel_for(
      "distribute_interface_dofs", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        const auto idx          = interface_dofs(i);
        Number     output_value = t_subdomain_dst_view(idx) - t_subdomain_src_view(idx);
        dst_view(i) += output_value;
      });

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
  }

  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::assemble_rhs_schur(
    InterfaceVectorType       &rhs_schur,
    const SubdomainVectorType &rhs_subdomain) const
  {
    Assert(rhs_schur.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
         interface vector partitioner."));

    rhs_schur = 0.;

    const auto interface_dofs = this->interface_dof_indices_subdomain;

    DeviceVector<Number> rhs_subdomain_view(rhs_subdomain.get_values(), rhs_subdomain.size());
    DeviceVector<Number> rhs_schur_view(rhs_schur.get_values(), interface_dofs.size());
    DeviceVector<Number> t_subdomain_dst_view(temp_subdomain_vector_dst.get_values(),
                                              temp_subdomain_vector_dst.size());

    // solve for interior, A_II^{-1} * F_I
    temp_subdomain_vector_src = 0.;
    dirichlet_solve_subdomain(temp_subdomain_vector_src, rhs_subdomain);

    // multiply by A_GI *A_II^{-1} * F_I
    this->subdomain_operator->vmult_interface_cell_range(temp_subdomain_vector_dst,
                                                         temp_subdomain_vector_src);

    // distribute interface dofs into rhs_schur: F_G - A_GI *A_II^{-1} * F_I
    Kokkos::parallel_for(
      "distribute_interface_dofs", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        const auto idx_subdomain = interface_dofs(i);
        Number     output_value =
          rhs_subdomain_view(idx_subdomain) - t_subdomain_dst_view(idx_subdomain);
        rhs_schur_view(i) += output_value;
      });

    rhs_schur.compress(VectorOperation::add);

    rhs_schur.update_ghost_values();
  }

  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::reconstruct_subdomain_solution_from_interface(
    SubdomainVectorType       &subdomain_solution,
    const InterfaceVectorType &interface_solution,
    const SubdomainVectorType &rhs_subdomain) const
  {
    Assert(interface_solution.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
         interface vector partitioner."));

    subdomain_solution = 0.;

    const auto interface_dofs = this->interface_dof_indices_subdomain;

    DeviceVector<Number> rhs_subdomain_view(rhs_subdomain.get_values(), rhs_subdomain.size()),
      interface_solution_view(interface_solution.get_values(), interface_dofs.size()),
      subdomain_solution_view(subdomain_solution.get_values(), subdomain_solution.size());

    DeviceVector<Number> t_subdomain_src_view(temp_subdomain_vector_src.get_values(),
                                              temp_subdomain_vector_src.size()),
      t_subdomain_dst_view(temp_subdomain_vector_dst.get_values(),
                           temp_subdomain_vector_dst.size()),
      t_subdomain_work_view(temp_subdomain_vector_work.get_values(),
                            temp_subdomain_vector_work.size());

    // read interface_values
    temp_subdomain_vector_src = 0.;
    Kokkos::parallel_for(
      "read_interface_solution", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        t_subdomain_src_view(interface_dofs(i)) = interface_solution_view(i);
      });

    // apply A_IG * src_interface
    this->subdomain_operator->vmult_interface_cell_range(temp_subdomain_vector_dst,
                                                         temp_subdomain_vector_src);

    // prepare F_I - A_IG * src_interface
    temp_subdomain_vector_src = rhs_subdomain;
    temp_subdomain_vector_src -= temp_subdomain_vector_dst;

    // solve interior, A_II^{-1} * (F_I-A_IG * src_interface)
    subdomain_solution = 0;
    dirichlet_solve_subdomain(subdomain_solution, temp_subdomain_vector_src);

    // distribute interface dofs into subdomain solution
    Kokkos::parallel_for(
      "distribute_interface_dofs", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        subdomain_solution_view(interface_dofs(i)) = interface_solution_view(i);
      });

    subdomain_solution.compress(VectorOperation::add);
  }


  template <int dim, typename Number>
  void
  SchurInterfaceOperator<dim, Number>::Tvmult(InterfaceVectorType       &dst,
                                              const InterfaceVectorType &src) const
  {
    this->vmult(dst, src);
  }

  template <int dim, typename Number>
  types::global_dof_index
  SchurInterfaceOperator<dim, Number>::m() const
  {
    return this->subdomain_dof_handler->get_interface_vector_partitioner()->size();
  }

  template <int dim, typename Number>
  types::global_dof_index
  SchurInterfaceOperator<dim, Number>::n() const
  {
    return this->subdomain_dof_handler->get_interface_vector_partitioner()->size();
  }

  template <int dim, typename Number>
  bool
  SchurInterfaceOperator<dim, Number>::enable_printing() const
  {
    return (this->subdomain_dof_handler->get_subdomain_id() == 0);
  }

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif