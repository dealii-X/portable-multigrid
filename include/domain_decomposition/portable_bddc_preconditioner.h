#ifndef portable_bddc_preconditioner_h
#define portable_bddc_preconditioner_h

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/solver_cg.h>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_block_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "multigrid/portable_block_vcycle_adapters.h"
#include "multigrid/portable_projected_chebyshev_smoother.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  /**
   * BddcSmootherType is the type of subdomain_mg_smoothers_bddc's element
   * in program.cc -- ProjectedChebyshevSmoother<LevelMatrixType,
   * BddcPreconditionerType, VectorTypeMG> in production -- needed here
   * (rather than only through the type-erased VCycleMultigridBase
   * subdomain_mg_preconditioner already stored below) so that
   * compute_local_coarse_matrix() can read each level's AdditionalData
   * (degree/max_eigenvalue/smoothing_range) back out via
   * get_additional_data() and build a *second*, block-shaped V-cycle for
   * the coarse problem -- VCycleMultigridBase's vmult()-only interface
   * has no way to expose the individual levels a block V-cycle needs.
   */
  template <int dim, typename Number, typename BddcSmootherType>
  class BDDCPreconditioner
  {
  public:
    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainPreconditioner = VCycleMultigridBase<dim, Number>;


    BDDCPreconditioner(
      const SchurInterfaceOperator<dim, Number>       &interface_operator,
      const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
      const SubdomainPreconditioner                   &subdomain_mg_preconditione,
      const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
        &level_bddc_matrices,
      const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers,
      const MGLevelObject<BddcSmootherType>                             &level_bddc_smoothers,
      const BDDCVariant variant = BDDCVariant::corner_edge_face);

    void
    vmult(InterfaceVectorType &dst, const InterfaceVectorType &src) const;

    void
    project_to_homogeneous_constraints_interface(InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const;

    void
    project_to_homogeneous_constraints_subdomain(SubdomainVectorType &subdomain_vector) const;


    void
    lift_coarse_to_subdomain(SubdomainVectorType  &interface_vector,
                             const Vector<Number> &coarse_vector) const;

    void
    compute_coarse_matrix();

    void
    vmult_fine_correction(SubdomainVectorType       &fine_solution,
                          const SubdomainVectorType &fine_residual) const;

    void
    vmult_coarse_correction(SubdomainVectorType       &coarse_solution,
                            const SubdomainVectorType &fine_residual) const;

    void
    coarse_to_global_interface(InterfaceVectorType  &interface_vector,
                               const Vector<Number> &coarse_vector) const;

    void
    global_interface_to_coarse(Vector<Number>            &coarse_vector,
                               const InterfaceVectorType &interface_vector) const;

    void
    gather_and_weight_global_interface(SubdomainVectorType       &dst,
                                       const InterfaceVectorType &src) const;

    void
    weight_local_interface_and_scatter(InterfaceVectorType       &dst,
                                       const SubdomainVectorType &src) const;



    void
    reset_timings() const;


    const std::array<double, 4> &
    get_timings() const;

    void
    reset_setup_timings() const;

    const std::array<double, 6> &
    get_setup_timings() const;

    unsigned int
    get_n_local_coarse_dofs() const;

    unsigned int
    get_n_global_coarse_dofs() const;


    struct SubdomainProjectorWrapper
    {
    public:
      const BDDCPreconditioner<dim, Number, BddcSmootherType> &parent;

      SubdomainProjectorWrapper(const BDDCPreconditioner<dim, Number, BddcSmootherType> &parent_preconditioner)
        : parent(parent_preconditioner)
      {}

      void
      project(SubdomainVectorType &subdomain_vector) const
      {
        parent.project_to_homogeneous_constraints_subdomain(subdomain_vector);
      }
    };

    unsigned int
    get_maximum_subdomain_mg_iterations() const;

  private:
    void
    setup_primal_constraint_views();

    void
    compute_local_coarse_matrix(LAPACKFullMatrix<Number> &local_coarse_matrix);

    ObserverPointer<const SchurInterfaceOperator<dim, Number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    const SubdomainPreconditioner &subdomain_mg_preconditioner;

    // Per-level BDDC matrices/transfers/smoothers, needed (alongside
    // subdomain_mg_preconditioner above) to build a second, block-shaped
    // V-cycle for compute_local_coarse_matrix() -- see the class-level
    // comment on BddcSmootherType for why the type-erased
    // subdomain_mg_preconditioner alone isn't enough.
    const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
                                                                        &level_bddc_matrices;
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers;
    const MGLevelObject<BddcSmootherType>                              &level_bddc_smoothers;

    SubdomainBDDCOperatorWrapper<dim, Number> subdomain_bddc_operator;

    LAPACKFullMatrix<Number> coarse_matrix;

    BDDCVariant bddc_variant;

    unsigned int       n_global_coarse_dofs;
    unsigned int       n_local_coarse_dofs;
    const unsigned int n_subdomain_dofs;
    const unsigned int interface_vector_size;

    const unsigned int coarse_problem_rank;
    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const InterfaceVectorType &interface_weights;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    // Flattened fine local interface DoF indices associated with each local primal constraint
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_interface_local;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_subdomain;

    // offset of the primal constraint dofs per each dof entity
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_offsets;

    // subdomain (local) to global constraint map
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> coarse_dofs_local_to_global;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> corner_dofs_subdomain;

    std::vector<unsigned int> coarse_dofs_local_to_global_vector_host;

    std::vector<SubdomainVectorType> coarse_basis_functions;

    mutable InterfaceVectorType temp_interface;
    mutable Vector<Number>      temp_coarse_local;
    mutable Vector<Number>      temp_coarse_global;


    mutable SubdomainVectorType temp_subdomain_dst;
    mutable SubdomainVectorType temp_subdomain_src;
    mutable SubdomainVectorType temp_subdomain_coarse;
    mutable SubdomainVectorType temp_subdomain_fine;


    mutable std::vector<Number> temp_global_coarse_std;

    /**
     * Per-vmult() solve-phase timings, reset externally (reset_timings())
     * and accumulated across repeated vmult() calls -- e.g. over the whole
     * outer interface CG solve -- so get_timings() reports totals for
     * whatever span the caller bracketed with reset_timings().
     *
     * timings[0] = gather_and_weight_global_interface + weight_local_interface_and_scatter
     * timings[1] = vmult_coarse_correction (global coarse problem solve)
     * timings[2] = vmult_fine_correction (local constrained CG solve)
     * timings[3] = total vmult() wall time
     */
    mutable std::array<double, 4> timings;

    /**
     * One-shot setup-phase timings for compute_coarse_matrix() /
     * compute_local_coarse_matrix(), reset internally at the start of each
     * compute_coarse_matrix() call (it runs once per preconditioner setup,
     * not per iteration, so there is no meaningful "per outer solve" span
     * to accumulate over the way there is for timings[] above).
     *
     * setup_timings[0] = lift_coarse_to_subdomain (building lifted constraint vectors)
     * setup_timings[1] = vmult_plain calls (rhs assembly + S*phi_j)
     * setup_timings[2] = vmult_fine_correction (one CG solve per local primal constraint)
     * setup_timings[3] = local coarse matrix inner products
     * setup_timings[4] = MPI sum (global reduction of local coarse matrix contributions)
     * setup_timings[5] = LU factorization of the global coarse matrix
     */
    mutable std::array<double, 6> setup_timings;

    mutable unsigned int max_subdomain_mg_iterations;
  };

  template <int dim, typename Number, typename BddcSmootherType>
  BDDCPreconditioner<dim, Number, BddcSmootherType>::BDDCPreconditioner(
    const SchurInterfaceOperator<dim, Number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, Number> &subdomain_operator,
    const SubdomainPreconditioner                   &subdomain_mg_preconditioner,
    const MGLevelObject<std::unique_ptr<SubdomainLaplaceOperatorBase<dim, Number>>>
                                                                        &level_bddc_matrices,
    const MGLevelObject<std::unique_ptr<MGTransferBase<dim, Number>>> &level_bddc_transfers,
    const MGLevelObject<BddcSmootherType>                              &level_bddc_smoothers,
    const BDDCVariant                                                   variant)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , subdomain_mg_preconditioner(subdomain_mg_preconditioner)
    , level_bddc_matrices(level_bddc_matrices)
    , level_bddc_transfers(level_bddc_transfers)
    , level_bddc_smoothers(level_bddc_smoothers)
    , subdomain_bddc_operator(subdomain_operator)
    , n_subdomain_dofs(subdomain_operator.get_subdomain_dof_handler().get_dof_handler().n_dofs())
    , interface_vector_size(subdomain_operator.get_interface_dof_indices_subdomain().size())
    , coarse_problem_rank(subdomain_dof_handler->n_subdomains() - 1)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
  {
    if (dim == 2 && variant == BDDCVariant::corner_edge_face)
      bddc_variant = BDDCVariant::corner_edge;
    else
      bddc_variant = variant;

    // store primal constraints
    {
      const auto &subdomain_dof_info = this->subdomain_dof_handler->get_dof_info();

      if (bddc_variant == BDDCVariant::corner)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[1]; // End of Vertices
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[1];

          if (n_local_coarse_dofs == 0)
            bddc_variant = BDDCVariant::corner_edge;
        }

      if (bddc_variant == BDDCVariant::corner_edge)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[2]; // End of Edges
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[2];

          if (n_local_coarse_dofs == 0)
            bddc_variant = BDDCVariant::corner_edge_face;
        }

      if (bddc_variant == BDDCVariant::corner_edge_face)
        {
          n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[3]; // End of Faces
          n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[3];
        }

      AssertThrow(n_global_coarse_dofs > 0, ExcMessage("There's zero global constraints"));
      AssertThrow(n_local_coarse_dofs > 0, ExcMessage("There's zero local constraints"));

      setup_primal_constraint_views();
    }

    temp_interface.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    temp_coarse_local.reinit(n_local_coarse_dofs);
    temp_coarse_global.reinit(n_global_coarse_dofs);

    temp_global_coarse_std.resize(n_global_coarse_dofs);

    subdomain_operator.initialize_dof_vector(temp_subdomain_dst);
    temp_subdomain_src.reinit(temp_subdomain_dst);
    temp_subdomain_coarse.reinit(temp_subdomain_dst);
    temp_subdomain_fine.reinit(temp_subdomain_dst);

    max_subdomain_mg_iterations = 0;

    reset_timings();
    reset_setup_timings();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::reset_timings() const
  {
    for (unsigned int i = 0; i < timings.size(); ++i)
      timings[i] = 0.;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::reset_setup_timings() const
  {
    for (unsigned int i = 0; i < setup_timings.size(); ++i)
      setup_timings[i] = 0.;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 6> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_setup_timings() const
  {
    return setup_timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_n_local_coarse_dofs() const
  {
    return n_local_coarse_dofs;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_n_global_coarse_dofs() const
  {
    return n_global_coarse_dofs;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  const std::array<double, 4> &
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_timings() const
  {
    return timings;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult(InterfaceVectorType       &dst,
                                         const InterfaceVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    Kokkos::fence();
    Timer total_time;

    dst = 0;
    src.update_ghost_values();

    Kokkos::fence();
    Timer time;
    gather_and_weight_global_interface(temp_subdomain_src, src);
    Kokkos::fence();
    timings[0] += time.wall_time();

    time.restart();
    vmult_coarse_correction(temp_subdomain_coarse, temp_subdomain_src);
    Kokkos::fence();
    timings[1] += time.wall_time();

    // vmult_coarse_correction() above needs the raw (unprojected) residual --
    // it takes inner products against the coarse basis functions, not Ahat/
    // Chat's sandwiched action -- so this project() has to happen here,
    // after the coarse correction and before the fine one, rather than
    // earlier. Projecting temp_subdomain_src in place lets
    // vmult_fine_correction() take its RHS by const reference directly
    // (no internal copy) since the "RHS must already be in V" precondition
    // is now satisfied by the caller.
    this->subdomain_bddc_operator.project(temp_subdomain_src);

    time.restart();
    vmult_fine_correction(temp_subdomain_fine, temp_subdomain_src);
    Kokkos::fence();
    timings[2] += time.wall_time();

    temp_subdomain_coarse += temp_subdomain_fine;

    time.restart();
    weight_local_interface_and_scatter(dst, temp_subdomain_coarse);
    Kokkos::fence();
    timings[0] += time.wall_time();

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    Kokkos::fence();
    timings[3] += total_time.wall_time();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_fine_correction(
    SubdomainVectorType       &fine_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(fine_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);

    fine_solution = 0;

    SolverControl solver_control(fine_residual.size(), 1e-12 * fine_residual.l2_norm());
    // ReductionControl solver_control(100, 1e-16, 1e-12);

    // subdomain_bddc_operator.vmult() == Pi*A*Pi and subdomain_mg_preconditioner.vmult()
    // are both symmetric and map into V = range(Pi) by construction (Pi is baked into
    // vmult() itself, and the BDDC-level smoother's elementary correction is sandwiched
    // as Pi*(omega*D^{-1})*Pi), so plain CG works directly -- only the RHS needs to be
    // in V up front.
    //
    // Precondition: fine_residual must already satisfy Pi*fine_residual == fine_residual
    // on entry -- this function does NOT project it. SolverCG::solve() takes its RHS by
    // const reference and never writes to it internally, so as long as the caller has
    // already projected the vector it's about to pass in (both callers do: vmult()
    // projects temp_subdomain_src in place right before calling this, and
    // compute_local_coarse_matrix() projects its local rhs before calling this), the
    // fine_residual reference can be handed straight to solve() with no extra copy.
    SolverCG<SubdomainVectorType> solver(solver_control);
    solver.solve(this->subdomain_bddc_operator,
                fine_solution,
                fine_residual,
                subdomain_mg_preconditioner);

    // std::cout << "Constrained projected solver converged in " << solver_control.last_step()
    //           << "  iterations." << std::endl;

    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));
  }

  template <int dim, typename Number, typename BddcSmootherType>
  unsigned int
  BDDCPreconditioner<dim, Number, BddcSmootherType>::get_maximum_subdomain_mg_iterations() const
  {
    return max_subdomain_mg_iterations;
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::vmult_coarse_correction(
    SubdomainVectorType       &coarse_solution,
    const SubdomainVectorType &fine_residual) const
  {
    AssertDimension(coarse_solution.size(), n_subdomain_dofs);
    AssertDimension(fine_residual.size(), n_subdomain_dofs);

    coarse_solution = 0;

    const unsigned int n_coarse_local  = this->n_local_coarse_dofs;
    const unsigned int n_coarse_global = this->n_global_coarse_dofs;

    const auto interface_dof_subdomain = this->interface_dof_indices_subdomain;

    temp_coarse_local = 0;

    DeviceVector<const Number> fine_residual_view(fine_residual.get_values(), fine_residual.size());
    DeviceVector<Number> coarse_solution_view(coarse_solution.get_values(), coarse_solution.size());

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        const SubdomainVectorType &basis_function = this->coarse_basis_functions[j];

        DeviceVector<const Number> basis_function_view(basis_function.get_values(),
                                                       basis_function.size());

        Number local_inner_product = 0;

        Kokkos::parallel_reduce(
          "coarse_rhs_inner_product",
          this->interface_vector_size,
          KOKKOS_LAMBDA(const int i, Number &sum) {
            const unsigned int subdomain_idx = interface_dof_subdomain(i);
            sum += fine_residual_view(subdomain_idx) * basis_function_view(subdomain_idx);
          },
          local_inner_product);
        Kokkos::fence();

        temp_coarse_local(j) = local_inner_product;
      }

    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    const auto &local_to_global = this->coarse_dofs_local_to_global_vector_host;

    for (unsigned int i = 0; i < n_coarse_local; ++i)
      {
        temp_global_coarse_std[local_to_global[i]] = temp_coarse_local(i);
      }

    Utilities::MPI::sum(temp_global_coarse_std,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        temp_global_coarse_std);

    temp_coarse_global = 0;

    for (unsigned int i = 0; i < n_coarse_global; ++i)
      temp_coarse_global(i) = temp_global_coarse_std[i];

    coarse_matrix.solve(temp_coarse_global);

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        const Number local_coarse_value = temp_coarse_global[local_to_global[j]];

        const SubdomainVectorType &basis_function = this->coarse_basis_functions[j];

        DeviceVector<const Number> basis_function_view(basis_function.get_values(),
                                                       basis_function.size());

        Kokkos::parallel_for(
          "coarse_prolongation", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
            const unsigned int subdomain_idx = interface_dof_subdomain(i);
            coarse_solution_view(subdomain_idx) +=
              local_coarse_value * basis_function_view(subdomain_idx);
          });
        Kokkos::fence();
      }
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::gather_and_weight_global_interface(
    SubdomainVectorType       &dst,
    const InterfaceVectorType &src) const
  {
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(dst.size(), n_subdomain_dofs);

    dst = 0;

    DeviceVector<Number>       dst_view(dst.get_values(), dst.size());
    DeviceVector<const Number> src_view(src.get_values(), this->interface_vector_size);

    const DeviceVector<const Number> weights(interface_weights.get_values(),
                                             this->interface_vector_size);
    const auto interface_dof_subdomain = this->interface_dof_indices_subdomain;

    Kokkos::parallel_for(
      "scale_interface_residual", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
        dst_view(interface_dof_subdomain(i)) = src_view(i) * weights(i);
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::weight_local_interface_and_scatter(
    InterfaceVectorType       &dst,
    const SubdomainVectorType &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(src.size(), n_subdomain_dofs);

    dst = 0;

    DeviceVector<Number>       dst_view(dst.get_values(), this->interface_vector_size);
    DeviceVector<const Number> src_view(src.get_values(), src.size());

    const DeviceVector<const Number> weights(interface_weights.get_values(),
                                             this->interface_vector_size);

    const auto interface_dof_subdomain = this->interface_dof_indices_subdomain;

    Kokkos::parallel_for(
      "scale_interface_residual", this->interface_vector_size, KOKKOS_LAMBDA(const int i) {
        dst_view(i) = src_view(interface_dof_subdomain(i)) * weights(i);
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::project_to_homogeneous_constraints_interface(
    InterfaceVectorType &interface_vector) const
  {
    auto interface_vector_view = interface_vector.get_values();

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;
    const auto weights               = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              interface_vector_view(local_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::
    project_to_homogeneous_constraints_interface_and_scatter_to_subdomain(
      SubdomainVectorType       &subdomain_vector,
      const InterfaceVectorType &interface_vector) const
  {
    subdomain_vector                 = 0;
    const auto interface_vector_view = interface_vector.get_values();
    auto       subdomain_vector_view = subdomain_vector.get_values();

    const auto offsets                   = this->primal_constraint_offsets;
    const auto local_constraint_dofs     = this->primal_constraint_dofs_interface_local;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_interface_and_scatter_to_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += interface_vector_view(local_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) =
                interface_vector_view(local_constraint_dofs(i)) - average;
          }
      });
    Kokkos::fence();
  }


  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::project_to_homogeneous_constraints_subdomain(
    SubdomainVectorType &subdomain_vector) const
  {
    DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(),
                                               subdomain_vector.size());

    const auto offsets                   = this->primal_constraint_offsets;
    const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
    const auto weights                   = this->coarse_weights;

    Kokkos::parallel_for(
      "project_to_homogeneous_constraints_subdomain",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const unsigned int n_dofs_per_coarse_dof = end - start;

        if (n_dofs_per_coarse_dof > 0)
          {
            Number average = 0;
            for (unsigned int i = start; i < end; ++i)
              average += subdomain_vector_view(subdomain_constraint_dofs(i));
            average *= weights(coarse_local_idx);

            for (unsigned int i = start; i < end; ++i)
              subdomain_vector_view(subdomain_constraint_dofs(i)) -= average;
          }
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::global_interface_to_coarse(
    Vector<Number>            &coarse_vector,
    const InterfaceVectorType &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);
    coarse_vector = 0;

    interface_vector.update_ghost_values();

    auto interface_vector_view = interface_vector.get_values();

    const auto weights = this->coarse_weights;

    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> local_coarse_contribution(
      "local_coarse_contribution", this->n_local_coarse_dofs);

    Kokkos::parallel_for(
      "global_interface_to_coarse_local_sum",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        Number average = 0;
        for (int i = start; i < end; ++i)
          average += interface_vector_view(local_constraint_dofs(i));

        average *= weights(coarse_local_idx);

        local_coarse_contribution(coarse_local_idx) = average;
      });

    Kokkos::fence();

    auto local_coarse_contribution_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_coarse_contribution);

    AssertDimension(temp_global_coarse_std.size(), this->n_global_coarse_dofs);
    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    for (unsigned int local_idx = 0; local_idx < this->n_local_coarse_dofs; ++local_idx)
      {
        const unsigned int global_coarse_idx = coarse_dofs_local_to_global_vector_host[local_idx];
        temp_global_coarse_std[global_coarse_idx] = local_coarse_contribution_host(local_idx);
      }

    Utilities::MPI::sum(temp_global_coarse_std,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        coarse_vector.get_values());

    interface_vector.zero_out_ghost_values();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::coarse_to_global_interface(
    InterfaceVectorType  &interface_vector,
    const Vector<Number> &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));

    AssertDimension(coarse_vector.size(), this->n_global_coarse_dofs);

    interface_vector = 0;

    AssertDimension(temp_global_coarse_std.size(), this->n_global_coarse_dofs);
    std::fill(temp_global_coarse_std.begin(), temp_global_coarse_std.end(), Number(0));

    for (unsigned int i = 0; i < this->n_local_coarse_dofs; ++i)
      temp_global_coarse_std[i] = coarse_vector(coarse_dofs_local_to_global_vector_host[i]);

    // copy to host
    Kokkos::View<Number *, Kokkos::HostSpace> local_coarse_host_view(temp_global_coarse_std.data(),
                                                                     this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);

    auto interface_vector_view = interface_vector.get_values();

    const auto weights               = this->coarse_weights;
    const auto offsets               = this->primal_constraint_offsets;
    const auto local_constraint_dofs = this->primal_constraint_dofs_interface_local;

    Kokkos::parallel_for(
      "coarse_to_global_interface_interpolate",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int coarse_local_idx) {
        const unsigned int start = offsets(coarse_local_idx);
        const unsigned int end   = offsets(coarse_local_idx + 1);

        const Number coarse_value =
          weights(coarse_local_idx) * local_coarse_device_view(coarse_local_idx);

        for (unsigned int i = start; i < end; ++i)
          interface_vector_view(local_constraint_dofs(i)) = coarse_value;
      });

    Kokkos::fence();

    // update globally
    interface_vector.compress(VectorOperation::insert);
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::lift_coarse_to_subdomain(
    SubdomainVectorType  &subdomain_vector,
    const Vector<Number> &coarse_vector) const
  {
    AssertDimension(subdomain_vector.size(), this->n_subdomain_dofs);

    AssertDimension(coarse_vector.size(), this->n_local_coarse_dofs);

    subdomain_vector = 0.;

    // copy to host
    const Kokkos::View<const Number *, Kokkos::HostSpace> local_coarse_host_view(
      coarse_vector.data(), this->n_local_coarse_dofs);

    auto local_coarse_device_view =
      Kokkos::create_mirror_view_and_copy(MemorySpace::Default::kokkos_space(),
                                          local_coarse_host_view);
    Kokkos::fence();

    DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(),
                                               subdomain_vector.size());

    const auto constraint_dof_subdomain = this->primal_constraint_dofs_subdomain;
    const auto offsets                  = this->primal_constraint_offsets;
    const auto weights                  = this->coarse_weights;

    Kokkos::parallel_for(
      "lift_coarse_constraints",
      this->n_local_coarse_dofs,
      KOKKOS_LAMBDA(const int local_coarse_idx) {
        const unsigned int start = offsets(local_coarse_idx);
        const unsigned int end   = offsets(local_coarse_idx + 1);

        const Number coarse_value =
          local_coarse_device_view(local_coarse_idx) * weights(local_coarse_idx);

        for (unsigned int i = start; i < end; ++i)
          {
            subdomain_vector_view(constraint_dof_subdomain(i)) = coarse_value;
          }
      });
    Kokkos::fence();
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::compute_local_coarse_matrix(
    LAPACKFullMatrix<Number> &local_coarse_matrix)
  {
    const unsigned int n_coarse_local = this->n_local_coarse_dofs;
    local_coarse_matrix.reinit(n_coarse_local, n_coarse_local);

    coarse_basis_functions.resize(n_coarse_local);
    for (unsigned int i = 0; i < n_coarse_local; ++i)
      coarse_basis_functions[i].reinit(temp_subdomain_dst);

    SubdomainVectorType S_per_phi_j;
    S_per_phi_j.reinit(temp_subdomain_dst);

    DeviceVector<Number> temp_interface_view(temp_interface.get_values(), interface_vector_size);

    std::vector<SubdomainVectorType> lifted_constraints(n_coarse_local);

    Vector<Number> e_k(n_coarse_local);

    Kokkos::fence();
    Timer time;

    for (unsigned int k = 0; k < n_coarse_local; ++k)
      {
        e_k    = 0.;
        e_k(k) = Number(1);

        SubdomainVectorType &lift = lifted_constraints[k];

        lift.reinit(temp_subdomain_src);

        this->lift_coarse_to_subdomain(lift, e_k);
      }

    Kokkos::fence();
    setup_timings[0] += time.wall_time();

    // Pack the n_coarse_local lifted constraint vectors into one block
    // vector (n_coarse_local blocks of n_subdomain_dofs each -- the same
    // layout vmult_plain_block()/project_block()/SolverBlockCG use
    // throughout). phi_j starts out equal to lifted_constraints[j] before
    // any fine correction is added (see the old per-j loop this replaces),
    // so this block vector is both the "phi_j at block-solve time" input
    // to vmult_plain_block() below *and*, unpacked back out further down,
    // becomes coarse_basis_functions' initial value.
    SubdomainVectorType lifted_block;
    lifted_block.reinit(static_cast<typename SubdomainVectorType::size_type>(n_coarse_local) *
                        n_subdomain_dofs);

    for (unsigned int k = 0; k < n_coarse_local; ++k)
      {
        DeviceVector<Number> src_view(lifted_constraints[k].get_values(), n_subdomain_dofs);
        DeviceVector<Number> dst_view(lifted_block.get_values() + k * n_subdomain_dofs,
                                      n_subdomain_dofs);
        Kokkos::deep_copy(dst_view, src_view);
      }
    Kokkos::fence();

    SubdomainVectorType rhs_block;
    rhs_block.reinit(lifted_block, true);

    time.restart();
    this->subdomain_bddc_operator.vmult_plain_block(rhs_block, lifted_block, n_coarse_local);
    Kokkos::fence();
    setup_timings[1] += time.wall_time();

    rhs_block *= Number(-1);

    this->subdomain_bddc_operator.project_block(rhs_block, n_coarse_local);

    // Build a block-shaped V-cycle over the same per-level BDDC matrices/
    // transfers/smoothers the scalar subdomain_mg_preconditioner above was
    // built from (see the class-level comment on BddcSmootherType), via
    // the adapter classes in portable_block_vcycle_adapters.h. Eigenvalue
    // bounds are read back from each level's already-initialized scalar
    // smoother rather than re-estimated: they depend only on the
    // operator/preconditioner pair, not on how many RHS columns are
    // solved at once.
    using BlockOperatorType = BlockBDDCOperatorAdapter<dim, Number>;
    using BlockTransferType = BlockTransferAdapter<dim, Number>;
    using BlockPreconditionerType =
      BlockProjectedDiagonalPreconditioner<BlockOperatorType, SubdomainVectorType>;
    using BlockSmootherType =
      ProjectedChebyshevSmoother<BlockOperatorType, BlockPreconditionerType, SubdomainVectorType>;

    const unsigned int minlevel = level_bddc_matrices.min_level();
    const unsigned int maxlevel = level_bddc_matrices.max_level();

    MGLevelObject<std::unique_ptr<BlockOperatorType>> block_matrices(minlevel, maxlevel);
    MGLevelObject<std::unique_ptr<BlockTransferType>> block_transfers(minlevel, maxlevel);
    MGLevelObject<BlockSmootherType>                  block_smoothers(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        const auto &concrete_matrix =
          static_cast<const SubdomainBDDCOperatorWrapper<dim, Number> &>(*level_bddc_matrices[level]);

        block_matrices[level] =
          std::make_unique<BlockOperatorType>(concrete_matrix, n_coarse_local);

        if (level > minlevel)
          block_transfers[level] =
            std::make_unique<BlockTransferType>(*level_bddc_transfers[level], n_coarse_local);

        typename BlockSmootherType::AdditionalData smoother_data;
        const auto &scalar_data          = level_bddc_smoothers[level].get_additional_data();
        smoother_data.degree             = scalar_data.degree;
        smoother_data.max_eigenvalue     = scalar_data.max_eigenvalue;
        smoother_data.smoothing_range    = scalar_data.smoothing_range;
        smoother_data.preconditioner     = std::make_shared<BlockPreconditionerType>(
          *block_matrices[level], level_bddc_matrices[level]->get_matrix_diagonal_inverse(), n_coarse_local);

        block_smoothers[level].initialize(*block_matrices[level], smoother_data);
      }

    SubdomainVCycleMultigrid<dim, Number, BlockOperatorType, BlockTransferType, BlockSmootherType>
      block_mg_preconditioner(block_matrices, block_transfers, block_smoothers);

    SubdomainVectorType correction_block;
    correction_block.reinit(rhs_block);

    SolverControl solver_control(rhs_block.size(), 1e-12 * rhs_block.l2_norm());
    SolverBlockCG<SubdomainVectorType> block_solver(solver_control);

    time.restart();
    block_solver.solve_block(*block_matrices[maxlevel],
                             correction_block,
                             rhs_block,
                             block_mg_preconditioner,
                             n_coarse_local);
    Kokkos::fence();
    setup_timings[2] += time.wall_time();

    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));

    // phi_j = lifted_constraints[j] + correction_j, unpacked straight into
    // coarse_basis_functions (matches the old loop's
    // `phi_j = lifted_constraints[j]; ...; phi_j.add(1., temp_subdomain_dst);`).
    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        SubdomainVectorType &phi_j = coarse_basis_functions[j];

        DeviceVector<Number> lifted_view(lifted_block.get_values() + j * n_subdomain_dofs,
                                         n_subdomain_dofs);
        DeviceVector<Number> correction_view(correction_block.get_values() + j * n_subdomain_dofs,
                                             n_subdomain_dofs);
        DeviceVector<Number> phi_view(phi_j.get_values(), n_subdomain_dofs);

        Kokkos::parallel_for(
          "unpack_phi_j",
          n_subdomain_dofs,
          KOKKOS_LAMBDA(const int i) { phi_view(i) = lifted_view(i) + correction_view(i); });
      }
    Kokkos::fence();

    // Verification: recompute phi_j via the old sequential
    // vmult_fine_correction() loop the block path above replaces, and diff
    // against the block result. Kept permanently (not a one-shot check to
    // delete once confirmed) as a running correctness guard on the block
    // coarse-solve path, since compute_local_coarse_matrix() only runs
    // once per preconditioner setup -- the extra n_coarse_local sequential
    // solves cost comparatively little next to that.
    {
      std::vector<SubdomainVectorType> phi_ref(n_coarse_local);
      SubdomainVectorType               rhs_ref, correction_ref;
      rhs_ref.reinit(temp_subdomain_src);
      correction_ref.reinit(temp_subdomain_dst);

      Number max_phi_diff = 0;

      for (unsigned int j = 0; j < n_coarse_local; ++j)
        {
          phi_ref[j] = lifted_constraints[j];

          this->subdomain_bddc_operator.vmult_plain(rhs_ref, phi_ref[j]);
          rhs_ref *= Number(-1);
          this->subdomain_bddc_operator.project(rhs_ref);

          correction_ref = 0;
          this->vmult_fine_correction(correction_ref, rhs_ref);

          phi_ref[j].add(Number(1), correction_ref);

          SubdomainVectorType diff = phi_ref[j];
          diff -= coarse_basis_functions[j];
          max_phi_diff = std::max(max_phi_diff, diff.linfty_norm());
        }

      std::cout << "[block coarse-matrix verification] subdomain " << this_subdomain
               << ": max |phi_block - phi_sequential| = " << max_phi_diff << std::endl;
    }

    for (unsigned int j = 0; j < n_coarse_local; ++j)
      {
        SubdomainVectorType &phi_j = coarse_basis_functions[j];

        time.restart();
        this->subdomain_bddc_operator.vmult_plain(S_per_phi_j, phi_j);
        Kokkos::fence();
        setup_timings[1] += time.wall_time();

        time.restart();
        for (unsigned int k = 0; k < n_coarse_local; ++k)
          local_coarse_matrix(k, j) = lifted_constraints[k] * S_per_phi_j;
        Kokkos::fence();
        setup_timings[3] += time.wall_time();
      }
  }

  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::compute_coarse_matrix()
  {
    reset_setup_timings();

    const unsigned int n_coarse_global = this->n_global_coarse_dofs;
    this->coarse_matrix.reinit(n_coarse_global, n_coarse_global);

    LAPACKFullMatrix<Number> local_coarse_matrix;
    this->compute_local_coarse_matrix(local_coarse_matrix);

    std::vector<Number> local_global_contribution(n_coarse_global * n_coarse_global, Number(0));

    const unsigned int n_coarse_local  = this->n_local_coarse_dofs;
    const auto        &local_to_global = this->coarse_dofs_local_to_global_vector_host;

    Kokkos::fence();

    for (unsigned int i = 0; i < n_coarse_local; ++i)
      {
        const unsigned int global_idx_i = local_to_global[i];

        for (unsigned int j = 0; j < n_coarse_local; ++j)
          {
            const unsigned int global_idx_j = local_to_global[j];

            local_global_contribution[global_idx_i * n_coarse_global + global_idx_j] =
              local_coarse_matrix(i, j);
          }
      }

    Kokkos::fence();

    std::vector<Number> globally_summed_matrix(n_coarse_global * n_coarse_global, Number(0));

    Timer mpi_sum_time;
    Utilities::MPI::sum(local_global_contribution,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        globally_summed_matrix);
    setup_timings[4] += mpi_sum_time.wall_time();

    for (unsigned int i = 0; i < n_coarse_global; ++i)
      for (unsigned int j = 0; j < n_coarse_global; ++j)
        this->coarse_matrix(i, j) = globally_summed_matrix[i * n_coarse_global + j];

#ifdef DEBUG
    {
      LAPACKFullMatrix<Number> check_matrix;
      check_matrix = this->coarse_matrix;

      check_matrix.compute_eigenvalues(false, false);

      bool   is_spd     = true;
      Number min_eigenv = std::numeric_limits<Number>::max();

      for (unsigned int i = 0; i < n_coarse_global; ++i)
        {
          const Number eigv = check_matrix(i, i);
          if (eigv < min_eigenv)
            min_eigenv = eigv;

          if (eigv <= 1e-10)
            {
              is_spd = false;
            }
        }

      if (Utilities::MPI::this_mpi_process(this->subdomain_dof_handler->get_mpi_communicator()) ==
          0)
        {
          std::cout << "============================================" << std::endl;
          std::cout << "BDDC COARSE MATRIX CHARACTERISTICS:" << std::endl;
          std::cout << "Minimum Eigenvalue: " << min_eigenv << std::endl;
          std::cout << "Is Strictly Positive Definite? " << (is_spd ? "YES" : "NO") << std::endl;
          std::cout << "============================================" << std::endl;

          AssertThrow(is_spd, ExcMessage("BDDC Global Coarse Matrix is not positive definite!"));
        }
    }
#endif

    Timer lu_time;
    this->coarse_matrix.compute_lu_factorization();
    setup_timings[5] += lu_time.wall_time();
  }



  template <int dim, typename Number, typename BddcSmootherType>
  void
  BDDCPreconditioner<dim, Number, BddcSmootherType>::setup_primal_constraint_views()
  {
    const auto &dof_info          = this->subdomain_dof_handler->get_dof_info();
    const auto &local_constraints = dof_info.local_primal_constraints;

    std::vector<unsigned int> primal_constraint_dofs_interface_local_v;
    std::vector<unsigned int> primal_constraint_dofs_subdomain_v;
    std::vector<unsigned int> constraint_dofs_offsets_v;
    std::vector<unsigned int> corner_dofs_subdomain_v;

    coarse_dofs_local_to_global_vector_host.clear();

    std::vector<Number> coarse_weights_v;

    constraint_dofs_offsets_v.push_back(0);
    for (const auto &constraint : local_constraints)
      {
        if (bddc_variant == BDDCVariant::corner && constraint.type != PrimalConstraintType::Vertex)
          continue;

        if (bddc_variant == BDDCVariant::corner_edge &&
            constraint.type == PrimalConstraintType::Face)
          continue;

        if (constraint.type == PrimalConstraintType::Vertex)
          {
            AssertDimension(constraint.interface_partitioner_dofs_local.size(), 1);
            AssertDimension(constraint.local_subdomain_dofs.size(), 1);

            corner_dofs_subdomain_v.push_back(constraint.local_subdomain_dofs[0]);
          }

        primal_constraint_dofs_interface_local_v.insert(
          primal_constraint_dofs_interface_local_v.end(),
          constraint.interface_partitioner_dofs_local.begin(),
          constraint.interface_partitioner_dofs_local.end());

        primal_constraint_dofs_subdomain_v.insert(primal_constraint_dofs_subdomain_v.end(),
                                                  constraint.local_subdomain_dofs.begin(),
                                                  constraint.local_subdomain_dofs.end());


        constraint_dofs_offsets_v.push_back(primal_constraint_dofs_interface_local_v.size());

        coarse_dofs_local_to_global_vector_host.push_back(constraint.global_coarse_dof_index);

        Number weight = Number(1) / Number(constraint.interface_partitioner_dofs_local.size());

        AssertThrow(weight > 0, ExcInternalError());
        coarse_weights_v.push_back(weight);
      }

    AssertThrow(primal_constraint_dofs_interface_local_v.size() > 0, ExcInternalError());
    AssertThrow(primal_constraint_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(constraint_dofs_offsets_v.size() > 0, ExcInternalError());
    // AssertThrow(corner_dofs_subdomain_v.size() > 0, ExcInternalError());
    AssertThrow(coarse_dofs_local_to_global_vector_host.size() > 0, ExcInternalError());
    AssertThrow(coarse_weights_v.size() > 0, ExcInternalError());


    // Allocate and copy to Device Kokkos::Views
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_interface_local_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "primal_constraint_dofs_interface_local_host"),
      primal_constraint_dofs_interface_local_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_dofs_subdomain_host"),
      primal_constraint_dofs_subdomain_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_constraint_offsets_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_constraint_offsets_host"),
      constraint_dofs_offsets_v.size());
    Kokkos::View<unsigned int *, Kokkos::HostSpace> coarse_dofs_local_to_global_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_dofs_local_to_global_host"),
      coarse_dofs_local_to_global_vector_host.size());

    Kokkos::View<unsigned int *, Kokkos::HostSpace> corner_dofs_subdomain_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "corner_dofs_subdomain_host"),
      corner_dofs_subdomain_v.size());

    Kokkos::View<Number *, Kokkos::HostSpace> coarse_weights_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_weights_host"),
      coarse_weights_v.size());

    std::copy(primal_constraint_dofs_interface_local_v.begin(),
              primal_constraint_dofs_interface_local_v.end(),
              primal_constraint_dofs_interface_local_host.data());
    std::copy(primal_constraint_dofs_subdomain_v.begin(),
              primal_constraint_dofs_subdomain_v.end(),
              primal_constraint_dofs_subdomain_host.data());
    std::copy(constraint_dofs_offsets_v.begin(),
              constraint_dofs_offsets_v.end(),
              primal_constraint_constraint_offsets_host.data());
    std::copy(coarse_dofs_local_to_global_vector_host.begin(),
              coarse_dofs_local_to_global_vector_host.end(),
              coarse_dofs_local_to_global_host.data());
    std::copy(corner_dofs_subdomain_v.begin(),
              corner_dofs_subdomain_v.end(),
              corner_dofs_subdomain_host.data());
    std::copy(coarse_weights_v.begin(), coarse_weights_v.end(), coarse_weights_host.data());

    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

    this->primal_constraint_dofs_interface_local =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_interface_local_host);
    this->primal_constraint_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_subdomain_host);
    this->primal_constraint_offsets =
      Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_constraint_offsets_host);
    this->coarse_dofs_local_to_global =
      Kokkos::create_mirror_view_and_copy(exec_space, coarse_dofs_local_to_global_host);
    this->corner_dofs_subdomain =
      Kokkos::create_mirror_view_and_copy(exec_space, corner_dofs_subdomain_host);
    this->coarse_weights = Kokkos::create_mirror_view_and_copy(exec_space, coarse_weights_host);

    exec_space.fence();
  }


} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif