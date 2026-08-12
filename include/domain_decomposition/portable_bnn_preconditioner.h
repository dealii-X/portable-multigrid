#ifndef portable_bnn_preconditioner_h
#define portable_bnn_preconditioner_h


#include <set>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/subdomain_dof_handler.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename number>
  class BNNPreconditioner
  {
  public:
    BNNPreconditioner(const SchurInterfaceOperator<dim, number>       &interface_operator,
                      const SubdomainLaplaceOperatorBase<dim, number> &subdomain_operator,
                      const VCycleMultigridBase<dim, number>          &neumann_preconditioner);

    // Vanilla (non-enhanced) BNN preconditioner action -- combines
    // vmult_fine_correction() and vmult_coarse_correction() the same way
    // BDDCPreconditioner::vmult() combines its own fine and coarse
    // corrections, so both can be driven through
    // SolverProjectedCG::solve_dd() for a like-for-like comparison.
    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    // Local Neumann-Neumann correction: a subdomain Neumann solve with the
    // mean subtracted before/after (the subdomain problem is only defined
    // up to a constant when it has no physical boundary). Analogous to
    // BDDCPreconditioner::vmult_fine_correction(), which is likewise a
    // local, floating/constrained subdomain solve.
    void
    vmult_fine_correction(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    project(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
            const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    // Balancing step R_0^T*S_0^{-1}*R_0. Analogous to
    // BDDCPreconditioner::vmult_coarse_correction() (same role: the global
    // coarse-space correction), even though the two coarse spaces and coarse
    // solves are constructed differently.
    void
    vmult_coarse_correction(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    vmult_coarse_correction_dummy(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
      const bool                                                              computation_on,
      const bool communication_on) const;

    void
    vmult_coarse_correction_and_S_update(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &S_per_dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

    void
    setup_coarse_matrix();

    void
    coarse_to_global_interface(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
      const Vector<number>                                             &coarse_vector) const;



    void
    coarse_to_global_interface_and_S_update(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &S_per_interface_vector,
      const Vector<number>                                             &coarse_vector) const;


    // Enhanced BNN preconditioner action: returns both the correction z and
    // S*z (s_tilde) together, reusing the coarse basis functions' S*phi_i
    // precomputed in setup_coarse_matrix() instead of a fresh Dirichlet
    // solve for the coarse contribution to S*z -- the "spared Dirichlet
    // solve" enhancement over vmult().
    void
    vmult_and_S_update(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &z,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &s_tilde,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &r) const;

    void
    global_interface_to_coarse(
      Vector<number>                                                         &coarse_vector,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector)
      const;

    void
    reset_timings() const;

    const std::array<double, 4> &
    get_timings() const;

    unsigned int
    get_maximum_subdomain_mg_iterations() const;

    void
    vmult_interface(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const;

  private:
    struct NeumannSubdomainOperator
    {
      NeumannSubdomainOperator(const SubdomainLaplaceOperatorBase<dim, number> &op)
        : op(op)
      {}

      void
      vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
            const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
      {
        op.vmult_neumann(dst, src);
      }

      const SubdomainLaplaceOperatorBase<dim, number> &op;
    };

    ObserverPointer<const SchurInterfaceOperator<dim, number>>       interface_operator;
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, number>> subdomain_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;
    ObserverPointer<const VCycleMultigridBase<dim, number>>          neumann_preconditioner;

    LAPACKFullMatrix<number> coarse_matrix;

    const unsigned int n_subdomains;
    const unsigned int this_subdomain;

    const DeviceVector<const number> interface_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      physical_boundary_dof_indices_subdomain;

    NeumannSubdomainOperator subdomain_neumann_operator;

    // Subdomain-space scratch buffers for vmult_fine_correction()'s Neumann
    // solve -- this rank's own local piece, not to be confused with
    // temp_interface below (interface-partitioner-shaped).
    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      temp_subdomain_vector_src, temp_subdomain_vector_dst;

    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default> temp_interface, z0,
      S_z0;

    // Scratch for the vanilla vmult() combine.
    mutable LinearAlgebra::distributed::Vector<number, MemorySpace::Default> temp_vmult_fine,
      temp_vmult_coarse;

    // Global coarse indices whose coarse basis function phi_i has a
    // nonzero-support S*phi_i on this rank: this_subdomain itself, plus
    // every subdomain this rank shares an interface dof with. A single
    // vmult()'s cell loops only ever touch interface-adjacent cells (see
    // setup_coarse_matrix()), so S*phi_i never spreads further than that --
    // for any other global index, S_per_coarse_basis_local would just be
    // all zero on this rank, so it isn't computed or stored at all.
    std::vector<unsigned int> relevant_coarse_indices;

    // S_per_coarse_basis_local[m] == S * phi_{relevant_coarse_indices[m]},
    // already cross-rank-merged (post-compress()) at this rank's own
    // interface entries -- computed once in setup_coarse_matrix() and
    // linearly recombined at solve time in
    // coarse_to_global_interface_and_S_update() to reconstruct S*z0
    // without an extra Dirichlet solve.
    std::vector<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>
      S_per_coarse_basis_local;

    // Send/receive buffer for the allreduce in global_interface_to_coarse():
    // every rank contributes its own scalar at index this_subdomain (zero
    // elsewhere) and ends up with the fully assembled length-n_subdomains
    // vector -- replicated identically everywhere, so no follow-up
    // scatter/broadcast is needed by callers.
    mutable std::vector<number> temp_coarse_allreduce;

    mutable Vector<number> temp_coarse_rhs;
    mutable Vector<number> temp_coarse_solution;

    /**
     * timings[0] = Dirichlet solve (S application, via vmult_interface())
     * timings[1] = fine correction (local Neumann-Neumann solve)
     * timings[2] = coarse correction
     * timings[3] = projection step (project())
     */
    mutable std::array<double, 4> timings;

    // Max CG iterations across calls to vmult_fine_correction()'s Neumann
    // solve on this rank -- the counterpart to
    // SchurInterfaceOperator::get_maximum_subdomain_mg_iterations(), which
    // now only tracks the Dirichlet solve (shared with BDDC).
    mutable unsigned int max_subdomain_mg_iterations;
  };

  template <int dim, typename number>
  BNNPreconditioner<dim, number>::BNNPreconditioner(
    const SchurInterfaceOperator<dim, number>       &interface_operator,
    const SubdomainLaplaceOperatorBase<dim, number> &subdomain_operator,
    const VCycleMultigridBase<dim, number>          &neumann_preconditioner)
    : interface_operator(&interface_operator)
    , subdomain_operator(&subdomain_operator)
    , subdomain_dof_handler(&subdomain_operator.get_subdomain_dof_handler())
    , neumann_preconditioner(&neumann_preconditioner)
    , n_subdomains(subdomain_dof_handler->n_subdomains())
    , this_subdomain(subdomain_dof_handler->get_subdomain_id())
    , interface_weights(interface_operator.get_interface_weights())
    , interface_dof_indices_subdomain(subdomain_operator.get_interface_dof_indices_subdomain())
    , physical_boundary_dof_indices_subdomain(
        subdomain_operator.get_physical_boundary_dof_indices_subdomain())
    , subdomain_neumann_operator(subdomain_operator)
  {
    this->subdomain_operator->initialize_dof_vector(this->temp_subdomain_vector_src);
    this->subdomain_operator->initialize_dof_vector(this->temp_subdomain_vector_dst);

    temp_interface.reinit(this->subdomain_dof_handler->get_interface_vector_partitioner());

    z0.reinit(temp_interface);
    S_z0.reinit(temp_interface);
    temp_vmult_fine.reinit(temp_interface);
    temp_vmult_coarse.reinit(temp_interface);

    {
      std::set<unsigned int> one_hop_neighbors;

      const auto &partitioner = *this->subdomain_dof_handler->get_interface_vector_partitioner();
      for (const auto &target : partitioner.ghost_targets())
        one_hop_neighbors.insert(target.first);
      for (const auto &target : partitioner.import_targets())
        one_hop_neighbors.insert(target.first);

      // A direct neighbor m's own contribution to S*phi_i comes from m's
      // *own* local Dirichlet solve, whose Green's function is dense over
      // m's entire subdomain (elliptic solves have no compact support) --
      // so m's output is generally nonzero at *all* of m's own interface
      // dofs, not just the ones shared with i. Those land on m's other
      // neighbors via compress(), so the true support of S*phi_i is the
      // 2-hop neighborhood of i, not just its direct neighbors. Gather
      // every rank's own 1-hop neighbor list so each rank can compute its
      // 2-hop closure locally -- a small, one-time setup cost.
      const std::vector<unsigned int> own_one_hop_neighbors(one_hop_neighbors.begin(),
                                                            one_hop_neighbors.end());

      const std::vector<std::vector<unsigned int>> all_one_hop_neighbors =
        Utilities::MPI::all_gather(this->subdomain_dof_handler->get_mpi_communicator(),
                                   own_one_hop_neighbors);

      std::set<unsigned int> relevant;
      relevant.insert(this_subdomain);
      for (const auto &m : one_hop_neighbors)
        {
          relevant.insert(m);
          for (const auto &k : all_one_hop_neighbors[m])
            relevant.insert(k);
        }

      relevant_coarse_indices.assign(relevant.begin(), relevant.end());
    }

    S_per_coarse_basis_local.resize(relevant_coarse_indices.size());
    for (auto &basis_function : S_per_coarse_basis_local)
      basis_function.reinit(temp_interface);

    temp_coarse_allreduce.resize(n_subdomains);
    temp_coarse_rhs.reinit(n_subdomains);
    temp_coarse_solution.reinit(n_subdomains);

    max_subdomain_mg_iterations = 0;
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::reset_timings() const
  {
    for (unsigned int i = 0; i < timings.size(); ++i)
      timings[i] = 0.;
  }

  template <int dim, typename number>
  const std::array<double, 4> &
  BNNPreconditioner<dim, number>::get_timings() const
  {
    return timings;
  }

  template <int dim, typename number>
  unsigned int
  BNNPreconditioner<dim, number>::get_maximum_subdomain_mg_iterations() const
  {
    return max_subdomain_mg_iterations;
  }


  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_and_S_update(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &z,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &s_tilde,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &r) const
  {
    Assert(z.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(r.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));


    // local NN preconditioner
    this->vmult_fine_correction(z, r);

    this->vmult_interface(s_tilde, z);

    temp_interface = r;
    temp_interface -= s_tilde;

    Kokkos::fence();
    Timer time;

    this->vmult_coarse_correction_and_S_update(z0, S_z0, temp_interface);

    Kokkos::fence();
    timings[2] += time.wall_time();

    s_tilde += S_z0;
    z += z0;
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(
      dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    // Vanilla (non-enhanced) BNN preconditioner action, written as the
    // symmetric sandwich (Q := vmult_coarse_correction, S := the Schur
    // operator applied via vmult_interface(), both symmetric;
    // P := project() = I - Q S, so P^T = I - S Q since Q and S are
    // symmetric):
    //   M_BNN^{-1} = P * vmult_fine_correction * P^T + Q
    //              = (I - Q S) M_NN^{-1} (I - S Q) + Q
    // Both the left *and* right projection are required for M_BNN^{-1}
    // itself to be symmetric -- CG needs that, and dropping the right-hand
    // (I - S Q) factor (applying vmult_fine_correction to the raw residual
    // instead of to (I - S Q) r) breaks symmetry and stalls convergence.
    // That right-hand factor costs one extra Dirichlet solve here, plus
    // another one inside project() for the left-hand factor -- exactly the
    // per-application Dirichlet-solve cost vmult_and_S_update() (the
    // enhanced variant) avoids by reusing precomputed coarse-basis actions
    // instead, so comparing solve_dd()+vmult() against
    // solve_enhanced()+vmult_and_S_update() gives a direct vanilla-vs-
    // enhanced timing comparison.
    this->vmult_coarse_correction(temp_vmult_coarse, src);     // w = Q r
    this->vmult_interface(temp_vmult_fine, temp_vmult_coarse); // temp_vmult_fine = S w
    temp_interface = src;
    temp_interface -= temp_vmult_fine;                            // temp_interface = (I - S Q) r
    this->vmult_fine_correction(temp_vmult_fine, temp_interface); // z = M_NN^{-1} (I - S Q) r
    this->project(dst, temp_vmult_fine);                          // dst = (I - Q S) z
    dst += temp_vmult_coarse;                                     // dst += Q r
  }

  /**
   * Local Neumann-Neumann correction: y = D*S_NN^{-1}*D
   * D - interface weights diagonal matrix
   * S_NN^{-1} defines pseudoinverse action via subdomain Neumann solve
   * with imposing zero mean. In principle, imposing zero mean is not necessary
   * as the coarse correction step of BNN preconditioner ensures that the rhs is
   * compatible for Neumann solve, but it is still good to keep it for numerical
   * stability.
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_fine_correction(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(
      dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    Kokkos::fence();
    Timer time;

    src.update_ghost_values();

    dst = 0.;

    DeviceVector<number> src_view(src.get_values(), interface_dof_indices_subdomain.size()),
      dst_view(dst.get_values(), interface_dof_indices_subdomain.size());

    const auto &weights_view = interface_weights;

    DeviceVector<number> t_subdomain_src_view(temp_subdomain_vector_src.get_values(),
                                              temp_subdomain_vector_src.size()),
      t_subdomain_dst_view(temp_subdomain_vector_dst.get_values(),
                           temp_subdomain_vector_dst.size());

    const auto interface_dofs = this->interface_dof_indices_subdomain;

    // read src interface values and apply weights
    temp_subdomain_vector_src = 0;
    Kokkos::parallel_for(
      "read_src_subdomain_neumann", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        t_subdomain_src_view(interface_dofs(i)) = weights_view(i) * src_view(i);
      });


    if (physical_boundary_dof_indices_subdomain.size() == 0)
      {
        number mean_value_src = temp_subdomain_vector_src.mean_value();
        temp_subdomain_vector_src.add(-mean_value_src);
      }

    SolverControl solver_control(temp_subdomain_vector_src.size(), 1e-12 * src.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<number, MemorySpace::Default>> cg(solver_control);
    temp_subdomain_vector_dst = 0.;
    cg.solve(this->subdomain_neumann_operator,
             temp_subdomain_vector_dst,
             temp_subdomain_vector_src,
             *neumann_preconditioner);
    max_subdomain_mg_iterations =
      std::max(max_subdomain_mg_iterations, static_cast<unsigned int>(solver_control.last_step()));

    // neumann_preconditioner->vmult(temp_subdomain_vector_dst, temp_subdomain_vector_src);
    // max_subdomain_mg_iterations = std::max(max_subdomain_mg_iterations, 1u);

    if (physical_boundary_dof_indices_subdomain.size() == 0)
      {
        number mean_value_dst = temp_subdomain_vector_dst.mean_value();
        temp_subdomain_vector_dst.add(-mean_value_dst);
      }

    // apply weights and write dst interface values
    Kokkos::parallel_for(
      "write_dst_subdomain_neumann", interface_dofs.size(), KOKKOS_LAMBDA(const int i) {
        dst_view(i) = weights_view(i) * t_subdomain_dst_view(interface_dofs(i));
      });

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();

    Kokkos::fence();
    timings[1] += time.wall_time();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(
      dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));
    Assert(
      src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
      ExcMessage(
        "This function expects a vector initialized by SubdomainDoFHandler's interface vector partitioner."));

    Kokkos::fence();
    Timer time;
    this->interface_operator->vmult(dst, src);

    Kokkos::fence();
    timings[0] += time.wall_time();
  }

  /**
   * Schwarz projection step Id-(R_0^T*S_0^{-1}*R_0)*S :
   *    1. Apply interface operator S (maps values into flux)
   *    2. vmult_coarse_correction (i.e, apply R_0^T*S_0^{-1}*R_0) to retrive mean value
   *    3. Id-(R_0^T*S_0^{-1}*R_0)*S - gives compatible vector for subdomain
   * Neumann solve Result: retrieved values in the kernel space (i.e., mean
   * values) (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e.
   * compatible for the subdomain Neumann solve
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::project(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    temp_interface = 0.;

    Kokkos::fence();
    Timer time;
    // dst = S*tmp
    this->interface_operator->vmult(temp_interface, src);

    Kokkos::fence();
    const double time_dirichlet = time.wall_time();
    timings[0] += time_dirichlet;

    time.restart();
    // tmp = R0^T*S_0^{-1}*R0*dst
    this->vmult_coarse_correction(dst, temp_interface);

    Kokkos::fence();
    timings[2] += time.wall_time();

    dst.sadd(-1., src);

    Kokkos::fence();
    timings[3] += time.wall_time() + time_dirichlet;
  }

  /**
   * Coarse correction R_0^T*S_0^{-1}*R_0:
   *    1. project from global interface to coarse space
   *    2. solve coarse problem
   *    3. project back to the global interface
   * Result: retrieved values in the kernel space (i.e., mean values)
   * (Id - R_0^T*S_0^{-1}*R_0) -- returns balanced vector, i.e. compatible for
   * the subdomain Neumann solve
   *
   * The coarse matrix is replicated and factorized identically on every
   * rank (see setup_coarse_matrix()), so the coarse solve below runs
   * redundantly on every rank against an already-replicated RHS -- no
   * elected "coarse_problem_rank" gather/scatter/broadcast is needed
   * anywhere in this step.
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_coarse_correction(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    // project from global interface to coarse space (replicated on every rank)
    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    // solve coarse problem redundantly on every rank
    this->temp_coarse_solution = 0.;
    this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);

    // project back to the global interface space
    this->coarse_to_global_interface(dst, this->temp_coarse_solution);
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_coarse_correction_and_S_update(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &S_per_dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    // project from global interface to coarse space (replicated on every rank)
    this->global_interface_to_coarse(this->temp_coarse_rhs, src);

    // solve coarse problem redundantly on every rank
    this->temp_coarse_solution = 0.;
    this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);

    // project back to the global interface space
    this->coarse_to_global_interface_and_S_update(dst, S_per_dst, this->temp_coarse_solution);
  }


  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::vmult_coarse_correction_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool                                                              computation_on,
    const bool                                                              communication_on) const
  {
    Assert(dst.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
             interface vector partitioner."));
    Assert(src.get_partitioner() == this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("This function expects a vector initialized by SubdomainDoFHandler's \
            interface vector partitioner."));

    DeviceVector<number> src_view(src.get_values(), interface_dof_indices_subdomain.size()),
      dst_view(dst.get_values(), interface_dof_indices_subdomain.size());

    const auto &weights_view = interface_weights;

    // project from global interface to coarse space: local weighted
    // reduction (computation) followed by an allreduce (communication)
    if (computation_on)
      {
        number subdomain_coarse_value = 0.;
        Kokkos::parallel_reduce(
          "vmult_coarse_correction_dummy_local_reduce",
          interface_dof_indices_subdomain.size(),
          KOKKOS_LAMBDA(const unsigned int i, number &coarse_value) {
            coarse_value += src_view(i) * weights_view(i);
          },
          subdomain_coarse_value);

        std::fill(this->temp_coarse_allreduce.begin(),
                  this->temp_coarse_allreduce.end(),
                  number(0));
        this->temp_coarse_allreduce[this_subdomain] = subdomain_coarse_value;
      }

    if (communication_on)
      Utilities::MPI::sum(this->temp_coarse_allreduce,
                          this->subdomain_dof_handler->get_mpi_communicator(),
                          this->temp_coarse_allreduce);

    // solve coarse problem redundantly on every rank (computation only)
    if (computation_on)
      {
        for (unsigned int i = 0; i < this->n_subdomains; ++i)
          this->temp_coarse_rhs[i] = this->temp_coarse_allreduce[i];

        this->temp_coarse_solution = 0.;
        this->coarse_matrix.vmult(this->temp_coarse_solution, this->temp_coarse_rhs);
      }

    // project back to the global interface space: local lift (computation),
    // shared-dof reconciliation (communication)
    if (computation_on)
      {
        const number subdomain_coarse_value = this->temp_coarse_solution[this_subdomain];

        dst = 0.;
        Kokkos::parallel_for(
          "vmult_coarse_correction_dummy_local_lift",
          interface_dof_indices_subdomain.size(),
          KOKKOS_LAMBDA(const unsigned int i) {
            dst_view(i) = subdomain_coarse_value * weights_view(i);
          });
      }

    if (communication_on)
      dst.compress(VectorOperation::add);
  }


  /**
   * Projects a length-n_subdomains vector, already replicated identically
   * on every rank (as produced by global_interface_to_coarse() or a coarse
   * solve run redundantly everywhere), back onto this rank's own local
   * interface entries. this_subdomain's own entry is read directly -- no
   * scatter/broadcast needed, since the caller already has the full vector.
   */
  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::coarse_to_global_interface(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
    const Vector<number>                                             &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    AssertDimension(coarse_vector.size(), this->n_subdomains);

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_dof_indices_subdomain.size());

    const auto &weights_view = interface_weights;

    const number subdomain_coarse_value = coarse_vector[this_subdomain];

    // propagate coarse value to the interface by applying weights
    interface_vector = 0.;
    Kokkos::parallel_for(
      "SubdomainLaplaceOperator::coarse_to_subdomain_interface",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i) {
        interface_vector_view(i) = subdomain_coarse_value * weights_view(i);
      });

    // condense
    interface_vector.compress(VectorOperation::add);
    interface_vector.update_ghost_values();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::coarse_to_global_interface_and_S_update(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &S_per_interface_vector,
    const Vector<number>                                             &coarse_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));
    AssertDimension(coarse_vector.size(), this->n_subdomains);

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_dof_indices_subdomain.size());


    const auto &weights_view = interface_weights;

    const number subdomain_coarse_value = coarse_vector[this_subdomain];

    // propagate coarse value to the interface by applying weights
    interface_vector = 0.;
    Kokkos::parallel_for(
      "SubdomainLaplaceOperator::coarse_to_subdomain_interface",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i) {
        interface_vector_view(i) = subdomain_coarse_value * weights_view(i);
      });

    // condense
    interface_vector.compress(VectorOperation::add);

    // S * z0 = sum_i c_i * (S * phi_i). Only i == this_subdomain and i's
    // interface neighbors ever contribute a nonzero S*phi_i on this rank
    // (relevant_coarse_indices), and each S_per_coarse_basis_local entry
    // already holds the full cross-rank-merged value at this rank's own
    // interface entries, so this local weighted sum reproduces S*z0 exactly
    // without needing another Dirichlet solve or any communication here.
    S_per_interface_vector = 0;
    for (unsigned int m = 0; m < this->relevant_coarse_indices.size(); ++m)
      S_per_interface_vector.add(coarse_vector[this->relevant_coarse_indices[m]],
                                 this->S_per_coarse_basis_local[m]);

    S_per_interface_vector.compress(VectorOperation::add);
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::global_interface_to_coarse(
    Vector<number>                                                         &coarse_vector,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &interface_vector) const
  {
    Assert(interface_vector.get_partitioner() ==
             this->subdomain_dof_handler->get_interface_vector_partitioner(),
           ExcMessage("Interface vector is not initialized correctly."));


    interface_vector.update_ghost_values();

    DeviceVector<number> interface_vector_view(interface_vector.get_values(),
                                               interface_dof_indices_subdomain.size());

    const auto &weights_view = interface_weights;

    // retrieve subdomain coarse value by the interface weighted sum
    number subdomain_coarse_value = 0.;
    Kokkos::parallel_reduce(
      "global_interface_to_coarse",
      interface_dof_indices_subdomain.size(),
      KOKKOS_LAMBDA(const unsigned int i, number &coarse_value) {
        coarse_value += interface_vector_view(i) * weights_view(i);
      },
      subdomain_coarse_value);

    // Every rank contributes its own scalar at index this_subdomain (zero
    // elsewhere); summing via allreduce, rather than gathering to one
    // elected rank, leaves every rank holding the same fully assembled
    // length-n_subdomains vector directly, so callers never need a
    // separate scatter/broadcast afterwards.
    std::fill(this->temp_coarse_allreduce.begin(), this->temp_coarse_allreduce.end(), number(0));
    this->temp_coarse_allreduce[this_subdomain] = subdomain_coarse_value;

    Utilities::MPI::sum(this->temp_coarse_allreduce,
                        this->subdomain_dof_handler->get_mpi_communicator(),
                        this->temp_coarse_allreduce);

    AssertDimension(coarse_vector.size(), this->n_subdomains);
    for (unsigned int i = 0; i < this->n_subdomains; ++i)
      coarse_vector[i] = this->temp_coarse_allreduce[i];

    interface_vector.zero_out_ghost_values();
  }

  template <int dim, typename number>
  void
  BNNPreconditioner<dim, number>::setup_coarse_matrix()
  {
    coarse_matrix.reinit(this->n_subdomains, this->n_subdomains);

    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> phi_j(
      this->subdomain_dof_handler->get_interface_vector_partitioner()),
      S_phi_j(this->subdomain_dof_handler->get_interface_vector_partitioner());

    Vector<number> e_j(this->n_subdomains), coarse_column(this->n_subdomains);

    // relevant_coarse_indices is sorted ascending (built from a std::set in
    // the constructor), matching the ascending j below, so a single
    // linear walk picks out exactly the columns worth keeping.
    unsigned int next_relevant = 0;

    for (unsigned int j = 0; j < this->n_subdomains; ++j)
      {
        // phi_j is nonzero only via this_subdomain's own contribution (all
        // other ranks write zero everywhere), so e_j can be built directly
        // and identically on every rank -- no communication needed to
        // determine it, unlike the coarse system solve itself below.
        e_j    = 0.;
        e_j[j] = 1.;

        this->coarse_to_global_interface(phi_j, e_j);

        this->interface_operator->vmult(S_phi_j, phi_j);

        this->global_interface_to_coarse(coarse_column, S_phi_j);

        for (unsigned int i = 0; i < this->n_subdomains; ++i)
          coarse_matrix(i, j) = coarse_column[i];

        if (next_relevant < this->relevant_coarse_indices.size() &&
            this->relevant_coarse_indices[next_relevant] == j)
          {
            this->S_per_coarse_basis_local[next_relevant] = S_phi_j;
            ++next_relevant;
          }
      }

    AssertDimension(next_relevant, this->relevant_coarse_indices.size());

    coarse_matrix.compute_inverse_svd(1e-12);

    // std::cout << "Singular values of the coarse matrix: " << std::endl;
    // for (unsigned int i = 0; i < this->n_subdomains; ++i)
    //   std::cout << coarse_matrix.singular_value(i) << " ";
    // std::cout << std::endl;
  }

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE


#endif
