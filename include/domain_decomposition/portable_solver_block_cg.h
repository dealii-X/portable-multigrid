#ifndef portable_solver_block_cg_h
#define portable_solver_block_cg_h

#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  /**
   * Block-CG driver: solves n_rhs independent systems A*x_k = b_k
   * simultaneously (same A, same preconditioner -- as in
   * compute_local_coarse_matrix()'s n_local_coarse_dofs sequential
   * vmult_fine_correction() solves) in one pass, with x/b/r/p/v laid out
   * as n_rhs blocks of dof_stride each -- the same convention
   * bk3_kokkos_kernel_block.h's KokkosKernelBlock() uses, so A and
   * the preconditioner passed in here are expected to operate on that
   * same block layout.
   *
   * Plain CG, not a projected variant: modeled on
   * BDDCPreconditioner::vmult_fine_correction(), which solves with plain
   * SolverCG rather than SolverProjectedCG. Ahat = Pi*A and Chat =
   * Pi*D^{-1} (or the MG V-cycle preconditioner, once its own block
   * version exists) already project their own *output* into V =
   * range(Pi) internally, so once the initial RHS is in V, every CG
   * iterate (r, p, v) stays in V for free -- no per-iteration project()
   * calls needed here. The caller is responsible for projecting b
   * before calling solve_block(), exactly as vmult_fine_correction()'s
   * callers project fine_residual before passing it in.
   *
   * Every scalar CG coefficient (r.v, p.Ap, alpha, beta) becomes an
   * n_rhs-length device array instead of a single number, computed via
   * block_dot()/block_axpy()/block_sadd()/block_add_and_dot() below
   * instead of VectorType's own dot()/add()/sadd()/add_and_dot().
   *
   * Deliberately NOT a general block-Krylov method: there is no
   * per-column adaptive convergence check here (that would need masking
   * out already-converged columns, a K x K Gram-matrix solve per
   * iteration, etc. -- see the bddc-preconditioner session's block-solve
   * discussion for why that's a bigger, separate step). Instead, every
   * column keeps iterating together, and the SolverControl passed to
   * the constructor is checked against the l2 norm of r treated as one
   * big vector of all n_rhs columns concatenated -- i.e. exactly what
   * VectorType::l2_norm() would give if called on r directly, matching
   * solve()'s own residual_norm = r.l2_norm(). Since that combined norm
   * is sqrt(sum_k ||r_k||^2) >= max_k ||r_k||, it's a slightly stricter
   * criterion than stopping on the worst column alone -- every column
   * ends up at least as converged as the tolerance, typically more so.
   * This wastes some extra iterations on columns that individually
   * would have stopped earlier, but that's a good trade given
   * dirichlet_mg_its/bddc_mg_its already cluster tightly (13-18) across
   * very different local coarse-dof counts in the production runs that
   * motivated this class -- the columns don't disagree enough on "when
   * to stop" for that to cost much.
   */
  template <typename VectorType>
  class SolverBlockCG : public SolverBase<VectorType>
  {
  public:
    using size_type = types::global_dof_index;
    using number     = typename VectorType::value_type;

    SolverBlockCG(SolverControl &cn, VectorMemory<VectorType> &mem)
      : SolverBase<VectorType>(cn, mem)
    {}

    explicit SolverBlockCG(SolverControl &cn)
      : SolverBase<VectorType>(cn)
    {}

    /**
     * x, b are block vectors of size n_rhs * dof_stride (dof_stride =
     * x.size() / n_rhs). A.vmult()/preconditioner.vmult() must both
     * operate on that same block layout (see
     * bk3_kokkos_kernel_block.h's KokkosKernelBlock() indexing
     * convention: block k occupies [k*dof_stride, (k+1)*dof_stride)).
     * Precondition: b must already be projected into V (see class
     * comment) -- this function does not project it.
     */
    template <typename MatrixType, typename PreconditionerType>
    void
    solve_block(const MatrixType         &A,
                 VectorType               &x,
                 const VectorType         &b,
                 const PreconditionerType &preconditioner,
                 const unsigned int        n_rhs);

  private:
    using DeviceArray = Kokkos::View<number *, MemorySpace::Default::kokkos_space>;

    // result(k) = sum_i a[k*stride+i] * b[k*stride+i], one reduction per
    // block computed in a single kernel launch (team per block).
    static DeviceArray
    block_dot(const number *a, const number *b, const unsigned int n_rhs, const unsigned int stride);

    // r[k*stride+i] -= alpha(k) * v[k*stride+i], fused with computing
    // result(k) = sum_i r[k*stride+i]^2 (the *updated* r) in the same
    // pass -- the block equivalent of VectorType::add_and_dot(), which
    // solve() uses for exactly this update+norm step rather than two
    // separate passes over r.
    static DeviceArray
    block_add_and_dot(number             *r,
                      const DeviceArray  &alpha,
                      const number       *v,
                      const unsigned int  n_rhs,
                      const unsigned int  stride);

    // sqrt(sum_k squared_norms(k)) -- the l2 norm of the full n_rhs-block
    // vector these per-block squared norms came from, fed to
    // SolverBase::iteration_status() each step. n_rhs is small (7-17 in
    // practice), so reducing on the host is simpler than a second
    // device-side reduction and costs nothing measurable next to
    // A.vmult()/preconditioner.vmult().
    static number
    combined_l2_norm(const DeviceArray &squared_norms, const unsigned int n_rhs);

    // x[k*stride+i] += alpha(k) * p[k*stride+i]
    static void
    block_axpy(number       *x,
              const DeviceArray &alpha,
              const number *p,
              const unsigned int n_rhs,
              const unsigned int stride);

    // dst[k*stride+i] = beta(k) * dst[k*stride+i] + src[k*stride+i]
    static void
    block_sadd(number       *dst,
              const DeviceArray &beta,
              const number *src,
              const unsigned int n_rhs,
              const unsigned int stride);

    // dst(k) = num(k) / den(k)
    static DeviceArray
    block_divide(const DeviceArray &num, const DeviceArray &den, const unsigned int n_rhs);
  };



  template <typename VectorType>
  typename SolverBlockCG<VectorType>::DeviceArray
  SolverBlockCG<VectorType>::block_dot(const number      *a,
                                         const number      *b,
                                         const unsigned int n_rhs,
                                         const unsigned int stride)
  {
    DeviceArray result(Kokkos::view_alloc("block_dot_result", Kokkos::WithoutInitializing), n_rhs);

    using TeamPolicy = Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;
    using TeamHandle = typename TeamPolicy::member_type;

    Kokkos::parallel_for(
      TeamPolicy(n_rhs, Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamHandle &team) {
        const unsigned int k      = team.league_rank();
        const unsigned int offset = k * stride;

        number block_sum = 0;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, stride),
          [&](const unsigned int i, number &sum) { sum += a[offset + i] * b[offset + i]; },
          block_sum);

        Kokkos::single(Kokkos::PerTeam(team), [&]() { result(k) = block_sum; });
      });
    Kokkos::fence();

    return result;
  }

  template <typename VectorType>
  typename SolverBlockCG<VectorType>::DeviceArray
  SolverBlockCG<VectorType>::block_add_and_dot(number             *r,
                                                 const DeviceArray  &alpha,
                                                 const number       *v,
                                                 const unsigned int  n_rhs,
                                                 const unsigned int  stride)
  {
    DeviceArray result(Kokkos::view_alloc("block_add_and_dot_result", Kokkos::WithoutInitializing),
                       n_rhs);

    using TeamPolicy = Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>;
    using TeamHandle = typename TeamPolicy::member_type;

    Kokkos::parallel_for(
      TeamPolicy(n_rhs, Kokkos::AUTO),
      KOKKOS_LAMBDA(const TeamHandle &team) {
        const unsigned int k      = team.league_rank();
        const unsigned int offset = k * stride;
        const number       a      = alpha(k);

        number block_sum = 0;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, stride),
          [&](const unsigned int i, number &sum) {
            const number updated_r = r[offset + i] - a * v[offset + i];
            r[offset + i]          = updated_r;
            sum += updated_r * updated_r;
          },
          block_sum);

        Kokkos::single(Kokkos::PerTeam(team), [&]() { result(k) = block_sum; });
      });
    Kokkos::fence();

    return result;
  }

  template <typename VectorType>
  typename SolverBlockCG<VectorType>::number
  SolverBlockCG<VectorType>::combined_l2_norm(const DeviceArray &squared_norms,
                                                const unsigned int n_rhs)
  {
    auto squared_norms_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), squared_norms);

    number total_squared_norm = 0;
    for (unsigned int k = 0; k < n_rhs; ++k)
      total_squared_norm += squared_norms_host(k);

    return std::sqrt(total_squared_norm);
  }

  template <typename VectorType>
  void
  SolverBlockCG<VectorType>::block_axpy(number             *x,
                                          const DeviceArray  &alpha,
                                          const number       *p,
                                          const unsigned int  n_rhs,
                                          const unsigned int  stride)
  {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, static_cast<std::size_t>(n_rhs) * stride),
      KOKKOS_LAMBDA(const std::size_t idx) {
        const unsigned int k = idx / stride;
        x[idx] += alpha(k) * p[idx];
      });
    Kokkos::fence();
  }

  template <typename VectorType>
  void
  SolverBlockCG<VectorType>::block_sadd(number             *dst,
                                          const DeviceArray  &beta,
                                          const number       *src,
                                          const unsigned int  n_rhs,
                                          const unsigned int  stride)
  {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, static_cast<std::size_t>(n_rhs) * stride),
      KOKKOS_LAMBDA(const std::size_t idx) {
        const unsigned int k = idx / stride;
        dst[idx]             = beta(k) * dst[idx] + src[idx];
      });
    Kokkos::fence();
  }

  template <typename VectorType>
  typename SolverBlockCG<VectorType>::DeviceArray
  SolverBlockCG<VectorType>::block_divide(const DeviceArray &num,
                                            const DeviceArray &den,
                                            const unsigned int n_rhs)
  {
    DeviceArray result(Kokkos::view_alloc("block_divide_result", Kokkos::WithoutInitializing),
                       n_rhs);

    Kokkos::parallel_for(
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0, n_rhs),
      KOKKOS_LAMBDA(const unsigned int k) {
        Assert(den(k) != number(), ExcDivideByZero());
        result(k) = num(k) / den(k);
      });
    Kokkos::fence();

    return result;
  }



  template <typename VectorType>
  template <typename MatrixType, typename PreconditionerType>
  void
  SolverBlockCG<VectorType>::solve_block(const MatrixType         &A,
                                             VectorType               &x,
                                             const VectorType         &b,
                                             const PreconditionerType &preconditioner,
                                             const unsigned int        n_rhs)
  {
    AssertDimension(x.size(), b.size());
    AssertDimension(x.size() % n_rhs, 0u);
    const unsigned int dof_stride = x.size() / n_rhs;

    typename VectorMemory<VectorType>::Pointer r_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer v_pointer(this->memory);

    VectorType &r = *r_pointer;
    VectorType &p = *p_pointer;
    VectorType &v = *v_pointer;

    r.reinit(x, true);
    p.reinit(x, true);
    v.reinit(x, true);

    number *x_ptr = x.get_values();
    number *r_ptr = r.get_values();
    number *p_ptr = p.get_values();
    number *v_ptr = v.get_values();

    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.sadd(-1., 1., b);
      }
    else
      r.equ(1., b);

    int it = 0;

    number residual_norm = combined_l2_norm(block_dot(r_ptr, r_ptr, n_rhs, dof_stride), n_rhs);

    SolverControl::State solver_state = this->iteration_status(it, residual_norm, x);
    if (solver_state != SolverControl::iterate)
      return;

    DeviceArray r_dot_preconditioner_dot_r(
      Kokkos::view_alloc("r_dot_preconditioner_dot_r", Kokkos::WithoutInitializing), n_rhs);
    Kokkos::deep_copy(r_dot_preconditioner_dot_r, number());

    constexpr bool use_identity = std::is_same<PreconditionerType, PreconditionIdentity>::value;

    while (solver_state == SolverControl::iterate)
      {
        it++;

        const DeviceArray old_r_dot_preconditioner_dot_r = r_dot_preconditioner_dot_r;

        if constexpr (!use_identity)
          {
            preconditioner.vmult(v, r);
            r_dot_preconditioner_dot_r = block_dot(r_ptr, v_ptr, n_rhs, dof_stride);
          }
        else
          r_dot_preconditioner_dot_r = block_dot(r_ptr, r_ptr, n_rhs, dof_stride);

        const number *direction_ptr = use_identity ? r_ptr : v_ptr;

        if (it > 1)
          {
            const DeviceArray beta = block_divide(r_dot_preconditioner_dot_r,
                                                  old_r_dot_preconditioner_dot_r,
                                                  n_rhs);
            block_sadd(p_ptr, beta, direction_ptr, n_rhs, dof_stride);
          }
        else
          Kokkos::deep_copy(
            Kokkos::View<number *, MemorySpace::Default::kokkos_space>(p_ptr,
                                                                       static_cast<std::size_t>(
                                                                         n_rhs) *
                                                                         dof_stride),
            Kokkos::View<const number *, MemorySpace::Default::kokkos_space>(
              direction_ptr, static_cast<std::size_t>(n_rhs) * dof_stride));

        A.vmult(v, p);

        const DeviceArray p_dot_A_dot_p = block_dot(p_ptr, v_ptr, n_rhs, dof_stride);
        const DeviceArray alpha = block_divide(r_dot_preconditioner_dot_r, p_dot_A_dot_p, n_rhs);

        block_axpy(x_ptr, alpha, p_ptr, n_rhs, dof_stride);

        // r -= alpha * v, fused with computing the updated r's squared
        // norms -- matches solve()'s r.add_and_dot(-alpha, v, r).
        const DeviceArray r_squared_norms = block_add_and_dot(r_ptr, alpha, v_ptr, n_rhs, dof_stride);

        residual_norm = combined_l2_norm(r_squared_norms, n_rhs);
        solver_state   = this->iteration_status(it, residual_norm, x);
      }

    AssertThrow(solver_state == SolverControl::success,
               SolverControl::NoConvergence(it, residual_norm));
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
