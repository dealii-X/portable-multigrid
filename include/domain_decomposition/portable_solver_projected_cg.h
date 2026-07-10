#ifndef portable_solver_projected_cg_h
#define portable_solver_projected_cg_h

#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <typename VectorType>
  class SolverProjectedCG : public SolverBase<VectorType>
  {
  public:
    /**
     * Declare type for container size.
     */
    using size_type = types::global_dof_index;

    /**
     * Constructor.
     */
    SolverProjectedCG(SolverControl &cn, VectorMemory<VectorType> &mem);

    /**
     * Constructor.
     */
    SolverProjectedCG(SolverControl &cn);

    /**
     * Virtual destructor.
     */
    virtual ~SolverProjectedCG() override = default;

    /**
     * Solve the linear system $Ax=b$ for x.
     */
    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType         &A,
          VectorType               &x,
          const VectorType         &b,
          const PreconditionerType &preconditioner);

    template <typename MatrixType, typename PreconditionerType>
    void
    solve_enhanced(const MatrixType         &A,
                   VectorType               &x,
                   const VectorType         &b,
                   const PreconditionerType &preconditioner);
  };



  template <typename VectorType>
  SolverProjectedCG<VectorType>::SolverProjectedCG(SolverControl &cn, VectorMemory<VectorType> &mem)
    : SolverBase<VectorType>(cn, mem)
  {}


  template <typename VectorType>
  SolverProjectedCG<VectorType>::SolverProjectedCG(SolverControl &cn)
    : SolverBase<VectorType>(cn)
  {}

  template <typename VectorType>
  template <typename MatrixType, typename PreconditionerType>
  void
  SolverProjectedCG<VectorType>::solve(const MatrixType         &A,
                                       VectorType               &x,
                                       const VectorType         &b,
                                       const PreconditionerType &preconditioner)
  {
    using number                      = typename VectorType::value_type;
    SolverControl::State solver_state = SolverControl::iterate;

    // Memory allocation
    typename VectorMemory<VectorType>::Pointer r_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer v_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer z_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer w_pointer(this->memory);


    VectorType &r = *r_pointer;
    VectorType &p = *p_pointer;
    VectorType &v = *v_pointer;
    VectorType &z = *z_pointer;
    VectorType &w = *w_pointer;


    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    r.reinit(x, true);
    p.reinit(x, true);
    v.reinit(x, true);
    z.reinit(x, true);
    w.reinit(x, true);

    int it = 0;

    number r_dot_preconditioner_dot_r = number();
    number beta                       = number();
    number alpha                      = number();


    if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
      preconditioner.balance(x, b);

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.sadd(-1., 1., b);
      }
    else
      r.equ(1., b);

    double residual_norm = r.l2_norm();
    solver_state         = this->iteration_status(0, residual_norm, x);


    if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
      preconditioner.reset_timings();

    if (solver_state != SolverControl::iterate)
      return;

    while (solver_state == SolverControl::iterate)
      {
        it++;

        const number old_r_dot_preconditioner_dot_r = r_dot_preconditioner_dot_r;

        if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
          {
            preconditioner.vmult(z, r);

            preconditioner.project(v, z);

            // preconditioner.balance(w, r);

            // preconditioner.vmult(z, r);

            // preconditioner.project(v, z);

            // v += w;

            r_dot_preconditioner_dot_r = r * v;
          }
        else
          r_dot_preconditioner_dot_r = residual_norm * residual_norm;

        const VectorType &direction =
          std::is_same<PreconditionerType, PreconditionIdentity>::value ? r : v;

        if (it > 1)
          {
            Assert(std::abs(old_r_dot_preconditioner_dot_r) != 0., ExcDivideByZero());

            beta = r_dot_preconditioner_dot_r / old_r_dot_preconditioner_dot_r;

            p.sadd(beta, 1., direction);
          }
        else
          p.equ(1., direction);

        // A.vmult(v, p);
        preconditioner.vmult_interface(v, p);

        const number p_dot_A_dot_p = p * v;
        Assert(std::abs(p_dot_A_dot_p) != 0., ExcDivideByZero());
        alpha = r_dot_preconditioner_dot_r / p_dot_A_dot_p;

        x.add(alpha, p);

        residual_norm = std::sqrt(std::abs(r.add_and_dot(-alpha, v, r)));

        if (A.enable_printing())
          std::cout << "residual_norm = " << residual_norm << std::endl;

        solver_state = this->iteration_status(it, residual_norm, x);
      }

    AssertThrow(solver_state == SolverControl::success,
                SolverControl::NoConvergence(it, residual_norm));
  }

  template <typename VectorType>
  template <typename MatrixType, typename PreconditionerType>
  void
  SolverProjectedCG<VectorType>::solve_enhanced(const MatrixType         &A,
                                                VectorType               &x,
                                                const VectorType         &b,
                                                const PreconditionerType &preconditioner)
  {
    using number                      = typename VectorType::value_type;
    SolverControl::State solver_state = SolverControl::iterate;

    // Memory allocation
    typename VectorMemory<VectorType>::Pointer r_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer v_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer z_pointer(this->memory);
    typename VectorMemory<VectorType>::Pointer s_tilde_pointer(this->memory);


    VectorType &r       = *r_pointer;
    VectorType &p       = *p_pointer;
    VectorType &v       = *v_pointer;
    VectorType &z       = *z_pointer;
    VectorType &s_tilde = *s_tilde_pointer;


    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    r.reinit(x, true);
    p.reinit(x, true);
    v.reinit(x, true);
    z.reinit(x, true);
    s_tilde.reinit(x, true);

    int it = 0;

    number r_dot_preconditioner_dot_r = number();
    number beta                       = number();
    number alpha                      = number();


    if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
      preconditioner.balance(x, b);

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.sadd(-1., 1., b);
      }
    else
      r.equ(1., b);

    double residual_norm = r.l2_norm();
    solver_state         = this->iteration_status(0, residual_norm, x);


    if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
      preconditioner.reset_timings();

    if (solver_state != SolverControl::iterate)
      return;

    while (solver_state == SolverControl::iterate)
      {
        it++;

        const number old_r_dot_preconditioner_dot_r = r_dot_preconditioner_dot_r;

        if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
          {
            // preconditioner.vmult(z, r);

            // preconditioner.project(v, z);

            // preconditioner.balance(w, r);

            // preconditioner.vmult(z, r);

            // preconditioner.project(v, z);

            // v += w;

            preconditioner.vmult_enhanced(z, s_tilde, r);

            r_dot_preconditioner_dot_r = r * z;
          }
        else
          r_dot_preconditioner_dot_r = residual_norm * residual_norm;

        const VectorType &direction =
          std::is_same<PreconditionerType, PreconditionIdentity>::value ? r : z;

        if (it > 1)
          {
            Assert(std::abs(old_r_dot_preconditioner_dot_r) != 0., ExcDivideByZero());

            beta = r_dot_preconditioner_dot_r / old_r_dot_preconditioner_dot_r;

            p.sadd(beta, 1., direction);
          }
        else
          p.equ(1., direction);

        // A.vmult(v, p);
        // preconditioner.vmult_interface(v, p);

        v.sadd(beta, 1., s_tilde);


        const number p_dot_A_dot_p = p * v;
        Assert(std::abs(p_dot_A_dot_p) != 0., ExcDivideByZero());
        alpha = r_dot_preconditioner_dot_r / p_dot_A_dot_p;

        x.add(alpha, p);

        residual_norm = std::sqrt(std::abs(r.add_and_dot(-alpha, v, r)));

        if (A.enable_printing())
          std::cout << "residual_norm = " << residual_norm << std::endl;

        solver_state = this->iteration_status(it, residual_norm, x);
      }

    AssertThrow(solver_state == SolverControl::success,
                SolverControl::NoConvergence(it, residual_norm));
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
