#ifndef portable_solver_projected_cg_h
#define portable_solver_projected_cg_h


#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>

#include "domain_decomposition/portable_interface_solver.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "operators/portable_subdomain_laplace_operator.h"

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
     * Standardized data struct to pipe additional data to the solver.
     * Here, it doesn't store anything but just exists for consistency
     * with the other solver classes.
     */
    // struct AdditionalData
    // {};

    /**
     * Constructor.
     */
    SolverProjectedCG(SolverControl &cn, VectorMemory<VectorType> &mem);
    //   const AdditionalData     &data = AdditionalData());

    /**
     * Constructor. Use an object of type GrowingVectorMemory as a default to
     * allocate memory.
     */
    SolverProjectedCG(SolverControl &cn);
    //   const AdditionalData &data = AdditionalData());

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

  private:
    /**
     * Additional parameters.
     */
    // AdditionalData additional_data;
  };


  // template <typename VectorType>
  // inline SolverProjectedCG<VectorType>::AdditionalData::AdditionalData()
  // {}

  template <typename VectorType>
  DEAL_II_CXX20_REQUIRES(concepts::is_vector_space_vector<VectorType>)
  SolverProjectedCG<VectorType>::SolverProjectedCG(
    SolverControl            &cn,
    VectorMemory<VectorType> &mem)
    // const AdditionalData     &data)
    : SolverBase<VectorType>(cn, mem)
  // , additional_data(data)
  {}


  template <typename VectorType>
  SolverProjectedCG<VectorType>::SolverProjectedCG(SolverControl &cn)
    //    const AdditionalData &data)
    : SolverBase<VectorType>(cn)
  // , additional_data(data)
  {}

  //   template <typename VectorType>
  //   template <typename MatrixType, typename PreconditionerType>
  //   void
  //   SolverProjectedCG<VectorType>::solve(const MatrixType         &S,
  //                                        VectorType               &u,
  //                                        const VectorType         &xi_global,
  //                                        const PreconditionerType
  //                                        &preconditioner)
  //   {
  //     using number                      = typename VectorType::value_type;
  //     SolverControl::State solver_state = SolverControl::iterate;

  //     // Memory allocation
  //     typename VectorMemory<VectorType>::Pointer r_pointer(this->memory);
  //     typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);
  //     typename VectorMemory<VectorType>::Pointer v_pointer(this->memory);
  //     typename VectorMemory<VectorType>::Pointer
  //     z_local_pointer(this->memory); typename
  //     VectorMemory<VectorType>::Pointer y_pointer(this->memory); typename
  //     VectorMemory<VectorType>::Pointer y_next_pointer(this->memory);



  //     // Define some aliases for simpler access, using the variables 'r' for
  //     the
  //     // residual b - A*x, 'p' for the search direction, and 'v' for the
  //     auxiliary
  //     // vector. This naming convention is used e.g. by the description on
  //     // https://en.wikipedia.org/wiki/Conjugate_gradient_method. The
  //     variable 'z'
  //     // gets only used for the flexible variant of the CG method.

  //     VectorType &r       = r_pointer;
  //     VectorType &v       = vasprintf;
  //     VectorType &p       = p_pointer;
  //     VectorType &y       = y_pointer;
  //     VectorType &y_next  = y_next_pointer;
  //     VectorType &z_local = z_local_pointer;



  //     // resize the vectors, but do not set the values since they'd be
  //     overwritten
  //     // soon anyway.
  //     r.reinit(u);
  //     p.reinit(u);
  //     y.reinit(u);
  //     v.reinit(u);
  //     y_next.reinit(u);
  //     z_local.reinit(u);


  //     int it = 0;

  //     number r_dot_preconditioner_dot_r = number();
  //     number beta                       = number();
  //     number alpha                      = number();


  //     preconditioner.project(u, xi_global);

  //     S.vmult(r, u);

  //     r.sadd(-1., 1., xi_global);

  //     preconditioner.vmult(z_local, r);

  //     S.vmult(v, p);

  //     preconditioner.project(p, v);

  //     // p.sadd(-1., 1., z);

  //     number rho = r * p;

  //     double residual_norm = r.l2_norm();
  //     solver_state         = this->iteration_status(0, residual_norm, x);

  //     if (solver_state != SolverControl::iterate)
  //       return;

  //     while (solver_state == SolverControl::iterate)
  //       {
  //         it++;

  //         // const number old_alpha = alpha;
  //         // const number old_r_dot_preconditioner_dot_r =
  //         //   r_dot_preconditioner_dot_r;

  //         // preconditioner.vmult(v, r);
  //         // r_dot_preconditioner_dot_r = r * v;

  //         // const VectorType &direction =
  //         //   std::is_same<PreconditionerType, PreconditionIdentity>::value
  //         ? r :
  //         //   v;

  //         // if (it > 1)
  //         //   {
  //         //     Assert(std::abs(old_r_dot_preconditioner_dot_r) != 0.,
  //         //            ExcDivideByZero());

  //         //     beta = r_dot_preconditioner_dot_r /
  //         //     old_r_dot_preconditioner_dot_r;

  //         //     p.sadd(beta, 1., direction);
  //         //   }
  //         // else
  //         //   p.equ(1., direction);

  //         // A.vmult(v, p);


  //         S.vmult(v, p);

  //         const number p_dot_A_dot_p = p * v;
  //         Assert(std::abs(p_dot_A_dot_p) != 0., ExcDivideByZero());
  //         alpha = r_dot_preconditioner_dot_r / p_dot_A_dot_p;

  //         x.add(alpha, p);
  //         residual_norm = std::sqrt(std::abs(r.add_and_dot(-alpha, v, r)));

  //         solver_state = this->iteration_status(it, residual_norm, x);
  //       }

  //     AssertThrow(solver_state == SolverControl::success,
  //                 SolverControl::NoConvergence(it, residual_norm));
  //   }



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



    VectorType &r = *r_pointer;
    VectorType &p = *p_pointer;
    VectorType &v = *v_pointer;
    VectorType &z = *z_pointer;



    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    r.reinit(x, true);
    p.reinit(x, true);
    v.reinit(x, true);
    z.reinit(x, true);

    int it = 0;

    number r_dot_preconditioner_dot_r = number();
    number beta                       = number();
    number alpha                      = number();

    preconditioner.balance(x, b);

    // x.print(std::cout);

    // compute residual. if vector is zero, then short-circuit the full
    // computation

    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.sadd(-1., 1., b);
      }
    else
      r.equ(1., b);

    // r.print(std::cout);



    double residual_norm = r.l2_norm();
    solver_state         = this->iteration_status(0, residual_norm, x);

    if (solver_state != SolverControl::iterate)
      return;

    while (solver_state == SolverControl::iterate)
      {
        it++;

        // const number old_alpha = alpha;
        const number old_r_dot_preconditioner_dot_r =
          r_dot_preconditioner_dot_r;

        if (std::is_same<PreconditionerType, PreconditionIdentity>::value ==
            false)
          {
            preconditioner.vmult(z, r);

            preconditioner.project(v, z);

            r_dot_preconditioner_dot_r = r * v;
          }
        else
          r_dot_preconditioner_dot_r = residual_norm * residual_norm;

        const VectorType &direction =
          std::is_same<PreconditionerType, PreconditionIdentity>::value ? r : v;

        if (it > 1)
          {
            Assert(std::abs(old_r_dot_preconditioner_dot_r) != 0.,
                   ExcDivideByZero());

            beta = r_dot_preconditioner_dot_r / old_r_dot_preconditioner_dot_r;

            p.sadd(beta, 1., direction);
          }
        else
          p.equ(1., direction);

        A.vmult(v, p);


        const number p_dot_A_dot_p = p * v;
        Assert(std::abs(p_dot_A_dot_p) != 0., ExcDivideByZero());
        alpha = r_dot_preconditioner_dot_r / p_dot_A_dot_p;
        // std::cout << "alpha = " << alpha << std::endl;

        x.add(alpha, p);

        // x.print(std::cout);
        residual_norm = std::sqrt(std::abs(r.add_and_dot(-alpha, v, r)));

        // std::cout << "residual_norm = " << residual_norm << std::endl;



        solver_state = this->iteration_status(it, residual_norm, x);
      }

    // AssertThrow(solver_state == SolverControl::success,
    //             SolverControl::NoConvergence(it, residual_norm));
  }



} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
