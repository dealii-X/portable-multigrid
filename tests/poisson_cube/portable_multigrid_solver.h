#ifndef portable_multigrid_solver_h
#define portable_multigrid_solver_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/multigrid/mg_base.h>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"


using namespace dealii;

namespace multigrid
{

  // A coarse solver defined via the smoother
  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const override
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };

  template <int dim, int fe_degree, typename number, typename SmootherType>
  class MultigridSolver
  {
  public:
    MultigridSolver(
      const MGLevelObject<DoFHandler<dim>>           &level_dof_handlers,
      const MGLevelObject<AffineConstraints<number>> &level_constraints,
      const MGLevelObject<
        std::unique_ptr<Portable::MGTransferBase<dim, number>>> &mg_transfers,
      const MGLevelObject<
        std::unique_ptr<Portable::LaplaceOperatorBase<dim, number>>>
                                        &level_matrices,
      const MGLevelObject<SmootherType> &smoothers,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                        &right_hand_side,
      const unsigned int degree_pre,
      const unsigned int degree_post)
      : minlevel(level_dof_handlers.min_level())
      , maxlevel(level_dof_handlers.max_level())
      , level_dof_handlers(level_dof_handlers)
      , level_constraints(level_constraints)
      , transfer(mg_transfers)
      , matrix(level_matrices)
      , smooth(smoothers)
      , solution(minlevel, maxlevel)
      , solution_host(minlevel, maxlevel)
      , rhs(right_hand_side)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , coarse(smooth, false)
      , degree_pre(degree_pre)
      , degree_post(degree_post)
      , timings(maxlevel + 1)
    {
      Assert(degree_post == degree_pre,
             ExcNotImplemented("Change of pre- and post-smoother degree "
                               "currently not possible with deal.II"));

      AssertDimension(fe_degree, level_dof_handlers.back().get_fe().degree);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          const auto &dof_handler = level_dof_handlers[level];

          IndexSet relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler);

          matrix[level]->initialize_dof_vector(solution[level]);
          defect[level] = solution[level];
          t[level]      = solution[level];

          solution_host[level].reinit(dof_handler.locally_owned_dofs(),
                                      relevant_dofs,
                                      dof_handler.get_mpi_communicator());
        }
    }
    // Print a summary of computation times on the various levels
    void
    print_wall_times()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Coarse solver " << (int)timings[minlevel][1]
                    << " times: " << timings[minlevel][0] << " tot prec "
                    << timings[minlevel][2] << std::endl;
          std::cout
            << "level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC"
            << std::endl;
          for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
            {
              std::cout << "L" << std::setw(2) << std::left << level << "    ";
              std::cout << std::setprecision(4) << std::setw(12)
                        << timings[level][5] << std::setw(10)
                        << timings[level][0] << std::setw(10)
                        << timings[level][4] << std::setw(10)
                        << timings[level][1] << std::setw(12)
                        << timings[level][2] << std::setw(10)
                        << timings[level][3] << std::endl;
            }
          std::cout << std::setprecision(5);
          for (unsigned int l = 0; l < timings.size(); ++l)
            for (unsigned int j = 0; j < timings[l].size(); ++j)
              timings[l][j] = 0.;
        }
    }

    void reset_timings()
    {
      for (unsigned int l = 0; l < timings.size(); ++l)
            for (unsigned int j = 0; j < timings[l].size(); ++j)
              timings[l][j] = 0.;
    }



    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per CG iteration
    std::pair<unsigned int, double>
    solve_cg()
    {
      reset_timings();

      ReductionControl           solver_control(100, 1e-16, 1e-9);
      SolverCG<VectorTypeDevice> solver_cg(solver_control);
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        solution_update = solution[maxlevel];
      solution_update   = 0;

      solver_cg.solve(*matrix[maxlevel], solution_update, rhs, *this);

      solution[maxlevel] = solution_update;

      return std::make_pair(solver_control.last_step(),
                            std::pow(solver_control.last_value() /
                                       solver_control.initial_value(),
                                     1. / solver_control.last_step()));
    }

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const
    {
      Timer time;


      for (unsigned int level = minlevel; level < maxlevel; ++level)
        {
          defect[level] = 0;
        }

      defect[maxlevel] = src;

      v_cycle(maxlevel);

      dst = solution[maxlevel];
    }

    void
    do_matvec()
    {
      matrix[maxlevel]->vmult(t[maxlevel], solution[maxlevel]);
    }

    void
    do_matvec_smoother()
    {
      matrix[maxlevel]->vmult(t[maxlevel], solution[maxlevel]);
    }

  private:
    /**
     * Implements the v-cycle
     */
    void
    v_cycle(const unsigned int level) const
    {
      
      if (level == minlevel)
        {
          Kokkos::fence();
          Timer time;
          (coarse)(level, solution[level], defect[level]);
          Kokkos::fence();
          timings[level][0] += time.wall_time();
          return;
        }
      Timer time;

      Kokkos::fence();
      time.restart();
      (smooth)[level].vmult(solution[level], defect[level]);
      // (smooth)[level].step(solution[level], defect[level]);
      Kokkos::fence();
      timings[level][5] += time.wall_time();


      Kokkos::fence();
      time.restart();
      (matrix)[level]->vmult(t[level], solution[level]);
      t[level].sadd(-1.0, 1.0, defect[level]);
      Kokkos::fence();
      timings[level][0] += time.wall_time();

      Kokkos::fence();
      time.restart();
      defect[level - 1] = 0;
      transfer[level]->restrict_and_add(defect[level - 1], t[level]);
      Kokkos::fence();
      timings[level][1] += time.wall_time();

      v_cycle(level - 1);

      Kokkos::fence();
      time.restart();
      transfer[level]->prolongate_and_add(solution[level], solution[level - 1]);
      Kokkos::fence();
      timings[level][2] += time.wall_time();

      Kokkos::fence();
      time.restart();
      (smooth)[level].step(solution[level], defect[level]);
      Kokkos::fence();
      timings[level][5] += time.wall_time();
    }

    // const ObserverPointer<const DoFHandler<dim>> dof_handler;


    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    const MGLevelObject<DoFHandler<dim>>           &level_dof_handlers;
    const MGLevelObject<AffineConstraints<number>> &level_constraints;


    MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, number>>>
      transfer;


    /**
     * The matrix for each level.
     */
    const MGLevelObject<
      std::unique_ptr<Portable::LaplaceOperatorBase<dim, number>>> &matrix;

    /**
     * The smoother object.
     */
    const MGLevelObject<SmootherType> &smooth;



    typedef LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      VectorTypeDevice;
    typedef LinearAlgebra::distributed::Vector<number, MemorySpace::Host>
      VectorTypeHost;


    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorTypeDevice> solution;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorTypeHost> solution_host;


    /**
     * Right hand side vector
     */
    VectorTypeDevice rhs;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorTypeDevice> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorTypeDevice> t;


    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorTypeDevice, MGLevelObject<SmootherType>> coarse;

    /**
     * Chebyshev degree for pre-smoothing
     */
    const unsigned int degree_pre;

    /**
     * Chebyshev degree for post-smoothing
     */
    const unsigned int degree_post;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<std::array<double, 6>> timings;
  };

} // namespace multigrid



#endif
