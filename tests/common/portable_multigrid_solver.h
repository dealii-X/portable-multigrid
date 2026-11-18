#ifndef portable_multigrid_solver_h
#define portable_multigrid_solver_h

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/multigrid/mg_base.h>

#include <type_traits>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
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

  // VectorType/TransferType: V-cycle (number) types, VectorTypeFine: outer CG
  // (number2) type. Defaults are the MemorySpace::Default ones.
  template <
    int dim,
    int fe_degree,
    typename number,
    typename number2,
    typename SmootherType,
    typename VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>,
    typename VectorTypeFine =
      LinearAlgebra::distributed::Vector<number2, MemorySpace::Default>,
    typename TransferType = Portable::MGTransferBase<dim, number>>
  class MultigridSolver
  {
    static_assert(std::is_same_v<typename VectorType::value_type, number>);
    static_assert(std::is_same_v<typename VectorTypeFine::value_type, number2>);

  public:
    using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorType>;
    using FineMatrixType  = Portable::LaplaceOperatorBase<dim, VectorTypeFine>;

    MultigridSolver(
      const std::unique_ptr<FineMatrixType>                 &fine_matrix,
      const MGLevelObject<DoFHandler<dim>>                  &level_dof_handlers,
      const MGLevelObject<AffineConstraints<number>>        &level_constraints,
      const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
      const MGLevelObject<std::unique_ptr<LevelMatrixType>> &level_matrices,
      const MGLevelObject<SmootherType>                     &smoothers,
      const VectorTypeFine                                  &right_hand_side,
      const unsigned int                                     degree_pre,
      const unsigned int                                     degree_post,
      const MGCoarseGridBase<VectorType>                    *coarse_solver = nullptr)
      : minlevel(level_dof_handlers.min_level())
      , maxlevel(level_dof_handlers.max_level())
      , fine_matrix(fine_matrix)
      , level_dof_handlers(level_dof_handlers)
      , level_constraints(level_constraints)
      , transfer(mg_transfers)
      , matrix(level_matrices)
      , smooth(smoothers)
      , solution(minlevel, maxlevel)
      , rhs(right_hand_side)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , coarse(smooth, false)
      , external_coarse(coarse_solver)
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
          matrix[level]->initialize_dof_vector(solution[level]);

          defect[level] = solution[level];
          t[level]      = solution[level];
        }

      fine_matrix->initialize_dof_vector(solution_fine);
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
          std::cout << "level  smoother    mg_mv     restrict  prolongate"
                    << std::endl;
          for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
            {
              std::cout << "L" << std::setw(2) << std::left << level << "    ";
              std::cout << std::setprecision(4) << std::setw(12)
                        << timings[level][5] << std::setw(10)
                        << timings[level][0] << std::setw(10)
                        << timings[level][1] << std::setw(12)
                        << timings[level][2] << std::endl;
            }
          std::cout << std::setprecision(5);
          for (unsigned int l = 0; l < timings.size(); ++l)
            for (unsigned int j = 0; j < timings[l].size(); ++j)
              timings[l][j] = 0.;
        }
    }

    void
    reset_timings()
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



      ReductionControl         solver_control(100, 1e-16, 1e-9);
      SolverCG<VectorTypeFine> solver_cg(solver_control);


      solution_fine = 0;
      solver_cg.solve(*fine_matrix, solution_fine, rhs, *this);

      solution[maxlevel].copy_locally_owned_data_from(solution_fine);

      return std::make_pair(solver_control.last_step(),
                            std::pow(solver_control.last_value() /
                                       solver_control.initial_value(),
                                     1. / solver_control.last_step()));
    }

    void
    vmult(VectorTypeFine &dst, const VectorTypeFine &src) const
    {
      fence();
      Timer time;


      for (unsigned int level = minlevel; level < maxlevel; ++level)
        {
          defect[level] = 0;
        }

      defect[maxlevel].copy_locally_owned_data_from(src);


      v_cycle(maxlevel);


      dst.copy_locally_owned_data_from(solution[maxlevel]);
      fence();

      // number of V-cycles and their total time, see print_wall_times()
      timings[minlevel][1] += 1;
      timings[minlevel][2] += time.wall_time();
    }

    void
    do_matvec()
    {
      fine_matrix->vmult(solution_fine, rhs);
    }

    void
    do_matvec_smoother()
    {
      matrix[maxlevel]->vmult(t[maxlevel], solution[maxlevel]);
    }

  private:
    /**
     * Kokkos::fence() for device vectors, no-op for host vectors.
     */
    static void
    fence()
    {
      if constexpr (!std::is_same_v<typename VectorType::memory_space,
                                    MemorySpace::Host>)
        Kokkos::fence();
    }

    /**
     * Implements the v-cycle
     */
    void
    v_cycle(const unsigned int level) const
    {
      // std::cout << "Entering v-cycle level " << level << std::endl;
      if (level == minlevel)
        {
          fence();
          Timer time;
          if (external_coarse != nullptr)
            (*external_coarse)(level, solution[level], defect[level]);
          else
            (coarse)(level, solution[level], defect[level]);
          fence();
          timings[level][0] += time.wall_time();

          // std::cout << "After coarse solve "  << level << std::endl;

          return;
        }

      Timer time;

      fence();
      time.restart();
      (smooth)[level].vmult(solution[level], defect[level]);
      // (smooth)[level].step(solution[level], defect[level]);
      fence();
      timings[level][5] += time.wall_time();

      fence();
      time.restart();
      (matrix)[level]->vmult(t[level], solution[level]);
      t[level].sadd(-1.0, 1.0, defect[level]);
      fence();
      timings[level][0] += time.wall_time();

      fence();
      time.restart();
      defect[level - 1] = 0;
      transfer[level]->restrict_and_add(defect[level - 1], t[level]);
      fence();
      timings[level][1] += time.wall_time();

      // std::cout << "After restrict_and_add " << level << std::endl;


      v_cycle(level - 1);


      fence();
      time.restart();
      transfer[level]->prolongate_and_add(solution[level], solution[level - 1]);
      fence();
      timings[level][2] += time.wall_time();

        // std::cout << "After prolongate_and_add " << level << std::endl;

      fence();
      time.restart();
      (smooth)[level].step(solution[level], defect[level]);
      fence();
      timings[level][5] += time.wall_time();

        // std::cout << "After post smooth " << level << std::endl;
    }

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    const std::unique_ptr<FineMatrixType> &fine_matrix;

    const MGLevelObject<DoFHandler<dim>>           &level_dof_handlers;
    const MGLevelObject<AffineConstraints<number>> &level_constraints;

    MGLevelObject<std::unique_ptr<TransferType>> transfer;

    /**
     * The matrix for each level.
     */
    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &matrix;

    /**
     * The smoother object.
     */
    const MGLevelObject<SmootherType> &smooth;

    VectorTypeFine solution_fine;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorType> solution;

    /**
     * Right hand side vector
     */
    const VectorTypeFine &rhs;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

    /**
     * Coarse solver passed to the constructor; replaces `coarse` if not null.
     */
    const MGCoarseGridBase<VectorType> *external_coarse;

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
