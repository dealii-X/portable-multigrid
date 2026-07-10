#ifndef portable_subdomain_v_cycle_multigrid_h
#define portable_subdomain_v_cycle_multigrid_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_smoother.h>

#include <Kokkos_Core.hpp>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim,
            typename number,
            typename LevelMatrixType,
            typename TransferType,
            typename SmootherType>
  class SubdomainVCycleMultigrid : public VCycleMultigridBase<dim, number>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

    SubdomainVCycleMultigrid(const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
                             const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
                             const MGLevelObject<SmootherType>                     &mg_smoothers,
                             const bool impose_zero_mean = false);

    void
    vmult(VectorType &dst, const VectorType &src) const override;

  private:
    /**
     * Implements the v-cycle
     */
    void
    v_cycle(const unsigned int level) const;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;


    /**
     * The matrix for each level.
     */
    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices;


    /**
     * The trasfer between each level.
     */
    const MGLevelObject<std::unique_ptr<TransferType>> &mg_transfers;


    /**
     * The smmother for each level.
     */

    const MGLevelObject<SmootherType> &mg_smoothers;

    /**
     * Whether to impose zero mean on the coarsest level. This is relevant for
     * purely Neumann problems.
     */
    const bool impose_zero_mean;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorType> solution;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;
  };

  template <int dim,
            typename number,
            typename LevelMatrixType,
            typename TransferType,
            typename SmootherType>
  SubdomainVCycleMultigrid<dim, number, LevelMatrixType, TransferType, SmootherType>::
    SubdomainVCycleMultigrid(const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
                             const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
                             const MGLevelObject<SmootherType>                     &mg_smoothers,
                             const bool impose_zero_mean)
    : minlevel(mg_matrices.min_level())
    , maxlevel(mg_matrices.max_level())
    , mg_matrices(mg_matrices)
    , mg_transfers(mg_transfers)
    , mg_smoothers(mg_smoothers)
    , impose_zero_mean(impose_zero_mean)
    , coarse(mg_smoothers, false)
    , solution(minlevel, maxlevel)
    , defect(minlevel, maxlevel)
    , t(minlevel, maxlevel)
  {
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        mg_matrices[level]->initialize_dof_vector(solution[level]);
        defect[level] = solution[level];
        t[level]      = solution[level];
      }
  }

  template <int dim,
            typename number,
            typename LevelMatrixType,
            typename TransferType,
            typename SmootherType>
  void
  SubdomainVCycleMultigrid<dim, number, LevelMatrixType, TransferType, SmootherType>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    defect[maxlevel] = src;

    v_cycle(maxlevel);

    dst = solution[maxlevel];
  }

  template <int dim,
            typename number,
            typename LevelMatrixType,
            typename TransferType,
            typename SmootherType>
  void
  SubdomainVCycleMultigrid<dim, number, LevelMatrixType, TransferType, SmootherType>::v_cycle(
    const unsigned int level) const
  {
    if (level == minlevel)
      {
        if (impose_zero_mean)
          {
            number mean_value = defect[level].mean_value();
            defect[level].add(-mean_value);
          }

        // Accuracy on coarsest level should be comparable to overall level
        // accuracy (~1e-3)
        (coarse)(level, solution[level], defect[level]);

        if (impose_zero_mean)
          {
            number mean_value = solution[level].mean_value();
            solution[level].add(-mean_value);
          }

        return;
      }

    // Pre-smoothing
    mg_smoothers[level].vmult(solution[level], defect[level]);

    // Compute residual
    mg_matrices[level]->vmult(t[level], solution[level]);
    t[level].sadd(-1.0, 1.0, defect[level]);

    // Restrict residual to the next coarser level
    defect[level - 1] = 0;
    mg_transfers[level]->restrict_and_add(defect[level - 1], t[level]);

    // Recursive call to v_cycle on the coarser level
    v_cycle(level - 1);

    // Prolongate coarse correction and add to current solution
    mg_transfers[level]->prolongate_and_add(solution[level], solution[level - 1]);

    // Post-smoothing
    mg_smoothers[level].step(solution[level], defect[level]);
  }


} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
