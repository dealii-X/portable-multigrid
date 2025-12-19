#ifndef portable_v_cycle_multigrid_h
#define portable_v_cycle_multigrid_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_smoother.h>

#include <Kokkos_Core.hpp>

#include "base/portable_laplace_operator_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename number, typename TransferType>
  class VCycleMultigrid : public EnableObserverPointer
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;
    using LevelMatrixType = LaplaceOperatorBase<dim, number>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorType>;

    VCycleMultigrid(
      const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
      const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
      const MGLevelObject<SmootherType>                     &mg_smoothers,
      const unsigned int pre_smoothing_steps,
      const unsigned int post_smoothing_steps);

    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    void
    smooth(VectorType        &u,
           const VectorType  &rhs,
           const unsigned int level) const;

    void
    v_cycle(VectorType        &dst,
            const VectorType  &src,
            const unsigned int level) const;

    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices;
    const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers;

    const MGLevelObject<SmootherType> &mg_smoothers;

    const unsigned int pre_smoothing_steps;
    const unsigned int post_smoothing_steps;
  };

  template <int dim, typename number, typename TransferType>
  VCycleMultigrid<dim, number, TransferType>::VCycleMultigrid(
    const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
    const MGLevelObject<std::unique_ptr<TransferType>>    &mg_transfers,
    const MGLevelObject<SmootherType>                     &mg_smoothers,
    const unsigned int                                     pre_smoothing_steps,
    const unsigned int                                     post_smoothing_steps)
    : mg_matrices(mg_matrices)
    , mg_transfers(mg_transfers)
    , mg_smoothers(mg_smoothers)
    , pre_smoothing_steps(pre_smoothing_steps)
    , post_smoothing_steps(post_smoothing_steps)
  {}

  template <int dim, typename number, typename TransferType>
  void
  VCycleMultigrid<dim, number, TransferType>::vmult(VectorType       &dst,
                                                    const VectorType &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() ==
             mg_matrices.back()->get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() ==
             mg_matrices.back()->get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    dst = 0;
    v_cycle(dst, src, mg_matrices.max_level());
  }

  template <int dim, typename number, typename TransferType>
  void
  VCycleMultigrid<dim, number, TransferType>::smooth(
    VectorType        &u,
    const VectorType  &rhs,
    const unsigned int level) const
  {
    Assert(level >= mg_matrices.min_level() && level <= mg_matrices.max_level(),
           ExcMessage("Level out of range"));

    AssertDimension(u.size(), rhs.size());

    Assert(u.get_partitioner() == mg_matrices[level]->get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    Assert(rhs.get_partitioner() ==
             mg_matrices[level]->get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));


    VectorType r, d;
    r.reinit(u, true);
    d.reinit(u, true);

    mg_matrices[level]->vmult(r, u);
    r.sadd(-1., rhs);

    mg_smoothers[level].vmult(d, r);

    u += d;
  }

  template <int dim, typename number, typename TransferType>
  void
  VCycleMultigrid<dim, number, TransferType>::v_cycle(
    VectorType        &dst,
    const VectorType  &src,
    const unsigned int level) const
  {
    {
      Assert(level >= mg_matrices.min_level() &&
               level <= mg_matrices.max_level(),
             ExcMessage("Level out of range"));

      AssertDimension(dst.size(), src.size());
      Assert(dst.get_partitioner() ==
               mg_matrices[level]->get_vector_partitioner(),
             ExcMessage("Vector is not correctly initialized."));
      Assert(src.get_partitioner() ==
               mg_matrices[level]->get_vector_partitioner(),
             ExcMessage("Vector is not correctly initialized."));

      if (level == mg_matrices.min_level())
        {
          // Accuracy on coarsest level should be comparable to overall level
          // accuracy (~1e-3)
          smooth(dst, src, mg_matrices.min_level());
          return;
        }

      // Pre-smoothing
      for (unsigned int step = 0; step < pre_smoothing_steps; ++step)
        {
          smooth(dst, src, level);
        }

      // Compute residual
      VectorType residual;
      mg_matrices[level]->initialize_dof_vector(residual);
      mg_matrices[level]->vmult(residual, dst);
      residual.sadd(-1., 1., src); // residual = src - A * dst

      // Restrict residual to the next coarser level
      VectorType coarse_residual;
      mg_matrices[level - 1]->initialize_dof_vector(coarse_residual);

      mg_transfers[level]->restrict_and_add(coarse_residual, residual);

      // Initialize coarse correction
      VectorType coarse_correction;
      mg_matrices[level - 1]->initialize_dof_vector(coarse_correction);

      // Recursive call to v_cycle on the coarser level
      v_cycle(coarse_correction, coarse_residual, level - 1);

      // Prolongate coarse correction and add to current solution
      mg_transfers[level]->prolongate_and_add(dst, coarse_correction);

      // Post-smoothing
      for (unsigned int step = 0; step < post_smoothing_steps; ++step)
        {
          smooth(dst, src, level);
        }
    }
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif