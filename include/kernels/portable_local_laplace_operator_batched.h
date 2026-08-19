#ifndef portable_local_laplace_operator_batched_h
#define portable_local_laplace_operator_batched_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include "matrix_free/portable_batched_fe_evaluation.h"
#include "matrix_free/portable_evaluation_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LocalLaplaceOperatorBatched
  {
  public:
    static constexpr unsigned int n_local_dofs_1d = fe_degree + 1;

    LocalLaplaceOperatorBatched() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const Custom::Parallel::BatchData<dim, Number> *data,
               const Custom::Parallel::DeviceView<Number>     &src,
               Custom::Parallel::DeviceView<Number>           &dst) const;
  };

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  DEAL_II_HOST_DEVICE void
  LocalLaplaceOperatorBatched<dim, fe_degree, n_q_points_1d, Number>::operator()(
    const Custom::Parallel::BatchData<dim, Number> *data,
    const Custom::Parallel::DeviceView<Number>     &src,
    Custom::Parallel::DeviceView<Number>           &dst) const
  {
    const Custom::Parallel::
      FEEvaluationImplTransformToCollocation<dim, n_local_dofs_1d, n_q_points_1d, Number>
        fe_eval(data->team_member,
                data->shape_values,
                data->co_shape_gradients,
                data->nelmtPerBatch,
                data->c_nelmtPerBatch,
                data->batchIdx,
                data->threadIdx,
                data->blockSize);

    Custom::Parallel::read_dof_values<dim, n_local_dofs_1d>(data->team_member,
                                                            src,
                                                            data->dof_indices,
                                                            data->values,
                                                            data->batchIdx,
                                                            data->nelmtPerBatch,
                                                            data->cell_range_ids,
                                                            data->c_nelmtPerBatch,
                                                            data->threadIdx,
                                                            data->blockSize);

    fe_eval.evaluate_values(data->values, data->values, data->scratch);

    fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(data->G_tensor,
                                                             data->cell_range_ids,
                                                             data->values,
                                                             data->gradients);

    fe_eval.integrate_gradients(data->gradients, data->values);

    fe_eval.integrate_values(data->values, data->values, data->scratch);

    Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(data->team_member,
                                                                       data->values,
                                                                       dst,
                                                                       data->dof_indices,
                                                                       data->batchIdx,
                                                                       data->nelmtPerBatch,
                                                                       data->cell_range_ids,
                                                                       data->c_nelmtPerBatch,
                                                                       data->threadIdx,
                                                                       data->blockSize);
  }



  // Batched analog of step-64's LocalHelmholtzOperator, minus the mass term,
  // built purely from Custom::Parallel::FEEvaluation's deal.II-Portable::
  // FEEvaluation-faithful accessor API (read_dof_values()/evaluate(flags)/
  // get_gradient()/submit_gradient()/integrate(flags)/
  // distribute_local_to_global()) -- the on-the-fly inv_jacobian/JxW
  // geometric-factor path, not LocalLaplaceOperatorBatched's G_tensor-fused
  // evaluate_gradients_and_multiply_symmetric_tensor() above. Same math,
  // same operator, different code path -- see
  // correctness_tests/check_correctness_batched_fe_evaluation for the
  // bit-identical-to-machine-precision cross-check against vmult().
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LocalLaplaceOperatorGeneric
  {
  public:
    LocalLaplaceOperatorGeneric() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const Custom::Parallel::BatchData<dim, Number> *data,
               const Custom::Parallel::DeviceView<Number>     &src,
               Custom::Parallel::DeviceView<Number>           &dst) const
    {
      Custom::Parallel::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      data->for_each_quad_point(
        [&](const int point) { fe_eval.submit_gradient(fe_eval.get_gradient(point), point); });

      fe_eval.integrate(EvaluationFlags::gradients);

      fe_eval.distribute_local_to_global(dst);
    }
  };

  // Same operator as LocalLaplaceOperatorGeneric above, but built from
  // FEEvaluation's split evaluate_values()/evaluate_gradients()/
  // integrate_gradients()/integrate_values() instead of the combined,
  // EvaluationFlags-driven evaluate()/integrate().
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LocalLaplaceOperatorGenericSplit
  {
  public:
    LocalLaplaceOperatorGenericSplit() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const Custom::Parallel::BatchData<dim, Number> *data,
               const Custom::Parallel::DeviceView<Number>     &src,
               Custom::Parallel::DeviceView<Number>           &dst) const
    {
      Custom::Parallel::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate_values();
      fe_eval.evaluate_gradients();

      data->for_each_quad_point(
        [&](const int point) { fe_eval.submit_gradient(fe_eval.get_gradient(point), point); });

      fe_eval.integrate_gradients();
      fe_eval.integrate_values();

      fe_eval.distribute_local_to_global(dst);
    }
  };



  // Batched analog of internal::ApplyCellKernel's role
  template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Functor>
  void
  cell_loop_batched_launch(
    Functor                                                                 func,
    const typename MatrixFree<dim, Number>::PrecomputedData                 precomputed_data,
    const Custom::Parallel::DoFIndicesView                                  dof_indices,
    const Kokkos::View<Number *, MemorySpace::Default::kokkos_space>        G_tensor,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const unsigned int                       n_blocks          = numbers::invalid_unsigned_int,
    const unsigned int                       threads_per_block = numbers::invalid_unsigned_int,
    const Custom::Parallel::CellRangeIdView &cell_range_ids = Custom::Parallel::CellRangeIdView())
  {
    const int nelmt = precomputed_data.n_cells;
    if (nelmt == 0)
      return;

    constexpr int n_1d     = fe_degree + 1;
    constexpr int nq_total = Utilities::pow(n_q_points_1d, dim);

    // 1 values slot + dim gradients-pool slots + (dim - 1) dedicated
    // evaluate_values()/integrate_values() scratch slots (0 for dim == 1, 1
    // for dim == 2, 2 for dim == 3 -- see their own doc comments) -- always
    // exactly 2 * dim.
    constexpr int n_scratch_arrays = 2 * dim;

    if (cell_range_ids.size() > 0)
      AssertDimension(cell_range_ids.size(), static_cast<unsigned int>(nelmt));

    // Batch-size heuristic and shared-memory layout copied verbatim from
    // BK3Custom::Parallel::KokkosKernel (kernels/bk3_kokkos_kernels_custom.h)
    // to keep this launcher's performance characteristics identical.
    constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

    const int nelmtPerBatch =
      std::max(1, static_cast<int>(shmemPerBlock / (n_scratch_arrays * nq_total) / sizeof(Number)));

    const int numBlocks = std::max(1,
                                   ((n_blocks == numbers::invalid_unsigned_int) ?
                                      ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                      static_cast<int>(n_blocks)));

    const int threadsPerBlock =
      std::max(1,
               ((threads_per_block == numbers::invalid_unsigned_int) ?
                  (Utilities::pow(n_q_points_1d, dim - 1) * nelmtPerBatch) :
                  static_cast<int>(threads_per_block)));

    const Custom::Parallel::DeviceView<Number> src_device(src.get_values(),
                                                          src.locally_owned_size());
    const Custom::Parallel::DeviceView<Number> dst_device(dst.get_values(),
                                                          dst.locally_owned_size());

    const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> shape_values_global =
      precomputed_data.shape_values;
    const Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients_global =
      precomputed_data.co_shape_gradients;
    const Kokkos::View<Number **[dim][dim], MemorySpace::Default::kokkos_space> inv_jacobian =
      precomputed_data.inv_jacobian;
    const Kokkos::View<Number **, MemorySpace::Default::kokkos_space> JxW = precomputed_data.JxW;

    const int ssize =
      n_1d * n_q_points_1d +                       // shape values
      n_q_points_1d * n_q_points_1d +              // co-shape gradients
      n_scratch_arrays * nelmtPerBatch * nq_total; // values + gradients pool + dedicated scratch

    const unsigned int shmem_size = ssize * sizeof(Number);

    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
    policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

    Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

        Number *s_shape_values       = scratch;
        Number *s_co_shape_gradients = s_shape_values + n_1d * n_q_points_1d;

        Number *s_values    = s_co_shape_gradients + n_q_points_1d * n_q_points_1d;
        Number *s_gradients = s_values + nelmtPerBatch * nq_total;
        Number *s_scratch   = s_gradients + dim * nelmtPerBatch * nq_total;

        const int quad_size_per_batch = nelmtPerBatch * nq_total;

        const int threadIdx = team_member.team_rank();
        const int blockSize = team_member.team_size();

        for (int tid = threadIdx; tid < n_1d * n_q_points_1d; tid += blockSize)
          s_shape_values[tid] = shape_values_global[tid];

        for (int tid = threadIdx; tid < n_q_points_1d * n_q_points_1d; tid += blockSize)
          s_co_shape_gradients[tid] = co_shape_gradients_global[tid];

        team_member.team_barrier();

        int batchIdx = team_member.league_rank();

        while (batchIdx < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
          {
            // current nelmtPerBatch (edge case, last batch size can be
            // less)
            const int c_nelmtPerBatch = (batchIdx * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                          (nelmt - batchIdx * nelmtPerBatch) :
                                          nelmtPerBatch;

            Custom::Parallel::BatchData<dim, Number> data{team_member,
                                                          s_shape_values,
                                                          s_co_shape_gradients,
                                                          G_tensor,
                                                          inv_jacobian,
                                                          JxW,
                                                          dof_indices,
                                                          batchIdx,
                                                          nelmtPerBatch,
                                                          c_nelmtPerBatch,
                                                          threadIdx,
                                                          blockSize,
                                                          cell_range_ids,
                                                          s_values,
                                                          s_gradients,
                                                          s_scratch,
                                                          quad_size_per_batch};

            Custom::Parallel::DeviceView<Number> nonconst_dst = dst_device;
            func(&data, src_device, nonconst_dst);

            batchIdx += team_member.league_size();
          }
      });

    Kokkos::fence();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
