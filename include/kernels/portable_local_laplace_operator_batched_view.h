#ifndef portable_local_laplace_operator_batched_view_h
#define portable_local_laplace_operator_batched_view_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include "matrix_free/portable_batched_fe_evaluation_view.h"
#include "matrix_free/portable_evaluation_kernels_view.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LocalLaplaceOperatorGenericView
  {
  public:
    // Tells cell_loop_batched_launch_view() what this functor actually
    // reads/submits at quad points (get_value()/submit_value() are never
    // called here, only get_gradient()/submit_gradient()), so it can size
    // shared memory accordingly.
    static constexpr EvaluationFlags::EvaluationFlags evaluation_flags = EvaluationFlags::gradients;

    LocalLaplaceOperatorGenericView() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const Custom::Parallel::BatchDataView<dim, Number> *data,
               const Custom::Parallel::DeviceView<Number>         &src,
               Custom::Parallel::DeviceView<Number>               &dst) const
    {
      Custom::Parallel::FEEvaluationView<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::gradients);

      data->for_each_quad_point([&](const int point)
                                  { fe_eval.submit_gradient(fe_eval.get_gradient(point), point); });

      fe_eval.integrate(EvaluationFlags::gradients);

      fe_eval.distribute_local_to_global(dst);
    }
  };


  namespace internal
  {
    // Batched analog of internal::ApplyKernel: a
    // proper functor class instead of a KOKKOS_LAMBDA, so Kokkos can query
    // team_shmem_size() to size the scratch pool automatically instead of
    // cell_loop_batched_launch_view() computing shmem_size by hand and
    // calling policy.set_scratch_size() itself.
    template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Functor>
    struct ApplyBatchedKernelView
    {
      using TeamHandle = Custom::Parallel::TeamHandle;
      using ScratchViewShapeDataType =
        typename Custom::Parallel::ShapeDataView<Number>::ScratchView;
      using ScratchViewType = typename Custom::Parallel::ShapeDataView<Number>::ScratchView;
      using SharedViewValuesType =
        typename Custom::Parallel::ShapeDataView<Number>::SharedViewValues;
      using SharedViewGradientsType =
        typename Custom::Parallel::ShapeDataView<Number>::SharedViewGradients;

      static constexpr int n_1d     = fe_degree + 1;
      static constexpr int nq_total = Utilities::pow(n_q_points_1d, dim);

      static constexpr unsigned int evaluation_flags_int =
        static_cast<unsigned int>(Functor::evaluation_flags);
      static constexpr bool needs_values =
        evaluation_flags_int & static_cast<unsigned int>(EvaluationFlags::values);
      static constexpr bool needs_gradients =
        evaluation_flags_int & static_cast<unsigned int>(EvaluationFlags::gradients);
      static_assert(needs_values || needs_gradients,
                    "Functor::evaluation_flags must request values and/or gradients.");

      static constexpr int n_scratch_arrays = (needs_values && needs_gradients) ? dim + 2 :
                                              needs_gradients                   ? dim + 1 :
                                                                                  2;

      ApplyBatchedKernelView(
        Functor                                                                 func,
        const typename MatrixFree<dim, Number>::PrecomputedData                 precomputed_data,
        const Custom::Parallel::DoFIndicesView                                  dof_indices,
        const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
        const int                                n_elements_per_batch,
        const Custom::Parallel::CellRangeIdView &cell_range_ids =
          Custom::Parallel::CellRangeIdView())
        : func(func)
        , precomputed_data(precomputed_data)
        , dof_indices(dof_indices)
        , cell_range_ids(cell_range_ids)
        , src(src.get_values(), src.locally_owned_size())
        , dst(dst.get_values(), dst.locally_owned_size())
        , n_elements_per_batch(n_elements_per_batch)
        , nelmt(precomputed_data.n_cells)
        , n_q_points_per_batch(n_elements_per_batch * nq_total)
      {}

      Functor func;

      const typename MatrixFree<dim, Number>::PrecomputedData precomputed_data;
      const Custom::Parallel::DoFIndicesView                  dof_indices;
      const Custom::Parallel::CellRangeIdView                 cell_range_ids;

      const Custom::Parallel::DeviceView<Number> src;
      Custom::Parallel::DeviceView<Number>       dst;

      const int n_elements_per_batch;
      const int nelmt;
      const int n_q_points_per_batch;

      // Provide the shared memory capacity.
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        std::size_t result =
          ScratchViewShapeDataType::shmem_size(precomputed_data.shape_values.size()) +
          ScratchViewShapeDataType::shmem_size(precomputed_data.co_shape_gradients.size()) +
          SharedViewValuesType::shmem_size(n_q_points_per_batch, 1);

        if constexpr (needs_gradients)
          {
            result += SharedViewGradientsType::shmem_size(n_q_points_per_batch, dim, 1);
            // if we don't need to keep values, we can reuse the gradients' memory for the scratch
            // array and no extra memory is needed; in general case we need to allocate scratch pad
            // separately
            if constexpr (needs_values)
              result += ScratchViewType::shmem_size(n_q_points_per_batch);
          }
        else
          {
            // values-only: no gradients and one scratch pad
            result += ScratchViewType::shmem_size(n_q_points_per_batch);
          }

        return result;
      }

      DEAL_II_HOST_DEVICE void
      operator()(const TeamHandle &team_member) const
      {
        const int n_q_points_per_batch = n_elements_per_batch * nq_total;

        ScratchViewType      v_shape_values(team_member.team_shmem(), n_1d * n_q_points_1d);
        ScratchViewType      v_co_shape_gradients(team_member.team_shmem(),
                                                  n_q_points_1d * n_q_points_1d);
        SharedViewValuesType v_values(team_member.team_shmem(), n_q_points_per_batch, 1);

        SharedViewGradientsType v_gradients;
        ScratchViewType         v_scratch;
        if constexpr (needs_gradients)
          {
            v_gradients =
              SharedViewGradientsType(team_member.team_shmem(), n_q_points_per_batch, dim, 1);
            if constexpr (needs_values)
              v_scratch =
                ScratchViewType(team_member.team_shmem(), n_q_points_per_batch); // general
            else
              v_scratch = ScratchViewType(v_gradients.data(),
                                          n_q_points_per_batch); // gradients-only: alias
          }
        else
          {
            v_gradients = SharedViewGradientsType(); // values-only: unused dummy, never
                                                     // dereferenced, no shmem drawn
            v_scratch =
              ScratchViewType(team_member.team_shmem(), n_q_points_per_batch); // values-only
          }

        const int thread_id  = team_member.team_rank();
        const int block_size = team_member.team_size();

        for (int tid = thread_id; tid < n_1d * n_q_points_1d; tid += block_size)
          v_shape_values(tid) = precomputed_data.shape_values[tid];

        for (int tid = thread_id; tid < n_q_points_1d * n_q_points_1d; tid += block_size)
          v_co_shape_gradients(tid) = precomputed_data.co_shape_gradients[tid];

        team_member.team_barrier();

        const Custom::Parallel::ShapeDataView<Number> shape_data{
          v_shape_values,
          ScratchViewType(), // shape_gradients -- not staged here, this launcher's
                             // functors only go through the collocation/transform-
                             // to-collocation path (co_shape_gradients only)
          v_co_shape_gradients,
          v_values,
          v_gradients,
          v_scratch};

        const Custom::Parallel::PrecomputedData<dim, Number> our_precomputed{precomputed_data,
                                                                             dof_indices,
                                                                             cell_range_ids};

        const int n_batches = (nelmt + n_elements_per_batch - 1) / n_elements_per_batch;

        int batch_index = team_member.league_rank();

        while (batch_index < n_batches)
          {
            // current n_elements_per_batch (edge case, last batch size can be
            // less)
            const int n_elements_in_current_batch =
              (batch_index * n_elements_per_batch + n_elements_per_batch > nelmt) ?
                (nelmt - batch_index * n_elements_per_batch) :
                n_elements_per_batch;

            const Custom::Parallel::BatchDataView<dim, Number> data{team_member,
                                                                    our_precomputed,
                                                                    shape_data,
                                                                    batch_index,
                                                                    n_elements_per_batch,
                                                                    n_elements_in_current_batch,
                                                                    n_q_points_per_batch,
                                                                    nq_total};

            Custom::Parallel::DeviceView<Number> nonconst_dst = dst;
            func(&data, src, nonconst_dst);

            batch_index += team_member.league_size();
          }
      }
    };
  } // namespace internal

  template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Functor>
  void
  cell_loop_batched_launch_view(
    Functor                                                                 func,
    const typename MatrixFree<dim, Number>::PrecomputedData                 precomputed_data,
    const Custom::Parallel::DoFIndicesView                                  dof_indices,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const unsigned int                       n_blocks          = numbers::invalid_unsigned_int,
    const unsigned int                       threads_per_block = numbers::invalid_unsigned_int,
    const unsigned int                       n_cells_per_batch = numbers::invalid_unsigned_int,
    const Custom::Parallel::CellRangeIdView &cell_range_ids = Custom::Parallel::CellRangeIdView())
  {
    const int nelmt = precomputed_data.n_cells;
    if (nelmt == 0)
      return;

    constexpr int nq_total = Utilities::pow(n_q_points_1d, dim);

    using KernelType =
      internal::ApplyBatchedKernelView<dim, fe_degree, n_q_points_1d, Number, Functor>;

    if (cell_range_ids.size() > 0)
      AssertDimension(cell_range_ids.size(), static_cast<unsigned int>(nelmt));

    // Batch-size heuristic copied verbatim from cell_loop_batched_launch()
    // to keep this launcher's performance characteristics identical.
    constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

    const int n_elements_per_batch =
      std::max(1,
               ((n_cells_per_batch == numbers::invalid_unsigned_int) ?
                  static_cast<int>(shmemPerBlock / (KernelType::n_scratch_arrays * nq_total) /
                                   sizeof(Number)) :
                  static_cast<int>(n_cells_per_batch)));

    const int numBlocks =
      std::max(1,
               ((n_blocks == numbers::invalid_unsigned_int) ?
                  ((nelmt + n_elements_per_batch - 1) / n_elements_per_batch / 2) :
                  static_cast<int>(n_blocks)));

    const int heuristic_threads_per_block =
      std::max(1,
               ((threads_per_block == numbers::invalid_unsigned_int) ?
                  (Utilities::pow(n_q_points_1d, dim - 1) * n_elements_per_batch) :
                  static_cast<int>(threads_per_block)));

    KernelType apply_kernel(
      func, precomputed_data, dof_indices, src, dst, n_elements_per_batch, cell_range_ids);

    const Kokkos::TeamPolicy<> probe_policy(numBlocks, Kokkos::AUTO);
    const bool                 heuristic_is_launchable =
      heuristic_threads_per_block <=
      probe_policy.team_size_max(apply_kernel, Kokkos::ParallelForTag());

    const Kokkos::TeamPolicy<> policy =
      heuristic_is_launchable ? Kokkos::TeamPolicy<>(numBlocks, heuristic_threads_per_block) :
                                Kokkos::TeamPolicy<>(numBlocks, Kokkos::AUTO);

    Kokkos::parallel_for(policy, apply_kernel);

    Kokkos::fence();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
