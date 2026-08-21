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

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LocalLaplaceOperatorGenericSplitView
  {
  public:
    LocalLaplaceOperatorGenericSplitView() = default;

    DEAL_II_HOST_DEVICE void
    operator()(const Custom::Parallel::BatchDataView<dim, Number> *data,
               const Custom::Parallel::DeviceView<Number>         &src,
               Custom::Parallel::DeviceView<Number>               &dst) const
    {
      Custom::Parallel::FEEvaluationView<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate_values();
      fe_eval.evaluate_gradients();

      data->for_each_quad_point([&](const int point)
                                  { fe_eval.submit_gradient(fe_eval.get_gradient(point), point); });

      fe_eval.integrate_gradients();
      fe_eval.integrate_values();

      fe_eval.distribute_local_to_global(dst);
    }
  };



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
    const Custom::Parallel::CellRangeIdView &cell_range_ids = Custom::Parallel::CellRangeIdView())
  {
    const int nelmt = precomputed_data.n_cells;
    if (nelmt == 0)
      return;

    constexpr int n_1d     = fe_degree + 1;
    constexpr int nq_total = Utilities::pow(n_q_points_1d, dim);

    // 1 values slot + dim gradients-pool slots + (dim - 1) dedicated
    // evaluate_values()/integrate_values() scratch slots -- see the class
    // doc comment above.
    constexpr int n_scratch_arrays = 2 * dim;

    if (cell_range_ids.size() > 0)
      AssertDimension(cell_range_ids.size(), static_cast<unsigned int>(nelmt));

    // Batch-size heuristic and shared-memory layout copied verbatim from
    // cell_loop_batched_launch() to keep this launcher's performance
    // characteristics identical.
    constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

    const int n_elements_per_batch =
      std::max(1, static_cast<int>(shmemPerBlock / (n_scratch_arrays * nq_total) / sizeof(Number)));

    const int numBlocks =
      std::max(1,
               ((n_blocks == numbers::invalid_unsigned_int) ?
                  ((nelmt + n_elements_per_batch - 1) / n_elements_per_batch / 2) :
                  static_cast<int>(n_blocks)));

    const int threadsPerBlock =
      std::max(1,
               ((threads_per_block == numbers::invalid_unsigned_int) ?
                  (Utilities::pow(n_q_points_1d, dim - 1) * n_elements_per_batch) :
                  static_cast<int>(threads_per_block)));

    const Custom::Parallel::DeviceView<Number> src_device(src.get_values(),
                                                          src.locally_owned_size());
    const Custom::Parallel::DeviceView<Number> dst_device(dst.get_values(),
                                                          dst.locally_owned_size());

    const int ssize = n_1d * n_q_points_1d +          // shape values
                      n_q_points_1d * n_q_points_1d + // co-shape gradients
                      n_scratch_arrays * n_elements_per_batch *
                        nq_total; // values + gradients pool + dedicated scratch

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
        Number *s_gradients = s_values + n_elements_per_batch * nq_total;
        Number *s_scratch   = s_gradients + dim * n_elements_per_batch * nq_total;

        const int n_q_points_per_batch = n_elements_per_batch * nq_total;

        const int thread_id  = team_member.team_rank();
        const int block_size = team_member.team_size();

        for (int tid = thread_id; tid < n_1d * n_q_points_1d; tid += block_size)
          s_shape_values[tid] = precomputed_data.shape_values[tid];

        for (int tid = thread_id; tid < n_q_points_1d * n_q_points_1d; tid += block_size)
          s_co_shape_gradients[tid] = precomputed_data.co_shape_gradients[tid];

        team_member.team_barrier();

        using ScratchViewType   = typename Custom::Parallel::ShapeDataView<Number>::ScratchView;
        using GradientsViewType = typename Custom::Parallel::ShapeDataView<Number>::GradientsView;

        ScratchViewType   v_shape_values(s_shape_values, n_1d * n_q_points_1d);
        ScratchViewType   v_co_shape_gradients(s_co_shape_gradients, n_q_points_1d * n_q_points_1d);
        ScratchViewType   v_values(s_values, n_q_points_per_batch);
        GradientsViewType v_gradients(s_gradients, n_q_points_per_batch, dim);
        ScratchViewType   v_scratch(s_scratch, (dim - 1) * n_q_points_per_batch);

        const Custom::Parallel::ShapeDataView<Number> shape_data{
          v_shape_values, v_co_shape_gradients, v_values, v_gradients, v_scratch};

        const Custom::Parallel::PrecomputedData<dim, Number> our_precomputed{precomputed_data,
                                                                             dof_indices,
                                                                             cell_range_ids};

        int batch_index = team_member.league_rank();

        while (batch_index < (nelmt + n_elements_per_batch - 1) / n_elements_per_batch)
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
                                                                    thread_id,
                                                                    block_size,
                                                                    n_q_points_per_batch,
                                                                    nq_total};

            Custom::Parallel::DeviceView<Number> nonconst_dst = dst_device;
            func(&data, src_device, nonconst_dst);

            batch_index += team_member.league_size();
          }
      });

    Kokkos::fence();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
