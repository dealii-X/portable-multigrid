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
    //
    // Currently UNUSED by cell_loop_batched_launch_view() below -- kept
    // defined for a bisection in progress: a JUPITER regression (~3.6-
    // 4.8% at large scale) was found comparing this functor-class design
    // against the pre-existing KOKKOS_LAMBDA one, and the scratch-
    // allocation mechanism (raw pointer vs sequential team_shmem()-View
    // construction) and the team_size_max()/AUTO safeguard have both
    // already been ruled out/only-partially-implicated as the cause --
    // the functor-class-vs-lambda conversion itself is the remaining
    // candidate, being isolated by reverting the launcher to a plain
    // KOKKOS_LAMBDA (with the same View-based scratch allocation kept)
    // while leaving this class here to swap back to once that's settled.
    template <int dim, int fe_degree, int n_q_points_1d, typename Number, typename Functor>
    struct ApplyBatchedKernelView
    {
      using TeamHandle      = Custom::Parallel::TeamHandle;
      using ScratchViewType = typename Custom::Parallel::SharedDataView<Number>::ScratchView;
      using SharedViewValuesType =
        typename Custom::Parallel::SharedDataView<Number>::SharedViewValues;
      using SharedViewGradientsType =
        typename Custom::Parallel::SharedDataView<Number>::SharedViewGradients;

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

      // See cell_loop_batched_launch_view()'s n_scratch_arrays comment for
      // the full derivation.
      static constexpr int n_scratch_arrays = needs_gradients ? dim + 1 : 2;

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

      // Provide the shared memory capacity. Mirrors operator()'s actual
      // sequence of team_shmem() draws one-to-one, via each View type's
      // own shmem_size() rather than a flat extent*sizeof(Number) sum --
      // shmem_size() adds an alignment padding term per draw
      // (required_span_size()*sizeof(T) + scratch_value_alignment,
      // confirmed in Kokkos_View.hpp), so a hand-computed flat total
      // would under-report what the View constructors in operator()
      // actually consume once there's more than one separate
      // team_shmem() draw -- exactly our case. DEAL_II_HOST_DEVICE since
      // nothing here calls it device-side, but harmless/consistent to
      // keep annotated the same as operator().
      DEAL_II_HOST_DEVICE
      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        std::size_t result = ScratchViewType::shmem_size(n_1d * n_q_points_1d) +
                             ScratchViewType::shmem_size(n_q_points_1d * n_q_points_1d) +
                             SharedViewValuesType::shmem_size(n_q_points_per_batch, 1);

        if constexpr (needs_gradients)
          {
            // v_scratch reuses v_gradients's own memory in operator() --
            // no separate team_shmem() draw, whether or not values is
            // also requested.
            result += SharedViewGradientsType::shmem_size(n_q_points_per_batch, dim, 1);
          }
        else
          {
            // values-only: v_gradients is never drawn from team_shmem()
            // at all (empty dummy), only v_scratch is.
            result += ScratchViewType::shmem_size(n_q_points_per_batch);
          }

        return result;
      }

      DEAL_II_HOST_DEVICE void
      operator()(const TeamHandle &team_member) const
      {
        // Each View constructor below draws its own chunk from
        // team_member.team_shmem() (a stateful bump allocator -- Kokkos
        // advances its internal offset, alignment included, on every
        // call), so no manual byte-offset arithmetic is needed for the
        // buffers that don't alias. Total bytes drawn here always equals
        // team_shmem_size() above.
        ScratchViewType      v_shape_values(team_member.team_shmem(), n_1d * n_q_points_1d);
        ScratchViewType      v_co_shape_gradients(team_member.team_shmem(),
                                                  n_q_points_1d * n_q_points_1d);
        SharedViewValuesType v_values(team_member.team_shmem(), n_q_points_per_batch, 1);

        // Layout depends on Functor::evaluation_flags -- see the
        // n_scratch_arrays derivation above for the reasoning.
        SharedViewGradientsType v_gradients;
        ScratchViewType         v_scratch;
        if constexpr (needs_gradients)
          {
            v_gradients =
              SharedViewGradientsType(team_member.team_shmem(), n_q_points_per_batch, dim, 1);
            v_scratch = ScratchViewType(v_gradients.data(),
                                        n_q_points_per_batch); // alias, whether or not
                                                               // values is also requested
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

        const Custom::Parallel::SharedDataView<Number> shared_data{
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
                                                                    shared_data,
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

  // Bisection step: back to a plain KOKKOS_LAMBDA (not
  // internal::ApplyBatchedKernelView) -- isolating whether the functor-
  // class-vs-lambda conversion itself is responsible for the remaining
  // ~3.6-4.8%-at-large-scale JUPITER regression found relative to the
  // pre-refactor version, now that the scratch-allocation mechanism and
  // the team_size_max()/AUTO safeguard have each been tested and ruled
  // out / only-partially-implicated in turn. Scratch is still allocated
  // "view-like" (sequential team_shmem()-View construction inside the
  // lambda, not manual byte-offset pointer arithmetic) -- only the
  // functor-vs-lambda variable changes here. Since a plain lambda has no
  // team_shmem_size() hook Kokkos can query, shmem_size is computed by
  // hand up front (mirroring each View's own shmem_size(), same
  // alignment-aware accounting as ApplyBatchedKernelView::team_shmem_
  // size()) and passed via policy.set_scratch_size(), matching the
  // pre-ApplyBatchedKernelView design.
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

    constexpr int n_1d     = fe_degree + 1;
    constexpr int nq_total = Utilities::pow(n_q_points_1d, dim);

    // Bitwise-AND on the plain ints rather than EvaluationFlags::operator&
    // -- that overload isn't marked constexpr in every deal.II version
    // this project builds against, but the enum's own integer conversion
    // always is.
    constexpr unsigned int evaluation_flags_int =
      static_cast<unsigned int>(Functor::evaluation_flags);
    constexpr bool needs_values =
      evaluation_flags_int & static_cast<unsigned int>(EvaluationFlags::values);
    constexpr bool needs_gradients =
      evaluation_flags_int & static_cast<unsigned int>(EvaluationFlags::gradients);
    static_assert(needs_values || needs_gradients,
                  "Functor::evaluation_flags must request values and/or gradients.");

    // The `values` buffer (1 slot) is always needed. The `gradients`
    // buffer (dim slots) is only allocated when the functor asks for it.
    // The dedicated scratch region is just 1 slot, needed only when
    // gradients isn't requested at all: whenever it is, evaluate()/
    // integrate() (FEEvaluationImplTransformToCollocationView) borrow
    // straight from the gradients buffer's own memory instead -- safe,
    // since the scratch-touching phase and the gradients-touching phase
    // never overlap in time (separated by a team_barrier()), and dim
    // slots is always enough (dim == 1 needs 1 borrowed slot for its
    // forced in_place step, dim == 2 needs 1 for its ping-pong, dim == 3
    // needs 2 to route all 3 directions with zero aliasing at all -- see
    // the evaluate()/integrate() comments for the routing itself). So a
    // dedicated scratch slot is only ever allocated in the values-only
    // case.
    constexpr int n_scratch_arrays = needs_gradients ? dim + 1 : 2;

    if (cell_range_ids.size() > 0)
      AssertDimension(cell_range_ids.size(), static_cast<unsigned int>(nelmt));

    // Batch-size heuristic copied verbatim from cell_loop_batched_launch()
    // to keep this launcher's performance characteristics identical.
    constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

    const int n_elements_per_batch =
      std::max(1,
               ((n_cells_per_batch == numbers::invalid_unsigned_int) ?
                  static_cast<int>(shmemPerBlock / (n_scratch_arrays * nq_total) / sizeof(Number)) :
                  static_cast<int>(n_cells_per_batch)));

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

    const int n_q_points_per_batch = n_elements_per_batch * nq_total;

    using ScratchViewType = typename Custom::Parallel::SharedDataView<Number>::ScratchView;
    using SharedViewValuesType =
      typename Custom::Parallel::SharedDataView<Number>::SharedViewValues;
    using SharedViewGradientsType =
      typename Custom::Parallel::SharedDataView<Number>::SharedViewGradients;

    // Matches the sequence of team_shmem() draws the lambda below makes,
    // one shmem_size() call per draw (alignment-padding-aware), not a
    // flat extent*sizeof(Number) sum -- see
    // ApplyBatchedKernelView::team_shmem_size()'s comment for why that
    // distinction matters.
    std::size_t shmem_size = ScratchViewType::shmem_size(n_1d * n_q_points_1d) +
                             ScratchViewType::shmem_size(n_q_points_1d * n_q_points_1d) +
                             SharedViewValuesType::shmem_size(n_q_points_per_batch, 1);
    if (needs_gradients)
      {
        // scratch reuses gradients' own memory below, whether or not
        // values is also requested -- no separate draw, no separate term.
        shmem_size += SharedViewGradientsType::shmem_size(n_q_points_per_batch, dim, 1);
      }
    else
      {
        shmem_size += ScratchViewType::shmem_size(n_q_points_per_batch); // values-only
      }

    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
    policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

    Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        ScratchViewType      v_shape_values(team_member.team_shmem(), n_1d * n_q_points_1d);
        ScratchViewType      v_co_shape_gradients(team_member.team_shmem(),
                                                  n_q_points_1d * n_q_points_1d);
        SharedViewValuesType v_values(team_member.team_shmem(), n_q_points_per_batch, 1);

        // Layout depends on Functor::evaluation_flags -- see
        // n_scratch_arrays above for the reasoning.
        SharedViewGradientsType v_gradients;
        ScratchViewType         v_scratch;
        if (needs_gradients)
          {
            v_gradients =
              SharedViewGradientsType(team_member.team_shmem(), n_q_points_per_batch, dim, 1);
            v_scratch = ScratchViewType(v_gradients.data(),
                                        n_q_points_per_batch); // alias, whether or not
                                                               // values is also requested
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

        const Custom::Parallel::SharedDataView<Number> shared_data{
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
                                                                    shared_data,
                                                                    batch_index,
                                                                    n_elements_per_batch,
                                                                    n_elements_in_current_batch,
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
