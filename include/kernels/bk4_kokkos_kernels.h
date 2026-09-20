#ifndef bk4_kokkos_kernels_h
#define bk4_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

#include "kernels/bk3_kokkos_kernels.h"
#include "matrix_free/portable_evaluation_kernels.h"

DEAL_II_NAMESPACE_OPEN

namespace BK4
{
  namespace Parallel
  {
    template <typename Number>
    using DeviceView = BK3::Parallel::DeviceView<Number>;

    using DoFIndicesView  = BK3::Parallel::DoFIndicesView;
    using CellRangeIdView = BK3::Parallel::CellRangeIdView;

    // Vector Laplacian on an FESystem(FE_Q(fe_degree), n_components) space
    // (n_components independent scalar fields, no coupling between them --
    // this is *not* linear elasticity). The operator is block-diagonal in
    // the components and all components share the same tensor-product shape
    // functions and geometry, so this is built directly on top of BK3's
    // *abstracted* per-cell-batch building blocks (read_dof_values,
    // evaluate_values, evaluate_gradients_and_multiply_symmetric_tensor,
    // integrate_gradients, integrate_values, distribute_local_to_global --
    // see KokkosKernelAbstracted() in bk3_kokkos_kernels.h) rather than on
    // the hand-unrolled register-blocked scalar kernel.
    //
    // Crucially, the loop over components lives *inside* the single
    // Kokkos::parallel_for below, not as n_components separate kernel
    // launches: launching many small kernels back-to-back is expensive on
    // GPUs (launch overhead, no work to hide it behind), so each team
    // processes its cell-batch once (shape data staged into shared memory
    // once) and then walks all components of that same batch sequentially,
    // reusing the same s_values/s_gradients scratch and re-reading/
    // re-distributing dofs per component via its own dof_indices map.
    //
    // dof_indices_per_component[c] must be laid out exactly like the
    // dof_indices argument BK3::Parallel::KokkosKernelAbstracted() expects
    // for a scalar problem, i.e. dof_indices_per_component[c](i, cell) is
    // the global dof of local (lexicographic) dof i of component c on cell
    // `cell`, or numbers::invalid_unsigned_int if constrained.
    template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
    void
    KokkosKernelAbstracted(
      const DeviceView<Number>                                    d_shape_values,
      const DeviceView<Number>                                    d_co_shape_gradients,
      const DeviceView<Number>                                    d_G,
      const DeviceView<Number>                                    d_in,
      DeviceView<Number>                                          d_out,
      const Kokkos::Array<DoFIndicesView, n_components>          &dof_indices_per_component,
      const unsigned int    n_cells,
      const unsigned int    n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int    threads_per_block = numbers::invalid_unsigned_int,
      const unsigned int    n_cells_per_batch = numbers::invalid_unsigned_int,
      const CellRangeIdView cell_range_ids    = CellRangeIdView())
    {
      if (n_cells == 0)
        return;

      constexpr int n_quad_points_total = Utilities::pow(n_q_points_1d, dim);
      constexpr int n_local_dofs_1d     = fe_degree + 1;

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 1 + dim;

      if (cell_range_ids.size() > 0)
        AssertDimension(cell_range_ids.size(), n_cells);

      const int nelmt = n_cells;

      const int nelmtPerBatch =
        std::max(1,
                 ((n_cells_per_batch == numbers::invalid_unsigned_int) ?
                    static_cast<int>(shmemPerBlock / (n_scratch_arrays * n_quad_points_total) /
                                     sizeof(Number)) :
                    static_cast<int>(n_cells_per_batch)));

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));

      const int threadsPerBlock =
        std::max(1,
                 ((threads_per_block == numbers::invalid_unsigned_int) ?
                    (Utilities::pow(n_q_points_1d, dim - 1) * nelmtPerBatch) :
                    static_cast<int>(threads_per_block)));

      // Shared memory footprint is identical to the scalar
      // KokkosKernelAbstracted() -- components are processed one at a time
      // (sequentially, per batch), reusing the same s_values/s_gradients
      // slots, so there is no per-component multiplier here.
      {
        const int ssize = n_local_dofs_1d * n_q_points_1d + // shape values
                          n_q_points_1d * n_q_points_1d +   // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            n_quad_points_total; // values slot + dim gradients-pool slots

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + n_q_points_1d * n_local_dofs_1d;

            Number *s_values    = s_co_shape_gradients + n_q_points_1d * n_q_points_1d;
            Number *s_gradients = s_values + nelmtPerBatch * n_quad_points_total;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < n_local_dofs_1d * n_q_points_1d; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (int tid = threadIdx; tid < n_q_points_1d * n_q_points_1d; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            // element batch iteration
            int batchIdx = team_member.league_rank();

            while (batchIdx < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch = (batchIdx * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                              (nelmt - batchIdx * nelmtPerBatch) :
                                              nelmtPerBatch;

                const Custom::Parallel::
                  FEEvaluationImplTransformToCollocation<dim, fe_degree, n_q_points_1d, Number>
                    fe_eval(team_member,
                            s_shape_values,
                            s_co_shape_gradients,
                            nelmtPerBatch,
                            c_nelmtPerBatch,
                            batchIdx,
                            threadIdx,
                            blockSize);

                // Same cell-batch, all components, one launch: no coupling
                // between components (block-diagonal operator), so this is
                // just the scalar pipeline repeated with a different
                // dof_indices map each time.
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    const DoFIndicesView &dof_indices = dof_indices_per_component[c];

                    // 1. read dof values from global memory to shared memory
                    Custom::Parallel::read_dof_values<dim, n_local_dofs_1d>(team_member,
                                                                            dof_indices,
                                                                            cell_range_ids,
                                                                            d_in,
                                                                            s_values,
                                                                            batchIdx,
                                                                            nelmtPerBatch,
                                                                            c_nelmtPerBatch,
                                                                            threadIdx,
                                                                            blockSize);

                    // 2. interpolate from dof values to quadrature points
                    fe_eval.evaluate_values(s_values, s_values, s_gradients);

                    // 3. Evaluate Laplacian operator at quadrature points
                    fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(d_G,
                                                                             cell_range_ids,
                                                                             s_values,
                                                                             s_gradients);

                    // 4. integrate gradients to dof values
                    fe_eval.integrate_gradients(s_gradients, s_values);

                    // 5. integrate values to dof values
                    fe_eval.integrate_values(s_values, s_values, s_gradients);

                    // 6. distribute dof values from shared memory to global
                    // memory
                    Custom::Parallel::distribute_local_to_global<dim, n_local_dofs_1d>(
                      team_member,
                      dof_indices,
                      cell_range_ids,
                      s_values,
                      d_out,
                      batchIdx,
                      nelmtPerBatch,
                      c_nelmtPerBatch,
                      threadIdx,
                      blockSize);
                  }

                batchIdx += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK4

DEAL_II_NAMESPACE_CLOSE

#endif
