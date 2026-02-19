#ifndef bk3_kokkos_kernel_h
#define bk3_kokkos_kernel_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace Parallel
  {

    template <typename number>
    using DeviceView =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>;

    template <typename number>
    using SharedView =
      Kokkos::View<number *,
                   MemorySpace::Default::kokkos_space::execution_space>;

    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;
    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int n_local_dofs_1d, int n_q_points_1d, typename number>
    void
    KokkosKernel_1D_Block(
      const DeviceView<number> shape_values_device,
      const DeviceView<number> co_shape_gradients_device,
      const DeviceView<number> G_device,
      const DeviceView<number> in_device,
      DeviceView<number>       out_device,
      const DoFIndicesView     dof_indices,
      const unsigned int       n_cells,
      unsigned int             numThreads      = numbers::invalid_unsigned_int,
      unsigned int             threadsPerBlock = numbers::invalid_unsigned_int)
    {
      constexpr unsigned n_q_points_total = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned n_local_dofs_total =
        Utilities::pow(n_local_dofs_1d, dim);

      if (numThreads == numbers::invalid_unsigned_int)
        numThreads = n_cells * n_q_points_total / 2;


      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = n_q_points_total;

      unsigned int numBlocks =
        numThreads / (std::min(n_q_points_total, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;


      {
        const unsigned int scratch_pad_size =
          5 * n_q_points_total; // working scratch arrays: s_wsp0, s_wsp1,
                                // rqr,rqq, rqt

        unsigned int ssize =
          n_q_points_1d * n_q_points_1d + // shape values
          n_q_points_1d * n_q_points_1d + // co-shape gradients
          scratch_pad_size;               // at most 5 tmp arrays

        const unsigned int shmem_size = ssize * sizeof(number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            SharedView<number> shape_values_scratch(team_member.team_shmem(),
                                                    n_q_points_1d *
                                                      n_q_points_1d);
            SharedView<number> co_shape_gradients_scratch(
              team_member.team_shmem(), n_q_points_1d * n_q_points_1d);

            SharedView<number> rqr(team_member.team_shmem(), n_q_points_total);
            SharedView<number> rqs(team_member.team_shmem(), n_q_points_total);
            SharedView<number> rqt(team_member.team_shmem(), n_q_points_total);

            SharedView<number> s_wsp0(team_member.team_shmem(),
                                      n_q_points_total);
            SharedView<number> s_wsp1(team_member.team_shmem(),
                                      n_q_points_total);

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();


            // copy to shared memory
            {
              for (unsigned int tid = threadIdx;
                   tid < n_local_dofs_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  shape_values_scratch[tid] = shape_values_device[tid];
                }

              for (unsigned int tid = threadIdx;
                   tid < n_q_points_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  co_shape_gradients_scratch[tid] =
                    co_shape_gradients_device[tid];
                }
            }

            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            unsigned int cell_index = team_member.league_rank();

            while (cell_index < n_cells)
              {
                team_member.team_barrier();
                {
                  // step-1 : Copy from in to the scratch values
                  for (unsigned int tid = threadIdx; tid < n_local_dofs_total;
                       tid += blockSize)
                    {
                      const int i = tid / (n_local_dofs_1d * n_local_dofs_1d);
                      const int j =
                        (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                        n_local_dofs_1d;
                      const int k = tid % n_local_dofs_1d;

                      const int local_idx =
                        i * n_local_dofs_1d * n_local_dofs_1d +
                        j * n_local_dofs_1d + k;

                      const unsigned int dof_index =
                        dof_indices(local_idx, cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        s_wsp0[tid] = 0;
                      else
                        s_wsp0[tid] = in_device[dof_index];
                    }
                }
                team_member.team_barrier();

                if constexpr (dim == 3)
                  {
                    // step-2 : direction 0
                    for (unsigned int tid = threadIdx;
                         tid <
                         n_q_points_1d * n_local_dofs_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                        const int j =
                          (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                          n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        number sum = 0.0;
                        for (unsigned int i = 0; i < n_local_dofs_1d; ++i)
                          {
                            sum +=
                              s_wsp0[i * n_local_dofs_1d * n_local_dofs_1d +
                                     j * n_local_dofs_1d + k] *
                              shape_values_scratch[i * n_local_dofs_1d + p];
                          }
                        s_wsp1[p * n_local_dofs_1d * n_local_dofs_1d +
                               j * n_local_dofs_1d + k] = sum;
                      }
                    team_member.team_barrier();

                    // step-3 : direction 1
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int q = tid / (n_q_points_1d * n_local_dofs_1d);
                        const int p =
                          (tid % (n_q_points_1d * n_local_dofs_1d)) /
                          n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        number sum = 0.0;
                        for (unsigned int j = 0; j < n_local_dofs_1d; j++)
                          {
                            sum +=
                              s_wsp1[p * n_local_dofs_1d * n_local_dofs_1d +
                                     j * n_local_dofs_1d + k] *
                              shape_values_scratch[j * n_local_dofs_1d + q];
                          }

                        s_wsp0[q * n_q_points_1d * n_local_dofs_1d +
                               p * n_local_dofs_1d + k] = sum;
                      }
                    team_member.team_barrier();

                    // step-4 : direction 2
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                         tid += blockSize)
                      {
                        const int p = tid / (n_q_points_1d * n_q_points_1d);
                        const int q = (tid % (n_q_points_1d * n_q_points_1d)) /
                                      n_q_points_1d;
                        const int r = tid % n_q_points_1d;

                        number sum = 0.0;
                        for (unsigned int k = 0; k < n_local_dofs_1d; ++k)
                          {
                            sum +=
                              s_wsp0[q * n_q_points_1d * n_local_dofs_1d +
                                     p * n_local_dofs_1d + k] *
                              shape_values_scratch[k * n_local_dofs_1d + r];
                          }
                        s_wsp1[p * n_q_points_1d * n_q_points_1d +
                               q * n_q_points_1d + r] = sum;
                      }
                    team_member.team_barrier();
                  }

                // Geometric vals
                number Grr, Grs, Grt, Gss, Gst, Gtt;
                number qr, qs, qt;

                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q =
                      (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    qr = 0;
                    qs = 0;
                    qt = 0;

                    // step-5 : Load Geometric Factors, coalesced access
                    Grr = G_device[cell_index * 6 * n_q_points_total +
                                   0 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grs = G_device[cell_index * 6 * n_q_points_total +
                                   1 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grt = G_device[cell_index * 6 * n_q_points_total +
                                   2 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gss = G_device[cell_index * 6 * n_q_points_total +
                                   3 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gst = G_device[cell_index * 6 * n_q_points_total +
                                   4 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gtt = G_device[cell_index * 6 * n_q_points_total +
                                   5 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];

                    // step-6 : Multiply by D
                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qr += s_wsp1[n * n_q_points_1d * n_q_points_1d +
                                     q * n_q_points_1d + r] *
                              co_shape_gradients_scratch[n * n_q_points_1d + p];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qs += s_wsp1[p * n_q_points_1d * n_q_points_1d +
                                     n * n_q_points_1d + r] *
                              co_shape_gradients_scratch[n * n_q_points_1d + q];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qt += s_wsp1[p * n_q_points_1d * n_q_points_1d +
                                     q * n_q_points_1d + n] *
                              co_shape_gradients_scratch[n * n_q_points_1d + r];
                      }

                    // step-7 : Apply chain rule
                    rqr[p * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        r] = Grr * qr + Grs * qs + Grt * qt;
                    rqs[p * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        r] = Grs * qr + Gss * qs + Gst * qt;
                    rqt[p * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        r] = Grt * qr + Gst * qs + Gtt * qt;
                  }
                team_member.team_barrier();

                // step-8 : Compute out vector in GL nodes
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q =
                      (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    number sum = 0;
                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqr[n * n_q_points_1d * n_q_points_1d +
                                 q * n_q_points_1d + r] *
                             co_shape_gradients_scratch[p * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqs[p * n_q_points_1d * n_q_points_1d +
                                 n * n_q_points_1d + r] *
                             co_shape_gradients_scratch[q * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqt[p * n_q_points_1d * n_q_points_1d +
                                 q * n_q_points_1d + n] *
                             co_shape_gradients_scratch[r * n_q_points_1d + n];

                    s_wsp1[p * n_q_points_1d * n_q_points_1d +
                           q * n_q_points_1d + r] = sum;
                  }
                team_member.team_barrier();

                /*
                Interpolate to GLL nodes
                */

                // step-9 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int q = tid / (n_q_points_1d * n_local_dofs_1d);
                    const int p = (tid % (n_q_points_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int r = 0; r < n_q_points_1d; ++r)
                      {
                        sum += s_wsp1[p * n_q_points_1d * n_q_points_1d +
                                      q * n_q_points_1d + r] *
                               shape_values_scratch[k * n_local_dofs_1d + r];
                      }
                    s_wsp0[q * n_q_points_1d * n_local_dofs_1d +
                           p * n_local_dofs_1d + k] = sum;
                  }
                team_member.team_barrier();

                // step-10 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < n_local_dofs_1d * n_local_dofs_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int q = 0; q < n_q_points_1d; q++)
                      {
                        sum += s_wsp0[q * n_q_points_1d * n_local_dofs_1d +
                                      p * n_local_dofs_1d + k] *
                               shape_values_scratch[j * n_local_dofs_1d + q];
                      }
                    s_wsp1[p * n_local_dofs_1d * n_local_dofs_1d +
                           j * n_local_dofs_1d + k] = sum;
                  }
                team_member.team_barrier();

                // step-11 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < n_local_dofs_1d * n_local_dofs_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int i = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int p = 0; p < n_q_points_1d; ++p)
                      {
                        sum += s_wsp1[p * n_local_dofs_1d * n_local_dofs_1d +
                                      j * n_local_dofs_1d + k] *
                               shape_values_scratch[i * n_local_dofs_1d + p];
                      }
                    s_wsp0[i * n_local_dofs_1d * n_local_dofs_1d +
                           j * n_local_dofs_1d + k] = sum;
                  }
                team_member.team_barrier();

                // step-12 : Copy wsp0 to out
                for (unsigned int tid = threadIdx; tid < n_local_dofs_total;
                     tid += blockSize)
                  {
                    const int i = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    const int local_idx =
                      i * n_local_dofs_1d * n_local_dofs_1d +
                      j * n_local_dofs_1d + k;

                    const unsigned int dof_index =
                      dof_indices(local_idx, cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        Kokkos::atomic_add(&out_device[dof_index], s_wsp0[tid]);
                      }
                  }
                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif