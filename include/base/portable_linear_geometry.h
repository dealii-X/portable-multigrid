#ifndef portable_linear_geometry_h
#define portable_linear_geometry_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {
    // G = JxW * J^{-1} J^{-T} (symmetric part) and JxW of the multilinear
    // map of each cell at the tensor-product Gauss points, in the layout of
    // LaplaceOperatorBK3: G[(cell * n_sym + c) * n_q + q], JxW[cell * n_q + q].
    // vertices[(cell * 2^dim + v) * dim + d], deal.II's vertex numbering.
    // J^{-1} and JxW are cast to Number before the contraction, as on the host.
    template <int dim, int n_q_points_1d, typename Number>
    void
    compute_linear_geometry(
      const MemorySpace::Default::kokkos_space::execution_space          &exec,
      const Kokkos::View<const double *, MemorySpace::Default::kokkos_space> vertices,
      const Kokkos::View<const double *, MemorySpace::Default::kokkos_space> points_1d,
      const Kokkos::View<const double *, MemorySpace::Default::kokkos_space> weights_1d,
      const unsigned int                                                   n_cells,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space>     G,
      const Kokkos::View<Number *, MemorySpace::Default::kokkos_space>     JxW)
    {
      static_assert(dim >= 1 && dim <= 3);

      constexpr unsigned int n_vertices = 1u << dim;
      constexpr unsigned int n_q        = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned int n_sym      = (dim * (dim + 1)) / 2;

      Kokkos::parallel_for(
        "compute_linear_geometry",
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(exec,
                                                                                 0,
                                                                                 n_cells * n_q),
        KOKKOS_LAMBDA(const unsigned int index) {
          const unsigned int cell = index / n_q;
          const unsigned int q    = index % n_q;

          double       x_hat[dim];
          double       weight = 1.;
          unsigned int q_d    = q;
          for (unsigned int d = 0; d < dim; ++d)
            {
              x_hat[d] = points_1d(q_d % n_q_points_1d);
              weight *= weights_1d(q_d % n_q_points_1d);
              q_d /= n_q_points_1d;
            }

          // J[k][j] = dx_k / dx_hat_j
          double J[dim][dim] = {};
          for (unsigned int v = 0; v < n_vertices; ++v)
            for (unsigned int j = 0; j < dim; ++j)
              {
                double dphi = 1.;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    const bool upper = (v >> d) & 1u;
                    if (d == j)
                      dphi *= upper ? 1. : -1.;
                    else
                      dphi *= upper ? x_hat[d] : 1. - x_hat[d];
                  }
                for (unsigned int k = 0; k < dim; ++k)
                  J[k][j] += vertices((cell * n_vertices + v) * dim + k) * dphi;
              }

          // K = J^{-1}, K[j][k] = dx_hat_j / dx_k
          double K[dim][dim];
          double det;
          if constexpr (dim == 1)
            {
              det     = J[0][0];
              K[0][0] = 1. / det;
            }
          else if constexpr (dim == 2)
            {
              det     = J[0][0] * J[1][1] - J[0][1] * J[1][0];
              K[0][0] = J[1][1] / det;
              K[0][1] = -J[0][1] / det;
              K[1][0] = -J[1][0] / det;
              K[1][1] = J[0][0] / det;
            }
          else
            {
              det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                    J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                    J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
              for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                  {
                    // cofactor of J[j][i]
                    const unsigned int r0 = (j + 1) % 3, r1 = (j + 2) % 3;
                    const unsigned int c0 = (i + 1) % 3, c1 = (i + 2) % 3;
                    K[i][j] = (J[r0][c0] * J[r1][c1] - J[r0][c1] * J[r1][c0]) / det;
                  }
            }

          const Number jxw = static_cast<Number>(det * weight);
          JxW(cell * n_q + q) = jxw;

          unsigned int c = 0;
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            for (unsigned int d2 = d1; d2 < dim; ++d2)
              {
                Number sum = 0;
                for (unsigned int k = 0; k < dim; ++k)
                  sum += static_cast<Number>(K[d1][k]) * static_cast<Number>(K[d2][k]);
                G((cell * n_sym + c) * n_q + q) = jxw * sum;
                ++c;
              }
        });
    }
  } // namespace internal
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
