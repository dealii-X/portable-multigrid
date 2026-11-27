#ifndef portable_symmetric_eigenbasis_h
#define portable_symmetric_eigenbasis_h

#include <deal.II/base/tensor.h>

#include <Kokkos_Core.hpp>

#include <limits>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  enum class EigenSolver
  {
    jacobi,
    closed_form
  };



  // A = sum_a values[a] vectors[a] (x) vectors[a], values in descending
  // order, vectors[a] orthonormal
  template <int dim, typename Number>
  struct SymmetricEigenbasis
  {
    Tensor<1, dim, Number> values;
    Tensor<2, dim, Number> vectors;
  };



  // Eigenbasis of the symmetric part of A by cyclic Jacobi rotations
  // (Golub, Van Loan, Sec. 8.5): no special cases for repeated eigenvalues,
  // which give an arbitrary orthonormal basis of the eigenspace
  template <int dim, typename Number>
  DEAL_II_HOST_DEVICE SymmetricEigenbasis<dim, Number>
  symmetric_eigenbasis(const Tensor<2, dim, Number> &A)
  {
    constexpr unsigned int max_sweeps = 20;
    constexpr Number       eps        = std::numeric_limits<Number>::epsilon();

    Number a[dim][dim];
    Number v[dim][dim];
    Number norm_sq = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        {
          a[i][j] = Number(0.5) * (A[i][j] + A[j][i]);
          v[i][j] = (i == j) ? Number(1.) : Number(0.);
          norm_sq += a[i][j] * a[i][j];
        }

    // the Frobenius norm is invariant under the rotations
    for (unsigned int sweep = 0; sweep < max_sweeps; ++sweep)
      {
        Number off_sq = 0.;
        for (unsigned int p = 0; p < dim; ++p)
          for (unsigned int q = p + 1; q < dim; ++q)
            off_sq += a[p][q] * a[p][q];
        if (!(off_sq > eps * eps * norm_sq))
          break;

        for (unsigned int p = 0; p < dim; ++p)
          for (unsigned int q = p + 1; q < dim; ++q)
            {
              const Number apq = a[p][q];
              if (apq == Number(0.))
                continue;

              // rotation zeroing a[p][q]: t = tan(phi) of the smaller root,
              // sqrt(theta^2 + 1) without overflow
              const Number theta     = (a[q][q] - a[p][p]) / (Number(2.) * apq);
              const Number abs_theta = Kokkos::abs(theta);
              const Number root =
                (abs_theta > Number(1.)) ?
                  abs_theta * Kokkos::sqrt(Number(1.) + Number(1.) / (theta * theta)) :
                  Kokkos::sqrt(theta * theta + Number(1.));
              const Number t = (theta < Number(0.) ? Number(-1.) : Number(1.)) / (abs_theta + root);
              const Number c = Number(1.) / Kokkos::sqrt(t * t + Number(1.));
              const Number s = t * c;

              a[p][p] -= t * apq;
              a[q][q] += t * apq;
              a[p][q] = a[q][p] = Number(0.);
              for (unsigned int r = 0; r < dim; ++r)
                {
                  if (r != p && r != q)
                    {
                      const Number arp = a[r][p];
                      const Number arq = a[r][q];
                      a[r][p] = a[p][r] = c * arp - s * arq;
                      a[r][q] = a[q][r] = s * arp + c * arq;
                    }
                  const Number vrp = v[r][p];
                  const Number vrq = v[r][q];
                  v[r][p]          = c * vrp - s * vrq;
                  v[r][q]          = s * vrp + c * vrq;
                }
            }
      }

    // eigenvectors are the columns of v; sort descending by swaps with
    // compile-time indices, which keeps a and v in registers
    for (unsigned int i = 0; i + 1 < dim; ++i)
      for (unsigned int j = 0; j + 1 < dim - i; ++j)
        if (a[j][j] < a[j + 1][j + 1])
          {
            const Number tmp = a[j][j];
            a[j][j]          = a[j + 1][j + 1];
            a[j + 1][j + 1]  = tmp;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const Number vd = v[d][j];
                v[d][j]         = v[d][j + 1];
                v[d][j + 1]     = vd;
              }
          }

    SymmetricEigenbasis<dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      {
        result.values[i] = a[i][i];
        for (unsigned int d = 0; d < dim; ++d)
          result.vectors[i][d] = v[d][i];
      }
    return result;
  }

  namespace internal
  {
    // c, s of the rotation zeroing a_pq of [[a_pp, a_pq], [a_pq, a_qq]]:
    // t = tan(phi) of the smaller root, sqrt(theta^2 + 1) without overflow
    template <typename Number>
    DEAL_II_HOST_DEVICE void
    jacobi_rotation(const Number app,
                    const Number apq,
                    const Number aqq,
                    Number      &t,
                    Number      &c,
                    Number      &s)
    {
      const Number theta     = (aqq - app) / (Number(2.) * apq);
      const Number abs_theta = Kokkos::abs(theta);
      const Number root      = (abs_theta > Number(1.)) ?
                                 abs_theta * Kokkos::sqrt(Number(1.) + Number(1.) / (theta * theta)) :
                                 Kokkos::sqrt(theta * theta + Number(1.));
      t                      = (theta < Number(0.) ? Number(-1.) : Number(1.)) / (abs_theta + root);
      c                      = Number(1.) / Kokkos::sqrt(t * t + Number(1.));
      s                      = t * c;
    }
  } // namespace internal



  // Eigenbasis of the symmetric part of A without iteration (cf. Eberly, A
  // robust eigensolver for 3 x 3 symmetric matrices, 2014). For B = A - q I,
  // q = tr A / 3: the root of det(B - beta I) farthest from the middle one
  // by the trigonometric formula (well conditioned also when the other two
  // coincide), its eigenvector w as the largest cross product of rows of
  // B - beta I, the other two by one Jacobi rotation of B restricted to w^perp;
  // eigenvalues as Rayleigh quotients. In 2D one rotation. Same result type
  // and ordering as symmetric_eigenbasis(). A diagonal A is only sorted.
  template <int dim, typename Number>
  DEAL_II_HOST_DEVICE SymmetricEigenbasis<dim, Number>
  symmetric_eigenbasis_closed_form(const Tensor<2, dim, Number> &A)
  {
    static_assert(dim == 2 || dim == 3);
    SymmetricEigenbasis<dim, Number> result;

    Number a[dim][dim];
    Number scale = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        {
          a[i][j] = Number(0.5) * (A[i][j] + A[j][i]);
          // comparisons, not fmax (a libm call on the host)
          const Number abs_a = Kokkos::abs(a[i][j]);
          scale              = (abs_a > scale) ? abs_a : scale;
        }
    if (scale == Number(0.))
      {
        for (unsigned int i = 0; i < dim; ++i)
          result.vectors[i][i] = 1.;
        return result;
      }

    Number value[dim];
    Number vector[dim][dim];
    Number off_diagonal = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        {
          const Number abs_a = Kokkos::abs(a[i][j]);
          off_diagonal       = (abs_a > off_diagonal) ? abs_a : off_diagonal;
        }
    if (off_diagonal == Number(0.))
      {
        // diagonal, e.g. C of an axis-aligned stretch
        for (unsigned int i = 0; i < dim; ++i)
          {
            value[i] = a[i][i];
            for (unsigned int d = 0; d < dim; ++d)
              vector[d][i] = (d == i) ? Number(1.) : Number(0.);
          }
      }
    else if constexpr (dim == 2)
      {
        Number t, c, s;
        internal::jacobi_rotation(a[0][0], a[0][1], a[1][1], t, c, s);
        value[0]     = a[0][0] - t * a[0][1];
        value[1]     = a[1][1] + t * a[0][1];
        vector[0][0] = c;
        vector[1][0] = -s;
        vector[0][1] = s;
        vector[1][1] = c;
      }
    else
      {
        // scaled to entries <= 1 and shifted
        const Number inv_scale = Number(1.) / scale;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            a[i][j] *= inv_scale;
        const Number q = (a[0][0] + a[1][1] + a[2][2]) / Number(3.);
        for (unsigned int i = 0; i < dim; ++i)
          a[i][i] -= q;
        const Number p_sq =
          (a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2] +
           Number(2.) * (a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2])) /
          Number(6.);
        if (p_sq == Number(0.))
          {
            for (unsigned int i = 0; i < dim; ++i)
              {
                result.values[i]     = scale * q;
                result.vectors[i][i] = 1.;
              }
            return result;
          }

        // beta = eigenvalue of (B / p) farthest from the middle one: the
        // largest for det(B / p) >= 0, else the smallest
        Number w[3];
        {
          const Number p     = Kokkos::sqrt(p_sq);
          const Number inv_p = Number(1.) / p;
          const Number det   = inv_p * inv_p * inv_p *
                             (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[1][2]) -
                              a[0][1] * (a[0][1] * a[2][2] - a[1][2] * a[0][2]) +
                              a[0][2] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]));
          const Number half_det_raw = Number(0.5) * det;
          const Number half_det     = (half_det_raw > Number(1.))  ? Number(1.) :
                                      (half_det_raw < Number(-1.)) ? Number(-1.) :
                                                                     half_det_raw;
          const Number phi          = Kokkos::acos(half_det) / Number(3.);
          const Number beta =
            Number(2.) *
            Kokkos::cos(half_det >= Number(0.) ? phi : phi + Number(2.0943951023931954923));
          const Number lambda = p * beta;

          // rows of B - lambda I, the largest of their cross products
          const Number r[3][3] = {{a[0][0] - lambda, a[0][1], a[0][2]},
                                  {a[0][1], a[1][1] - lambda, a[1][2]},
                                  {a[0][2], a[1][2], a[2][2] - lambda}};
          Number       norm_sq = -1.;
          for (unsigned int k = 0; k < 3; ++k)
            {
              const unsigned int i    = (k == 2) ? 1 : 0;
              const unsigned int j    = (k == 0) ? 1 : 2;
              const Number       x[3] = {r[i][1] * r[j][2] - r[i][2] * r[j][1],
                                         r[i][2] * r[j][0] - r[i][0] * r[j][2],
                                         r[i][0] * r[j][1] - r[i][1] * r[j][0]};
              const Number       n    = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
              if (n > norm_sq)
                {
                  norm_sq = n;
                  for (unsigned int d = 0; d < 3; ++d)
                    w[d] = x[d];
                }
            }
          // rows of rank <= 1: only for B ~ 0 within roundoff
          if (!(norm_sq > Number(0.)))
            return symmetric_eigenbasis(A);
          const Number inv_norm = Number(1.) / Kokkos::sqrt(norm_sq);
          for (unsigned int d = 0; d < 3; ++d)
            w[d] *= inv_norm;
        }

        // orthonormal basis U, V of w^perp
        Number U[3];
        if (Kokkos::abs(w[0]) > Kokkos::abs(w[1]))
          {
            const Number inv = Number(1.) / Kokkos::sqrt(w[0] * w[0] + w[2] * w[2]);
            U[0]             = -w[2] * inv;
            U[1]             = 0.;
            U[2]             = w[0] * inv;
          }
        else
          {
            const Number inv = Number(1.) / Kokkos::sqrt(w[1] * w[1] + w[2] * w[2]);
            U[0]             = 0.;
            U[1]             = w[2] * inv;
            U[2]             = -w[1] * inv;
          }
        const Number V[3] = {w[1] * U[2] - w[2] * U[1],
                             w[2] * U[0] - w[0] * U[2],
                             w[0] * U[1] - w[1] * U[0]};

        Number Bw[3], BU[3], BV[3];
        for (unsigned int i = 0; i < 3; ++i)
          {
            Bw[i] = a[i][0] * w[0] + a[i][1] * w[1] + a[i][2] * w[2];
            BU[i] = a[i][0] * U[0] + a[i][1] * U[1] + a[i][2] * U[2];
            BV[i] = a[i][0] * V[0] + a[i][1] * V[1] + a[i][2] * V[2];
          }
        const Number m_ww = w[0] * Bw[0] + w[1] * Bw[1] + w[2] * Bw[2];
        const Number m_uu = U[0] * BU[0] + U[1] * BU[1] + U[2] * BU[2];
        const Number m_uv = U[0] * BV[0] + U[1] * BV[1] + U[2] * BV[2];
        const Number m_vv = V[0] * BV[0] + V[1] * BV[1] + V[2] * BV[2];

        value[0] = m_ww;
        for (unsigned int d = 0; d < 3; ++d)
          vector[d][0] = w[d];
        if (m_uv == Number(0.))
          {
            value[1] = m_uu;
            value[2] = m_vv;
            for (unsigned int d = 0; d < 3; ++d)
              {
                vector[d][1] = U[d];
                vector[d][2] = V[d];
              }
          }
        else
          {
            Number t, c, s;
            internal::jacobi_rotation(m_uu, m_uv, m_vv, t, c, s);
            value[1] = m_uu - t * m_uv;
            value[2] = m_vv + t * m_uv;
            for (unsigned int d = 0; d < 3; ++d)
              {
                vector[d][1] = c * U[d] - s * V[d];
                vector[d][2] = s * U[d] + c * V[d];
              }
          }
        for (unsigned int k = 0; k < 3; ++k)
          value[k] = scale * (value[k] + q);
      }

    // vector[d][a] = d-th component of the a-th eigenvector; sort descending
    // by swaps with compile-time indices
    for (unsigned int i = 0; i + 1 < dim; ++i)
      for (unsigned int j = 0; j + 1 < dim - i; ++j)
        if (value[j] < value[j + 1])
          {
            const Number tmp = value[j];
            value[j]         = value[j + 1];
            value[j + 1]     = tmp;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const Number vd  = vector[d][j];
                vector[d][j]     = vector[d][j + 1];
                vector[d][j + 1] = vd;
              }
          }

    for (unsigned int i = 0; i < dim; ++i)
      {
        result.values[i] = value[i];
        for (unsigned int d = 0; d < dim; ++d)
          result.vectors[i][d] = vector[d][i];
      }
    return result;
  }
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
