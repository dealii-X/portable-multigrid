#ifndef deformed_mesh_test_utils_h
#define deformed_mesh_test_utils_h

// Shared pieces of the deformed-mesh correctness tests
// (correctness_tests/*_deformed/): a curved, sheared, non-affine mesh on
// which every entry of the geometry tensor varies per quadrature point, and
// a CPU FEValues reference for the Laplace operator in the Dirichlet
// convention all Portable operators use.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace deformed_mesh_test
{
  using namespace dealii;

  template <typename Number>
  using DeviceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  // Smooth, anisotropic, sheared and non-affine diffeomorphism of the unit
  // cube with an analytic inverse. The coarse hyper_cube is pushed through
  // map_forward() and the manifold is attached, so refinement produces
  // genuinely curved cells for MappingQ(p > 1).
  template <int dim>
  class CurvedShearedManifold : public ChartManifold<dim, dim, dim>
  {
  public:
    static Point<dim>
    map_forward(const Point<dim> &chart_point)
    {
      if constexpr (dim == 2)
        {
          const double xi  = chart_point[0];
          const double eta = chart_point[1];

          const double y = 1.1 * eta;
          const double x = 1.3 * xi + 0.18 * y + 0.15 * std::sin(numbers::PI * eta);
          return Point<dim>(x, y);
        }
      else
        {
          const double xi   = chart_point[0];
          const double eta  = chart_point[1];
          const double zeta = chart_point[2];

          const double z = 0.7 * zeta;
          const double y = 1.1 * eta + 0.12 * std::sin(numbers::PI * zeta);
          const double x = 2.3 * xi + 0.18 * y +
                           0.15 * std::sin(numbers::PI * eta) * std::cos(numbers::PI * zeta);
          return Point<dim>(x, y, z);
        }
    }

    static Point<dim>
    map_backward(const Point<dim> &space_point)
    {
      if constexpr (dim == 2)
        {
          const double x = space_point[0];
          const double y = space_point[1];

          const double eta = y / 1.1;
          const double xi  = (x - 0.18 * y - 0.15 * std::sin(numbers::PI * eta)) / 1.3;
          return Point<dim>(xi, eta);
        }
      else
        {
          const double x = space_point[0];
          const double y = space_point[1];
          const double z = space_point[2];

          const double zeta = z / 0.7;
          const double eta  = (y - 0.12 * std::sin(numbers::PI * zeta)) / 1.1;
          const double xi =
            (x - 0.18 * y - 0.15 * std::sin(numbers::PI * eta) * std::cos(numbers::PI * zeta)) /
            2.3;
          return Point<dim>(xi, eta, zeta);
        }
    }

    Point<dim>
    push_forward(const Point<dim> &chart_point) const override
    {
      return map_forward(chart_point);
    }

    Point<dim>
    pull_back(const Point<dim> &space_point) const override
    {
      return map_backward(space_point);
    }

    std::unique_ptr<Manifold<dim, dim>>
    clone() const override
    {
      return std::make_unique<CurvedShearedManifold<dim>>();
    }
  };



  // The unit hyper-cube pushed through the map, with the manifold attached
  // (it must outlive the triangulation) and refined n_refine times.
  template <int dim>
  void
  make_deformed_triangulation(parallel::distributed::Triangulation<dim> &triangulation,
                              const CurvedShearedManifold<dim>          &manifold,
                              const unsigned int                         n_refine)
  {
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    GridTools::transform(&CurvedShearedManifold<dim>::map_forward, triangulation);
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, manifold);
    triangulation.refine_global(n_refine);
  }



  // Deterministic, rank-independent fill of the locally owned entries. With
  // `constraints`, the constrained entries are set to zero instead (for
  // operators that read src at constrained DoFs, like the deal.II cell_loop
  // paths); without, they are deliberately nonzero so that the final
  // dst[c] = src[c] copy is exercised.
  template <typename Number>
  void
  fill_source(DeviceVectorType<Number>        &vec,
              const IndexSet                  &locally_owned_dofs,
              const AffineConstraints<Number> *constraints = nullptr)
  {
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = (constraints != nullptr && constraints->is_constrained(i)) ?
                Number(0) :
                static_cast<Number>(std::sin(0.37 * static_cast<double>(i + 1)) +
                                    0.5 * std::cos(0.13 * static_cast<double>(i + 3)));
    vec.import_elements(rw, VectorOperation::insert);
  }



  template <typename Number>
  double
  relative_l2_difference(const DeviceVectorType<Number> &a, const DeviceVectorType<Number> &b)
  {
    DeviceVectorType<Number> diff(a);
    diff -= b;
    const double norm_b = b.l2_norm();
    return norm_b > 0 ? diff.l2_norm() / norm_b : diff.l2_norm();
  }



  // CPU reference for the Laplace operator (component-wise for vector-valued
  // elements): dst = A src and diag = diag(A), assembled cell by cell with
  // FEValues in double, in the Dirichlet convention of the Portable
  // operators: constrained src entries are read as zero, constrained rows are
  // not accumulated, and afterwards dst[c] = src[c] and diag[c] = 1 on every
  // constrained DoF c.
  template <int dim, typename Number>
  void
  reference_laplace(const Mapping<dim>              &mapping,
                    const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<Number> &constraints,
                    const unsigned int               n_q_points_1d,
                    const IndexSet                  &locally_owned_dofs,
                    const IndexSet                  &locally_relevant_dofs,
                    const MPI_Comm                   mpi_communicator,
                    const DeviceVectorType<Number>  &src_device,
                    DeviceVectorType<Number>        &dst_device,
                    DeviceVectorType<Number>        &diag_device)
  {
    using HostVector = LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;

    HostVector src_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    {
      LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
      rw.import_elements(src_device, VectorOperation::insert);
      for (const auto i : locally_owned_dofs)
        src_host(i) = rw(i);
    }
    src_host.update_ghost_values();

    HostVector dst_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    HostVector diag_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const QGauss<dim>         quadrature(n_q_points_1d);
    FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();
    const unsigned int n_components  = fe.n_components();

    FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                       x_local(dofs_per_cell), y_local(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0.;

          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  double grad_i_dot_grad_j = 0.;
                  for (unsigned int c = 0; c < n_components; ++c)
                    grad_i_dot_grad_j +=
                      fe_values.shape_grad_component(i, q, c) * fe_values.shape_grad_component(j, q, c);
                  cell_matrix(i, j) += grad_i_dot_grad_j * fe_values.JxW(q);
                }

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            x_local(j) = constraints.is_constrained(local_dof_indices[j]) ?
                           0. :
                           src_host(local_dof_indices[j]);

          cell_matrix.vmult(y_local, x_local);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (!constraints.is_constrained(local_dof_indices[i]))
              {
                dst_host(local_dof_indices[i]) += y_local(i);
                diag_host(local_dof_indices[i]) += cell_matrix(i, i);
              }
        }

    dst_host.compress(VectorOperation::add);
    diag_host.compress(VectorOperation::add);

    for (const auto i : locally_owned_dofs)
      if (constraints.is_constrained(i))
        {
          dst_host(i)  = src_host(i);
          diag_host(i) = 1.;
        }

    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = static_cast<Number>(dst_host(i));
    dst_device.import_elements(rw, VectorOperation::insert);

    for (const auto i : locally_owned_dofs)
      rw(i) = static_cast<Number>(diag_host(i));
    diag_device.import_elements(rw, VectorOperation::insert);
  }



  // max_i |1 / inverse_diagonal_i - diag_ref_i| / |diag_ref_i| over the
  // owned DoFs, reduced over the communicator.
  template <typename Number>
  double
  max_relative_diagonal_error(const DeviceVectorType<Number> &inverse_diagonal,
                              const DeviceVectorType<Number> &diag_ref,
                              const IndexSet                 &locally_owned_dofs,
                              const MPI_Comm                  mpi_communicator)
  {
    LinearAlgebra::ReadWriteVector<Number> rw_inv(locally_owned_dofs), rw_ref(locally_owned_dofs);
    rw_inv.import_elements(inverse_diagonal, VectorOperation::insert);
    rw_ref.import_elements(diag_ref, VectorOperation::insert);

    double err = 0.;
    for (const auto i : locally_owned_dofs)
      {
        const double d     = 1. / static_cast<double>(rw_inv(i));
        const double d_ref = static_cast<double>(rw_ref(i));
        err                = std::max(err, std::abs(d - d_ref) / std::abs(d_ref));
      }
    return Utilities::MPI::max(err, mpi_communicator);
  }



  // Constructs an operator, or returns null (and prints SKIPPED) when the
  // build refuses communication/computation overlap without device-aware
  // MPI: that is a property of the build, not a bug in the operator.
  template <typename OperatorType, typename... Args>
  std::unique_ptr<OperatorType>
  make_operator_or_skip(ConditionalOStream &pcout, const bool overlap, Args &&...args)
  {
    try
      {
        return std::make_unique<OperatorType>(std::forward<Args>(args)...);
      }
    catch (const ExceptionBase &exc)
      {
        if (overlap && std::string(exc.what()).find("device-aware MPI") != std::string::npos)
          {
            pcout << "  SKIPPED: overlap needs device-aware MPI on this build" << std::endl
                  << std::endl;
            return nullptr;
          }
        throw;
      }
  }



  // Per-variant bookkeeping: one line per compared vmult variant, and the
  // largest error seen.
  struct Results
  {
    double       max_err   = 0.;
    unsigned int n_skipped = 0;

    void
    record(ConditionalOStream &pcout, const std::string &name, const double err)
    {
      pcout << "    " << name << ": rel_err = " << err << std::endl;
      max_err = std::max(max_err, err);
    }
  };

} // namespace deformed_mesh_test

#endif
