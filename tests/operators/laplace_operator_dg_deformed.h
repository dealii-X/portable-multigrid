// Portable::LaplaceOperatorDG (SIPG, FE_DGQ) on a curved, sheared mesh or
// on the axis-aligned unit cube: vmult, Tvmult and compute_diagonal()
// against a CPU FEValues/FEInterfaceValues assembly of the SIPG operator.
// compute_diagonal() returns the diagonal of the cell terms only, which is
// what it is compared with.

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q.h>

#include "deformed_mesh_util.h"
#include "operators/portable_laplace_operator_dg.h"

using namespace dealii;
using namespace deformed_mesh_test;

// dst = A_SIPG src, diag_full = diag(A_SIPG), diag_cell = diag(cell terms)
template <int dim, typename Number>
void
reference_sipg(const Mapping<dim>             &mapping,
               const DoFHandler<dim>          &dof_handler,
               const unsigned int              fe_degree,
               const unsigned int              n_q_points_1d,
               const IndexSet                 &locally_owned_dofs,
               const IndexSet                 &locally_relevant_dofs,
               const MPI_Comm                  mpi_communicator,
               const DeviceVectorType<Number> &src_device,
               DeviceVectorType<Number>       &dst_device,
               DeviceVectorType<Number>       &diag_full_device,
               DeviceVectorType<Number>       &diag_cell_device)
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
  HostVector diag_full_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  HostVector diag_cell_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  const FiniteElement<dim> &fe            = dof_handler.get_fe();
  const unsigned int        dofs_per_cell = fe.n_dofs_per_cell();

  const QGauss<dim>     cell_quadrature(n_q_points_1d);
  const QGauss<dim - 1> face_quadrature(n_q_points_1d);

  FEValues<dim> fe_values(mapping, fe, cell_quadrature, update_gradients | update_JxW_values);
  FEInterfaceValues<dim> fe_iv(mapping,
                               fe,
                               face_quadrature,
                               update_values | update_gradients | update_JxW_values |
                                 update_normal_vectors);
  FEFaceValues<dim>      fe_face_values(mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_gradients | update_JxW_values |
                                     update_normal_vectors);

  const auto penalty_factor = [fe_degree](const double extent_minus, const double extent_plus) {
    return fe_degree * (fe_degree + 1.) * 0.5 * (1. / extent_minus + 1. / extent_plus);
  };

  // apply a local matrix to the local src values and scatter rows
  const auto apply_local = [&](const FullMatrix<double>                   &matrix,
                               const std::vector<types::global_dof_index> &dofs,
                               HostVector                                 *diag) {
    const unsigned int n = dofs.size();
    Vector<double>     x(n), y(n);
    for (unsigned int j = 0; j < n; ++j)
      x(j) = src_host(dofs[j]);
    matrix.vmult(y, x);
    for (unsigned int i = 0; i < n; ++i)
      {
        dst_host(dofs[i]) += y(i);
        if (diag != nullptr)
          (*diag)(dofs[i]) += matrix(i, i);
        diag_full_host(dofs[i]) += matrix(i, i);
      }
  };

  FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      // cell term
      fe_values.reinit(cell);
      cell_matrix = 0.;
      for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) +=
              fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
      cell->get_dof_indices(local_dof_indices);
      apply_local(cell_matrix, local_dof_indices, &diag_cell_host);

      for (const unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            {
              // Dirichlet boundary (id 0 in the kernel's convention)
              Assert(cell->face(f)->boundary_id() == 0, ExcInternalError());

              const double extent = cell->measure() / cell->face(f)->measure();
              const double sigma  = penalty_factor(extent, extent);

              fe_face_values.reinit(cell, f);
              cell_matrix = 0.;
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const Tensor<1, dim> &n = fe_face_values.normal_vector(q);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const double u    = fe_face_values.shape_value(j, q);
                        const double dn_u = fe_face_values.shape_grad(j, q) * n;
                        const double v    = fe_face_values.shape_value(i, q);
                        const double dn_v = fe_face_values.shape_grad(i, q) * n;
                        cell_matrix(i, j) +=
                          ((2. * sigma * u - dn_u) * v - u * dn_v) * fe_face_values.JxW(q);
                      }
                }
              apply_local(cell_matrix, local_dof_indices, nullptr);
              continue;
            }

          // interior face: assembled once, by the cell with the lower id
          // among owned cells, and by the lower subdomain id for faces
          // shared with another rank
          const auto neighbor = cell->neighbor(f);
          Assert(!neighbor->has_children() && cell->level() == neighbor->level(),
                 ExcMessage("The reference assumes a globally refined mesh."));

          if (neighbor->is_locally_owned())
            {
              if (neighbor->id() < cell->id())
                continue;
            }
          else if (neighbor->subdomain_id() < cell->subdomain_id())
            continue;

          const unsigned int nf = cell->neighbor_of_neighbor(f);

          const double extent_minus = cell->measure() / cell->face(f)->measure();
          const double extent_plus  = neighbor->measure() / neighbor->face(nf)->measure();
          const double sigma        = penalty_factor(extent_minus, extent_plus);

          fe_iv.reinit(cell, f, numbers::invalid_unsigned_int, neighbor, nf, numbers::invalid_unsigned_int);

          const unsigned int n_iv_dofs = fe_iv.n_current_interface_dofs();
          FullMatrix<double> face_matrix(n_iv_dofs, n_iv_dofs);
          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
              const Tensor<1, dim> &n = fe_iv.normal_vector(q);
              for (unsigned int i = 0; i < n_iv_dofs; ++i)
                for (unsigned int j = 0; j < n_iv_dofs; ++j)
                  {
                    const double jump_u = fe_iv.jump_in_shape_values(j, q);
                    const double avg_dn_u = fe_iv.average_of_shape_gradients(j, q) * n;
                    const double jump_v = fe_iv.jump_in_shape_values(i, q);
                    const double avg_dn_v = fe_iv.average_of_shape_gradients(i, q) * n;
                    face_matrix(i, j) +=
                      (sigma * jump_u * jump_v - avg_dn_u * jump_v - jump_u * avg_dn_v) *
                      fe_iv.JxW(q);
                  }
            }
          apply_local(face_matrix, fe_iv.get_interface_dof_indices(), nullptr);
        }
    }

  dst_host.compress(VectorOperation::add);
  diag_full_host.compress(VectorOperation::add);
  diag_cell_host.compress(VectorOperation::add);

  LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
  for (const auto i : locally_owned_dofs)
    rw(i) = static_cast<Number>(dst_host(i));
  dst_device.import_elements(rw, VectorOperation::insert);
  for (const auto i : locally_owned_dofs)
    rw(i) = static_cast<Number>(diag_full_host(i));
  diag_full_device.import_elements(rw, VectorOperation::insert);
  for (const auto i : locally_owned_dofs)
    rw(i) = static_cast<Number>(diag_cell_host(i));
  diag_cell_device.import_elements(rw, VectorOperation::insert);
}




template <int dim, int fe_degree>
void
test(const bool overlap, const bool deformed)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Operator   = Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, Number>;

  constexpr double tolerance = 1e-12;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  if (deformed)
    make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);
  else
    {
      GridGenerator::hyper_cube(triangulation, 0., 1.);
      triangulation.refine_global(dim == 2 ? 3 : 2);
    }

  const FE_DGQ<dim>   fe(fe_degree);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // weak boundary conditions: no constraints
  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  Operator op(mapping, dof_handler, constraints, overlap);

  VectorType src, dst, dst_ref, diag_full_ref, diag_cell_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_full_ref);
  op.initialize_dof_vector(diag_cell_ref);

  fill_source(src, locally_owned_dofs);

  reference_sipg<dim, Number>(mapping,
                              dof_handler,
                              fe_degree,
                              fe_degree + 1,
                              locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD,
                              src,
                              dst_ref,
                              diag_full_ref,
                              diag_cell_ref);

  deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;

  dst = 0.;
  op.vmult(dst, src);
  Kokkos::fence();
  report("vmult", relative_l2_difference(dst, dst_ref), tolerance);

  dst = 0.;
  op.Tvmult(dst, src);
  Kokkos::fence();
  report("Tvmult", relative_l2_difference(dst, dst_ref), tolerance);

  op.compute_diagonal();
  Kokkos::fence();
  report("compute_diagonal (cell terms)",
         max_relative_diagonal_error(op.get_matrix_diagonal_inverse()->get_vector(),
                                     diag_cell_ref,
                                     locally_owned_dofs,
                                     MPI_COMM_WORLD),
         tolerance);
}



inline void
test_all(const bool overlap, const bool deformed)
{
  test<2, 1>(overlap, deformed);
  test<2, 2>(overlap, deformed);
  test<2, 3>(overlap, deformed);
  test<3, 1>(overlap, deformed);
  test<3, 2>(overlap, deformed);
  test<3, 3>(overlap, deformed);
}
