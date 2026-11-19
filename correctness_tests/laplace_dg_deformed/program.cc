// Deformed-mesh correctness check for Portable::LaplaceOperatorDG
// (portable_laplace_operator_dg.h, SIPG with FE_DGQ): vmult() and Tvmult()
// are compared against a CPU reference assembled with FEValues and
// FEInterfaceValues on the curved, sheared mesh of
// correctness_tests/common/deformed_mesh_test_utils.h with MappingQ(p).
//
// The reference reproduces the kernels of laplace_dg_kokkos_kernels.h:
//   cells:          (grad u, grad v)
//   interior faces: sigma [u][v] - {grad u}.n [v] - [u] {grad v}.n,
//                   sigma = p (p + 1) / 2 (1/h- + 1/h+), h = |cell| / |face|,
//                   n the normal of the first cell, [.] = first - second
//   Dirichlet faces (boundary id 0, mirror u_out = -u_in, dn u_out = dn u_in):
//                   2 sigma_b u v - dn u v - u dn v, sigma_b = p (p + 1) / h.
// Each interior face is assembled once (the rank with the lower subdomain id
// takes shared faces), then compress(add).
//
// compute_diagonal() goes through MatrixFreeTools::compute_diagonal(), which
// only sees the cell terms; its result is compared against both the full
// SIPG diagonal and the cell-only diagonal, and the smaller error counts.
//
// Prints one line per variant and PASSED/FAILED with exit code 0/1.

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q.h>

#include <iostream>

#include "deformed_mesh_test_utils.h"
#include "operators/portable_laplace_operator_dg.h"

namespace multigrid
{
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



  // With `deformed == false` the axis-aligned unit cube is used instead: a
  // sanity check of the reference itself, since on axis-aligned cells the
  // face geometry is trivial and every reasonable implementation agrees.
  template <int dim, int fe_degree>
  double
  run_test(const MPI_Comm      mpi_communicator,
           ConditionalOStream &pcout,
           const bool          overlap,
           const bool          deformed,
           unsigned int       &n_skipped)
  {
    using Number     = double;
    using VectorType = DeviceVectorType<Number>;
    using Operator   = Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, Number>;

    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", overlap = " << (overlap ? "true" : "false")
          << ", mesh = " << (deformed ? "deformed" : "axis-aligned cube") << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
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

    // DG: boundary conditions are weak, no constraints
    AffineConstraints<Number> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    constraints.close();

    const auto op =
      make_operator_or_skip<Operator>(pcout, overlap, mapping, dof_handler, constraints, overlap);
    if (!op)
      {
        ++n_skipped;
        return -1.;
      }

    VectorType src, dst, dst_ref, diag_full_ref, diag_cell_ref;
    op->initialize_dof_vector(src);
    op->initialize_dof_vector(dst);
    op->initialize_dof_vector(dst_ref);
    op->initialize_dof_vector(diag_full_ref);
    op->initialize_dof_vector(diag_cell_ref);

    fill_source(src, locally_owned_dofs);

    reference_sipg<dim, Number>(mapping,
                                dof_handler,
                                fe_degree,
                                fe_degree + 1,
                                locally_owned_dofs,
                                locally_relevant_dofs,
                                mpi_communicator,
                                src,
                                dst_ref,
                                diag_full_ref,
                                diag_cell_ref);

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_ref| = " << dst_ref.l2_norm()
          << std::endl;

    Results results;

    const auto check = [&](const std::string &name, auto &&apply) {
      dst = 0.;
      apply(dst, src);
      Kokkos::fence();
      results.record(pcout, name, relative_l2_difference(dst, dst_ref));
    };

    check("vmult", [&](VectorType &d, const VectorType &s) { op->vmult(d, s); });
    check("Tvmult", [&](VectorType &d, const VectorType &s) { op->Tvmult(d, s); });

    op->compute_diagonal();
    Kokkos::fence();
    {
      const VectorType &inverse_diagonal = op->get_matrix_diagonal_inverse()->get_vector();
      const double      err_full         = max_relative_diagonal_error(
        inverse_diagonal, diag_full_ref, locally_owned_dofs, mpi_communicator);
      const double err_cell = max_relative_diagonal_error(
        inverse_diagonal, diag_cell_ref, locally_owned_dofs, mpi_communicator);
      pcout << "    compute_diagonal vs full SIPG diagonal: max_rel_err = " << err_full
            << std::endl
            << "    compute_diagonal vs cell-term diagonal: max_rel_err = " << err_cell
            << std::endl;
      results.record(pcout, "compute_diagonal (closer of the two)", std::min(err_full, err_cell));
    }

    pcout << std::endl;
    return results.max_err;
  }

  template <int dim, int fe_degree>
  void
  run_both_overlap_modes(const MPI_Comm      mpi_communicator,
                         ConditionalOStream &pcout,
                         const bool          deformed,
                         double             &max_err,
                         unsigned int       &n_skipped)
  {
    for (const bool overlap : {false, true})
      {
        const double err =
          run_test<dim, fe_degree>(mpi_communicator, pcout, overlap, deformed, n_skipped);
        if (err >= 0.)
          max_err = std::max(max_err, err);
      }
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double       max_err   = 0.;
  unsigned int n_skipped = 0;

  // "./program cube" runs on the axis-aligned unit cube instead
  const bool deformed = !(argc > 1 && std::string(argv[1]) == "cube");

  try
    {
      run_both_overlap_modes<3, 1>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);
      run_both_overlap_modes<3, 2>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);
      run_both_overlap_modes<3, 3>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);

      run_both_overlap_modes<2, 1>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);
      run_both_overlap_modes<2, 2>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);
      run_both_overlap_modes<2, 3>(MPI_COMM_WORLD, pcout, deformed, max_err, n_skipped);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-12;

  if (n_skipped > 0)
    pcout << "Note: " << n_skipped
          << " overlap configuration(s) skipped (build without device-aware MPI)." << std::endl;

  if (max_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_err << " (tolerance " << tolerance << ")"
        << std::endl;
  return 0;
}
