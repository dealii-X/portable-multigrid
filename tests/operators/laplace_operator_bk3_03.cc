// Portable::LaplaceOperatorBK3::vmult() with each cell kernel (abstracted
// and plain, each with the shape values in the team scratch and in the
// kernel closure) and the compute_diagonal() that goes with it on a curved,
// sheared mesh against the CPU FEValues reference, in double and float, with
// p + 1 and p + 2 quadrature points, with and without
// communication/computation overlap.

#include "../tests.h"

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "deformed_mesh_util.h"
#include "operators/portable_laplace_operator_bk3.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
void
test(const bool overlap)
{
  using VectorType = DeviceVectorType<Number>;
  using Operator   = Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d>;

  const double tolerance = std::is_same_v<Number, double> ? 1e-12 : 1e-5;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << ", "
          << (std::is_same_v<Number, double> ? "double" : "float")
          << ", overlap = " << (overlap ? "true" : "false") << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FE_Q<dim>     fe(fe_degree);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(), constraints);
  constraints.close();

  Operator op(dof_handler, constraints, overlap);

  VectorType src, dst, dst_ref, diag_ref;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref);
  op.initialize_dof_vector(diag_ref);

  fill_source(src, locally_owned_dofs);

  reference_laplace<dim, Number>(mapping,
                                 dof_handler,
                                 constraints,
                                 n_q_points_1d,
                                 locally_owned_dofs,
                                 locally_relevant_dofs,
                                 MPI_COMM_WORLD,
                                 src,
                                 dst_ref,
                                 diag_ref);

  const auto check = [&](const std::string &name, const Portable::BK3VmultKernel kernel) {
    op.set_vmult_kernel(kernel);
    AssertThrow(op.get_vmult_kernel() == kernel, ExcInternalError());

    dst = 0.;
    op.vmult(dst, src);
    Kokkos::fence();
    report(name, relative_l2_difference(dst, dst_ref), tolerance);

    // compute_diagonal() follows the vmult() kernel
    op.compute_diagonal();
    Kokkos::fence();
    report(name + " diagonal",
           max_relative_diagonal_error(op.get_matrix_diagonal_inverse()->get_vector(),
                                       diag_ref,
                                       locally_owned_dofs,
                                       MPI_COMM_WORLD),
           tolerance);
  };

  check("abstracted", Portable::BK3VmultKernel::abstracted);
  check("abstracted_constant_shape_values",
        Portable::BK3VmultKernel::abstracted_constant_shape_values);
  check("plain", Portable::BK3VmultKernel::plain);
  check("plain_constant_shape_values", Portable::BK3VmultKernel::plain_constant_shape_values);
}



template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
void
test_both_overlap_modes()
{
  test<dim, fe_degree, Number, n_q_points_1d>(false);
  test<dim, fe_degree, Number, n_q_points_1d>(true);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_both_overlap_modes<2, 1, double>();
  test_both_overlap_modes<2, 3, double>();

  test_both_overlap_modes<3, 1, double>();
  test_both_overlap_modes<3, 2, double>();
  test_both_overlap_modes<3, 3, double>();
  test_both_overlap_modes<3, 4, double>();

  // over-integration as in performance_tests/bp3
  test_both_overlap_modes<3, 2, double, 4>();

  test_both_overlap_modes<3, 1, float>();
  test_both_overlap_modes<3, 4, float>();
}
