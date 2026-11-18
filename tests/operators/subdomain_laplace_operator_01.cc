// Portable::SubdomainLaplaceOperator on the subdomain of each rank of a
// curved, sheared mesh: vmult, Tvmult and vmult_bk3 with Dirichlet
// conditions on the physical boundary and the subdomain interface,
// vmult_neumann with Dirichlet conditions on the physical boundary only, and
// both diagonals, against the CPU FEValues reference. Errors are per
// subdomain; the largest one over all ranks is reported.

#include "../tests.h"

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <memory>

#include "deformed_mesh_util.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "operators/portable_subdomain_laplace_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int fe_degree>
void
test()
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using Operator   = Portable::SubdomainLaplaceOperator<dim, fe_degree, Number>;

  constexpr double tolerance = 1e-12;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

  const FE_Q<dim>     fe(fe_degree);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     global_dof_handler(triangulation);
  global_dof_handler.distribute_dofs(fe);

  const auto subdomain_triangulation = std::make_shared<SubdomainTriangulation<dim>>();
  subdomain_triangulation->create_subdomain_triangulation(triangulation);
  subdomain_triangulation->set_manifold(0, manifold);

  SubdomainDoFHandler<dim> subdomain_dof_handler;
  subdomain_dof_handler.reinit(subdomain_triangulation, global_dof_handler);
  subdomain_dof_handler.distribute_subdomain_dofs();

  const DoFHandler<dim> &dof_handler = subdomain_dof_handler.get_dof_handler();

  // serial triangulation of the subdomain: every DoF is owned
  const IndexSet all_dofs = complete_index_set(dof_handler.n_dofs());

  const Functions::ZeroFunction<dim, Number> zero;

  AffineConstraints<Number> constraints_dirichlet;
  {
    const std::map<types::boundary_id, const Function<dim, Number> *> functions = {
      {types::boundary_id(0), &zero}, {subdomain_dof_handler.get_interface_id(), &zero}};
    VectorTools::interpolate_boundary_values(mapping, dof_handler, functions, constraints_dirichlet);
    constraints_dirichlet.close();
  }

  AffineConstraints<Number> constraints_physical;
  {
    const std::map<types::boundary_id, const Function<dim, Number> *> functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(mapping, dof_handler, functions, constraints_physical);
    constraints_physical.close();
  }

  Operator op(subdomain_dof_handler, constraints_dirichlet, /* overlap = */ false);

  VectorType src, dst, dst_ref_dirichlet, diag_ref_dirichlet, dst_ref_physical, diag_ref_physical;
  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  op.initialize_dof_vector(dst_ref_dirichlet);
  op.initialize_dof_vector(diag_ref_dirichlet);
  op.initialize_dof_vector(dst_ref_physical);
  op.initialize_dof_vector(diag_ref_physical);

  // nonzero on constrained DoFs: the kernels mask them, the final copy must
  // reproduce them
  fill_source(src, all_dofs);

  reference_laplace<dim, Number>(mapping,
                                 dof_handler,
                                 constraints_dirichlet,
                                 fe_degree + 1,
                                 all_dofs,
                                 all_dofs,
                                 MPI_COMM_SELF,
                                 src,
                                 dst_ref_dirichlet,
                                 diag_ref_dirichlet);
  reference_laplace<dim, Number>(mapping,
                                 dof_handler,
                                 constraints_physical,
                                 fe_degree + 1,
                                 all_dofs,
                                 all_dofs,
                                 MPI_COMM_SELF,
                                 src,
                                 dst_ref_physical,
                                 diag_ref_physical);

  deallog << "subdomain n_dofs = " << dof_handler.n_dofs() << ", interface n_dofs (all ranks) = "
          << Utilities::MPI::sum(
               subdomain_dof_handler.get_dof_info().subdomain_interface_dofs.n_elements(),
               MPI_COMM_WORLD)
          << std::endl;

  const auto check = [&](const std::string &name, const VectorType &dst_ref, auto &&apply) {
    dst = 0.;
    apply(dst, src);
    Kokkos::fence();
    report(name,
           Utilities::MPI::max(relative_l2_difference(dst, dst_ref), MPI_COMM_WORLD),
           tolerance);
  };

  check("vmult (Dirichlet)", dst_ref_dirichlet, [&](VectorType &d, const VectorType &s) {
    op.vmult(d, s);
  });
  check("Tvmult (Dirichlet)", dst_ref_dirichlet, [&](VectorType &d, const VectorType &s) {
    op.Tvmult(d, s);
  });
  check("vmult_bk3 (Dirichlet)", dst_ref_dirichlet, [&](VectorType &d, const VectorType &s) {
    op.vmult_bk3(d, s);
  });
  check("vmult_neumann", dst_ref_physical, [&](VectorType &d, const VectorType &s) {
    op.vmult_neumann(d, s);
  });

  op.compute_diagonal();
  Kokkos::fence();
  report("compute_diagonal (Dirichlet)",
         Utilities::MPI::max(max_relative_diagonal_error(
                               op.get_matrix_diagonal_inverse()->get_vector(),
                               diag_ref_dirichlet,
                               all_dofs,
                               MPI_COMM_SELF),
                             MPI_COMM_WORLD),
         tolerance);
  report("compute_diagonal (Neumann)",
         Utilities::MPI::max(max_relative_diagonal_error(
                               op.get_matrix_diagonal_inverse_neumann()->get_vector(),
                               diag_ref_physical,
                               all_dofs,
                               MPI_COMM_SELF),
                             MPI_COMM_WORLD),
         tolerance);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>();
  test<2, 2>();
  test<2, 3>();
  test<3, 1>();
  test<3, 2>();
  test<3, 3>();
}
