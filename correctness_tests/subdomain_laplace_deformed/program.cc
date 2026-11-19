// Deformed-mesh correctness check for Portable::SubdomainLaplaceOperator
// (portable_subdomain_laplace_operator.h), the operator behind the
// BNN/BDDC domain-decomposition family. Each MPI rank extracts its locally
// owned cells into a SubdomainTriangulation (serial, interface faces get
// boundary id 100 + rank) and builds a SubdomainDoFHandler on it, exactly
// as source/domain_decomposition/program.cc does. The subdomain copy keeps
// the curved vertex positions of the global mesh but straight edges, so
// its cells are non-affine trilinear hexahedra: non-orthogonal Jacobians
// everywhere, which is what the G-tensor contraction has to get right.
//
// Compared against the CPU FEValues reference of
// correctness_tests/common/deformed_mesh_test_utils.h on the subdomain
// DoFHandler with MappingQ(fe_degree):
//   vmult(), vmult_bk3(), Tvmult(), get_matrix_diagonal_inverse():
//     Dirichlet problem, interface + physical boundary constrained;
//   vmult_neumann(), get_matrix_diagonal_inverse_neumann():
//     only the physical boundary constrained.
// With 2 ranks every subdomain has an interface; with 1 rank there is none.
//
// This is the regression guard for the contraction-index fix in
// compute_G_tensors() (fb60d37): the pre-fix operator contracted the
// inverse Jacobian over the wrong index and fails here (passes on
// axis-aligned meshes, where the two agree).
//
// Prints one line per variant and PASSED/FAILED with exit code 0/1.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <memory>

#include "deformed_mesh_test_utils.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "operators/portable_subdomain_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;
  using namespace deformed_mesh_test;

  template <int dim, int fe_degree>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    using Number     = double;
    using VectorType = DeviceVectorType<Number>;
    using Operator   = Portable::SubdomainLaplaceOperator<dim, fe_degree, Number>;

    pcout << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    const CurvedShearedManifold<dim>          manifold;
    make_deformed_triangulation(triangulation, manifold, dim == 2 ? 3 : 2);

    const FE_Q<dim>     fe(fe_degree);
    const MappingQ<dim> mapping(fe_degree);
    DoFHandler<dim>     global_dof_handler(triangulation);
    global_dof_handler.distribute_dofs(fe);

    // this rank's subdomain
    const auto subdomain_triangulation = std::make_shared<SubdomainTriangulation<dim>>();
    subdomain_triangulation->create_subdomain_triangulation(triangulation);

    SubdomainDoFHandler<dim> subdomain_dof_handler;
    subdomain_dof_handler.reinit(subdomain_triangulation, global_dof_handler);
    subdomain_dof_handler.distribute_subdomain_dofs();

    const DoFHandler<dim> &dof_handler = subdomain_dof_handler.get_dof_handler();

    // The subdomain DoFHandler lives on a serial triangulation: every DoF is
    // owned, and the reference runs on MPI_COMM_SELF.
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

    VectorType src, dst, dst_ref_dirichlet, diag_ref_dirichlet, dst_ref_physical,
      diag_ref_physical;
    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst);
    op.initialize_dof_vector(dst_ref_dirichlet);
    op.initialize_dof_vector(diag_ref_dirichlet);
    op.initialize_dof_vector(dst_ref_physical);
    op.initialize_dof_vector(diag_ref_physical);

    // nonzero on constrained DoFs too: the operator's kernels mask them and
    // the final copy must reproduce them
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

    const unsigned int n_interface_dofs = Utilities::MPI::sum(
      subdomain_dof_handler.get_dof_info().subdomain_interface_dofs.n_elements(),
      mpi_communicator);
    pcout << "  subdomain dofs (rank 0) = " << dof_handler.n_dofs()
          << ", interface dofs (all ranks) = " << n_interface_dofs << std::endl;

    // Errors are per subdomain; report and judge the worst rank.
    Results results;

    const auto check = [&](const std::string &name, const VectorType &dst_ref, auto &&apply) {
      dst = 0.;
      apply(dst, src);
      Kokkos::fence();
      results.record(pcout,
                     name,
                     Utilities::MPI::max(relative_l2_difference(dst, dst_ref), mpi_communicator));
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
    check("vmult_neumann (physical boundary only)",
          dst_ref_physical,
          [&](VectorType &d, const VectorType &s) { op.vmult_neumann(d, s); });

    op.compute_diagonal();
    Kokkos::fence();
    results.record(pcout,
                   "compute_diagonal (Dirichlet)",
                   Utilities::MPI::max(max_relative_diagonal_error(
                                         op.get_matrix_diagonal_inverse()->get_vector(),
                                         diag_ref_dirichlet,
                                         all_dofs,
                                         MPI_COMM_SELF),
                                       mpi_communicator));
    results.record(pcout,
                   "compute_diagonal (Neumann)",
                   Utilities::MPI::max(max_relative_diagonal_error(
                                         op.get_matrix_diagonal_inverse_neumann()->get_vector(),
                                         diag_ref_physical,
                                         all_dofs,
                                         MPI_COMM_SELF),
                                       mpi_communicator));

    pcout << std::endl;
    return results.max_err;
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double max_err = 0.;

  try
    {
      max_err = std::max(max_err, run_test<3, 1>(MPI_COMM_WORLD, pcout));
      max_err = std::max(max_err, run_test<3, 2>(MPI_COMM_WORLD, pcout));
      max_err = std::max(max_err, run_test<3, 3>(MPI_COMM_WORLD, pcout));

      max_err = std::max(max_err, run_test<2, 1>(MPI_COMM_WORLD, pcout));
      max_err = std::max(max_err, run_test<2, 2>(MPI_COMM_WORLD, pcout));
      max_err = std::max(max_err, run_test<2, 3>(MPI_COMM_WORLD, pcout));
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-12;

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
