// Correctness check for Portable::VectorLaplaceOperator / BK4::Parallel::
// KokkosKernel: compares vmult_bk4() (the custom Kokkos kernel) against
// vmult_dealii() (real deal.II's own Portable::FEEvaluation path, taken as
// ground truth) on a random vector, and sanity-checks compute_diagonal().
//
// This is deliberately small and single-shot -- see tests/
// abstracted_batched_laplace/program.cc for the full timing/speedup/error
// comparison table this is modeled after, applied here to a genuinely
// vector-valued (FESystem) problem instead of a scalar one.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>
#include <random>

#include "operators/portable_vector_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;

  // Runs the comparison for one (dim, fe_degree, n_components) combination
  // on a small hyper-cube mesh. Returns the relative error between
  // vmult_bk4() and vmult_dealii(), which should be at machine precision.
  template <int dim, int fe_degree, int n_components>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_components = " << n_components << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);

    const FE_Q<dim> scalar_fe(fe_degree);
    FESystem<dim>   fe(scalar_fe, n_components);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero(n_components);
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);
    constraints.close();

    Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double> op(
      dof_handler, constraints, /* overlap_communication_computation = */ false);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src, dst_dealii, dst_bk4;
    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst_dealii);
    op.initialize_dof_vector(dst_bk4);

    {
      std::mt19937                           gen(42);
      std::uniform_real_distribution<double> dist(-1., 1.);

      LinearAlgebra::ReadWriteVector<double> rw(locally_owned_dofs);
      for (const auto idx : locally_owned_dofs)
        rw(idx) = dist(gen);
      src.import_elements(rw, VectorOperation::insert);
    }

    // vmult_dealii() reads src at constrained DoFs directly (it relies on
    // copy_constrained_values() afterwards), while vmult_bk4() never reads
    // constrained entries at all -- zero them up front so both variants see
    // the same effective input, exactly as abstracted_batched_laplace/
    // program.cc does for the scalar operator.
    op.get_matrix_free().set_constrained_values(0., src);

    op.vmult_dealii(dst_dealii, src);
    op.vmult_bk4(dst_bk4, src);

    // vmult_tensor_core() is dim == 3 only and needs an actual nvcc/CUDA
    // build to do anything (see bk4_cuda_kernels.cuh) -- this sandbox has
    // neither a CUDA compiler nor a CUDA-enabled Kokkos build, so this is
    // dead code (never executed) purely to force the compiler to
    // instantiate and typecheck vmult_tensor_core()'s non-CUDA fallback
    // branch (the Assert(false, ...) path) as part of this compile check.
    if constexpr (dim == 3)
      if (false)
        op.template vmult_tensor_core<2>(dst_bk4, src);

    op.get_matrix_free().set_constrained_values(0., dst_dealii);
    op.get_matrix_free().set_constrained_values(0., dst_bk4);

    const double norm_dealii = dst_dealii.l2_norm();

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = dst_bk4;
    diff -= dst_dealii;
    const double rel_err = norm_dealii > 0 ? diff.l2_norm() / norm_dealii : diff.l2_norm();

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_dealii| = " << norm_dealii
          << ", rel_err(bk4 vs dealii) = " << rel_err << std::endl;

    // Sanity-check compute_diagonal(): a Laplacian's diagonal must be
    // strictly positive everywhere (compute_diagonal() itself Asserts this
    // in debug mode already -- this just makes the check visible here too).
    op.compute_diagonal();
    const auto &inverse_diagonal = op.get_matrix_diagonal_inverse()->get_vector();
    LinearAlgebra::ReadWriteVector<double> diagonal_host(locally_owned_dofs);
    diagonal_host.import_elements(inverse_diagonal, VectorOperation::insert);
    double min_inverse_diagonal = std::numeric_limits<double>::max();
    for (const auto idx : locally_owned_dofs)
      min_inverse_diagonal = std::min(min_inverse_diagonal, diagonal_host(idx));
    pcout << "  min(1/diagonal) = " << min_inverse_diagonal << std::endl << std::endl;

    return rel_err;
  }
} // namespace multigrid

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double max_rel_err = 0.;

  try
    {
      max_rel_err = std::max(max_rel_err, run_test<2, 2, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 2, 3>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 3, 2>(MPI_COMM_WORLD, pcout));
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-10;

  if (max_rel_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_rel_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_rel_err << " (tolerance " << tolerance << ")"
        << std::endl;

  return 0;
}
