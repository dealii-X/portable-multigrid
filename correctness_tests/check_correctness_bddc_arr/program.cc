// Correctness check for SubdomainBDDCOperator::vmult_primal_pinned() (the
// new A_RR/static-condensation apply, include/operators/
// portable_subdomain_bddc_operator_wrapper.h) against the existing
// vmult() (Ahat = Pi*A*Pi, subtract-mean projected).
//
// These are genuinely different operators in general (subtract-the-group-
// average vs. pin-every-dof-in-the-group-to-zero), EXCEPT for
// BDDCVariant::corner, where every primal-constraint group is a single
// dof (a mesh vertex) -- for a singleton group, "subtract the group's own
// average" and "pin to zero" are the same operation. So with
// BDDCVariant::corner and a test vector already projected into V (zero at
// every primal dof, the precondition both operators share), vmult() and
// vmult_primal_pinned() must agree to floating-point precision. That's
// what this test checks.
//
// Needs a real multi-subdomain interface to get any primal constraints at
// all (a single, undecomposed domain has none) -- reuses the same
// SubdomainTriangulation/SubdomainDoFHandler machinery
// source/bddc_preconditioner/program.cc's create_subdomain_triangulations()/
// setup_dofs() use, trimmed to a single mesh level (no h/p-multigrid
// hierarchy needed just to exercise the operator once). Run with
// `mpirun -n N ./program` for N >= 2 (N >= 4 in 3D to see a real edge/face
// mix, though BDDCVariant::corner itself only ever needs vertices).

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "operators/portable_subdomain_bddc_operator_wrapper.h"
#include "operators/portable_subdomain_laplace_operator.h"

using namespace dealii;

constexpr int dim       = 3;
constexpr int fe_degree = 3;
using Number             = double;
using VectorType         = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  const MPI_Comm                   mpi_communicator = MPI_COMM_WORLD;
  ConditionalOStream                pcout(std::cout,
                                          Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

  const unsigned int n_subdomains = Utilities::MPI::n_mpi_processes(mpi_communicator);

  // --- split into n_subdomains coarse cells, one per rank (same
  //     best-divisor-per-axis logic as create_subdomain_triangulations()) ---
  std::vector<unsigned int> subdomains_per_axis(dim);
  {
    int remaining = static_cast<int>(n_subdomains);
    for (int d = dim; d > 0; --d)
      {
        int n_this_axis = static_cast<int>(std::pow(remaining, 1.0 / d));

        int best_divisor = 1;
        for (int j = n_this_axis; j >= 1; --j)
          if (remaining % j == 0)
            {
              best_divisor = j;
              break;
            }
        subdomains_per_axis[d - 1] = best_divisor;
        remaining /= best_divisor;
      }
  }

  Triangulation<dim> coarse_triangulation;
  Point<dim>          p1, p2;
  for (int d = 0; d < dim; ++d)
    p2[d] = 1.;
  GridGenerator::subdivided_hyper_rectangle(coarse_triangulation, subdomains_per_axis, p1, p2);

  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  // A couple of global refinements so each subdomain has more than one
  // cell -- otherwise fe_degree=3 alone already gives plenty of dofs per
  // subdomain, but a single-cell subdomain is a degenerate/uninteresting
  // case for a matrix-free operator check.
  coarse_triangulation.refine_global(2);

  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(coarse_triangulation,
                                                                                mpi_communicator);

  parallel::fullydistributed::Triangulation<dim> triangulation(mpi_communicator);
  triangulation.create_triangulation(description);

  auto subdomain_triangulation = std::make_shared<SubdomainTriangulation<dim>>();
  subdomain_triangulation->create_subdomain_triangulation(triangulation);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  SubdomainDoFHandler<dim> subdomain_dof_handler;
  subdomain_dof_handler.reinit(subdomain_triangulation, dof_handler);
  subdomain_dof_handler.distribute_subdomain_dofs();

  // Zero Dirichlet everywhere (physical boundary_id 0, plus the interface
  // id for the "full" constraints below) -- DoFTools::make_zero_boundary_
  // constraints() rather than VectorTools::interpolate_boundary_values()
  // with a ZeroFunction, since the latter drags in
  // <deal.II/numerics/vector_tools.h>'s own FEEvaluation/MatrixFreeTools
  // machinery, which collides with this codebase's Portable:: matrix-free
  // kernels at this dealii version (same reason check_correctness_cg's
  // test uses make_zero_boundary_constraints() too, not interpolate_
  // boundary_values()).
  AffineConstraints<Number> constraints;
  DoFTools::make_hanging_node_constraints(subdomain_dof_handler.get_dof_handler(), constraints);
  DoFTools::make_zero_boundary_constraints(subdomain_dof_handler.get_dof_handler(),
                                           types::boundary_id(0),
                                           constraints);
  DoFTools::make_zero_boundary_constraints(subdomain_dof_handler.get_dof_handler(),
                                           subdomain_triangulation->get_interface_id(),
                                           constraints);
  constraints.close();

  AffineConstraints<Number> constraints_physical;
  DoFTools::make_hanging_node_constraints(subdomain_dof_handler.get_dof_handler(),
                                          constraints_physical);
  DoFTools::make_zero_boundary_constraints(subdomain_dof_handler.get_dof_handler(),
                                           types::boundary_id(0),
                                           constraints_physical);
  constraints_physical.close();

  Portable::SubdomainLaplaceOperator<dim, fe_degree, Number> dirichlet_operator(
    subdomain_dof_handler, constraints, constraints_physical);

  const unsigned int n_local_coarse_dofs_corner =
    subdomain_dof_handler.get_dof_info().local_coarse_offsets[1];
  const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_communicator);

  // Every rank must reach the collective MPI::sum() below regardless of
  // whether it has anything to check locally (a rank with no shared
  // corners has nothing to compare, but must not skip the collective) --
  // so this stays a flag, not an early return.
  int local_result = 0;

  if (n_local_coarse_dofs_corner == 0)
    {
      // No vertex constraints on this rank (can happen with few enough
      // subdomains that this rank shares no corner with any other) --
      // nothing to check here, but not a failure either.
      std::cout << "Rank " << rank << ": no vertex primal constraints, skipping." << std::endl;
    }
  else
    {
      Portable::SubdomainBDDCOperator<dim, Number> corner_only_bddc_operator(
        dirichlet_operator, Portable::BDDCVariant::corner);

      VectorType src, dst_projected, dst_primal_pinned;
      corner_only_bddc_operator.initialize_dof_vector(src);
      dst_projected.reinit(src);
      dst_primal_pinned.reinit(src);

      {
        LinearAlgebra::ReadWriteVector<Number> rw(src.size());
        std::mt19937                          rng(1234 + rank);
        std::uniform_real_distribution<Number> dist(-1., 1.);
        for (unsigned int i = 0; i < rw.size(); ++i)
          rw(i) = dist(rng);
        src.import_elements(rw, VectorOperation::insert);
      }

      // Project into V (zero at every primal/vertex dof) -- the
      // precondition both vmult() and vmult_primal_pinned() share.
      corner_only_bddc_operator.project(src);

      corner_only_bddc_operator.vmult(dst_projected, src);
      corner_only_bddc_operator.vmult_primal_pinned(dst_primal_pinned, src);

      VectorType diff = dst_projected;
      diff -= dst_primal_pinned;

      const Number diff_norm      = diff.l2_norm();
      const Number reference_norm = dst_projected.l2_norm();
      const bool   passed         = diff_norm < 1e-10 * std::max(reference_norm, Number(1.0));

      std::cout << "Rank " << rank << ": n_subdomain_dofs = " << src.size()
                << ", n_vertex_constraints = " << n_local_coarse_dofs_corner
                << ", ||vmult - vmult_primal_pinned||_2 = " << diff_norm
                << ", ||vmult||_2 = " << reference_norm << (passed ? "  PASSED" : "  FAILED")
                << std::endl;

      local_result = passed ? 0 : 1;
    }

  const int global_result = Utilities::MPI::sum(local_result, mpi_communicator);

  if (global_result != 0)
    {
      pcout << "FAILED" << std::endl;
      return 1;
    }

  pcout << "PASSED" << std::endl;
  return 0;
}
