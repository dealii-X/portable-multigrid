#ifndef testsuite_gmg_util_h
#define testsuite_gmg_util_h

// Geometric multigrid on a refined [-1, 1]^dim cube with zero Dirichlet
// values and a unit load: levels from create_geometric_coarsening_sequence(),
// GeometricTransfer, Chebyshev smoothing (coarse level: Chebyshev as
// solver), Portable::VCycleMultigrid as preconditioner of CG.

#include <deal.II/base/function.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <memory>

#include "../operators/deformed_mesh_util.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"

namespace gmg_test
{
  using namespace dealii;

  using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  // OperatorType(dof_handler, constraints, overlap) must derive from
  // Portable::LaplaceOperatorBase<dim, VectorType>
  template <int dim, int fe_degree, typename OperatorType>
  void
  solve(const unsigned int n_refinements,
        const unsigned int smoothing_degree,
        const unsigned int min_iterations,
        const unsigned int max_iterations)
  {
    using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorType>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorType>;

    deallog << "dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_refinements = " << n_refinements << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(n_refinements);

    const auto coarse_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        triangulation, RepartitioningPolicyTools::MinimalGranularityPolicy<dim>(16));

    const unsigned int max_level = coarse_triangulations.size() - 1;

    const FE_Q<dim>                                 fe(fe_degree);
    const MappingQ<dim>                             mapping(fe_degree);
    MGLevelObject<DoFHandler<dim>>                  dof_handlers(0, max_level);
    MGLevelObject<AffineConstraints<double>>        constraints(0, max_level);
    MGLevelObject<std::unique_ptr<LevelMatrixType>> matrices(0, max_level);
    MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, double>>> transfers(0, max_level);
    MGLevelObject<SmootherType>                                           smoothers(0, max_level);

    for (unsigned int level = 0; level <= max_level; ++level)
      {
        dof_handlers[level].reinit(*coarse_triangulations[level]);
        dof_handlers[level].distribute_dofs(fe);

        constraints[level].reinit(dof_handlers[level].locally_owned_dofs(),
                                  DoFTools::extract_locally_relevant_dofs(dof_handlers[level]));
        VectorTools::interpolate_boundary_values(
          mapping, dof_handlers[level], 0, Functions::ZeroFunction<dim>(), constraints[level]);
        constraints[level].close();

        matrices[level] =
          std::make_unique<OperatorType>(dof_handlers[level], constraints[level], false);
      }

    for (unsigned int level = 1; level <= max_level; ++level)
      {
        transfers[level] = std::make_unique<Portable::GeometricTransfer<dim, fe_degree, double>>();
        transfers[level]->reinit(*matrices[level - 1],
                                 *matrices[level],
                                 constraints[level - 1],
                                 constraints[level]);
      }

    for (unsigned int level = 0; level <= max_level; ++level)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (level > 0)
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = smoothing_degree;
            smoother_data.eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = matrices[0]->m();
          }
        matrices[level]->compute_diagonal();
        smoother_data.preconditioner = matrices[level]->get_matrix_diagonal_inverse();
        smoothers[level].initialize(*matrices[level], smoother_data);
      }

    const DoFHandler<dim> &dof_handler = dof_handlers[max_level];

    VectorType solution, rhs;
    matrices[max_level]->initialize_dof_vector(solution);
    matrices[max_level]->initialize_dof_vector(rhs);
    deformed_mesh_test::reference_unit_rhs<dim, double>(mapping,
                                                        dof_handler,
                                                        constraints[max_level],
                                                        fe_degree + 1,
                                                        dof_handler.locally_owned_dofs(),
                                                        DoFTools::extract_locally_relevant_dofs(
                                                          dof_handler),
                                                        MPI_COMM_WORLD,
                                                        rhs);

    const Portable::VCycleMultigrid<dim, double, Portable::MGTransferBase<dim, double>> mg(
      matrices, transfers, smoothers);

    deallog << "n_levels = " << max_level + 1 << ", n_dofs = " << dof_handler.n_dofs() << std::endl;

    SolverControl        control(100, 1e-10 * rhs.l2_norm());
    SolverCG<VectorType> cg(control);
    check_solver_within_range(cg.solve(*matrices[max_level], solution, rhs, mg),
                              control.last_step(),
                              min_iterations,
                              max_iterations);
  }
} // namespace gmg_test

#endif
