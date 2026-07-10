#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "base/portable_v_cycle_multigrid_base.h"
#include "domain_decomposition/portable_bnn_preconditioner.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_projected_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_laplace_operator.h"
#include "operators/portable_subdomain_neumann_operator_wrapper.h"



using namespace dealii;


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int n_pre_smooth, const unsigned int n_post_smooth);

  void
  run();

  void
  test_coarse_problem();

private:
  void
  create_subdomain_triangulations(unsigned int n_refinement_cycles);

  void
  setup_dofs();

  void
  compute_interface_weights();

  void
  setup_matrix_free();

  void
  setup_mg_transfers();

  void
  setup_smoothers();

  void
  setup_mg_preconditioners();

  void
  setup_interface_system();

  void
  setup_bnn_preconditioner();

  void
  assemble_rhs();

  void
  solve_interface();

  void
  matvec_ghost_timing();

  void
  postprocess_subdomain_solution();

  void
  output_results(const unsigned int cycle) const;

  void
  test_triangulation(int n_refinements);

  MPI_Comm mpi_communicator;

  parallel::fullydistributed::Triangulation<dim> triangulation;

  FE_Q<dim> fe;

  MGLevelObject<std::unique_ptr<FE_Q<dim>>> p_level_fes;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::vector<std::shared_ptr<SubdomainTriangulation<dim>>> level_subdomain_triangulations;

  std::vector<std::shared_ptr<parallel::fullydistributed::Triangulation<dim>>> level_triangulations;

  MGLevelObject<SubdomainDoFHandler<dim>> level_subdomain_dof_handlers;
  MGLevelObject<DoFHandler<dim>>          level_dof_handlers;

  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints;
  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints_physical;

  using VectorTypeMG = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  using LevelMatrixType = Portable::SubdomainLaplaceOperatorBase<dim, double>;

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;

  using TransferType = Portable::MGTransferBase<dim, double>;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_matrices;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> level_subdomain_neumann_matrices;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_dirichlet;

  MGLevelObject<std::unique_ptr<TransferType>> subdomain_mg_transfers_neumann;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_dirichlet;

  MGLevelObject<SmootherType> subdomain_mg_smoothers_neumann;

  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>> subdomain_mg_preconditioner_dirichlet;

  std::unique_ptr<Portable::VCycleMultigridBase<dim, double>> subdomain_mg_preconditioner_neumann;

  std::unique_ptr<Portable::SchurInterfaceOperator<dim, double>> interface_operator;

  std::unique_ptr<Portable::BNNPreconditioner<dim, double>> bnn_preconditioner;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> global_solution_host,
    subdomain_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> subdomain_solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> subdomain_rhs_device, schur_rhs;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> rhs_schur_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> solution_interface_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> global_interface_weights;

  const unsigned int n_pre_smooth;
  const unsigned int n_post_smooth;

  double             setup_time;
  ConditionalOStream pcout;
  ConditionalOStream time_details;

  ConvergenceTable timing_table;

  ConvergenceTable timing_table_per_iteration;

  ConvergenceTable ghost_timing_table;

  unsigned int n_cells_total;

  struct SubdomainLaplaceOperatorRunner
  {
    const unsigned int              level;
    SubdomainDoFHandler<dim>       &subomain_dof_handler;
    AffineConstraints<double>      &constraints;
    bool                            overlap_communication_computation;
    LaplaceProblem<dim, fe_degree> &parent_problem;

    template <unsigned int degree>
    void
    run()
    {
      parent_problem.level_subdomain_matrices[level] =
        std::make_unique<Portable::SubdomainLaplaceOperator<dim, degree, double>>(
          subomain_dof_handler, constraints, overlap_communication_computation);


      parent_problem.level_subdomain_neumann_matrices[level] =
        std::make_unique<typename Portable::SubdomainNeumannOperatorWrapper<dim, degree, double>>(
          *parent_problem.level_subdomain_matrices[level]);
    }
  };

  struct PolynomialTransferRunner
  {
    const unsigned int                       level;
    const Portable::MatrixFree<dim, double> &mf_coarse;
    const Portable::MatrixFree<dim, double> &mf_fine;
    AffineConstraints<double>               &constraints_coarse;
    AffineConstraints<double>               &constraints_fine;
    AffineConstraints<double>               &physical_constraints_coarse;
    AffineConstraints<double>               &physical_constraints_fine;

    LaplaceProblem<dim, fe_degree> &parent_problem;

    template <unsigned int degree_coarse, unsigned int degree_fine>
    void
    run()
    {
      parent_problem.subdomain_mg_transfers_dirichlet[level] =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();

      parent_problem.subdomain_mg_transfers_dirichlet[level]->reinit(mf_coarse,
                                                                     mf_fine,
                                                                     constraints_coarse,
                                                                     constraints_fine);

      parent_problem.subdomain_mg_transfers_neumann[level] =
        std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine, double>>();

      parent_problem.subdomain_mg_transfers_neumann[level]->reinit(mf_coarse,
                                                                   mf_fine,
                                                                   physical_constraints_coarse,
                                                                   physical_constraints_fine);
    }
  };
};
template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(const unsigned int n_pre_smooth,
                                               const unsigned int n_post_smooth)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , n_pre_smooth(n_pre_smooth)
  , n_post_smooth(n_post_smooth)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout, true && Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
  Assert(n_pre_smooth == n_post_smooth,
         ExcNotImplemented("Change of pre- and post-smoother degree "
                           "currently not possible with deal.II"));
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_subdomain_triangulations(unsigned int n_refinement_cycles)
{
  Timer time;

  const unsigned int n_subdomains = Utilities::MPI::n_mpi_processes(mpi_communicator);

  std::vector<unsigned int> subdomains_per_axis(dim);

  int remaining = n_subdomains;
  for (int d = dim; d > 0; --d)
    {
      int n_this_axis = std::pow(remaining, 1.0 / d);

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


  Triangulation<dim> coarse_triangulation;

  Point<dim> p1, p2;
  for (int d = 0; d < dim; ++d)
    p2[d] = 1.;

  GridGenerator::subdivided_hyper_rectangle(coarse_triangulation, subdomains_per_axis, p1, p2);

  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  this->level_subdomain_triangulations.clear();
  this->level_triangulations.clear();

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      if (cycle > 0)
        coarse_triangulation.refine_global(1);

      n_cells_total = coarse_triangulation.n_global_active_cells();

      const TriangulationDescription::Description<dim> description =
        TriangulationDescription::Utilities::create_description_from_triangulation(
          coarse_triangulation, mpi_communicator);

      this->triangulation.clear();
      this->triangulation.create_triangulation(description);

      this->level_triangulations.push_back(
        std::make_shared<parallel::fullydistributed::Triangulation<dim>>(mpi_communicator));
      this->level_triangulations.back()->create_triangulation(description);

      this->level_subdomain_triangulations.push_back(
        std::make_shared<SubdomainTriangulation<dim>>());

      if (cycle == 0)
        this->level_subdomain_triangulations.back()->create_subdomain_triangulation(
          *level_triangulations.back());
      else
        {
          this->level_subdomain_triangulations.back()->copy_subdomain_triangulation(
            *level_subdomain_triangulations[cycle - 1]);
          this->level_subdomain_triangulations.back()->refine_global(1);
        }
    }
  setup_time += time.wall_time();
  time_details << "           Subdomain triangulations extracted        (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  // const double subdomain_diameter = Utilities::MPI::max(
  //   GridTools::diameter(
  //     level_subdomain_triangulations.back()->get_triangulation()),
  //   mpi_communicator);

  // const double subdomain_mesh_size = Utilities::MPI::max(
  //   GridTools::maximal_cell_diameter(
  //     level_subdomain_triangulations.back()->get_triangulation()),
  //   mpi_communicator);


  // pcout << "H/h = " << subdomain_diameter / subdomain_mesh_size << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  Timer time;

  const unsigned int n_h_levels = static_cast<unsigned int>(level_triangulations.size());

  AssertDimension(n_h_levels, level_subdomain_triangulations.size());

  std::vector<unsigned int> p_levels({fe.degree});

  while (p_levels.back() > 1)
    p_levels.push_back(std::max(p_levels.back() - 1, 1u));

  p_level_fes.resize(0, p_levels.size() - 1);

  for (unsigned int level = 0; level < p_levels.size(); ++level)
    p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

  level_dof_handlers.resize(0, n_h_levels - 1 + p_level_fes.max_level());
  level_subdomain_dof_handlers.resize(0, n_h_levels - 1 + p_level_fes.max_level());

  level_subdomain_constraints.resize(0, level_dof_handlers.max_level());
  level_subdomain_constraints_physical.resize(0, level_dof_handlers.max_level());

  Functions::ZeroFunction<dim>                        homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
    {types::boundary_id(0), &homogeneous_dirichlet_bc},
    {level_subdomain_triangulations.back()->get_interface_id(), &homogeneous_dirichlet_bc}};
  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions_physical = {
    {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
    {
      DoFHandler<dim> &dof_h = level_dof_handlers[level];
      dof_h.reinit(*level_triangulations[std::min(level, n_h_levels - 1)]);

      if (level < n_h_levels)
        dof_h.distribute_dofs(*p_level_fes[0]);
      else
        dof_h.distribute_dofs(*p_level_fes[level + 1 - n_h_levels]);

      SubdomainDoFHandler<dim> &subdomain_dof_h = level_subdomain_dof_handlers[level];

      subdomain_dof_h.reinit(level_subdomain_triangulations[std::min(level, n_h_levels - 1)],
                             dof_h);
      subdomain_dof_h.distribute_subdomain_dofs();

      {
        AffineConstraints<double> &constraints = level_subdomain_constraints[level];

        constraints.clear();

        DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(), constraints);
        VectorTools::interpolate_boundary_values(subdomain_dof_h.get_dof_handler(),
                                                 dirichlet_boundary_functions,
                                                 constraints);
        constraints.close();
      }

      {
        AffineConstraints<double> &constraints_physical =
          level_subdomain_constraints_physical[level];

        constraints_physical.clear();

        DoFTools::make_hanging_node_constraints(subdomain_dof_h.get_dof_handler(),
                                                constraints_physical);
        VectorTools::interpolate_boundary_values(subdomain_dof_h.get_dof_handler(),
                                                 dirichlet_boundary_functions_physical,
                                                 constraints_physical);
        constraints_physical.close();
      }
    }

  locally_owned_dofs    = level_dof_handlers.back().locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(level_dof_handlers.back());

  pcout << "           Total number of DoFs: " << level_dof_handlers.back().n_dofs() << std::endl;

  global_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  subdomain_solution_host.reinit(level_subdomain_dof_handlers.back().get_dof_handler().n_dofs());

  setup_time += time.wall_time();
  time_details << "           Subdomain DoFs setup                      (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_interface_weights()
{
  if (!level_subdomain_dof_handlers.back().get_interface_vector_partitioner())
    return;

  level_subdomain_dof_handlers.back().initialize_interface_dof_vector(global_interface_weights);

  const unsigned int n_locally_relevant_interface_indices =
    this->level_subdomain_dof_handlers.back().n_locally_relevant_interface_indices();
  for (unsigned int i = 0; i < n_locally_relevant_interface_indices; ++i)
    global_interface_weights[this->level_subdomain_dof_handlers.back()
                               .local_to_global_interface_partitioner(i)] += 1.0;

  global_interface_weights.compress(VectorOperation::add);

  for (unsigned int i = 0; i < global_interface_weights.locally_owned_size(); ++i)
    global_interface_weights.local_element(i) = 1. / global_interface_weights.local_element(i);

  global_interface_weights.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  const unsigned int n_h_levels = static_cast<unsigned int>(level_triangulations.size());

  Kokkos::fence();
  Timer time;

  level_subdomain_matrices.resize(0, level_dof_handlers.max_level());

  level_subdomain_neumann_matrices.resize(0, level_dof_handlers.max_level());

  for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
    {
      if (level < n_h_levels)
        {
          level_subdomain_matrices[level] =
            std::make_unique<Portable::SubdomainLaplaceOperator<dim, 1, double>>(
              level_subdomain_dof_handlers[level], level_subdomain_constraints[level]);


          level_subdomain_neumann_matrices[level] =
            std::make_unique<typename Portable::SubdomainNeumannOperatorWrapper<dim, 1, double>>(
              *level_subdomain_matrices[level]);
        }
      else
        {
          SubdomainLaplaceOperatorRunner runner{level,
                                                level_subdomain_dof_handlers[level],
                                                level_subdomain_constraints[level],
                                                false,
                                                *this};


          bool success = Portable::SubdomainOperatorDispatchFactory::dispatch(
            p_level_fes[level + 1 - n_h_levels]->degree, runner);

          Assert(success, ExcMessage("Failed to find a matching polynomial degree in dispatcher."));
        }
    }

  level_subdomain_matrices.back()->initialize_dof_vector(subdomain_solution_device);
  level_subdomain_matrices.back()->initialize_dof_vector(subdomain_rhs_device);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Matrix-free operators setup               (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_transfers()
{
  Kokkos::fence();
  Timer time;

  const unsigned int n_h_levels = static_cast<unsigned int>(level_triangulations.size());

  subdomain_mg_transfers_dirichlet.resize(level_subdomain_matrices.min_level(),
                                          level_subdomain_matrices.max_level());

  subdomain_mg_transfers_neumann.resize(level_subdomain_matrices.min_level(),
                                        level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level() + 1;
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      if (level < n_h_levels)
        {
          subdomain_mg_transfers_dirichlet[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          subdomain_mg_transfers_dirichlet[level]->reinit(
            level_subdomain_matrices[level - 1]->get_matrix_free(),
            level_subdomain_matrices[level]->get_matrix_free(),
            level_subdomain_constraints[level - 1],
            level_subdomain_constraints[level]);

          subdomain_mg_transfers_neumann[level] =
            std::make_unique<Portable::GeometricTransfer<dim, 1, double>>();
          subdomain_mg_transfers_neumann[level]->reinit(
            level_subdomain_matrices[level - 1]->get_matrix_free(),
            level_subdomain_matrices[level]->get_matrix_free(),
            level_subdomain_constraints_physical[level - 1],
            level_subdomain_constraints_physical[level]);
        }
      else
        {
          const unsigned int p_coarse = p_level_fes[level - n_h_levels]->degree;
          const unsigned int p_fine   = p_level_fes[level + 1 - n_h_levels]->degree;

          PolynomialTransferRunner runner{level,
                                          level_subdomain_matrices[level - 1]->get_matrix_free(),
                                          level_subdomain_matrices[level]->get_matrix_free(),
                                          level_subdomain_constraints[level - 1],
                                          level_subdomain_constraints[level],
                                          level_subdomain_constraints_physical[level - 1],
                                          level_subdomain_constraints_physical[level],
                                          *this};

          bool success =
            Portable::PolynomialTransferDispatchFactory::dispatch(p_coarse, p_fine, runner);

          Assert(success,
                 ExcMessage("Failed to find a matching polynomial degree "
                            "pair in transfer dispatcher."));
        }
    }

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           MG transfers setup                        (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_smoothers()
{
  Kokkos::fence();
  Timer time;

  subdomain_mg_smoothers_dirichlet.resize(level_subdomain_matrices.min_level(),
                                          level_subdomain_matrices.max_level());

  subdomain_mg_smoothers_neumann.resize(level_subdomain_matrices.min_level(),
                                        level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level();
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      typename SmootherType::AdditionalData smoother_data_dirichlet;
      typename SmootherType::AdditionalData smoother_data_neumann;
      if (level > 0)
        {
          smoother_data_dirichlet.smoothing_range     = 15.;
          smoother_data_dirichlet.degree              = n_pre_smooth;
          smoother_data_dirichlet.eig_cg_n_iterations = 10;

          smoother_data_neumann.smoothing_range     = 15.;
          smoother_data_neumann.degree              = n_pre_smooth;
          smoother_data_neumann.eig_cg_n_iterations = 10;
        }
      else
        {
          smoother_data_dirichlet.smoothing_range     = 1e-3;
          smoother_data_dirichlet.degree              = numbers::invalid_unsigned_int;
          smoother_data_dirichlet.eig_cg_n_iterations = level_subdomain_matrices[0]->m();

          smoother_data_neumann.smoothing_range     = 1e-3;
          smoother_data_neumann.degree              = numbers::invalid_unsigned_int;
          smoother_data_neumann.eig_cg_n_iterations = level_subdomain_matrices[0]->m();
        }

      level_subdomain_matrices[level]->compute_diagonal();
      smoother_data_dirichlet.preconditioner =
        level_subdomain_matrices[level]->get_matrix_diagonal_inverse();

      smoother_data_neumann.preconditioner =
        level_subdomain_matrices[level]->get_matrix_diagonal_inverse_neumann();

      subdomain_mg_smoothers_dirichlet[level].initialize(*level_subdomain_matrices[level],
                                                         smoother_data_dirichlet);

      subdomain_mg_smoothers_neumann[level].initialize(*level_subdomain_neumann_matrices[level],
                                                       smoother_data_neumann);
    }

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Smoothers setup                           (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_preconditioners()
{
  Kokkos::fence();
  Timer time;

  subdomain_mg_preconditioner_dirichlet = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_matrices, subdomain_mg_transfers_dirichlet, subdomain_mg_smoothers_dirichlet);

  const bool impose_zero_mean =
    level_subdomain_matrices.back()->get_physical_boundary_dof_indices_subdomain().size() == 0;

  subdomain_mg_preconditioner_neumann = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim, double, LevelMatrixType, TransferType, SmootherType>>(
    level_subdomain_neumann_matrices,
    subdomain_mg_transfers_neumann,
    subdomain_mg_smoothers_neumann,
    impose_zero_mean);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           MG Preconditioners setup                  (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_interface_system()
{
  Kokkos::fence();
  Timer time;

  interface_operator = std::make_unique<Portable::SchurInterfaceOperator<dim, double>>(
    *level_subdomain_matrices.back(),
    *subdomain_mg_preconditioner_dirichlet,
    *subdomain_mg_preconditioner_neumann);

  rhs_schur_device.reinit(
    this->level_subdomain_dof_handlers.back().get_interface_vector_partitioner());

  solution_interface_device.reinit(
    this->level_subdomain_dof_handlers.back().get_interface_vector_partitioner());

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Interface system setup                    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_bnn_preconditioner()
{
  Kokkos::fence();
  Timer time;

  this->bnn_preconditioner =
    std::make_unique<Portable::BNNPreconditioner<dim, double>>(*interface_operator,
                                                               *level_subdomain_matrices.back());

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           BNN preconditioner setup                  (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  Kokkos::fence();
  time.restart();

  this->bnn_preconditioner->setup_coarse_matrix();

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Coarse matrix for BNN computed            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  Timer time;
  Kokkos::fence();

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    level_subdomain_dof_handlers.back().get_dof_handler().n_dofs());

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell :
       level_subdomain_dof_handlers.back().get_dof_handler().active_cell_iterators())
    {
      cell_rhs = 0;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_host[local_dof_indices[i]] += cell_rhs[i];
    }

  for (const auto &index :
       level_subdomain_dof_handlers.back().get_dof_info().subdomain_physical_boundary_dofs)
    system_rhs_host[index] = 0.;


  LinearAlgebra::ReadWriteVector<double> rw_vector(
    level_subdomain_dof_handlers.back().get_dof_handler().n_dofs());

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  subdomain_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           RHS assembled                             (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  Kokkos::fence();
  time.restart();
  this->interface_operator->assemble_rhs_schur(rhs_schur_device, subdomain_rhs_device);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Schur RHS assembled                       (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve_interface()
{
  Timer time;
  Kokkos::fence();
  // SolverControl solver_control(1000, 1e-9 * rhs_schur_device.l2_norm());
  // ReductionControl solver_control(1000, 1e-14, 1e-7);
  ReductionControl solver_control(100, 1e-16, 1e-9);

  Portable::SolverProjectedCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  // solution_interface_device = 0.;
  // cg.solve(*interface_operator,
  //          solution_interface_device,
  //          rhs_schur_device,
  //          *bnn_preconditioner);


  // pcout << "           Interface solver converged in "
  //       << solver_control.last_step() << " iterations.    (CPU/wall) "
  //       << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  solution_interface_device = 0.;

  cg.solve_enhanced(*interface_operator,
                    solution_interface_device,
                    rhs_schur_device,
                    *bnn_preconditioner);


  // SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  // cg(
  //   solver_control);

  // cg.solve(*interface_operator,
  //          solution_interface_device,
  //          rhs_schur_device,
  //          PreconditionIdentity());

  solution_interface_device.update_ghost_values();

  Kokkos::fence();
  const double time_solve = time.wall_time();

  pcout << "           Interface solver converged in " << solver_control.last_step()
        << " iterations.    (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time() << 's'
        << std::endl;

  const auto max_mg_iterations = interface_operator->get_maximum_subdomain_mg_iterations();

  const auto max_mg_iterations_dir = Utilities::MPI::max(max_mg_iterations.first, mpi_communicator);

  const auto max_mg_iterations_neu =
    Utilities::MPI::max(max_mg_iterations.second, mpi_communicator);

  const auto iterations = std::max(solver_control.last_step(), 1u);

  const std::array<double, 4> timings = bnn_preconditioner->get_timings();

  timing_table.add_value("cells", n_cells_total);
  timing_table.add_value("dofs", level_dof_handlers.back().n_dofs());
  timing_table.add_value("Dirichlet", timings[0]);
  timing_table.add_value("Neumann", timings[1]);
  timing_table.add_value("Coarse", timings[2]);
  timing_table.add_value("Project", timings[3]);
  timing_table.add_value("CG_time", time_solve);
  timing_table.add_value("Iters", solver_control.last_step());

  timing_table_per_iteration.add_value("Dir_per_iter", timings[0] / iterations);
  timing_table_per_iteration.add_value("Neu_per_iter", timings[1] / iterations);
  timing_table_per_iteration.add_value("Crs_per_iter", timings[2] / iterations);
  timing_table_per_iteration.add_value("Prj_per_iter", timings[3] / iterations);
  timing_table_per_iteration.add_value("CG_per_iter", time_solve / iterations);
  timing_table_per_iteration.add_value("dir_mg_its", max_mg_iterations_dir);
  timing_table_per_iteration.add_value("neu_mg_its", max_mg_iterations_neu);
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::matvec_ghost_timing()
{
  const bool communication_on = true;
  const bool computation_on   = true;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dummy_solution, dummy_rhs;
  level_subdomain_dof_handlers.back().initialize_interface_dof_vector(dummy_solution);
  dummy_rhs.reinit(dummy_solution);

  dummy_rhs = 1.;

  Timer time;


  std::array<double, 2> best_mv_both{{1e10, 1e10}};
  std::array<double, 2> best_only_ghost{{1e10, 1e10}};
  std::array<double, 2> best_only_comp{{1e10, 1e10}};

  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = 50;

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          communication_on,
                                          communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv_both[0] = std::min(best_mv_both[0], stat.max);
      }


      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->balance_dummy(dummy_solution,
                                            dummy_rhs,
                                            computation_on,
                                            communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv_both[1] = std::min(best_mv_both[1], stat.max);
      }
      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          !computation_on,
                                          communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_ghost[0] = std::min(best_only_ghost[0], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->balance_dummy(dummy_solution,
                                            dummy_rhs,
                                            !computation_on,
                                            communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_ghost[1] = std::min(best_only_ghost[1], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          interface_operator->vmult_dummy(dummy_solution,
                                          dummy_rhs,
                                          computation_on,
                                          !communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_comp[0] = std::min(best_only_comp[0], stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          bnn_preconditioner->balance_dummy(dummy_solution,
                                            dummy_rhs,
                                            computation_on,
                                            !communication_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_comp[1] = std::min(best_only_comp[1], stat.max);
      }
    }


  ghost_timing_table.add_value("cells", n_cells_total);
  ghost_timing_table.add_value("dofs", level_dof_handlers.back().n_dofs());

  ghost_timing_table.add_value("subdomain_total", best_mv_both[0]);
  ghost_timing_table.add_value("subdomain_compute", best_only_comp[0]);
  ghost_timing_table.add_value("subdomain_communicate", best_only_ghost[0]);

  ghost_timing_table.add_value("coarse_total", best_mv_both[1]);
  ghost_timing_table.add_value("coarse_compute", best_only_comp[1]);
  ghost_timing_table.add_value("coarse_communicate", best_only_ghost[1]);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::postprocess_subdomain_solution()
{
  Timer time;
  Kokkos::fence();
  this->interface_operator->reconstruct_subdomain_solution_from_interface(subdomain_solution_device,
                                                                          solution_interface_device,
                                                                          subdomain_rhs_device);

  LinearAlgebra::ReadWriteVector<double> rw_vector(
    level_subdomain_dof_handlers.back().get_dof_handler().n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_solution_host.update_ghost_values();

  level_subdomain_constraints_physical.back().distribute(subdomain_solution_host);

  const auto &subdomain_to_global_dof_map =
    level_subdomain_dof_handlers.back().get_dof_info().subdomain_to_global_dof_map;

  for (unsigned int i = 0; i < subdomain_to_global_dof_map.size(); ++i)
    {
      const auto global_index = subdomain_to_global_dof_map[i];

      global_solution_host[global_index] = subdomain_solution_host[i];
    }

  global_solution_host.compress(VectorOperation::add);


  for (unsigned int i = 0;
       i < level_subdomain_dof_handlers.back().n_locally_relevant_interface_indices();
       ++i)
    {
      const auto subdomain_index =
        level_subdomain_dof_handlers.back().local_interface_to_subdomain(i);
      const auto global_index = subdomain_to_global_dof_map[subdomain_index];
      global_solution_host[global_index] *=
        global_interface_weights[level_subdomain_dof_handlers.back()
                                   .local_to_global_interface_partitioner(i)];
    }

  global_solution_host.update_ghost_values();

  Kokkos::fence();
  time_details << "           Subdomain solution post-processed         (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  Kokkos::fence();
  Timer time;
  (void)cycle;

  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(level_dof_handlers.back());
  // data_out.add_data_vector(global_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", cycle, mpi_communicator, 2);

  Vector<float> cellwise_norm(level_triangulations.back()->n_active_cells());
  VectorTools::integrate_difference(level_dof_handlers.back(),
                                    global_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);
  const double global_norm = VectorTools::compute_global_error(*level_triangulations.back(),
                                                               cellwise_norm,
                                                               VectorTools::L2_norm);

  pcout << "        solution norm: " << global_norm << std::endl;

  Kokkos::fence();
  time_details << "           Output results                            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;

  // Vector<float> cellwise_norm_subdomain(
  //   subdomain_triangulation.get_triangulation().n_active_cells());
  // VectorTools::integrate_difference(subdomain_dof_handler.get_dof_handler(),
  //                                   subdomain_solution_host,
  //                                   Functions::ZeroFunction<dim>(),
  //                                   cellwise_norm_subdomain,
  //                                   QGauss<dim>(fe.degree + 2),
  //                                   VectorTools::L2_norm);
  // const double subdomain_norm = VectorTools::compute_global_error(
  //   subdomain_triangulation.get_triangulation(),
  //   cellwise_norm_subdomain,
  //   VectorTools::L2_norm);


  // std::cout << " solution norm on subdomain "
  //           << subdomain_dof_handler.get_subdomain_id() << ": "
  //           << subdomain_norm << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test_coarse_problem()
{
  MGLevelObject<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> temp1, temp2, src,
    err;

  temp1.resize(0, this->level_subdomain_matrices.max_level());
  temp2.resize(0, this->level_subdomain_matrices.max_level());
  src.resize(0, this->level_subdomain_matrices.max_level());
  err.resize(0, this->level_subdomain_matrices.max_level());

  for (unsigned int level = 0; level <= this->level_subdomain_matrices.max_level(); ++level)
    {
      level_subdomain_matrices[level]->initialize_dof_vector(src[level]);
      src[level] = 1.;

      temp1[level].reinit(src[level]);
      temp2[level].reinit(src[level]);

      // level_subdomain_constraints[level].distribute(src[level]);


      this->level_subdomain_matrices[level]->vmult(temp1[level], src[level]);
      this->level_subdomain_matrices[level]->vmult_bk3(temp2[level], src[level]);

      // temp1[level].print(std::cout);
      // temp2[level].print(std::cout);



      err[level] = temp1[level];
      err[level] -= temp2[level];

      MPI_Barrier(mpi_communicator);

      // std::cout << "temp1[" << level << "] = ";
      // for (const auto x : temp1[level])
      //   std::cout << x << "  ";
      // std::cout << std::endl;

      // MPI_Barrier(mpi_communicator);

      // std::cout << "temp2[" << level << "] = ";
      // for (const auto x : temp2[level])
      //   std::cout << x << "  ";
      // std::cout << std::endl;
      // MPI_Barrier(mpi_communicator);

      std::cout << "  temp1[" << level << "].l2_norm() = " << temp1[level].l2_norm() << std::endl;
      std::cout << "  temp2[" << level << "].l2_norm() = " << temp2[level].l2_norm() << std::endl;
      std::cout << "  error[" << level << "].l2_norm() = " << err[level].l2_norm() << std::endl;
      MPI_Barrier(mpi_communicator);
    }

  // LinearAlgebra::distributed::Vector<double, MemorySpace::Default> temp1,
  // temp2,
  //   src, err;

  // const auto &matrix = *level_subdomain_matrices.back();
  // matrix.initialize_dof_vector(src);
  // src = 1.;

  // temp1.reinit(src);
  // temp2.reinit(src);

  // MPI_Barrier(mpi_communicator);

  // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  //   matrix.vmult(temp1, src);
  // MPI_Barrier(mpi_communicator);

  // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  //   matrix.vmult_bk3(temp2, src);

  // MPI_Barrier(mpi_communicator);

  // err = temp1;
  // err -= temp2;

  // MPI_Barrier(mpi_communicator);

  // std::cout << "temp1[" << level << "] = ";
  // for (const auto x : temp1[level])
  //   std::cout << x << "  ";
  // std::cout << std::endl;

  // MPI_Barrier(mpi_communicator);

  // std::cout << "temp2[" << level << "] = ";
  // for (const auto x : temp2[level])
  //   std::cout << x << "  ";
  // std::cout << std::endl;
  // MPI_Barrier(mpi_communicator);

  // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  //   {
  //     std::cout << "  temp1.l2_norm() = " << temp1.l2_norm() << std::endl;
  //     std::cout << "  temp2.l2_norm() = " << temp2.l2_norm() << std::endl;
  //     std::cout << "  error.l2_norm() = " << err.l2_norm() << std::endl;
  //   }
  // Kokkos::fence();
  // MPI_Barrier(mpi_communicator);
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test_triangulation(int n_refinements)
{}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  for (unsigned int cycle = 0; cycle < 12; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      create_subdomain_triangulations(cycle + 1);

      pcout << "    N_cells = " << level_triangulations.back()->n_global_active_cells()
            << std::endl;

      setup_dofs();

      compute_interface_weights();

      setup_matrix_free();

      setup_mg_transfers();

      setup_smoothers();

      setup_mg_preconditioners();

      setup_interface_system();

      setup_bnn_preconditioner();

      assemble_rhs();

      pcout << "           setup time: " << setup_time << "s" << std::endl;

      solve_interface();

      matvec_ghost_timing();

      // test_coarse_problem();

      postprocess_subdomain_solution();

      output_results(cycle);

      // test_triangulation();

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << std::endl << std::endl;

          timing_table.set_scientific("Dirichlet", true);
          timing_table.set_precision("Dirichlet", 3);
          timing_table.set_scientific("Neumann", true);
          timing_table.set_precision("Neumann", 3);
          timing_table.set_scientific("Coarse", true);
          timing_table.set_precision("Coarse", 3);
          timing_table.set_scientific("Project", true);
          timing_table.set_precision("Project", 3);
          timing_table.set_scientific("CG_time", true);
          timing_table.set_precision("CG_time", 3);

          timing_table_per_iteration.set_scientific("Dir_per_iter", true);
          timing_table_per_iteration.set_precision("Dir_per_iter", 3);
          timing_table_per_iteration.set_scientific("Neu_per_iter", true);
          timing_table_per_iteration.set_precision("Neu_per_iter", 3);
          timing_table_per_iteration.set_scientific("Crs_per_iter", true);
          timing_table_per_iteration.set_precision("Crs_per_iter", 3);
          timing_table_per_iteration.set_scientific("Prj_per_iter", true);
          timing_table_per_iteration.set_precision("Prj_per_iter", 3);
          timing_table_per_iteration.set_scientific("CG_per_iter", true);
          timing_table_per_iteration.set_precision("CG_per_iter", 3);

          timing_table.write_text(std::cout);

          std::cout << std::endl << std::endl;

          timing_table_per_iteration.write_text(std::cout);

          std::cout << std::endl << std::endl;


          ghost_timing_table.set_scientific("subdomain_total", true);
          ghost_timing_table.set_precision("subdomain_total", 4);
          ghost_timing_table.set_scientific("subdomain_compute", true);
          ghost_timing_table.set_precision("subdomain_compute", 4);
          ghost_timing_table.set_scientific("subdomain_communicate", true);
          ghost_timing_table.set_precision("subdomain_communicate", 4);


          ghost_timing_table.set_scientific("coarse_total", true);
          ghost_timing_table.set_precision("coarse_total", 4);
          ghost_timing_table.set_scientific("coarse_compute", true);
          ghost_timing_table.set_precision("coarse_compute", 4);
          ghost_timing_table.set_scientific("coarse_communicate", true);
          ghost_timing_table.set_precision("coarse_communicate", 4);

          ghost_timing_table.write_text(std::cout);

          std::cout << std::endl << std::endl;
        }
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      constexpr int dim       = 3;
      constexpr int fe_degree = 4;

      const unsigned int n_pre_smooth  = 5;
      const unsigned int n_post_smooth = 5;

      LaplaceProblem<dim, fe_degree> laplace_problem(n_pre_smooth, n_post_smooth);
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}