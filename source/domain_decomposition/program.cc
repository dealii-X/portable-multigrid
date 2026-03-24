#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
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

#include "domain_decomposition/portable_bnn_preconditioner.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_projected_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_hyper_cube_triangulation.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"
#include "operators/portable_subdomain_laplace_operator.h"


using namespace dealii;


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem();

  void
  run();

private:
  void
  setup_grid(unsigned int cycle);

  void
  create_subdomain_triangulations();

  void
  setup_dofs();

  void
  compute_interface_weights();

  void
  setup_matrix_free();

  void
  setup_interface_system();

  void
  setup_bnn_preconditioner();

  void
  assemble_rhs();

  void
  solve_interface();

  void
  postprocess_subdomain_solution();

  void
  output_results(const unsigned int cycle) const;

  void
  test_coarse_problem();

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  SubdomainTriangulation<dim> subdomain_triangulation;
  SubdomainDoFHandler<dim>    subdomain_dof_handler;

  FE_Q<dim> fe;

  AffineConstraints<double> subdomain_constraints;
  AffineConstraints<double> subdomain_constraints_physical;


  std::vector<types::global_dof_index> local_to_global_dof_map;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    global_interface_weights;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    global_solution_host;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    subdomain_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_solution_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_rhs_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> schur_rhs;

  std::unique_ptr<Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>
    subdomain_matrix;

  std::unique_ptr<Portable::SchurInterfaceOperator<dim, fe_degree, double>>
    interface_operator;

  std::unique_ptr<Portable::BNNPreconditioner<dim, fe_degree, double>>
    bnn_preconditioner;


  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    rhs_schur_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_interface_device;

  double             setup_time;
  ConditionalOStream pcout;
  ConditionalOStream time_details;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , fe(fe_degree)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout,
                 true &&
                   Utilities::MPI::this_mpi_process(mpi_communicator) == 0)

{}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_grid(unsigned int cycle)
{
  Timer time;
  setup_time = 0;

  triangulation.clear();

  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(cycle);

  setup_time += time.wall_time();
  time_details << "           Distributed triangulation created    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_subdomain_triangulations()
{
  Timer time;
  this->subdomain_triangulation.create_subdomain_triangulation(triangulation);

  setup_time += time.wall_time();
  time_details << "           Subdomain triangulations extracted    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  const double subdomain_diameter =
    GridTools::diameter(subdomain_triangulation.get_triangulation());

  const double subdomain_mesh_size = GridTools::maximal_cell_diameter(
    subdomain_triangulation.get_triangulation());

  std::cout << "  On subdomain "
            << Utilities::MPI::this_mpi_process(mpi_communicator)
            << ": H/h = " << subdomain_diameter / subdomain_mesh_size
            << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  Timer time;

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  subdomain_dof_handler.reinit(subdomain_triangulation, dof_handler);
  subdomain_dof_handler.distribute_subdomain_dofs();

  pcout << "           Total number of DoFs: " << dof_handler.n_dofs()
        << std::endl;

  {
    Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
    std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions = {
        {types::boundary_id(0), &homogeneous_dirichlet_bc},
        {subdomain_dof_handler.get_interface_id(), &homogeneous_dirichlet_bc}};

    subdomain_constraints.clear();

    DoFTools::make_hanging_node_constraints(
      subdomain_dof_handler.get_dof_handler(), subdomain_constraints);

    VectorTools::interpolate_boundary_values(
      subdomain_dof_handler.get_dof_handler(),
      dirichlet_boundary_functions,
      subdomain_constraints);
    subdomain_constraints.close();
  }
  {
    Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;

    subdomain_constraints_physical.clear();
    std::map<types::boundary_id, const Function<dim> *>
      dirichlet_boundary_functions_physical = {
        {types::boundary_id(0), &homogeneous_dirichlet_bc}};

    DoFTools::make_hanging_node_constraints(
      subdomain_dof_handler.get_dof_handler(), subdomain_constraints_physical);

    VectorTools::interpolate_boundary_values(
      subdomain_dof_handler.get_dof_handler(),
      dirichlet_boundary_functions_physical,
      subdomain_constraints_physical);
    subdomain_constraints_physical.close();
  }

  global_solution_host.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);

  time_details << "           Distributed DoFs setup              (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
  time.restart();

  subdomain_solution_host.reinit(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  setup_time += time.wall_time();

  time_details << "           Subdomain DoFs setup                (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_interface_weights()
{
  Timer time;
  subdomain_dof_handler.initialize_interface_dof_vector(
    global_interface_weights);

  const unsigned int n_locally_relevant_interface_indices =
    this->subdomain_dof_handler.n_locally_relevant_interface_indices();
  for (unsigned int i = 0; i < n_locally_relevant_interface_indices; ++i)
    global_interface_weights[this->subdomain_dof_handler
                               .local_to_global_interface_partitioner(i)] +=
      1.0;

  global_interface_weights.compress(VectorOperation::add);

  for (unsigned int i = 0; i < global_interface_weights.locally_owned_size();
       ++i)
    global_interface_weights.local_element(i) =
      1. / global_interface_weights.local_element(i);

  global_interface_weights.update_ghost_values();

  setup_time += time.wall_time();
  time_details << "           Interface weights computed          (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  Timer time;
  subdomain_matrix = std::make_unique<
    Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>(
    subdomain_dof_handler, subdomain_constraints);

  subdomain_matrix->initialize_dof_vector(subdomain_solution_device);
  subdomain_matrix->initialize_dof_vector(subdomain_rhs_device);

  setup_time += time.wall_time();
  time_details << "           Matrix-free operator setup          (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_interface_system()
{
  Kokkos::fence();
  Timer time;

  interface_operator =
    std::make_unique<Portable::SchurInterfaceOperator<dim, fe_degree, double>>(
      *subdomain_matrix);

  rhs_schur_device.reinit(
    this->subdomain_dof_handler.get_interface_vector_partitioner());

  solution_interface_device.reinit(
    this->subdomain_dof_handler.get_interface_vector_partitioner());

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Interface system setup              (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_bnn_preconditioner()
{
  Timer time;

  Kokkos::fence();
  this->bnn_preconditioner =
    std::make_unique<Portable::BNNPreconditioner<dim, fe_degree, double>>(
      *interface_operator, *subdomain_matrix);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           BNN preconditioner setup            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
  time.restart();
  this->bnn_preconditioner->setup_coarse_matrix();

  setup_time += time.wall_time();
  time_details << "           Coarse matrix for BNN computed      (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  Timer time;
  Kokkos::fence();

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell :
       subdomain_dof_handler.get_dof_handler().active_cell_iterators())
    {
      cell_rhs = 0;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) +=
            (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_host[local_dof_indices[i]] += cell_rhs[i];
    }

  for (const auto &index :
       subdomain_dof_handler.get_dof_info().subdomain_physical_boundary_dofs)
    system_rhs_host[index] = 0.;


  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  subdomain_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           RHS assembled                       (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
  this->interface_operator->assemble_rhs_schur(rhs_schur_device,
                                               subdomain_rhs_device);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Schur RHS assembled                 (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve_interface()
{
  Timer time;
  Kokkos::fence();
  ReductionControl solver_control(1000, 1e-16, 1e-9);

  Portable::SolverProjectedCG<
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
    cg(solver_control);

  solution_interface_device = 0.;
  cg.solve(*interface_operator,
           solution_interface_device,
           rhs_schur_device,
           *bnn_preconditioner);

  solution_interface_device.update_ghost_values();

  Kokkos::fence();

  pcout << "           Interface solver converged in "
        << solver_control.last_step() << " iterations.    (CPU/wall) "
        << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::postprocess_subdomain_solution()
{
  Timer time;
  Kokkos::fence();
  this->interface_operator->reconstruct_subdomain_solution_from_interface(
    subdomain_solution_device, solution_interface_device, subdomain_rhs_device);

  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_solution_host.update_ghost_values();

  subdomain_constraints_physical.distribute(subdomain_solution_host);

  const auto &subdomain_to_global_dof_map =
    subdomain_dof_handler.get_dof_info().subdomain_to_global_dof_map;

  for (unsigned int i = 0; i < subdomain_to_global_dof_map.size(); ++i)
    {
      const auto global_index = subdomain_to_global_dof_map[i];

      global_solution_host[global_index] = subdomain_solution_host[i];
    }

  global_solution_host.compress(VectorOperation::add);

  for (unsigned int i = 0;
       i < subdomain_dof_handler.n_locally_relevant_interface_indices();
       ++i)
    {
      const auto subdomain_index =
        subdomain_dof_handler.local_interface_to_subdomain(i);
      const auto global_index = subdomain_to_global_dof_map[subdomain_index];
      global_solution_host[global_index] *=
        global_interface_weights[subdomain_dof_handler
                                   .local_to_global_interface_partitioner(i)];
    }

  global_solution_host.update_ghost_values();

  Kokkos::fence();
  time_details << "           Subdomain solution post-processed    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  Kokkos::fence();
  Timer time;
  (void)cycle;

  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(global_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", cycle, mpi_communicator, 2);

  Vector<float> cellwise_norm(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    global_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);
  const double global_norm =
    VectorTools::compute_global_error(triangulation,
                                      cellwise_norm,
                                      VectorTools::L2_norm);

  pcout << "        solution norm: " << global_norm << std::endl;

  Kokkos::fence();
  time_details << "           Output results    (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;

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
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> test_src(
    this->subdomain_dof_handler.get_interface_vector_partitioner()),
    test_dst(this->subdomain_dof_handler.get_interface_vector_partitioner());

  test_src = 1.0;
  test_src.update_ghost_values();

  // bnn_preconditioner->vmult(test_dst, test_src);

  // bnn_preconditioner->project(test_dst, test_src);

  // test_dst.update_ghost_values();

  // test_dst.print(std::cout);

  // rhs_schur_device.print(std::cout);

  // ReductionControl solver_control(rhs_schur_device.size(),
  //                                 1e-9 * rhs_schur_device.l2_norm());

  // ReductionControl solver_control(rhs_schur_device.size(),1e-16, 1e-9);



  // Portable::SolverProjectedCG<
  //   LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  //   cg(solver_control);

  // solution_interface_device = 0.;
  // cg.solve(*interface_operator,
  //          solution_interface_device,
  //          rhs_schur_device,
  //          *bnn_preconditioner);


  SubdomainHyperCubeTriangulation<dim> tria(this->mpi_communicator);
  tria.generate_subdomain_triangulations(2);

  // tria.refine_global(1);

  tria.save_triangulations();


  DoFHandler<dim> dh_fully_distributed;
  dh_fully_distributed.reinit(tria.get_distributed_triangulation());
  dh_fully_distributed.distribute_dofs(fe);

  locally_owned_dofs = dh_fully_distributed.locally_owned_dofs();
  locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dh_fully_distributed);

  SubdomainDoFHandler<dim> dh_subdomain;

  dh_subdomain.reinit(tria.get_subdomain_triangulation(), dh_fully_distributed);
  dh_subdomain.distribute_subdomain_dofs();

  pcout << "           Total number of DoFs: " << dh_fully_distributed.n_dofs()
        << std::endl;


  std::cout << "           DoFs on subdomain "
            << Utilities::MPI::this_mpi_process(mpi_communicator) << ": "
            << dof_handler.n_dofs() << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  setup_grid(2);

  for (unsigned int cycle = 0; cycle < 8; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      triangulation.refine_global(1);

      create_subdomain_triangulations();

      pcout << "N_cells = " << triangulation.n_global_active_cells()
            << std::endl;


      setup_dofs();

      compute_interface_weights();

      setup_matrix_free();

      setup_interface_system();

      setup_bnn_preconditioner();

      assemble_rhs();

      pcout << "           setup time: " << setup_time << "s" << std::endl;

      solve_interface();

      // test_coarse_problem();

      // postprocess_subdomain_solution();

      // output_results(cycle);
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      constexpr int dim       = 2;
      constexpr int fe_degree = 1;

      LaplaceProblem<dim, fe_degree> laplace_problem;
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}