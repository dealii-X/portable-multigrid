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

#include "domain_decomposition/subdomain_dof_handler.h"
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
  setup_grid();

  void
  create_subdomain_triangulations();

  void
  setup_dofs();

  void
  compute_interface_weights();

  void
  setup_matrix_free();

  void
  assemble_rhs();

  void
  solve_subdomain();

  void
  post_process_subdomain_solution();

  void
  output_results(const unsigned int cycle) const;

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
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_rhs_interior;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_rhs_interface;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> schur_rhs;

  std::unique_ptr<Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>
    subdomain_matrix;

  ConditionalOStream pcout;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , fe(fe_degree)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_subdomain_triangulations()
{
  this->subdomain_triangulation.create_subdomain_triangulation(triangulation);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  subdomain_dof_handler.reinit(subdomain_triangulation, dof_handler);
  subdomain_dof_handler.distribute_subdomain_dofs();

  pcout << "  Total number of DoFs: " << dof_handler.n_dofs() << std::endl;

  std::cout << "    Number of DoFs on subdomain "
            << subdomain_dof_handler.get_subdomain_id() << ": "
            << subdomain_dof_handler.get_dof_handler().n_dofs() << std::endl;

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

  subdomain_solution_host.reinit(
    subdomain_dof_handler.get_dof_handler().n_dofs());
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_interface_weights()
{
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

  // LinearAlgebra::distributed::Vector<double> vec(
  //   this->subdomain_dof_handler.get_interface_vector_partitioner());
  // std::cout << "On subdomain " <<
  // this->subdomain_dof_handler.get_subdomain_id()
  //           << ": ";
  // for (unsigned int i = 0;
  //      i < vec.locally_owned_size() +
  //            this->subdomain_dof_handler.get_interface_vector_partitioner()
  //              ->n_ghost_indices();
  //      ++i)
  //   std::cout << i << " /  "
  //             <<
  //             this->subdomain_dof_handler.get_interface_vector_partitioner()
  //                  ->local_to_global(i)
  //             << " ; ";
  // std::cout << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  subdomain_matrix = std::make_unique<
    Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>(
    subdomain_dof_handler, subdomain_constraints);

  subdomain_matrix->initialize_dof_vector(subdomain_solution_device);
  subdomain_matrix->initialize_dof_vector(subdomain_rhs_device);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  std::cout << "Before assembly process \n";


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

  std::cout << "Before physical bondary dofs\n";
  std::cout << std::endl;

  for (const auto &index :
       subdomain_dof_handler.get_dof_info().subdomain_physical_boundary_dofs)
    system_rhs_host[index] = 0.;


  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  std::cout << "Before import\n";
  std::cout << std::endl;


  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  subdomain_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  std::cout << "after import\n";
  std::cout << std::endl;


  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    rhs_schur_device(
      this->subdomain_dof_handler.get_interface_vector_partitioner());

  std::cout << "after create schur\n";
  std::cout << std::endl;



  this->subdomain_matrix->assemble_rhs_schur(rhs_schur_device,
                                             subdomain_rhs_device);

  rhs_schur_device.update_ghost_values();

  std::cout << "after assemble schur\n";
  std::cout << std::endl;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> rhs_schur_host(
    this->subdomain_dof_handler.get_interface_vector_partitioner());


  rw_vector.reinit(rhs_schur_device.locally_owned_elements());
  // rw_vector.import_elements(rhs_schur_device, VectorOperation::insert);


  // rhs_schur_host.import_elements(rw_vector, VectorOperation::add);
  // rhs_schur_host.update_ghost_values();

  // rhs_schur_host.print(std::cout);
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve_subdomain()
{
  SolverControl solver_control(subdomain_rhs_device.size(),
                               1e-12 * subdomain_rhs_device.l2_norm());

  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  cg.solve(*subdomain_matrix,
           subdomain_solution_device,
           subdomain_rhs_device,
           PreconditionIdentity());

  std::cout << "   On subdomain "
            << subdomain_triangulation.get_topology_info().subdomain_id
            << " solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::post_process_subdomain_solution()
{
  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_constraints_physical.distribute(subdomain_solution_host);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  (void)cycle;

  const auto subdomain_topology =
    this->subdomain_triangulation.get_topology_info();

  DataOut<dim> data_out;

  data_out.attach_dof_handler(subdomain_dof_handler.get_dof_handler());
  data_out.add_data_vector(subdomain_solution_host, "solution");
  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
    "./",
    "solution_subdomain_" + std::to_string(subdomain_topology.subdomain_id),
    cycle,
    mpi_communicator);


  Vector<float> cellwise_norm(
    subdomain_triangulation.get_triangulation().n_active_cells());
  VectorTools::integrate_difference(subdomain_dof_handler.get_dof_handler(),
                                    subdomain_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  const double global_norm = VectorTools::compute_global_error(
    subdomain_triangulation.get_triangulation(),
    cellwise_norm,
    VectorTools::L2_norm);

  std::cout << "    On subdomain " << subdomain_dof_handler.get_subdomain_id()
            << "  solution norm: " << global_norm << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  setup_grid();

  for (unsigned int cycle = 0; cycle < 3; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      triangulation.refine_global(1);

      create_subdomain_triangulations();


      std::cout << "after create tria\n";
      std::cout << std::endl;

      setup_dofs();


      std::cout << "after setup dofs\n";
      std::cout << std::endl;

      compute_interface_weights();


      std::cout << "after compute weights\n";
      std::cout << std::endl;

      setup_matrix_free();

      std::cout << "after setup MF\n";
      std::cout << std::endl;

      assemble_rhs();


      std::cout << "after assmble RHS\n";
      std::cout << std::endl;

      // solve_subdomain();


      // std::cout << "after solve\n";
      // std::cout << std::endl;

      // post_process_subdomain_solution();


      // std::cout << "after post_process \n";
      // std::cout << std::endl;

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