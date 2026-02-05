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

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"

#include <deal.II/numerics/data_out.h>

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
  setup_matrix_free();

  void
  assemble_rhs();

  void
  solve_subdomain();

  void
  post_process_solution();
  
  void
  output_results(const unsigned int cycle) const;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  Triangulation<dim> subdomain_triangulation;

  DoFHandler<dim> subdomain_dof_handler;

  FE_Q<dim> fe;

  AffineConstraints<double> subdomain_constraints;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_device;

  const unsigned int subdomain_id;

  std::unique_ptr<Portable::LaplaceOperatorBase<dim, double>> subdomain_matrix;

  ConditionalOStream pcout;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , fe(fe_degree)
  , subdomain_id(Utilities::MPI::this_mpi_process(mpi_communicator))
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
  // std::cout << "On MPI rank "
  //           << Utilities::MPI::this_mpi_process(mpi_communicator)
  //           << ", locally_owned_subdomain = "
  //           << triangulation.locally_owned_subdomain()
  //           << ", n_cells = " << triangulation.n_locally_owned_active_cells()
  //           << std::endl;


  std::vector<Point<dim>>    subdomain_vertices;
  std::vector<CellData<dim>> subdomain_cell_data;
  SubCellData                subcell_data;

  std::map<unsigned int, unsigned int> global_to_local_vertex_map;

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          CellData<dim> cell_data;
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              const unsigned int global_vertex_index = cell->vertex_index(v);

              if (global_to_local_vertex_map.find(global_vertex_index) ==
                  global_to_local_vertex_map.end())
                {
                  global_to_local_vertex_map[global_vertex_index] =
                    subdomain_vertices.size();
                  subdomain_vertices.push_back(cell->vertex(v));
                }
              cell_data.vertices[v] =
                global_to_local_vertex_map[global_vertex_index];
            }

          cell_data.material_id = cell->material_id();
          cell_data.manifold_id = cell->manifold_id();
          subdomain_cell_data.push_back(cell_data);

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->at_boundary(f))
                {
                  CellData<dim - 1> face_data;
                  for (unsigned int fv = 0;
                       fv < GeometryInfo<dim>::vertices_per_face;
                       ++fv)
                       {
                    face_data.vertices[fv] =
                      global_to_local_vertex_map[cell->face(f)->vertex_index(
                        fv)];
                        std::cout<<cell->face(f)->vertex(fv)<<"; ";
                      }
                        std::cout<<std::endl;

                      

                  face_data.boundary_id = cell->face(f)->boundary_id();
                  face_data.manifold_id = cell->face(f)->manifold_id();
                      

                  if constexpr (dim == 2)
                    subcell_data.boundary_lines.push_back(face_data);
                  else if constexpr (dim == 3)
                    subcell_data.boundary_quads.push_back(face_data);



                  Assert(subcell_data.check_consistency(dim),
                         ExcMessage(
                           "Subcell data are not filled consistenly."));
                }
            }
        }

    }
                        std::cout<<std::endl<<std::endl;


  GridTools::consistently_order_cells<dim>(subdomain_cell_data);

  this->subdomain_triangulation.clear();
  this->subdomain_triangulation.create_triangulation(subdomain_vertices,
                                                     subdomain_cell_data,
                                                     subcell_data);

                                                     const types::boundary_id interface_id = 101; // Choose an ID that isn't 0

  for (auto &cell : subdomain_triangulation.active_cell_iterators())
    {
      for (unsigned int f : cell->face_indices())
        {
          if (cell->at_boundary(f))
            {
              // Check if this face was explicitly set in our extraction loop.
              // If it has the 'internal_face_boundary_id', it's a new interface.
              if (cell->face(f)->boundary_id() == numbers::internal_face_boundary_id)
                {
                  cell->face(f)->set_boundary_id(interface_id);
                }
            }
        }
    }
  // std::cout << "Verification on Rank "
  //           << Utilities::MPI::this_mpi_process(mpi_communicator) << ":"
  //           << std::endl
  //           << "  Distributed locally owned cells: "
  //           << triangulation.n_locally_owned_active_cells() << std::endl
  //           << "  Extracted subdomain cells      : "
  //           << subdomain_triangulation.n_active_cells() << std::endl;

  AssertDimension(subdomain_triangulation.n_active_cells(),
                  triangulation.n_locally_owned_active_cells());

                  unsigned int real_boundary_count = 0;
unsigned int interface_count = 0;

for (const auto &cell : subdomain_triangulation.active_cell_iterators())
  for (unsigned int f : cell->face_indices())
    if (cell->at_boundary(f))
      {
        if (cell->face(f)->boundary_id() == 0)
          real_boundary_count++;
        else
          interface_count++;
      }

std::cout << "Rank " << subdomain_id 
          << ": Physical faces = " << real_boundary_count 
          << ", Interface faces = " << interface_count << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  // Timer time;

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  pcout << "  Total number of DoFs: " << dof_handler.n_dofs() << std::endl;

  subdomain_dof_handler.reinit(subdomain_triangulation);
  subdomain_dof_handler.distribute_dofs(fe);

  std::cout << "    Number of DoFs on subdomain " << subdomain_id << ": "
            << subdomain_dof_handler.n_dofs() << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *>
    dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  subdomain_constraints.clear();

  DoFTools::make_hanging_node_constraints(subdomain_dof_handler,
                                          subdomain_constraints);

  VectorTools::interpolate_boundary_values(subdomain_dof_handler,
                                           dirichlet_boundary_functions,
                                           subdomain_constraints);
  subdomain_constraints.close();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  subdomain_matrix =
    std::make_unique<Portable::LaplaceOperator<dim, fe_degree, double>>(
      subdomain_dof_handler, subdomain_constraints, false);


  subdomain_matrix->initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  solution_host.reinit(subdomain_dof_handler.n_dofs());
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    subdomain_dof_handler.n_dofs());

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      // if (cell->is_locally_owned())
      //   {
      cell_rhs = 0;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) +=
            (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

      cell->get_dof_indices(local_dof_indices);
      subdomain_constraints.distribute_local_to_global(cell_rhs,
                                                       local_dof_indices,
                                                       system_rhs_host);
      // }
    }

  // system_rhs_host.compress(VectorOperation::add);
  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.n_dofs());

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  // std::cout << "    On subdomain " << subdomain_id
  //           << ": rhs.l2_norm() = " << system_rhs_host.l2_norm() <<
  //           std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve_subdomain()
{


  SolverControl solver_control(system_rhs_device.size(),
                               1e-12 * system_rhs_device.l2_norm());

  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  cg.solve(*subdomain_matrix,
           solution_device,
           system_rhs_device,
           PreconditionIdentity());

  std::cout  << "   On subdomain " << subdomain_id << " solver converged in " << solver_control.last_step()
        << " iterations." << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::post_process_solution()
{
  LinearAlgebra::ReadWriteVector<double> rw_vector(subdomain_dof_handler.n_dofs());
  rw_vector.import_elements(solution_device, VectorOperation::insert);
  solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_constraints.distribute(solution_host);

  // solution_host.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  (void)cycle;

  DataOut<dim> data_out;

  data_out.attach_dof_handler(subdomain_dof_handler);
  data_out.add_data_vector(solution_host, "solution");
  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
    "./", "solution_subdomain_"+std::to_string(subdomain_id), cycle, mpi_communicator);
    

  Vector<float> cellwise_norm(subdomain_triangulation.n_active_cells());
  VectorTools::integrate_difference(subdomain_dof_handler,
                                    solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  const double global_norm =
    VectorTools::compute_global_error(subdomain_triangulation,
                                      cellwise_norm,
                                      VectorTools::L2_norm);

  std::cout << "    On subdomain " << subdomain_id << "  solution norm: " << global_norm << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  setup_grid();
  for (unsigned int cycle = 0; cycle < 2; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      triangulation.refine_global(1);

      create_subdomain_triangulations();

      setup_dofs();

      setup_matrix_free();

      assemble_rhs();

      solve_subdomain();

      post_process_solution();

      output_results(cycle);
    }

  // GridOut       grid_out;
  // std::ofstream output(
  //   "subdomain_" +
  //   std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
  //   ".vtu");
  // grid_out.write_vtu(subdomain_triangulation, output);
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