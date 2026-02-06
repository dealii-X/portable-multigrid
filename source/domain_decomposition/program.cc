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

#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_v_cycle_multigrid.h"
#include "operators/portable_laplace_operator.h"

using namespace dealii;

template <int dim>
struct SubdomainTopologyInfo
{
  unsigned int            subdomain_id;
  Triangulation<dim>      triangulation;
  std::vector<Point<dim>> vertices;
  std::vector<bool>       interface_vertex_ids;
  std::vector<bool>       physical_boundary_vertex_ids;
  types::boundary_id      interface_id;

  void
  clear()
  {
    subdomain_id = numbers::invalid_unsigned_int;
    triangulation.clear();
    vertices.clear();
    interface_vertex_ids.clear();
    physical_boundary_vertex_ids.clear();
    interface_id = numbers::invalid_boundary_id;
  }
};

struct SubdomainDoFInfo
{
  IndexSet interface_dofs_global;

  std::vector<unsigned int>            local_interface_dofs;
  std::vector<types::global_dof_index> interface_local_to_global_map;

  std::vector<unsigned int> local_physical_boundary_dofs;

  void
  clear()
  {
    interface_dofs_global.clear();
    local_interface_dofs.clear();
    interface_local_to_global_map.clear();
    local_physical_boundary_dofs.clear();
  }
};

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
  create_dof_mapping();

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

  SubdomainTopologyInfo<dim> subdomain_topology;
  SubdomainDoFInfo           subdomain_dofs;

  FE_Q<dim> fe;

  DoFHandler<dim>           subdomain_dof_handler;
  AffineConstraints<double> subdomain_constraints;

  std::vector<types::global_dof_index> local_to_global_dof_map;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    global_solution_host;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    subdomain_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_rhs_device;


  std::unique_ptr<Portable::LaplaceOperatorBase<dim, double>> subdomain_matrix;

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
  this->subdomain_topology.clear();
  this->subdomain_topology.subdomain_id =
    Utilities::MPI::this_mpi_process(mpi_communicator);

  this->subdomain_topology.interface_id =
    100 + this->subdomain_topology.subdomain_id;

  std::vector<CellData<dim>> subdomain_cell_data;
  SubCellData                subcell_data;
  std::vector<bool>          is_physical_boundary;

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
                    this->subdomain_topology.vertices.size();
                  this->subdomain_topology.vertices.push_back(cell->vertex(v));
                }
              cell_data.vertices[v] =
                global_to_local_vertex_map[global_vertex_index];
            }

          cell_data.material_id = cell->material_id();
          cell_data.manifold_id = cell->manifold_id();
          subdomain_cell_data.push_back(cell_data);

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              bool on_physical_boundary = cell->at_boundary(f);
              bool on_interface         = false;

              if (!on_physical_boundary)
                {
                  if (cell->neighbor(f)->is_ghost())
                    on_interface = true;
                }
              if (on_physical_boundary || on_interface)
                {
                  CellData<dim - 1> face_data;
                  for (unsigned int fv = 0;
                       fv < GeometryInfo<dim>::vertices_per_face;
                       ++fv)
                    face_data.vertices[fv] =
                      global_to_local_vertex_map[cell->face(f)->vertex_index(
                        fv)];

                  face_data.boundary_id =
                    on_physical_boundary ?
                      cell->face(f)->boundary_id() :
                      this->subdomain_topology.interface_id;

                  face_data.manifold_id = cell->face(f)->manifold_id();

                  if constexpr (dim == 2)
                    subcell_data.boundary_lines.push_back(face_data);

                  if constexpr (dim == 3)
                    subcell_data.boundary_quads.push_back(face_data);

                  is_physical_boundary.push_back(true);
                }
            }
        }
    }

  Assert(subcell_data.check_consistency(dim),
         ExcMessage("Subcell data are not filled consistenly."));

  GridTools::consistently_order_cells<dim>(subdomain_cell_data);

  this->subdomain_topology.triangulation.create_triangulation(
    this->subdomain_topology.vertices, subdomain_cell_data, subcell_data);

  this->subdomain_topology.physical_boundary_vertex_ids.resize(
    this->subdomain_topology.triangulation.n_vertices(), false);

  for (auto &cell :
       this->subdomain_topology.triangulation.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->at_boundary(f))
            continue;

          const auto bid = cell->face(f)->boundary_id();

          if (bid != this->subdomain_topology.interface_id)
            {
              for (unsigned int fv = 0;
                   fv < GeometryInfo<dim>::vertices_per_face;
                   ++fv)
                {
                  const unsigned int vertex_idx =
                    cell->face(f)->vertex_index(fv);
                  this->subdomain_topology
                    .physical_boundary_vertex_ids[vertex_idx] = true;
                }
            }
        }
    }

  this->subdomain_topology.interface_vertex_ids.resize(
    subdomain_topology.triangulation.n_vertices(), false);

  for (auto &cell :
       this->subdomain_topology.triangulation.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (!cell->at_boundary(f))
            continue;

          const auto bid = cell->face(f)->boundary_id();

          if (bid == this->subdomain_topology.interface_id)
            {
              for (unsigned int fv = 0;
                   fv < GeometryInfo<dim>::vertices_per_face;
                   ++fv)
                {
                  const unsigned int vertex_idx =
                    cell->face(f)->vertex_index(fv);

                  if (!this->subdomain_topology
                         .physical_boundary_vertex_ids[vertex_idx])
                    this->subdomain_topology.interface_vertex_ids[vertex_idx] =
                      true;
                }
            }
        }
    }

  AssertDimension(this->subdomain_topology.triangulation.n_active_cells(),
                  triangulation.n_locally_owned_active_cells());
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  pcout << "  Total number of DoFs: " << dof_handler.n_dofs() << std::endl;

  subdomain_dof_handler.reinit(subdomain_topology.triangulation);
  subdomain_dof_handler.distribute_dofs(fe);

  std::cout << "    Number of DoFs on subdomain "
            << this->subdomain_topology.subdomain_id << ": "
            << subdomain_dof_handler.n_dofs() << std::endl;


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

  global_solution_host.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);

  subdomain_solution_host.reinit(subdomain_dof_handler.n_dofs());
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_dof_mapping()
{
  local_to_global_dof_map.resize(subdomain_dof_handler.n_dofs());
  {
    auto global_cell     = dof_handler.begin_active();
    auto global_cell_end = dof_handler.end();

    auto local_cell = subdomain_dof_handler.begin_active();

    std::vector<types::global_dof_index> global_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (; global_cell != global_cell_end; ++global_cell)
      {
        if (global_cell->is_locally_owned())
          {
            global_cell->get_dof_indices(global_dof_indices);
            local_cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              {
                local_to_global_dof_map[local_dof_indices[i]] =
                  global_dof_indices[i];
              }

            ++local_cell;
          }
      }
  }

  subdomain_dofs.clear();
  subdomain_dofs.interface_dofs_global.set_size(dof_handler.n_dofs());

  IndexSet local_physical_boundary_dofs(subdomain_dof_handler.n_dofs());
  IndexSet local_interface_dofs(subdomain_dof_handler.n_dofs());

  const unsigned int n_dofs_per_cell = fe.dofs_per_cell;

  std::vector<types::global_dof_index> cell_dofs(n_dofs_per_cell);

  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      if (!cell->at_boundary())
        continue;

      cell->get_dof_indices(cell_dofs);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (cell->at_boundary(f) &&
              cell->face(f)->boundary_id() != subdomain_topology.interface_id)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                {
                  if (fe.has_support_on_face(i, f))
                    local_physical_boundary_dofs.add_index(cell_dofs[i]);
                }
            }
        }
    }

  for (const auto &cell : subdomain_dof_handler.active_cell_iterators())
    {
      if (!cell->at_boundary())
        continue;

      cell->get_dof_indices(cell_dofs);

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if (cell->at_boundary(f) &&
              cell->face(f)->boundary_id() == subdomain_topology.interface_id)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                {
                  if (fe.has_support_on_face(i, f))
                    local_interface_dofs.add_index(cell_dofs[i]);
                }
            }
        }
    }

  local_interface_dofs.subtract_set(local_physical_boundary_dofs);

  for (auto index : local_physical_boundary_dofs)
    {
      subdomain_dofs.local_physical_boundary_dofs.push_back(index);
    }

  for (auto index : local_interface_dofs)
    {
      const types::global_dof_index global_index =
        local_to_global_dof_map[index];

      subdomain_dofs.local_interface_dofs.push_back(index);
      subdomain_dofs.interface_local_to_global_map.push_back(global_index);
      subdomain_dofs.interface_dofs_global.add_index(global_index);
    }

  std::cout << "On subdomain " << this->subdomain_topology.subdomain_id
            << " interface dofs: " << std::endl;
  for (unsigned int i = 0; i < subdomain_dofs.local_interface_dofs.size(); ++i)
    {
      std::cout << subdomain_dofs.local_interface_dofs[i] << ", ";
    }
  std::cout << std::endl;

  std::cout << "On subdomain " << this->subdomain_topology.subdomain_id
            << " physical boundary dofs: " << std::endl;
  for (unsigned int i = 0;
       i < subdomain_dofs.local_physical_boundary_dofs.size();
       ++i)
    {
      std::cout << subdomain_dofs.local_physical_boundary_dofs[i] << ", ";
    }
  std::cout << std::endl;

  std::cout << "On subdomain " << this->subdomain_topology.subdomain_id
            << " local_to_global: " << std::endl;
  for (unsigned int i = 0;
       i < subdomain_dofs.interface_local_to_global_map.size();
       ++i)
    {
      std::cout << subdomain_dofs.interface_local_to_global_map[i] << ", ";
    }
  std::cout << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  subdomain_matrix =
    std::make_unique<Portable::LaplaceOperator<dim, fe_degree, double>>(
      subdomain_dof_handler, subdomain_constraints, false);


  subdomain_matrix->initialize_dof_vector(subdomain_solution_device);
  subdomain_rhs_device.reinit(subdomain_solution_device);
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
    }

  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.n_dofs());

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  subdomain_rhs_device.import_elements(rw_vector, VectorOperation::insert);
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

  std::cout << "   On subdomain " << this->subdomain_topology.subdomain_id
            << " solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::post_process_subdomain_solution()
{
  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_constraints.distribute(subdomain_solution_host);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  (void)cycle;

  DataOut<dim> data_out;

  data_out.attach_dof_handler(subdomain_dof_handler);
  data_out.add_data_vector(subdomain_solution_host, "solution");
  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
    "./",
    "solution_subdomain_" +
      std::to_string(this->subdomain_topology.subdomain_id),
    cycle,
    mpi_communicator);


  Vector<float> cellwise_norm(
    this->subdomain_topology.triangulation.n_active_cells());
  VectorTools::integrate_difference(subdomain_dof_handler,
                                    subdomain_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  const double global_norm =
    VectorTools::compute_global_error(subdomain_topology.triangulation,
                                      cellwise_norm,
                                      VectorTools::L2_norm);

  std::cout << "    On subdomain " << this->subdomain_topology.subdomain_id
            << "  solution norm: " << global_norm << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  setup_grid();
  for (unsigned int cycle = 0; cycle < 1; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      triangulation.refine_global(1);

      create_subdomain_triangulations();

      setup_dofs();

      create_dof_mapping();

      setup_matrix_free();

      assemble_rhs();

      solve_subdomain();

      post_process_subdomain_solution();

      output_results(cycle);
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      constexpr int dim       = 3;
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