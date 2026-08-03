#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/evaluation_kernels_face.h>
#include <deal.II/matrix_free/evaluation_template_face_factory.templates.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "operators/portable_laplace_operator_dg.h"



using namespace dealii;

template <int dim, typename Number = double>
class LaplaceOperatorCPU
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;

  LaplaceOperatorCPU() = default;

  void
  reinit(const MatrixFree<dim, Number> &matrix_free)
  {
    this->matrix_free = &matrix_free;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    this->matrix_free->initialize_dof_vector(vec);
  }
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    // src.update_ghost_values();
    matrix_free->loop(&LaplaceOperatorCPU::cell_operation,
                      &LaplaceOperatorCPU::inner_face_operation,
                      &LaplaceOperatorCPU::boundary_face_operation,
                      this,
                      dst,
                      src,
                      true,
                      MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                      MatrixFree<dim, Number>::DataAccessOnFaces::gradients);

    // dst.compress(VectorOperation::add);
    // src.zero_out_ghost_values();

    for (unsigned int i : matrix_free->get_constrained_dofs())
      dst.local_element(i) = src.local_element(i);
  }

  void
  compute_diagonal(VectorType &diagonal) const
  {
    matrix_free->initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal<dim, -1, 0, 1, Number, VectorizedArray<Number>>(
      *matrix_free,
      diagonal,
      [](FEEvaluation<dim, -1, 0, 1, double> &phi)
        {
          phi.evaluate(EvaluationFlags::values);
          for (const unsigned int q : phi.quadrature_point_indices())
            phi.submit_value(phi.get_value(q), q);
          phi.integrate(EvaluationFlags::values);
        });
  }

private:
  ObserverPointer<const MatrixFree<dim, Number>> matrix_free;

  void
  cell_operation(const MatrixFree<dim, Number>               &matrix_free,
                 VectorType                                  &dst,
                 const VectorType                            &src,
                 const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, 1, Number> eval(matrix_free);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::gradients);
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            const auto grad = eval.get_gradient(q);
            eval.submit_gradient(make_vectorized_array<Number>(1.0) * grad, q);
          }
        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  void
  inner_face_operation(const MatrixFree<dim, Number>               &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, Number> phi_inner(data, true);
    FEFaceEvaluation<dim, -1, 0, 1, Number> phi_outer(data, false);

    const int actual_degree = data.get_dof_handler().get_fe().degree;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_inner.reinit(face);
        phi_inner.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_outer.reinit(face);
        phi_outer.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<Number> inverse_length_normal_to_face =
          0.5 * (std::abs((phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]) +
                 std::abs((phi_outer.normal_vector(0) * phi_outer.inverse_jacobian(0))[dim - 1]));
        const VectorizedArray<Number> sigma =
          inverse_length_normal_to_face * Number(actual_degree * (actual_degree + 1));

        for (const unsigned int q : phi_inner.quadrature_point_indices())
          {
            const VectorizedArray<Number> solution_jump =
              (phi_inner.get_value(q) - phi_outer.get_value(q));

            const VectorizedArray<Number> average_normal_derivative =
              (phi_inner.get_normal_derivative(q) + phi_outer.get_normal_derivative(q)) *
              Number(0.5);

            const VectorizedArray<Number> test_by_value =
              solution_jump * sigma - average_normal_derivative;

            phi_inner.submit_value(test_by_value, q);
            phi_outer.submit_value(-test_by_value, q);

            phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
            phi_outer.submit_normal_derivative(-solution_jump * Number(0.5), q);
          }

        phi_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);

        phi_outer.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }


  void
  boundary_face_operation(const MatrixFree<dim, Number>               &data,
                          VectorType                                  &dst,
                          const VectorType                            &src,
                          const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, 1, Number> phi_inner(data, true);

    const int actual_degree = data.get_dof_handler().get_fe().degree;


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_inner.reinit(face);
        phi_inner.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<Number> inverse_length_normal_to_face =
          std::abs((phi_inner.normal_vector(0) * phi_inner.inverse_jacobian(0))[dim - 1]);
        const VectorizedArray<Number> sigma =
          inverse_length_normal_to_face * Number(actual_degree * (actual_degree + 1));
        const bool is_dirichlet = (data.get_boundary_id(face) == 0);

        for (const unsigned int q : phi_inner.quadrature_point_indices())
          {
            const VectorizedArray<Number> u_inner = phi_inner.get_value(q);
            const VectorizedArray<Number> u_outer = is_dirichlet ? -u_inner : u_inner;
            const VectorizedArray<Number> normal_derivative_inner =
              phi_inner.get_normal_derivative(q);
            const VectorizedArray<Number> normal_derivative_outer =
              is_dirichlet ? normal_derivative_inner : -normal_derivative_inner;
            const VectorizedArray<Number> solution_jump = (u_inner - u_outer);
            const VectorizedArray<Number> average_normal_derivative =
              (normal_derivative_inner + normal_derivative_outer) * Number(0.5);
            const VectorizedArray<Number> test_by_value =
              solution_jump * sigma - average_normal_derivative;
            phi_inner.submit_normal_derivative(-solution_jump * Number(0.5), q);
            phi_inner.submit_value(test_by_value, q);
          }
        phi_inner.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }
};

template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(bool overlap_communication_computation = false);

  void
  run();

private:
  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  assemble_rhs();

  void
  solve();

  void
  test();

  void
  output_results(const unsigned int cycle) const;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_DGQ<dim> fe;

  MappingQ<dim> mapping;

  Portable::MatrixFree<dim, double> matrix_free;

  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::set<types::boundary_id> dirichlet_boundary_ids = {0};

  MatrixFree<dim, double>         matrix_free_cpu;
  LaplaceOperatorCPU<dim, double> system_matrix_cpu;

  Portable::LaplaceOperatorDG<dim, fe_degree, fe_degree + 1, double> system_matrix;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>      ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>   solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>      solution_cpu;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> system_rhs_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>    system_rhs_cpu;

  bool overlap_communication_computation;

  ConditionalOStream pcout;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(bool overlap_communication_computation)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , mapping(fe_degree)
  , dof_handler(triangulation)
  , overlap_communication_computation(overlap_communication_computation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  pcout << " Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  {
    QGauss<1>                                        quadrature_1d(fe_degree + 1);
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags  = update_values | update_gradients | update_JxW_values |
                                            update_quadrature_points | update_normal_vectors;

    additional_data.mapping_update_flags_inner_faces =
      update_values | update_gradients | update_JxW_values;
    additional_data.mapping_update_flags_boundary_faces =
      update_values | update_gradients | update_JxW_values | update_quadrature_points;

    matrix_free_cpu.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

    system_matrix_cpu.reinit(matrix_free_cpu);
  }
  system_matrix.reinit(mapping, dof_handler, constraints, overlap_communication_computation);


  system_matrix.initialize_dof_vector(solution_device);

  system_rhs_device.reinit(solution_device);
  ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);


  matrix_free_cpu.initialize_dof_vector(system_rhs_cpu);
  matrix_free_cpu.initialize_dof_vector(solution_cpu);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs_cpu);
        }
    }

  system_rhs_cpu.compress(VectorOperation::add);
  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(system_rhs_cpu, VectorOperation::insert);
  system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  pcout << "  RHS assembled" << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve()
{
  SolverControl solver_control(system_rhs_device.size(), 1e-12 * system_rhs_device.l2_norm());
  SolverControl solver_control_cpu(system_rhs_cpu.size(), 1e-12 * system_rhs_cpu.l2_norm());

  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(solver_control);
  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Host>>    cg_cpu(
    solver_control_cpu);

  solution_device = 0;
  solution_cpu    = 0;

  cg.solve(system_matrix, solution_device, system_rhs_device, PreconditionIdentity());

  pcout << "  Solver converged in     " << solver_control.last_step() << " iterations."
        << std::endl;

  cg_cpu.solve(system_matrix_cpu, solution_cpu, system_rhs_cpu, PreconditionIdentity());

  pcout << "  Solver CPU converged in " << solver_control.last_step() << " iterations." << std::endl
        << std::endl;

  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(solution_device, VectorOperation::insert);
  ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

  constraints.distribute(ghost_solution_host);

  ghost_solution_host.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  (void)cycle;
  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(ghost_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", cycle, mpi_communicator, 2);

  Vector<float> cellwise_norm(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    ghost_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);

  const double global_norm =
    VectorTools::compute_global_error(triangulation, cellwise_norm, VectorTools::L2_norm);

  pcout << "  solution norm: " << global_norm << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test()
{
  using VecTypeHost = LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;
  using VecType     = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  VecTypeHost dst_cpu, src_cpu;

  matrix_free_cpu.initialize_dof_vector(dst_cpu);
  matrix_free_cpu.initialize_dof_vector(src_cpu);

  for (unsigned int i = 0; i < src_cpu.locally_owned_size(); ++i)
    src_cpu.local_element(i) = (double)(i);

  src_cpu.compress(VectorOperation::add);
  src_cpu.update_ghost_values();

  system_matrix_cpu.vmult(dst_cpu, src_cpu);

  VecType dst, src;
  system_matrix.initialize_dof_vector(dst);
  system_matrix.initialize_dof_vector(src);

  LinearAlgebra::ReadWriteVector<double> rw(src.locally_owned_elements());
  rw.import_elements(src_cpu, VectorOperation::insert);
  src.import_elements(rw, VectorOperation::insert);

  system_matrix.vmult(dst, src);

  VecTypeHost err(dst_cpu.get_partitioner());

  rw.import_elements(dst, VectorOperation::insert);
  err.import_elements(rw, VectorOperation::insert);

  err -= dst_cpu;

  pcout << "dst_cpu.l2_norm = " << dst_cpu.l2_norm() << std::endl;
  pcout << "dst.l2_norm     = " << dst.l2_norm() << std::endl;
  pcout << "err.l2_norm     = " << err.l2_norm() << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  pcout << "============== dim = " << dim << ", " << " fe_degree = " << fe_degree
        << " ============== \n\n";

  for (unsigned int cycle = 0; cycle < 5; ++cycle)
    {
      pcout << std::endl << std::endl;
      pcout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation, 0., 1.);
          triangulation.refine_global(1);
        }
      else
        {
          triangulation.refine_global(1);
        }


      setup_dofs();
      setup_matrix_free();
      assemble_rhs();

      // test();


      solve();
      output_results(cycle);


      pcout << std::endl;
    }
}

template <int dim, int degree>
void
solve_for_degree(int fe_degree)
{
  if (degree == fe_degree)
    {
      LaplaceProblem<dim, degree> laplace_problem(false);
      laplace_problem.run();
    }
  else if constexpr (degree < 5)
    solve_for_degree<dim, degree + 1>(fe_degree);
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      {
        const int dim           = 2;
        const int max_fe_degree = 4;

        for (int fe_degree = 1; fe_degree <= max_fe_degree; ++fe_degree)
          {
            solve_for_degree<dim, 1>(fe_degree);
          }
      }
      {
        const int dim           = 3;
        const int max_fe_degree = 4;

        for (int fe_degree = 1; fe_degree <= max_fe_degree; ++fe_degree)
          {
            solve_for_degree<dim, 1>(fe_degree);
          }
      }
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
