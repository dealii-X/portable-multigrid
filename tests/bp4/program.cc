// BP4-style benchmark (vector Poisson, n_components == dim -- the vector
// counterpart of tests/bp_35's scalar BP3-style benchmark): same overall
// structure as bp_35/program.cc (doubling-mesh refinement cycles, one
// accumulating convergence_table re-printed after every cycle), but for
// Portable::VectorLaplaceOperator, and reporting *two* backends side by
// side in that same table each cycle:
//   - "bk4"          -- vmult_bk4(), the custom Kokkos kernel (bk4_kokkos_
//                       kernels.h), portable to any Kokkos backend.
//   - "tensor_core"  -- vmult_tensor_core(), the FP64 Tensor Core kernel
//                       (bk4_cuda_kernels.cuh), dim == 3 + double only,
//                       and only actually runs when this translation unit
//                       is compiled by nvcc against a CUDA-enabled Kokkos
//                       -- see tests/vector_laplace_tensor_core/ for the
//                       same graceful-degradation pattern. On a Serial/
//                       OpenMP-only Kokkos build (such as this project's
//                       own dev box), the tensor_core columns are simply
//                       left out of the table, not filled with
//                       placeholders.
//
// Both backends solve the exact same linear system with unpreconditioned
// CG, so cg_its/cg_reduction agreeing between them each cycle is itself a
// running correctness check, not just a timing comparison -- see tests/
// vector_laplace_bk4/ and tests/vector_laplace_tensor_core/ for the
// dedicated (non-benchmark) correctness checks this reuses.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "operators/portable_vector_laplace_operator.h"

namespace BP4
{
  using namespace dealii;

  // Same "precompile a degree range, pick one at runtime" scheme as
  // bp_35 -- the degree can't be a plain runtime variable because it's a
  // template parameter of LaplaceProblem/VectorLaplaceOperator.
  const unsigned int dimension      = 3; // bk4_cuda_kernels.cuh: dim == 3 only
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 8;

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();

    void
    run(const std::size_t min_size, const std::size_t max_size, const bool use_doubling_mesh);

  private:
    static constexpr int n_components = dim; // BP4: vector Poisson, n_components == dim

    using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
    using OperatorType =
      Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double, fe_degree + 2>;

    void
    setup_grid();

    void
    setup_dofs();

    void
    setup_matrix_free();

    void
    compute_rhs();

    // Runs CG + a separate matvec-only timing loop for one backend
    // ("bk4" or "tensor_core"), appending "cg_time_" + name / "cg_its_" +
    // name / "cg_reduction_" + name / "matvec_" + name columns to
    // convergence_table for this cycle. vmult must be one of
    // &OperatorType::vmult_bk4 / &OperatorType::vmult_tensor_core.
    void
    solve_and_time(const std::string &name,
                  void (OperatorType::*vmult)(VectorType &, const VectorType &) const);

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       scalar_fe;
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    VectorType solution_device;
    VectorType system_rhs_device;

    AffineConstraints<double> constraints;

    std::unique_ptr<OperatorType> system_matrix;

    const unsigned int refinement_cycles = 10;

    const bool overlap_communication_computation = false;

    double setup_time;

    ConvergenceTable convergence_table;

    ConditionalOStream pcout;
    ConditionalOStream time_details;
  };



  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , scalar_fe(fe_degree)
    , fe(scalar_fe, n_components)
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , time_details(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_dofs()
  {
    Timer time;

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(fe);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = "
          << n_components << " x (" << fe.degree << " + 1)^" << dim << std::endl;

    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    Functions::ZeroFunction<dim>                        homogeneous_dirichlet_bc(n_components);
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet_boundary_functions,
                                             constraints);
    constraints.close();

    setup_time += time.wall_time();

    time_details << "DoFs and constraint setup  (CPU/wall)" << time.cpu_time() << "s/"
                 << time.wall_time() << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_matrix_free()
  {
    Kokkos::fence();

    Timer time;

    system_matrix =
      std::make_unique<OperatorType>(dof_handler, constraints, overlap_communication_computation);
    system_matrix->initialize_dof_vector(solution_device);
    system_rhs_device.reinit(solution_device);
    Kokkos::fence();

    setup_time += time.wall_time();

    time_details << "Setup matrices   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::compute_rhs()
  {
    Timer time;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Constant unit load on every component -- same "1.0" RHS bp_35 uses,
    // just assembled component-wise for the vector-valued FESystem.
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_rhs = 0;

            fe_values.reinit(cell);

            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int c = 0; c < n_components; ++c)
                  cell_rhs(i) +=
                    fe_values.shape_value_component(i, q_index, c) * 1.0 * fe_values.JxW(q_index);

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs_host);
          }
      }

    system_rhs_host.compress(VectorOperation::add);
    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

    rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
    system_rhs_device.import_elements(rw_vector, VectorOperation::insert);

    setup_time += time.wall_time();

    time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << time.wall_time()
                 << 's' << std::endl;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve_and_time(
    const std::string &name,
    void (OperatorType::*vmult)(VectorType &, const VectorType &) const)
  {
    // helper: an operator that duck-types as SolverCG's MatrixType by
    // forwarding vmult() to whichever backend member function pointer was
    // passed in -- lets solve_and_time() reuse the exact same CG loop for
    // vmult_bk4() and vmult_tensor_core() instead of duplicating it.
    struct BackendOperator
    {
      const OperatorType &op;
      void (OperatorType::*vmult_impl)(VectorType &, const VectorType &) const;

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        (op.*vmult_impl)(dst, src);
      }
    };
    const BackendOperator backend_operator{*system_matrix, vmult};

    double                          time_cg = 1e10;
    std::pair<unsigned int, double> cg_details;

    for (unsigned int i = 0; i < 10; ++i)
      {
        Timer                     time;
        ReductionControl          solver_control(dof_handler.n_dofs(), 1e-16, 1e-9);
        SolverCG<VectorType>      solver_cg(solver_control);

        Kokkos::fence();
        time.restart();
        solution_device = 0;
        solver_cg.solve(backend_operator, solution_device, system_rhs_device, PreconditionIdentity());
        Kokkos::fence();

        if (time.wall_time() < time_cg)
          {
            time_cg           = time.wall_time();
            cg_details.first  = solver_control.last_step();
            cg_details.second = solver_control.last_value();
          }
      }

    pcout << "  [" << name << "] cg time " << time_cg << "  its " << cg_details.first
          << "  reduction " << cg_details.second << std::endl;

    double best_mv = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

        Kokkos::fence();
        Timer time;
        for (unsigned int j = 0; j < n_mv; ++j)
          backend_operator.vmult(solution_device, system_rhs_device);
        Kokkos::fence();

        const Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv = std::min(best_mv, stat.max);
      }

    pcout << "  [" << name << "] matvec time " << best_mv << std::endl;

    convergence_table.add_value("cg_time_" + name, time_cg);
    convergence_table.add_value("cg_its_" + name, cg_details.first);
    convergence_table.add_value("cg_reduction_" + name, cg_details.second);
    convergence_table.add_value("matvec_" + name, best_mv);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const std::size_t min_size,
                                      const std::size_t max_size,
                                      const bool        use_doubling_mesh)
  {
    (void)use_doubling_mesh; // only the doubling mesh is implemented below

    pcout << "Testing " << fe.get_name() << " (n_components = " << n_components << ")"
          << std::endl;

    for (unsigned int cycle = 0; cycle < refinement_cycles; ++cycle)
      {
        triangulation.clear();

        setup_time = 0.;

        pcout << "Cycle " << cycle << std::endl;

        // Same doubling-mesh sizing as bp_35::run() -- two-out-of-three
        // dimensions get subdivided each step, so cell count roughly
        // doubles per cycle instead of jumping by 8x on every dimension
        // at once.
        const unsigned int n_refine  = cycle / 3;
        const unsigned int remainder = cycle % 3;
        Point<dim>         p1;
        for (unsigned int d = 0; d < dim; ++d)
          p1[d] = -1;
        Point<dim> p2;
        for (unsigned int d = 0; d < remainder; ++d)
          p2[d] = 2.8;
        for (unsigned int d = remainder; d < dim; ++d)
          p2[d] = 0.9;
        std::vector<unsigned int> subdivisions(dim, 1);
        for (unsigned int d = 0; d < remainder; ++d)
          subdivisions[d] = 2;
        const unsigned int base_refine = (1u << n_refine);

        std::size_t projected_size = 1;
        for (unsigned int d = 0; d < dim; ++d)
          projected_size *= base_refine * subdivisions[d] * fe_degree + 1;

        GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

        if (projected_size < min_size)
          continue;

        if (projected_size > max_size)
          {
            pcout << "Projected size " << projected_size << " higher than max size, terminating."
                  << std::endl;
            pcout << std::endl;
            break;
          }

        triangulation.refine_global(n_refine);

        setup_dofs();

        setup_matrix_free();

        compute_rhs();

        pcout << "Total setup time: " << setup_time << std::endl;

        convergence_table.add_value("cells", triangulation.n_global_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());

        solve_and_time("bk4", &OperatorType::vmult_bk4);

#ifdef __CUDACC__
        // if constexpr (not just #ifdef): &OperatorType::vmult_tensor_core
        // below requires instantiating that member function to take its
        // address, which trips its static_assert(dim == 3, ...) for any
        // other dim -- dead-branch-discarding via if constexpr (dim ==
        // dimension is always 3 in this file, but the class template
        // itself is generic) keeps that from ever actually firing.
        if constexpr (dim == 3)
          solve_and_time("tensor_core", &OperatorType::vmult_tensor_core);
#else
        pcout << "  [tensor_core] skipped (not built with nvcc)" << std::endl;
#endif

        pcout << std::endl;

        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            for (const char *name : {"bk4"
#ifdef __CUDACC__
                                     ,
                                     "tensor_core"
#endif
                })
              {
                convergence_table.set_scientific(std::string("cg_time_") + name, true);
                convergence_table.set_precision(std::string("cg_time_") + name, 3);
                convergence_table.set_scientific(std::string("cg_reduction_") + name, true);
                convergence_table.set_precision(std::string("cg_reduction_") + name, 3);
                convergence_table.set_scientific(std::string("matvec_") + name, true);
                convergence_table.set_precision(std::string("matvec_") + name, 3);
              }

            convergence_table.write_text(std::cout);

            std::cout << std::endl << std::endl;
          }
      }
  }

  template <int dim, int min_degree, int max_degree>
  class LaplaceRunTime
  {
  public:
    LaplaceRunTime(const unsigned int target_degree,
                   const std::size_t  min_size,
                   const std::size_t  max_size,
                   const bool         use_doubling_mesh)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem;
          laplace_problem.run(min_size, max_size, use_doubling_mesh);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree, min_size, max_size, use_doubling_mesh);
    }
  };
} // namespace BP4

int
main(int argc, char *argv[])
{
  try
    {
      using namespace BP4;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree            = numbers::invalid_unsigned_int;
      std::size_t  maxsize           = static_cast<std::size_t>(-1);
      std::size_t  minsize           = 1;
      bool         use_doubling_mesh = true;

      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program degree minsize maxsize" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Settings of parameters: " << std::endl
                    << "Number of MPI ranks:            "
                    << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                    << "Polynomial degree:              " << degree << std::endl
                    << "Minimum size:                   " << minsize << std::endl
                    << "Maximum size:                   " << maxsize << std::endl
#ifdef __CUDACC__
                    << "Built with nvcc: tensor_core columns will run for real." << std::endl
#else
                    << "NOT built with nvcc: tensor_core columns will be skipped." << std::endl
#endif
                    << std::endl;
        }

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(degree,
                                                                    minsize,
                                                                    maxsize,
                                                                    use_doubling_mesh);
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
