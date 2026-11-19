#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/grid/grid_generator.h>
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

#include "operators/portable_momentum_operator_rt.h"

using namespace dealii;

// Here at the top of the file, we collect the main global settings. The
// degree can be passed as the first argument to the program, but due to the
// templates we need to precompile the respective programs. Here we specify
// a minimum and maximum degree we want to support. Degrees outside this
// range will not do any work.
const unsigned int dimension      = 3;
const unsigned int minimal_degree = 1;
const unsigned int maximal_degree = 8;
const double       wave_number    = 3.;
const bool         deform_grid    = false;

// We also select a mixed-precision approach as default. You can
// independently change the number type for the outer iteration via
// full_number and the number type for the multigrid v-cycle.
using vcycle_number = double;
using full_number   = double;



template <int dim, int fe_degree>
class RaviartThomasOperator
{
public:
  RaviartThomasOperator();

  void
  run(const std::size_t min_size, const std::size_t max_size, const bool use_doubling_mesh);

  using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

private:
  void
  setup_dofs();

  void
  setup_matrix_free();


  void
  test_cell_operator();


  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;


  FE_RaviartThomasNodal<dim> fe;
  DoFHandler<dim>            dof_handler;
  AffineConstraints<double>  constraints_double;
  AffineConstraints<float>   constraints_float;
  MappingQ<dim>              mapping;



  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::set<types::boundary_id> dirichlet_boundary_ids;


  Portable::RT::RaviartThomasOperatorBase<dim, double> rt_operator_double;
  Portable::RT::RaviartThomasOperatorBase<dim, float>  rt_operator_float;


  double setup_time;

  ConvergenceTable convergence_table;

  // ConvergenceTable ghost_timing_table;

  ConditionalOStream pcout;

  ConditionalOStream time_details;


  // struct LaplaceOperatorRunner
  // {
  //   const unsigned int              level;
  //   DoFHandler<dim>                &dof_handler;
  //   AffineConstraints<double>      &constraints;
  //   bool                            overlap_communication_computation;
  //   LaplaceProblem<dim, fe_degree> &parent_problem;

  //   template <unsigned int degree>
  //   void
  //   run()
  //   {
  //     parent_problem.level_matrices[level] =
  //       std::make_unique<Portable::LaplaceOperatorBK3<dim, degree, double>>(
  //         dof_handler, constraints, overlap_communication_computation);
  //   }
  // };


  // struct PolynomialTransferRunner
  // {
  //   const unsigned int                       level;
  //   const Portable::MatrixFree<dim, double> &mf_coarse;
  //   const Portable::MatrixFree<dim, double> &mf_fine;
  //   AffineConstraints<double>               &constraints_coarse;
  //   AffineConstraints<double>               &constraints_fine;

  //   LaplaceProblem<dim, fe_degree> &parent_problem;

  //   template <unsigned int degree_coarse, unsigned int degree_fine>
  //   void
  //   run()
  //   {
  //     parent_problem.mg_transfers[level] =
  //       std::make_unique<Portable::PolynomialTransfer<dim, degree_coarse, degree_fine,
  //       double>>();

  //     parent_problem.mg_transfers[level]->reinit(mf_coarse,
  //                                                mf_fine,
  //                                                constraints_coarse,
  //                                                constraints_fine);
  //   }
  // };
};

template <int dim, int fe_degree>
RaviartThomasOperator<dim, fe_degree>::RaviartThomasOperator()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , mapping(fe_degree)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout, true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)

{
  // dirichlet_boundary_ids.insert(0);
}



template <int dim, int fe_degree>
void
RaviartThomasOperator<dim, fe_degree>::setup_dofs()
{
  Timer time;

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  // pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
  //       << ((int)std::pow(dof_handler.n_dofs() * 1.0000001, 1. / dim) - 1) / fe.degree << " x "
  //       << fe.degree << " + 1)^" << dim << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);


  // Functions::ZeroFunction<dim>                        homogeneous_dirichlet_bc;
  // std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
  //   {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  // std::vector<unsigned int> p_levels({fe.degree});

  // while (p_levels.back() > 1)
  //   p_levels.push_back(std::max(p_levels.back() - 1, 1u));

  // p_level_fes.resize(0, p_levels.size() - 1);

  // for (unsigned int level = 0; level < p_levels.size(); ++level)
  //   p_level_fes[level] = std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

  // level_dof_handlers.resize(0, coarse_triangulations.size() - 1 + p_level_fes.max_level());
  // level_constraints.resize(0, level_dof_handlers.max_level());

  // for (unsigned int level = level_dof_handlers.min_level();
  //      level <= level_dof_handlers.max_level();
  //      ++level)
  //   {
  //     DoFHandler<dim> &dof_h = level_dof_handlers[level];

  //     dof_h.reinit(*coarse_triangulations[std::min(level, triangulation.n_global_levels() -
  //     1)]);

  //     if (level < coarse_triangulations.size())
  //       dof_h.distribute_dofs(*p_level_fes[0]);
  //     else
  //       dof_h.distribute_dofs(*p_level_fes[level + 1 - coarse_triangulations.size()]);

  //     IndexSet level_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_h);

  //     AffineConstraints<double> &constraints = level_constraints[level];

  //     constraints.reinit(dof_h.locally_owned_dofs(), level_relevant_dofs);

  //     DoFTools::make_hanging_node_constraints(dof_h, constraints);

  //     VectorTools::interpolate_boundary_values(dof_h, dirichlet_boundary_functions,
  //     constraints); constraints.close();
  //   }

  // setup_time += time.wall_time();

  // time_details << "DoFs and constraint setup  (CPU/wall)" << time.cpu_time() << "s/"
  //              << time.wall_time() << 's' << std::endl;

  constraints_double.clear();
  constraints_double.reinit(locally_owned_dofs, locally_relevant_dofs);
  // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary_functions,
  // constraints);
  constraints_double.close();

  constraints_float.copy_from(constraints_double);

  pcout << " Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
}


template <int dim, int fe_degree>
void
RaviartThomasOperator<dim, fe_degree>::setup_matrix_free()
{
  QGauss<1> quadrature_1d(fe_degree + 2);

  rt_operator_double.reinit(mapping, dof_handler, constraints_double, quadrature_1d);
  rt_operator_float.reinit(mapping, dof_handler, constraints_float, quadrature_1d);
}


template <int dim, int fe_degree>
void
RaviartThomasOperator<dim, fe_degree>::test_cell_operator()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dst_double, src_double;
  LinearAlgebra::distributed::Vector<float, MemorySpace::Default>  dst_float, src_float;

  rt_operator_double.initialize_dof_vector(dst_double);
  rt_operator_double.initialize_dof_vector(src_double);

  rt_operator_float.initialize_dof_vector(dst_float);
  rt_operator_float.initialize_dof_vector(src_float);


  src_double = double(1);
  src_float  = float(1);

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

  pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index << "] " << memory.avg
        << " " << memory.max << " [p" << memory.max_index << "]" << std::endl;

  Timer time;

  double best_mv_double = 1e10;
  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

      Kokkos::fence();
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        rt_operator_double.test_cell_operator(dst_double, src_double);
      Kokkos::fence();

      Utilities::MPI::MinMaxAvg stat =
        Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

      best_mv_double = std::min(best_mv_double, stat.max);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "matvec time double " << stat.min << " [p" << stat.min_index << "] "
                  << stat.avg << " " << stat.max << " [p" << stat.max_index << "]"
                  << " DoFs/s: " << dof_handler.n_dofs() / stat.max << std::endl;
    }

  pcout << std::endl << std::endl;
  double best_mv_float = 1e10;
  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

      Kokkos::fence();
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        rt_operator_float.test_cell_operator(dst_float, src_float);
      Kokkos::fence();

      Utilities::MPI::MinMaxAvg stat =
        Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

      best_mv_float = std::min(best_mv_float, stat.max);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "matvec time float " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                  << " " << stat.max << " [p" << stat.max_index << "]"
                  << " DoFs/s: " << dof_handler.n_dofs() / stat.max << std::endl;
    }
  convergence_table.add_value("cells", triangulation.n_global_active_cells());
  convergence_table.add_value("dofs", dof_handler.n_dofs());
  convergence_table.add_value("mv_double", best_mv_double);
  convergence_table.add_value("mv_float", best_mv_float);
}


template <int dim, int fe_degree>
void
RaviartThomasOperator<dim, fe_degree>::run(const std::size_t min_size,
                                           const std::size_t max_size,
                                           const bool        use_doubling_mesh)
{
  pcout << "Testing " << fe.get_name() << std::endl;
  const unsigned int sizes[] = {1,   2,   3,   4,   5,   6,   7,   8,   10,  12,   14,   16,  20,
                                24,  28,  32,  40,  48,  56,  64,  80,  96,  112,  128,  160, 192,
                                224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536};



  for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int); ++cycle)
    {
      triangulation.clear();

      setup_time = 0.;

      pcout << "Cycle " << cycle << std::endl;

      std::size_t  projected_size = numbers::invalid_size_type;
      unsigned int n_refine       = 0;

      if (use_doubling_mesh)
        {
          n_refine                     = cycle / 3;
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
          const unsigned int base_refine = (1 << n_refine);
          projected_size                 = 1;
          for (unsigned int d = 0; d < dim; ++d)
            projected_size *= base_refine * subdivisions[d] * fe_degree + 1;
          GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
        }
      else
        {
          n_refine              = 0;
          unsigned int n_subdiv = sizes[cycle];
          if (n_subdiv > 1)
            while (n_subdiv % 2 == 0)
              {
                n_refine += 1;
                n_subdiv /= 2;
              }
          if (dim == 2)
            n_refine += 3;
          GridGenerator::subdivided_hyper_cube(triangulation, n_subdiv, -0.9, 1.0);
          const unsigned int base_refine = (1 << n_refine);
          projected_size = Utilities::pow(base_refine * n_subdiv * fe_degree + 1, dim);
        }

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


      pcout << "Total setup time: " << setup_time << std::endl;

      pcout << std::endl;
      test_cell_operator();
      pcout << std::endl;

      // pcout << std::endl;
      // pcout << std::endl;
      // matvec_ghost_timing();
      // pcout << std::endl;
      // pcout << std::endl;


      // if (cycle >= 0)
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          convergence_table.set_scientific("mv_double", true);
          convergence_table.set_precision("mv_double", 3);
          convergence_table.set_scientific("mv_float", true);
          convergence_table.set_precision("mv_float", 3);
          // convergence_table.set_scientific("cg_reduction", true);
          // convergence_table.set_precision("cg_reduction", 3);
          // convergence_table.set_scientific("cg_time", true);
          // convergence_table.set_precision("cg_time", 3);

          convergence_table.write_text(std::cout);

          // std::cout << std::endl << std::endl;

          // ghost_timing_table.set_scientific("mv_ghost_and_compute", true);
          // ghost_timing_table.set_precision("mv_ghost_and_compute", 4);
          // ghost_timing_table.set_scientific("mv_compute_only", true);
          // ghost_timing_table.set_precision("mv_compute_only", 4);
          // ghost_timing_table.set_scientific("mv_ghost_only", true);
          // ghost_timing_table.set_precision("mv_ghost_only", 4);

          // ghost_timing_table.write_text(std::cout);

          // std::cout << std::endl << std::endl;
        }
    }
}
template <int dim, int min_degree, int max_degree>
class RTOperatorRunTime
{
public:
  RTOperatorRunTime(const unsigned int target_degree,
                    const std::size_t  min_size,
                    const std::size_t  max_size,
                    const bool         use_doubling_mesh)
  {
    if (min_degree > max_degree)
      return;
    if (min_degree == target_degree)
      {
        RaviartThomasOperator<dim, min_degree> laplace_problem;
        laplace_problem.run(min_size, max_size, use_doubling_mesh);
      }
    RTOperatorRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree>
      m(target_degree, min_size, max_size, use_doubling_mesh);
  }
};

int
main(int argc, char *argv[])
{
  try
    {
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
                      << "./program degree minsize maxsize doubling" << std::endl
                      << "The parameters degree to n_post_smooth are integers, "
                      << "the last selects between a square mesh or a doubling mesh" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);
      if (argc > 4)
        use_doubling_mesh = argv[4][0] == 'd';

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh << std::endl
                  << std::endl;

      RTOperatorRunTime<dimension, minimal_degree, maximal_degree> run(degree,
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