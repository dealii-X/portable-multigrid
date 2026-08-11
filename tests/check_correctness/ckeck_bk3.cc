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
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_evaluation_kernels.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "operators/portable_laplace_operator_bk3.h"
#include "operators/portable_laplace_operator.h"




template <int dim, int fe_degree>
class LaplaceOperatorTest
{
public:
  LaplaceProblemLaplaceOperatorTest();

  void
  run_test();

  void
  compute_G_tensor();


private:
  void
  setup_grid();

  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  do_test();

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  Portable::LaplaceOperator<dim, fe_degree, double>    dealii_operator;
  Portable::LaplaceOperatorBK3<dim, fe_degree, double> bk3_operator;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  const bool overlap_communication_computation = false;

  Kokkos::View<number *, MemorySpace::Default::kokkos_space> G_tensor;
};



template <int dim, int fe_degree>
LaplaceOperatorTest<dim, fe_degree>::LaplaceOperatorTest()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)

{}

template <int dim, int fe_degree>
void
LaplaceOperatorTest<dim, fe_degree>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
}



template <int dim, int fe_degree>
void
LaplaceOperatorTest<dim, fe_degree>::setup_dofs()
{
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.reinit(dof_h.locally_owned_dofs(), locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_h, constraints);
  constraints.close();
}


template <int dim, int fe_degree>
void
LaplaceOperatorTest<dim, fe_degree>::setup_matrix_free()
{
  typename Portable::MatrixFree<dim, number>::AdditionalData additional_data;

  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;
  additional_data.overlap_communication_computation = false;

  const MappingQ<dim> mapping(fe_degree);

  const QGauss<1> quadrature_1d(fe_degree + 1);

  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature_1d, additional_data);

  compute_G_tensor();
}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorTest<dim, fe_degree, number>::compute_G_tensor()
{
  constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

  const auto        &precomputed_data = matrix_free.get_data(0);
  const unsigned int n_cells          = precomputed_data.n_cells;

  const auto &inv_jacobian = precomputed_data.inv_jacobian;
  const auto &JxW          = precomputed_data.JxW;

  G_tensor = Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
    Kokkos::view_alloc("initialize_G_tensor", Kokkos::WithoutInitializing),
    symmetric_tensor_dim * n_cells * n_q_points);

  Kokkos::parallel_for(
    "Fill_G_tensor",
    Kokkos::RangePolicy<
      dealii::MemorySpace::Default::kokkos_space::execution_space>(0, n_cells),
    KOKKOS_LAMBDA(const int cell_id) {
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          number components[symmetric_tensor_dim];

          int idx = 0;
          for (int d1 = 0; d1 < dim; ++d1)
            for (int d2 = d1; d2 < dim; ++d2)
              {
                number sum = 0;
                for (int k = 0; k < dim; ++k)
                  sum += inv_jacobian(q_point, cell_id, k, d1) *
                         inv_jacobian(q_point, cell_id, k, d2);
                components[idx] = JxW(q_point, cell_id) * sum;
                ++idx;
              }

          for (int c = 0; c < symmetric_tensor_dim; ++c)
            {
              G_tensor[cell_id * symmetric_tensor_dim * n_q_points +
                       c * n_q_points + q_point] = components[c];
            }
        }
    });
  Kokkos::fence();
}



template <int dim, int fe_degree>
void
LaplaceOperatorTest<dim, fe_degree>::do_test()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src,
    dst_dealii, dst_bk3;

  matrix_free.initialize_dof_vector(src);
  dst_dealii.reinit(src);
  dst_bk3.reinit(src);


  LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src_host(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  for (unsigned int i = 0; i < src_host.locally_owned_size(); ++i)
    src_host.local_element(i) = rand() / RAND_MAX;

  src_host.compress(VectorOperation::instert);
  src_host.update_ghost_values();

  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(src_host, VectorOperation::insert);

  src.import_elements(rw_vector, VectorOperation::insert);

  Portable::DeviceVector<double> src_device(src.get_values(), src.size());
  Portable::DeviceVector<double> dst_dealii_device(dst_dealii.get_values(),
                                                   dst_dealii.size());
  Portable::DeviceVector<double> dst_bk3_device(dst_bk3.get_values(),
                                                dst_bk3.size());

  const auto &precomputed_data = matrix_free.get_data(0);

  constexpr bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace,
                 Kokkos::DefaultHostExecutionSpace>::value;

  unsigned int numBlocks       = numbers::invalid_unsigned_int;
  unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
  if (is_serial)
    {
      numBlocks       = 1u;
      threadsPerBlock = 1u;
    }

  // BK3::Parallel::
  //   KokkosKernel_1D_Block<dim, fe_degree + 1, fe_degree + 1,
  //   number>(
  //     precomputed_data.shape_values,
  //     precomputed_data.co_shape_gradients,
  //     G_tensor,
  //     src_device,
  //     dst_device,
  //     dof_indices_per_color[color],
  //     n_cells,
  //     numBlocks,
  //     threadsPerBlock);

  BK3::Parallel::KokkosKernel<dim, fe_degree + 1, fe_degree + 1, number>(
    precomputed_data.shape_values,
    precomputed_data.co_shape_gradients,
    G_tensor,
    src_device,
    dst_bk3_device,
    precomputed_data.local_to_global,
    n_cells,
    numBlocks,
    threadsPerBlock);

  Kokkos::fence();

  Portabe::internal::FEEvaluationImplTransformToCollocation<dim,
                                                            fe_degree + 1,
                                                            fe_degree + 1,
                                                            double>::evaluate()
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run(const std::size_t  min_size,
                                    const std::size_t  max_size,
                                    const unsigned int n_pre_smooth,
                                    const unsigned int n_post_smooth,
                                    const bool         use_doubling_mesh)
{
  pcout << "Testing " << fe.get_name() << std::endl;
  const unsigned int sizes[] = {1,   2,   3,   4,    5,    6,   7,   8,
                                10,  12,  14,  16,   20,   24,  28,  32,
                                40,  48,  56,  64,   80,   96,  112, 128,
                                160, 192, 224, 256,  320,  384, 448, 512,
                                640, 768, 896, 1024, 1280, 1536};



  for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int);
       ++cycle)
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
          GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                    subdivisions,
                                                    p1,
                                                    p2);
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
          GridGenerator::subdivided_hyper_cube(triangulation,
                                               n_subdiv,
                                               -0.9,
                                               1.0);
          const unsigned int base_refine = (1 << n_refine);
          projected_size =
            Utilities::pow(base_refine * n_subdiv * fe_degree + 1, dim);
        }

      if (projected_size < min_size)
        continue;

      if (projected_size > max_size)
        {
          pcout << "Projected size " << projected_size
                << " higher than max size, terminating." << std::endl;
          pcout << std::endl;
          break;
        }


      triangulation.refine_global(n_refine);

      create_coarse_triangulations();

      setup_dofs();

      setup_matrix_free();

      setup_mg_transfers();

      setup_smoothers(n_pre_smooth, n_post_smooth);

      compute_rhs();

      pcout << "Total setup time: " << setup_time << std::endl;

      solve(n_pre_smooth, n_post_smooth);
      pcout << std::endl;

      pcout << std::endl;
      pcout << std::endl;
      matvec_ghost_timing();
      pcout << std::endl;
      pcout << std::endl;


      if (cycle >= 10)
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            convergence_table.set_scientific("mv_outer", true);
            convergence_table.set_precision("mv_outer", 3);
            convergence_table.set_scientific("mv_inner", true);
            convergence_table.set_precision("mv_inner", 3);
            convergence_table.set_scientific("cg_reduction", true);
            convergence_table.set_precision("cg_reduction", 3);
            convergence_table.set_scientific("cg_time", true);
            convergence_table.set_precision("cg_time", 3);

            convergence_table.write_text(std::cout);

            std::cout << std::endl << std::endl;

            ghost_timing_table.set_scientific("mv_ghost_and_compute", true);
            ghost_timing_table.set_precision("mv_ghost_and_compute", 4);
            ghost_timing_table.set_scientific("mv_compute_only", true);
            ghost_timing_table.set_precision("mv_compute_only", 4);
            ghost_timing_table.set_scientific("mv_ghost_only", true);
            ghost_timing_table.set_precision("mv_ghost_only", 4);

            ghost_timing_table.write_text(std::cout);

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
                 const unsigned int n_pre_smooth,
                 const unsigned int n_post_smooth,
                 const bool         use_doubling_mesh)
  {
    if (min_degree > max_degree)
      return;
    if (min_degree == target_degree)
      {
        LaplaceProblem<dim, min_degree> laplace_problem;
        laplace_problem.run(
          min_size, max_size, n_pre_smooth, n_post_smooth, use_doubling_mesh);
      }
    LaplaceRunTime<dim,
                   (min_degree <= max_degree ? (min_degree + 1) : min_degree),
                   max_degree>
      m(target_degree,
        min_size,
        max_size,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh);
  }
};

int
main(int argc, char *argv[])
{
  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree            = numbers::invalid_unsigned_int;
      std::size_t  maxsize           = static_cast<std::size_t>(-1);
      std::size_t  minsize           = 1;
      unsigned int n_pre_smooth      = 3;
      unsigned int n_post_smooth     = 3;
      bool         use_doubling_mesh = true;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "Expected at least one argument." << std::endl
              << "Usage:" << std::endl
              << "./program degree minsize maxsize n_pre_smooth n_post_smooth doubling"
              << std::endl
              << "The parameters degree to n_post_smooth are integers, "
              << "the last selects between a square mesh or a doubling mesh"
              << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);
      if (argc > 4)
        n_pre_smooth = std::atoi(argv[4]);
      if (argc > 5)
        n_post_smooth = std::atoi(argv[5]);
      if (argc > 6)
        use_doubling_mesh = argv[6][0] == 'd';

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                  << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Number of pre-smoother iters:   " << n_pre_smooth
                  << std::endl
                  << "Number of post-smoother iters:  " << n_post_smooth
                  << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh
                  << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(
        degree,
        minsize,
        maxsize,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh);
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