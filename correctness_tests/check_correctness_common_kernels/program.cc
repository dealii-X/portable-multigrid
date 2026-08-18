// Correctness check for BK3Custom::Parallel::KokkosKernel
// (include/kernels/bk3_kokkos_kernels_custom.h, composed from the generic
// building blocks in include/kernels/portable_tensor_product_kernels.h)
// against the original BK3::Parallel::KokkosKernel
// (include/kernels/bk3_kokkos_kernel.h).
//
// Builds a small Portable::MatrixFree Laplace setup, mirroring
// Portable::LaplaceOperatorBK3's constructor/setup_dof_indices_per_color()/
// compute_G_tensors() (include/operators/portable_laplace_operator_bk3.h,
// duplicated here rather than modifying that header), applies both kernels
// to the same random right-hand side, and compares the results -- for
// several (dim, fe_degree) combinations, to exercise both the dim == 2 and
// dim == 3 branches in the new kernel.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>
#include <random>

#include "kernels/bk3_kokkos_kernel.h"
#include "kernels/bk3_kokkos_kernels_custom.h"

using namespace dealii;

using Number = double;

// n_quad_1d defaults to fe_degree + 1 (matching every real call site in
// this codebase, where nm == nq), but can be set higher to exercise the
// rectangular (n_rows != n_columns) path in apply_matrix_vector_product --
// BK3's own "GL vs GLL nodes" comments describe exactly this
// over-integrated case, even though it isn't used anywhere yet.
template <int dim, int fe_degree, int n_quad_1d = fe_degree + 1>
bool
run_test()
{
  constexpr int n_1d         = fe_degree + 1;
  constexpr int n_local_dofs = Utilities::pow(n_1d, dim);

  static_assert(n_quad_1d >= n_1d, "under-integration is not supported");

  const MappingQ<dim> mapping(fe_degree);
  const FE_Q<dim>     fe(fe_degree);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  constraints.close();

  typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_gradients | update_JxW_values | update_quadrature_points;

  const QGauss<1> quadrature_1d(n_quad_1d);

  Portable::MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

  const auto        &colored_graph = matrix_free.get_colored_graph();
  const unsigned int n_colors      = colored_graph.size();

  // -- setup_dof_indices_per_color(), following
  //    Portable::LaplaceOperatorBK3::setup_dof_indices_per_color() --
  std::vector<unsigned int> lex_numbering(n_local_dofs);
  {
    const Quadrature<1>                                       dummy_quadrature(
      std::vector<Point<1>>(1, Point<1>()));
    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
    lex_numbering = shape_info.lexicographic_numbering;
  }

  std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
    dof_indices_per_color(n_colors);

  {
    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
    std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() == 0)
          continue;

        const auto &mf_data = matrix_free.get_data(color);
        const auto &graph   = colored_graph[color];

        dof_indices_per_color[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices_" + std::to_string(color), Kokkos::WithoutInitializing),
            n_local_dofs,
            mf_data.n_cells);

        auto dof_indices_host = Kokkos::create_mirror_view(dof_indices_per_color[color]);

        for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
          {
            auto triacell = graph[cell_id];

            typename DoFHandler<dim>::cell_iterator cell =
              triacell->as_dof_handler_iterator(dof_handler);

            cell->get_dof_indices(local_dof_indices);
            triacell->get_dof_indices(subdomain_local_dof_indices);

            if (partitioner)
              for (auto &index : local_dof_indices)
                index = partitioner->global_to_local(index);

            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                const auto global_dof          = local_dof_indices[lex_numbering[i]];
                const auto subdomain_local_dof = subdomain_local_dof_indices[lex_numbering[i]];

                if (constraints.is_constrained(subdomain_local_dof))
                  dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                else
                  dof_indices_host(i, cell_id) = global_dof;
              }
          }

        Kokkos::deep_copy(dof_indices_per_color[color], dof_indices_host);
        Kokkos::fence();
      }
  }

  // -- compute_G_tensors(), following
  //    Portable::LaplaceOperatorBK3::compute_G_tensors() --
  constexpr int          symmetric_tensor_dim = (dim * (dim + 1)) / 2;
  constexpr unsigned int n_q_points           = Utilities::pow(n_quad_1d, dim);

  std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> G_tensors(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color)
    {
      if (colored_graph[color].size() == 0)
        continue;

      const auto        &precomputed_data = matrix_free.get_data(color);
      const unsigned int n_cells          = precomputed_data.n_cells;

      const auto &inv_jacobian = precomputed_data.inv_jacobian;
      const auto &JxW          = precomputed_data.JxW;

      G_tensors[color] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("G_tensor_color_" + std::to_string(color), Kokkos::WithoutInitializing),
        symmetric_tensor_dim * n_cells * n_q_points);

      auto G = G_tensors[color];

      Kokkos::parallel_for(
        "Fill_G_tensor_color" + std::to_string(color),
        Kokkos::RangePolicy<dealii::MemorySpace::Default::kokkos_space::execution_space>(0, n_cells),
        KOKKOS_LAMBDA(const int cell_id) {
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              Number components[symmetric_tensor_dim];

              int idx = 0;
              for (int d1 = 0; d1 < dim; ++d1)
                for (int d2 = d1; d2 < dim; ++d2)
                  {
                    Number sum = 0;
                    for (int k = 0; k < dim; ++k)
                      sum += inv_jacobian(q_point, cell_id, d1, k) *
                             inv_jacobian(q_point, cell_id, d2, k);
                    components[idx] = JxW(q_point, cell_id) * sum;
                    ++idx;
                  }

              for (int c = 0; c < symmetric_tensor_dim; ++c)
                G[cell_id * symmetric_tensor_dim * n_q_points + c * n_q_points + q_point] =
                  components[c];
            }
        });
      Kokkos::fence();
    }

  // -- random input, apply both kernels, compare --
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src, dst_orig, dst_common;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_orig);
  matrix_free.initialize_dof_vector(dst_common);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw(src.locally_owned_elements());
    for (const auto idx : src.locally_owned_elements())
      rw(idx) = dist(gen);
    src.import_elements(rw, VectorOperation::insert);
  }

  dst_orig   = 0.;
  dst_common = 0.;

  Portable::DeviceVector<Number> src_device(src.get_values(), src.locally_owned_size());
  Portable::DeviceVector<Number> dst_orig_device(dst_orig.get_values(), dst_orig.locally_owned_size());
  Portable::DeviceVector<Number> dst_common_device(dst_common.get_values(),
                                                   dst_common.locally_owned_size());

  // Force team_size == 1 on a serial Kokkos host backend, matching
  // Portable::LaplaceOperatorBK3::vmult()'s own handling of this case -- the
  // kernels' default launch-heuristic team sizes assume a GPU backend and
  // are too large for Kokkos::Serial.
  constexpr bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

  unsigned int numBlocks       = numbers::invalid_unsigned_int;
  unsigned int threadsPerBlock = numbers::invalid_unsigned_int;
  if (is_serial)
    {
      numBlocks       = 1u;
      threadsPerBlock = 1u;
    }

  for (unsigned int color = 0; color < n_colors; ++color)
    {
      const unsigned int n_cells = colored_graph[color].size();
      if (n_cells == 0)
        continue;

      const auto &precomputed_data = matrix_free.get_data(color);

      BK3::Parallel::KokkosKernel<dim, n_1d, n_quad_1d, Number>(precomputed_data.shape_values,
                                                                precomputed_data.co_shape_gradients,
                                                                G_tensors[color],
                                                                src_device,
                                                                dst_orig_device,
                                                                dof_indices_per_color[color],
                                                                n_cells,
                                                                numBlocks,
                                                                threadsPerBlock);

      BK3Custom::Parallel::KokkosKernel<dim, n_1d, n_quad_1d, Number>(precomputed_data.shape_values,
                                                                      precomputed_data.co_shape_gradients,
                                                                      G_tensors[color],
                                                                      src_device,
                                                                      dst_common_device,
                                                                      dof_indices_per_color[color],
                                                                      n_cells,
                                                                      numBlocks,
                                                                      threadsPerBlock);

      Kokkos::fence();
    }

  LinearAlgebra::ReadWriteVector<Number> rw_orig(dst_orig.locally_owned_elements());
  LinearAlgebra::ReadWriteVector<Number> rw_common(dst_common.locally_owned_elements());
  rw_orig.import_elements(dst_orig, VectorOperation::insert);
  rw_common.import_elements(dst_common, VectorOperation::insert);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (const auto idx : dst_orig.locally_owned_elements())
    {
      max_abs_diff = std::max(max_abs_diff, std::abs(rw_orig(idx) - rw_common(idx)));
      max_abs_val  = std::max(max_abs_val, std::abs(rw_orig(idx)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "dim = " << dim << ", fe_degree = " << fe_degree << ", n_quad_1d = " << n_quad_1d
            << ", n_dofs = " << dof_handler.n_dofs() << std::endl;
  std::cout << "  max |orig|          = " << max_abs_val << std::endl;
  std::cout << "  max |orig - common| = " << max_abs_diff << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}

// Custom::Parallel::evaluate_gradients() (kernels/portable_evaluation_kernels.h)
// has no caller anywhere yet (added "for completeness", per the
// unify-kernels session), so nothing
// above exercises it -- being a template, it isn't even compiled until
// something instantiates it. Cross-check it directly against
// evaluate_gradients_and_multiply_symmetric_tensor() using an identity d_G (one
// "cell", no batching/MatrixFree setup needed): with G == identity, the
// tensor multiply is a no-op, so the dim raw directional derivatives
// evaluate_gradients() writes should exactly equal what
// evaluate_gradients_and_multiply_symmetric_tensor() writes -- which is itself
// already validated end-to-end by run_test() above.
template <int dim, int nq>
bool
run_evaluate_test()
{
  constexpr int nq_total                   = Utilities::pow(nq, dim);
  constexpr int symmetric_tensor_dimension = (dim * (dim + 1)) / 2;

  // scratch layout: [0) values, [1*nq_total, (1+dim)*nq_total)
  // evaluate_gradients()'s output slots, [(1+dim)*nq_total,
  // (1+2*dim)*nq_total) evaluate_gradients_and_multiply_symmetric_tensor()'s output
  // slots.
  const int values_offset  = 0;
  const int grad_a_offset  = nq_total;
  const int grad_b_offset  = (1 + dim) * nq_total;
  const int total_scratch  = (1 + 2 * dim) * nq_total;

  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> co_shape_gradients("co_shape_gradients",
                                                                                nq * nq);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> d_G("d_G",
                                                                  symmetric_tensor_dimension * nq_total);
  Kokkos::View<Number *, MemorySpace::Default::kokkos_space> scratch("scratch", total_scratch);

  {
    std::mt19937                           gen(7);
    std::uniform_real_distribution<double> dist(-1., 1.);

    auto co_shape_gradients_host = Kokkos::create_mirror_view(co_shape_gradients);
    for (int i = 0; i < nq * nq; ++i)
      co_shape_gradients_host(i) = dist(gen);
    Kokkos::deep_copy(co_shape_gradients, co_shape_gradients_host);

    auto scratch_host = Kokkos::create_mirror_view(scratch);
    for (int i = 0; i < nq_total; ++i)
      scratch_host(values_offset + i) = dist(gen);
    Kokkos::deep_copy(scratch, scratch_host);

    // Identity tensor: diagonal (d, d) components are 1, everything else 0.
    auto d_G_host = Kokkos::create_mirror_view(d_G);
    for (int c = 0; c < symmetric_tensor_dimension; ++c)
      for (int q = 0; q < nq_total; ++q)
        d_G_host(c * nq_total + q) = 0;
    for (int d = 0; d < dim; ++d)
      {
        const int c = d * dim - (d * (d - 1)) / 2; // packed index of the (d, d) diagonal entry
        for (int q = 0; q < nq_total; ++q)
          d_G_host(c * nq_total + q) = 1;
      }
    Kokkos::deep_copy(d_G, d_G_host);
  }

  const bool is_serial =
    std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int threadsPerBlock = is_serial ? 1 : Utilities::pow(nq, dim - 1);

  Kokkos::TeamPolicy<> policy(1, threadsPerBlock);
  policy.set_scratch_size(0, Kokkos::PerTeam(0));

  Kokkos::parallel_for(
    policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
      const int threadIdx = team_member.team_rank();
      const int blockSize = team_member.team_size();

      Number *co_shape_gradients_ptr = co_shape_gradients.data();
      Number *scratch_ptr            = scratch.data();

      // matrix (shape_values) isn't used by evaluate_gradients()/
      //  evaluate_gradients_and_multiply_symmetric_tensor(), so nullptr/nq are fine
      // stand-ins for it/n_rows here.
      const Custom::Parallel::FEEvaluationImplTransformToCollocation<dim, nq, nq, Number> fe_eval(
        team_member, nullptr, co_shape_gradients_ptr, 1, threadIdx, blockSize);

      fe_eval.evaluate_gradients(scratch_ptr + values_offset, scratch_ptr + grad_a_offset, nq_total);

      fe_eval.evaluate_gradients_and_multiply_symmetric_tensor(d_G,
                                                     0,
                                                     1,
                                                     Custom::Parallel::CellRangeIdView(),
                                                     scratch_ptr + values_offset,
                                                     scratch_ptr + grad_b_offset,
                                                     nq_total);
    });
  Kokkos::fence();

  auto scratch_host = Kokkos::create_mirror_view(scratch);
  Kokkos::deep_copy(scratch_host, scratch);

  double max_abs_diff = 0.;
  double max_abs_val  = 0.;
  for (int i = 0; i < dim * nq_total; ++i)
    {
      const double diff =
        std::abs(scratch_host(grad_a_offset + i) - scratch_host(grad_b_offset + i));
      max_abs_diff = std::max(max_abs_diff, diff);
      max_abs_val  = std::max(max_abs_val, std::abs(scratch_host(grad_a_offset + i)));
    }

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val);

  std::cout << "evaluate_gradients() vs evaluate_gradients_and_multiply_symmetric_tensor()+identity: dim = "
            << dim << ", nq = " << nq << std::endl;
  std::cout << "  max |grad|          = " << max_abs_val << std::endl;
  std::cout << "  max |grad_a - grad_b| = " << max_abs_diff << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  bool all_passed = true;
  all_passed &= run_test<3, 3>();
  all_passed &= run_test<3, 1>();
  all_passed &= run_test<2, 4>();
  all_passed &= run_test<2, 1>();

  // nm != nq (over-integrated quadrature): exercises the rectangular
  // n_rows != n_columns path in apply_matrix_vector_product, not touched by
  // any of the nm == nq cases above.
  all_passed &= run_test<3, 2, 5>();
  all_passed &= run_test<2, 3, 6>();

  all_passed &= run_evaluate_test<3, 4>();
  all_passed &= run_evaluate_test<2, 5>();

  std::cout << (all_passed ? "ALL PASS" : "SOME FAILED") << std::endl;

  return all_passed ? 0 : 1;
}
