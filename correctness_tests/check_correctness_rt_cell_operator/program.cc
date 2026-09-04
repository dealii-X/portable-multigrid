// Correctness check for the Portable (GPU) Raviart-Thomas cell operator
// Portable::RT::compute_cell (include/kernels/kokkos_kernels_rt.h), driven
// through Portable::RT::RaviartThomasOperatorBase::test() (include/operators/
// portable_momentum_operator_rt.h).
//
// Reference: a plain deal.II MatrixFree::cell_loop() with FEEvaluation over
// FE_RaviartThomasNodal, computing the same Helmholtz cell integrand
//
//     a(u, v) = \int_K ( factor_mass * u . v  +  factor_lapl * grad u : grad v ) dx
//
// with u, v Piola-mapped to physical space. No face terms on either side --
// this exercises the cell kernel only.
//
// Caveats:
//  * compute_cell omits the Jacobian-gradient (curvature) terms of the Piola
//    map, so it agrees with FEEvaluation only on affine cells. The test mesh
//    is therefore an undeformed hyper_cube (affine for every refinement).
//  * RaviartThomasOperatorBase::reinit() renumbers DoFs into its own global
//    ordering; that ordering is the identity only on a single MPI rank, and
//    test() moves data host<->device by matching global indices. The
//    comparison is therefore only meaningful on one rank -- main() skips the
//    run otherwise.
//  * For FE_RaviartThomasNodal(k) the anisotropic ShapeInfo reports
//    data[0].fe_degree == k + 1, and test() dispatches on that value with
//    n_q = (k + 1) + 1 = k + 2. So reinit() and the reference MatrixFree are
//    both built with QGauss<1>(k + 2), and the supported element degrees are
//    k in {1, 2, 3, 4} (data[0].fe_degree in {2, 3, 4, 5}).

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <cmath>
#include <iostream>
#include <random>

#include "operators/portable_momentum_operator_rt.h"

using namespace dealii;

using Number = double;

// deal.II reference: Helmholtz *cell* operator for the vector RT element.
template <int dim>
class HelmholtzCellReference
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  HelmholtzCellReference(const MatrixFree<dim, Number> &matrix_free,
                         const Number                   factor_mass,
                         const Number                   factor_lapl)
    : matrix_free(matrix_free)
    , factor_mass(factor_mass)
    , factor_lapl(factor_lapl)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.cell_loop(&HelmholtzCellReference::local_apply, this, dst, src, true);
  }

private:
  void
  local_apply(const MatrixFree<dim, Number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval(matrix_free);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : eval.quadrature_point_indices())
          {
            eval.submit_value(make_vectorized_array<Number>(factor_mass) * eval.get_value(q), q);
            eval.submit_gradient(make_vectorized_array<Number>(factor_lapl) * eval.get_gradient(q),
                                 q);
          }

        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  const MatrixFree<dim, Number> &matrix_free;
  const Number                   factor_mass;
  const Number                   factor_lapl;
};


// deal.II reference: full Helmholtz operator = cell term + SIPG viscous flux on
// interior faces (Neumann boundary -> no boundary terms). Uses the manual
// normal-vector form, which is the correct one for the Piola-mapped RT element.
template <int dim>
class HelmholtzReference
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  HelmholtzReference(const MatrixFree<dim, Number> &matrix_free,
                     const Number                   factor_mass,
                     const Number                   factor_lapl)
    : matrix_free(matrix_free)
    , factor_mass(factor_mass)
    , factor_lapl(factor_lapl)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.loop(&HelmholtzReference::cell_op,
                     &HelmholtzReference::inner_face_op,
                     &HelmholtzReference::boundary_op,
                     this,
                     dst,
                     src,
                     true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
                     MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
  }

private:
  void
  cell_op(const MatrixFree<dim, Number>               &mf,
          VectorType                                  &dst,
          const VectorType                            &src,
          const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, -1, 0, dim, Number> eval(mf);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        for (const unsigned int q : eval.quadrature_point_indices())
          {
            eval.submit_value(make_vectorized_array<Number>(factor_mass) * eval.get_value(q), q);
            eval.submit_gradient(make_vectorized_array<Number>(factor_lapl) * eval.get_gradient(q),
                                 q);
          }
        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  void
  inner_face_op(const MatrixFree<dim, Number>               &mf,
                VectorType                                  &dst,
                const VectorType                            &src,
                const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> phi_m(mf, true);
    FEFaceEvaluation<dim, -1, 0, dim, Number> phi_p(mf, false);
    const unsigned int degree = mf.get_dof_handler().get_fe().degree;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);
        phi_p.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        // penalty: (|J^-1 n|_minus + |J^-1 n|_plus) * max(p,1)(p+1) * factor_lapl.
        // Axis-aligned: |J^-1 n| == |(J^-1 n)_d|, matching
        // RaviartThomasOperatorBase::penalty_parameters.
        const auto n0     = phi_m.normal_vector(0);
        const auto m_m    = n0 * phi_m.inverse_jacobian(0); // J^-1 n
        const auto m_p    = n0 * phi_p.inverse_jacobian(0);
        const auto sigmaF =
          (m_m.norm() + m_p.norm()) *
          make_vectorized_array<Number>(std::max<unsigned int>(degree, 1) * (degree + 1.0) *
                                        factor_lapl);

        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
          {
            const auto normal = phi_m.normal_vector(q);
            const auto u_m    = phi_m.get_value(q);
            const auto u_p    = phi_p.get_value(q);
            const auto dn_m   = phi_m.get_gradient(q) * normal;
            const auto dn_p   = phi_p.get_gradient(q) * normal;

            const auto viscous_value_flux =
              make_vectorized_array<Number>(0.5 * factor_lapl) * (dn_m + dn_p) -
              sigmaF * (u_m - u_p);
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(0.5 * factor_lapl) * (u_p - u_m);

            phi_m.submit_value(-viscous_value_flux, q);
            phi_p.submit_value(viscous_value_flux, q);
            phi_m.submit_gradient(outer_product(viscous_gradient_flux, normal), q);
            phi_p.submit_gradient(outer_product(viscous_gradient_flux, normal), q);
          }
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  // Single-sided mirror of inner_face_op: boundary_id == 0 is Neumann (natural,
  // zero contribution -- u_outer = u_inner, dn_outer = -dn_inner cancels
  // everything); boundary_id != 0 is homogeneous Dirichlet (g = 0), mirrored as
  // u_outer = -u_inner, dn_outer = dn_inner. FEFaceEvaluation::get_value /
  // get_gradient already return the Piola-mapped physical quantities for RT, so
  // no manual Piola handling is needed here (unlike the GPU kernel).
  void
  boundary_op(const MatrixFree<dim, Number>               &mf,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, -1, 0, dim, Number> phi(mf, true);
    const unsigned int degree = mf.get_dof_handler().get_fe().degree;

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const bool dirichlet = mf.get_boundary_id(face) != 0;

        // penalty: 2 * |J^-1 n| * max(p,1)(p+1) * factor_lapl, matching
        // inner_face_op's sigmaF with m_p == m_m (same cell on both sides).
        const auto n0     = phi.normal_vector(0);
        const auto m0     = n0 * phi.inverse_jacobian(0);
        const auto sigmaF =
          make_vectorized_array<Number>(2.) * m0.norm() *
          make_vectorized_array<Number>(std::max<unsigned int>(degree, 1) * (degree + 1.0) *
                                        factor_lapl);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto normal   = phi.normal_vector(q);
            const auto u_inner  = phi.get_value(q);
            const auto dn_inner = phi.get_gradient(q) * normal;

            const auto u_outer  = dirichlet ? -u_inner : u_inner;
            const auto dn_outer = dirichlet ? dn_inner : -dn_inner;

            const auto viscous_value_flux =
              make_vectorized_array<Number>(0.5 * factor_lapl) * (dn_inner + dn_outer) -
              sigmaF * (u_inner - u_outer);
            const auto viscous_gradient_flux =
              make_vectorized_array<Number>(0.5 * factor_lapl) * (u_outer - u_inner);

            phi.submit_value(-viscous_value_flux, q);
            phi.submit_gradient(outer_product(viscous_gradient_flux, normal), q);
          }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  const MatrixFree<dim, Number> &matrix_free;
  const Number                   factor_mass;
  const Number                   factor_lapl;
};


// Fixed invertible linear map -> sheared parallelepiped cells (still affine, so
// MappingQ(1) is exact and compute_*_info's single-Jacobian read is valid).
template <int dim>
Point<dim>
shear(const Point<dim> &p)
{
  if (dim == 2)
    return Point<dim>(p[0] + 0.30 * p[1], p[1] + 0.15 * p[0]);
  return Point<dim>(p[0] + 0.30 * p[1] + 0.10 * p[2],
                    p[1] + 0.15 * p[0] + 0.20 * p[2],
                    p[2] + 0.05 * p[0] + 0.25 * p[1]);
}

template <int dim, int fe_degree>
bool
run_test(const Number factor_mass,
        const Number factor_lapl,
        const bool   deform    = false,
        const bool   mixed_bc  = false)
{
  const FE_RaviartThomasNodal<dim> fe(fe_degree);

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 2.);
  triangulation.refine_global(dim == 2 ? 2 : 1);

  // Tag the x == 0 boundary face Dirichlet (boundary_id 1); everything else
  // stays Neumann (boundary_id 0, the hyper_cube default). Done before any
  // shearing so the geometric test ("x == 0") is unambiguous.
  if (mixed_bc)
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const unsigned int f : cell->face_indices())
        if (cell->face(f)->at_boundary() && cell->face(f)->center()[0] < 1e-10)
          cell->face(f)->set_boundary_id(1);

  if (deform)
    GridTools::transform(&shear<dim>, triangulation);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Empty constraints -- matches how RaviartThomasOperatorBase is used
  // elsewhere (no boundary conditions on the cell operator).
  AffineConstraints<Number> constraints;
  constraints.close();

  const MappingQ<dim> mapping(1);                 // affine cube
  const QGauss<1>     quadrature(fe_degree + 2);  // n_q_points_1d == compute_cell's n_q (= k + 2)

  // -- GPU path --
  Portable::RT::RaviartThomasOperatorBase<dim, Number> rt_operator;
  rt_operator.reinit(mapping, dof_handler, constraints, quadrature);

  // -- deal.II reference --
  MatrixFree<dim, Number> matrix_free;
  {
    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags                = update_values | update_gradients | update_JxW_values;
    data.mapping_update_flags_inner_faces    = update_values | update_gradients | update_JxW_values |
                                            update_normal_vectors | update_jacobians;
    data.mapping_update_flags_boundary_faces = update_values | update_gradients | update_JxW_values |
                                               update_normal_vectors | update_jacobians;
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, data);
  }
  const HelmholtzCellReference<dim> reference(matrix_free, factor_mass, factor_lapl);
  const HelmholtzReference<dim>     reference_full(matrix_free, factor_mass, factor_lapl);

  // -- random src --
  LinearAlgebra::distributed::Vector<Number> src, dst_ref, dst_ref_full, dst_gpu, dst_gpu_faces;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_ref);
  matrix_free.initialize_dof_vector(dst_ref_full);
  matrix_free.initialize_dof_vector(dst_gpu);
  matrix_free.initialize_dof_vector(dst_gpu_faces);

  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<Number> rw(src.locally_owned_elements());
    for (const auto idx : src.locally_owned_elements())
      rw(idx) = dist(gen);
    src.import_elements(rw, VectorOperation::insert);
  }

  reference.vmult(dst_ref, src);            // cell term only
  reference_full.vmult(dst_ref_full, src);  // cell + SIPG inner faces

  // test() takes host vectors and does its own device round-trip.
  //  interpolate_to_faces = false -> volume term only
  //  interpolate_to_faces = true  -> volume + SIPG face kernels
  rt_operator.test(dst_gpu, src, factor_mass, factor_lapl, /*interpolate_to_faces=*/false);
  rt_operator.test(dst_gpu_faces, src, factor_mass, factor_lapl, /*interpolate_to_faces=*/true);

  double max_abs_diff  = 0.; // cell-only:   dst_ref      vs dst_gpu
  double max_abs_diff2 = 0.; // cell + face: dst_ref_full vs dst_gpu_faces
  double max_abs_val   = 0.;
  double max_abs_val2  = 0.;
  for (const auto idx : dst_ref.locally_owned_elements())
    {
      max_abs_diff  = std::max(max_abs_diff, std::abs(dst_ref(idx) - dst_gpu(idx)));
      max_abs_diff2 = std::max(max_abs_diff2, std::abs(dst_ref_full(idx) - dst_gpu_faces(idx)));
      max_abs_val   = std::max(max_abs_val, std::abs(dst_ref(idx)));
      max_abs_val2  = std::max(max_abs_val2, std::abs(dst_ref_full(idx)));
    }
  max_abs_diff  = Utilities::MPI::max(max_abs_diff, MPI_COMM_WORLD);
  max_abs_diff2 = Utilities::MPI::max(max_abs_diff2, MPI_COMM_WORLD);
  max_abs_val   = Utilities::MPI::max(max_abs_val, MPI_COMM_WORLD);
  max_abs_val2  = Utilities::MPI::max(max_abs_val2, MPI_COMM_WORLD);

  const bool pass = max_abs_diff < 1e-10 * std::max(1., max_abs_val) &&
                    max_abs_diff2 < 1e-10 * std::max(1., max_abs_val2);

  std::cout << "dim = " << dim << ", fe_degree = " << fe_degree
            << ", n_dofs = " << dof_handler.n_dofs()
            << " (factor_mass = " << factor_mass << ", factor_lapl = " << factor_lapl
            << (deform ? ", sheared" : "") << (mixed_bc ? ", mixed Dirichlet/Neumann" : "") << ")"
            << std::endl;
  std::cout << "  cell only:  max|ref| = " << max_abs_val << "   max|ref - gpu| = " << max_abs_diff
            << std::endl;
  std::cout << "  cell+face:  max|ref| = " << max_abs_val2 << "   max|ref - gpu| = " << max_abs_diff2
            << std::endl;
  std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

  return pass;
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) != 1)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "This test requires a single MPI rank "
                     "(RaviartThomasOperatorBase renumbers DoFs); skipping."
                  << std::endl;
      return 0;
    }

  bool all_passed = true;

  // Helmholtz (both terms), as used by RaviartThomasOperatorBase::test().
  const Number factor_mass = 800.;
  const Number factor_lapl = 1e-6;

  all_passed &= run_test<2, 1>(factor_mass, factor_lapl);
  all_passed &= run_test<2, 2>(factor_mass, factor_lapl);
  all_passed &= run_test<2, 3>(factor_mass, factor_lapl);
  all_passed &= run_test<2, 4>(factor_mass, factor_lapl);

  all_passed &= run_test<3, 1>(factor_mass, factor_lapl);
  all_passed &= run_test<3, 2>(factor_mass, factor_lapl);
  all_passed &= run_test<3, 3>(factor_mass, factor_lapl);
  all_passed &= run_test<3, 4>(factor_mass, factor_lapl);

  // Pure mass (no face terms) and pure stiffness (the SIPG face flux is a
  // large fraction of the result -- this is what exercises the face kernels).
  all_passed &= run_test<2, 2>(1., 0.);
  all_passed &= run_test<3, 2>(1., 0.);

  all_passed &= run_test<2, 1>(0., 1.);
  all_passed &= run_test<2, 2>(0., 1.);
  all_passed &= run_test<2, 3>(0., 1.);
  all_passed &= run_test<2, 4>(0., 1.);
  all_passed &= run_test<3, 1>(0., 1.);
  all_passed &= run_test<3, 2>(0., 1.);
  all_passed &= run_test<3, 3>(0., 1.);
  all_passed &= run_test<3, 4>(0., 1.);

  // General affine (sheared parallelepipeds): exercises the tangential
  // reference-derivative path in compute_inner_faces.
  all_passed &= run_test<2, 1>(0., 1., /*deform=*/true);
  all_passed &= run_test<2, 2>(0., 1., true);
  all_passed &= run_test<2, 3>(0., 1., true);
  all_passed &= run_test<3, 1>(0., 1., true);
  all_passed &= run_test<3, 2>(0., 1., true);
  all_passed &= run_test<3, 3>(0., 1., true);

  all_passed &= run_test<2, 2>(800., 1e-6, true);
  all_passed &= run_test<3, 2>(800., 1e-6, true);

  // Mixed Dirichlet (x == 0 face) / Neumann (everywhere else) boundary:
  // exercises compute_boundary_faces on axis-aligned and sheared meshes.
  all_passed &= run_test<2, 1>(0., 1., /*deform=*/false, /*mixed_bc=*/true);
  all_passed &= run_test<2, 2>(0., 1., false, true);
  all_passed &= run_test<2, 3>(0., 1., false, true);
  all_passed &= run_test<2, 4>(0., 1., false, true);
  all_passed &= run_test<3, 1>(0., 1., false, true);
  all_passed &= run_test<3, 2>(0., 1., false, true);
  all_passed &= run_test<3, 3>(0., 1., false, true);
  all_passed &= run_test<3, 4>(0., 1., false, true);

  all_passed &= run_test<2, 2>(800., 1e-6, false, true);
  all_passed &= run_test<3, 2>(800., 1e-6, false, true);

  all_passed &= run_test<2, 2>(0., 1., /*deform=*/true, /*mixed_bc=*/true);
  all_passed &= run_test<3, 2>(0., 1., true, true);

  std::cout << (all_passed ? "ALL PASS" : "SOME FAILED") << std::endl;

  return all_passed ? 0 : 1;
}
