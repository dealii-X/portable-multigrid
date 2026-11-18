#ifndef testsuite_cube_random_util_h
#define testsuite_cube_random_util_h

// Setup of the CUDA kernel tests: a refined [-1, 1]^dim cube with zero
// Dirichlet values, a random source vector, and the relative l2 difference
// of two results.

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <random>
#include <string>

namespace cube_random
{
  using namespace dealii;

  using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  template <int dim>
  void
  make_cube(parallel::distributed::Triangulation<dim> &triangulation)
  {
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);
  }

  template <int dim>
  AffineConstraints<double>
  make_dirichlet_constraints(const DoFHandler<dim> &dof_handler)
  {
    AffineConstraints<double> constraints;
    constraints.reinit(dof_handler.locally_owned_dofs(),
                       DoFTools::extract_locally_relevant_dofs(dof_handler));
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(
                                               dof_handler.get_fe().n_components()),
                                             constraints);
    constraints.close();
    return constraints;
  }

  // uniform in [-1, 1], zero on constrained DoFs (the custom kernels never
  // read them, deal.II's cell loop does)
  inline void
  fill_random(VectorType                      &vec,
              const IndexSet                  &locally_owned_dofs,
              const AffineConstraints<double> &constraints)
  {
    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1., 1.);

    LinearAlgebra::ReadWriteVector<double> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = constraints.is_constrained(i) ? 0. : dist(gen);
    vec.import_elements(rw, VectorOperation::insert);
  }

  // relative difference on the unconstrained DoFs
  inline double
  relative_difference(const VectorType                &a,
                      const VectorType                &b,
                      const IndexSet                  &locally_owned_dofs,
                      const AffineConstraints<double> &constraints)
  {
    LinearAlgebra::ReadWriteVector<double> ra(locally_owned_dofs), rb(locally_owned_dofs);
    ra.import_elements(a, VectorOperation::insert);
    rb.import_elements(b, VectorOperation::insert);

    double diff = 0., norm = 0.;
    for (const auto i : locally_owned_dofs)
      if (!constraints.is_constrained(i))
        {
          diff += (ra(i) - rb(i)) * (ra(i) - rb(i));
          norm += rb(i) * rb(i);
        }
    diff = Utilities::MPI::sum(diff, MPI_COMM_WORLD);
    norm = Utilities::MPI::sum(norm, MPI_COMM_WORLD);
    return norm > 0. ? std::sqrt(diff / norm) : std::sqrt(diff);
  }

  inline void
  report(const std::string &name, const double err, const double tolerance)
  {
    if (err <= tolerance)
      deallog << name << ": OK" << std::endl;
    else
      deallog << name << ": rel_err = " << err << " > " << tolerance << std::endl;
  }
} // namespace cube_random

#endif
