// Properties of Portable::ElasticityOperator that do not need a reference
// operator, on a curved, sheared mesh without Dirichlet constraints: rigid
// body modes are in the kernel (rotations only for the symmetric gradient)
// and (v, A u) = (u, A v). With IdentityMaterial the operator is the vector
// Laplacian, compared with the FEValues reference of deformed_mesh_util.h.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "deformed_mesh_util.h"
#include "operators/portable_elasticity_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

// u = t + W x with W antisymmetric: t = e_mode for mode < dim, otherwise
// the rotation in the plane (a, b) number mode - dim
template <int dim>
class RigidBodyMode : public Function<dim>
{
public:
  RigidBodyMode(const unsigned int mode)
    : Function<dim>(dim)
    , mode(mode)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    if (mode < dim)
      return component == mode ? 1. : 0.;

    unsigned int plane = mode - dim;
    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int b = a + 1; b < dim; ++b, --plane)
        if (plane == 0)
          return component == a ? -p[b] : (component == b ? p[a] : 0.);
    return 0.;
  }

private:
  const unsigned int mode;
};



template <int dim, typename Number>
void
interpolate_to_device(const Mapping<dim>       &mapping,
                      const DoFHandler<dim>    &dof_handler,
                      const Function<dim>      &function,
                      DeviceVectorType<Number> &vec)
{
  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> host(
    locally_owned_dofs, DoFTools::extract_locally_relevant_dofs(dof_handler), MPI_COMM_WORLD);
  VectorTools::interpolate(mapping, dof_handler, function, host);

  LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
  for (const auto i : locally_owned_dofs)
    rw(i) = host(i);
  vec.import_elements(rw, VectorOperation::insert);
}



template <int dim, int fe_degree>
void
test()
{
  const bool overlap = false;

  using Number     = double;
  using VectorType = DeviceVectorType<Number>;

  deallog << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, 2);

  const FESystem<dim> fe(FE_Q<dim>{fe_degree}, dim);
  const MappingQ<dim> mapping(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number> no_constraints;
  no_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  no_constraints.close();

  {
    using Material = Portable::LinearElasticMaterial<dim, Number>;
    const Portable::ElasticityOperator<dim, fe_degree, Number, Material> op(dof_handler,
                                                                            no_constraints,
                                                                            Material{1.7, 0.6},
                                                                            overlap);

    VectorType u, v, Au, Av;
    op.initialize_dof_vector(u);
    op.initialize_dof_vector(v);
    op.initialize_dof_vector(Au);
    op.initialize_dof_vector(Av);

    fill_source(u, locally_owned_dofs);
    op.vmult(Au, u);
    const double scale = Au.l2_norm() / u.l2_norm();

    for (unsigned int mode = 0; mode < dim + (dim * (dim - 1)) / 2; ++mode)
      {
        interpolate_to_device(mapping, dof_handler, RigidBodyMode<dim>(mode), v);
        op.vmult(Av, v);
        report("rigid body mode " + std::to_string(mode),
               Av.l2_norm() / (scale * v.l2_norm()),
               1e-11);
      }

    // a second vector that is not a multiple of the first
    v = u;
    v.scale(u);
    op.vmult(Av, v);
    const double vAu = v * Au, uAv = u * Av;
    report("symmetry", std::abs(vAu - uAv) / std::abs(vAu), 1e-12);
  }

  {
    using Material = Portable::IdentityMaterial<dim, Number>;
    const Portable::ElasticityOperator<dim, fe_degree, Number, Material> op(dof_handler,
                                                                            no_constraints,
                                                                            Material(),
                                                                            overlap);

    VectorType src, dst, dst_ref, diag_ref;
    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst);
    op.initialize_dof_vector(dst_ref);
    op.initialize_dof_vector(diag_ref);

    fill_source(src, locally_owned_dofs);
    reference_laplace<dim, Number>(mapping,
                                   dof_handler,
                                   no_constraints,
                                   fe_degree + 1,
                                   locally_owned_dofs,
                                   locally_relevant_dofs,
                                   MPI_COMM_WORLD,
                                   src,
                                   dst_ref,
                                   diag_ref);

    op.vmult(dst, src);
    Kokkos::fence();
    report("identity material = vector Laplacian", relative_l2_difference(dst, dst_ref), 1e-12);
  }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1>();
  test<2, 2>();
  test<2, 3>();
  test<3, 1>();
  test<3, 2>();
}
