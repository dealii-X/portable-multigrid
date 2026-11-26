// PolynomialTransfer for FESystem(FE_Q(p), dim), with zero Dirichlet values
// on the first component only, on the curved, sheared mesh of
// deformed_mesh_util.h:
// - prolongate_and_add() against FETools::interpolate() of the coarse
//   function into the fine space
// - restrict_and_add() as its transpose, (R v, u) = (v, P u)

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "../operators/deformed_mesh_util.h"
#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_elasticity_operator.h"

using namespace dealii;
using namespace deformed_mesh_test;

template <int dim, int p_coarse, int p_fine>
void
test(const unsigned int n_refinements)
{
  using Number     = double;
  using VectorType = DeviceVectorType<Number>;
  using HostVector = LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
  using Material   = Portable::LinearElasticMaterial<dim, Number>;

  deallog << "dim = " << dim << ", p_coarse = " << p_coarse << ", p_fine = " << p_fine << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  const CurvedShearedManifold<dim>          manifold;
  make_deformed_triangulation(triangulation, manifold, n_refinements);

  const FESystem<dim> fe_coarse(FE_Q<dim>{p_coarse}, dim), fe_fine(FE_Q<dim>{p_fine}, dim);
  DoFHandler<dim>     dof_handler_coarse(triangulation), dof_handler_fine(triangulation);
  dof_handler_coarse.distribute_dofs(fe_coarse);
  dof_handler_fine.distribute_dofs(fe_fine);

  const auto make_constraints = [](const DoFHandler<dim> &dof_handler)
  {
    AffineConstraints<Number> constraints(dof_handler.locally_owned_dofs(),
                                          DoFTools::extract_locally_relevant_dofs(dof_handler));
    ComponentMask             mask(dim, false);
    mask.set(0, true);
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints, mask);
    constraints.close();
    return constraints;
  };
  const AffineConstraints<Number> constraints_coarse = make_constraints(dof_handler_coarse);
  const AffineConstraints<Number> constraints_fine   = make_constraints(dof_handler_fine);

  const Material                                                      material{};
  const Portable::ElasticityOperator<dim, p_coarse, Number, Material> op_coarse(dof_handler_coarse,
                                                                                constraints_coarse,
                                                                                material,
                                                                                false);
  const Portable::ElasticityOperator<dim, p_fine, Number, Material>   op_fine(dof_handler_fine,
                                                                            constraints_fine,
                                                                            material,
                                                                            false);

  Portable::PolynomialTransfer<dim, p_coarse, p_fine, Number> transfer;
  transfer.reinit(op_coarse, op_fine, constraints_coarse, constraints_fine);

  VectorType u_coarse, r_coarse, v_fine, p_fine_vec, p_fine_ref;
  op_coarse.initialize_dof_vector(u_coarse);
  op_coarse.initialize_dof_vector(r_coarse);
  op_fine.initialize_dof_vector(v_fine);
  op_fine.initialize_dof_vector(p_fine_vec);
  op_fine.initialize_dof_vector(p_fine_ref);

  fill_source(u_coarse, dof_handler_coarse.locally_owned_dofs(), &constraints_coarse);
  fill_source(v_fine, dof_handler_fine.locally_owned_dofs(), &constraints_fine);

  deallog << "n_dofs coarse = " << dof_handler_coarse.n_dofs()
          << ", fine = " << dof_handler_fine.n_dofs() << std::endl;

  p_fine_vec = 0.;
  transfer.prolongate_and_add(p_fine_vec, u_coarse);

  // reference
  {
    HostVector                             u_host(dof_handler_coarse.locally_owned_dofs(),
                      DoFTools::extract_locally_relevant_dofs(dof_handler_coarse),
                      MPI_COMM_WORLD);
    LinearAlgebra::ReadWriteVector<Number> rw(dof_handler_coarse.locally_owned_dofs());
    rw.import_elements(u_coarse, VectorOperation::insert);
    u_host.import_elements(rw, VectorOperation::insert);
    u_host.update_ghost_values();

    HostVector p_host(dof_handler_fine.locally_owned_dofs(),
                      DoFTools::extract_locally_relevant_dofs(dof_handler_fine),
                      MPI_COMM_WORLD);
    FETools::interpolate(dof_handler_coarse, u_host, dof_handler_fine, constraints_fine, p_host);
    constraints_fine.set_zero(p_host);

    LinearAlgebra::ReadWriteVector<Number> rw_fine(dof_handler_fine.locally_owned_dofs());
    rw_fine.import_elements(p_host, VectorOperation::insert);
    p_fine_ref.import_elements(rw_fine, VectorOperation::insert);
  }
  report("prolongate_and_add", relative_l2_difference(p_fine_vec, p_fine_ref), 1e-13);

  r_coarse = 0.;
  transfer.restrict_and_add(r_coarse, v_fine);
  const double r_u = r_coarse * u_coarse;
  report("restrict_and_add", std::abs(r_u - v_fine * p_fine_vec) / std::abs(r_u), 1e-13);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<2, 1, 2>(3);
  test<2, 2, 4>(2);
  test<3, 1, 2>(2);
  test<3, 2, 4>(1);
  test<3, 1, 3>(1);
}
