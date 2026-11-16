// Correctness check for Copy::Portable::MGTwoLevelTransfer
// (include/multigrid_dealii/portable_mg_transfer_global_coarsening.templates.h),
// the copy of deal.II's Portable two-level transfer whose cell kernels are
// rewritten on top of Copy::Portable::internal::batched::EvaluatorTensorProduct.
//
// For every configuration, prolongate_and_add() and restrict_and_add() are
// applied to random vectors and the result is compared against two references:
//
//   (a) dealii::MGTwoLevelTransfer on the host, the matrix-free CPU transfer
//       whose setup the portable transfer wraps, and
//   (b) the installed dealii::Portable::MGTwoLevelTransfer, i.e. the kernels
//       the copy started from.
//
// Configurations: dim = 2, 3; FE_Q degree 1..4; scalar and dim-component
// FESystem(FE_Q); double and float; and four meshes:
//
//   uniform    two independent triangulations, globally refined once and
//              twice (deal.II test mg_transfer_kokkos_h_01), only the
//              embedding scheme;
//   adaptive   one adaptive refinement step, the coarse mesh is the fine mesh
//              coarsened once (mg_transfer_kokkos_a_01): identity and
//              embedding scheme, hanging nodes on the fine level only;
//   hanging    the coarse mesh is adaptively refined and has hanging nodes,
//              the fine mesh refines part of it: identity and embedding
//              scheme with hanging-node masks on the coarse level;
//   dirichlet  uniform plus homogeneous Dirichlet constraints on both levels:
//              the invalid-index gather/scatter path;
//   antiperiodic
//              uniform plus anti-periodicity in x on both levels, i.e.
//              constraints with a single entry of weight -1. (Weight +1
//              would be turned into an index redirection by the matrix-free
//              constraint setup.) This selects the CRS gather/scatter path.
//
// The copy is applied with several numbers of coarse cells per batch
// (automatic, 1, 3 and 7). Run with one and with two MPI ranks. Exit code 0
// means all configurations agree with both references within the tolerance.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/portable_mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "multigrid_dealii/portable_mg_transfer_global_coarsening.templates.h"

using namespace dealii;

// Access to the private list of transfer schemes of the copy, to report which
// code paths (embedding/identity scheme, CRS constraints, hanging-node masks)
// a configuration exercises. Same mechanism as in
// multigrid_dealii/mg_transfer_private_access.h; the tag has to live in the
// namespace of PrivateMemberAccess, where the explicit instantiation defines
// the friend function.
DEAL_II_NAMESPACE_OPEN
namespace Copy
{
  namespace Portable
  {
    namespace internal
    {
      template <int dim, typename Number>
      struct CopySchemesTag
      {
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
        friend auto private_member(CopySchemesTag);
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
      };

#define INSTANTIATE_COPY_SCHEMES_ACCESS(dim, Number)                                             \
  template struct PrivateMemberAccess<                                                           \
    CopySchemesTag<dim, Number>,                                                                 \
    &MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>:: \
      schemes>;

      INSTANTIATE_COPY_SCHEMES_ACCESS(2, double)
      INSTANTIATE_COPY_SCHEMES_ACCESS(3, double)
      INSTANTIATE_COPY_SCHEMES_ACCESS(2, float)
      INSTANTIATE_COPY_SCHEMES_ACCESS(3, float)

#undef INSTANTIATE_COPY_SCHEMES_ACCESS
    } // namespace internal
  } // namespace Portable
} // namespace Copy
DEAL_II_NAMESPACE_CLOSE

/**
 * One letter per scheme with cells: E for the embedding (interpolating)
 * scheme, I for the identity scheme, followed by c if the CRS constraint
 * path is used and h if hanging-node masks are present.
 */
template <int dim, typename TransferType>
std::string
scheme_paths(const TransferType &transfer)
{
  using Number = typename TransferType::Number;

  std::string result;
  for (const auto &scheme :
       transfer.*private_member(Copy::Portable::internal::CopySchemesTag<dim, Number>{}))
    {
      if (scheme.n_coarse_cells == 0)
        continue;
      result += (scheme.prolongation_matrix.extent(0) > 0) ? "E" : "I";
      if (scheme.use_constraint_crs)
        result += "c";
      if (scheme.hanging_node_masks.extent(0) > 0)
        result += "h";
      result += " ";
    }
  return result;
}

enum class MeshType
{
  uniform,
  adaptive,
  hanging,
  dirichlet,
  antiperiodic
};

std::string
to_string(const MeshType type)
{
  switch (type)
    {
      case MeshType::uniform:
        return "uniform";
      case MeshType::adaptive:
        return "adaptive";
      case MeshType::hanging:
        return "hanging";
      case MeshType::dirichlet:
        return "dirichlet";
      case MeshType::antiperiodic:
        return "antiperiodic";
    }
  return "";
}



template <int dim>
void
create_meshes(const MeshType                             type,
              parallel::distributed::Triangulation<dim> &tria_coarse,
              parallel::distributed::Triangulation<dim> &tria_fine)
{
  if (type == MeshType::adaptive)
    {
      // mg_transfer_kokkos_a_01: refine the left half once more, the coarse
      // mesh is the fine mesh coarsened once
      const auto create_fine_grid = [](Triangulation<dim> &tria)
      {
        GridGenerator::hyper_cube(tria);
        tria.refine_global();
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && cell->center()[0] < 0.5)
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      };

      create_fine_grid(tria_coarse);
      for (const auto &cell : tria_coarse.active_cell_iterators())
        if (cell->is_locally_owned())
          cell->set_coarsen_flag();
      tria_coarse.execute_coarsening_and_refinement();

      create_fine_grid(tria_fine);
      return;
    }

  if (type == MeshType::hanging)
    {
      // the coarse mesh has hanging nodes, the fine mesh refines the cells
      // with y < 0.5 once more (which keeps the 2:1 balance, so no further
      // cells are refined)
      const auto create_coarse_grid = [](Triangulation<dim> &tria)
      {
        GridGenerator::hyper_cube(tria);
        tria.refine_global();
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() && cell->center()[0] < 0.5)
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      };

      create_coarse_grid(tria_coarse);

      create_coarse_grid(tria_fine);
      for (const auto &cell : tria_fine.active_cell_iterators())
        if (cell->is_locally_owned() && cell->center()[1] < 0.5)
          cell->set_refine_flag();
      tria_fine.execute_coarsening_and_refinement();
      return;
    }

  // mg_transfer_kokkos_h_01: two independent, globally refined meshes
  const bool colorize = (type == MeshType::antiperiodic);
  for (auto *tria : {&tria_coarse, &tria_fine})
    {
      GridGenerator::hyper_cube(*tria, 0., 1., colorize);
      if (type == MeshType::antiperiodic)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename parallel::distributed::Triangulation<dim>::cell_iterator>>
            matched_pairs;
          GridTools::collect_periodic_faces(*tria, 0, 1, 0, matched_pairs);
          tria->add_periodicity(matched_pairs);
        }
    }
  tria_coarse.refine_global(1);
  tria_fine.refine_global(2);
}



template <int dim, typename Number>
void
make_constraints(const MeshType             type,
                 const DoFHandler<dim>     &dof_handler,
                 AffineConstraints<Number> &constraints)
{
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  if (type == MeshType::dirichlet)
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim, Number>(
                                               dof_handler.get_fe().n_components()),
                                             constraints);
  if (type == MeshType::antiperiodic)
    DoFTools::make_periodicity_constraints(
      dof_handler, 0, 1, 0, constraints, ComponentMask(), Number(-1.));
  constraints.close();
}



template <int dim, typename Number, typename MemorySpaceType>
void
initialize_vector(LinearAlgebra::distributed::Vector<Number, MemorySpaceType> &vec,
                  const DoFHandler<dim>                                       &dof_handler)
{
  vec.reinit(dof_handler.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler),
             dof_handler.get_mpi_communicator());
}



template <typename Number>
void
copy_to_device(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>    &dst,
               const LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &src)
{
  LinearAlgebra::ReadWriteVector<Number> rw_vector(src.get_partitioner()->locally_owned_range());
  rw_vector.import_elements(src, VectorOperation::insert);
  dst.import_elements(rw_vector, VectorOperation::insert);
}



template <typename Number>
void
copy_to_host(LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>          &dst,
             const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src)
{
  LinearAlgebra::ReadWriteVector<Number> rw_vector(src.get_partitioner()->locally_owned_range());
  rw_vector.import_elements(src, VectorOperation::insert);
  dst.import_elements(rw_vector, VectorOperation::insert);
}



/**
 * Relative difference max|a - b| / max|b| over the locally owned entries,
 * reduced over all ranks.
 */
template <typename Number>
double
relative_difference(const LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &a,
                    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &b)
{
  double max_diff = 0., max_ref = 0.;
  for (unsigned int i = 0; i < b.locally_owned_size(); ++i)
    {
      max_diff = std::max<double>(max_diff, std::abs(a.local_element(i) - b.local_element(i)));
      max_ref  = std::max<double>(max_ref, std::abs(b.local_element(i)));
    }
  max_diff = Utilities::MPI::max(max_diff, b.get_mpi_communicator());
  max_ref  = Utilities::MPI::max(max_ref, b.get_mpi_communicator());
  return max_diff / std::max(max_ref, 1e-300);
}



template <int dim, typename Number>
bool
run(const unsigned int degree, const unsigned int n_components, const MeshType mesh_type)
{
  using HostVector   = LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>;
  using DeviceVector = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  const unsigned int rank      = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const double       tolerance = std::is_same_v<Number, double> ? 1e-12 : 1e-4;

  // meshes and DoFs
  parallel::distributed::Triangulation<dim> tria_coarse(MPI_COMM_WORLD);
  parallel::distributed::Triangulation<dim> tria_fine(MPI_COMM_WORLD);
  create_meshes(mesh_type, tria_coarse, tria_fine);

  const FESystem<dim> fe(FE_Q<dim>(degree), n_components);

  DoFHandler<dim> dof_handler_fine(tria_fine);
  dof_handler_fine.distribute_dofs(fe);
  DoFHandler<dim> dof_handler_coarse(tria_coarse);
  dof_handler_coarse.distribute_dofs(fe);

  AffineConstraints<Number> constraints_fine, constraints_coarse;
  make_constraints(mesh_type, dof_handler_fine, constraints_fine);
  make_constraints(mesh_type, dof_handler_coarse, constraints_coarse);

  // transfer operators: reference on the host, installed portable, copy
  MGTwoLevelTransfer<dim, HostVector> transfer_host;
  transfer_host.reinit_geometric_transfer(dof_handler_fine,
                                          dof_handler_coarse,
                                          constraints_fine,
                                          constraints_coarse);

  Portable::MGTwoLevelTransfer<dim, DeviceVector> transfer_installed;
  transfer_installed.reinit_geometric_transfer(dof_handler_fine,
                                               dof_handler_coarse,
                                               constraints_fine,
                                               constraints_coarse);

  Copy::Portable::MGTwoLevelTransfer<dim, DeviceVector> transfer_copy;
  transfer_copy.reinit_geometric_transfer(dof_handler_fine,
                                          dof_handler_coarse,
                                          constraints_fine,
                                          constraints_coarse);

  // random input vectors on the host
  HostVector src_coarse_host, src_fine_host, dst_coarse_host, dst_fine_host;
  initialize_vector(src_coarse_host, dof_handler_coarse);
  initialize_vector(dst_coarse_host, dof_handler_coarse);
  initialize_vector(src_fine_host, dof_handler_fine);
  initialize_vector(dst_fine_host, dof_handler_fine);

  std::mt19937                           generator(42 + rank);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  for (unsigned int i = 0; i < src_coarse_host.locally_owned_size(); ++i)
    src_coarse_host.local_element(i) = distribution(generator);
  for (unsigned int i = 0; i < src_fine_host.locally_owned_size(); ++i)
    src_fine_host.local_element(i) = distribution(generator);

  DeviceVector src_coarse_device, src_fine_device, dst_coarse_device, dst_fine_device;
  initialize_vector(src_coarse_device, dof_handler_coarse);
  initialize_vector(dst_coarse_device, dof_handler_coarse);
  initialize_vector(src_fine_device, dof_handler_fine);
  initialize_vector(dst_fine_device, dof_handler_fine);
  copy_to_device(src_coarse_device, src_coarse_host);
  copy_to_device(src_fine_device, src_fine_host);

  // reference results on the host
  dst_fine_host = 0.;
  transfer_host.prolongate_and_add(dst_fine_host, src_coarse_host);
  dst_coarse_host = 0.;
  transfer_host.restrict_and_add(dst_coarse_host, src_fine_host);

  HostVector result_host;
  const auto apply_and_compare =
    [&](const auto &transfer, double &error_prolongation, double &error_restriction)
  {
    dst_fine_device = 0.;
    transfer.prolongate_and_add(dst_fine_device, src_coarse_device);
    result_host.reinit(dst_fine_host);
    copy_to_host(result_host, dst_fine_device);
    error_prolongation = relative_difference(result_host, dst_fine_host);

    dst_coarse_device = 0.;
    transfer.restrict_and_add(dst_coarse_device, src_fine_device);
    result_host.reinit(dst_coarse_host);
    copy_to_host(result_host, dst_coarse_device);
    error_restriction = relative_difference(result_host, dst_coarse_host);
  };

  double error_installed_p, error_installed_r;
  apply_and_compare(transfer_installed, error_installed_p, error_installed_r);

  // the copy with several numbers of cells per batch: the automatic choice,
  // one cell per team, and odd sizes that leave a shorter last batch; the
  // reported error is the maximum over all of them
  double error_copy_p = 0., error_copy_r = 0.;
  for (const unsigned int cells_per_batch : {numbers::invalid_unsigned_int, 1u, 3u, 7u})
    {
      transfer_copy.set_cells_per_batch(cells_per_batch);
      double error_p, error_r;
      apply_and_compare(transfer_copy, error_p, error_r);
      error_copy_p = std::max(error_copy_p, error_p);
      error_copy_r = std::max(error_copy_r, error_r);
    }

  const bool ok = error_copy_p < tolerance && error_copy_r < tolerance &&
                  error_installed_p < tolerance && error_installed_r < tolerance;

  // collective operations before the rank-0 output
  const double norm_prolongation = dst_fine_host.l2_norm();
  const double norm_restriction  = dst_coarse_host.l2_norm();

  if (rank == 0)
    std::cout << "dim=" << dim << " degree=" << degree << " components=" << n_components
              << " mesh=" << std::setw(12) << std::left << to_string(mesh_type) << std::right
              << (std::is_same_v<Number, double> ? " double" : " float ")
              << "  n_dofs=" << std::setw(7) << dof_handler_fine.n_dofs() << "/" << std::setw(6)
              << dof_handler_coarse.n_dofs() << "  schemes=" << std::setw(8) << std::left
              << scheme_paths<dim>(transfer_copy) << std::right << std::scientific
              << std::setprecision(1) << " |P|,|R| " << norm_prolongation << " " << norm_restriction
              << "  installed P/R " << error_installed_p << " " << error_installed_r
              << "  copy P/R " << error_copy_p << " " << error_copy_r << "  "
              << (ok ? "OK" : "FAILED") << std::defaultfloat << std::endl;

  return ok;
}



template <int dim, typename Number>
unsigned int
run_all()
{
  unsigned int n_failures = 0;
  for (const MeshType mesh_type : {MeshType::uniform,
                                   MeshType::adaptive,
                                   MeshType::hanging,
                                   MeshType::dirichlet,
                                   MeshType::antiperiodic})
    for (const unsigned int n_components : {1u, static_cast<unsigned int>(dim)})
      for (unsigned int degree = 1; degree <= 4; ++degree)
        if (run<dim, Number>(degree, n_components, mesh_type) == false)
          ++n_failures;
  return n_failures;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  unsigned int n_failures = 0;
  n_failures += run_all<2, double>();
  n_failures += run_all<3, double>();
  n_failures += run_all<2, float>();
  n_failures += run_all<3, float>();

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << (n_failures == 0 ? "All configurations OK" :
                                    std::to_string(n_failures) + " configurations FAILED")
              << std::endl;

  return n_failures == 0 ? 0 : 1;
}
