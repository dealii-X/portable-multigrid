// Copy::Portable::MGTwoLevelTransfer (multigrid_dealii/, batched kernels):
// prolongate_and_add() and restrict_and_add() against an independent
// reference on serial copies of both meshes,
//
//   P = Z_f I C_c,   R = P^T,
//
// C_c: coarse constraints applied to the coarse vector (constrained entries
// ignored), I: evaluation of the coarse FE function at the fine support points
// (FE shape values on the coarse cell), Z_f: zero at the fine constrained DoFs.
// DoFs of the distributed and the serial DoFHandlers are matched by support
// point and component; the input vectors are a hash of both.
//
// dim = 2, 3; FE_Q degree 1..4, scalar and dim components; double and float;
// meshes: uniform (two refinement levels), adaptive (hanging nodes on the fine
// level), hanging (also on the coarse level), dirichlet (homogeneous Dirichlet
// on both levels), antiperiodic (single-entry constraints with weight -1, the
// CRS path); 1, 3, 7 and the automatic number of coarse cells per batch.

#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <array>
#include <cmath>
#include <map>
#include <tuple>

#include "../tests.h"

#include "multigrid_dealii/portable_mg_transfer_global_coarsening.templates.h"

enum class MeshType
{
  uniform,
  adaptive,
  hanging,
  dirichlet,
  antiperiodic
};

const char *
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

// geometric refinement criteria only, so that the serial and the distributed
// meshes are the same
template <int dim>
void
refine_where(Triangulation<dim> &tria, const unsigned int direction)
{
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[direction] < 0.5)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
}

template <int dim>
void
create_mesh(const MeshType type, const bool fine, Triangulation<dim> &tria)
{
  GridGenerator::hyper_cube(tria, 0., 1., type == MeshType::antiperiodic);
  if (type == MeshType::antiperiodic)
    {
      std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> pairs;
      GridTools::collect_periodic_faces(tria, 0, 1, 0, pairs);
      tria.add_periodicity(pairs);
    }

  tria.refine_global(1);
  if (type == MeshType::adaptive)
    {
      if (fine)
        refine_where(tria, 0);
    }
  else if (type == MeshType::hanging)
    {
      refine_where(tria, 0);
      if (fine)
        refine_where(tria, 1);
    }
  else if (fine)
    tria.refine_global(1);
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

// support point (rounded) and component of a DoF
template <int dim>
struct DoFKey
{
  std::array<long long, dim> first;
  unsigned int               second;
  // a hanging DoF can share its support point with an unconstrained one
  bool constrained;

  bool
  operator<(const DoFKey &other) const
  {
    return std::tie(first, second, constrained) <
           std::tie(other.first, other.second, other.constrained);
  }
};

template <int dim>
DoFKey<dim>
make_key(const Point<dim> &p, const unsigned int component, const bool constrained)
{
  DoFKey<dim> key;
  for (unsigned int d = 0; d < dim; ++d)
    key.first[d] = std::llround(p[d] * 1e8);
  key.second      = component;
  key.constrained = constrained;
  return key;
}

// pseudo-random value in [-1, 1) of a DoF, the same in every numbering
template <int dim>
double
hash_value(const DoFKey<dim> &key)
{
  double arg = 4.581 * key.second;
  for (unsigned int d = 0; d < dim; ++d)
    arg += (d == 0 ? 12.9898 : d == 1 ? 78.233 : 37.719) * (key.first[d] * 1e-8);
  const double s = std::sin(arg) * 43758.5453;
  return 2. * (s - std::floor(s)) - 1.;
}

// keys of the DoFs on locally owned cells
template <int dim, typename Number>
std::map<types::global_dof_index, DoFKey<dim>>
dof_keys(const DoFHandler<dim> &dof_handler, const AffineConstraints<Number> &constraints)
{
  const auto                                    &fe = dof_handler.get_fe();
  FEValues<dim>                                  fe_values(fe,
                          Quadrature<dim>(fe.get_unit_support_points()),
                          update_quadrature_points);
  std::vector<types::global_dof_index>           dof_indices(fe.n_dofs_per_cell());
  std::map<types::global_dof_index, DoFKey<dim>> keys;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
          keys[dof_indices[k]] = make_key(fe_values.quadrature_point(k),
                                          fe.system_to_component_index(k).first,
                                          constraints.is_constrained(dof_indices[k]));
      }
  return keys;
}

// reference prolongation of the hashed coarse vector and restriction of the
// hashed fine vector, by key
template <int dim>
std::pair<std::map<DoFKey<dim>, double>, std::map<DoFKey<dim>, double>>
reference_transfer(const MeshType type, const FiniteElement<dim> &fe)
{
  // the 2:1 balance across vertices of p4est
  Triangulation<dim> tria_coarse(Triangulation<dim>::limit_level_difference_at_vertices),
    tria_fine(Triangulation<dim>::limit_level_difference_at_vertices);
  create_mesh(type, false, tria_coarse);
  create_mesh(type, true, tria_fine);
  DoFHandler<dim> dof_coarse(tria_coarse), dof_fine(tria_fine);
  dof_coarse.distribute_dofs(fe);
  dof_fine.distribute_dofs(fe);
  AffineConstraints<double> constraints_coarse, constraints_fine;
  make_constraints(type, dof_coarse, constraints_coarse);
  make_constraints(type, dof_fine, constraints_fine);

  const auto keys_coarse = dof_keys(dof_coarse, constraints_coarse);
  const auto keys_fine   = dof_keys(dof_fine, constraints_fine);

  // C_c u_c and Z_f v_f
  Vector<double> u_coarse(dof_coarse.n_dofs()), v_fine(dof_fine.n_dofs());
  for (const auto &[i, key] : keys_coarse)
    u_coarse[i] = hash_value(key);
  constraints_coarse.distribute(u_coarse);
  for (const auto &[i, key] : keys_fine)
    v_fine[i] = constraints_fine.is_constrained(i) ? 0. : hash_value(key);

  Vector<double>    prolongation(dof_fine.n_dofs()), restriction(dof_coarse.n_dofs());
  std::vector<bool> visited(dof_fine.n_dofs(), false);

  const MappingQ1<dim>                 mapping;
  FEValues<dim>                        fe_values(fe,
                          Quadrature<dim>(fe.get_unit_support_points()),
                          update_quadrature_points);
  std::vector<types::global_dof_index> dofs_fine(fe.n_dofs_per_cell()),
    dofs_coarse(fe.n_dofs_per_cell());
  for (const auto &cell : dof_fine.active_cell_iterators())
    {
      // the coarse cell that contains this fine cell
      const auto coarse_cell =
        GridTools::find_active_cell_around_point(mapping, dof_coarse, cell->center()).first;
      coarse_cell->get_dof_indices(dofs_coarse);
      fe_values.reinit(cell);
      cell->get_dof_indices(dofs_fine);
      for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
        {
          const auto i = dofs_fine[k];
          if (visited[i])
            continue;
          visited[i] = true;

          const unsigned int c = fe.system_to_component_index(k).first;
          const Point<dim>   unit_point =
            mapping.transform_real_to_unit_cell(coarse_cell, fe_values.quadrature_point(k));
          for (unsigned int m = 0; m < fe.n_dofs_per_cell(); ++m)
            if (fe.system_to_component_index(m).first == c)
              {
                const double phi = fe.shape_value(m, unit_point);
                prolongation[i] += phi * u_coarse[dofs_coarse[m]];
                restriction[dofs_coarse[m]] += phi * v_fine[i];
              }
        }
    }
  for (const auto &[i, key] : keys_fine)
    if (constraints_fine.is_constrained(i))
      prolongation[i] = 0.;
  constraints_coarse.condense(restriction);

  std::map<DoFKey<dim>, double> ref_prolongation, ref_restriction;
  for (const auto &[i, key] : keys_fine)
    ref_prolongation[key] = prolongation[i];
  for (const auto &[i, key] : keys_coarse)
    ref_restriction[key] = restriction[i];
  return {ref_prolongation, ref_restriction};
}

template <typename Number>
using DeviceVector = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

template <int dim, typename Number>
void
fill_hashed(DeviceVector<Number>                                 &vec,
            const DoFHandler<dim>                                &dof_handler,
            const std::map<types::global_dof_index, DoFKey<dim>> &keys)
{
  vec.reinit(dof_handler.locally_owned_dofs(),
             DoFTools::extract_locally_relevant_dofs(dof_handler),
             dof_handler.get_mpi_communicator());
  const IndexSet                        &owned = dof_handler.locally_owned_dofs();
  LinearAlgebra::ReadWriteVector<Number> rw(owned);
  for (const auto &[i, key] : keys)
    if (owned.is_element(i))
      rw(i) = hash_value(key);
  vec.import_elements(rw, VectorOperation::insert);
}

// max |result - reference| / max |reference| over the locally owned DoFs
template <int dim, typename Number>
double
relative_error(const DeviceVector<Number>                           &result,
               const DoFHandler<dim>                                &dof_handler,
               const std::map<types::global_dof_index, DoFKey<dim>> &keys,
               const std::map<DoFKey<dim>, double>                  &reference)
{
  const IndexSet                        &owned = dof_handler.locally_owned_dofs();
  LinearAlgebra::ReadWriteVector<Number> rw(owned);
  rw.import_elements(result, VectorOperation::insert);

  double max_diff = 0., max_ref = 0.;
  for (const auto &[i, key] : keys)
    if (owned.is_element(i))
      {
        const double ref = reference.at(key);
        max_diff         = std::max(max_diff, std::abs(rw(i) - ref));
        max_ref          = std::max(max_ref, std::abs(ref));
      }
  const MPI_Comm comm = dof_handler.get_mpi_communicator();
  return Utilities::MPI::max(max_diff, comm) / Utilities::MPI::max(max_ref, comm);
}

template <int dim, typename Number>
void
test(const unsigned int degree, const unsigned int n_components, const MeshType type)
{
  const FESystem<dim> fe(FE_Q<dim>(degree), n_components);

  parallel::distributed::Triangulation<dim> tria_coarse(MPI_COMM_WORLD), tria_fine(MPI_COMM_WORLD);
  create_mesh(type, false, tria_coarse);
  create_mesh(type, true, tria_fine);
  DoFHandler<dim> dof_coarse(tria_coarse), dof_fine(tria_fine);
  dof_coarse.distribute_dofs(fe);
  dof_fine.distribute_dofs(fe);
  AffineConstraints<Number> constraints_coarse, constraints_fine;
  make_constraints(type, dof_coarse, constraints_coarse);
  make_constraints(type, dof_fine, constraints_fine);

  Copy::Portable::MGTwoLevelTransfer<dim, DeviceVector<Number>> transfer;
  transfer.reinit_geometric_transfer(dof_fine, dof_coarse, constraints_fine, constraints_coarse);

  const auto [ref_prolongation, ref_restriction] = reference_transfer(type, fe);
  const auto keys_coarse                         = dof_keys(dof_coarse, constraints_coarse);
  const auto keys_fine                           = dof_keys(dof_fine, constraints_fine);

  DeviceVector<Number> src_coarse, src_fine, dst_coarse, dst_fine;
  fill_hashed(src_coarse, dof_coarse, keys_coarse);
  fill_hashed(src_fine, dof_fine, keys_fine);
  dst_coarse.reinit(src_coarse);
  dst_fine.reinit(src_fine);

  double error = 0.;
  for (const unsigned int cells_per_batch : {numbers::invalid_unsigned_int, 1u, 3u, 7u})
    {
      transfer.set_cells_per_batch(cells_per_batch);
      dst_fine = 0.;
      transfer.prolongate_and_add(dst_fine, src_coarse);
      error = std::max(error, relative_error<dim>(dst_fine, dof_fine, keys_fine, ref_prolongation));
      dst_coarse = 0.;
      transfer.restrict_and_add(dst_coarse, src_fine);
      error =
        std::max(error, relative_error<dim>(dst_coarse, dof_coarse, keys_coarse, ref_restriction));
    }

  const double tolerance = std::is_same_v<Number, double> ? 1e-12 : 1e-5;
  deallog << "dim=" << dim << " degree=" << degree << " components=" << n_components
          << " mesh=" << to_string(type) << (std::is_same_v<Number, double> ? " double" : " float")
          << " n_dofs=" << dof_fine.n_dofs() << "/" << dof_coarse.n_dofs() << ": "
          << (error < tolerance ? std::string("OK") : "FAILED, error " + std::to_string(error))
          << std::endl;
}

template <int dim, typename Number>
void
test_all()
{
  for (const MeshType type : {MeshType::uniform,
                              MeshType::adaptive,
                              MeshType::hanging,
                              MeshType::dirichlet,
                              MeshType::antiperiodic})
    for (const unsigned int n_components : {1u, static_cast<unsigned int>(dim)})
      for (unsigned int degree = 1; degree <= 4; ++degree)
        if (!(dim == 3 && std::is_same_v<Number, float> && degree == 3)) // run time
          test<dim, Number>(degree, n_components, type);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test_all<2, double>();
  test_all<3, double>();
  test_all<2, float>();
  test_all<3, float>();
}
