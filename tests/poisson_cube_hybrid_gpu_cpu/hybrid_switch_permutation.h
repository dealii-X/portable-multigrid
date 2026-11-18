#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>
// the library instantiates the exchange only for float/double; setup also
// exchanges DoF indices
#include <deal.II/base/mpi_noncontiguous_partitioner.templates.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/cell_id_translator.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>

#include "hybrid_cell_id_translator.h"
#include "hybrid_mpi_layout.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * P: moves switch-level values between the GPU side (owned by the GPU
   * ranks, GPU DoF numbering) and the CPU side (owned by all ranks, CPU DoF
   * numbering). Both sides hold the same Q1 space on the same mesh, so P is a
   * permutation of the DoFs and cpu_to_gpu() is the exact inverse of
   * gpu_to_cpu().
   *
   * Values are the locally owned entries of the level vectors, in the order
   * of locally_owned_dofs(). CPU-only ranks pass an empty GPU-side array.
   */
  template <int dim>
  class SwitchLevelPermutation
  {
  public:
    // Collective on comm_world. gpu_dofs is nullptr on CPU-only ranks.
    void
    reinit(const MPILayout &layout, const DoFHandler<dim> *gpu_dofs, const DoFHandler<dim> &cpu_dofs);

    // GPU -> CPU. Collective on comm_world.
    template <typename Number>
    void
    gpu_to_cpu(const ArrayView<const Number> &gpu_values, const ArrayView<Number> &cpu_values) const
    {
      AssertDimension(gpu_values.size(), n_gpu_values);
      AssertDimension(cpu_values.size(), n_cpu_values);
      partitioner.export_to_ghosted_array(gpu_values, cpu_values);
    }

    // CPU -> GPU, overwrites gpu_values. Collective on comm_world.
    // cpu_values is not modified (deal.II's interface takes it non-const).
    template <typename Number>
    void
    cpu_to_gpu(const ArrayView<Number> &cpu_values, const ArrayView<Number> &gpu_values) const
    {
      AssertDimension(gpu_values.size(), n_gpu_values);
      AssertDimension(cpu_values.size(), n_cpu_values);

      // deal.II only implements 'add' into a zeroed array; every GPU entry
      // gets exactly one value, so this overwrites
      std::fill(gpu_values.begin(), gpu_values.end(), Number(0));
      partitioner.import_from_ghosted_array(VectorOperation::add, cpu_values, gpu_values);
    }

  private:
    Utilities::MPI::NoncontiguousPartitioner partitioner;

    std::size_t n_gpu_values = 0;
    std::size_t n_cpu_values = 0;
  };



  namespace internal
  {
    // AssertThrow that fails on all ranks of comm if it fails on one
    inline void
    assert_on_all_ranks(const bool condition, const MPI_Comm comm, const std::string &message)
    {
      AssertThrow(Utilities::MPI::min(condition ? 1u : 0u, comm) == 1u, ExcMessage(message));
    }
  } // namespace internal



  template <int dim>
  void
  SwitchLevelPermutation<dim>::reinit(const MPILayout       &layout,
                                      const DoFHandler<dim> *gpu_dofs,
                                      const DoFHandler<dim> &cpu_dofs)
  {
    const MPI_Comm comm = layout.comm_world;

    internal::assert_on_all_ranks(layout.is_gpu_rank == (gpu_dofs != nullptr),
                                  comm,
                                  "gpu_dofs has to be given exactly on the GPU ranks.");

    // one integer per cell id, the same on both sides
    const auto translator = make_cell_id_translator(cpu_dofs.get_triangulation());
    const unsigned int dofs_per_cell = cpu_dofs.get_fe().n_dofs_per_cell();

    internal::assert_on_all_ranks(gpu_dofs == nullptr ||
                                    gpu_dofs->get_fe().n_dofs_per_cell() == dofs_per_cell,
                                  comm,
                                  "GPU and CPU switch levels use different elements.");
    internal::assert_on_all_ranks(translator.size() <
                                    std::numeric_limits<types::global_dof_index>::max() /
                                      dofs_per_cell,
                                  comm,
                                  "Too many cells for the cell-DoF slots in this index type.");

    // 1. Learn the GPU DoF index of every CPU-side cell DoF. Slot (cell, i) =
    //    cell id * dofs_per_cell + i. GPU ranks provide the GPU DoF index of
    //    the slots of their cells; every rank requests the slots of its
    //    CPU-side cells.
    std::vector<types::global_dof_index> provided_slots, gpu_dof_of_slot, requested_slots;
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    if (gpu_dofs != nullptr)
      for (const auto &cell : gpu_dofs->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const types::global_dof_index id = translator.translate(cell);
            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                provided_slots.push_back(id * dofs_per_cell + i);
                gpu_dof_of_slot.push_back(dof_indices[i]);
              }
          }

    for (const auto &cell : cpu_dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const types::global_dof_index id = translator.translate(cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            requested_slots.push_back(id * dofs_per_cell + i);
        }

    std::vector<types::global_dof_index> received(requested_slots.size());
    {
      const Utilities::MPI::NoncontiguousPartitioner slot_exchange(provided_slots,
                                                                   requested_slots,
                                                                   comm);
      slot_exchange.export_to_ghosted_array(
        ArrayView<const types::global_dof_index>(gpu_dof_of_slot),
        ArrayView<types::global_dof_index>(received));
    }

    // 2. GPU DoF index of each locally owned CPU DoF. Every owned DoF lies on
    //    an owned cell; cells sharing a DoF have to agree.
    const IndexSet &cpu_owned = cpu_dofs.locally_owned_dofs();

    std::vector<types::global_dof_index> gpu_dof_of_cpu_dof(cpu_owned.n_elements(),
                                                            numbers::invalid_dof_index);
    bool                                 consistent = true;
    unsigned int                         slot       = 0;

    for (const auto &cell : cpu_dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i, ++slot)
            if (cpu_owned.is_element(dof_indices[i]))
              {
                types::global_dof_index &gpu_dof =
                  gpu_dof_of_cpu_dof[cpu_owned.index_within_set(dof_indices[i])];
                consistent = consistent &&
                             (gpu_dof == numbers::invalid_dof_index || gpu_dof == received[slot]);
                gpu_dof = received[slot];
              }
        }

    internal::assert_on_all_ranks(consistent, comm, "Cells disagree on the GPU index of a DoF.");
    internal::assert_on_all_ranks(std::find(gpu_dof_of_cpu_dof.begin(),
                                            gpu_dof_of_cpu_dof.end(),
                                            numbers::invalid_dof_index) ==
                                    gpu_dof_of_cpu_dof.end(),
                                  comm,
                                  "Some locally owned CPU DoF is on no locally owned cell.");

    // 3. P itself: GPU ranks own their GPU DoFs in vector order, every rank
    //    requests the GPU DoFs of its CPU DoFs in vector order
    std::vector<types::global_dof_index> gpu_owned;
    if (gpu_dofs != nullptr)
      for (const types::global_dof_index i : gpu_dofs->locally_owned_dofs())
        gpu_owned.push_back(i);

    partitioner.reinit(gpu_owned, gpu_dof_of_cpu_dof, comm);

    n_gpu_values = gpu_owned.size();
    n_cpu_values = gpu_dof_of_cpu_dof.size();
  }



  // Collective on comm_world. Interpolates f(x) = x + sqrt(2) y + pi z on both
  // sides, moves the GPU-side values to the CPU side and back with P and
  // prints the maximal differences. The irrational coefficients make f differ
  // on any two vertices (by >~1e-3 here), so a misplaced DoF cannot hide;
  // a correct P leaves only roundoff of interpolate(). Returns the larger one.
  template <int dim>
  double
  check_switch_level_permutation(const MPILayout                   &layout,
                                 const SwitchLevelPermutation<dim> &permutation,
                                 const DoFHandler<dim>             *gpu_dofs,
                                 const DoFHandler<dim>             &cpu_dofs,
                                 std::ostream                      &out)
  {
    using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;

    const ScalarFunctionFromFunctionObject<dim> f([](const Point<dim> &p) {
      const double coefficients[3] = {1., std::sqrt(2.), numbers::PI};
      double       value           = 0;
      for (unsigned int d = 0; d < dim; ++d)
        value += coefficients[d] * p[d];
      return value;
    });

    // interpolate() writes all DoFs of the owned cells -> vectors with ghosts;
    // only their owned entries (stored first) are used below
    VectorType gpu_values; // empty on CPU-only ranks
    if (gpu_dofs != nullptr)
      {
        gpu_values.reinit(gpu_dofs->locally_owned_dofs(),
                          DoFTools::extract_locally_relevant_dofs(*gpu_dofs),
                          layout.comm_gpu);
        VectorTools::interpolate(*gpu_dofs, f, gpu_values);
      }

    VectorType cpu_expected(cpu_dofs.locally_owned_dofs(),
                            DoFTools::extract_locally_relevant_dofs(cpu_dofs),
                            layout.comm_world);
    VectorTools::interpolate(cpu_dofs, f, cpu_expected);

    const auto owned_view = [](VectorType &v) {
      return ArrayView<double>(v.begin(), v.locally_owned_size());
    };

    // down: GPU -> CPU, compare with the CPU-side interpolation
    VectorType cpu_values(cpu_dofs.locally_owned_dofs(), layout.comm_world);
    permutation.gpu_to_cpu(ArrayView<const double>(owned_view(gpu_values)), owned_view(cpu_values));

    double down_error = 0;
    for (unsigned int i = 0; i < cpu_values.locally_owned_size(); ++i)
      down_error =
        std::max(down_error, std::abs(cpu_values.local_element(i) - cpu_expected.local_element(i)));

    // up: CPU -> GPU, compare with the original GPU-side values
    VectorType gpu_back;
    if (gpu_dofs != nullptr)
      gpu_back.reinit(gpu_dofs->locally_owned_dofs(), layout.comm_gpu);
    permutation.cpu_to_gpu(owned_view(cpu_values), owned_view(gpu_back));

    double up_error = 0;
    for (unsigned int i = 0; i < gpu_back.locally_owned_size(); ++i)
      up_error =
        std::max(up_error, std::abs(gpu_back.local_element(i) - gpu_values.local_element(i)));

    down_error = Utilities::MPI::max(down_error, layout.comm_world);
    up_error   = Utilities::MPI::max(up_error, layout.comm_world);

    if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
      out << "P check, f = x + sqrt(2) y + pi z on " << cpu_dofs.n_dofs() << " DoFs:" << std::endl
          << "  GPU -> CPU max difference: " << down_error << std::endl
          << "  CPU -> GPU max difference: " << up_error << std::endl
          << std::endl;

    return std::max(down_error, up_error);
  }
} // namespace hybrid
