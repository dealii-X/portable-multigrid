#ifndef portable_operator_info_base_h
#define portable_operator_info_base_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Without device-aware MPI, deal.II (9.8 and later) stages the exchange of
  // MemorySpace::Default vectors through a host buffer: compress_start()
  // snapshots the vector and compress_finish() writes the snapshot back, so
  // no kernel may add to it in between. The 3-color overlap loops of the
  // operators and transfers order their colors accordingly.
#ifdef DEAL_II_MPI_WITH_DEVICE_SUPPORT
  inline constexpr bool buffered_ghost_exchange = false;
#else
  inline constexpr bool buffered_ghost_exchange = true;
#endif

  // The DoF layout of an operator, as the multigrid transfers see it: the
  // DoFHandler, the colored list of locally owned cells the operator loops
  // over, the partitioner its vectors use, and whether the cell loop is
  // split into the 3 colors of the communication/computation overlap scheme.
  // Operators implement it; the transfers depend on nothing else.
  template <int dim>
  class OperatorInfoBase : public EnableObserverPointer
  {
  public:
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;
    using ColoredGraph = std::vector<std::vector<CellIterator>>;

    virtual ~OperatorInfoBase() = default;

    virtual const DoFHandler<dim> &
    get_dof_handler() const = 0;

    virtual const ColoredGraph &
    get_colored_graph() const = 0;

    // Null for a non-parallel triangulation, like deal.II's portable matrix-free class.
    virtual const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const = 0;

    virtual bool
    use_overlap_communication_computation() const = 0;
  };



  // Stand-alone implementation built from a bare DoFHandler: partitioner from
  // the locally owned and locally relevant DoFs (the same layout
  // deal.II's portable matrix-free class builds), one color with all locally owned active
  // cells in iteration order, no overlap.
  template <int dim>
  class DoFHandlerOperatorInfo : public OperatorInfoBase<dim>
  {
  public:
    using typename OperatorInfoBase<dim>::ColoredGraph;

    explicit DoFHandlerOperatorInfo(const DoFHandler<dim> &dof_handler);

    const DoFHandler<dim> &
    get_dof_handler() const override;

    const ColoredGraph &
    get_colored_graph() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    bool
    use_overlap_communication_computation() const override;

  private:
    ObserverPointer<const DoFHandler<dim>>             dof_handler;
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    ColoredGraph                                       colored_graph;
  };



  // Strips deal.II's FilteredIterator wrapper from a portable matrix-free
  // colored graph (std::vector<std::vector<CellFilter>>).
  template <int dim, typename FilteredCellIterator>
  typename OperatorInfoBase<dim>::ColoredGraph
  make_colored_graph(const std::vector<std::vector<FilteredCellIterator>> &filtered_graph)
  {
    typename OperatorInfoBase<dim>::ColoredGraph graph(filtered_graph.size());
    for (unsigned int color = 0; color < filtered_graph.size(); ++color)
      graph[color].assign(filtered_graph[color].begin(), filtered_graph[color].end());
    return graph;
  }



  // True when a vector's partitioner describes the same layout as an
  // operator's: the same object, or separately built but globally
  // compatible. For debug asserts; is_globally_compatible() is collective.
  inline bool
  partitioners_are_compatible(const std::shared_ptr<const Utilities::MPI::Partitioner> &a,
                              const std::shared_ptr<const Utilities::MPI::Partitioner> &b)
  {
    if (a == b)
      return true;
    if (a == nullptr || b == nullptr)
      return false;
    return a->is_globally_compatible(*b);
  }



  template <int dim>
  DoFHandlerOperatorInfo<dim>::DoFHandlerOperatorInfo(const DoFHandler<dim> &dof_handler)
    : dof_handler(&dof_handler)
  {
    if (const auto *parallel_triangulation =
          dynamic_cast<const parallel::TriangulationBase<dim> *>(
            &dof_handler.get_triangulation()))
      partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(dof_handler),
        parallel_triangulation->get_mpi_communicator());

    colored_graph.resize(1);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        colored_graph[0].push_back(cell);
  }

  template <int dim>
  const DoFHandler<dim> &
  DoFHandlerOperatorInfo<dim>::get_dof_handler() const
  {
    return *dof_handler;
  }

  template <int dim>
  const typename DoFHandlerOperatorInfo<dim>::ColoredGraph &
  DoFHandlerOperatorInfo<dim>::get_colored_graph() const
  {
    return colored_graph;
  }

  template <int dim>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  DoFHandlerOperatorInfo<dim>::get_vector_partitioner() const
  {
    return partitioner;
  }

  template <int dim>
  bool
  DoFHandlerOperatorInfo<dim>::use_overlap_communication_computation() const
  {
    return false;
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
