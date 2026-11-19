#ifndef portable_laplace_operator_base_h
#define portable_laplace_operator_base_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/memory_space.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "base/portable_operator_info_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Operator interface for both backends: VectorType is a distributed vector
  // in MemorySpace::Default (portable operators) or MemorySpace::Host (CPU
  // operators).
  template <int dim, typename VectorType>
  class LaplaceOperatorBase : public OperatorInfoBase<dim>
  {
  public:
    using value_type = typename VectorType::value_type;

    virtual ~LaplaceOperatorBase() = default;

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    vmult_dummy(VectorType       &dst,
                const VectorType &src,
                const bool        ghost_exchange_on,
                const bool        computation_on) const = 0;

    virtual void
    Tvmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    initialize_dof_vector(VectorType &vec) const = 0;

    virtual void
    compute_diagonal() = 0;

    virtual std::shared_ptr<DiagonalMatrix<VectorType>>
    get_matrix_diagonal_inverse() const = 0;

    virtual types::global_dof_index
    m() const = 0;

    virtual types::global_dof_index
    n() const = 0;

    virtual value_type
    el(const types::global_dof_index row, const types::global_dof_index col) const = 0;

    // get_dof_handler(), get_colored_graph(), get_vector_partitioner() and
    // use_overlap_communication_computation() are inherited from
    // OperatorInfoBase<dim>.
  };

  class OperatorDispatchFactory
  {
  public:
    static constexpr unsigned int max_degree = 9;

    template <typename OperatorRunner>
    static bool
    dispatch(const unsigned int runtime_degree, OperatorRunner &runner)
    {
      return recursive_dispatch<OperatorRunner, max_degree>(runtime_degree, runner);
    }

  private:
    template <typename OperatorRunner, unsigned int degree>
    static bool
    recursive_dispatch(const unsigned int runtime_degree, OperatorRunner &runner)
    {
      if (runtime_degree == degree)
        {
          runner.template run<degree>();
          return true;
        }
      else if constexpr (degree > 1)
        {
          return recursive_dispatch<OperatorRunner, degree - 1>(runtime_degree, runner);
        }
      else
        {
          return false;
        }
    }
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
