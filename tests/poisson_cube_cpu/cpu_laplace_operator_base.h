#ifndef cpu_laplace_operator_base_h
#define cpu_laplace_operator_base_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/memory_space.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <type_traits>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Operator interface for both backends: MemorySpace::Default vectors pair with
  // Portable::MatrixFree, MemorySpace::Host vectors with deal.II's CPU MatrixFree.
  template <int dim, typename VectorType>
  class LaplaceOperatorBase : public EnableObserverPointer
  {
  public:
    using value_type = typename VectorType::value_type;

    using MatrixFreeType =
      std::conditional_t<std::is_same_v<typename VectorType::memory_space, MemorySpace::Host>,
                         dealii::MatrixFree<dim, value_type>,
                         Portable::MatrixFree<dim, value_type>>;

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

    virtual const MatrixFreeType &
    get_matrix_free() const = 0;

    virtual const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const = 0;
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
