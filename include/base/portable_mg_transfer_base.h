#ifndef portable_mg_transfer_base_h
#define portable_mg_transfer_base_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  class MGTransferBase : public EnableObserverPointer
  {
  public:
    ~MGTransferBase() = default;

    virtual void
    prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    virtual void
    restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src) const = 0;

    virtual void
    reinit(const MatrixFree<dim, number>   &mf_coarse,
           const MatrixFree<dim, number>   &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) = 0;
  };

  class PolynomialTransferDispatchFactory
  {
  public:
    static constexpr unsigned int max_degree = 9;

    template <typename Runner>
    static bool
    dispatch(const int runtime_p_coarse, const int runtime_p_fine, Runner &runner)
    {
      return recursive_dispatch<Runner, max_degree, max_degree>(runtime_p_coarse,
                                                                runtime_p_fine,
                                                                runner);
    }

  private:
    template <typename Runner, unsigned int degree_coarse, unsigned int degree_fine>
    static bool
    recursive_dispatch(const int runtime_p_coarse, const int runtime_p_fine, Runner &runner)
    {
      if (runtime_p_fine == degree_fine)
        {
          if (runtime_p_coarse == degree_coarse)
            {
              runner.template run<degree_coarse, degree_fine>();
              return true;
            }
          else if constexpr (degree_coarse > 1)
            {
              return recursive_dispatch<Runner, degree_coarse - 1, degree_fine>(runtime_p_coarse,
                                                                                runtime_p_fine,
                                                                                runner);
            }
          else
            {
              return false;
            }
        }
      else if constexpr (degree_fine > 1)
        {
          return recursive_dispatch<Runner, degree_fine - 2, degree_fine - 1>(runtime_p_coarse,
                                                                              runtime_p_fine,
                                                                              runner);
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
