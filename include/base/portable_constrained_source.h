#ifndef portable_constrained_source_h
#define portable_constrained_source_h

#include <deal.II/base/index_set.h>
#include <deal.II/base/memory_space.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // Portable::FEEvaluation::read_dof_values() reads the source vector at
  // constrained DoFs (the CPU MatrixFree masks them since deal.II PR 14324).
  // An operator applied through the deal.II cell loop therefore has to see
  // zeros there: zero_constrained() saves and zeroes the locally owned
  // constrained entries of src before the loop (ghost copies follow through
  // the loop's own ghost exchange), restore_constrained() puts them back,
  // as MatrixFreeOperators::Base::vmult() does on the CPU.
  template <typename Number>
  class ConstrainedSource
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    template <typename ConstraintNumber>
    void
    reinit(const AffineConstraints<ConstraintNumber> &constraints,
           const IndexSet                            &locally_owned_dofs)
    {
      std::vector<unsigned int> local_indices;
      for (const auto i : locally_owned_dofs)
        if (constraints.is_constrained(i))
          local_indices.push_back(locally_owned_dofs.index_within_set(i));

      indices = Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("constrained_source_indices", Kokkos::WithoutInitializing),
        local_indices.size());
      values = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("constrained_source_values", Kokkos::WithoutInitializing),
        local_indices.size());

      Kokkos::View<unsigned int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host(local_indices.data(), local_indices.size());
      Kokkos::deep_copy(indices, host);
    }

    void
    zero_constrained(const VectorType &src) const
    {
      Number    *src_ptr = const_cast<VectorType &>(src).get_values();
      const auto idx     = indices;
      const auto saved   = values;
      Kokkos::parallel_for(
        "zero_constrained_source",
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0, idx.size()),
        KOKKOS_LAMBDA(const int k) {
          saved(k)        = src_ptr[idx(k)];
          src_ptr[idx(k)] = Number(0);
        });
    }

    void
    restore_constrained(const VectorType &src) const
    {
      Number    *src_ptr = const_cast<VectorType &>(src).get_values();
      const auto idx     = indices;
      const auto saved   = values;
      Kokkos::parallel_for(
        "restore_constrained_source",
        Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(0, idx.size()),
        KOKKOS_LAMBDA(const int k) { src_ptr[idx(k)] = saved(k); });
    }

  private:
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> indices;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>       values;
  };
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
