#ifndef shape_values_in_closure_h
#define shape_values_in_closure_h

#include <deal.II/base/config.h>

#include <Kokkos_Core.hpp>

#include <string>
#include <type_traits>

DEAL_II_NAMESPACE_OPEN

namespace Custom
{
  namespace Parallel
  {
    // 1D shape values (or a 1D transfer matrix) for the kernel closure
    // instead of a device view copied to the team scratch. On CUDA the
    // closure is a kernel argument, so the reads bypass shared memory.
    template <typename Number, int n_entries>
    struct ShapeValuesInClosure
    {
      Number values[n_entries];
    };

    namespace internal
    {
      template <typename ShapeValuesType>
      struct is_shape_values_in_closure : std::false_type
      {};

      template <typename Number, int n_entries>
      struct is_shape_values_in_closure<ShapeValuesInClosure<Number, n_entries>> : std::true_type
      {};

      // Shape values as a team reads them: a device view is copied to the
      // team scratch (the caller places the barrier), the closure is read
      // in place.
      template <typename ViewType, typename Number>
      KOKKOS_INLINE_FUNCTION const Number *
      team_shape_values(const ViewType &shape_values,
                        Number         *scratch,
                        const int       n_entries,
                        const int       thread_id,
                        const int       block_size)
      {
        for (int tid = thread_id; tid < n_entries; tid += block_size)
          scratch[tid] = shape_values[tid];
        return scratch;
      }

      template <typename Number, int n_entries>
      KOKKOS_INLINE_FUNCTION const Number *
      team_shape_values(const ShapeValuesInClosure<Number, n_entries> &shape_values,
                        Number *,
                        const int,
                        const int,
                        const int)
      {
        return shape_values.values;
      }

      // parallel_for of a team kernel. With the shape values in the closure
      // it stays a kernel argument (CUDA), also above the size Kokkos would
      // move to the constant buffer.
      template <typename ShapeValuesType, typename Policy, typename Functor>
      void
      launch_team_kernel(const std::string &label, const Policy &policy, const Functor &functor)
      {
        if constexpr (is_shape_values_in_closure<ShapeValuesType>::value)
        {
          // Kokkos passes the closure as a kernel argument only below 4 KB
          // (CudaTraits::KernelArgumentLimit, functor + policy), above it
          // silently from global memory.
          static_assert(sizeof(Functor) <= 3072,
                        "Kernel closure too large to stay a kernel argument; "
                        "use the shape values from a device view instead.");
          Kokkos::parallel_for(
            label,
            Kokkos::Experimental::require(policy,
                                          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
            functor);
        }
        else
          Kokkos::parallel_for(label, policy, functor);
      }
    } // namespace internal
  }   // namespace Parallel
} // namespace Custom

DEAL_II_NAMESPACE_CLOSE

#endif
