// -----------------------------------------------------------------------------
//
// Accessors for the private members of the host transfer
// dealii::MGTwoLevelTransfer<dim, Vector<Number, MemorySpace::Host>> that the
// portable transfer reads during its setup.
//
// deal.II grants this access by declaring dealii::Portable::internal::
// MGTwoLevelTransferImplementation a friend of the host transfer. The copy in
// namespace Copy is not that friend, and the installed library cannot be
// changed, so the member pointers are obtained through an explicit template
// instantiation: the usual access checking does not apply to the names used in
// an explicit instantiation ([temp.explicit]), which makes
// &MGTwoLevelTransfer::schemes a valid template argument there. The resulting
// member pointer is handed out by a friend function found through argument
// dependent lookup on a tag type. This is standard C++17 and changes nothing
// about the class layout, in contrast to redefining `private` as `public`.
//
// Only the (dim, Number) combinations instantiated at the end of this file can
// be used; a missing combination shows up as an undefined reference to
// private_member(...) at link time.
//
// -----------------------------------------------------------------------------

#ifndef multigrid_dealii_mg_transfer_private_access_h
#define multigrid_dealii_mg_transfer_private_access_h

#include <deal.II/base/config.h>

#include <deal.II/base/memory_space.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Copy
{
  namespace Portable
  {
    namespace internal
    {
      /**
       * The host transfer whose private members are accessed.
       */
      template <int dim, typename Number>
      using CpuTransfer =
        dealii::MGTwoLevelTransfer<dim,
                                   LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>>;

      /**
       * Explicitly instantiating this class with a pointer to a private member
       * defines the friend function private_member(Tag) that returns this
       * pointer.
       */
      template <typename Tag, auto MemberPointer>
      struct PrivateMemberAccess
      {
        friend auto
        private_member(Tag)
        {
          return MemberPointer;
        }
      };

// The friend declaration in the tag types below intentionally declares a
// non-template function; silence the corresponding warning of g++ and of the
// nvcc front end (1301-D).
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
#if defined(__NVCC__)
#  pragma nv_diag_suppress 1301
#endif

      // For every member: a tag type carrying the friend declaration and two
      // accessors (const and non-const) that dereference the member pointer.
#define DEALII_COPY_CPU_TRANSFER_ACCESSOR(member)                            \
  template <int dim, typename Number>                                        \
  struct CpuTransferTag_##member                                             \
  {                                                                          \
    friend auto private_member(CpuTransferTag_##member);                     \
  };                                                                         \
                                                                             \
  template <int dim, typename Number>                                        \
  inline auto &cpu_##member(CpuTransfer<dim, Number> &transfer)              \
  {                                                                          \
    return transfer.*private_member(CpuTransferTag_##member<dim, Number>{}); \
  }                                                                          \
                                                                             \
  template <int dim, typename Number>                                        \
  inline const auto &cpu_##member(const CpuTransfer<dim, Number> &transfer)  \
  {                                                                          \
    return transfer.*private_member(CpuTransferTag_##member<dim, Number>{}); \
  }

      DEALII_COPY_CPU_TRANSFER_ACCESSOR(schemes)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(n_components)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(constraint_info_coarse)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(constraint_info_fine)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(weights)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(weights_start)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(weights_are_compressed)
      // the following three are protected members of the base class
      // dealii::internal::MGTwoLevelTransferCore
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(fine_element_is_continuous)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(partitioner_coarse)
      DEALII_COPY_CPU_TRANSFER_ACCESSOR(partitioner_fine)

#undef DEALII_COPY_CPU_TRANSFER_ACCESSOR

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
#if defined(__NVCC__)
#  pragma nv_diag_default 1301
#endif

      // Explicit instantiations: this is where the access check is bypassed.
#define DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, member)    \
  template struct PrivateMemberAccess<CpuTransferTag_##member<dim, Number>, \
                                      &CpuTransfer<dim, Number>::member>;

#define DEALII_COPY_CPU_TRANSFER_INSTANTIATE(dim, Number)                              \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, schemes)                    \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, n_components)               \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, constraint_info_coarse)     \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, constraint_info_fine)       \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, weights)                    \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, weights_start)              \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, weights_are_compressed)     \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, fine_element_is_continuous) \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, partitioner_coarse)         \
  DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER(dim, Number, partitioner_fine)

      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(1, float)
      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(2, float)
      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(3, float)
      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(1, double)
      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(2, double)
      DEALII_COPY_CPU_TRANSFER_INSTANTIATE(3, double)

#undef DEALII_COPY_CPU_TRANSFER_INSTANTIATE
#undef DEALII_COPY_CPU_TRANSFER_INSTANTIATE_MEMBER

    } // namespace internal
  } // namespace Portable
} // namespace Copy

DEAL_II_NAMESPACE_CLOSE

#endif
