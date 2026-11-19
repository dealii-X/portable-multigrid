#ifndef portable_mapping_data_accessor_h
#define portable_mapping_data_accessor_h

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // (cell, q)-indexed access to the inverse Jacobians and JxW values of
  // Portable::MatrixFree::PrecomputedData, hiding the storage layout:
  // deal.II < 9.9 stores one entry per quadrature point, indexed (q, cell);
  // deal.II >= 9.9 stores them compressed, addressed through
  // data_index_offsets(cell): a single entry for a Cartesian/affine cell
  // (JxW is then det(J), to be multiplied by the quadrature weight) and one
  // per quadrature point for a general cell. @p data is either the
  // PrecomputedData itself or a MappingDataAccessor (same member names).
  template <typename Data>
  DEAL_II_HOST_DEVICE inline auto
  inverse_jacobian(const Data        &data,
                   const unsigned int cell,
                   const unsigned int q,
                   const unsigned int d1,
                   const unsigned int d2)
  {
#if DEAL_II_VERSION_GTE(9, 9, 0)
    const unsigned int index =
      data.data_index_offsets(cell) +
      ((data.cell_type(cell) == ::dealii::internal::MatrixFreeFunctions::general) ? q : 0);
    return data.inv_jacobian(index, d1, d2);
#else
    return data.inv_jacobian(q, cell, d1, d2);
#endif
  }

  template <typename Data>
  DEAL_II_HOST_DEVICE inline auto
  JxW_value(const Data &data, const unsigned int cell, const unsigned int q)
  {
#if DEAL_II_VERSION_GTE(9, 9, 0)
    const unsigned int offset = data.data_index_offsets(cell);
    return (data.cell_type(cell) == ::dealii::internal::MatrixFreeFunctions::general) ?
             data.JxW(offset + q) :
             data.JxW(offset) * data.q_weights(q);
#else
    return data.JxW(q, cell);
#endif
  }

  namespace internal
  {
    template <typename Data, typename = void>
    struct has_first_cell : std::false_type
    {};

    template <typename Data>
    struct has_first_cell<Data, std::void_t<decltype(std::declval<Data>().first_cell)>>
      : std::true_type
    {};
  } // namespace internal

  // Process-local index of the first cell of the color @p data describes.
  // deal.II master replaced row_start/padding_length by first_cell within the
  // 9.9.0-pre series, so this is detected rather than version-switched.
  template <typename Data>
  DEAL_II_HOST_DEVICE inline unsigned int
  first_cell_index(const Data &data)
  {
    if constexpr (internal::has_first_cell<Data>::value)
      return data.first_cell;
    else
      return data.row_start / data.padding_length;
  }

  // Shallow copies of the views the functions above read, for kernels that
  // receive the mapping data separately from the rest of PrecomputedData.
  template <int dim, typename Number>
  struct MappingDataAccessor
  {
    using PrecomputedData = typename MatrixFree<dim, Number>::PrecomputedData;

    MappingDataAccessor() = default;

    MappingDataAccessor(const PrecomputedData &data)
      : inv_jacobian(data.inv_jacobian)
      , JxW(data.JxW)
#if DEAL_II_VERSION_GTE(9, 9, 0)
      , cell_type(data.cell_type)
      , data_index_offsets(data.data_index_offsets)
      , q_weights(data.q_weights)
#endif
    {}

    DEAL_II_HOST_DEVICE Number
    inverse_jacobian(const unsigned int cell,
                     const unsigned int q,
                     const unsigned int d1,
                     const unsigned int d2) const
    {
      return Portable::inverse_jacobian(*this, cell, q, d1, d2);
    }

    DEAL_II_HOST_DEVICE Number
    JxW_value(const unsigned int cell, const unsigned int q) const
    {
      return Portable::JxW_value(*this, cell, q);
    }

    decltype(PrecomputedData::inv_jacobian) inv_jacobian;
    decltype(PrecomputedData::JxW)          JxW;
#if DEAL_II_VERSION_GTE(9, 9, 0)
    decltype(PrecomputedData::cell_type)          cell_type;
    decltype(PrecomputedData::data_index_offsets) data_index_offsets;
    decltype(PrecomputedData::q_weights)          q_weights;
#endif
  };
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
