#ifndef portable_shape_info_h
#define portable_shape_info_h

#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>


DEAL_II_NAMESPACE_OPEN


template <typename Number>
using DeviceVector = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

namespace Portable
{
  namespace internal
  {

    template <typename Number>
    struct UnivariateShapeData
    {
      UnivariateShapeData();

      void
      reinit(const ::dealii::internal::MatrixFreeFunctions::UnivariateShapeData<Number>
               &univariate_shape_data_cpu);

      /**
       * Encodes the type of element detected at construction. FEEvaluation
       * will select the most efficient algorithm based on the given element
       * type.
       */
      ::dealii::internal::MatrixFreeFunctions::ElementType element_type;

      /**
       * Stores the shape values of the 1d finite element evaluated at all 1d
       * quadrature points. The length of
       * this array is <tt>n_dofs_1d * n_q_points_1d</tt> and quadrature
       * points are the index running fastest.
       */
      DeviceVector<Number> shape_values;

      /**
       * Stores the shape gradients of the 1d finite element evaluated at all
       * 1d quadrature points. The length of
       * this array is <tt>n_dofs_1d * n_q_points_1d</tt> and quadrature
       * points are the index running fastest.
       */
      DeviceVector<Number> shape_gradients;

      /**
       * Stores the shape Hessians of the 1d finite element evaluated at all
       * 1d quadrature points. The length of
       * this array is <tt>n_dofs_1d * n_q_points_1d</tt> and quadrature
       * points are the index running fastest.
       */
      DeviceVector<Number> shape_hessians;

      /**
       * Stores the shape gradients of the shape function space associated to
       * the quadrature (collocation), given by FE_DGQ<1>(Quadrature<1>).
       */
      DeviceVector<Number> shape_gradients_collocation;

      /**
       * Stores the shape hessians of the shape function space associated to
       * the quadrature (collocation), given by FE_DGQ<1>(Quadrature<1>).
       */
      DeviceVector<Number> shape_hessians_collocation;

      /**
       * Stores the shape values in a different format, namely the so-called
       * even-odd scheme where the symmetries in shape_values are used for
       * faster evaluation.
       */
      DeviceVector<Number> shape_values_eo;

      /**
       * Stores the shape gradients in a different format, namely the
       * so-called even-odd scheme where the symmetries in shape_gradients are
       * used for faster evaluation.
       */
      DeviceVector<Number> shape_gradients_eo;

      /**
       * Stores the shape second derivatives in a different format, namely the
       * so-called even-odd scheme where the symmetries in shape_hessians are
       * used for faster evaluation.
       */
      DeviceVector<Number> shape_hessians_eo;

      /**
       * Stores the shape gradients of the shape function space associated to
       * the quadrature (collocation), given by FE_DGQ<1>(Quadrature<1>). This
       * array provides an alternative representation of the
       * shape_gradients_collocation field in the even-odd format.
       */
      DeviceVector<Number> shape_gradients_collocation_eo;

      /**
       * Stores the shape hessians of the shape function space associated to
       * the quadrature (collocation), given by FE_DGQ<1>(Quadrature<1>). This
       * array provides an alternative representation of the
       * shape_hessians_collocation field in the even-odd format.
       */
      DeviceVector<Number> shape_hessians_collocation_eo;

      /**
       * Stores the inverse transformation from the data at quadrature points
       * to the basis defined by the shape_values fields. The data at
       * quadrature points is interpreted either implicitly by its polynomial
       * interpolation, or explicitly in terms of separate polynomials such as
       * with the `_collocation` fields. The size of the array equals the
       * layout of the `shape_values` array, and it is combined with the shape
       * values array such that this matrix is the pseudo inverse of
       * shape_values. In case the number of 1d quadrature points equals the
       * size of the basis, this array is exactly the inverse of the
       * shape_values array. The length of this array is <tt>n_dofs_1d *
       * n_q_points_1d</tt> and quadrature points are the index running
       * fastest.
       */
      DeviceVector<Number> inverse_shape_values;

      /**
       * Stores the even-odd variant of the `inverse_shape_values` field.
       */
      DeviceVector<Number> inverse_shape_values_eo;

      /**
       * Collects all data of 1d shape values evaluated at the point 0 and 1
       * (the vertices) in one data structure. The sorting of data is to
       * start with the values, then gradients, then second derivatives.
       */
      Kokkos::Array<DeviceVector<Number>, 2> shape_data_on_face;

      /**
       * Collects all data of 1d nodal shape values (defined by the Lagrange
       * polynomials in the points of the quadrature rule) evaluated at the
       * point 0 and 1 (the vertices) in one data structure.
       *
       * This data structure can be used to interpolate from the cell to the
       * face quadrature points. The sorting of data is to start with the
       * values, then gradients, then second derivatives.
       */
      Kokkos::Array<DeviceVector<Number>, 2> quadrature_data_on_face;

      /**
       * Stores one-dimensional values of shape functions on subface. Since
       * there are two subfaces, store two variants.
       */
      Kokkos::Array<DeviceVector<Number>, 2> values_within_subface;

      /**
       * Stores one-dimensional gradients of shape functions on subface. Since
       * there are two subfaces, store two variants.
       */
      Kokkos::Array<DeviceVector<Number>, 2> gradients_within_subface;

      /**
       * Stores one-dimensional gradients of shape functions on subface. Since
       * there are two subfaces, store two variants.
       */
      Kokkos::Array<DeviceVector<Number>, 2> hessians_within_subface;

      /**
       * A 1d subface interpolation matrices to the first and second quadrant.
       * This data structure is only set up for FE_Q for dim > 1.
       */
      Kokkos::Array<DeviceVector<Number>, 2> subface_interpolation_matrices;

      /**
       * Same as above but stored in a scalar format independent of the type of
       * Number
       */
      Kokkos::
        Array<DeviceVector<typename dealii::internal::VectorizedArrayTrait<Number>::value_type>, 2>
          subface_interpolation_matrices_scalar;


      /**
       * We store a copy of the one-dimensional quadrature formula
       * used for initialization.
       */
      Quadrature<1> quadrature;

      /**
       * Stores the degree of the element.
       */
      unsigned int fe_degree;

      /**
       * Stores the number of quadrature points per dimension.
       */
      unsigned int n_q_points_1d;

      /**
       * Indicates whether the basis functions are nodal in 0 and 1, i.e., the
       * end points of the unit cell.
       */
      bool nodal_at_cell_boundaries;
    };


    template <typename Number>
    UnivariateShapeData<Number>::UnivariateShapeData()
      : element_type(::dealii::internal::MatrixFreeFunctions::ElementType::tensor_general)
      , fe_degree(0)
      , n_q_points_1d(0)
      , nodal_at_cell_boundaries(false)
    {}

    template <typename Number>
    void
    UnivariateShapeData<Number>::reinit(
      const ::dealii::internal::MatrixFreeFunctions::UnivariateShapeData<Number>
        &univariate_shape_data_cpu)
    {
      const auto &data_cpu = univariate_shape_data_cpu;

      element_type             = data_cpu.element_type;
      quadrature               = data_cpu.quadrature;
      fe_degree                = data_cpu.fe_degree;
      n_q_points_1d            = data_cpu.n_q_points_1d;
      nodal_at_cell_boundaries = data_cpu.nodal_at_cell_boundaries;

      ::dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;


      if (data_cpu.shape_values.size() > 0)
        {
          shape_values =
            DeviceVector<Number>(Kokkos::view_alloc("shape_values", Kokkos::WithoutInitializing),
                                 data_cpu.shape_values.size());
          Kokkos::deep_copy(exec_space,
                            shape_values,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_values.data(), data_cpu.shape_values.size()));
        }

      if (data_cpu.shape_gradients.size() > 0)
        {
          shape_gradients =
            DeviceVector<Number>(Kokkos::view_alloc("shape_gradients", Kokkos::WithoutInitializing),
                                 data_cpu.shape_gradients.size());
          Kokkos::deep_copy(exec_space,
                            shape_gradients,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_gradients.data(), data_cpu.shape_gradients.size()));
        }
      if (data_cpu.shape_hessians.size() > 0)
        {
          shape_hessians =
            DeviceVector<Number>(Kokkos::view_alloc("shape_hessians", Kokkos::WithoutInitializing),
                                 data_cpu.shape_hessians.size());
          Kokkos::deep_copy(exec_space,
                            shape_hessians,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_hessians.data(), data_cpu.shape_hessians.size()));
        }
      if (data_cpu.shape_gradients_collocation.size() > 0)
        {
          shape_gradients_collocation =
            DeviceVector<Number>(Kokkos::view_alloc("shape_gradients_collocation",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.shape_gradients_collocation.size());
          Kokkos::deep_copy(exec_space,
                            shape_gradients_collocation,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_gradients_collocation.data(),
                              data_cpu.shape_gradients_collocation.size()));
        }
      if (data_cpu.shape_hessians_collocation.size() > 0)
        {
          shape_hessians_collocation =
            DeviceVector<Number>(Kokkos::view_alloc("shape_hessians_collocation",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.shape_hessians_collocation.size());
          Kokkos::deep_copy(exec_space,
                            shape_hessians_collocation,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_hessians_collocation.data(),
                              data_cpu.shape_hessians_collocation.size()));
        }
      if (data_cpu.shape_values_eo.size() > 0)
        {
          shape_values_eo =
            DeviceVector<Number>(Kokkos::view_alloc("shape_values_eo", Kokkos::WithoutInitializing),
                                 data_cpu.shape_values_eo.size());
          Kokkos::deep_copy(exec_space,
                            shape_values_eo,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_values_eo.data(), data_cpu.shape_values_eo.size()));
        }
      if (data_cpu.shape_gradients_eo.size() > 0)
        {
          shape_gradients_eo = DeviceVector<Number>(Kokkos::view_alloc("shape_gradients_eo",
                                                                       Kokkos::WithoutInitializing),
                                                    data_cpu.shape_gradients_eo.size());
          Kokkos::deep_copy(
            exec_space,
            shape_gradients_eo,
            Kokkos::View<const Number *, Kokkos::HostSpace>(data_cpu.shape_gradients_eo.data(),
                                                            data_cpu.shape_gradients_eo.size()));
        }
      if (data_cpu.shape_hessians_eo.size() > 0)
        {
          shape_hessians_eo = DeviceVector<Number>(Kokkos::view_alloc("shape_hessians_eo",
                                                                      Kokkos::WithoutInitializing),
                                                   data_cpu.shape_hessians_eo.size());
          Kokkos::deep_copy(
            exec_space,
            shape_hessians_eo,
            Kokkos::View<const Number *, Kokkos::HostSpace>(data_cpu.shape_hessians_eo.data(),
                                                            data_cpu.shape_hessians_eo.size()));
        }
      if (data_cpu.shape_gradients_collocation_eo.size() > 0)
        {
          shape_gradients_collocation_eo =
            DeviceVector<Number>(Kokkos::view_alloc("shape_gradients_collocation_eo",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.shape_gradients_collocation_eo.size());
          Kokkos::deep_copy(exec_space,
                            shape_gradients_collocation_eo,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_gradients_collocation_eo.data(),
                              data_cpu.shape_gradients_collocation_eo.size()));
        }

      if (data_cpu.shape_hessians_collocation_eo.size() > 0)
        {
          shape_hessians_collocation_eo =
            DeviceVector<Number>(Kokkos::view_alloc("shape_hessians_collocation_eo",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.shape_hessians_collocation_eo.size());
          Kokkos::deep_copy(exec_space,
                            shape_hessians_collocation_eo,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.shape_hessians_collocation_eo.data(),
                              data_cpu.shape_hessians_collocation_eo.size()));
        }


      if (data_cpu.inverse_shape_values.size() > 0)
        {
          inverse_shape_values =
            DeviceVector<Number>(Kokkos::view_alloc("inverse_shape_values",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.inverse_shape_values.size());
          Kokkos::deep_copy(
            exec_space,
            inverse_shape_values,
            Kokkos::View<const Number *, Kokkos::HostSpace>(data_cpu.inverse_shape_values.data(),
                                                            data_cpu.inverse_shape_values.size()));
        }

      if (data_cpu.inverse_shape_values_eo.size() > 0)
        {
          inverse_shape_values_eo =
            DeviceVector<Number>(Kokkos::view_alloc("inverse_shape_values_eo",
                                                    Kokkos::WithoutInitializing),
                                 data_cpu.inverse_shape_values_eo.size());
          Kokkos::deep_copy(exec_space,
                            inverse_shape_values_eo,
                            Kokkos::View<const Number *, Kokkos::HostSpace>(
                              data_cpu.inverse_shape_values_eo.data(),
                              data_cpu.inverse_shape_values_eo.size()));
        }

      for (unsigned int c = 0; c < 2; ++c)
        {
          if (data_cpu.shape_data_on_face[c].size() > 0)
            {
              shape_data_on_face[c] =
                DeviceVector<Number>(Kokkos::view_alloc("shape_data_on_face_" + std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.shape_data_on_face[c].size());
              Kokkos::deep_copy(exec_space,
                                shape_data_on_face[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.shape_data_on_face[c].data(),
                                  data_cpu.shape_data_on_face[c].size()));
            }

          if (data_cpu.quadrature_data_on_face[c].size() > 0)
            {
              quadrature_data_on_face[c] =
                DeviceVector<Number>(Kokkos::view_alloc("quadrature_data_on_face_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.quadrature_data_on_face[c].size());
              Kokkos::deep_copy(exec_space,
                                quadrature_data_on_face[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.quadrature_data_on_face[c].data(),
                                  data_cpu.quadrature_data_on_face[c].size()));
            }

          if (data_cpu.values_within_subface[c].size() > 0)
            {
              values_within_subface[c] =
                DeviceVector<Number>(Kokkos::view_alloc("values_within_subface_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.values_within_subface[c].size());
              Kokkos::deep_copy(exec_space,
                                values_within_subface[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.values_within_subface[c].data(),
                                  data_cpu.values_within_subface[c].size()));
            }

          if (data_cpu.gradients_within_subface[c].size() > 0)
            {
              gradients_within_subface[c] =
                DeviceVector<Number>(Kokkos::view_alloc("gradients_within_subface_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.gradients_within_subface[c].size());
              Kokkos::deep_copy(exec_space,
                                gradients_within_subface[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.gradients_within_subface[c].data(),
                                  data_cpu.gradients_within_subface[c].size()));
            }

          if (data_cpu.hessians_within_subface[c].size() > 0)
            {
              hessians_within_subface[c] =
                DeviceVector<Number>(Kokkos::view_alloc("hessians_within_subface_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.hessians_within_subface[c].size());
              Kokkos::deep_copy(exec_space,
                                hessians_within_subface[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.hessians_within_subface[c].data(),
                                  data_cpu.hessians_within_subface[c].size()));
            }

          if (data_cpu.subface_interpolation_matrices[c].size() > 0)
            {
              subface_interpolation_matrices[c] =
                DeviceVector<Number>(Kokkos::view_alloc("subface_interpolation_matrices_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.subface_interpolation_matrices[c].size());
              Kokkos::deep_copy(exec_space,
                                subface_interpolation_matrices[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.subface_interpolation_matrices[c].data(),
                                  data_cpu.subface_interpolation_matrices[c].size()));
            }

          if (data_cpu.subface_interpolation_matrices_scalar[c].size() > 0)
            {
              subface_interpolation_matrices_scalar[c] =
                DeviceVector<Number>(Kokkos::view_alloc("subface_interpolation_matrices_scalar_" +
                                                          std::to_string(c),
                                                        Kokkos::WithoutInitializing),
                                     data_cpu.subface_interpolation_matrices_scalar[c].size());
              Kokkos::deep_copy(exec_space,
                                subface_interpolation_matrices_scalar[c],
                                Kokkos::View<const Number *, Kokkos::HostSpace>(
                                  data_cpu.subface_interpolation_matrices_scalar[c].data(),
                                  data_cpu.subface_interpolation_matrices_scalar[c].size()));
            }
        }

#if DEAL_II_KOKKOS_VERSION_GTE(3, 6, 0)
      exec_space.fence("RT::RaviartThomasOperatorBase::UnivariateShapeData::reinit(): end");
#else
      exec_space.fence();
#endif
    }

  } // namespace internal

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif