#ifndef portable_laplace_operator_dg_h
#define portable_laplace_operator_dg_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "kernels/laplace_dg_kokkos_kernels.h"
#include "kernels/portable_local_laplace_operator.h"
#include "matrix_free/portable_shape_info.h"
#include "operators/portable_laplace_operator_quad.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LaplaceOperatorDG : public LaplaceOperatorBase<dim, Number>
  {
  public:
    LaplaceOperatorDG() = default;

    LaplaceOperatorDG(const Mapping<dim>              &mapping,
                      const DoFHandler<dim>           &dof_handler,
                      const AffineConstraints<Number> &constraints,
                      bool                             overlap_communication_computation);


    void
    reinit(const Mapping<dim>              &mapping,
           const DoFHandler<dim>           &dof_handler,
           const AffineConstraints<Number> &constraints,
           bool                             overlap_communication_computation);

    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override;


    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override;

    void
    compute_diagonal() override;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override;

    const MatrixFree<dim, Number> &
    get_matrix_free() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;


  private:
    using TeamHandle =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues =
      Kokkos::View<Number *,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients =
      Kokkos::View<Number **,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static constexpr unsigned int n_local_dofs    = Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points      = Utilities::pow(n_q_points_1d, dim);
    static constexpr unsigned int n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);


    MatrixFree<dim, Number> matrix_free;

    ObserverPointer<const AffineConstraints<Number>> constraints;

    ObserverPointer<const Mapping<dim>> mapping;

    ObserverPointer<const DoFHandler<dim>> dof_handler;

    std::unique_ptr<QGauss<1>> quadrature_1d;

    internal::MatrixFreeFunctions::UnivariateShapeData<Number> shape_data;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    // cell data
    Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices;
    std::vector<std::pair<unsigned int, unsigned int>>                cell_local_info;
    std::vector<std::vector<unsigned int>>                            cell_level_index_map;
    std::array<unsigned int, n_local_dofs>                            lexicographic_numbering;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>
      geometric_transformation_symmetric_cell;


    // face interpolation data
    Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_values_at_quads;
    Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads;
    Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> interpolate_quad_to_boundary;

    // face data: 0 - inner faces, 1 - boundary faces
    std::array<std::vector<std::array<unsigned int, 5>>, 2> face_info_cpu;
    Kokkos::Array<Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space>, 2> face_info;
    Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jacobians_times_normal_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> jacobians_times_normal_boundary_face;
    Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jxw_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>    jxw_boundary_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>    penalty_parameters_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>    penalty_parameters_boundary_face;

    void
    compute_cell_info();

    void
    compute_face_info();

    Number
    get_penalty_factor(const double cell_extent_minus, const double cell_extent_plus) const
    {
      return fe_degree * (fe_degree + 1.) * 0.5 * (1. / cell_extent_minus + 1. / cell_extent_plus);
    }
  };

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::LaplaceOperatorDG(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints,
    bool                             overlap_communication_computation)
  {
    this->reinit(mapping, dof_handler, constraints, overlap_communication_computation);
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::reinit(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints,
    bool                             overlap_communication_computation)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    this->constraints = &constraints;
    this->mapping     = &mapping;
    this->dof_handler = &dof_handler;

    this->quadrature_1d = std::make_unique<QGauss<1>>(n_q_points_1d);

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    matrix_free.reinit(mapping, dof_handler, constraints, *quadrature_1d, additional_data);

    {
      dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(*quadrature_1d,
                                                                          dof_handler.get_fe());

      shape_data.reinit(shape_info.get_shape_data());

      AssertDimension(shape_info.lexicographic_numbering.size(), n_local_dofs);

      for (unsigned int i = 0; i < n_local_dofs; ++i)
        {
          this->lexicographic_numbering[i] = shape_info.lexicographic_numbering[i];
        }
    }

    compute_cell_info();

    compute_face_info();

    // We need to compute the values of the shape functions at the quadrature points on the boundary
    // for the face integrals
    {
      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quadrature_1d->get_points());

      AssertDimension(basis.size(), n_q_points_1d);

      interpolate_quad_to_boundary = Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("interpolate_quad_to_boundary", Kokkos::WithoutInitializing),
        2,            // location at 0 or 1
        basis.size(), // quad points
        2);           // values and derivative

      auto interpolate_quad_to_boundary_host =
        Kokkos::create_mirror_view(interpolate_quad_to_boundary);

      std::vector<double> val_and_der(2);

      for (unsigned int i = 0; i < basis.size(); ++i)
        {
          basis[i].value(0., val_and_der);
          interpolate_quad_to_boundary_host(0, i, 0) = val_and_der[0];
          interpolate_quad_to_boundary_host(1, i, 0) = val_and_der[1];

          basis[i].value(1., val_and_der);
          interpolate_quad_to_boundary_host(0, i, 1) = val_and_der[0];
          interpolate_quad_to_boundary_host(1, i, 1) = val_and_der[1];
        }

      Kokkos::deep_copy(interpolate_quad_to_boundary, interpolate_quad_to_boundary_host);
      Kokkos::fence();

      face_values_at_quads = Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("face_values_at_quads", Kokkos::WithoutInitializing),
        n_q_points_face,
        2 * dim,
        cell_local_info.size());

      face_normal_derivatives_at_quads =
        Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("face_normal_derivatives_at_quads", Kokkos::WithoutInitializing),
          n_q_points_face,
          2 * dim,
          cell_local_info.size());
    }
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::vmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    src.update_ghost_values();

    const unsigned int locally_relevant_size = cell_local_info.size() * n_local_dofs;

    DeviceVector<Number> src_device(src.get_values(), locally_relevant_size),
      dst_device(dst.get_values(), locally_relevant_size);

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_blocks          = numbers::invalid_unsigned_int;
    unsigned int threads_per_block = numbers::invalid_unsigned_int;

    if (is_serial)
      {
        n_blocks          = 1u;
        threads_per_block = 1u;
      }

    // n_cells_per_batch = 1u;

    const unsigned int n_cells          = cell_local_info.size();
    const unsigned int n_inner_faces    = face_info_cpu[0].size();
    const unsigned int n_boundary_faces = face_info_cpu[1].size();

    if (n_cells > 0)
      {

        BK3::DG::compute_cell<dim, fe_degree + 1, n_q_points_1d, Number>(
          shape_data.shape_values,
          shape_data.shape_gradients_collocation,
          geometric_transformation_symmetric_cell,
          src_device,
          dst_device,
          interpolate_quad_to_boundary,
          face_values_at_quads,
          face_normal_derivatives_at_quads,
          dof_indices,
          n_cells,
          n_cells_per_batch,
          n_blocks,
          threads_per_block);

        if (n_inner_faces > 0)
          BK3::DG::compute_inner_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
            shape_data.shape_gradients_collocation,
            jacobians_times_normal_inner_face,
            jxw_inner_face,
            penalty_parameters_inner_face,
            face_values_at_quads,
            face_normal_derivatives_at_quads,
            face_info[0],
            n_inner_faces,
            n_cells_per_batch,
            n_blocks,
            threads_per_block);

        if (n_boundary_faces > 0)
          BK3::DG::compute_boundary_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
            shape_data.shape_gradients_collocation,
            jacobians_times_normal_boundary_face,
            jxw_boundary_face,
            penalty_parameters_boundary_face,
            face_values_at_quads,
            face_normal_derivatives_at_quads,
            face_info[1],
            n_boundary_faces,
            n_cells_per_batch,
            n_blocks,
            threads_per_block);

        BK3::DG::distribute_face_to_global<dim, fe_degree + 1, n_q_points_1d, Number>(
          shape_data.shape_values,
          dst_device,
          interpolate_quad_to_boundary,
          face_values_at_quads,
          face_normal_derivatives_at_quads,
          dof_indices,
          n_cells,
          n_cells_per_batch,
          n_blocks,
          threads_per_block);
      }


    src.zero_out_ghost_values();
    dst.zero_out_ghost_values();


    matrix_free.copy_constrained_values(src, dst);

  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const
  {
    if (ghost_exchange_on)
      src.update_ghost_values();

    const unsigned int locally_relevant_size = cell_local_info.size() * n_local_dofs;

    DeviceVector<Number> src_device(src.get_values(), locally_relevant_size),
      dst_device(dst.get_values(), locally_relevant_size);

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_blocks          = numbers::invalid_unsigned_int;
    unsigned int threads_per_block = numbers::invalid_unsigned_int;
    if (is_serial)
      {
        n_cells_per_batch = 1u;
        n_blocks          = 1u;
        threads_per_block = 1u;
      }

    const unsigned int n_cells          = cell_local_info.size();
    const unsigned int n_inner_faces    = face_info_cpu[0].size();
    const unsigned int n_boundary_faces = face_info_cpu[1].size();


    if (computation_on)
      {
        if (n_cells > 0)
          {
            BK3::DG::compute_cell<dim, fe_degree + 1, n_q_points_1d, Number>(
              shape_data.shape_values,
              shape_data.shape_gradients_collocation,
              geometric_transformation_symmetric_cell,
              src_device,
              dst_device,
              interpolate_quad_to_boundary,
              face_values_at_quads,
              face_normal_derivatives_at_quads,
              dof_indices,
              n_cells,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);

            Kokkos::fence();


            if (n_inner_faces > 0)
              BK3::DG::compute_inner_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
                shape_data.shape_gradients_collocation,
                jacobians_times_normal_inner_face,
                jxw_inner_face,
                penalty_parameters_inner_face,
                face_values_at_quads,
                face_normal_derivatives_at_quads,
                face_info[0],
                n_inner_faces,
                n_cells_per_batch,
                n_blocks,
                threads_per_block);

            Kokkos::fence();

            if (n_boundary_faces > 0)
              BK3::DG::compute_boundary_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
                shape_data.shape_gradients_collocation,
                jacobians_times_normal_boundary_face,
                jxw_boundary_face,
                penalty_parameters_boundary_face,
                face_values_at_quads,
                face_normal_derivatives_at_quads,
                face_info[1],
                n_boundary_faces,
                n_cells_per_batch,
                n_blocks,
                threads_per_block);

            Kokkos::fence();


            BK3::DG::distribute_face_to_global<dim, fe_degree + 1, n_q_points_1d, Number>(
              shape_data.shape_values,
              dst_device,
              interpolate_quad_to_boundary,
              face_values_at_quads,
              face_normal_derivatives_at_quads,
              dof_indices,
              n_cells,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);

            Kokkos::fence();
          }
      }

    if (ghost_exchange_on)
      {
        src.zero_out_ghost_values();
        dst.zero_out_ghost_values();

        matrix_free.copy_constrained_values(src, dst);
      }
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_cell_info()
  {
    const auto &triangulation = dof_handler->get_triangulation();

    cell_level_index_map.resize(triangulation.n_levels());

    for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
      cell_level_index_map[level].resize(triangulation.n_raw_cells(level));

    this->cell_local_info.clear();

    std::vector<unsigned char> cell_touched(triangulation.n_active_cells(), 0);

    unsigned int counter = 0;

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_touched[cell->index()]                        = 1;
          cell_level_index_map[cell->level()][cell->index()] = counter++;
          cell_local_info.push_back(std::make_pair(cell->level(), cell->index()));
        }

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int face : cell->face_indices())
              {
                if (!cell->at_boundary(face) || cell->has_periodic_neighbor(face))
                  {
                    const auto neighbor = cell->neighbor_or_periodic_neighbor(face);

                    AssertThrow(neighbor->is_artificial() == false,
                                ExcMessage("Neighbor should be either locally owned or ghost"));

                    if (neighbor->is_active())
                      {
                        if (cell_touched[neighbor->index()] == 0)
                          {
                            cell_touched[neighbor->index()]                            = 1;
                            cell_level_index_map[neighbor->level()][neighbor->index()] = counter++;

                            cell_local_info.push_back(
                              std::make_pair(neighbor->level(), neighbor->index()));
                          }
                      }
                    else
                      AssertThrow(false, ExcMessage("The neighbor cell is more refined than us."));
                  }
              }
          }
      }

    {
      this->dof_indices = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("dof_indices", Kokkos::WithoutInitializing),
        n_local_dofs,
        cell_local_info.size());

      auto dof_indices_host = Kokkos::create_mirror_view(this->dof_indices);

      const auto &partitioner = matrix_free.get_vector_partitioner();

      std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);

      for (unsigned int cell_id = 0; cell_id < this->cell_local_info.size(); ++cell_id)
        {
          const auto &cell =
            typename dealii::DoFHandler<dim>::active_cell_iterator(&triangulation,
                                                                   cell_local_info[cell_id].first,
                                                                   cell_local_info[cell_id].second,
                                                                   dof_handler);

          cell->get_dof_indices(local_dof_indices);

          if (partitioner)
            for (auto &index : local_dof_indices)
              index = partitioner->global_to_local(index);

          for (unsigned int i = 0; i < n_local_dofs; ++i)
            {
              const unsigned int local_dof = local_dof_indices[lexicographic_numbering[i]];
              if (this->constraints->is_constrained(local_dof))
                dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
              else
                dof_indices_host(i, cell_id) = local_dof;
            }
        }

      Kokkos::deep_copy(dof_indices, dof_indices_host);
      Kokkos::fence();
    }

    {
      constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

      this->geometric_transformation_symmetric_cell =
        Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("geometric_transformation_symmetric_cell",
                             Kokkos::WithoutInitializing),
          cell_local_info.size() * symmetric_tensor_dim * n_q_points);

      auto geometry_tensor_host =
        Kokkos::create_mirror_view(geometric_transformation_symmetric_cell);

      Quadrature<dim> quadrature(*quadrature_1d);
      FEValues<dim>   fe_values(*mapping,
                                dof_handler->get_fe(),
                                quadrature,
                                update_gradients | update_JxW_values | update_jacobians);

      for (unsigned int cell_id = 0; cell_id < this->cell_local_info.size(); ++cell_id)
        {
          const auto &cell = typename dealii::Triangulation<dim>::active_cell_iterator(
            &triangulation, cell_local_info[cell_id].first, cell_local_info[cell_id].second);

          fe_values.reinit(cell);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<2, dim, double> inv_jacobian(fe_values.jacobian(q).covariant_form());
              const double                 jxw = fe_values.JxW(q);

              Number components[symmetric_tensor_dim];

              int idx = 0;
              for (unsigned int d1 = 0; d1 < dim; ++d1)
                for (int d2 = d1; d2 < dim; ++d2)
                  {
                    Number sum = 0;
                    for (int k = 0; k < dim; ++k)
                      sum += inv_jacobian[k][d1] * inv_jacobian[k][d2];
                    components[idx] = jxw * sum;
                    ++idx;
                  }

              for (int c = 0; c < symmetric_tensor_dim; ++c)
                geometry_tensor_host[cell_id * symmetric_tensor_dim * n_q_points + c * n_q_points +
                                     q] = components[c];
            }
        }

      Kokkos::deep_copy(geometric_transformation_symmetric_cell, geometry_tensor_host);
      Kokkos::fence();
    }
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_face_info()
  {
    const auto &triangulation = dof_handler->get_triangulation();


    this->face_info_cpu[0].clear();
    this->face_info_cpu[1].clear();

    // get interior and boundary faces information
    // 0 - interior cell
    // 1 - exterior cell
    // 2 - interior face no
    // 3 - exterior face no
    // 4 - face orientation
    {
      std::vector<unsigned char> visited_face(triangulation.n_raw_faces(), 0);

      for (const auto &cell : dof_handler->active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              for (const auto f : cell->face_indices())
                {
                  const auto &face = cell->face(f);

                  const unsigned int face_index = face->index();

                  if (visited_face[face_index] == 1)
                    continue;

                  visited_face[face_index] = 1;

                  if (face->at_boundary())
                    {
                      std::array<unsigned int, 5> info;
                      info[0] = cell_level_index_map[cell->level()][cell->index()];
                      info[1] = numbers::invalid_unsigned_int;
                      info[2] = f;
                      info[3] = face->boundary_id();
                      info[4] = 0;

                      face_info_cpu[1].push_back(info);
                    }
                  else
                    {
                      typename dealii::Triangulation<dim>::cell_iterator neighbor =
                        cell->neighbor_or_periodic_neighbor(f);

                      const unsigned int neighbor_face_no = cell->has_periodic_neighbor(f) ?
                                                              cell->periodic_neighbor_face_no(f) :
                                                              cell->neighbor_face_no(f);

                      std::array<unsigned int, 5> info;
                      info[0] = cell_level_index_map[cell->level()][cell->index()];
                      info[1] = cell_level_index_map[neighbor->level()][neighbor->index()];
                      info[2] = f;
                      info[3] = neighbor_face_no;
                      info[4] = 0;

                      face_info_cpu[0].push_back(info);
                    }
                }
            }
        }

      std::string name_handle[] = {"inner_faces_info", "boundary_faces_info"};

      for (unsigned int i = 0; i < 2; ++i)
        {
          const unsigned int n_faces = face_info_cpu[i].size();

          face_info[i] = Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc(name_handle[i], Kokkos::WithoutInitializing), n_faces);

          if (n_faces > 0)
            {
              auto face_info_host = Kokkos::create_mirror_view(face_info[i]);

              for (unsigned int f = 0; f < n_faces; ++f)
                for (unsigned int k = 0; k < 5; ++k)
                  face_info_host(f, k) = face_info_cpu[i][f][k];

              Kokkos::deep_copy(face_info[i], face_info_host);
            }
        }

      Kokkos::fence();
    }

    {
      Quadrature<dim - 1> face_quadrature(*quadrature_1d);
      FEFaceValues<dim>   fe_face_values(*mapping,
                                         dof_handler->get_fe(),
                                         face_quadrature,
                                         update_gradients | update_JxW_values |
                                           update_normal_vectors | update_jacobians);
      FEFaceValues<dim>   fe_face_values_neighbor(*mapping,
                                                  dof_handler->get_fe(),
                                                  face_quadrature,
                                                  update_gradients | update_JxW_values |
                                                    update_normal_vectors | update_jacobians);

      const unsigned int n_inner_faces    = face_info_cpu[0].size();
      const unsigned int n_boundary_faces = face_info_cpu[1].size();

      jacobians_times_normal_inner_face =
        Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("jacobian_times_normal_inner_face", Kokkos::WithoutInitializing),
          n_inner_faces * dim * n_q_points_face);

      jxw_inner_face = Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("jxw_inner_face", Kokkos::WithoutInitializing),
        n_inner_faces * n_q_points_face);

      penalty_parameters_inner_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("penalty_parameters_inner_face", Kokkos::WithoutInitializing),
        n_inner_faces);

      jacobians_times_normal_boundary_face =
        Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("jacobian_times_normal_boundary_face", Kokkos::WithoutInitializing),
          n_boundary_faces * dim * n_q_points_face);

      jxw_boundary_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("jxw_boundary_face", Kokkos::WithoutInitializing),
        n_boundary_faces * n_q_points_face);

      penalty_parameters_boundary_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("penalty_parameters_boundary_face", Kokkos::WithoutInitializing),
        n_boundary_faces);

      auto jacobians_times_normal_inner_face_host =
        Kokkos::create_mirror_view(jacobians_times_normal_inner_face);
      auto jacobians_times_normal_boundary_face_host =
        Kokkos::create_mirror_view(jacobians_times_normal_boundary_face);
      auto jxw_inner_face_host    = Kokkos::create_mirror_view(jxw_inner_face);
      auto jxw_boundary_face_host = Kokkos::create_mirror_view(jxw_boundary_face);
      auto penalty_parameters_inner_face_host =
        Kokkos::create_mirror_view(penalty_parameters_inner_face);
      auto penalty_parameters_boundary_face_host =
        Kokkos::create_mirror_view(penalty_parameters_boundary_face);

      for (unsigned int f = 0; f < n_inner_faces; ++f)
        {
          const unsigned int cell_minus = face_info_cpu[0][f][0];
          const unsigned int cell_plus  = face_info_cpu[0][f][1];
          const unsigned int f_minus    = face_info_cpu[0][f][2];
          const unsigned int f_plus     = face_info_cpu[0][f][3];

          const typename dealii::Triangulation<dim>::cell_iterator cell_it_minus(
            &triangulation, cell_local_info[cell_minus].first, cell_local_info[cell_minus].second);

          const typename dealii::Triangulation<dim>::cell_iterator cell_it_plus(
            &triangulation, cell_local_info[cell_plus].first, cell_local_info[cell_plus].second);

          // compute penalty factors
          {
            const double extent1 =
              cell_it_minus->measure() / cell_it_minus->face(f_minus)->measure();
            const double extent2 = cell_it_plus->measure() / cell_it_plus->face(f_plus)->measure();

            penalty_parameters_inner_face_host(f) = get_penalty_factor(extent1, extent2);

            fe_face_values.reinit(cell_it_minus, f_minus);
            fe_face_values_neighbor.reinit(cell_it_plus, f_plus);

            for (unsigned int q = 0; q < n_q_points_face; ++q)
              {
                const Tensor<1, dim> n_minus = fe_face_values.normal_vector(q);

                const Tensor<2, dim, double> inv_jacobian_minus(
                  fe_face_values.jacobian(q).covariant_form());

                const Tensor<2, dim, double> inv_jacobian_plus(
                  fe_face_values_neighbor.jacobian(q).covariant_form());

                Tensor<1, dim, double> jac_x_n_minus = inv_jacobian_minus * n_minus;

                Tensor<1, dim, double> jac_x_n_plus = inv_jacobian_plus * n_minus;

                jxw_inner_face_host(f * n_q_points_face + q, 0) = fe_face_values.JxW(q);

                jxw_inner_face_host(f * n_q_points_face + q, 1) = fe_face_values_neighbor.JxW(q);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    jacobians_times_normal_inner_face_host(f * dim * n_q_points_face +
                                                             d * n_q_points_face + q,
                                                           0) = jac_x_n_minus[d];
                    jacobians_times_normal_inner_face_host(f * dim * n_q_points_face +
                                                             d * n_q_points_face + q,
                                                           1) = jac_x_n_plus[d];
                  }
              }
          }
        }
      for (unsigned int f = 0; f < n_boundary_faces; ++f)
        {
          const unsigned int cell = face_info_cpu[1][f][0];
          const unsigned int face = face_info_cpu[1][f][2];

          Assert(face_info_cpu[1][f][1] == numbers::invalid_unsigned_int, ExcInternalError());

          const typename dealii::Triangulation<dim>::cell_iterator cell_it(
            &triangulation, cell_local_info[cell].first, cell_local_info[cell].second);

          // compute penalty factors
          {
            const double extent = cell_it->measure() / cell_it->face(face)->measure();

            penalty_parameters_boundary_face_host(f) = get_penalty_factor(extent, extent);
          }

          fe_face_values.reinit(cell_it, face);

          for (unsigned int q = 0; q < n_q_points_face; ++q)
            {
              const Tensor<1, dim> n = fe_face_values.normal_vector(q);

              const Tensor<2, dim, double> inv_jacobian(
                fe_face_values.jacobian(q).covariant_form());

              Tensor<1, dim, double> jac_x_n = inv_jacobian * n;

              jxw_boundary_face_host(f * n_q_points_face + q) = fe_face_values.JxW(q);

              for (unsigned int d = 0; d < dim; d++)
                {
                  jacobians_times_normal_boundary_face_host(f * dim * n_q_points_face +
                                                            d * n_q_points_face + q) = jac_x_n[d];
                }
            }
        }
      Kokkos::deep_copy(jacobians_times_normal_inner_face, jacobians_times_normal_inner_face_host);
      Kokkos::deep_copy(jacobians_times_normal_boundary_face,
                        jacobians_times_normal_boundary_face_host);
      Kokkos::deep_copy(jxw_inner_face, jxw_inner_face_host);
      Kokkos::deep_copy(jxw_boundary_face, jxw_boundary_face_host);
      Kokkos::deep_copy(penalty_parameters_inner_face, penalty_parameters_inner_face_host);
      Kokkos::deep_copy(penalty_parameters_boundary_face, penalty_parameters_boundary_face_host);

      Kokkos::fence();
    }
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::Tvmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  const MatrixFree<dim, Number> &
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &inverse_diagonal =
      inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    internal::LaplaceOperatorQuad<dim, fe_degree, n_q_points_1d, Number> operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, 1, Number>(
      matrix_free,
      inverse_diagonal,
      operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    Number *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  types::global_dof_index
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  types::global_dof_index
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  Number
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif