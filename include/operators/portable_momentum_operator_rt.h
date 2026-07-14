#ifndef portable_momentum_operator_rt_h
#define portable_momentum_operator_rt_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include "kernels/kokkos_kernels_rt.h"
#include "matrix_free/portable_shape_info.h"



DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace RT
  {
    template <int dim, typename Number = double>
    class RaviartThomasOperatorBase : public EnableObserverPointer
    {
    public:
      using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

      using KokkosScratchSpace =
        MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space;

      using ViewValues =
        Kokkos::View<Number *, KokkosScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      RaviartThomasOperatorBase() = default;


      template <typename OtherNumber>
      void
      reinit(const Mapping<dim>                   &mapping,
             const DoFHandler<dim>                &dof_handler,
             const AffineConstraints<OtherNumber> &constraints,
             const Quadrature<1>                  &quadrature)
      {
        this->mapping     = &mapping;
        this->dof_handler = &dof_handler;
        this->quadrature  = &quadrature;

        const FiniteElement<dim> &fe = dof_handler.get_fe();

        AssertThrow(dynamic_cast<const FE_RaviartThomasNodal<dim> *>(&fe),
                    ExcMessage("This class only works for Raviart-Thomas elements."));


        this->shape_info_cpu.reinit(quadrature, fe);

        AssertDimension(shape_info_cpu.data.size(), 2);

        this->shape_info[0].reinit(shape_info_cpu.data[0]);
        this->shape_info[1].reinit(shape_info_cpu.data[1]);

        // calculate the number of cells
        n_cells = 0;
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            ++n_cells;

        {
          const MPI_Comm comm = dof_handler.get_mpi_communicator();

          IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

          unsigned int n_constrained_locally_owned_dofs = 0;
          for (const auto &line : constraints.get_lines())
            if (locally_owned_dofs.is_element(line.index))
              ++n_constrained_locally_owned_dofs;

          const types::global_dof_index n_unconstrained_owned_dofs =
            locally_owned_dofs.n_elements() - n_constrained_locally_owned_dofs;

          // Assign DoF numbers for the unconstrained degrees of freedom only
          std::pair<types::global_dof_index, types::global_dof_index> positions =
            Utilities::MPI::partial_and_total_sum(n_unconstrained_owned_dofs, comm);

          std::vector<types::global_dof_index> dof_numbers(locally_owned_dofs.n_elements(),
                                                           numbers::invalid_dof_index);
          types::global_dof_index              counter = positions.first;
          for (unsigned int i = 0; i < n_unconstrained_owned_dofs; ++i)
            dof_numbers[i] = counter++;

          // Extract ghost entries for which we need to query the numbers from
          // remote processes - we do not know the start indices of the
          // respective processes even though we could query the DoF index owner
          // through the triangulation, so we need to perform a lookup anyway
          // and do that by a ghost exchange of all data. While there, also
          // extract a compressed representation, taking the first number on
          // each entity (face, cell) in global index space, which we later
          // translate to local numbers


          std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
          std::vector<types::global_dof_index> lexicographic_dof_indices(fe.dofs_per_cell);

          const std::vector<unsigned int> &lexicographic_numbering =
            shape_info_cpu.lexicographic_numbering;

          // We store the start of the indices per each geometric entity (first
          // 2*dim faces and then the cell dofs).
          std::vector<std::array<types::global_dof_index, 2 * dim + 1>> dof_indices_per_entity(
            n_cells);

          std::vector<types::global_dof_index> ghost_indices;

          unsigned int cell_counter = 0;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(local_dof_indices);

                  // Adjust dof indices due to periodicity
                  for (types::global_dof_index &a : local_dof_indices)
                    {
                      const auto line = constraints.get_constraint_entries(a);
                      if (line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
                        a = (*line)[0].first;
                    }

                  for (types::global_dof_index a : local_dof_indices)
                    if (!locally_owned_dofs.is_element(a) && !constraints.is_constrained(a))
                      ghost_indices.push_back(a);

                  const unsigned int dofs_per_face = fe.dofs_per_face;

                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    {
                      for (unsigned int i = 0; i < dofs_per_face; ++i)
                        AssertThrow(local_dof_indices[f * dofs_per_face + i] ==
                                      local_dof_indices[f * dofs_per_face] + i,
                                    ExcInternalError());

                      dof_indices_per_entity[cell_counter][f] =
                        local_dof_indices[f * dofs_per_face];
                    }

                  const unsigned int start_cell_dofs = 2 * dim * dofs_per_face;
                  for (unsigned int i = 0; i < fe.dofs_per_cell - start_cell_dofs; ++i)
                    AssertThrow(local_dof_indices[start_cell_dofs + i] ==
                                  local_dof_indices[start_cell_dofs] + i,
                                ExcInternalError());

                  dof_indices_per_entity[cell_counter][2 * dim] =
                    local_dof_indices[start_cell_dofs];

                  ++cell_counter;
                }
            }

          IndexSet ghost_index_set(locally_owned_dofs.size());
          ghost_index_set.add_indices(ghost_indices.begin(), ghost_indices.end());
          ghost_index_set.compress();
          Utilities::MPI::Partitioner partitioner_dofs(locally_owned_dofs, ghost_index_set, comm);

          std::vector<types::global_dof_index> tmp_array(partitioner_dofs.n_import_indices());
          std::vector<types::global_dof_index> numbers_ghosts(partitioner_dofs.n_ghost_indices());
          std::vector<MPI_Request>             requests;
          partitioner_dofs.export_to_ghosted_array_start(3,
                                                         make_const_array_view(dof_numbers),
                                                         make_array_view(tmp_array),
                                                         make_array_view(numbers_ghosts),
                                                         requests);
          partitioner_dofs.export_to_ghosted_array_finish(make_array_view(numbers_ghosts),
                                                          requests);

          IndexSet owned_dofs(positions.second);
          owned_dofs.add_range(positions.first, positions.first + n_unconstrained_owned_dofs);
          std::vector<types::global_dof_index> compressed_ghost;
          compressed_ghost.reserve(numbers_ghosts.size());
          for (const types::global_dof_index a : numbers_ghosts)
            if (a != numbers::invalid_dof_index)
              compressed_ghost.push_back(a);
          IndexSet ghost_dofs(positions.second);
          ghost_dofs.add_indices(compressed_ghost.begin(), compressed_ghost.end());
          ghost_dofs.compress();

          partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned_dofs, ghost_dofs, comm);


          dof_indices = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices", Kokkos::WithoutInitializing), 2 * dim + 1, n_cells);

          neighbor_cells = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("neighbor_cells", Kokkos::WithoutInitializing), 2 * dim, n_cells);

          auto dof_indices_host = Kokkos::create_mirror_view(dof_indices);

          auto neighbor_cells_host = Kokkos::create_mirror_view(neighbor_cells);

          dof_indices_per_cell = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices_per_cell", Kokkos::WithoutInitializing),
            fe.dofs_per_cell,
            n_cells);

          auto dof_indices_per_cell_host = Kokkos::create_mirror_view(dof_indices_per_cell);

          {
            for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
              {
                for (unsigned int i = 0; i < 2 * dim; ++i)
                  {
                    neighbor_cells_host(i, cell_id) = numbers::invalid_unsigned_int;
                    dof_indices_host(i, cell_id)    = numbers::invalid_unsigned_int;
                  }

                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  dof_indices_per_cell_host(i, cell_id) = numbers::invalid_unsigned_int;

                dof_indices_host(2 * dim, cell_id) = numbers::invalid_unsigned_int;
              }
          }

          std::array<std::vector<unsigned int>, 2 * dim> cells_at_dirichlet_boundary_by_face;
          std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                                 numbers::invalid_unsigned_int);

          cell_counter = 0;
          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int f = 0; f < 2 * dim + 1; ++f)
                    {
                      const types::global_dof_index index = dof_indices_per_entity[cell_counter][f];
                      const unsigned int      my_index    = partitioner_dofs.global_to_local(index);
                      types::global_dof_index number_compressed;

                      if (my_index < locally_owned_dofs.n_elements())
                        number_compressed = dof_numbers[my_index];
                      else
                        number_compressed = numbers_ghosts[my_index - dof_numbers.size()];

                      Assert(number_compressed != numbers::invalid_dof_index ||
                               constraints.is_constrained(index),
                             ExcInternalError());

                      if (number_compressed != numbers::invalid_dof_index)
                        dof_indices_host(f, cell_counter) =
                          partitioner->global_to_local(number_compressed);
                      else
                        dof_indices_host(f, cell_counter) = numbers::invalid_unsigned_int;
                    }

                  cell_indices[cell->active_cell_index()] = cell_counter;


                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    if (dof_indices_host(f, cell_counter) == numbers::invalid_unsigned_int)
                      cells_at_dirichlet_boundary_by_face[f].push_back(cell_counter);


                  cell->get_dof_indices(local_dof_indices);

                  // Adjust dof indices due to periodicity
                  for (types::global_dof_index &a : local_dof_indices)
                    {
                      const auto line = constraints.get_constraint_entries(a);
                      if (line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
                        a = (*line)[0].first;
                    }

                  if (partitioner)
                    for (auto &index : local_dof_indices)
                      index = partitioner->global_to_local(index);

                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    {
                      const types::global_dof_index dof_index =
                        local_dof_indices[lexicographic_numbering[i]];

                      if (constraints.is_constrained(dof_index))
                        dof_indices_per_cell_host(i, cell_counter) = numbers::invalid_unsigned_int;
                      else
                        dof_indices_per_cell_host(i, cell_counter) = dof_index;
                    }

                  ++cell_counter;
                }
            }

          // put cells at Dirichlet boundary together in one data structure
          unsigned int n_cells_at_dirichlet_boundary = 0;
          for (unsigned int f = 0; f < 2 * dim; ++f)
            n_cells_at_dirichlet_boundary += cells_at_dirichlet_boundary_by_face[f].size();

          std::map<unsigned int, std::vector<std::array<types::global_dof_index, 5>>>
            proc_neighbors;

          cell_counter = 0;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    {
                      const bool at_boundary = cell->at_boundary(f);
                      const bool has_periodic_neighbor =
                        at_boundary && cell->has_periodic_neighbor(f);

                      if (at_boundary == false || has_periodic_neighbor)
                        {
                          const auto neighbor =
                            has_periodic_neighbor ? cell->periodic_neighbor(f) : cell->neighbor(f);

                          if (neighbor->is_locally_owned())
                            {
                              AssertIndexRange(neighbor->active_cell_index(), cell_indices.size());
                              neighbor_cells_host(f, cell_counter) =
                                cell_indices[neighbor->active_cell_index()];
                            }
                          else
                            {
                              std::array<types::global_dof_index, 5> neighbor_data;
                              neighbor_data[0] = cell_counter;
                              neighbor_data[1] = has_periodic_neighbor ?
                                                   cell->periodic_neighbor_face_no(f) :
                                                   cell->neighbor_face_no(f);
                              neighbor_data[2] = cell->global_active_cell_index();
                              neighbor_data[3] = neighbor->global_active_cell_index();
                              neighbor_data[4] = f;
                              proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                              // set dummy
                              neighbor_cells_host(f, cell_counter) =
                                cell_indices[cell->active_cell_index()];
                            }
                        }
                    }

                  ++cell_counter;
                }
            }
          Kokkos::deep_copy(dof_indices, dof_indices_host);
          Kokkos::fence();

          Kokkos::deep_copy(neighbor_cells, neighbor_cells_host);
          Kokkos::fence();

          Kokkos::deep_copy(dof_indices_per_cell, dof_indices_per_cell_host);
          Kokkos::fence();

          {
            std::vector<Polynomials::Polynomial<double>> basis =
              Polynomials::generate_complete_Lagrange_basis(quadrature.get_points());


            interpolate_quad_to_boundary =
              Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("interpolate_quad_to_boundary", Kokkos::WithoutInitializing),
                2,
                basis.size(),
                2);
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

            // std::cout << basis.size() << " == " << fe.degree << std::endl;
            // for (unsigned int i = 0; i < 2; ++i)
            //   for (unsigned int j = 0; j < basis.size(); ++j)
            //     for (int k = 0; k < 2; ++k)
            //       std::cout << interpolate_quad_to_boundary_host(i, j, k) << "  ";
            // std::cout << std::endl;
          }

          quad_values = Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("quad_values", Kokkos::WithoutInitializing),
            Utilities::pow(quadrature.size(), dim),
            dim,
            n_cells);
        }

        compute_geometric_tensors(mapping, quadrature, dof_handler, n_cells);

        // Kokkos::Array<DeviceVector<Number>, 2> shape_values;
        shape_values[0] = shape_info[0].shape_values;
        shape_values[1] = shape_info[1].shape_values;
      }

      void
      compute_geometric_tensors(const Mapping<dim>    &mapping,
                                const Quadrature<1>   &quadrature_1d,
                                const DoFHandler<dim> &dof_handler,
                                const unsigned int     n_cells)
      {
        const unsigned int n_q_points_1d = quadrature_1d.size();
        const unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);

        constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

        geometric_tensor_mass =
          DeviceVector<Number>(Kokkos::view_alloc("geometric_tensor_mass",
                                                  Kokkos::WithoutInitializing),
                               n_cells * symmetric_tensor_dim * n_q_points);

        geometric_tensor_stiffness =
          DeviceVector<Number>(Kokkos::view_alloc("geometric_tensor_stiffness",
                                                  Kokkos::WithoutInitializing),
                               n_cells * symmetric_tensor_dim * n_q_points);

        cell_inverse_jacobians_transpose =
          DeviceVector<Number>(Kokkos::view_alloc("inverse_jacobian_transpose",
                                                  Kokkos::WithoutInitializing),
                               n_cells * dim * dim * n_q_points);

        cell_inverse_jacobians =
          DeviceVector<Number>(Kokkos::view_alloc("inverse_jacobian", Kokkos::WithoutInitializing),
                               n_cells * dim * dim * n_q_points);

        auto geometric_tensor_mass_host = Kokkos::create_mirror_view(geometric_tensor_mass);

        auto geometric_tensor_stiffness_host =
          Kokkos::create_mirror_view(geometric_tensor_stiffness);

        auto inverse_jacobian_transpose_host =
          Kokkos::create_mirror_view(cell_inverse_jacobians_transpose);
        auto inverse_jacobian_host = Kokkos::create_mirror_view(cell_inverse_jacobians);

        // build tensor product quadrature
        Quadrature<dim> quadrature(quadrature_1d);

        const auto &quad_weights = quadrature.get_weights();

        const FiniteElement<dim> &fe = dof_handler.get_fe();

        FEValues<dim> fe_values(mapping,
                                fe,
                                quadrature,
                                update_jacobians | update_jacobian_grads |
                                  update_quadrature_points | update_inverse_jacobians);

        unsigned cell_counter = 0;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const DerivativeForm<1, dim, dim> jacobian     = fe_values.jacobian(q);
                    const DerivativeForm<1, dim, dim> inv_jacobian = fe_values.inverse_jacobian(q);

                    const DerivativeForm<1, dim, dim> inv_jacobian_transpose =
                      transpose(inv_jacobian);

                    const auto determinant = jacobian.determinant();

                    AssertThrow(determinant > 0,
                                ExcMessage("Jacobian determinant must be positive , but it is " +
                                           std::to_string(determinant) + "."));

                    // Number components[symmetric_tensor_dim];
                    // Number components_stiffness[symmetric_tensor_dim];


                    // we only need the symmetric part of the tensor, so we store it
                    // in a compressed format
                    int index = 0;

                    for (int d1 = 0; d1 < dim; ++d1)
                      {
                        for (int d2 = d1; d2 < dim; ++d2)
                          {
                            Number sum = 0, sum_stiffness = 0;
                            for (int k = 0; k < dim; ++k)
                              {
                                sum += jacobian[k][d1] * jacobian[k][d2];
                                sum_stiffness += inv_jacobian[d1][k] * inv_jacobian[d2][k];
                              }
                            geometric_tensor_mass_host(
                              cell_counter * symmetric_tensor_dim * n_q_points +
                              index * n_q_points + q) = sum * quad_weights[q] / determinant;

                            geometric_tensor_stiffness_host(cell_counter * symmetric_tensor_dim *
                                                              n_q_points +
                                                            index * n_q_points + q) = sum_stiffness;

                            ++index;
                          }
                        for (int d2 = 0; d2 < dim; ++d2)
                          {
                            for (int k = 0; k < dim; ++k)
                              {
                                inverse_jacobian_host(cell_counter * dim * dim * n_q_points +
                                                      (d1 * dim + d2) * n_q_points + q) =
                                  inv_jacobian[d1][d2];

                                inverse_jacobian_transpose_host(cell_counter * dim * dim *
                                                                  n_q_points +
                                                                (d1 * dim + d2) * n_q_points + q) =
                                  inv_jacobian_transpose[d1][d2];
                              }
                          }
                      }


                    // for (unsigned int c = 0; c < symmetric_tensor_dim; ++c)
                    //   {
                    //     geometric_tensor_mass_host(
                    //       cell_counter * symmetric_tensor_dim * n_q_points + c * n_q_points + q)
                    //       = components[c];
                    //     geometric_tensor_stiffness_host(
                    //       cell_counter * symmetric_tensor_dim * n_q_points + c * n_q_points + q)
                    //       = components_stiffness[c];
                    // }
                  }
                ++cell_counter;
              }
          }

        Kokkos::deep_copy(geometric_tensor_mass, geometric_tensor_mass_host);
        Kokkos::fence();
        Kokkos::deep_copy(geometric_tensor_stiffness, geometric_tensor_stiffness_host);
        Kokkos::fence();

        Kokkos::deep_copy(cell_inverse_jacobians_transpose, inverse_jacobian_transpose_host);
        Kokkos::fence();
        Kokkos::deep_copy(cell_inverse_jacobians, inverse_jacobian_host);
        Kokkos::fence();
      }

      template <int n_t, int n_q, int component = 0>
      void
      test_cpu(const Number               *in,
               Number                     *out,
               DeviceVector<Number>       &src_device,
               const DeviceVector<Number> &dst_device)
      {
        constexpr unsigned int n_n = n_t + 1;

        Kokkos::Array<DeviceVector<Number>, 2> shape_values;
        shape_values[0] = shape_info[0].shape_values;
        shape_values[1] = shape_info[1].shape_values;

        Portable::RT::mass_operator<dim, n_t, n_q, Number>(shape_values,
                                                           geometric_tensor_mass,
                                                           src_device,
                                                           dst_device,
                                                           dof_indices_per_cell,
                                                           n_cells,
                                                           1u,
                                                           1u,
                                                           1u);

        std::vector<Number> temp1(n_q * Utilities::pow(n_q, dim - 1));
        std::vector<Number> temp2(n_q * Utilities::pow(n_q, dim - 1));


        if (component == 0)
          {
            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_t; ++i0)
                dealii::internal::apply_matrix_vector_product<
                  dealii::internal::EvaluatorVariant::evaluate_general,
                  dealii::internal::EvaluatorQuantity::value,
                  n_n,
                  n_q,
                  1,
                  1,
                  true,
                  false,
                  Number,
                  Number>(shape_info_cpu.data[0].shape_values.data(),
                          in + i1 * n_n * Utilities::pow(n_t, dim - 2) + i0 * n_n,
                          temp1.data() + i1 * n_q * Utilities::pow(n_t, dim - 2) + i0 * n_q);


            if (dim == 2)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  for (unsigned int i0 = 0; i0 < n_q; ++i0)
                    dealii::internal::apply_matrix_vector_product<
                      dealii::internal::EvaluatorVariant::evaluate_general,
                      dealii::internal::EvaluatorQuantity::value,
                      n_t,
                      n_q,
                      n_q,
                      n_q,
                      true,
                      false,
                      Number,
                      Number>(shape_info_cpu.data[1].shape_values.data(),
                              temp1.data() + i1 * n_q * n_t + i0,
                              out + i1 * n_q * n_q + i0);
              }
            if (dim == 3)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  for (unsigned int i0 = 0; i0 < n_q; ++i0)
                    dealii::internal::apply_matrix_vector_product<
                      dealii::internal::EvaluatorVariant::evaluate_general,
                      dealii::internal::EvaluatorQuantity::value,
                      n_t,
                      n_q,
                      n_q,
                      n_q,
                      true,
                      false,
                      Number,
                      Number>(shape_info_cpu.data[1].shape_values.data(),
                              temp1.data() + i1 * n_q * n_t + i0,
                              temp2.data() + i1 * n_q * n_q + i0);

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          Utilities::pow(n_q, dim - 1),
                          Utilities::pow(n_q, dim - 1),
                          true,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  temp2.data() + i0 * n_q + i1,
                                  out + i0 * n_q + i1);
                      }
                    // std::cout << std::endl;
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          n_q * Utilities::pow(n_q, dim - 2),
                          n_q * Utilities::pow(n_q, dim - 2),
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  out + i0 * n_q + i1,
                                  temp1.data() + i0 * n_q + i1);
                      }
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          n_q,
                          n_q,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  temp1.data() + i1 * n_q * n_q + i0,
                                  temp2.data() + i1 * n_q * n_t + i0);
                      }
                  }
              }
            std::cout << std::endl << std::endl;

            if (dim == 2)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          n_q,
                          n_q,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  out + i1 * n_q + i0,
                                  temp2.data() + i1 * n_q + i0);
                      }
                  }
              }
            std::cout << std::endl << std::endl;

            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_t; ++i0)
                {
                  dealii::internal::apply_matrix_vector_product<
                    dealii::internal::EvaluatorVariant::evaluate_general,
                    dealii::internal::EvaluatorQuantity::value,
                    n_t + 1,
                    n_q,
                    1,
                    1,
                    false,
                    false,
                    Number,
                    Number>(shape_info_cpu.data[0].shape_values.data(),
                            temp2.data() + i1 * n_q * n_t + i0 * n_q,
                            out + i1 * n_n * n_t + i0 * n_n);

                  // for (int j = 0; j < n_q; ++j)
                  //   std::cout << temp2[i1 * n_q * n_t + i0 * n_q + j] << " ";
                  // std::cout << std::endl;
                }


            // std::cout << std::endl << std::endl;
            // for (unsigned int i = 0; i < n_q * n_t * Utilities::pow(n_q, dim - 2); ++i)
            //   {
            //     std::cout << temp1[i] << " ";
            //   }

            // std::cout << std::endl << std::endl;
            // if (dim == 2)
            // for (unsigned int i = 0; i < n_q * Utilities::pow(n_t, dim - 1); ++i)
            //   {
            //     std::cout << temp2[i] << " ";
            //   }

            // for (unsigned int i = 0; i < n_n * Utilities::pow(n_t, dim - 1); ++i)
            //   {
            //     std::cout << out[i] << " ";
            //   }
          }

        if (component == 1)
          {
            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_n; ++i0)
                dealii::internal::apply_matrix_vector_product<
                  dealii::internal::EvaluatorVariant::evaluate_general,
                  dealii::internal::EvaluatorQuantity::value,
                  n_t,
                  n_q,
                  1,
                  1,
                  true,
                  false,
                  Number,
                  Number>(shape_info_cpu.data[1].shape_values.data(),
                          in + i1 * n_t * n_n + i0 * n_t,
                          temp1.data() + i1 * n_q * n_n + i0 * n_q);


            if (dim == 2)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  for (unsigned int i0 = 0; i0 < n_q; ++i0)
                    dealii::internal::apply_matrix_vector_product<
                      dealii::internal::EvaluatorVariant::evaluate_general,
                      dealii::internal::EvaluatorQuantity::value,
                      n_n,
                      n_q,
                      n_q,
                      n_q,
                      true,
                      false,
                      Number,
                      Number>(shape_info_cpu.data[0].shape_values.data(),
                              temp1.data() + i1 * n_q * n_n + i0,
                              out + i1 * n_q * n_n + i0);
              }



            if (dim == 3)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  for (unsigned int i0 = 0; i0 < n_q; ++i0)
                    dealii::internal::apply_matrix_vector_product<
                      dealii::internal::EvaluatorVariant::evaluate_general,
                      dealii::internal::EvaluatorQuantity::value,
                      n_n,
                      n_q,
                      n_q,
                      n_q,
                      true,
                      false,
                      Number,
                      Number>(shape_info_cpu.data[0].shape_values.data(),
                              temp1.data() + i1 * n_q * n_n + i0,
                              temp2.data() + i1 * n_q * n_n + i0);

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          Utilities::pow(n_q, dim - 1),
                          Utilities::pow(n_q, dim - 1),
                          true,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  temp2.data() + i0 * n_q + i1,
                                  out + i0 * n_q + i1);
                      }
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          Utilities::pow(n_q, dim - 1),
                          Utilities::pow(n_q, dim - 1),
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  out + i0 * n_q + i1,
                                  temp1.data() + i0 * n_q + i1);
                      }
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_n,
                          n_q,
                          n_q,
                          n_q,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[0].shape_values.data(),
                                  temp1.data() + i1 * n_q * n_q + i0,
                                  temp2.data() + i1 * n_q * n_n + i0);
                      }
                  }
              }
            if (dim == 2)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_n,
                          n_q,
                          n_q,
                          n_q,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[0].shape_values.data(),
                                  out + i1 * n_q + i0,
                                  temp2.data() + i1 * n_q + i0);
                      }
                  }
              }

            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_t : 1); ++i1)
              {
                for (unsigned int i0 = 0; i0 < n_n; ++i0)
                  {
                    dealii::internal::apply_matrix_vector_product<
                      dealii::internal::EvaluatorVariant::evaluate_general,
                      dealii::internal::EvaluatorQuantity::value,
                      n_t,
                      n_q,
                      1,
                      1,
                      false,
                      false,
                      Number,
                      Number>(shape_info_cpu.data[1].shape_values.data(),
                              temp2.data() + i1 * n_q * n_n + i0 * n_q,
                              out + i1 * n_t * n_n + i0 * n_t);
                  }
              }


            std::cout << std::endl << std::endl;
            for (unsigned int i = 0; i < n_n * Utilities::pow(n_t, dim - 1); ++i)
              {
                std::cout << out[i] << " ";
              }


            // std::cout << std::endl << std::endl;
            // for (unsigned int i = 0; i < n_q * n_q * Utilities::pow(n_q, dim - 2); ++i)
            //   {
            //     std::cout << out[i] << " ";
            //   }

            // std::cout << std::endl << std::endl;
            // for (unsigned int i = 0; i < n_q * n_t * Utilities::pow(n_q, dim - 2); ++i)
            //   {
            //     std::cout << temp1[i] << " ";
            //   }

            // std::cout << std::endl << std::endl;
            // for (unsigned int i = 0; i < n_q * n_n * Utilities::pow(n_t, dim - 2); ++i)
            //   {
            //     std::cout << temp2[i] << " ";
            //   }
          }

        if (component == 2)
          {
            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_n : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_t; ++i0)
                dealii::internal::apply_matrix_vector_product<
                  dealii::internal::EvaluatorVariant::evaluate_general,
                  dealii::internal::EvaluatorQuantity::value,
                  n_t,
                  n_q,
                  1,
                  1,
                  true,
                  false,
                  Number,
                  Number>(shape_info_cpu.data[1].shape_values.data(),
                          in + i1 * n_t * n_t + i0 * n_t,
                          temp1.data() + i1 * n_q * n_t + i0 * n_q);


            for (unsigned int i1 = 0; i1 < (dim == 3 ? n_n : 1); ++i1)
              for (unsigned int i0 = 0; i0 < n_q; ++i0)
                dealii::internal::apply_matrix_vector_product<
                  dealii::internal::EvaluatorVariant::evaluate_general,
                  dealii::internal::EvaluatorQuantity::value,
                  n_t,
                  n_q,
                  n_q,
                  n_q,
                  true,
                  false,
                  Number,
                  Number>(shape_info_cpu.data[1].shape_values.data(),
                          temp1.data() + i1 * n_q * n_t + i0,
                          temp2.data() + i1 * n_q * n_q + i0);



            if (dim == 3)
              {
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_n,
                          n_q,
                          Utilities::pow(n_q, dim - 1),
                          Utilities::pow(n_q, dim - 1),
                          true,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[0].shape_values.data(),
                                  temp2.data() + i0 * n_q + i1,
                                  out + i0 * n_q + i1);
                      }
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_q : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_n,
                          n_q,
                          Utilities::pow(n_q, dim - 1),
                          Utilities::pow(n_q, dim - 1),
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[0].shape_values.data(),
                                  out + i1 * n_q + i0,
                                  temp1.data() + i1 * n_q + i0);
                      }
                  }

                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_n : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_q; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          n_q,
                          n_q,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  temp1.data() + i1 * n_q * n_q + i0,
                                  temp2.data() + i1 * n_q * n_t + i0);
                      }
                  }
                for (unsigned int i1 = 0; i1 < (dim == 3 ? n_n : 1); ++i1)
                  {
                    for (unsigned int i0 = 0; i0 < n_t; ++i0)
                      {
                        dealii::internal::apply_matrix_vector_product<
                          dealii::internal::EvaluatorVariant::evaluate_general,
                          dealii::internal::EvaluatorQuantity::value,
                          n_t,
                          n_q,
                          1,
                          1,
                          false,
                          false,
                          Number,
                          Number>(shape_info_cpu.data[1].shape_values.data(),
                                  temp2.data() + i1 * n_q * n_t + i0 * n_q,
                                  out + i1 * n_t * n_t + i0 * n_t);
                      }
                  }
              }
          }
        std::cout << std::endl << std::endl;
        // for (unsigned int i = 0; i < n_q * n_t * n_n; ++i)
        //   {
        //     std::cout << temp2[i] << " ";
        //   }
        // std::cout << std::endl << std::endl;
        // for (unsigned int i = 0; i < n_q * n_q * Utilities::pow(n_n, dim - 2); ++i)
        //   {
        //     std::cout << temp2[i] << " ";
        //   }

        // std::cout << std::endl << std::endl;
        for (unsigned int i = 0; i < n_t * n_t * n_n; ++i)
          {
            std::cout << out[i] << " ";
          }
      }


      void
      initialize_dof_vector(
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const
      {
        vec.reinit(this->partitioner);
      }
      void
      test_cell_operator(
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
        const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
        const Number factor_mass    = Number(1),
        const Number factor_laplace = Number(1)) const
      {
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

        src.update_ghost_values();
        DeviceVector<Number> src_device(src.get_values(), src.locally_owned_size());
        DeviceVector<Number> dst_device(dst.get_values(), dst.locally_owned_size());



        if (shape_info[0].fe_degree == 2)
          {
            constexpr int n_t = 2, n_q = 3;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }

        if (shape_info[0].fe_degree == 2)
          {
            constexpr int n_t = 2, n_q = 3;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }

        if (shape_info[0].fe_degree == 3)
          {
            constexpr int n_t = 3, n_q = 4;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }

        if (shape_info[0].fe_degree == 4)
          {
            constexpr int n_t = 4, n_q = 5;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }


        if (shape_info[0].fe_degree == 5)
          {
            constexpr int n_t = 5, n_q = 6;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }


        if (shape_info[0].fe_degree == 6)
          {
            constexpr int n_t = 6, n_q = 7;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }


        if (shape_info[0].fe_degree == 7)
          {
            constexpr int n_t = 7, n_q = 8;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }



        if (shape_info[0].fe_degree == 8)
          {
            constexpr int n_t = 8, n_q = 9;


            Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }

        dst.compress(VectorOperation::add);
        src.zero_out_ghost_values();
      }

      void
      test(LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>       &dst_host,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &src_host,
           const Number factor_mass    = Number(1),
           const Number factor_laplace = Number(1))
      {
        LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src, dst;
        src.reinit(partitioner);
        dst.reinit(partitioner);

        // std::cout << src.size() << " " << dst.size() << std::endl;
        // std::cout << src_host.size() << " " << dst_host.size() << std::endl;

        LinearAlgebra::ReadWriteVector<Number> rw(src_host.locally_owned_elements());
        rw.import_elements(src_host, VectorOperation::insert);
        src.reinit(partitioner);
        src.import_elements(rw, VectorOperation::insert);

        // for (unsigned int cell = 0; cell < n_cells; ++cell)
        //   {
        //     std::cout << "Cell " << cell << ": ";
        //     for (unsigned int i = 0; i < dof_indices_per_cell.extent(0); ++i)
        //       {
        //         const auto index = dof_indices_per_cell(i, cell);

        //         std::cout << index << " ";
        //       }
        //     std::cout << std::endl;
        //   }


        DeviceVector<Number> src_device(src.get_values(), src.locally_owned_size());
        DeviceVector<Number> dst_device(dst.get_values(), dst.locally_owned_size());


        Kokkos::Array<DeviceVector<Number>, 2> shape_values;
        shape_values[0] = shape_info[0].shape_values;
        shape_values[1] = shape_info[1].shape_values;

        // const unsigned int n_dofs_per_component =
        //   (shape_info[0].fe_degree + 1) * Utilities::pow(shape_info[0].fe_degree, dim - 1);
        // const unsigned int n_q_points = Utilities::pow(shape_info[0].n_q_points_1d, dim);

        // std::vector<Number> in0(n_dofs_per_component);
        // std::vector<Number> in1(n_dofs_per_component);
        // std::vector<Number> in2(n_dofs_per_component);
        // std::vector<Number> out0(n_q_points);
        // std::vector<Number> out1(n_q_points);
        // std::vector<Number> out2(n_q_points);
        // std::vector<Number> temp1(n_q_points);
        // std::vector<Number> temp2(n_q_points);



        // for (unsigned int i = 0; i < n_dofs_per_component; ++i)
        //   {
        //     in0[i] = src_host[dof_indices_per_cell(0 * n_dofs_per_component + i, 0)];
        //     in1[i] = src_host[dof_indices_per_cell(1 * n_dofs_per_component + i, 0)];
        //     if (dim > 2)
        //       in2[i] = src_host[dof_indices_per_cell(2 * n_dofs_per_component + i, 0)];
        //   }


        // const unsigned int nn_n = shape_info[0].fe_degree + 1;
        // const unsigned int nn_t = shape_info[0].fe_degree;
        // const unsigned int nn_q = shape_info[0].n_q_points_1d;

        std::cout << "-----------------------------------\n";
        std::cout << "shape_info[0].shape_data_on_face[0].size() = "
                  << shape_info[0].shape_data_on_face[0].size() << std::endl;
        std::cout << "shape_info[0].shape_data_on_face[1].size() = "
                  << shape_info[0].shape_data_on_face[1].size() << std::endl;
        std::cout << "-----------------------------------\n";

        std::cout << "-----------------------------------\n";
        std::cout << "shape_info[1].shape_data_on_face[0].size() = "
                  << shape_info[1].shape_data_on_face[0].size() << std::endl;
        std::cout << "shape_info[1].shape_data_on_face[1].size() = "
                  << shape_info[1].shape_data_on_face[1].size() << std::endl;
        std::cout << "-----------------------------------\n";

        std::cout << "-----------------------------------\n";
        for (unsigned int i = 0; i < shape_info[0].shape_data_on_face[0].size(); ++i)
          std::cout << shape_info[0].shape_data_on_face[0][i] << "  ";
        std::cout << std::endl << std::endl;

        for (unsigned int i = 0; i < shape_info[0].shape_data_on_face[1].size(); ++i)
          std::cout << shape_info[0].shape_data_on_face[1][i] << "  ";
        std::cout << std::endl << std::endl;

        for (unsigned int i = 0; i < shape_info[1].shape_data_on_face[0].size(); ++i)
          std::cout << shape_info[1].shape_data_on_face[0][i] << "  ";
        std::cout << std::endl << std::endl;


        for (unsigned int i = 0; i < shape_info[1].shape_data_on_face[1].size(); ++i)
          std::cout << shape_info[1].shape_data_on_face[1][i] << "  ";
        std::cout << std::endl << std::endl;

        std::cout << "-----------------------------------\n";


        // std::cout << std::endl << std::endl;
        if (shape_info[0].fe_degree == 2)
          {
            constexpr int n_t = 2, n_q = 3;

            // Portable::RT::mass_operator<dim, n_t, n_q, Number>(shape_values,
            //                                                    geometric_tensor_mass,
            //                                                    src_device,
            //                                                    dst_device,
            //                                                    dof_indices_per_cell,
            //                                                    n_cells,
            //                                                    1u,
            //                                                    1u,
            //                                                    1u);



            // Portable::RT::stiffness_operator<dim, n_t, n_q, Number>(
            //   shape_values,
            //   shape_info[0].shape_gradients_collocation,
            //   geometric_tensor_mass,
            //   geometric_tensor_stiffness,
            //   src_device,
            //   dst_device,
            //   dof_indices_per_cell,
            //   n_cells,
            //   1u,
            //   1u,
            //   1u);

            // Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
            //   shape_values,
            //   shape_info[0].shape_gradients_collocation,
            //   geometric_tensor_mass,
            //   geometric_tensor_stiffness,
            //   src_device,
            //   dst_device,
            //   dof_indices_per_cell,
            //   n_cells,
            //   factor_mass,
            //   factor_laplace,
            //   1u,
            //   1u,
            //   1u);

            Portable::RT::compute_cell<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              quad_values,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              1u,
              1u,
              1u);

            // test_cpu<n_t, n_q>(in0.data(), temp1.data(), src_device, dst_device);
            // test_cpu<n_t, n_q, 1>(in1.data(), out1.data(), src_device, dst_device);
            // if (dim == 3)
            //   test_cpu<n_t, n_q, 2>(in2.data(), out2.data(), src_device, dst_device);
          }
        else if (shape_info[0].fe_degree == 3)
          {
            constexpr int n_t = 3, n_q = 4;

            // test_cpu<n_t, n_q>(in0.data(), temp1.data(), src_device, dst_device);
            // test_cpu<n_t, n_q, 1>(in1.data(), out1.data(), src_device, dst_device);

            // if (dim == 3)
            //   test_cpu<n_t, n_q, 2>(in2.data(), out2.data(), src_device, dst_device);

            // Portable::RT::mass_operator<dim, n_t, n_q, Number>(shape_values,
            //                                                    geometric_tensor_mass,
            //                                                    src_device,
            //                                                    dst_device,
            //                                                    dof_indices_per_cell,
            //                                                    n_cells,
            //                                                    1u,
            //                                                    1u,
            //                                                    1u);

            // Portable::RT::stiffness_operator<dim, n_t, n_q, Number>(
            //   shape_values,
            //   shape_info[0].shape_gradients_collocation,
            //   geometric_tensor_mass,
            //   geometric_tensor_stiffness,
            //   src_device,
            //   dst_device,
            //   dof_indices_per_cell,
            //   n_cells,
            //   1u,
            //   1u,
            //   1u);

            // Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
            // shape_values,
            // shape_info[0].shape_gradients_collocation,
            // geometric_tensor_mass,
            // geometric_tensor_stiffness,
            // src_device,
            // dst_device,
            // dof_indices_per_cell,
            // n_cells,
            // factor_mass,
            // factor_laplace,
            // 1u,
            // 1u,
            // 1u);

            Portable::RT::compute_cell<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              quad_values,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              1u,
              1u,
              1u);
          }
        else if (shape_info[0].fe_degree == 4)
          {
            constexpr int n_t = 4, n_q = 5;

            // test_cpu<n_t, n_q>(in0.data(), temp1.data(), src_device, dst_device);
            // test_cpu<n_t, n_q, 1>(in1.data(), out1.data(), src_device, dst_device);
            // if (dim == 3)
            //   test_cpu<n_t, n_q, 2>(in2.data(), out2.data(), src_device, dst_device);

            // Portable::RT::mass_operator<dim, n_t, n_q, Number>(shape_values,
            //                                                    geometric_tensor_mass,
            //                                                    src_device,
            //                                                    dst_device,
            //                                                    dof_indices_per_cell,
            //                                                    n_cells,
            //                                                    1u,
            //                                                    1u,
            //                                                    1u);


            // Portable::RT::stiffness_operator<dim, n_t, n_q, Number>(
            //   shape_values,
            //   shape_info[0].shape_gradients_collocation,
            //   geometric_tensor_mass,
            //   geometric_tensor_stiffness,
            //   src_device,
            //   dst_device,
            //   dof_indices_per_cell,
            //   n_cells,
            //   1u,
            //   1u,
            //   1u);

            // Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
            // shape_values,
            // shape_info[0].shape_gradients_collocation,
            // geometric_tensor_mass,
            // geometric_tensor_stiffness,
            // src_device,
            // dst_device,
            // dof_indices_per_cell,
            // n_cells,
            // factor_mass,
            // factor_laplace,
            // 1u,
            // 1u,
            // 1u);

            Portable::RT::compute_cell<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              quad_values,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              1u,
              1u,
              1u);
          }
        else if (shape_info[0].fe_degree == 5)
          {
            constexpr int n_t = 5, n_q = 6;

            // test_cpu<n_t, n_q>(in0.data(), temp1.data(), src_device, dst_device);
            // test_cpu<n_t, n_q, 1>(in1.data(), out1.data(), src_device, dst_device);
            // if (dim == 3)
            //   test_cpu<n_t, n_q, 2>(in2.data(), out2.data(), src_device, dst_device);

            // Portable::RT::mass_operator<dim, n_t, n_q, Number>(shape_values,
            //                                                    geometric_tensor_mass,
            //                                                    src_device,
            //                                                    dst_device,
            //                                                    dof_indices_per_cell,
            //                                                    n_cells,
            //                                                    1u,
            //                                                    1u,
            //                                                    1u);


            // Portable::RT::stiffness_operator<dim, n_t, n_q, Number>(
            //   shape_values,
            //   shape_info[0].shape_gradients_collocation,
            //   geometric_tensor_mass,
            //   geometric_tensor_stiffness,
            //   src_device,
            //   dst_device,
            //   dof_indices_per_cell,
            //   n_cells,
            //   1u,
            //   1u,
            //   1u);

            // Portable::RT::helmholtz_operator<dim, n_t, n_q, Number>(
            // shape_values,
            // shape_info[0].shape_gradients_collocation,
            // geometric_tensor_mass,
            // geometric_tensor_stiffness,
            // src_device,
            // dst_device,
            // dof_indices_per_cell,
            // n_cells,
            // factor_mass,
            // factor_laplace,
            // 1u,
            // 1u,
            // 1u);

            Portable::RT::compute_cell<dim, n_t, n_q, Number>(
              shape_values,
              shape_info[0].shape_gradients_collocation,
              geometric_tensor_mass,
              geometric_tensor_stiffness,
              src_device,
              dst_device,
              quad_values,
              dof_indices_per_cell,
              n_cells,
              factor_mass,
              factor_laplace,
              1u,
              1u,
              1u);
          }

        dst.compress(VectorOperation::add);

        rw.import_elements(dst, VectorOperation::insert);
        dst_host.import_elements(rw, VectorOperation::insert);
      }


    private:
      enum class CellOperation
      {
        helmholtz,
        convective,
        convective_and_divergence
      };

      ObserverPointer<const Mapping<dim>>    mapping;
      ObserverPointer<const DoFHandler<dim>> dof_handler;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_per_cell;

      // used as a boundary mask to distinguish which faces are on the boundary and which are not
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> neighbor_cells;

      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> quad_values;
      std::shared_ptr<const Utilities::MPI::Partitioner>           partitioner;

      mutable std::array<double, 15> timings;

      dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info_cpu;

      Kokkos::Array<internal::UnivariateShapeData<Number>, 2> shape_info;

      ObserverPointer<const Quadrature<1>> quadrature;

      DeviceVector<Number> geometric_tensor_mass;
      DeviceVector<Number> geometric_tensor_stiffness;

      DeviceVector<Number> cell_inverse_jacobians_transpose;
      DeviceVector<Number> cell_inverse_jacobians;

      // std::array<std::vector<std::array<Number, 2>>, 2> interpolate_quad_to_boundary;

      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> interpolate_quad_to_boundary;

      Kokkos::Array<DeviceVector<Number>, 2> shape_values;


      unsigned int n_cells;
    };

  } // namespace RT

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif