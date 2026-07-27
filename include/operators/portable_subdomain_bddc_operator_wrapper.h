#ifndef portable_subdomain_bddc_operator_wrapper_h
#define portable_subdomain_bddc_operator_wrapper_h

#include <deal.II/base/observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"
#include "operators/portable_subdomain_laplace_operator.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  enum class BDDCVariant
  {
    corner,
    corner_edge,
    corner_edge_face
  };

  template <int dim, typename Number>
  class SubdomainBDDCOperatorWrapper : public SubdomainLaplaceOperatorBase<dim, Number>
  {
  public:
    using InterfaceVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    using SubdomainVectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    SubdomainBDDCOperatorWrapper(
      const SubdomainLaplaceOperatorBase<dim, Number> &dirichlet_operator,
      const BDDCVariant                                variant = BDDCVariant::corner_edge_face)
      : dirichlet_operator(&dirichlet_operator)
      , subdomain_dof_handler(&dirichlet_operator.get_subdomain_dof_handler())
      , n_subdomain_dofs(dirichlet_operator.get_subdomain_dof_handler().get_dof_handler().n_dofs())
      , interface_vector_size(dirichlet_operator.get_interface_dof_indices_subdomain().size())
      , interface_dof_indices_subdomain(dirichlet_operator.get_interface_dof_indices_subdomain())
      , inverse_diagonal_entries(
          std::make_shared<
            DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>())
    {
      if (dim == 2 && variant == BDDCVariant::corner_edge_face)
        bddc_variant = BDDCVariant::corner_edge;
      else
        bddc_variant = variant;

      // store primal constraints
      {
        const auto &subdomain_dof_info = this->subdomain_dof_handler->get_dof_info();

        if (bddc_variant == BDDCVariant::corner)
          {
            n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[1]; // End of Vertices
            n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[1];

            if (n_local_coarse_dofs == 0)
              bddc_variant = BDDCVariant::corner_edge;
          }

        if (bddc_variant == BDDCVariant::corner_edge)
          {
            n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[2]; // End of Edges
            n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[2];

            if (n_local_coarse_dofs == 0)
              bddc_variant = BDDCVariant::corner_edge_face;
          }

        if (bddc_variant == BDDCVariant::corner_edge_face)
          {
            n_global_coarse_dofs = subdomain_dof_info.global_coarse_offsets[3]; // End of Faces
            n_local_coarse_dofs  = subdomain_dof_info.local_coarse_offsets[3];
          }

        AssertThrow(n_global_coarse_dofs > 0, ExcMessage("There's zero global constraints"));
        AssertThrow(n_local_coarse_dofs > 0, ExcMessage("There's zero local constraints"));

        setup_primal_constraint_views();

        dirichlet_operator.initialize_dof_vector(temp_subdomain_vector);
      }
    }

    void
    project(SubdomainVectorType &subdomain_vector) const override
    {
      DeviceVector<Number> subdomain_vector_view(subdomain_vector.get_values(), n_subdomain_dofs);

      const auto offsets                   = this->primal_constraint_offsets;
      const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
      const auto weights                   = this->coarse_weights;

      Kokkos::parallel_for(
        "project_to_homogeneous_constraints_subdomain",
        this->n_local_coarse_dofs,
        KOKKOS_LAMBDA(const int coarse_local_idx) {
          const unsigned int start = offsets(coarse_local_idx);
          const unsigned int end   = offsets(coarse_local_idx + 1);

          const unsigned int n_dofs_per_coarse_dof = end - start;

          if (n_dofs_per_coarse_dof > 0)
            {
              Number average = 0;
              for (unsigned int i = start; i < end; ++i)
                average += subdomain_vector_view(subdomain_constraint_dofs(i));
              average *= weights(coarse_local_idx);

              for (unsigned int i = start; i < end; ++i)
                subdomain_vector_view(subdomain_constraint_dofs(i)) -= average;
            }
        });
      Kokkos::fence();
    }

    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);
    }


    void
    vmult_plain(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_plain(dst, src);
    }

    void
    vmult_bk3(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      (void)dst;
      (void)src;
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override
    {
      (void)dst;
      (void)src;
      (void)ghost_exchange_on;
      (void)computation_on;

      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_interface_cell_range(dst, src);
    }

    void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      dirichlet_operator->vmult_neumann(dst, src);
    }


    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override
    {
      this->vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override
    {
      dirichlet_operator->initialize_dof_vector(vec);
    }

    void
    compute_diagonal() override
    {
      inverse_diagonal_entries->reinit(
        dirichlet_operator->get_matrix_diagonal_inverse_neumann()->get_vector());

      // LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &inverse_diagonal_vector =
      //   inverse_diagonal_entries->get_vector();

      // DeviceVector<Number> inv_diag_view(inverse_diagonal_vector.get_values(),
      //                                    inverse_diagonal_vector.size());

      // const auto offsets                   = this->primal_constraint_offsets;
      // const auto subdomain_constraint_dofs = this->primal_constraint_dofs_subdomain;
      // const auto weights                   = this->coarse_weights;

      // Kokkos::parallel_for(
      //   "project_to_homogeneous_constraints_subdomain",
      //   this->n_local_coarse_dofs,
      //   KOKKOS_LAMBDA(const int coarse_local_idx) {
      //     const unsigned int start = offsets(coarse_local_idx);
      //     const unsigned int end   = offsets(coarse_local_idx + 1);

      //     const unsigned int n_dofs_per_coarse_dof = end - start;

      //     if (n_dofs_per_coarse_dof > 0)
      //       {
      //         for (unsigned int i = start; i < end; ++i)
      //           inv_diag_view(subdomain_constraint_dofs(i)) = Number(1);
      //       }
      //   });
      // Kokkos::fence();
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override
    {
      return this->inverse_diagonal_entries;
    }

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const override
    {
      return dirichlet_operator->get_matrix_diagonal_inverse_neumann();
    }

    types::global_dof_index
    m() const override
    {
      return dirichlet_operator->m();
    }

    types::global_dof_index
    n() const override
    {
      return dirichlet_operator->n();
    }

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override
    {
      (void)col;
      Assert(row == col, ExcNotImplemented());

      Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
             ExcNotInitialized());

      return 1.0 / (*inverse_diagonal_entries)(row, row);
    }

    const MatrixFree<dim, Number> &
    get_matrix_free() const override
    {
      return dirichlet_operator->get_matrix_free();
    }

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override
    {
      return dirichlet_operator->get_vector_partitioner();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_interface_dof_indices_subdomain();
    }

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const override
    {
      return dirichlet_operator->get_physical_boundary_dof_indices_subdomain();
    }

    const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const override
    {
      return dirichlet_operator->get_subdomain_dof_handler();
    }


  private:
    ObserverPointer<const SubdomainLaplaceOperatorBase<dim, Number>> dirichlet_operator;
    ObserverPointer<const SubdomainDoFHandler<dim>>                  subdomain_dof_handler;

    BDDCVariant bddc_variant;

    mutable SubdomainVectorType temp_subdomain_vector;

    unsigned int       n_global_coarse_dofs;
    unsigned int       n_local_coarse_dofs;
    const unsigned int n_subdomain_dofs;
    const unsigned int interface_vector_size;

    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> coarse_weights;

    const Kokkos::View<const unsigned int *, MemorySpace::Default::kokkos_space>
      interface_dof_indices_subdomain;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    // Flattened fine local interface DoF indices associated with each local primal constraint
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_interface_local;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      primal_constraint_dofs_subdomain;

    // offset of the primal constraint dofs per each dof entity
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> primal_constraint_offsets;

    // subdomain (local) to global constraint map
    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> coarse_dofs_local_to_global;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space> corner_dofs_subdomain;

    std::vector<unsigned int> coarse_dofs_local_to_global_vector_host;

    void
    setup_primal_constraint_views()
    {
      const auto &dof_info          = this->subdomain_dof_handler->get_dof_info();
      const auto &local_constraints = dof_info.local_primal_constraints;

      std::vector<unsigned int> primal_constraint_dofs_interface_local_v;
      std::vector<unsigned int> primal_constraint_dofs_subdomain_v;
      std::vector<unsigned int> constraint_dofs_offsets_v;
      std::vector<unsigned int> corner_dofs_subdomain_v;

      coarse_dofs_local_to_global_vector_host.clear();

      std::vector<Number> coarse_weights_v;

      constraint_dofs_offsets_v.push_back(0);
      for (const auto &constraint : local_constraints)
        {
          if (bddc_variant == BDDCVariant::corner &&
              constraint.type != PrimalConstraintType::Vertex)
            continue;

          if (bddc_variant == BDDCVariant::corner_edge &&
              constraint.type == PrimalConstraintType::Face)
            continue;

          if (constraint.type == PrimalConstraintType::Vertex)
            {
              AssertDimension(constraint.interface_partitioner_dofs_local.size(), 1);
              AssertDimension(constraint.local_subdomain_dofs.size(), 1);

              corner_dofs_subdomain_v.push_back(constraint.local_subdomain_dofs[0]);
            }

          primal_constraint_dofs_interface_local_v.insert(
            primal_constraint_dofs_interface_local_v.end(),
            constraint.interface_partitioner_dofs_local.begin(),
            constraint.interface_partitioner_dofs_local.end());

          primal_constraint_dofs_subdomain_v.insert(primal_constraint_dofs_subdomain_v.end(),
                                                    constraint.local_subdomain_dofs.begin(),
                                                    constraint.local_subdomain_dofs.end());


          constraint_dofs_offsets_v.push_back(primal_constraint_dofs_interface_local_v.size());

          coarse_dofs_local_to_global_vector_host.push_back(constraint.global_coarse_dof_index);

          Number weight = Number(1) / Number(constraint.interface_partitioner_dofs_local.size());

          AssertThrow(weight > 0, ExcInternalError());
          coarse_weights_v.push_back(weight);
        }

      AssertThrow(primal_constraint_dofs_interface_local_v.size() > 0, ExcInternalError());
      AssertThrow(primal_constraint_dofs_subdomain_v.size() > 0, ExcInternalError());
      AssertThrow(constraint_dofs_offsets_v.size() > 0, ExcInternalError());
      // AssertThrow(corner_dofs_subdomain_v.size() > 0, ExcInternalError());
      AssertThrow(coarse_dofs_local_to_global_vector_host.size() > 0, ExcInternalError());
      AssertThrow(coarse_weights_v.size() > 0, ExcInternalError());


      // Allocate and copy to Device Kokkos::Views
      Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_interface_local_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "primal_constraint_dofs_interface_local_host"),
        primal_constraint_dofs_interface_local_v.size());
      Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_dofs_subdomain_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "primal_constraint_dofs_subdomain_host"),
        primal_constraint_dofs_subdomain_v.size());
      Kokkos::View<unsigned int *, Kokkos::HostSpace> primal_constraint_constraint_offsets_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "primal_constraint_constraint_offsets_host"),
        constraint_dofs_offsets_v.size());
      Kokkos::View<unsigned int *, Kokkos::HostSpace> coarse_dofs_local_to_global_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_dofs_local_to_global_host"),
        coarse_dofs_local_to_global_vector_host.size());

      Kokkos::View<unsigned int *, Kokkos::HostSpace> corner_dofs_subdomain_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "corner_dofs_subdomain_host"),
        corner_dofs_subdomain_v.size());

      Kokkos::View<Number *, Kokkos::HostSpace> coarse_weights_host(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "coarse_weights_host"),
        coarse_weights_v.size());

      std::copy(primal_constraint_dofs_interface_local_v.begin(),
                primal_constraint_dofs_interface_local_v.end(),
                primal_constraint_dofs_interface_local_host.data());
      std::copy(primal_constraint_dofs_subdomain_v.begin(),
                primal_constraint_dofs_subdomain_v.end(),
                primal_constraint_dofs_subdomain_host.data());
      std::copy(constraint_dofs_offsets_v.begin(),
                constraint_dofs_offsets_v.end(),
                primal_constraint_constraint_offsets_host.data());
      std::copy(coarse_dofs_local_to_global_vector_host.begin(),
                coarse_dofs_local_to_global_vector_host.end(),
                coarse_dofs_local_to_global_host.data());
      std::copy(corner_dofs_subdomain_v.begin(),
                corner_dofs_subdomain_v.end(),
                corner_dofs_subdomain_host.data());
      std::copy(coarse_weights_v.begin(), coarse_weights_v.end(), coarse_weights_host.data());

      dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;

      this->primal_constraint_dofs_interface_local =
        Kokkos::create_mirror_view_and_copy(exec_space,
                                            primal_constraint_dofs_interface_local_host);
      this->primal_constraint_dofs_subdomain =
        Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_dofs_subdomain_host);
      this->primal_constraint_offsets =
        Kokkos::create_mirror_view_and_copy(exec_space, primal_constraint_constraint_offsets_host);
      this->coarse_dofs_local_to_global =
        Kokkos::create_mirror_view_and_copy(exec_space, coarse_dofs_local_to_global_host);
      this->corner_dofs_subdomain =
        Kokkos::create_mirror_view_and_copy(exec_space, corner_dofs_subdomain_host);
      this->coarse_weights = Kokkos::create_mirror_view_and_copy(exec_space, coarse_weights_host);

      exec_space.fence();
    }
  };
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif