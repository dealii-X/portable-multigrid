#ifndef portable_geometric_transfer_core_h
#define portable_geometric_transfer_core_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.templates.h>

#include <Kokkos_Core.hpp>

#include "base/portable_mg_transfer_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace internal
  {

    template <typename Number>
    struct SimpleVectorDataExchange
    {
      SimpleVectorDataExchange(
        const std::shared_ptr<const Utilities::MPI::Partitioner>
                              &embedded_partitioner,
        AlignedVector<Number> &buffer)
        : embedded_partitioner(embedded_partitioner)
        , buffer(buffer)
      {}

      template <typename VectorType>
      void
      update_ghost_values(const VectorType &vec) const
      {
        update_ghost_values_start(vec);
        update_ghost_values_finish(vec);
      }

      template <typename VectorType>
      void
      update_ghost_values_start(const VectorType &vec) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());
        (void)vec;
#else
        const auto &vector_partitioner = vec.get_partitioner();

        buffer.resize_fast(embedded_partitioner->n_import_indices());

        embedded_partitioner
          ->template export_to_ghosted_array_start<Number, MemorySpace::Host>(
            0,
            dealii::ArrayView<const Number>(
              vec.begin(), embedded_partitioner->locally_owned_size()),
            dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
            dealii::ArrayView<Number>(
              const_cast<Number *>(vec.begin()) +
                embedded_partitioner->locally_owned_size(),
              vector_partitioner->n_ghost_indices()),
            requests);
#endif
      }

      template <typename VectorType>
      void
      update_ghost_values_finish(const VectorType &vec) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());
        (void)vec;
#else
        const auto &vector_partitioner = vec.get_partitioner();

        embedded_partitioner
          ->template export_to_ghosted_array_finish<Number, MemorySpace::Host>(
            dealii::ArrayView<Number>(
              const_cast<Number *>(vec.begin()) +
                embedded_partitioner->locally_owned_size(),
              vector_partitioner->n_ghost_indices()),
            requests);

        vec.set_ghost_state(true);
#endif
      }

      template <typename VectorType>
      void
      compress(VectorType &vec) const
      {
        compress_start(vec);
        compress_finish(vec);
      }

      template <typename VectorType>
      void
      compress_start(VectorType &vec) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());
        (void)vec;
#else
        const auto &vector_partitioner = vec.get_partitioner();

        buffer.resize_fast(embedded_partitioner->n_import_indices());

        embedded_partitioner
          ->template import_from_ghosted_array_start<Number, MemorySpace::Host>(
            VectorOperation::add,
            0,
            dealii::ArrayView<Number>(
              const_cast<Number *>(vec.begin()) +
                embedded_partitioner->locally_owned_size(),
              vector_partitioner->n_ghost_indices()),
            dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
            requests);
#endif
      }

      template <typename VectorType>
      void
      compress_finish(VectorType &vec) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());
        (void)vec;
#else
        const auto &vector_partitioner = vec.get_partitioner();

        embedded_partitioner
          ->template import_from_ghosted_array_finish<Number,
                                                      MemorySpace::Host>(
            VectorOperation::add,
            dealii::ArrayView<const Number>(buffer.begin(), buffer.size()),
            dealii::ArrayView<Number>(
              vec.begin(), embedded_partitioner->locally_owned_size()),
            dealii::ArrayView<Number>(
              const_cast<Number *>(vec.begin()) +
                embedded_partitioner->locally_owned_size(),
              vector_partitioner->n_ghost_indices()),
            requests);
#endif
      }

      template <typename VectorType>
      void
      zero_out_ghost_values(const VectorType &vec) const
      {
        const auto &vector_partitioner = vec.get_partitioner();

        ArrayView<Number> ghost_array(
          const_cast<VectorType &>(vec).begin() +
            vector_partitioner->locally_owned_size(),
          vector_partitioner->n_ghost_indices());

        for (const auto &my_ghosts :
             embedded_partitioner->ghost_indices_within_larger_ghost_set())
          for (unsigned int j = my_ghosts.first; j < my_ghosts.second; ++j)
            ghost_array[j] = 0.;

        vec.set_ghost_state(false);
      }

    private:
      const std::shared_ptr<const Utilities::MPI::Partitioner>
                                       embedded_partitioner;
      dealii::AlignedVector<Number>   &buffer;
      mutable std::vector<MPI_Request> requests;
    };

    /**
     * Base class for Portable::GeometricTrasnfer.
     */
    template <int dim, typename number>
    class GeometricTransferCore : public MGTransferBase<dim, number>
    {
    public:
      GeometricTransferCore();

      using VectorType =
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;


      void
      prolongate_and_add(VectorType &dst, const VectorType &src) const override;
      void
      restrict_and_add(VectorType &dst, const VectorType &src) const override;

    protected:
      /**
       * Perform prolongation on vectors with correct ghosting.
       */
      virtual void
      prolongate_and_add_internal(VectorType       &dst,
                                  const VectorType &src) const = 0;

      /**
       * Perform restriction on vectors with correct ghosting.
       */
      virtual void
      restrict_and_add_internal(VectorType       &dst,
                                const VectorType &src) const = 0;

      /**
       * A wrapper around update_ghost_values() optimized in case the
       * present vector has the same parallel layout of one of the external
       * partitioners.
       */
      void
      update_ghost_values(const VectorType &vec) const;

      /**
       * A wrapper around compress() optimized in case the
       * present vector has the same parallel layout of one of the external
       * partitioners.
       */
      void
      compress(VectorType &vec, const VectorOperation::values op) const;

      /**
       * A wrapper around zero_out_ghost_values() optimized in case the
       * present vector has the same parallel layout of one of the external
       * partitioners.
       */
      void
      zero_out_ghost_values(const VectorType &vec) const;

      /**
       * Enable inplace vector operations if external and internal vectors
       * are compatible.
       */
      template <std::size_t width, typename IndexType>
      std::pair<bool, bool>
      internal_enable_inplace_operations_if_possible(
        const std::shared_ptr<const Utilities::MPI::Partitioner>
          &partitioner_coarse,
        const std::shared_ptr<const Utilities::MPI::Partitioner>
             &partitioner_fine,
        bool &vec_fine_needs_ghost_update,
        dealii::internal::MatrixFreeFunctions::
          ConstraintInfo<dim, VectorizedArray<number, width>, IndexType>
                                  &constraint_info_coarse,
        std::vector<unsigned int> &dof_indices_fine);

      /**
       * Flag if the finite elements on the fine cells are continuous. If yes,
       * the multiplicity of DoF sharing a vertex/line as well as constraints
       * have to be taken into account via weights.
       */
      bool fine_element_is_continuous;

    protected:
      /**
       * Internal vector on which the actual prolongation/restriction is
       * performed.
       */
      mutable VectorType vec_coarse;

      /**
       * Internal vector needed for collecting all degrees of freedom of the
       * fine cells. It is only initialized if the fine-level DoF indices touch
       * DoFs other than the locally active ones (which we always assume can be
       * accessed by the given vectors in the prolongate/restrict functions),
       * otherwise it is left at size zero.
       */
      mutable VectorType vec_fine;


      /**
       * Partitioner needed by the intermediate vector.
       */
      std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

      /**
       * Partitioner needed by the intermediate vector.
       */
      std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;

      /**
       * Bool indicating whether fine vector has relevant ghost values.
       */
      bool vec_fine_needs_ghost_update;

      /**
       * Embedded partitioner for efficient communication if locally relevant
       * DoFs are a subset of an external Partitioner object.
       */
      std::shared_ptr<const Utilities::MPI::Partitioner>
        partitioner_coarse_embedded;

      /**
       * Embedded partitioner for efficient communication if locally relevant
       * DoFs are a subset of an external Partitioner object.
       */
      std::shared_ptr<const Utilities::MPI::Partitioner>
        partitioner_fine_embedded;

      /**
       * Buffer for efficient communication if locally relevant DoFs
       * are a subset of an external Partitioner object.
       */
      mutable AlignedVector<number> buffer_coarse_embedded;

      /**
       * Buffer for efficient communication if locally relevant DoFs
       * are a subset of an external Partitioner object.
       */
      mutable AlignedVector<number> buffer_fine_embedded;
    };


    template <int dim, typename number>
    GeometricTransferCore<dim, number>::GeometricTransferCore()
      : vec_fine_needs_ghost_update(true)
    {}

    template <int dim, typename number>
    void
    GeometricTransferCore<dim, number>::prolongate_and_add(
      VectorType       &dst,
      const VectorType &src) const
    {
      const bool  use_dst_inplace = this->vec_fine.size() == 0;
      auto *const vec_fine_ptr    = use_dst_inplace ? &dst : &this->vec_fine;
      Assert(vec_fine_ptr->get_partitioner().get() ==
               this->partitioner_fine.get(),
             ExcInternalError());

      const bool        use_src_inplace = this->vec_coarse.size() == 0;
      const auto *const vec_coarse_ptr =
        use_src_inplace ? &src : &this->vec_coarse;
      Assert(vec_coarse_ptr->get_partitioner().get() ==
               this->partitioner_coarse.get(),
             ExcInternalError());

      const bool src_ghosts_have_been_set = src.has_ghost_elements();

      if (use_src_inplace == false)
        this->vec_coarse.copy_locally_owned_data_from(src);

      if ((use_src_inplace == false) || (src_ghosts_have_been_set == false))
        this->update_ghost_values(*vec_coarse_ptr);

      if (use_dst_inplace == false)
        *vec_fine_ptr = number(0.);

      this->prolongate_and_add_internal(*vec_fine_ptr, *vec_coarse_ptr);

      if (this->vec_fine_needs_ghost_update || use_dst_inplace == false)
        this->compress(*vec_fine_ptr, VectorOperation::add);

      if (use_dst_inplace == false)
        dst += this->vec_fine;

      if (use_src_inplace && (src_ghosts_have_been_set == false))
        this->zero_out_ghost_values(*vec_coarse_ptr);
    }

    template <int dim, typename number>
    void
    GeometricTransferCore<dim, number>::restrict_and_add(
      VectorType       &dst,
      const VectorType &src) const
    {
      const bool        use_src_inplace = this->vec_fine.size() == 0;
      const auto *const vec_fine_ptr = use_src_inplace ? &src : &this->vec_fine;
      Assert(vec_fine_ptr->get_partitioner().get() ==
               this->partitioner_fine.get(),
             ExcInternalError());

      const bool  use_dst_inplace = this->vec_coarse.size() == 0;
      auto *const vec_coarse_ptr  = use_dst_inplace ? &dst : &this->vec_coarse;
      Assert(vec_coarse_ptr->get_partitioner().get() ==
               this->partitioner_coarse.get(),
             ExcInternalError());

      const bool src_ghosts_have_been_set = src.has_ghost_elements();

      if (use_src_inplace == false)
        this->vec_fine.copy_locally_owned_data_from(src);

      if ((use_src_inplace == false) ||
          (vec_fine_needs_ghost_update && (src_ghosts_have_been_set == false)))
        this->update_ghost_values(*vec_fine_ptr);

      if (use_dst_inplace == false)
        *vec_coarse_ptr = number(0.0);

      // since we might add into the ghost values and call compress
      this->zero_out_ghost_values(*vec_coarse_ptr);

      this->restrict_and_add_internal(*vec_coarse_ptr, *vec_fine_ptr);

      // clean up related to update_ghost_values()
      if (vec_fine_needs_ghost_update == false && use_src_inplace == false)
        this->zero_out_ghost_values(*vec_fine_ptr); // internal vector (DG)
      else if (vec_fine_needs_ghost_update && use_src_inplace == false)
        vec_fine_ptr->set_ghost_state(false); // internal vector (CG)
      else if (vec_fine_needs_ghost_update &&
               (src_ghosts_have_been_set == false))
        this->zero_out_ghost_values(*vec_fine_ptr); // external vector

      this->compress(*vec_coarse_ptr, VectorOperation::add);

      if (use_dst_inplace == false)
        dst += this->vec_coarse;
    }

    inline bool
    is_partitioner_contained(
      const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
      const std::shared_ptr<const Utilities::MPI::Partitioner>
        &external_partitioner)
    {
      // no external partitioner has been given
      if (external_partitioner.get() == nullptr)
        return false;

      // check if locally owned ranges are the same
      if (external_partitioner->size() != partitioner->size())
        return false;

      if (external_partitioner->locally_owned_range() !=
          partitioner->locally_owned_range())
        return false;

      const bool ghosts_locally_contained =
        (external_partitioner->ghost_indices() &
         partitioner->ghost_indices()) == partitioner->ghost_indices();

      // check if ghost values are contained in external partititioner
      return Utilities::MPI::logical_and(ghosts_locally_contained,
                                         partitioner->get_mpi_communicator());
    }

    inline std::shared_ptr<Utilities::MPI::Partitioner>
    create_embedded_partitioner(
      const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
      const std::shared_ptr<const Utilities::MPI::Partitioner>
        &larger_partitioner)
    {
      auto embedded_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        larger_partitioner->locally_owned_range(),
        larger_partitioner->get_mpi_communicator());

      embedded_partitioner->set_ghost_indices(
        partitioner->ghost_indices(), larger_partitioner->ghost_indices());

      return embedded_partitioner;
    }



    template <int dim, typename number>
    template <std::size_t width, typename IndexType>
    std::pair<bool, bool>
    GeometricTransferCore<dim, number>::
      internal_enable_inplace_operations_if_possible(
        const std::shared_ptr<const Utilities::MPI::Partitioner>
          &external_partitioner_coarse,
        const std::shared_ptr<const Utilities::MPI::Partitioner>
             &external_partitioner_fine,
        bool &vec_fine_needs_ghost_update,
        dealii::internal::MatrixFreeFunctions::
          ConstraintInfo<dim, VectorizedArray<number, width>, IndexType>
                                  &constraint_info_coarse,
        std::vector<unsigned int> &dof_indices_fine)
    {
      std::pair<bool, bool> success_flags = {false, false};

      if (this->partitioner_coarse->is_globally_compatible(
            *external_partitioner_coarse))
        {
          this->vec_coarse.reinit(0);
          this->partitioner_coarse = external_partitioner_coarse;
          success_flags.first      = true;
        }
      else if (internal::is_partitioner_contained(this->partitioner_coarse,
                                                  external_partitioner_coarse))
        {
          this->vec_coarse.reinit(0);

          for (auto &i : constraint_info_coarse.dof_indices)
            i = external_partitioner_coarse->global_to_local(
              this->partitioner_coarse->local_to_global(i));

          for (auto &i : constraint_info_coarse.plain_dof_indices)
            i = external_partitioner_coarse->global_to_local(
              this->partitioner_coarse->local_to_global(i));

          this->partitioner_coarse_embedded =
            internal::create_embedded_partitioner(this->partitioner_coarse,
                                                  external_partitioner_coarse);

          this->partitioner_coarse = external_partitioner_coarse;
          success_flags.first      = true;
        }

      vec_fine_needs_ghost_update =
        Utilities::MPI::max(
          this->partitioner_fine->ghost_indices().n_elements(),
          this->partitioner_fine->get_mpi_communicator()) != 0;

      if (this->partitioner_fine->is_globally_compatible(
            *external_partitioner_fine))
        {
          this->vec_fine.reinit(0);
          this->partitioner_fine = external_partitioner_fine;
          success_flags.second   = true;
        }
      else if (internal::is_partitioner_contained(this->partitioner_fine,
                                                  external_partitioner_fine))
        {
          this->vec_fine.reinit(0);

          for (auto &i : dof_indices_fine)
            i = external_partitioner_fine->global_to_local(
              this->partitioner_fine->local_to_global(i));

          this->partitioner_fine_embedded =
            internal::create_embedded_partitioner(this->partitioner_fine,
                                                  external_partitioner_fine);

          this->partitioner_fine = external_partitioner_fine;
          success_flags.second   = true;
        }

      return success_flags;
    }

    template <int dim, typename number>
    void
    GeometricTransferCore<dim, number>::update_ghost_values(
      const VectorType &vec) const
    {
      if ((vec.get_partitioner().get() == this->partitioner_coarse.get()) &&
          (this->partitioner_coarse_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_coarse_embedded, this->buffer_coarse_embedded)
          .update_ghost_values(vec);
      else if ((vec.get_partitioner().get() == this->partitioner_fine.get()) &&
               (this->partitioner_fine_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_fine_embedded, this->buffer_fine_embedded)
          .update_ghost_values(vec);
      else
        vec.update_ghost_values();
    }



    template <int dim, typename number>
    void
    GeometricTransferCore<dim, number>::compress(
      VectorType                   &vec,
      const VectorOperation::values op) const
    {
      Assert(op == VectorOperation::add, ExcNotImplemented());

      if ((vec.get_partitioner().get() == this->partitioner_coarse.get()) &&
          (this->partitioner_coarse_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_coarse_embedded, this->buffer_coarse_embedded)
          .compress(vec);
      else if ((vec.get_partitioner().get() == this->partitioner_fine.get()) &&
               (this->partitioner_fine_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_fine_embedded, this->buffer_fine_embedded)
          .compress(vec);
      else
        vec.compress(op);
    }



    template <int dim, typename number>
    void
    GeometricTransferCore<dim, number>::zero_out_ghost_values(
      const VectorType &vec) const
    {
      if ((vec.get_partitioner().get() == this->partitioner_coarse.get()) &&
          (this->partitioner_coarse_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_coarse_embedded, this->buffer_coarse_embedded)
          .zero_out_ghost_values(vec);
      else if ((vec.get_partitioner().get() == (this->partitioner_fine.get()) &&
                this->partitioner_fine_embedded != nullptr))
        internal::SimpleVectorDataExchange<number>(
          this->partitioner_fine_embedded, this->buffer_fine_embedded)
          .zero_out_ghost_values(vec);
      else
        vec.zero_out_ghost_values();
    }



  } // namespace internal

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE


#endif

