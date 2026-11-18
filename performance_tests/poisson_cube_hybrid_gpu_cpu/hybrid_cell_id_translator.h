#pragma once

#include <deal.II/grid/cell_id_translator.h>
#include <deal.II/grid/tria.h>

#include <type_traits>

namespace hybrid
{
  // CellIDTranslator of a triangulation. Newer deal.II builds it from the
  // triangulation, the mg-transfer branch from (n_global_coarse_cells,
  // n_global_levels).
  template <int dim>
  dealii::internal::CellIDTranslator<dim>
  make_cell_id_translator(const dealii::Triangulation<dim> &tria)
  {
    using Translator = dealii::internal::CellIDTranslator<dim>;
    if constexpr (std::is_constructible_v<Translator, const dealii::Triangulation<dim> &>)
      return Translator(tria);
    else
      return Translator(tria.n_global_coarse_cells(), tria.n_global_levels());
  }
} // namespace hybrid
