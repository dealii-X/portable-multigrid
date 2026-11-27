#ifndef newton_elasticity_tangent_pmg_h
#define newton_elasticity_tangent_pmg_h

// Polynomial multigrid for the linearized operator of the Newton driver:
// levels of degree p, p / 2, ..., 1 on the same mesh, each an
// ElasticityTangentLevelOperator linearized at the interpolation of u,
// PolynomialTransfer between them, Chebyshev smoothing with the inverse
// diagonal, and CG with the inverse diagonal to a relative tolerance on
// degree 1. The prescribed DoFs are homogeneous constraints of the transfers
// and the smoothers.

#include <deal.II/base/mg_level_object.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <functional>
#include <memory>
#include <vector>

#include "multigrid/portable_polynomial_transfer.h"
#include "operators/portable_elasticity_tangent_level_operator.h"

DEAL_II_NAMESPACE_OPEN

template <int dim, typename Number, typename Material>
class TangentPMG
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
  using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorType>;
  using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorType>;

  // the constraints and the prescribed DoFs of a level, those on the locally
  // owned cells
  using LevelSetup =
    std::function<void(const DoFHandler<dim> &, AffineConstraints<Number> &, IndexSet &)>;

  template <int fe_degree>
  TangentPMG(Portable::ElasticityTangentLevelOperator<dim, fe_degree, Number, Material> &fine,
             const AffineConstraints<Number> &fine_constraints,
             const IndexSet                  &fine_prescribed,
             const Material                  &material,
             const LevelSetup                &level_setup,
             const unsigned int               smoothing_degree,
             const double                     coarse_tolerance);

  // linearize the coarser levels at the interpolation of the fine u (the fine
  // level is linearized by the caller), set up diagonals and smoothers
  void
  update(const VectorType &u_fine);

  void
  vmult(VectorType &dst, const VectorType &src) const;

  unsigned int
  n_levels() const
  {
    return levels.size();
  }

  unsigned int
  degree(const unsigned int level) const
  {
    return levels[level].degree;
  }

  // CG iterations on the coarse level since the last reset
  mutable unsigned int n_coarse_iterations = 0;
  mutable unsigned int n_vcycles           = 0;

private:
  struct Level
  {
    unsigned int                     degree = 0;
    std::unique_ptr<FESystem<dim>>   fe;
    std::unique_ptr<DoFHandler<dim>> dof_handler;
    // behind pointers: the operators and transfers keep references
    std::unique_ptr<AffineConstraints<Number>>             constraints;
    std::unique_ptr<AffineConstraints<Number>>             transfer_constraints;
    IndexSet                                               prescribed;
    std::unique_ptr<LevelMatrixType>                       owned_matrix;
    LevelMatrixType                                       *matrix = nullptr;
    std::function<void(const VectorType &)>                linearize;
    std::unique_ptr<Portable::MGTransferBase<dim, Number>> transfer; // to the next finer
    VectorType                                             state;
  };

  template <int fe_degree>
  void
  add_coarser_levels(const Triangulation<dim> &triangulation);

  void
  v_cycle(const unsigned int level) const;

  // level 0 is the coarsest
  std::vector<Level> levels;

  const Material     material;
  const LevelSetup   level_setup;
  const unsigned int smoothing_degree;
  const double       coarse_tolerance;

  std::vector<SmootherType> smoothers;

  mutable std::vector<VectorType> solution, defect, t;
};



template <int dim, typename Number, typename Material>
template <int fe_degree>
TangentPMG<dim, Number, Material>::TangentPMG(
  Portable::ElasticityTangentLevelOperator<dim, fe_degree, Number, Material> &fine,
  const AffineConstraints<Number>                                            &fine_constraints,
  const IndexSet                                                             &fine_prescribed,
  const Material                                                             &material,
  const LevelSetup                                                           &level_setup,
  const unsigned int                                                          smoothing_degree,
  const double                                                                coarse_tolerance)
  : material(material)
  , level_setup(level_setup)
  , smoothing_degree(smoothing_degree)
  , coarse_tolerance(coarse_tolerance)
{
  add_coarser_levels<fe_degree>(fine.get_dof_handler().get_triangulation());

  Level fine_level;
  fine_level.degree      = fe_degree;
  fine_level.constraints = std::make_unique<AffineConstraints<Number>>();
  fine_level.constraints->copy_from(fine_constraints);
  fine_level.prescribed = fine_prescribed;
  fine_level.matrix     = &fine;
  levels.push_back(std::move(fine_level));

  const unsigned int n = levels.size();
  for (unsigned int l = 0; l < n; ++l)
    {
      const DoFHandler<dim> &dof_handler          = levels[l].matrix->get_dof_handler();
      auto                  &transfer_constraints = levels[l].transfer_constraints;
      transfer_constraints                        = std::make_unique<AffineConstraints<Number>>(
        dof_handler.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dof_handler));
      transfer_constraints->merge(*levels[l].constraints);
      for (const auto i : levels[l].prescribed)
        if (!transfer_constraints->is_constrained(i))
          transfer_constraints->add_line(i);
      transfer_constraints->close();
    }

  // transfers between level l - 1 and l, created by add_coarser_levels() on
  // the coarser level; reinit here, with all operators in place
  for (unsigned int l = 1; l < n; ++l)
    levels[l - 1].transfer->reinit(*levels[l - 1].matrix,
                                   *levels[l].matrix,
                                   *levels[l - 1].transfer_constraints,
                                   *levels[l].transfer_constraints);

  smoothers.resize(n);
  solution.resize(n);
  defect.resize(n);
  t.resize(n);
  for (unsigned int l = 0; l < n; ++l)
    {
      levels[l].matrix->initialize_dof_vector(solution[l]);
      defect[l] = solution[l];
      t[l]      = solution[l];
      if (l + 1 < n)
        levels[l].matrix->initialize_dof_vector(levels[l].state);
    }
}



template <int dim, typename Number, typename Material>
template <int fe_degree>
void
TangentPMG<dim, Number, Material>::add_coarser_levels(const Triangulation<dim> &triangulation)
{
  if constexpr (fe_degree > 1)
    {
      constexpr int coarse_degree = fe_degree / 2;
      add_coarser_levels<coarse_degree>(triangulation);

      using CoarseOperator =
        Portable::ElasticityTangentLevelOperator<dim, coarse_degree, Number, Material>;

      Level level;
      level.degree      = coarse_degree;
      level.fe          = std::make_unique<FESystem<dim>>(FE_Q<dim>(coarse_degree), dim);
      level.dof_handler = std::make_unique<DoFHandler<dim>>(triangulation);
      level.dof_handler->distribute_dofs(*level.fe);
      level.constraints = std::make_unique<AffineConstraints<Number>>();
      level_setup(*level.dof_handler, *level.constraints, level.prescribed);

      auto            op     = std::make_unique<CoarseOperator>(*level.dof_handler,
                                                 *level.constraints,
                                                 material,
                                                 level.prescribed);
      CoarseOperator *op_ptr = op.get();
      level.linearize        = [op_ptr](const VectorType &u) { op_ptr->linearize(u); };
      level.matrix           = op.get();
      level.owned_matrix     = std::move(op);
      level.transfer =
        std::make_unique<Portable::PolynomialTransfer<dim, coarse_degree, fe_degree, Number>>();
      levels.push_back(std::move(level));
    }
}



template <int dim, typename Number, typename Material>
void
TangentPMG<dim, Number, Material>::update(const VectorType &u_fine)
{
  const unsigned int n = levels.size();
  for (unsigned int l = n - 1; l-- > 0;)
    {
      const VectorType &u_above = (l + 1 == n - 1) ? u_fine : levels[l + 1].state;
      Portable::interpolate_state(levels[l + 1].matrix->get_dof_handler(),
                                  u_above,
                                  levels[l].matrix->get_dof_handler(),
                                  *levels[l].constraints,
                                  levels[l].state);
      levels[l].linearize(levels[l].state);
    }

  for (unsigned int l = 0; l < n; ++l)
    {
      levels[l].matrix->compute_diagonal();
      if (l == 0)
        continue;
      typename SmootherType::AdditionalData data;
      data.smoothing_range     = 15.;
      data.degree              = smoothing_degree;
      data.eig_cg_n_iterations = 15;
      data.preconditioner      = levels[l].matrix->get_matrix_diagonal_inverse();
      data.constraints.copy_from(*levels[l].transfer_constraints);
      smoothers[l].initialize(*levels[l].matrix, data);
    }
}



template <int dim, typename Number, typename Material>
void
TangentPMG<dim, Number, Material>::vmult(VectorType &dst, const VectorType &src) const
{
  const unsigned int top = levels.size() - 1;
  defect[top]            = src;
  v_cycle(top);
  dst = solution[top];
  ++n_vcycles;
}



template <int dim, typename Number, typename Material>
void
TangentPMG<dim, Number, Material>::v_cycle(const unsigned int level) const
{
  if (level == 0)
    {
      SolverControl        control(100000, coarse_tolerance * defect[0].l2_norm());
      SolverCG<VectorType> cg(control);
      solution[0] = 0.;
      cg.solve(*levels[0].matrix,
               solution[0],
               defect[0],
               *levels[0].matrix->get_matrix_diagonal_inverse());
      n_coarse_iterations += control.last_step();
      return;
    }

  smoothers[level].vmult(solution[level], defect[level]);

  levels[level].matrix->vmult(t[level], solution[level]);
  t[level].sadd(-1., 1., defect[level]);

  defect[level - 1] = 0.;
  levels[level - 1].transfer->restrict_and_add(defect[level - 1], t[level]);

  v_cycle(level - 1);

  levels[level - 1].transfer->prolongate_and_add(solution[level], solution[level - 1]);

  smoothers[level].step(solution[level], defect[level]);
}

DEAL_II_NAMESPACE_CLOSE

#endif
