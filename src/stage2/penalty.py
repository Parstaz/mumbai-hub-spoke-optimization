"""The adaptive multiplier the GA prices lateness at while it searches.

Time windows are **soft** (CLAUDE.md §1.1): missing one costs ``tw_penalty_per_hour`` and is never
forbidden. That is the right model — a drop half an hour late may genuinely be cheaper than the
detour that would have made it on time — but it is a poor thing to *search* against from a random
start. At the configured ₹250/h, lateness is a small fraction of a tour's cost, so early
generations trade a window away for a few hundred metres and the population settles into orders
that are cheap and chronically late, from which no single or-opt move escapes.

The fix is standard and is search guidance, not a change to the model: price lateness high while
the population is mostly infeasible, and relax as it becomes feasible, so the configured rate
decides the final trade-off rather than the initial conditions. :class:`AdaptivePenalty` is that
multiplier, adjusted every :attr:`~src.config.GAConfig.penalty_adapt_interval` generations against
the share of the population carrying any lateness at all.

**The target is deliberately not zero.** Driving violations to zero would make windows hard by the
back door, and the instance does not admit a time-feasible plan at any sensible cost — the greedy
baseline misses 68 of 800. Aiming at
:attr:`~src.config.GAConfig.penalty_target_violation_rate` keeps pressure on without pretending the
constraint is one the model has.

**The multiplier floors at 1.0, and that bound is not symmetric with the ceiling.** Above 1.0 the
search over-prices lateness: it looks for time-feasible orders first and relaxes toward the true
trade-off as the population finds them, which is guidance — the optimum it converges on is still
the configured model's. Below 1.0 it would *under*-price lateness relative to
:func:`~src.scoring.evaluate_solution`, and that is not weaker guidance but a different objective:
the search would prefer plans that score worse when reported, and would do so more the further the
multiplier fell. There is no reading of "search guidance" under which making a penalised thing
cheaper than the cost model says it is helps. Raising
:attr:`~src.config.GAConfig.penalty_max_multiplier` is a tuning decision; lowering
:attr:`~src.config.GAConfig.penalty_min_multiplier` below 1.0 is a modelling error.

**This multiplier never reaches a reported number.** It scales the ``CostConfig`` handed to
:func:`~src.stage2.split.split`, so arc weights — and therefore
:attr:`~src.stage2.split.SplitPlan.search_objective_inr` — are priced at the search rate, while
:func:`~src.scoring.evaluate_solution` prices the resulting routes at the configured one. Two runs
whose multipliers happened to diverge must still be comparable, and they are only comparable
through the reported figure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.config import CostConfig, GAConfig


@dataclass(frozen=True, slots=True)
class AdaptivePenalty:
    """A multiplier on the configured lateness rate, and the rule for moving it.

    Frozen, and :meth:`adapt` returns a new instance rather than mutating: the GA's generation loop
    is the only thing that advances it, and a penalty that could be changed from anywhere would
    make a run's trajectory depend on call order rather than on its seed.
    """

    multiplier: float

    @classmethod
    def neutral(cls) -> AdaptivePenalty:
        """Start at the configured rate, so generation zero searches the model as written."""
        return cls(multiplier=1.0)

    def rates(self, cost_config: CostConfig) -> CostConfig:
        """The rates to price arcs at: the configured ones, with lateness scaled.

        Only ``tw_penalty_per_hour`` moves. Scaling distance or time as well would change which
        tour is cheaper for reasons that have nothing to do with windows, and the search would stop
        optimising the cost model at all.
        """
        return replace(
            cost_config, tw_penalty_per_hour=cost_config.tw_penalty_per_hour * self.multiplier
        )

    def adapt(self, violating_fraction: float, ga: GAConfig) -> AdaptivePenalty:
        """Move the multiplier toward the configured target violation rate.

        Multiplicative rather than additive, and clamped: the useful range runs from the configured
        rate up to a penalty strong enough to dominate a 20-stop tour, and stepping through that
        additively would spend the run's whole generation budget travelling it. A population
        already inside the target therefore relaxes back toward 1.0 and stops there, which is the
        cost model as written.

        Args:
            violating_fraction: Share of the population whose plan has any lateness, in [0, 1].
            ga: Supplies the target rate, the step and the bounds.

        Returns:
            The penalty for the next block of generations. Unchanged if the population is already
            at the target, which is the fixed point the rule is aiming at.
        """
        if violating_fraction > ga.penalty_target_violation_rate:
            moved = self.multiplier * ga.penalty_step
        elif violating_fraction < ga.penalty_target_violation_rate:
            moved = self.multiplier / ga.penalty_step
        else:
            return self
        clamped = min(max(moved, ga.penalty_min_multiplier), ga.penalty_max_multiplier)
        return AdaptivePenalty(multiplier=clamped)
