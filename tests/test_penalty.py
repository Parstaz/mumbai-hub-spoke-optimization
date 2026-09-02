"""Tests for the adaptive time-window penalty.

Two things are being protected here. The first is that the multiplier moves in the right direction
and settles: a rule that oscillates or runs away would make a run's answer depend on when it
happened to stop. The second, and the one that matters to every reported figure, is that the
multiplier stays *out* of the cost model — it scales what the search prices arcs at and nothing
else, so two runs whose multipliers diverged are still comparable through
:func:`~src.scoring.evaluate_solution`.
"""

from __future__ import annotations

import pytest

from src.config import CostConfig, GAConfig
from src.exceptions import ConfigurationError
from src.stage2.penalty import AdaptivePenalty

GA = GAConfig()
COSTS = CostConfig()
ABOVE_TARGET = GA.penalty_target_violation_rate + 0.2
BELOW_TARGET = GA.penalty_target_violation_rate / 2.0


def test_a_neutral_penalty_prices_at_the_configured_rate() -> None:
    """Generation zero searches the model as written, before any evidence has arrived."""
    assert AdaptivePenalty.neutral().rates(COSTS) == COSTS


def test_a_population_missing_too_many_windows_raises_the_penalty() -> None:
    """Chronic lateness means the search is not being charged enough for it."""
    penalty = AdaptivePenalty.neutral().adapt(ABOVE_TARGET, GA)
    assert penalty.multiplier == pytest.approx(GA.penalty_step)


def test_a_population_inside_the_target_relaxes_the_penalty() -> None:
    """Pressure comes off as the population becomes time-feasible, so the real rate decides."""
    raised = AdaptivePenalty(multiplier=GA.penalty_step**3)
    assert raised.adapt(BELOW_TARGET, GA).multiplier == pytest.approx(GA.penalty_step**2)


def test_a_population_exactly_on_target_is_left_alone() -> None:
    """The fixed point the rule aims at. Moving here would make the multiplier oscillate."""
    penalty = AdaptivePenalty(multiplier=4.0)
    assert penalty.adapt(GA.penalty_target_violation_rate, GA) is penalty


def test_the_penalty_never_falls_below_the_configured_rate() -> None:
    """The asymmetric bound, and the reason it is asymmetric.

    Under 1.0 the search would price lateness below what ``evaluate_solution`` charges, so it
    would prefer plans that report worse — a different objective, not gentler guidance. Relaxing
    all the way down must therefore stop at the cost model rather than pass through it.
    """
    penalty = AdaptivePenalty.neutral()
    for _ in range(20):
        penalty = penalty.adapt(0.0, GA)
    assert penalty.multiplier == pytest.approx(GA.penalty_min_multiplier)
    assert penalty.multiplier >= 1.0
    assert penalty.rates(COSTS).tw_penalty_per_hour >= COSTS.tw_penalty_per_hour


def test_the_penalty_is_capped_however_late_the_population_stays() -> None:
    """An instance with no time-feasible plan must not send the multiplier to infinity.

    Without the ceiling, a hub whose windows simply cannot all be met would drive lateness to
    dominate every other term, and the search would stop optimising distance at all.
    """
    penalty = AdaptivePenalty.neutral()
    for _ in range(40):
        penalty = penalty.adapt(1.0, GA)
    assert penalty.multiplier == pytest.approx(GA.penalty_max_multiplier)


def test_only_the_lateness_rate_is_scaled() -> None:
    """Scaling distance or time too would change which tour is cheaper for unrelated reasons."""
    rates = AdaptivePenalty(multiplier=8.0).rates(COSTS)
    assert rates.tw_penalty_per_hour == pytest.approx(COSTS.tw_penalty_per_hour * 8.0)
    assert rates.variable_per_km == COSTS.variable_per_km
    assert rates.driver_per_hour == COSTS.driver_per_hour
    assert rates.fixed_per_vehicle == COSTS.fixed_per_vehicle


def test_adapting_returns_a_new_penalty_and_leaves_the_old_one_alone() -> None:
    """Frozen, so a run's trajectory depends on its seed rather than on who called what."""
    penalty = AdaptivePenalty.neutral()
    penalty.adapt(ABOVE_TARGET, GA)
    assert penalty.multiplier == pytest.approx(1.0)


def test_a_zero_rate_cost_model_stays_at_zero_however_the_penalty_moves() -> None:
    """An ablation that switches lateness off must not have it switched back on by the search."""
    switched_off = CostConfig(tw_penalty_per_hour=0.0)
    assert AdaptivePenalty(multiplier=64.0).rates(switched_off).tw_penalty_per_hour == 0.0


@pytest.mark.parametrize(
    "override",
    [
        {"penalty_min_multiplier": 0.5},
        {"penalty_step": 1.0},
        {"penalty_adapt_interval": 0},
        {"penalty_target_violation_rate": 1.5},
        {"penalty_max_multiplier": 0.5},
    ],
)
def test_an_impossible_penalty_configuration_is_refused_at_construction(
    override: dict[str, float],
) -> None:
    """Including a floor below 1.0, which is a modelling error rather than an aggressive setting."""
    with pytest.raises(ConfigurationError):
        GAConfig(**override)  # type: ignore[arg-type]  # deliberately ill-typed overrides
