"""Tests for the step 8 report.

CLI entry points are exempt from the coverage standard (CLAUDE.md §3), but what this module does
is not presentation — it decides whether a gap may be reported at all, and in which direction.
Three things are worth pinning: that an incomplete reference plan yields no cost per drop, that
the verdict names the actual leader rather than assuming one, and that the two ways a run can be
misread — a first-solution-capped reference, and hubs cut off mid-descent — announce themselves.
"""

from __future__ import annotations

from typing import cast

import pytest
from ortools.constraint_solver import routing_enums_pb2

from src.cli.reference import (
    Comparison,
    budget_lines,
    compare,
    report_lines,
    status_lines,
    verdict_lines,
)
from src.config import CostConfig
from src.data.instance import Instance
from src.scoring import CostBreakdown, Metrics
from src.stage2.ortools_reference import HubReference, ReferencePlan
from src.units import Rupees

_SUCCESS = int(routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS)
_TRUNCATED = int(
    routing_enums_pb2.RoutingSearchStatus.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED
)
_TIMEOUT = int(routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT)

_UNUSED_INSTANCE = cast(Instance, None)
"""Stands in where the instance is provably never read: ``compare`` returns before scoring.

Passing ``None`` makes that laziness part of the contract rather than an implementation detail.
If the code ever started scoring an incomplete plan, these tests would fail loudly instead of
quietly reporting a partial column as though it were a result.
"""


def hub(
    hub_id: int = 0,
    *,
    solved: bool = True,
    status: int = _SUCCESS,
    fleet: tuple[int, int] = (3, 2),
    solution_limit: int = 0,
) -> HubReference:
    """One hub's reference result, shaped by what the test is about.

    ``fleet`` pairs vehicles offered with vehicles deployed because only their *relation* matters
    to anything here — equal means the hub sat at its ceiling — and they are never varied apart.
    """
    offered, deployed = fleet
    return HubReference(
        hub_id=hub_id,
        orders=tuple((position,) for position in range(deployed)) if solved else None,
        status=status,
        budget_s=12.0,
        elapsed_s=11.5,
        vehicles_offered=offered,
        stops=20,
        solution_limit=solution_limit,
    )


def metrics(cost_per_drop: float) -> Metrics:
    """A ``Metrics`` carrying only what the report reads, so a test states its own inputs."""
    breakdown = CostBreakdown(
        variable_inr=Rupees(1.0),
        driver_inr=Rupees(1.0),
        fixed_inr=Rupees(1.0),
        tw_penalty_inr=Rupees(1.0),
    )
    return Metrics(
        total_cost_inr=Rupees(cost_per_drop * 100.0),
        cost_per_drop_inr=Rupees(cost_per_drop),
        total_distance_km=10.0,
        total_duration_hr=2.0,
        vehicles_used=3,
        tw_violations=1,
        tw_lateness_hr=0.5,
        stops_per_hour=4.0,
        capacity_utilisation=0.5,
        breakdown=breakdown,
    )


def comparison(ga: float, reference: float) -> Comparison:
    """Two scored columns, built directly so a verdict test states the figures it asserts on."""
    return Comparison(
        ga=metrics(ga),
        reference=metrics(reference),
        ga_stage2_inr=Rupees(ga * 50.0),
        reference_stage2_inr=Rupees(reference * 50.0),
    )


def test_an_incomplete_plan_yields_no_scored_comparison() -> None:
    """The load-bearing branch: a plan missing a hub has no honest cost per drop."""
    plan = ReferencePlan(hubs=(hub(0), hub(1, solved=False, status=_TIMEOUT)), routes=())

    assert not plan.complete
    assert compare(_UNUSED_INSTANCE, (), (), plan, CostConfig()) is None
    assert verdict_lines(None) == [
        "No gap can be reported: the reference has no complete plan, so it has no cost per",
        "drop. See the per-hub status above.",
    ]


def test_an_unsolved_hub_is_named_and_the_refusal_to_substitute_is_stated() -> None:
    """A dropped hub must be visible, and the report must say nothing was put in its place."""
    plan = ReferencePlan(hubs=(hub(0), hub(7, solved=False, status=_TIMEOUT)), routes=())

    text = "\n".join(status_lines(plan))

    assert "no solution on 1 of 2 hubs" in text
    assert "hub 7" in text
    assert "substituted" in text
    assert "finding about the budget, not a failed run" in text


def test_an_all_solved_run_still_prints_every_status() -> None:
    """Success on every hub is the evidence the budget sufficed, and must not be left implicit."""
    plan = ReferencePlan(hubs=(hub(0), hub(1)), routes=())

    text = "\n".join(status_lines(plan))

    assert "ROUTING_SUCCESS" in text
    assert "no solution on" not in text
    assert "still improving" not in text


@pytest.mark.parametrize(
    ("ga", "reference", "leader"),
    [(100.0, 120.0, "GA"), (120.0, 100.0, "reference")],
)
def test_the_verdict_names_whichever_solver_actually_won(
    ga: float, reference: float, leader: str
) -> None:
    """Written to be able to report that the GA lost — §1.1 forbids tuning the gap away."""
    lines = verdict_lines(comparison(ga, reference))

    assert f"the {leader} is ahead" in lines[0]
    assert "not tuned against this figure" in lines[1]


def test_a_tie_is_reported_as_a_tie_rather_than_a_zero_percent_win() -> None:
    """A signed zero would read as a win for whichever column came second."""
    lines = verdict_lines(comparison(100.0, 100.0))

    assert lines == ["The GA and the reference tie at ₹100.00 per drop."]


def test_hubs_cut_off_mid_descent_make_the_gap_a_bound_and_say_so() -> None:
    """The mirror of the ablation's truncation caveat, in the reference's direction."""
    plan = ReferencePlan(hubs=(hub(0, status=_TRUNCATED), hub(1)), routes=())

    text = "\n".join(status_lines(plan))

    assert "1 of 2 hubs were still improving" in text
    assert "upper bound on the GA's advantage" in text


def test_a_hub_at_its_fleet_ceiling_is_flagged_as_possibly_constrained() -> None:
    """Deploying every vehicle offered means the cap, not the search, may have set the answer."""
    plan = ReferencePlan(hubs=(hub(0, status=_TRUNCATED, fleet=(2, 2)),), routes=())

    text = "\n".join(status_lines(plan))

    assert "deployed every vehicle offered" in text
    assert "hub 0" in text


def test_a_first_solution_capped_reference_warns_that_this_is_not_the_measured_gap() -> None:
    """``--deterministic`` buys reproducibility by crippling the reference; that must be loud.

    Without this the table looks like a measurement and reads as a large GA win, when what it
    actually shows is the GA against an unoptimised construction heuristic.
    """
    plan = ReferencePlan(hubs=(hub(0, solution_limit=1), hub(1, solution_limit=1)), routes=())

    text = "\n".join(budget_lines(plan))

    assert "WARNING" in text
    assert "NOT the measured step 8 gap" in text
    assert "Drop --deterministic to measure" in text


def test_a_normally_budgeted_reference_carries_no_such_warning() -> None:
    """The caveat must not become decoration that appears on every run."""
    plan = ReferencePlan(hubs=(hub(0), hub(1)), routes=())

    assert "WARNING" not in "\n".join(budget_lines(plan))


def test_the_budget_block_reports_both_the_per_hub_spread_and_the_totals() -> None:
    """A matched-budget claim invites exactly this question, so both must be printed."""
    plan = ReferencePlan(hubs=(hub(0), hub(1)), routes=())

    text = "\n".join(budget_lines(plan))

    assert "per-hub budget" in text
    assert "reference total" in text
    assert "matched-budget, not matched-to-convergence" in text


def test_the_report_degrades_to_status_and_caveats_when_there_is_nothing_to_score() -> None:
    """End to end on the branch a real run may never take: no table, no headline, no gap."""
    plan = ReferencePlan(hubs=(hub(0, solved=False, status=_TIMEOUT),), routes=())

    text = "\n".join(report_lines(_UNUSED_INSTANCE, (), (), plan, CostConfig()))

    assert "cost per drop ₹" not in text
    assert "No gap can be reported" in text
    assert "Per-hub solver status" in text
    assert "static dispatch-hour traffic multiplier" in text
