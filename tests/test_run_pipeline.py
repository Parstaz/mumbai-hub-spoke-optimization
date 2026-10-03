"""Tests for the pipeline entry point's own logic.

CLI entry points are exempt from the coverage standard (CLAUDE.md §3), but two things in here are
logic rather than presentation and are worth pinning: the stage composition, which is where a
customer could silently be served from the wrong hub, and the delta column, which is the number a
reader will quote.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.cli.format import delta, generations_used
from src.cli.run_pipeline import (
    budget_lines,
    comparison_lines,
    main,
    solve_pipeline,
)
from src.config import Config, GAConfig, GeoConfig, RunConfig, Stage1Config
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.scoring import evaluate_solution
from src.stage1.assignment import unconstrained
from src.stage2.ga import HubOutcome
from src.units import Rupees

SMALL = Config(
    geo=GeoConfig(
        n_hubs=3, n_sources=8, n_customers=14, n_density_clusters=2, hub_candidate_pool=100
    ),
    ga=GAConfig(
        population_size=8,
        generations=3,
        tournament_k=3,
        elitism_count=1,
        seeded_individuals=1,
        penalty_adapt_interval=2,
    ),
    stage1=Stage1Config(cvrp_solution_limit=1, workers=1),
    run=RunConfig(seed=11, use_osrm=False),
)


def test_the_composed_plan_passes_the_scorer() -> None:
    """Both legs together must satisfy every hard constraint, including completeness.

    The scorer is what would catch a customer served from a hub its parcel never reached — the
    stage would leave someone undelivered and it raises rather than reporting a cheaper plan.
    """
    instance = generate_instance(SMALL.geo, SMALL.fleet, SMALL.schedule, SMALL.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=SMALL.run.circuity_factor, speed_kmph=SMALL.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    matrices = CostMatrices(distance_m=distance_m, duration_s=duration_s)
    traffic = TrafficModel.from_config(SMALL.traffic)

    solution, _ = solve_pipeline(instance, matrices, traffic, unconstrained, SMALL)
    metrics = evaluate_solution(solution, instance, SMALL.cost)
    assert metrics.cost_per_drop_inr > 0.0
    assert len(solution.stage2_routes) > 0


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (100.0, 90.0, "-10.0%"),
        (100.0, 110.0, "+10.0%"),
        (100.0, 100.0, "+0.0%"),
        (0.0, 5.0, "—"),
    ],
)
def test_the_delta_column_reads_negative_for_an_improvement(
    before: float, after: float, expected: str
) -> None:
    """An improvement is a cost going down, so the sign has to say so without a legend.

    The zero-baseline case is the one that matters: a baseline with no window violations at all
    would otherwise divide by zero while rendering a table.
    """
    assert delta(before, after) == expected


def test_the_comparison_table_has_a_row_per_reported_measure() -> None:
    """Header, rule, then one row each — a table that silently dropped a row would still print."""
    instance = generate_instance(SMALL.geo, SMALL.fleet, SMALL.schedule, SMALL.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=SMALL.run.circuity_factor, speed_kmph=SMALL.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    matrices = CostMatrices(distance_m=distance_m, duration_s=duration_s)
    traffic = TrafficModel.from_config(SMALL.traffic)
    solution, _ = solve_pipeline(instance, matrices, traffic, unconstrained, SMALL)
    metrics = evaluate_solution(solution, instance, SMALL.cost)

    lines = comparison_lines(metrics, metrics, SMALL.cost)
    assert len(lines) == 13
    assert lines[0].split() == ["greedy", "pipeline", "change"]
    assert all("+0.0%" in line for line in lines[2:] if "—" not in line)


def test_the_budget_line_reports_what_was_used_not_what_was_configured() -> None:
    """The header describes the run, so it must not claim a budget the run did not spend.

    A hub stops when ``stagnation_limit`` generations pass without improving its incumbent, and on
    the default instance every hub does. A header reading "150x600" without this line describes a
    search that did not happen.
    """
    instance = generate_instance(SMALL.geo, SMALL.fleet, SMALL.schedule, SMALL.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=SMALL.run.circuity_factor, speed_kmph=SMALL.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    matrices = CostMatrices(distance_m=distance_m, duration_s=duration_s)
    traffic = TrafficModel.from_config(SMALL.traffic)
    _, plan = solve_pipeline(instance, matrices, traffic, unconstrained, SMALL)
    hubs = plan.outcomes

    lines = budget_lines(hubs, SMALL.ga)
    assert f"of {SMALL.ga.generations} budget" in lines[0]
    used = [outcome.generations_run for outcome in hubs]
    assert f"range {min(used)}-{max(used)}" in lines[0]


def test_the_budget_line_is_empty_when_no_hub_ran() -> None:
    """An instance with nothing to deliver prints no budget line rather than dividing by zero."""
    assert generations_used((), SMALL.ga) is None
    assert budget_lines((), SMALL.ga) == []


def test_the_generations_summary_counts_only_hubs_that_stopped_short() -> None:
    """The early-stop count is what tells a reader the budget was not the binding constraint.

    Pinned on hand-built outcomes rather than a real run so the boundary is exact: a hub landing
    *on* the budget spent it and has not stopped early, and the one a generation below it has.
    """
    ga = dataclasses.replace(SMALL.ga, generations=10)
    hubs = tuple(
        HubOutcome(
            hub_id=hub_id,
            permutation=(0,),
            objective_inr=Rupees(1.0),
            generations_run=ran,
            final_multiplier=1.0,
        )
        for hub_id, ran in enumerate((4, 9, 10))
    )

    summary = generations_used(hubs, ga)
    assert summary is not None
    assert (summary.median, summary.lowest, summary.highest) == (9, 4, 10)
    assert (summary.stopped_early, summary.hubs) == (2, 3)


def test_the_entry_point_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """A smoke test of the real ``main``, at a budget a test can afford.

    ``--deterministic`` is not optional here. Stage 1's guided local search under a wall-clock
    limit returns whatever it reached when the clock ran out, so a busier machine yields a
    different plan — CLAUDE.md's gotchas require tests to stop at the first-solution heuristic.

    Deliberately not asserting a cost: a suite that pinned the pipeline's number would become a
    tuning target, and the run's own output is what step 6's verification inspects.
    """
    exit_code = main(
        [
            "--seed",
            "11",
            "--no-osrm",
            "--generations",
            "2",
            "--population",
            "8",
            "--workers",
            "1",
            "--deterministic",
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "cost per drop" in printed
    assert "greedy" in printed and "pipeline" in printed
