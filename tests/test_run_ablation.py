"""Tests for step 7's ablation — the parts that are logic rather than presentation.

CLI entry points are exempt from the coverage standard (CLAUDE.md §3), but four things here decide
whether the reported result means anything and are pinned accordingly: that both arms of a strategy
deliver from the *same* inbound plan, that the noise probe moves the GA seed and nothing else, that
the truncation caveat travels with the table, and that the verdict is capable of reporting a loss.

No cost is asserted anywhere. A suite that pinned the ablation's number would turn it into a tuning
target, which is exactly what step 7 must not become.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.baseline.greedy import solve_baseline
from src.cli.ablation import (
    Arm,
    Priced,
    Problem,
    inbound_leg,
    outbound_arm,
    priced,
    probe_arms,
    solve_arms,
    without_local_search,
)
from src.cli.run_ablation import (
    generation_lines,
    main,
    probe_lines,
    table_lines,
    truncation_lines,
    verdict_lines,
)
from src.config import Config, GAConfig, GeoConfig, RunConfig, Stage1Config
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.scoring import CostBreakdown, Metrics, stage_cost
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
    # cvrp_solution_limit=1 is not optional: Stage 1's guided local search under a wall-clock
    # limit returns whatever it reached when the clock ran out, so a busier machine would yield a
    # different inbound plan and these assertions would drift with the load average.
    stage1=Stage1Config(cvrp_solution_limit=1, workers=1),
    run=RunConfig(seed=11, use_osrm=False),
)


@pytest.fixture(name="problem")
def problem_fixture() -> Problem:
    """The small instance, priced by the haversine fallback so no test touches the network."""
    instance = generate_instance(SMALL.geo, SMALL.fleet, SMALL.schedule, SMALL.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=SMALL.run.circuity_factor, speed_kmph=SMALL.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    return Problem(
        instance=instance,
        matrices=CostMatrices(distance_m=distance_m, duration_s=duration_s),
        traffic=TrafficModel.from_config(SMALL.traffic),
    )


def fake_arm(strategy: str, local_search: bool, cost_per_drop: float, **kwargs: float) -> Arm:
    """An arm with a chosen headline figure, for testing what the report says about numbers.

    Hand-built rather than solved: the verdict and caveat renderers have to be exercised on
    outcomes a real run may not produce on demand — a local search that lost, or a run that spent
    its whole budget — and waiting for one to appear is not a test.
    """
    metrics = Metrics(
        total_cost_inr=Rupees(cost_per_drop * 10.0),
        cost_per_drop_inr=Rupees(cost_per_drop),
        total_distance_km=100.0,
        total_duration_hr=10.0,
        vehicles_used=4,
        tw_violations=1,
        tw_lateness_hr=0.5,
        stops_per_hour=2.0,
        capacity_utilisation=0.5,
        breakdown=CostBreakdown(Rupees(1.0), Rupees(2.0), Rupees(3.0), Rupees(4.0)),
    )
    return Arm(
        strategy=strategy,
        local_search=local_search,
        ga_seed=int(kwargs.get("ga_seed", 42)),
        priced=Priced(
            metrics=metrics,
            stage1_inr=Rupees(kwargs.get("stage1_inr", 500.0)),
            stage2_inr=Rupees(kwargs.get("stage2_inr", 500.0)),
        ),
        outcomes=(
            HubOutcome(
                hub_id=0,
                permutation=(0,),
                objective_inr=Rupees(1.0),
                generations_run=int(kwargs.get("generations_run", 40)),
                final_multiplier=1.0,
            ),
        ),
        elapsed_s=1.0,
    )


def test_both_arms_of_a_strategy_deliver_from_the_inbound_plan_they_were_given(
    problem: Problem,
) -> None:
    """The load-bearing design property: Stage 1 is solved once and shared by both arms.

    Stage 1's guided local search is not reproducible, so an ablation that re-solved it per arm
    would measure local search across two different inbound plans and two different customer-to-hub
    mappings. Asserted through the priced result rather than by counting calls: if either arm had
    solved its own inbound leg, its stage 1 column would stop matching the leg it was handed.
    """
    leg = inbound_leg("nearest", problem, SMALL)
    expected = stage_cost(leg.inbound, problem.instance, SMALL.cost).breakdown.total_inr

    arms = solve_arms([leg], problem, SMALL)

    assert [arm.local_search for arm in arms] == [True, False]
    assert all(arm.priced.stage1_inr == expected for arm in arms)
    assert all(arm.strategy == "nearest" for arm in arms)


def test_the_two_by_two_runs_every_cell_once(problem: Problem) -> None:
    """Four arms, one per cell, and no cell reported twice."""
    legs = [inbound_leg(name, problem, SMALL) for name in ("nearest", "balanced")]

    arms = solve_arms(legs, problem, SMALL)

    assert len(arms) == 4
    assert len({(arm.strategy, arm.local_search) for arm in arms}) == 4


def test_switching_local_search_off_changes_nothing_else() -> None:
    """The ablation arm must differ from the shipping config in exactly one field.

    Any second difference would make the measured effect unattributable, which is the failure this
    whole exercise exists to avoid.
    """
    off = without_local_search(SMALL)

    assert off.ga.local_search_pct == 0.0
    assert dataclasses.replace(off.ga, local_search_pct=SMALL.ga.local_search_pct) == SMALL.ga
    assert (off.run, off.stage1, off.cost, off.geo) == (
        SMALL.run,
        SMALL.stage1,
        SMALL.cost,
        SMALL.geo,
    )


def test_the_noise_probe_moves_the_ga_seed_and_holds_the_problem_fixed(problem: Problem) -> None:
    """The probe has to measure search noise, not a different problem.

    If it regenerated the instance it would be measuring instance-to-instance variation and the
    spread would be meaningless as an error bar on this instance's 2x2. The inbound plan staying
    priced identically is what proves only the GA was reseeded.
    """
    leg = inbound_leg("nearest", problem, SMALL)
    reference = outbound_arm(leg, problem, SMALL)

    samples = probe_arms(leg, problem, SMALL, 2)

    assert [arm.ga_seed for arm in samples] == [SMALL.run.seed + 1, SMALL.run.seed + 2]
    assert all(arm.priced.stage1_inr == reference.priced.stage1_inr for arm in samples)
    assert all(arm.local_search == reference.local_search for arm in samples)


def test_the_probe_is_reproducible_from_its_seeds(problem: Problem) -> None:
    """A noise floor that moved between runs would not be a floor."""
    leg = inbound_leg("nearest", problem, SMALL)

    first = probe_arms(leg, problem, SMALL, 1)
    second = probe_arms(leg, problem, SMALL, 1)

    assert [arm.priced.metrics for arm in first] == [arm.priced.metrics for arm in second]


def test_the_table_has_a_column_per_arm_and_the_benchmark(problem: Problem) -> None:
    """A table that silently dropped an arm would still render, so the shape is pinned."""
    leg = inbound_leg("nearest", problem, SMALL)
    arms = solve_arms([leg], problem, SMALL)
    benchmark = priced(
        solve_baseline(problem.instance, problem.matrices, problem.traffic),
        problem.instance,
        SMALL.cost,
    )

    lines = table_lines(benchmark, arms, SMALL.cost)

    assert lines[0].split() == ["greedy", "nearest", "nearest"]
    assert lines[1].split() == ["benchmark", "with", "l.s.", "no", "l.s."]
    assert lines[3].startswith("cost per drop")
    assert lines[4].startswith("vs greedy")
    # Header, second header, rule, headline, vs greedy, then one line per remaining row.
    assert len(lines) == 5 + 12


def test_the_headline_row_is_the_total_and_the_legs_sit_beneath_it() -> None:
    """Reading the two legs separately is what hid step 4's cross-stage interaction.

    So the total is the headline and the legs are indented under the cost components. The row order
    is the argument, and it is pinned here rather than trusted to survive an edit.
    """
    arms = [fake_arm("nearest", True, 100.0), fake_arm("nearest", False, 110.0)]

    labels = [line.split("₹")[0].rstrip() for line in table_lines(arms[0].priced, arms, SMALL.cost)]

    assert labels[3] == "cost per drop"
    assert labels.index("  stage 1 inbound") > labels.index("total cost")
    assert labels.index("  stage 2 final mile") > labels.index("  stage 1 inbound")


def test_generations_are_reported_for_every_arm() -> None:
    """A figure quoted against the budget rather than against what ran is not about that arm."""
    arms = [
        fake_arm("nearest", True, 100.0),
        fake_arm("nearest", False, 110.0),
        fake_arm("balanced", True, 105.0),
        fake_arm("balanced", False, 115.0),
    ]

    lines = generation_lines(arms, SMALL.ga)

    assert f"against the {SMALL.ga.generations} budget" in lines[0]
    assert len(lines) == 1 + len(arms)
    for arm, line in zip(arms, lines[1:], strict=True):
        assert arm.strategy in line and arm.search_label in line


def test_the_truncation_caveat_travels_with_the_table() -> None:
    """The caveat has to be in the output, because the output is what gets copied out."""
    truncated = [fake_arm("nearest", True, 100.0, generations_run=SMALL.ga.generations - 1)]

    lines = truncation_lines(truncated, SMALL.ga)

    assert any("lower bound" in line for line in lines)
    assert any("not the differences" in line for line in lines)


def test_the_caveat_is_silent_when_every_arm_spent_its_budget() -> None:
    """The boundary: an arm landing exactly on the budget did not stop early."""
    full = [fake_arm("nearest", True, 100.0, generations_run=SMALL.ga.generations)]

    assert truncation_lines(full, SMALL.ga) == []
    assert truncation_lines([], SMALL.ga) == []


def test_the_caveat_refuses_to_claim_symmetry_it_has_not_checked() -> None:
    """The real seed-42 shape: the budget ceiling was hit in one arm and not the other.

    Hub 9 exhausted the 600 generations in both no-local-search arms and in neither local-search
    arm, so the arms were *not* truncated identically and the direction is not neutral — the
    comparison arm was cut off rather than converged. A caveat asserting symmetry here would be
    false about the run printing it, which is worse than no caveat.
    """
    arms = [
        fake_arm("nearest", True, 100.0, generations_run=SMALL.ga.generations - 1),
        fake_arm("nearest", False, 110.0, generations_run=SMALL.ga.generations),
    ]

    lines = truncation_lines(arms, SMALL.ga)

    assert any("NOT truncated identically" in line for line in lines)
    assert any("upper bound on the other arm's benefit" in line for line in lines)
    assert any(
        "nearest with l.s. 0 of 1" in line and "nearest no l.s. 1 of 1" in line for line in lines
    )
    assert not any("applies to all arms identically" in line for line in lines)


def test_the_symmetric_claim_survives_when_the_counts_do_agree() -> None:
    """Equal truncation is the case the original caveat was written for, and it still holds."""
    arms = [
        fake_arm("nearest", True, 100.0, generations_run=SMALL.ga.generations - 1),
        fake_arm("nearest", False, 110.0, generations_run=SMALL.ga.generations - 2),
    ]

    lines = truncation_lines(arms, SMALL.ga)

    assert any("applies to all arms identically" in line for line in lines)
    assert not any("NOT truncated identically" in line for line in lines)


def test_the_verdict_reports_local_search_losing_as_a_loss() -> None:
    """The failure mode this ablation must be able to report.

    Capacity-balanced assignment not paying on the inbound leg shipped as a negative result; if
    local search does not pay, this is the function that has to say so rather than soften it.
    """
    arms = [
        fake_arm("nearest", True, 110.0),
        fake_arm("nearest", False, 100.0),
        fake_arm("balanced", True, 112.0),
        fake_arm("balanced", False, 102.0),
    ]

    lines = verdict_lines(arms)

    assert "₹100.00 without → ₹110.00 with, +10.0% per drop." in lines[0]
    assert "agree on the sign" in lines[2]


def test_the_verdict_says_so_when_the_two_assignments_disagree() -> None:
    """An effect with a different sign under each assignment has not been shown to exist."""
    arms = [
        fake_arm("nearest", True, 90.0),
        fake_arm("nearest", False, 100.0),
        fake_arm("balanced", True, 112.0),
        fake_arm("balanced", False, 102.0),
    ]

    lines = verdict_lines(arms)

    assert "disagree on the sign" in lines[2]
    assert "not separable from the assignment" in lines[2]


def test_the_assignment_verdict_contrasts_the_total_with_both_legs() -> None:
    """Step 4's verdict was inbound-only; the point of step 7 is to print every leg beside it."""
    arms = [
        fake_arm("nearest", True, 100.0, stage1_inr=1000.0, stage2_inr=2000.0),
        fake_arm("nearest", False, 105.0, stage1_inr=1000.0, stage2_inr=2100.0),
        fake_arm("balanced", True, 98.0, stage1_inr=1017.0, stage2_inr=1900.0),
        fake_arm("balanced", False, 103.0, stage1_inr=1017.0, stage2_inr=2050.0),
    ]

    lines = verdict_lines(arms)

    assert "-2.0% on total cost per drop" in lines[-3]
    assert "inbound +1.7%" in lines[-2]
    assert "final mile -5.0%" in lines[-2]
    assert "made the final mile cheaper" in lines[-2]


def test_a_coincidental_equality_cannot_read_as_stage_2_neutrality() -> None:
    """Seed 42's shape: total and inbound both land on +1.7% while the final mile is dearer.

    A reader seeing only those two percentages concludes Stage 2 contributed nothing. It
    contributed +1.8% in the same direction. Step 6's hypothesis was specifically about the final
    mile, so this is the row that decides it and it must be named rather than inferred.
    """
    arms = [
        fake_arm("nearest", True, 264.63, stage1_inr=66261.0, stage2_inr=145443.0),
        fake_arm("nearest", False, 272.36, stage1_inr=66261.0, stage2_inr=151631.0),
        fake_arm("balanced", True, 269.25, stage1_inr=67365.0, stage2_inr=148031.0),
        fake_arm("balanced", False, 275.50, stage1_inr=67365.0, stage2_inr=153038.0),
    ]

    lines = verdict_lines(arms)

    assert "+1.7% on total cost per drop" in lines[-3]
    assert "inbound +1.7%" in lines[-2] and "final mile +1.8%" in lines[-2]
    assert "made the final mile dearer" in lines[-2]
    assert "not the inbound figure passed through" in lines[-1]


def test_the_probe_says_whether_an_effect_clears_the_noise_floor() -> None:
    """An effect smaller than seed-to-seed variation has not been measured."""
    arms = [
        fake_arm("nearest", True, 100.0),
        fake_arm("nearest", False, 100.50),
        fake_arm("balanced", True, 100.0),
        fake_arm("balanced", False, 110.0),
    ]
    samples = [fake_arm("nearest", True, 102.0, ga_seed=43)]

    lines = probe_lines(arms[0], samples, arms)

    assert "spread" in lines[-3]
    assert "local search on nearest: ₹0.50, does not clear the noise floor." in lines[-2]
    assert "local search on balanced: ₹10.00, clears the noise floor." in lines[-1]


def test_the_entry_point_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """A smoke test of the real ``main`` at a budget a test can afford.

    ``--deterministic`` is required: CLAUDE.md's gotchas forbid tests depending on OR-Tools guided
    local search under a wall-clock limit. Deliberately asserts on structure, never on a cost.
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
            "--noise-probe",
            "1",
            "--deterministic",
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "cost per drop" in printed
    assert "nearest" in printed and "balanced" in printed
    assert "with l.s." in printed and "no l.s." in printed
    assert "generations used" in printed
    assert "Noise probe" in printed


def test_the_entry_point_can_skip_the_probe(capsys: pytest.CaptureFixture[str]) -> None:
    """The boundary on ``--noise-probe``: zero repeats runs the 2x2 and stops."""
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
            "--noise-probe",
            "0",
            "--deterministic",
        ]
    )

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "cost per drop" in printed
    assert "Noise probe" not in printed
