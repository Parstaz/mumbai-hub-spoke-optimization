"""Tests for the per-hub evolution loop.

A GA is easy to test badly. Asserting that it finds a particular tour makes the suite a record of
one seed's trajectory, and asserting that it beats some fixed number makes it a tuning target that
will be quietly relaxed the first time it fails. So the properties here are structural: the answer
is a valid chromosome, the run is reproducible from its seed, the incumbent never gets worse, the
search objective never leaks into the reported one, and the ablation switch actually switches
something off.

Instances are tiny and the generation budget is small, because none of that depends on scale.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from src.config import CostConfig, FleetConfig, GAConfig, GeoConfig, ScheduleConfig
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import (
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.stage2.ga import evolve
from src.stage2.penalty import AdaptivePenalty
from src.stage2.population import initial_population, nearest_neighbour_order
from src.stage2.pricing import hub_pricing
from src.stage2.split import SplitContext, split
from src.tour import RoutingContext
from src.units import DistanceMatrix, NodeId, Seconds
from src.workload import HubWorkload

CAPACITY_KG = 750.0
PARCEL_KG = 37.5
METRES_PER_SECOND = 10.0
EIGHT_AM_S = Seconds(8 * 3600.0)
SERVICE_S = Seconds(300.0)

FLAT = TrafficModel(hourly_multipliers=(1.0,) * 24)

SMALL_GA = GAConfig(
    population_size=12,
    generations=8,
    tournament_k=3,
    elitism_count=2,
    local_search_pct=0.25,
    stagnation_limit=100,
    seeded_individuals=2,
    penalty_adapt_interval=3,
)
"""A run small enough to execute in a test but structurally identical to a real one."""


def ring_matrix(n_nodes: int, seed: int) -> DistanceMatrix:
    """An asymmetric random matrix, so a tour and its reverse are genuinely different."""
    rng = np.random.default_rng(seed)
    matrix: DistanceMatrix = rng.uniform(300.0, 12_000.0, size=(n_nodes, n_nodes))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def make_context(
    n_stops: int,
    windows: tuple[TimeWindow | None, ...] | None = None,
    seed: int = 5,
    demand_kg: float = PARCEL_KG,
) -> SplitContext:
    """A hub of ``n_stops`` customers over an asymmetric random network."""
    resolved = windows or (None,) * n_stops
    geo = GeoConfig(
        n_hubs=1, n_sources=1, n_customers=n_stops, n_density_clusters=1, hub_candidate_pool=10
    )
    instance = Instance(
        seed=0,
        geo=geo,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)),),
        sources=(Source(0, Coordinate(19.01, 72.91)),),
        customers=tuple(
            Customer(index, Coordinate(19.02 + 0.001 * index, 72.92), resolved[index])
            for index in range(n_stops)
        ),
        shipments=tuple(
            Shipment(index, source_id=0, customer_id=index, size_kg=PARCEL_KG)
            for index in range(n_stops)
        ),
    )
    matrix = ring_matrix(instance.n_nodes, seed)
    workload = HubWorkload(
        hub_id=0,
        hub_node=NodeId(0),
        nodes=np.array([instance.customer_node(index) for index in range(n_stops)], dtype=np.intp),
        demand_kg=np.full(n_stops, demand_kg, dtype=np.float64),
    )
    return SplitContext(
        workload=workload,
        routing=RoutingContext(
            matrices=CostMatrices(distance_m=matrix, duration_s=matrix / METRES_PER_SECOND),
            traffic=FLAT,
            start_time_s=EIGHT_AM_S,
            service_time_s=SERVICE_S,
            capacity_kg=CAPACITY_KG,
        ),
        pricing=hub_pricing(workload, instance),
        cost_config=CostConfig(),
    )


# --------------------------------------------------------------------------------------------
# What a run returns
# --------------------------------------------------------------------------------------------


def test_the_result_is_a_chromosome_the_split_accepts() -> None:
    """A GA that returned a broken permutation would be caught by split, but only downstream."""
    context = make_context(9)
    outcome = evolve(context, SMALL_GA, np.random.default_rng(1))
    assert sorted(outcome.permutation) == list(range(9))
    assert split(outcome.permutation, context).search_objective_inr == outcome.objective_inr


def test_the_outcome_is_tagged_with_the_hub_it_belongs_to() -> None:
    """Results come back from the pool in completion order, so they must identify themselves.

    An untagged outcome reassembled against the wrong hub's workload yields a complete,
    capacity-legal plan that delivers to the wrong customers — the expensive kind of wrong.
    """
    context = make_context(6)
    assert evolve(context, SMALL_GA, np.random.default_rng(1)).hub_id == context.workload.hub_id


def test_the_reported_objective_is_priced_at_the_configured_rates() -> None:
    """The adaptive multiplier must not reach the number the run reports.

    Windows here are unmeetable, so the penalty climbs away from 1.0 during the run. The returned
    objective still has to be exactly what splitting that chromosome at the *configured* rates
    costs — if the multiplier leaked, this is where it would show.
    """
    impossible = TimeWindow(Seconds(8 * 3600.0), Seconds(8 * 3600.0 + 60.0))
    context = make_context(8, (impossible,) * 8)
    outcome = evolve(context, SMALL_GA, np.random.default_rng(2))
    assert outcome.final_multiplier > 1.0, "the fixture is meant to drive the penalty up"
    assert outcome.objective_inr == split(outcome.permutation, context).search_objective_inr


def test_a_run_is_reproducible_from_its_seed() -> None:
    """Two runs of one seed agree exactly, which is what a multi-seed evaluation rests on.

    The generator is injected rather than global precisely so this holds under the per-hub process
    pool — see CLAUDE.md §1.1 and its gotcha about a shared generator.
    """
    context = make_context(10)
    first = evolve(context, SMALL_GA, np.random.default_rng(7))
    second = evolve(context, SMALL_GA, np.random.default_rng(7))
    assert first == second


def test_tracing_does_not_change_the_search(caplog: pytest.LogCaptureFixture) -> None:
    """Instrumentation must be read-only, or it measures a run that would not otherwise happen.

    This is the guarantee that lets §8.5's open question be settled by turning tracing on rather
    than by changing a setting: the traced run *is* the untraced run, so nothing observed in it is
    an artefact of observing it.
    """
    context = make_context(9)
    plain = evolve(context, SMALL_GA, np.random.default_rng(5))
    traced_config = dataclasses.replace(SMALL_GA, trace_generations=True)
    with caplog.at_level(logging.INFO, logger="src.stage2.ga"):
        traced = evolve(context, traced_config, np.random.default_rng(5))
    assert traced == plain
    assert any("population min" in record.message for record in caplog.records)


def test_tracing_reports_when_the_incumbent_missed_a_better_plan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The line that would confirm the mechanism has to be emitted when the condition holds.

    ``_configured_best`` re-prices only the champion by search objective, so whenever the
    population's configured-rate minimum beats the incumbent, an improvement was available and
    invisible. The trace has to say so explicitly rather than leave it to be derived.
    """
    impossible = TimeWindow(Seconds(8 * 3600.0), Seconds(8 * 3600.0 + 60.0))
    context = make_context(9, (impossible,) * 9)
    traced_config = dataclasses.replace(SMALL_GA, trace_generations=True)
    with caplog.at_level(logging.INFO, logger="src.stage2.ga"):
        evolve(context, traced_config, np.random.default_rng(6))
    traced = [record.message for record in caplog.records if "gen" in record.message]
    assert traced, "a traced run must emit one line per generation"
    assert all("incumbent" in line and "population min" in line for line in traced)


def test_different_seeds_explore_differently() -> None:
    """If the seed did not matter, a multi-seed evaluation would be reporting one run n times."""
    context = make_context(12)
    outcomes = {
        evolve(context, SMALL_GA, np.random.default_rng(seed)).permutation for seed in range(6)
    }
    assert len(outcomes) > 1


def test_the_incumbent_is_never_worse_than_the_first_generation() -> None:
    """Elitism plus an incumbent tracked at configured rates means the answer only improves.

    Not a quality claim — the GA is not asserted to beat anything here — but a monotonicity one:
    a run that ended worse than it started would mean the incumbent is being overwritten by a
    number it is not comparable with.
    """
    context = make_context(11)
    rng = np.random.default_rng(3)
    generation_zero = min(
        float(split(order, context).search_objective_inr)
        for order in initial_population(context, SMALL_GA, np.random.default_rng(3))
    )
    assert float(evolve(context, SMALL_GA, rng).objective_inr) <= generation_zero


# --------------------------------------------------------------------------------------------
# The knobs do what they say
# --------------------------------------------------------------------------------------------


def test_local_search_can_be_switched_off_entirely() -> None:
    """Step 7's ablation arm. At zero the memetic step must not run at all.

    Asserted through behaviour: with local search off the run is still valid and reproducible, and
    it differs from the same seed with local search on — otherwise the ablation would be comparing
    a configuration against itself.
    """
    context = make_context(10)
    without = dataclasses.replace(SMALL_GA, local_search_pct=0.0)
    plain = evolve(context, without, np.random.default_rng(4))
    memetic = evolve(context, SMALL_GA, np.random.default_rng(4))
    assert sorted(plain.permutation) == list(range(10))
    assert plain != memetic


def test_stagnation_stops_the_run_before_the_generation_budget() -> None:
    """A converged hub should not spend 600 generations proving it has converged."""
    context = make_context(4)
    impatient = dataclasses.replace(SMALL_GA, generations=200, stagnation_limit=2)
    outcome = evolve(context, impatient, np.random.default_rng(8))
    assert outcome.generations_run < impatient.generations


def test_the_penalty_relaxes_when_every_window_is_met() -> None:
    """With no windows to miss, the search has no reason to keep over-pricing lateness.

    It floors at 1.0 rather than falling through it, which is the asymmetry
    :mod:`src.stage2.penalty` exists to enforce.
    """
    context = make_context(8)
    outcome = evolve(context, SMALL_GA, np.random.default_rng(9))
    assert outcome.final_multiplier == pytest.approx(AdaptivePenalty.neutral().multiplier)


# --------------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------------


def test_a_hub_with_one_stop_still_produces_a_plan() -> None:
    """One stop admits one order; the loop must not divide by zero or refuse to start."""
    context = make_context(1)
    outcome = evolve(context, SMALL_GA, np.random.default_rng(6))
    assert outcome.permutation == (0,)


def test_a_hub_needing_several_vehicles_is_searched_as_one_chromosome() -> None:
    """The chromosome carries no delimiters however many vehicles the split ends up deploying."""
    context = make_context(9, demand_kg=300.0)
    outcome = evolve(context, SMALL_GA, np.random.default_rng(10))
    assert len(split(outcome.permutation, context).routes) >= 3
    assert sorted(outcome.permutation) == list(range(9))


# --------------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------------


def test_nearest_neighbour_seeds_differ_by_starting_stop() -> None:
    """Distinct starts give distinct tours — the reason starts are drawn without replacement."""
    context = make_context(8)
    orders = {
        nearest_neighbour_order(context.workload, context.routing, start) for start in range(8)
    }
    assert len(orders) > 1


def test_a_nearest_neighbour_seed_visits_every_stop_once() -> None:
    """A seed is a chromosome like any other and has to satisfy the same invariant."""
    context = make_context(8)
    assert sorted(nearest_neighbour_order(context.workload, context.routing, 3)) == list(range(8))


def test_a_seed_beats_a_random_order_on_average() -> None:
    """The point of seeding: selection starts with something worth selecting.

    Averaged over random orders rather than compared against one, so the assertion is about the
    construction rather than about which random draw the seed happened to face.
    """
    context = make_context(12)
    rng = np.random.default_rng(12)
    seeded = float(
        split(
            nearest_neighbour_order(context.workload, context.routing, 0), context
        ).search_objective_inr
    )
    random_mean = float(
        np.mean(
            [
                float(
                    split(tuple(int(p) for p in rng.permutation(12)), context).search_objective_inr
                )
                for _ in range(30)
            ]
        )
    )
    assert seeded < random_mean


def test_the_population_is_exactly_the_configured_size() -> None:
    """Seeds occupy slots rather than adding to them, whatever the hub's stop count."""
    context = make_context(2)
    population = initial_population(context, SMALL_GA, np.random.default_rng(0))
    assert len(population) == SMALL_GA.population_size
    assert all(sorted(order) == [0, 1] for order in population)
