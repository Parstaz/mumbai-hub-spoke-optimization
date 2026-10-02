"""Tests for the Stage 2 orchestration: what crosses the process boundary, and what comes back.

Two concerns. The first is plumbing that is easy to get subtly wrong and impossible to notice: a
worker's node ids are local to its slice, so an off-by-``n_hubs`` in the translation produces a
complete, capacity-legal, plausible plan that delivers to the wrong customers. The scorer catches
that, but only because it checks completeness — these tests catch it here.

The second is CLAUDE.md §1.1's rule about what a worker may know. That one cannot be caught
downstream at all: a shared generator or a mutable config still produces a correct-looking run,
and only stops the numbers repeating. :func:`test_a_worker_payload_carries_no_shared_state` is a
structural check because there is no behavioural one.
"""

from __future__ import annotations

import dataclasses
import math

from src.config import Config, GAConfig, GeoConfig, RunConfig, Stage1Config
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.data.instance import Instance
from src.scoring import evaluate_solution
from src.solution import Route, Solution
from src.stage1.assignment import unconstrained
from src.stage1.cvrp import solve_stage1
from src.stage2.solve import Stage2Task, hub_of_source, solve_stage2
from src.workload import hub_of_customer

SMALL_GEO = GeoConfig(
    n_hubs=3,
    n_sources=8,
    n_customers=14,
    n_density_clusters=2,
    hub_candidate_pool=100,
)

TINY_GA = GAConfig(
    population_size=8,
    generations=4,
    tournament_k=3,
    elitism_count=1,
    local_search_pct=0.25,
    seeded_individuals=1,
    penalty_adapt_interval=2,
)


def small_config(workers: int = 1) -> Config:
    """A run small enough for a test but structurally identical to a real one."""
    return Config(
        geo=SMALL_GEO,
        ga=TINY_GA,
        stage1=Stage1Config(cvrp_solution_limit=1, workers=workers),
        run=RunConfig(seed=11, use_osrm=False),
    )


def build(config: Config) -> tuple[Instance, CostMatrices, TrafficModel]:
    """An instance and its haversine matrices — no network, per CLAUDE.md §3."""
    instance = generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=config.run.circuity_factor, speed_kmph=config.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    return (
        instance,
        CostMatrices(distance_m=distance_m, duration_s=duration_s),
        TrafficModel.from_config(config.traffic),
    )


def inbound_leg(
    instance: Instance, matrices: CostMatrices, traffic: TrafficModel, config: Config
) -> tuple[Route, ...]:
    """Stage 1's tours, which is where Stage 2 reads its hub assignment from."""
    return solve_stage1(instance, matrices, traffic, unconstrained, config)


# --------------------------------------------------------------------------------------------
# Reading Stage 1's assignment back off its plan
# --------------------------------------------------------------------------------------------


def test_hub_of_source_recovers_the_hub_that_collected_each_source() -> None:
    """Every source a tour visited is recorded against the hub whose tour that was."""
    config = small_config()
    instance, matrices, traffic = build(config)
    routes = inbound_leg(instance, matrices, traffic, config)
    hubs = hub_of_source(instance, routes)
    for route in routes:
        for node in route.interior_nodes:
            assert hubs[int(node) - len(instance.hubs)] == route.hub_id


def test_a_source_no_shipment_originates_at_is_never_looked_up() -> None:
    """Idle sources keep the fill value, and nothing may read it.

    A source with nothing waiting is skipped by Stage 1 rather than visited, so it has no hub.
    That is only safe while :func:`~src.workload.hub_of_customer` reads exactly the sources a
    shipment names — this asserts the two agree.
    """
    config = small_config()
    instance, matrices, traffic = build(config)
    hubs = hub_of_source(instance, inbound_leg(instance, matrices, traffic, config))
    named = {shipment.source_id for shipment in instance.shipments}
    assert all(hubs[source_id] >= 0 for source_id in named)
    idle = set(range(len(instance.sources))) - named
    assert all(hubs[source_id] == -1 for source_id in idle)


def test_a_customer_is_served_from_the_hub_its_shipment_reached() -> None:
    """The composition the whole two-stage decomposition rests on, end to end."""
    config = small_config()
    instance, matrices, traffic = build(config)
    routes = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, routes))
    landed = {int(node): route.hub_id for route in routes for node in route.interior_nodes}
    for shipment in instance.shipments:
        assert (
            customer_hubs[shipment.customer_id] == landed[instance.source_node(shipment.source_id)]
        )


# --------------------------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------------------------


def test_the_plan_delivers_to_every_customer_exactly_once() -> None:
    """A local-to-global translation error produces a plausible plan for the wrong customers."""
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    outbound = solve_stage2(instance, matrices, traffic, customer_hubs, config).routes

    served = [int(node) for route in outbound for node in route.interior_nodes]
    expected = [instance.customer_node(c.customer_id) for c in instance.customers]
    assert sorted(served) == sorted(expected)


def test_every_tour_leaves_from_the_hub_that_holds_its_parcels() -> None:
    """A worker's hub is local node 0; the parent has to put the right global hub back."""
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    outbound = solve_stage2(instance, matrices, traffic, customer_hubs, config).routes

    for route in outbound:
        assert route.nodes[0] == instance.hub_node(route.hub_id)
        for node in route.interior_nodes:
            assert customer_hubs[int(node) - len(instance.hubs) - len(instance.sources)] == (
                route.hub_id
            )


def test_the_whole_plan_scores_through_the_single_scoring_path() -> None:
    """Both legs together must satisfy every hard constraint ``evaluate_solution`` asserts."""
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    outbound = solve_stage2(instance, matrices, traffic, customer_hubs, config).routes

    metrics = evaluate_solution(
        Solution(stage1_routes=inbound, stage2_routes=outbound), instance, config.cost
    )
    assert metrics.cost_per_drop_inr > 0.0
    assert metrics.vehicles_used == len(inbound) + len(outbound)


def test_a_run_is_reproducible_across_repeats() -> None:
    """The per-hub seed is derived from the run seed and the hub id, so the answer is fixed."""
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))

    first = solve_stage2(instance, matrices, traffic, customer_hubs, config).routes
    second = solve_stage2(instance, matrices, traffic, customer_hubs, config).routes
    assert [route.nodes for route in first] == [route.nodes for route in second]


def test_the_pool_and_the_sequential_path_agree() -> None:
    """Parallelism is speedup, not a different search.

    A hub's seed comes from the run seed and its own id rather than from a shared generator, so
    which worker picks up which hub cannot change the answer. This is the test that would fail if
    a ``Generator`` were ever put in the task payload.
    """
    sequential_config = small_config(workers=1)
    instance, matrices, traffic = build(sequential_config)
    inbound = inbound_leg(instance, matrices, traffic, sequential_config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))

    sequential = solve_stage2(instance, matrices, traffic, customer_hubs, sequential_config).routes
    pooled = solve_stage2(
        instance, matrices, traffic, customer_hubs, small_config(workers=2)
    ).routes
    assert [route.nodes for route in sequential] == [route.nodes for route in pooled]


def test_a_plan_is_reassembled_against_its_own_hub() -> None:
    """The pool yields results as hubs finish, so pairing is by hub id rather than by position.

    Asserted through the outcome the guard protects: every delivered customer belongs to the hub
    whose tour carries it. A mispairing survives capacity and completeness checks and shows up
    only here.
    """
    config = small_config(workers=2)
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    offset = len(instance.hubs) + len(instance.sources)
    for route in solve_stage2(instance, matrices, traffic, customer_hubs, config).routes:
        for node in route.interior_nodes:
            assert customer_hubs[int(node) - offset] == route.hub_id


# --------------------------------------------------------------------------------------------
# What a worker is allowed to know
# --------------------------------------------------------------------------------------------


def test_a_worker_payload_carries_no_shared_state() -> None:
    """CLAUDE.md §1.1, checked structurally because it has no behavioural symptom.

    A worker holding an ``Instance``, a ``Config`` or a live ``Generator`` still produces a
    correct-looking run — it just stops producing the same one twice. There is nothing downstream
    that fails, so the rule is enforced on the payload's own type.
    """
    annotations = {str(field.type) for field in dataclasses.fields(Stage2Task)}
    assert "Instance" not in annotations, "a worker must not be handed the whole instance"
    assert "Config" not in annotations, "a worker takes frozen value types, not the run config"
    assert not any("Generator" in annotation for annotation in annotations), (
        "randomness is rebuilt from a seed, never shipped: a shared Generator makes a run "
        "succeed and stop repeating"
    )


def test_the_plan_reports_one_outcome_per_hub() -> None:
    """The entry point needs what each hub actually did, not just the tours it produced."""
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    plan = solve_stage2(instance, matrices, traffic, customer_hubs, config)
    assert len(plan.outcomes) == len(set(customer_hubs.tolist()))
    assert all(outcome.generations_run >= 1 for outcome in plan.outcomes)


def test_the_payload_is_frozen() -> None:
    """Anything shared across hubs must be immutable or passed by value."""
    assert Stage2Task.__dataclass_params__.frozen


def test_every_hub_reports_the_wall_clock_its_search_consumed() -> None:
    """The OR-Tools reference spends this number as its own per-hub budget.

    Asserted as positive-and-finite rather than against a value: the point is that a real figure
    is recorded for every hub, not how fast this machine happens to be.
    """
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))

    plan = solve_stage2(instance, matrices, traffic, customer_hubs, config)

    assert len(plan.runs) == len(plan.outcomes)
    assert all(run.elapsed_s > 0.0 for run in plan.runs)
    assert all(math.isfinite(run.elapsed_s) for run in plan.runs)


def test_the_outcomes_property_stays_aligned_with_the_runs_it_reads() -> None:
    """Callers report on outcomes and only the reference wants timings; the two must not diverge.

    Hub order is the contract — ``solve_stage2`` zips workloads against runs — so an ``outcomes``
    that reordered or dropped anything would pair a plan with the wrong hub's workload.
    """
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))

    plan = solve_stage2(instance, matrices, traffic, customer_hubs, config)

    assert plan.outcomes == tuple(run.outcome for run in plan.runs)
    assert [outcome.hub_id for outcome in plan.outcomes] == sorted(
        outcome.hub_id for outcome in plan.outcomes
    )


def test_two_identical_runs_agree_on_the_search_but_not_on_the_clock() -> None:
    """Why ``elapsed_s`` is on :class:`HubRun` and not on ``HubOutcome``.

    ``tests/test_ga.py`` compares whole outcomes for equality, which is only sound while an
    outcome holds nothing that varies between runs of the same search. This pins both halves: the
    search repeats exactly, and the timing is kept somewhere that equality does not reach.
    """
    config = small_config()
    instance, matrices, traffic = build(config)
    inbound = inbound_leg(instance, matrices, traffic, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))

    first = solve_stage2(instance, matrices, traffic, customer_hubs, config)
    second = solve_stage2(instance, matrices, traffic, customer_hubs, config)

    assert first.outcomes == second.outcomes, "the same seed must search identically"
    assert first.routes == second.routes
