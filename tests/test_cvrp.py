"""Tests for the per-hub inbound CVRP.

Every case here sets ``cvrp_solution_limit=1``, which stops OR-Tools at the first-solution
heuristic. That is not a shortcut for speed: guided local search under a wall-clock time limit
returns whatever it had reached when the clock ran out, so a test asserting on a GLS result would
pass or fail according to how busy the machine was. Verified separately that ``solution_limit=1``
gives the same objective at a 50 ms limit as at a 5 s one, and the same as with the metaheuristic
switched off entirely.

The properties asserted are the ones that would let a wrong plan look right: capacity is never
exceeded (it is a hard OR-Tools dimension, never a penalty), every assigned source is collected
from exactly once, the vehicle count is the mass floor rather than merely *some* feasible number,
no empty tour is ever emitted, and the parallel and sequential paths return identical plans —
because if they did not, the reported cost would depend on the machine's core count.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.config import (
    Config,
    FleetConfig,
    GeoConfig,
    ScheduleConfig,
    Stage1Config,
)
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel
from src.data.instance import Coordinate, Customer, Hub, Instance, Shipment, Source
from src.exceptions import InfeasibleInstanceError
from src.scoring import evaluate_solution
from src.solution import Route, Solution
from src.stage1.assignment import capacity_balanced, unconstrained
from src.stage1.cvrp import solve_stage1
from src.units import DistanceMatrix

SHIPMENT_KG = 37.5

DETERMINISTIC = Stage1Config(cvrp_solution_limit=1, workers=1)
"""First-solution only, one process: the only configuration a test may assert a plan against."""


def instance_with(origins: list[int], n_hubs: int = 1, capacity_kg: float = 750.0) -> Instance:
    """Sources laid out east of the hubs, one shipment per customer originating at ``origins[i]``.

    ``origins`` controls how mass piles onto a Stage 1 stop, which is how a test forces a second
    vehicle or pushes a stop past capacity.
    """
    n_sources = max(origins) + 1
    geo = GeoConfig(
        n_hubs=n_hubs,
        n_sources=n_sources,
        n_customers=len(origins),
        n_density_clusters=1,
        hub_candidate_pool=10,
    )
    return Instance(
        seed=0,
        geo=geo,
        fleet=FleetConfig(vehicle_capacity_kg=capacity_kg),
        schedule=ScheduleConfig(),
        hubs=tuple(Hub(i, Coordinate(19.00 + 0.10 * i, 72.80)) for i in range(n_hubs)),
        sources=tuple(
            Source(i, Coordinate(19.00 + 0.01 * i, 72.90 + 0.005 * i)) for i in range(n_sources)
        ),
        customers=tuple(
            Customer(i, Coordinate(19.20, 73.00 + 0.001 * i), None) for i in range(len(origins))
        ),
        shipments=tuple(
            Shipment(i, source_id=origin, customer_id=i, size_kg=SHIPMENT_KG)
            for i, origin in enumerate(origins)
        ),
    )


def matrices_for(instance: Instance) -> CostMatrices:
    """Real-geometry matrices with no network access."""
    distance_m, duration_s = HaversineProvider(circuity_factor=1.3, speed_kmph=24.0).matrix(
        instance.coordinates()
    )
    return CostMatrices(distance_m=distance_m, duration_s=duration_s)


def solve(
    instance: Instance,
    stage1: Stage1Config = DETERMINISTIC,
    strategy: object = unconstrained,
    matrices: CostMatrices | None = None,
) -> tuple[Route, ...]:
    """Run Stage 1 over ``instance`` with a deterministic search by default."""
    config = dataclasses.replace(Config(), stage1=stage1)
    return solve_stage1(
        instance,
        matrices if matrices is not None else matrices_for(instance),
        TrafficModel.from_config(config.traffic),
        strategy,  # type: ignore[arg-type]  # parametrised over the two strategy functions
        config,
    )


def collected_nodes(routes: tuple[Route, ...]) -> list[int]:
    """Every stop visited across a set of tours, in no particular order."""
    return [int(node) for route in routes for node in route.interior_nodes]


def test_one_source_yields_one_tour_through_it() -> None:
    """The smallest possible inbound plan: hub, source, hub."""
    routes = solve(instance_with([0]))
    assert len(routes) == 1
    assert routes[0].nodes == (0, 1, 0)
    assert routes[0].load_kg == pytest.approx(SHIPMENT_KG)


def test_every_source_with_something_waiting_is_collected_exactly_once() -> None:
    """Nodes 1-4 are the four sources; all four must appear, none twice."""
    routes = solve(instance_with([0, 1, 2, 3]))
    assert sorted(collected_nodes(routes)) == [1, 2, 3, 4]


def test_a_source_with_nothing_waiting_is_not_visited() -> None:
    """Source 1 (node 2) originates no shipment, so a tour through it moves zero mass."""
    routes = solve(instance_with([0, 0, 2]))
    assert sorted(collected_nodes(routes)) == [1, 3]


def test_capacity_forces_exactly_the_second_vehicle_and_no_more() -> None:
    """Three 300 kg stops against a 750 kg vehicle: 900 kg has a mass floor of exactly two.

    Two of the stops fit one vehicle (600 kg) and the third cannot, so the right answer is two
    tours. Asserted as equality rather than ``>= 2`` on purpose: a three- or four-vehicle plan is
    also feasible and also respects capacity, so an inequality here would pass for a wasteful
    solver and the fixed vehicle charge in the objective would go unverified.
    """
    instance = instance_with([0] * 8 + [1] * 8 + [2] * 8, capacity_kg=750.0)
    routes = solve(instance)
    assert len(routes) == 2
    assert all(route.load_kg <= 750.0 + 1e-6 for route in routes)
    assert sorted(collected_nodes(routes)) == [1, 2, 3]


def test_a_stop_holding_exactly_one_vehicle_load_is_a_tour_of_its_own() -> None:
    """The boundary: 20 × 37.5 kg is exactly 750 kg, and must fit one vehicle rather than two."""
    instance = instance_with([0] * 20, capacity_kg=750.0)
    routes = solve(instance)
    assert len(routes) == 1
    assert routes[0].load_kg == pytest.approx(750.0)


def test_a_stop_holding_more_than_one_vehicle_load_is_rejected() -> None:
    """One shipment over capacity is infeasible: Stage 1 never splits a source.

    The same rule and the same exception as the greedy baseline, from the same function in
    ``src/workload.py``, so step 8 cannot end up comparing solvers under different constraints.
    """
    instance = instance_with([0] * 21, capacity_kg=750.0)
    with pytest.raises(InfeasibleInstanceError, match="no stop is ever split across vehicles"):
        solve(instance)


def test_no_empty_tour_is_emitted_when_vehicles_go_unused() -> None:
    """The slack factor offers spare vehicles; unused ones must be dropped, not returned.

    One 37.5 kg source has a mass floor of one vehicle, and the 1.15 slack rounds that up to two.
    The second must never appear as a hub-to-hub route with no load.
    """
    routes = solve(instance_with([0]))
    assert all(route.n_stops >= 1 for route in routes)
    assert all(route.load_kg > 0.0 for route in routes)


def test_every_tour_leaves_from_and_returns_to_its_own_hub() -> None:
    """With four hubs, a tour's node sequence must close at the hub it claims."""
    routes = solve(instance_with([0, 1, 2, 3, 4, 5], n_hubs=4))
    for route in routes:
        assert route.nodes[0] == route.nodes[-1] == route.hub_id


def test_the_plan_scores_through_the_shared_scoring_path() -> None:
    """Stage 1 routes must be acceptable to evaluate_solution, not merely well-formed.

    Paired with a hand-built Stage 2 covering every customer, so the scorer's completeness and
    capacity checks both run over the CVRP's output.
    """
    instance = instance_with([0, 1])
    stage1 = solve(instance)
    stage2 = Route(
        hub_id=0,
        nodes=(0, 3, 4, 0),
        load_kg=2 * SHIPMENT_KG,
        distance_m=10_000.0,
        duration_s=3_600.0,
        arrival_s=(28_800.0, 30_000.0, 31_200.0, 32_400.0),
    )
    metrics = evaluate_solution(Solution(stage1, (stage2,)), instance, Config().cost)
    assert metrics.total_cost_inr > 0.0
    assert metrics.vehicles_used == len(stage1) + 1


def test_the_parallel_and_sequential_paths_return_identical_plans() -> None:
    """A pool must be pure speedup. If it were not, cost would depend on the core count.

    Four hubs give the pool four independent tasks to distribute; ``pool.map`` preserves input
    order, so the tours must come back in the same sequence as well as with the same contents.
    """
    instance = instance_with([0, 1, 2, 3, 4, 5, 6, 7], n_hubs=4)
    sequential = solve(instance, DETERMINISTIC)
    parallel = solve(instance, dataclasses.replace(DETERMINISTIC, workers=2))
    assert [route.nodes for route in parallel] == [route.nodes for route in sequential]
    assert [route.distance_m for route in parallel] == [route.distance_m for route in sequential]


def test_the_solve_is_reproducible_under_the_deterministic_search() -> None:
    """Two runs on one instance must produce byte-identical plans."""
    instance = instance_with([0, 1, 2, 3, 4])
    assert [r.nodes for r in solve(instance)] == [r.nodes for r in solve(instance)]


def test_the_balanced_strategy_moves_load_off_an_oversubscribed_hub() -> None:
    """Every source sits beside hub 0; a cap of 3 must spread six of them over two hubs.

    Asserted on the *hub* distribution rather than on cost, because whether balancing pays is an
    empirical question the CLI answers, not something a unit test should assume.
    """
    matrix: DistanceMatrix = np.full((8, 8), 50_000.0, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for stop in range(2, 8):
        matrix[0, stop] = matrix[stop, 0] = 1_000.0
        matrix[1, stop] = matrix[stop, 1] = 40_000.0
    instance = instance_with([0, 1, 2, 3, 4, 5], n_hubs=2)
    matrices = CostMatrices(distance_m=matrix, duration_s=matrix / 10.0)

    nearest = solve(instance, DETERMINISTIC, unconstrained, matrices)
    balanced = solve(
        instance,
        dataclasses.replace(DETERMINISTIC, hub_balance_slack=1.0),
        capacity_balanced,
        matrices,
    )
    assert {route.hub_id for route in nearest} == {0}
    assert {route.hub_id for route in balanced} == {0, 1}
    assert sorted(collected_nodes(balanced)) == [2, 3, 4, 5, 6, 7]


def test_arrival_times_come_from_the_cumulative_traffic_model() -> None:
    """OR-Tools sees a static arc cost, but the reported tour must carry the real timeline.

    A tour dispatched into the 08-11 band at 1.6x must take longer than free-flow would predict.
    If this ever equalled the free-flow duration, the static proxy would have leaked into the
    reported figures.
    """
    instance = instance_with([0, 1, 2])
    routes = solve(instance)
    matrices = matrices_for(instance)
    for route in routes:
        path = np.array(route.nodes, dtype=np.intp)
        free_flow_s = float(matrices.duration_s[path[:-1], path[1:]].sum())
        service_s = route.n_stops * instance.fleet.service_time_per_stop_s
        assert route.duration_s > free_flow_s + service_s
