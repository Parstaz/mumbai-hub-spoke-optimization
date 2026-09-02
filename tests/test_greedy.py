"""Tests for the greedy nearest-neighbour benchmark.

Most cases here drive the solver with a **hand-written distance matrix** rather than a generated
one. The baseline's every decision is an ``argmin`` over road distance, so writing the matrix down
is how an expected hub assignment or an expected visit order becomes a fact about the algorithm
instead of a snapshot of whatever it produced last time.

Two properties get the most attention, because they are the two ways this benchmark could quietly
stop being fair: that a customer is served from the hub its *shipment* reached — not its own
nearest hub, which would be a free optimization the pipeline is supposed to earn — and that
nothing here scores anything, so every rupee in the suite comes from ``evaluate_solution``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from src.baseline.greedy import solve_baseline
from src.config import (
    SECONDS_PER_HOUR,
    CostConfig,
    FleetConfig,
    GeoConfig,
    ScheduleConfig,
    TrafficConfig,
)
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel, route_timeline
from src.data.instance import (
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.exceptions import InfeasibleInstanceError
from src.scoring import evaluate_solution
from src.solution import Route, Solution
from src.units import DistanceMatrix, NodeId, Seconds
from tests.conftest import SMALL_GEO, build_instance

COSTS = CostConfig()
SHIPMENT_KG = 37.5

FREE_FLOW = TrafficModel(hourly_multipliers=(1.0,) * 24)
"""A flat schedule, so a test that is about routing is not also about traffic."""

TWO_HUB_GEO = GeoConfig(
    n_hubs=2,
    n_sources=2,
    n_customers=2,
    n_density_clusters=1,
    hub_candidate_pool=10,
)
"""Node layout: hubs 0–1 are nodes 0–1, sources 0–1 are nodes 2–3, customers 0–1 are nodes 4–5."""


def traffic_model() -> TrafficModel:
    """The project's real band schedule, for the cases that are about traffic."""
    return TrafficModel.from_config(TrafficConfig())


def uniform_matrices(distance_m: DistanceMatrix, speed_mps: float = 10.0) -> CostMatrices:
    """Wrap a hand-written distance matrix, deriving duration from a flat speed.

    Free-flow duration proportional to distance keeps the hand-written cases readable: a matrix
    entry is simultaneously the distance the tour builder chooses on and the time the timeline
    accumulates.
    """
    return CostMatrices(distance_m=distance_m, duration_s=distance_m / speed_mps)


def haversine_matrices(instance: Instance) -> CostMatrices:
    """Real-geometry matrices for the instance, with no network access."""
    distance_m, duration_s = HaversineProvider(circuity_factor=1.3, speed_kmph=24.0).matrix(
        instance.coordinates()
    )
    return CostMatrices(distance_m=distance_m, duration_s=duration_s)


def symmetric_matrix(
    n_nodes: int, legs: dict[tuple[int, int], float], far: float
) -> DistanceMatrix:
    """Build a symmetric distance matrix: named legs get their length, everything else ``far``.

    For cases that assert *which* node an ``argmin`` picks. Not for cases that compare one tour's
    length against another's — see :func:`planar_matrix` for why.
    """
    distance = np.full((n_nodes, n_nodes), far, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    for (origin, destination), metres in legs.items():
        distance[origin, destination] = metres
        distance[destination, origin] = metres
    return distance


def planar_matrix(points: Sequence[tuple[float, float]]) -> DistanceMatrix:
    """Euclidean distances in metres between planar points, one per node.

    Used wherever a test claims the greedy tour is longer than some other tour. A Euclidean
    matrix satisfies the triangle inequality by construction, which is what makes that claim a
    fact about nearest-neighbour rather than an artefact of an inconsistent hand-written matrix.
    """
    array = np.asarray(points, dtype=np.float64)
    deltas = array[:, np.newaxis, :] - array[np.newaxis, :, :]
    matrix: DistanceMatrix = np.sqrt((deltas**2).sum(axis=2))
    return matrix


def one_hub_instance(n_customers: int, source_ids: Sequence[int] | None = None) -> Instance:
    """One hub (node 0), then sources, then customers, one shipment per customer.

    ``source_ids`` gives the origin of shipment ``i`` and so controls how mass piles up on a
    Stage 1 stop; it defaults to every shipment originating at a single source.
    """
    origins = list(source_ids) if source_ids is not None else [0] * n_customers
    n_sources = max(origins) + 1
    geo = GeoConfig(
        n_hubs=1,
        n_sources=n_sources,
        n_customers=n_customers,
        n_density_clusters=1,
        hub_candidate_pool=10,
    )
    return Instance(
        seed=0,
        geo=geo,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)),),
        sources=tuple(
            Source(index, Coordinate(19.02 + 0.01 * index, 72.92)) for index in range(n_sources)
        ),
        customers=tuple(
            Customer(index, Coordinate(19.05, 72.95 + 0.001 * index), None)
            for index in range(n_customers)
        ),
        shipments=tuple(
            Shipment(index, source_id=origin, customer_id=index, size_kg=SHIPMENT_KG)
            for index, origin in enumerate(origins)
        ),
    )


def two_hub_instance(shipments: tuple[Shipment, ...]) -> Instance:
    """Two hubs, two sources, two customers, with the shipment wiring under test control."""
    return Instance(
        seed=0,
        geo=TWO_HUB_GEO,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)), Hub(1, Coordinate(19.20, 73.00))),
        sources=(Source(0, Coordinate(19.02, 72.92)), Source(1, Coordinate(19.18, 72.98))),
        customers=(
            Customer(0, Coordinate(19.04, 72.94), None),
            Customer(1, Coordinate(19.16, 72.96), None),
        ),
        shipments=shipments,
    )


def one_to_one_shipments() -> tuple[Shipment, ...]:
    """Source 0 supplies customer 0, source 1 supplies customer 1."""
    return (
        Shipment(0, source_id=0, customer_id=0, size_kg=SHIPMENT_KG),
        Shipment(1, source_id=1, customer_id=1, size_kg=SHIPMENT_KG),
    )


def stops_of(routes: tuple[Route, ...]) -> list[tuple[int, ...]]:
    """The interior stops of each route, as plain ints, for readable assertions."""
    return [tuple(int(node) for node in route.interior_nodes) for route in routes]


def test_baseline_solution_is_scored_by_the_shared_path(tiny_instance: Instance) -> None:
    """The whole contract: the plan is a Solution that ``evaluate_solution`` accepts and prices."""
    solution = solve_baseline(tiny_instance, haversine_matrices(tiny_instance), traffic_model())

    metrics = evaluate_solution(solution, tiny_instance, COSTS)

    assert metrics.total_cost_inr > 0.0
    assert metrics.cost_per_drop_inr == pytest.approx(
        metrics.total_cost_inr / tiny_instance.n_deliveries
    )
    assert metrics.vehicles_used == len(solution.all_routes)


def test_every_source_with_freight_is_collected_and_every_customer_delivered() -> None:
    """Completeness, asserted through the scorer, which raises if a stage leaves work undone."""
    instance = Instance(
        seed=7,
        geo=SMALL_GEO,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)), Hub(1, Coordinate(19.20, 73.00))),
        sources=(
            Source(0, Coordinate(19.02, 72.92)),
            Source(1, Coordinate(19.18, 72.98)),
            Source(2, Coordinate(19.10, 72.95)),
        ),
        customers=tuple(
            Customer(index, Coordinate(19.05 + 0.03 * index, 72.93), None) for index in range(4)
        ),
        # Source 2 supplies nothing, so it must not be visited at all.
        shipments=tuple(
            Shipment(index, source_id=index % 2, customer_id=index, size_kg=SHIPMENT_KG)
            for index in range(4)
        ),
    )

    solution = solve_baseline(instance, haversine_matrices(instance), traffic_model())

    collected = {node for route in solution.stage1_routes for node in route.interior_nodes}
    delivered = {node for route in solution.stage2_routes for node in route.interior_nodes}
    assert collected == {instance.source_node(0), instance.source_node(1)}
    assert delivered == {instance.customer_node(cid) for cid in range(4)}
    evaluate_solution(solution, instance, COSTS)


def test_source_is_assigned_to_its_nearest_hub_by_road_distance() -> None:
    """Assignment follows the matrix, not the coordinates: source 1 is nearer hub 0 by road."""
    instance = two_hub_instance(one_to_one_shipments())
    # Source 1 (node 3) sits beside hub 1 geographically, but the road to hub 0 is shorter.
    distance = symmetric_matrix(
        instance.n_nodes,
        legs={(0, 2): 1_000.0, (1, 2): 9_000.0, (0, 3): 2_000.0, (1, 3): 8_000.0},
        far=50_000.0,
    )

    solution = solve_baseline(instance, uniform_matrices(distance), FREE_FLOW)

    assert [route.hub_id for route in solution.stage1_routes] == [0]
    assert stops_of(solution.stage1_routes) == [(2, 3)]


def test_customer_is_served_from_the_hub_holding_its_shipment_not_its_nearest_hub() -> None:
    """The structural property that makes the two-stage decomposition visible.

    Customer 0 (node 4) is a kilometre from hub 0 and forty from hub 1, but its shipment
    originates at source 1, which consolidates at hub 1. It must be delivered from hub 1: the
    parcel is only ever in one place, and choosing the nearer hub would dispatch a parcel that is
    not there.
    """
    instance = two_hub_instance(
        shipments=(
            Shipment(0, source_id=1, customer_id=0, size_kg=SHIPMENT_KG),
            Shipment(1, source_id=0, customer_id=1, size_kg=SHIPMENT_KG),
        )
    )
    distance = symmetric_matrix(
        instance.n_nodes,
        legs={
            (0, 2): 1_000.0,  # source 0 consolidates at hub 0
            (1, 2): 9_000.0,
            (0, 3): 9_000.0,
            (1, 3): 1_000.0,  # source 1 consolidates at hub 1
            (0, 4): 1_000.0,  # customer 0 is next door to hub 0 ...
            (1, 4): 40_000.0,  # ... and a long way from hub 1, which holds its parcel
            (0, 5): 40_000.0,
            (1, 5): 1_000.0,
        },
        far=50_000.0,
    )

    solution = solve_baseline(instance, uniform_matrices(distance), FREE_FLOW)

    served_from = {
        int(node): route.hub_id for route in solution.stage2_routes for node in route.interior_nodes
    }
    assert served_from == {4: 1, 5: 0}


def test_visit_order_is_nearest_neighbour_and_is_not_the_shortest_tour() -> None:
    """The canonical straw-man failure: greedy takes the near bait and strands the far stop.

    Geometry, in metres: the hub at the origin, A 1000 m due north, B 1050 m due south, C 3000 m
    due east. Nearest neighbour goes north to A, crosses back south to B — 2050 m — and only then
    drives east to C, for 9228 m. Sweeping A, C, B instead costs 8391 m. A local-search pass would
    find that with a single 2-opt move; the benchmark must not, or there is nothing to improve on.
    """
    instance = one_hub_instance(3)
    hub, source, a, b, c = 0, 1, 2, 3, 4
    distance = planar_matrix(
        [
            (0.0, 0.0),  # hub
            (100.0, 100.0),  # source, close to the hub so Stage 1 is trivial
            (0.0, 1_000.0),  # A
            (0.0, -1_050.0),  # B
            (3_000.0, 0.0),  # C
        ]
    )

    solution = solve_baseline(instance, uniform_matrices(distance), FREE_FLOW)

    (delivery,) = solution.stage2_routes
    assert stops_of(solution.stage2_routes) == [(a, b, c)]
    swept_m = sum(distance[pair] for pair in ((hub, a), (a, c), (c, b), (b, hub)))
    assert delivery.distance_m == pytest.approx(9_228.4, abs=0.1)
    assert swept_m == pytest.approx(8_390.7, abs=0.1)
    assert swept_m < delivery.distance_m
    assert stops_of(solution.stage1_routes) == [(source,)]


@pytest.mark.parametrize(
    ("n_customers", "source_ids", "expected_vehicles"),
    [
        (20, None, 1),
        (21, [index % 2 for index in range(21)], 2),
    ],
    ids=["exactly-at-capacity", "one-shipment-over"],
)
def test_capacity_is_hard_at_the_boundary(
    n_customers: int, source_ids: list[int] | None, expected_vehicles: int
) -> None:
    """Exactly a vehicle-load rides on one vehicle; one shipment more forces a second.

    750 kg capacity over 37.5 kg shipments is exactly 20 drops, so this brackets the limit rather
    than probing near it — and the 20-drop case also proves the float tolerance does its job, since
    twenty accumulated 37.5s need not land exactly on 750.
    """
    instance = one_hub_instance(n_customers, source_ids)

    solution = solve_baseline(instance, haversine_matrices(instance), traffic_model())

    assert len(solution.stage2_routes) == expected_vehicles
    assert all(
        route.load_kg <= instance.fleet.vehicle_capacity_kg for route in solution.stage2_routes
    )
    evaluate_solution(solution, instance, COSTS)


def test_a_stage_one_tour_closes_on_mass_not_on_stop_count() -> None:
    """A Stage 1 stop carries everything waiting there, so capacity binds on kilograms.

    Source 0 holds 15 shipments (562.5 kg) and source 1 holds 10 (375 kg). Source 0 is nearer the
    hub, so the first vehicle takes it; source 1's 375 kg will not fit in the 187.5 kg left, and
    goes out on a second vehicle. Two stops, two vehicles — nothing about that is a stop count.
    """
    instance = one_hub_instance(25, source_ids=[0] * 15 + [1] * 10)

    solution = solve_baseline(instance, haversine_matrices(instance), traffic_model())

    assert stops_of(solution.stage1_routes) == [(1,), (2,)]
    assert [route.load_kg for route in solution.stage1_routes] == [562.5, 375.0]


def test_a_source_holding_more_than_a_vehicle_is_rejected() -> None:
    """The failure mode: the baseline never splits a stop, so it must refuse rather than guess."""
    # All 21 shipments — 787.5 kg — wait at the single source, over the 750 kg vehicle.
    instance = one_hub_instance(21)

    with pytest.raises(InfeasibleInstanceError, match="no stop is ever split across vehicles"):
        solve_baseline(instance, haversine_matrices(instance), traffic_model())


def test_time_windows_are_recorded_and_never_avoided() -> None:
    """An impossible window changes nothing about the plan, and shows up only in the metrics.

    The same instance is solved twice, once with a window nobody could meet. The routes must be
    identical — the baseline has no window awareness to exercise — and the only difference must be
    the violation the scorer reports.
    """
    blind = build_instance()
    windowed = build_instance(windows=(TimeWindow(Seconds(0.0), Seconds(60.0)), None))
    matrices = haversine_matrices(blind)

    blind_solution = solve_baseline(blind, matrices, traffic_model())
    windowed_solution = solve_baseline(windowed, matrices, traffic_model())

    assert stops_of(windowed_solution.stage2_routes) == stops_of(blind_solution.stage2_routes)
    assert evaluate_solution(blind_solution, blind, COSTS).tw_violations == 0
    windowed_metrics = evaluate_solution(windowed_solution, windowed, COSTS)
    assert windowed_metrics.tw_violations == 1
    assert windowed_metrics.tw_lateness_hr > 0.0
    assert windowed_metrics.breakdown.tw_penalty_inr > 0.0


def test_arrival_times_come_from_the_shared_cumulative_traffic_model(
    tiny_instance: Instance,
) -> None:
    """Arrivals must match ``route_timeline`` exactly, not a per-route multiplier applied once."""
    matrices = haversine_matrices(tiny_instance)
    traffic = traffic_model()

    solution = solve_baseline(tiny_instance, matrices, traffic)

    for route in solution.all_routes:
        expected = route_timeline(
            route.nodes,
            matrices.duration_s,
            Seconds(tiny_instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
            traffic,
            Seconds(tiny_instance.fleet.service_time_per_stop_s),
        )
        assert route.arrival_s == pytest.approx(expected)
        assert route.duration_s == pytest.approx(expected[-1] - expected[0])


def test_traffic_lengthens_the_day_without_changing_the_plan(tiny_instance: Instance) -> None:
    """Distance is a property of the routing; duration is a property of when it is driven.

    Same stops, same kilometres, more hours — which is what makes ``driver_per_hour`` the channel
    the traffic model reaches the reported cost through.
    """
    matrices = haversine_matrices(tiny_instance)

    free_flow = solve_baseline(tiny_instance, matrices, FREE_FLOW)
    peak = solve_baseline(tiny_instance, matrices, traffic_model())

    assert stops_of(peak.all_routes) == stops_of(free_flow.all_routes)
    assert peak.total_distance_m == pytest.approx(free_flow.total_distance_m)
    assert peak.total_duration_s > free_flow.total_duration_s
    peak_cost = evaluate_solution(peak, tiny_instance, COSTS).total_cost_inr
    assert peak_cost > evaluate_solution(free_flow, tiny_instance, COSTS).total_cost_inr


def test_route_distance_is_the_sum_of_its_legs_in_the_shared_matrix(
    tiny_instance: Instance,
) -> None:
    """No second distance model: every metre reported is a matrix entry the solver indexed."""
    matrices = haversine_matrices(tiny_instance)

    solution = solve_baseline(tiny_instance, matrices, FREE_FLOW)

    for route in solution.all_routes:
        legs = zip(route.nodes, route.nodes[1:], strict=False)
        assert route.distance_m == pytest.approx(
            sum(matrices.distance_m[origin, destination] for origin, destination in legs)
        )


def test_the_baseline_is_deterministic(small_instance: Instance) -> None:
    """No randomness anywhere: two solves of one instance are the same plan, tie-breaks included."""
    matrices = haversine_matrices(small_instance)

    first = solve_baseline(small_instance, matrices, traffic_model())
    second = solve_baseline(small_instance, matrices, traffic_model())

    assert first == second


def test_a_single_stop_hub_still_produces_a_legal_tour() -> None:
    """The smallest possible tour — hub, one stop, hub — is a boundary the Route type enforces."""
    instance = build_instance()

    solution = solve_baseline(instance, haversine_matrices(instance), FREE_FLOW)

    (inbound,) = solution.stage1_routes
    assert inbound.n_stops == 1
    assert inbound.nodes[0] == inbound.nodes[-1] == NodeId(0)
    assert inbound.load_kg == pytest.approx(2 * SHIPMENT_KG)


def test_solution_holds_both_stages_separately(tiny_instance: Instance) -> None:
    """Stage 1 stops are sources and Stage 2 stops are customers — the scorer rejects a mix-up."""
    solution = solve_baseline(tiny_instance, haversine_matrices(tiny_instance), FREE_FLOW)

    assert isinstance(solution, Solution)
    assert all(
        tiny_instance.is_source_node(node)
        for route in solution.stage1_routes
        for node in route.interior_nodes
    )
    assert all(
        tiny_instance.is_customer_node(node)
        for route in solution.stage2_routes
        for node in route.interior_nodes
    )
