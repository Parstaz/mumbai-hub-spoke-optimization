"""Tests for the shared route builder.

Expected arrivals here are worked out by hand against a distance matrix chosen so that one metre
is one tenth of a second: a 1000 m leg is 100 s of free-flow driving. That is what lets a test
assert the timeline is *right* rather than assert it equals whatever
:func:`~src.costs.traffic.route_timeline` returned, which would be a tautology.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.exceptions import InfeasibleSolutionError
from src.tour import RoutingContext, build_route
from src.units import DistanceMatrix, NodeId, Seconds
from src.workload import HubWorkload

EIGHT_AM_S = Seconds(8 * 3600.0)
SERVICE_S = Seconds(300.0)
METRES_PER_SECOND = 10.0

FREE_FLOW = TrafficModel(hourly_multipliers=(1.0,) * 24)
"""A flat schedule, so a test about route assembly is not also a test about traffic."""

MORNING_PEAK = TrafficModel(hourly_multipliers=(1.6,) * 24)
"""The 08–11 band held constant, for the one case that is about the multiplier."""


def context(distance_m: DistanceMatrix, traffic: TrafficModel = FREE_FLOW) -> RoutingContext:
    """Bundle a hand-written matrix into a routing context dispatching at 08:00."""
    return RoutingContext(
        matrices=CostMatrices(distance_m=distance_m, duration_s=distance_m / METRES_PER_SECOND),
        traffic=traffic,
        start_time_s=EIGHT_AM_S,
        service_time_s=SERVICE_S,
        capacity_kg=750.0,
    )


def line_matrix() -> DistanceMatrix:
    """Nodes 0..3 on a line 1000 m apart, so leg lengths are readable off the node ids."""
    positions = np.arange(4, dtype=np.float64) * 1000.0
    matrix: DistanceMatrix = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
    return matrix


def workload(nodes: list[int], demand_kg: list[float]) -> HubWorkload:
    """A hub at node 0 with the given stops and masses."""
    return HubWorkload(
        hub_id=0,
        hub_node=NodeId(0),
        nodes=np.array(nodes, dtype=np.intp),
        demand_kg=np.array(demand_kg, dtype=np.float64),
    )


def test_the_tour_opens_and_closes_at_its_hub() -> None:
    """A vehicle returns to the hub it left; that closure is what makes it a tour."""
    route = build_route(workload([1, 2], [10.0, 20.0]), (0, 1), context(line_matrix()))
    assert route.nodes == (NodeId(0), NodeId(1), NodeId(2), NodeId(0))


def test_distance_is_the_sum_of_the_legs_actually_driven() -> None:
    """0 -> 1 -> 2 -> 0 on the line is 1000 + 1000 + 2000 metres."""
    route = build_route(workload([1, 2], [10.0, 20.0]), (0, 1), context(line_matrix()))
    assert route.distance_m == pytest.approx(4000.0)


def test_visit_order_follows_positions_rather_than_the_workload_order() -> None:
    """The builder is handed an ordering; reversing it must produce the reversed tour.

    0 -> 2 -> 1 -> 0 is 2000 + 1000 + 1000, the same total on a symmetric line, so the assertion
    is on the node sequence — a builder that sorted its input would pass a distance check here.
    """
    route = build_route(workload([1, 2], [10.0, 20.0]), (1, 0), context(line_matrix()))
    assert route.nodes == (NodeId(0), NodeId(2), NodeId(1), NodeId(0))


def test_load_sums_only_the_stops_this_vehicle_serves() -> None:
    """A hub's other stops belong to other vehicles and must not be charged to this one."""
    route = build_route(workload([1, 2, 3], [10.0, 20.0, 40.0]), (0, 2), context(line_matrix()))
    assert route.load_kg == pytest.approx(50.0)


def test_arrivals_charge_service_time_on_leaving_a_stop_but_not_the_hub() -> None:
    """08:00 depart, 100 s to node 1, 300 s dwell, 100 s to node 2, 300 s dwell, 200 s home."""
    route = build_route(workload([1, 2], [10.0, 20.0]), (0, 1), context(line_matrix()))
    assert route.arrival_s == (
        Seconds(28800.0),
        Seconds(28900.0),
        Seconds(29300.0),
        Seconds(29800.0),
    )


def test_duration_is_read_off_the_timeline_including_service() -> None:
    """29800 - 28800 = 1000 s: 400 s driving plus 600 s of dwell at two stops."""
    route = build_route(workload([1, 2], [10.0, 20.0]), (0, 1), context(line_matrix()))
    assert route.duration_s == pytest.approx(1000.0)


def test_traffic_stretches_the_driving_but_not_the_dwell() -> None:
    """At 1.6x, the 400 s of driving becomes 640 s; the 600 s of service is unaffected."""
    route = build_route(
        workload([1, 2], [10.0, 20.0]), (0, 1), context(line_matrix(), MORNING_PEAK)
    )
    assert route.duration_s == pytest.approx(640.0 + 600.0)


def test_a_single_stop_tour_is_legal() -> None:
    """Hub, one stop, hub is the shortest thing a vehicle can be sent out to do."""
    route = build_route(workload([1], [10.0]), (0,), context(line_matrix()))
    assert route.n_stops == 1
    assert route.distance_m == pytest.approx(2000.0)


def test_an_empty_ordering_is_rejected_rather_than_returned_as_a_hub_to_hub_route() -> None:
    """Route validation is the guard, so callers must drop empty vehicles before building."""
    with pytest.raises(InfeasibleSolutionError, match="hub -> at least one stop -> hub"):
        build_route(workload([1], [10.0]), (), context(line_matrix()))
