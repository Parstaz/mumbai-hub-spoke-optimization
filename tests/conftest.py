"""Shared fixtures.

Instances here are deliberately tiny — one hub, one source, two customers — so that every
expected cost in the suite can be worked out by hand and checked against the cost model rather
than against whatever the code happened to produce.
"""

from __future__ import annotations

import pytest

from src.config import FleetConfig, GeoConfig, ScheduleConfig
from src.data.generate import generate_instance
from src.data.instance import (
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.solution import Route
from src.units import Metres, NodeId, Seconds

TINY_GEO = GeoConfig(
    n_hubs=1,
    n_sources=1,
    n_customers=2,
    n_density_clusters=1,
    hub_candidate_pool=10,
)
"""One hub (node 0), one source (node 1), two customers (nodes 2 and 3)."""

SMALL_GEO = GeoConfig(
    n_hubs=2,
    n_sources=3,
    n_customers=4,
    n_density_clusters=2,
    hub_candidate_pool=200,
)

EIGHT_AM_S = Seconds(8 * 3600.0)
SHIPMENT_KG = 37.5


def build_instance(windows: tuple[TimeWindow | None, ...] = (None, None)) -> Instance:
    """Hand-build the tiny instance, with the customers' time windows under test control."""
    return Instance(
        seed=0,
        geo=TINY_GEO,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)),),
        sources=(Source(0, Coordinate(19.05, 72.95)),),
        customers=(
            Customer(0, Coordinate(19.10, 73.00), windows[0]),
            Customer(1, Coordinate(19.15, 73.05), windows[1]),
        ),
        shipments=(
            Shipment(0, source_id=0, customer_id=0, size_kg=SHIPMENT_KG),
            Shipment(1, source_id=0, customer_id=1, size_kg=SHIPMENT_KG),
        ),
    )


def make_route(
    nodes: tuple[int, ...],
    load_kg: float = 75.0,
    distance_m: float = 10_000.0,
    duration_s: float = 3600.0,
    arrival_s: tuple[float, ...] | None = None,
) -> Route:
    """Build a route, defaulting to arrivals evenly spaced across its duration from 08:00."""
    if arrival_s is None:
        step = duration_s / (len(nodes) - 1)
        arrival_s = tuple(EIGHT_AM_S + step * index for index in range(len(nodes)))
    return Route(
        hub_id=nodes[0],
        nodes=tuple(NodeId(node) for node in nodes),
        load_kg=load_kg,
        distance_m=Metres(distance_m),
        duration_s=Seconds(duration_s),
        arrival_s=tuple(Seconds(value) for value in arrival_s),
    )


@pytest.fixture
def tiny_instance() -> Instance:
    """The hand-built instance with no time windows."""
    return build_instance()


@pytest.fixture
def small_instance() -> Instance:
    """A generated instance small enough to assert over exhaustively."""
    return generate_instance(SMALL_GEO, FleetConfig(), ScheduleConfig(), seed=7)
