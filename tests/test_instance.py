"""Tests for the instance model: node addressing, validation, and JSON round-tripping.

The flat node space is shared by every matrix in the codebase, so its boundaries — last source
versus first customer — are tested explicitly. An off-by-one there would mis-price every route
in the results table while raising nothing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import FleetConfig, ScheduleConfig
from src.data.instance import (
    FORMAT_VERSION,
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.exceptions import InstanceError
from src.units import NodeId, Seconds
from tests.conftest import TINY_GEO, build_instance


def test_node_layout_is_hubs_then_sources_then_customers(tiny_instance: Instance) -> None:
    """Node ids are assigned block-wise, which is what lets a matrix be indexed directly."""
    assert tiny_instance.hub_node(0) == 0
    assert tiny_instance.source_node(0) == 1
    assert tiny_instance.customer_node(0) == 2
    assert tiny_instance.customer_node(1) == 3
    assert tiny_instance.n_nodes == 4


@pytest.mark.parametrize(
    ("node", "is_source", "is_customer"),
    [
        (0, False, False),  # the hub
        (1, True, False),  # the only source — last node before the customer block
        (2, False, True),  # first customer
        (3, False, True),  # last customer
    ],
)
def test_node_kind_boundaries(
    tiny_instance: Instance, node: int, is_source: bool, is_customer: bool
) -> None:
    """The source/customer boundary is exactly where the block layout says it is."""
    assert tiny_instance.is_source_node(NodeId(node)) is is_source
    assert tiny_instance.is_customer_node(NodeId(node)) is is_customer


def test_customer_at_round_trips_every_customer(small_instance: Instance) -> None:
    """Resolving a customer node returns the customer whose id produced it."""
    for customer in small_instance.customers:
        node = small_instance.customer_node(customer.customer_id)
        assert small_instance.customer_at(node) is customer


def test_customer_at_rejects_a_non_customer_node(tiny_instance: Instance) -> None:
    """Asking for the window of a hub is a bug, and fails loudly rather than returning None."""
    with pytest.raises(InstanceError):
        tiny_instance.customer_at(tiny_instance.hub_node(0))


@pytest.mark.parametrize("accessor", ["hub_node", "source_node", "customer_node"])
def test_node_accessors_reject_unknown_ids(tiny_instance: Instance, accessor: str) -> None:
    """Out-of-range ids raise instead of producing a plausible-looking node id."""
    with pytest.raises(InstanceError):
        getattr(tiny_instance, accessor)(99)


def test_coordinates_array_matches_node_order(tiny_instance: Instance) -> None:
    """The coordinate array is in flat node order, so row i is node i."""
    coords = tiny_instance.coordinates()
    assert coords.shape == (4, 2)
    assert coords[0].tolist() == [19.00, 72.90]
    assert coords[1].tolist() == [19.05, 72.95]
    assert coords[3].tolist() == [19.15, 73.05]


def test_coordinate_digest_is_stable_and_geometry_sensitive() -> None:
    """The cache key changes when the geometry changes, and only then."""
    first = build_instance()
    second = build_instance()
    assert first.coordinate_digest() == second.coordinate_digest()

    moved = dataclasses.replace(
        first,
        hubs=(Hub(0, Coordinate(19.01, 72.90)),),
    )
    assert moved.coordinate_digest() != first.coordinate_digest()


def test_json_round_trip_is_exact(small_instance: Instance) -> None:
    """An instance reconstructed from JSON is equal to the original, configs included."""
    restored = Instance.from_json(small_instance.to_json())
    assert restored == small_instance
    assert restored.coordinate_digest() == small_instance.coordinate_digest()


def test_json_round_trip_preserves_windows_and_all_day_customers() -> None:
    """``None`` survives the round trip as all-day rather than collapsing to a window."""
    original = build_instance((TimeWindow(Seconds(28800.0), Seconds(32400.0)), None))
    restored = Instance.from_json(original.to_json())
    assert restored.customers[0].window == original.customers[0].window
    assert restored.customers[1].window is None


def test_write_and_read_json(tmp_path: Path, small_instance: Instance) -> None:
    """Writing creates missing parent directories and reading returns the same instance."""
    destination = tmp_path / "nested" / "instance.json"
    small_instance.write_json(destination)
    assert Instance.read_json(destination) == small_instance


def test_from_json_rejects_an_unknown_format_version(small_instance: Instance) -> None:
    """A future or missing format version fails loudly instead of being parsed optimistically."""
    payload = small_instance.to_json().replace(
        f'"format_version": {FORMAT_VERSION}', '"format_version": 99'
    )
    with pytest.raises(InstanceError):
        Instance.from_json(payload)


def test_instance_rejects_count_disagreeing_with_config() -> None:
    """Counts must match the GeoConfig that generated them, or provenance is a lie."""
    with pytest.raises(InstanceError):
        Instance(
            seed=0,
            geo=TINY_GEO,
            fleet=FleetConfig(),
            schedule=ScheduleConfig(),
            hubs=(Hub(0, Coordinate(19.0, 72.9)), Hub(1, Coordinate(19.1, 72.95))),
            sources=(Source(0, Coordinate(19.05, 72.95)),),
            customers=(
                Customer(0, Coordinate(19.10, 73.00), None),
                Customer(1, Coordinate(19.15, 73.05), None),
            ),
            shipments=(Shipment(0, 0, 0, 37.5), Shipment(1, 0, 1, 37.5)),
        )


def test_instance_rejects_non_contiguous_ids() -> None:
    """Ids double as array indices, so a gap would silently shift every lookup."""
    base = build_instance()
    with pytest.raises(InstanceError):
        dataclasses.replace(
            base,
            customers=(base.customers[0], Customer(5, Coordinate(19.15, 73.05), None)),
        )


def test_instance_rejects_coordinates_outside_the_bounding_box() -> None:
    """A point outside the box means the generator is broken; it is not quietly clamped here."""
    base = build_instance()
    with pytest.raises(InstanceError):
        dataclasses.replace(base, hubs=(Hub(0, Coordinate(28.61, 77.21)),))


def test_instance_rejects_shipment_with_unknown_endpoints() -> None:
    """A shipment must resolve to a real source and a real customer."""
    base = build_instance()
    with pytest.raises(InstanceError):
        dataclasses.replace(base, shipments=(Shipment(0, source_id=7, customer_id=0, size_kg=1.0),))
    with pytest.raises(InstanceError):
        dataclasses.replace(base, shipments=(Shipment(0, source_id=0, customer_id=7, size_kg=1.0),))


def test_instance_accepts_a_shipment_exactly_at_vehicle_capacity() -> None:
    """The capacity boundary is inclusive: a full-vehicle shipment is legal."""
    base = build_instance()
    instance = dataclasses.replace(
        base, shipments=(Shipment(0, source_id=0, customer_id=0, size_kg=750.0),)
    )
    assert instance.shipments[0].size_kg == 750.0


def test_instance_rejects_a_shipment_over_vehicle_capacity() -> None:
    """One unit over capacity is rejected: no vehicle could carry it."""
    base = build_instance()
    with pytest.raises(InstanceError):
        dataclasses.replace(
            base, shipments=(Shipment(0, source_id=0, customer_id=0, size_kg=750.001),)
        )


def test_instance_rejects_an_empty_shipment_list() -> None:
    """An instance with nothing to move is not a problem instance."""
    with pytest.raises(InstanceError):
        dataclasses.replace(build_instance(), shipments=())


def test_n_deliveries_equals_customer_count(small_instance: Instance) -> None:
    """The KPI denominator is the customer count, since each customer receives one shipment."""
    assert small_instance.n_deliveries == len(small_instance.customers)


@pytest.mark.parametrize(
    ("start_s", "end_s"),
    [(32400.0, 28800.0), (28800.0, 28800.0), (-3600.0, 28800.0)],
)
def test_time_window_rejects_impossible_bounds(start_s: float, end_s: float) -> None:
    """Inverted, empty and pre-midnight windows are all rejected at construction."""
    with pytest.raises(InstanceError):
        TimeWindow(Seconds(start_s), Seconds(end_s))


@pytest.mark.parametrize(
    ("arrival_s", "expected_lateness_s"),
    [
        (28_800.0, 0.0),  # before the window opens: waiting, not lateness
        (30_000.0, 0.0),  # inside the window
        (32_400.0, 0.0),  # exactly at the close: on time
        (32_401.0, 1.0),  # one second late
        (36_000.0, 3600.0),  # an hour late
    ],
)
def test_time_window_lateness(arrival_s: float, expected_lateness_s: float) -> None:
    """Lateness is measured from the close of the window, and early arrival is never lateness."""
    window = TimeWindow(start_s=Seconds(29_700.0), end_s=Seconds(32_400.0))
    assert window.lateness_s(Seconds(arrival_s)) == expected_lateness_s
