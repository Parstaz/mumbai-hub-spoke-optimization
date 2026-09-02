"""Tests for the shared workload layer.

Two of these are load-bearing beyond their own module. :func:`~src.workload.require_servable` is
the codebase's only statement of the no-split invariant, so its boundary — exactly one vehicle
load is legal, a microgram more is not — is asserted here and nowhere else. And
:func:`~src.workload.nearest_hub` is driven with a deliberately **asymmetric** hand-written matrix,
because the direction it measures in (hub out to stop, the way the vehicle leaves) is a decision
that a symmetric matrix could not distinguish from its opposite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import FleetConfig, GeoConfig, ScheduleConfig
from src.data.instance import Coordinate, Customer, Hub, Instance, Shipment, Source
from src.exceptions import InfeasibleInstanceError
from src.units import DemandArray, DistanceMatrix, NodeId
from src.workload import (
    group_by_hub,
    nearest_hub,
    node_array,
    require_servable,
    stage_demands,
)

CAPACITY_KG = 750.0
SHIPMENT_KG = 37.5


def instance_with_origins(origins: list[int], n_sources: int) -> Instance:
    """One hub, ``n_sources`` sources, one customer per shipment originating at ``origins[i]``.

    ``origins`` is how a test piles several shipments onto one Stage 1 stop, or leaves a source
    with nothing waiting at it.
    """
    geo = GeoConfig(
        n_hubs=1,
        n_sources=n_sources,
        n_customers=len(origins),
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
            for index in range(len(origins))
        ),
        shipments=tuple(
            Shipment(index, source_id=origin, customer_id=index, size_kg=SHIPMENT_KG)
            for index, origin in enumerate(origins)
        ),
    )


def test_source_demand_is_summed_per_source_not_per_node() -> None:
    """Three shipments at source 0 make one 112.5 kg stop, not three 37.5 kg ones."""
    source_demand_kg, _ = stage_demands(instance_with_origins([0, 0, 0], n_sources=2))
    assert source_demand_kg.tolist() == [3 * SHIPMENT_KG, 0.0]


def test_a_source_no_shipment_originates_at_has_zero_demand() -> None:
    """A source with nothing waiting is not work; grouping drops it, so it must read as zero."""
    source_demand_kg, _ = stage_demands(instance_with_origins([1], n_sources=3))
    assert source_demand_kg.tolist() == [0.0, SHIPMENT_KG, 0.0]


def test_every_customer_owes_exactly_one_shipment() -> None:
    """Customer demand comes off the shipments too, and the instance guarantees a 1:1 mapping."""
    _, customer_demand_kg = stage_demands(instance_with_origins([0, 1, 1], n_sources=2))
    assert customer_demand_kg.tolist() == [SHIPMENT_KG] * 3


def test_a_source_holding_exactly_one_vehicle_load_is_servable() -> None:
    """The boundary is inclusive: a full vehicle is a legal vehicle."""
    require_servable(np.array([CAPACITY_KG], dtype=np.float64), CAPACITY_KG, "source")


def test_a_source_holding_one_vehicle_load_plus_a_gram_is_rejected() -> None:
    """One gram over capacity is infeasible, because no solver here splits a stop."""
    with pytest.raises(InfeasibleInstanceError, match="no stop is ever split across vehicles"):
        require_servable(np.array([CAPACITY_KG + 0.001], dtype=np.float64), CAPACITY_KG, "source")


def test_the_rejection_message_names_the_worst_source_and_the_capacity() -> None:
    """A refusal has to say what it refused, or it is indistinguishable from a bug."""
    demand: DemandArray = np.array([CAPACITY_KG, 900.0, 1200.0], dtype=np.float64)
    with pytest.raises(InfeasibleInstanceError, match=r"2 source\(s\).*worst 1200.0 kg.*750.0 kg"):
        require_servable(demand, CAPACITY_KG, "source")


def test_the_rejection_message_names_the_stop_kind_the_caller_gave_it() -> None:
    """Stage 2 checks customers, so a refusal must not say "source" and cite the wrong solver.

    The noun is a required argument precisely so this cannot drift: the message is the only place
    a reader learns what was refused.
    """
    demand: DemandArray = np.array([900.0], dtype=np.float64)
    with pytest.raises(InfeasibleInstanceError, match=r"1 customer\(s\) hold more than one"):
        require_servable(demand, CAPACITY_KG, "customer")


def test_a_float_sum_landing_a_microgram_over_capacity_is_still_servable() -> None:
    """Twenty 37.5 kg parcels is exactly a vehicle; float addition must not make it illegal."""
    accumulated = float(np.array([SHIPMENT_KG] * 20, dtype=np.float64).sum())
    require_servable(np.array([accumulated], dtype=np.float64), CAPACITY_KG, "source")


def test_node_array_indexes_the_flat_node_space_through_the_instance_accessors() -> None:
    """Node ids come from the accessors, so a layout change cannot silently shift a stop."""
    instance = instance_with_origins([0, 1], n_sources=2)
    nodes = node_array(instance.source_node(s.source_id) for s in instance.sources)
    assert nodes.tolist() == [1, 2]
    assert nodes.dtype == np.intp


def test_nearest_hub_measures_the_drive_out_to_the_stop_not_the_return_leg() -> None:
    """Hub 1 is the shorter drive *out*; hub 0 is only closer on the way back.

    The whole point of the chosen direction. A solver that measured stop-to-hub, or averaged the
    two, would pick hub 0 here.
    """
    distance_m: DistanceMatrix = np.zeros((3, 3), dtype=np.float64)
    distance_m[0, 2], distance_m[2, 0] = 100.0, 1.0
    distance_m[1, 2], distance_m[2, 1] = 50.0, 500.0
    assignment = nearest_hub(
        node_array([NodeId(0), NodeId(1)]), node_array([NodeId(2)]), distance_m
    )
    assert assignment.tolist() == [1]


def test_nearest_hub_breaks_a_tie_towards_the_lower_hub_id() -> None:
    """Ties must resolve deterministically, or two runs on one instance disagree."""
    distance_m = np.full((3, 3), 42.0, dtype=np.float64)
    assignment = nearest_hub(
        node_array([NodeId(0), NodeId(1)]), node_array([NodeId(2)]), distance_m
    )
    assert assignment.tolist() == [0]


def test_group_by_hub_drops_stops_with_nothing_waiting_at_them() -> None:
    """A zero-demand stop would cost distance and a fixed vehicle charge to move no mass."""
    workloads = group_by_hub(
        hub_nodes=node_array([NodeId(0)]),
        hub_of_stop=np.array([0, 0, 0], dtype=np.intp),
        stop_nodes=node_array([NodeId(1), NodeId(2), NodeId(3)]),
        demand_kg=np.array([SHIPMENT_KG, 0.0, SHIPMENT_KG], dtype=np.float64),
    )
    assert len(workloads) == 1
    assert workloads[0].nodes.tolist() == [1, 3]
    assert workloads[0].demand_kg.tolist() == [SHIPMENT_KG, SHIPMENT_KG]


def test_group_by_hub_omits_a_hub_with_no_work_rather_than_yielding_an_empty_tour() -> None:
    """An empty workload would build a hub-to-hub route, which is not a tour."""
    workloads = group_by_hub(
        hub_nodes=node_array([NodeId(0), NodeId(1)]),
        hub_of_stop=np.array([1], dtype=np.intp),
        stop_nodes=node_array([NodeId(2)]),
        demand_kg=np.array([SHIPMENT_KG], dtype=np.float64),
    )
    assert [workload.hub_id for workload in workloads] == [1]
    assert workloads[0].hub_node == NodeId(1)


def test_group_by_hub_keeps_demand_aligned_with_the_stops_it_kept() -> None:
    """The two arrays are read positionally by every tour builder, so a shift is a wrong load."""
    workloads = group_by_hub(
        hub_nodes=node_array([NodeId(0), NodeId(1)]),
        hub_of_stop=np.array([1, 0, 1], dtype=np.intp),
        stop_nodes=node_array([NodeId(2), NodeId(3), NodeId(4)]),
        demand_kg=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )
    by_hub = {workload.hub_id: workload for workload in workloads}
    assert by_hub[0].nodes.tolist() == [3]
    assert by_hub[0].demand_kg.tolist() == [20.0]
    assert by_hub[1].nodes.tolist() == [2, 4]
    assert by_hub[1].demand_kg.tolist() == [10.0, 30.0]
