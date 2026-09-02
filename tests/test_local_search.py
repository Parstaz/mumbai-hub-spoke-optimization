"""Tests for the memetic local search.

The load-bearing test here is :func:`test_refining_then_resplitting_never_costs_more`. The module's
whole claim is that the Lamarckian write-back is safe — that handing the GA a concatenated,
locally-improved order can only help — and that claim rests on an argument about arc weights
depending solely on the contiguous run they cover. An argument in a docstring is a hypothesis; this
is where it is checked.

Geometry is planar and Euclidean so that a crossing is a crossing and a test about 2-opt is not
also a test about the road network. Traffic is flat except where a test is about windows.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import (
    CostConfig,
    FleetConfig,
    GAConfig,
    GeoConfig,
    ScheduleConfig,
)
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
from src.stage2.local_search import refine
from src.stage2.pricing import TourPricer, hub_pricing, ordered_tour
from src.stage2.split import Permutation, SplitContext, split
from src.tour import RoutingContext
from src.units import DistanceMatrix, NodeId, Seconds
from src.workload import HubWorkload

CAPACITY_KG = 750.0

STOP_KG = 250.0
"""Three stops to a vehicle, so a refined plan spans several routes rather than one."""

PARCEL_KG = 37.5
METRES_PER_SECOND = 10.0
EIGHT_AM_S = Seconds(8 * 3600.0)
SERVICE_S = Seconds(300.0)
MAX_PASSES = GAConfig().local_search_max_passes

FLAT = TrafficModel(hourly_multipliers=(1.0,) * 24)
"""A flat schedule, so a test about reordering is not also a test about traffic."""


def euclidean_matrix(points_m: tuple[tuple[float, float], ...]) -> DistanceMatrix:
    """Straight-line distances between planar points, so a crossing costs what it looks like."""
    coords = np.asarray(points_m, dtype=np.float64)
    deltas = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matrix: DistanceMatrix = np.sqrt((deltas**2).sum(axis=-1))
    return matrix


def instance_with_customers(n_customers: int, windows: tuple[TimeWindow | None, ...]) -> Instance:
    """One hub, one unused source, and ``n_customers`` customers carrying the given windows."""
    geo = GeoConfig(
        n_hubs=1,
        n_sources=1,
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
        sources=(Source(0, Coordinate(19.01, 72.91)),),
        customers=tuple(
            Customer(index, Coordinate(19.02 + 0.001 * index, 72.92), windows[index])
            for index in range(n_customers)
        ),
        shipments=tuple(
            Shipment(index, source_id=0, customer_id=index, size_kg=PARCEL_KG)
            for index in range(n_customers)
        ),
    )


def make_context(
    points_m: tuple[tuple[float, float], ...],
    demand_kg: list[float],
    windows: tuple[TimeWindow | None, ...] | None = None,
    cost_config: CostConfig | None = None,
) -> SplitContext:
    """A split context over planar geometry: hub, then an unused source, then the customers."""
    n_stops = len(demand_kg)
    instance = instance_with_customers(n_stops, windows or (None,) * n_stops)
    matrix = euclidean_matrix(points_m)
    workload = HubWorkload(
        hub_id=0,
        hub_node=NodeId(0),
        nodes=np.array([instance.customer_node(index) for index in range(n_stops)], dtype=np.intp),
        demand_kg=np.array(demand_kg, dtype=np.float64),
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
        cost_config=cost_config or CostConfig(),
    )


def line_points(n_stops: int, spacing_m: float = 1_000.0) -> tuple[tuple[float, float], ...]:
    """Hub and source at the origin, then ``n_stops`` customers evenly spaced along the x axis."""
    return ((0.0, 0.0), (0.0, 0.0), *((spacing_m * (index + 1), 0.0) for index in range(n_stops)))


def tour_weight(tour: Permutation, context: SplitContext) -> float:
    """Price one complete route, through the same public pricer the local search uses."""
    return float(
        TourPricer(
            tour=ordered_tour(tour, context.workload, context.pricing, context.routing),
            routing=context.routing,
            cost_config=context.cost_config,
        ).whole_weight()
    )


# --------------------------------------------------------------------------------------------
# The guarantee the write-back rests on
# --------------------------------------------------------------------------------------------


@st.composite
def refinement_case(draw: st.DrawFn) -> tuple[Permutation, SplitContext]:
    """A random planar hub, a random visit order over it, and mixed windows."""
    n_stops = draw(st.integers(min_value=1, max_value=9))
    coords = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=-8_000.0, max_value=8_000.0),
                st.floats(min_value=-8_000.0, max_value=8_000.0),
            ),
            min_size=n_stops,
            max_size=n_stops,
        )
    )
    windows = tuple(
        draw(
            st.one_of(
                st.none(),
                st.builds(
                    lambda opens, span: TimeWindow(Seconds(opens), Seconds(opens + span)),
                    st.floats(min_value=8 * 3600.0, max_value=15 * 3600.0),
                    st.floats(min_value=1800.0, max_value=3 * 3600.0),
                ),
            )
        )
        for _ in range(n_stops)
    )
    context = make_context(((0.0, 0.0), (0.0, 0.0), *coords), [STOP_KG] * n_stops, windows)
    return tuple(draw(st.permutations(range(n_stops)))), context


@settings(max_examples=120, deadline=None)
@given(case=refinement_case())
def test_refining_then_resplitting_never_costs_more(case: tuple[Permutation, SplitContext]) -> None:
    """The monotonicity the Lamarckian write-back depends on.

    Each improved route is exactly the weight of its own arc in the concatenated order's DAG,
    because an arc weight depends only on the contiguous run it covers — the fleet departs
    together. Their sum is therefore one path through that DAG, and ``split`` returns the cheapest.
    So re-splitting a refined chromosome can match the original but never lose to it.

    If this ever fails, the write-back has stopped preserving the routes as a contiguous partition
    and the GA is being handed fitnesses its own plans cannot reproduce.
    """
    permutation, context = case
    before = split(permutation, context)
    after = split(refine(before.tours, context, MAX_PASSES), context)
    assert after.search_objective_inr <= before.search_objective_inr


@settings(max_examples=120, deadline=None)
@given(case=refinement_case())
def test_refinement_returns_a_permutation_of_the_same_stops(
    case: tuple[Permutation, SplitContext],
) -> None:
    """A refined chromosome still delivers to everyone, or its cost is meaningless."""
    permutation, context = case
    refined = refine(split(permutation, context).tours, context, MAX_PASSES)
    assert sorted(refined) == sorted(permutation)


@settings(max_examples=60, deadline=None)
@given(case=refinement_case())
def test_refinement_concatenates_the_routes_it_improved(
    case: tuple[Permutation, SplitContext],
) -> None:
    """The improved routes must survive as contiguous runs, which is what makes the bound hold.

    A refinement that reordered stops *across* two routes would still return a permutation, and
    would still usually be cheaper — but the sum of its routes would no longer be a path through
    the concatenated order's DAG, and the guarantee above would become a coincidence.
    """
    permutation, context = case
    tours = split(permutation, context).tours
    refined = refine(tours, context, MAX_PASSES)
    offset = 0
    for tour in tours:
        assert sorted(refined[offset : offset + len(tour)]) == sorted(tour)
        offset += len(tour)
    assert offset == len(refined)


# --------------------------------------------------------------------------------------------
# The moves actually improve something
# --------------------------------------------------------------------------------------------


def test_two_opt_undoes_a_crossing() -> None:
    """Four stops on a line visited 1-3-2-4 should come back costing what 1-2-3-4 costs.

    The canonical 2-opt fixture. Asserted on the resulting cost rather than on the exact tuple: on
    a symmetric matrix the sorted order and its reverse are the same tour, and pinning one of them
    would be asserting on the search's traversal order rather than on its outcome.
    """
    context = make_context(line_points(4), [PARCEL_KG] * 4)
    refined = refine(split((0, 2, 1, 3), context).tours, context, MAX_PASSES)
    assert split(refined, context).search_objective_inr == pytest.approx(
        float(split((0, 1, 2, 3), context).search_objective_inr)
    )


def test_refinement_leaves_an_already_optimal_route_alone() -> None:
    """Nothing to gain, so nothing changes — and the pass budget is not spent chasing ties."""
    context = make_context(line_points(4), [PARCEL_KG] * 4)
    assert refine(split((0, 1, 2, 3), context).tours, context, MAX_PASSES) == (0, 1, 2, 3)


def test_refinement_improves_a_route_the_split_cannot_fix() -> None:
    """Split chooses cuts, never order — so a badly ordered single tour is local search's job.

    With every stop fitting one vehicle there is only one partition, and the split objective is
    whatever the order happens to cost. Any improvement here is entirely the local search's.
    """
    context = make_context(line_points(5), [PARCEL_KG] * 5)
    before = split((4, 0, 3, 1, 2), context)
    after = split(refine(before.tours, context, MAX_PASSES), context)
    assert len(before.routes) == 1, "the fixture is meant to admit exactly one partition"
    assert after.search_objective_inr < before.search_objective_inr


# --------------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_stops", [1, 2])
def test_a_route_too_short_to_reorder_is_returned_unchanged(n_stops: int) -> None:
    """One stop has one ordering; two have one non-trivial move, left to the GA's operators."""
    context = make_context(line_points(n_stops), [PARCEL_KG] * n_stops)
    tour = tuple(range(n_stops))
    assert refine((tour,), context, MAX_PASSES) == tour


def test_refinement_of_no_tours_is_an_empty_chromosome() -> None:
    """A hub with nothing to do refines to nothing, rather than raising."""
    context = make_context(line_points(1), [PARCEL_KG])
    assert refine((), context, MAX_PASSES) == ()


def test_a_single_pass_still_improves_and_terminates() -> None:
    """The pass cap is a budget, not a correctness condition: one pass must be legal and finite."""
    context = make_context(line_points(5), [PARCEL_KG] * 5)
    before = split((4, 0, 3, 1, 2), context)
    after = split(refine(before.tours, context, 1), context)
    assert after.search_objective_inr <= before.search_objective_inr


def test_more_passes_never_return_a_worse_tour() -> None:
    """Budget only ever helps — asserted per tour, which is the level the guarantee holds at.

    It does **not** extend to the post-split objective. A better-ordered route concatenates into a
    different chromosome, and splitting that chromosome is a fresh optimisation whose answer is not
    comparable with the other's; the module's bound relates a re-split to the routes it was built
    from and to nothing else. Comparing two refinements through ``split`` would be asserting a
    property the algorithm does not have.
    """
    context = make_context(line_points(6), [PARCEL_KG] * 6)
    (tour,) = split((5, 1, 4, 0, 3, 2), context).tours
    cheap = refine((tour,), context, 1)
    thorough = refine((tour,), context, MAX_PASSES)
    assert tour_weight(thorough, context) <= tour_weight(cheap, context)
