"""The equivalence test :mod:`src.stage2.pricing` exists on sufferance of.

The arc pricer is an optimisation of the shared path, not an alternative to it. That claim is only
worth anything if it is checked, so this suite pins every arc weight to
:func:`~src.tour.build_route` followed by :func:`~src.scoring.route_cost` under ``==`` rather than
a tolerance. A tolerance would let the fast path become a second cost model by degrees, which is
exactly what CLAUDE.md §1.1's single scoring path forbids.

Matrices here are **asymmetric** and drawn pseudo-randomly. A symmetric matrix — the line geometry
``test_split.py`` uses, quite reasonably, for questions about partitioning — cannot tell a leg out
of the hub from a leg back to it, so it would pass a pricer that had the two confused.

Traffic is the configured band schedule rather than a flat one, because a flat schedule cannot
tell a cumulative traffic model from a per-leg one and the whole point of pricing along a prefix is
that the clock keeps running.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import (
    CAPACITY_TOLERANCE_KG,
    SECONDS_PER_HOUR,
    CostConfig,
    FleetConfig,
    GeoConfig,
    ScheduleConfig,
    TrafficConfig,
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
from src.exceptions import InfeasibleSolutionError
from src.scoring import route_cost, route_window_outcome
from src.stage2.pricing import TourPricer, hub_pricing, ordered_tour
from src.stage2.split import Permutation
from src.tour import RoutingContext, build_route
from src.units import DemandArray, NodeId, Rupees, Seconds
from src.workload import HubWorkload

PARCEL_KG = 37.5
CAPACITY_KG = 750.0
SERVICE_S = Seconds(300.0)
MAX_PROPERTY_STOPS = 7

BANDED = TrafficModel.from_config(TrafficConfig())
"""The configured bands. A leg crossing 11:00 blends 1.6 and 1.2, which is the interesting case."""

RATE_SETS: tuple[CostConfig, ...] = (
    CostConfig(),
    CostConfig(
        variable_per_km=1.0, driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=0.0
    ),
    CostConfig(
        variable_per_km=0.0, driver_per_hour=1.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=0.0
    ),
    CostConfig(
        variable_per_km=0.0, driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=1.0
    ),
    CostConfig(tw_penalty_per_hour=CostConfig().tw_penalty_per_hour * 8.0),
)
"""Default rates, then each component alone, then the adaptive penalty's shape.

Isolating a component is how this suite reaches distance, duration and lateness separately without
the pricer having to expose them: priced at ₹1/km and nothing else, an arc weight *is* its distance
in kilometres, so a leg taken from the wrong matrix entry cannot hide inside a total.
"""


def asymmetric_matrices(n_nodes: int, seed: int) -> CostMatrices:
    """Random distances with an independent random duration, so the two cannot be inferred.

    Asymmetric on purpose, and with a zero diagonal so a degenerate hub-to-hub leg costs nothing.
    """
    rng = np.random.default_rng(seed)
    distance_m = rng.uniform(200.0, 25_000.0, size=(n_nodes, n_nodes))
    duration_s = distance_m / rng.uniform(4.0, 12.0, size=(n_nodes, n_nodes))
    np.fill_diagonal(distance_m, 0.0)
    np.fill_diagonal(duration_s, 0.0)
    return CostMatrices(distance_m=distance_m, duration_s=duration_s)


def instance_with_windows(windows: tuple[TimeWindow | None, ...]) -> Instance:
    """One hub, one unused source, and one customer per entry of ``windows``."""
    n_customers = len(windows)
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


def make_workload(instance: Instance, demand_kg: DemandArray) -> HubWorkload:
    """The hub's stops, as positions into the instance's customer block."""
    return HubWorkload(
        hub_id=0,
        hub_node=NodeId(0),
        nodes=np.array(
            [instance.customer_node(index) for index in range(len(demand_kg))], dtype=np.intp
        ),
        demand_kg=demand_kg,
    )


def make_routing(matrices: CostMatrices, dispatch_hour: float) -> RoutingContext:
    """Routing at the configured traffic bands, dispatching at ``dispatch_hour``."""
    return RoutingContext(
        matrices=matrices,
        traffic=BANDED,
        start_time_s=Seconds(dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=SERVICE_S,
        capacity_kg=CAPACITY_KG,
    )


def prefix_load_kg(permutation: Permutation, demand_kg: DemandArray) -> DemandArray:
    """Cumulative load along the permutation, as :func:`~src.stage2.split.split` computes it."""
    ordered = demand_kg[np.asarray(permutation, dtype=np.intp)]
    prefix: DemandArray = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ordered)))
    return prefix


def priced_arcs(
    permutation: Permutation,
    workload: HubWorkload,
    instance: Instance,
    routing: RoutingContext,
    cost_config: CostConfig,
) -> dict[tuple[int, int], Rupees]:
    """Every arc the pricer yields for ``permutation``, keyed by ``(start, end)``."""
    pricer = TourPricer(
        tour=ordered_tour(permutation, workload, hub_pricing(workload, instance), routing),
        routing=routing,
        cost_config=cost_config,
    )
    prefix_kg = prefix_load_kg(permutation, workload.demand_kg)
    room_kg = routing.capacity_kg + CAPACITY_TOLERANCE_KG
    return {
        (start, end): weight
        for start in range(len(permutation))
        for end, weight in pricer.weights_from(start, prefix_kg, room_kg)
    }


def oracle_weight(
    stops: Permutation,
    workload: HubWorkload,
    instance: Instance,
    routing: RoutingContext,
    cost_config: CostConfig,
) -> Rupees:
    """What the arc serving ``stops`` costs down the shared path, tour built and all."""
    route = build_route(workload, stops, routing)
    return route_cost(route, cost_config, route_window_outcome(route, instance)).total_inr


# --------------------------------------------------------------------------------------------
# The equivalence property
# --------------------------------------------------------------------------------------------


@st.composite
def pricing_case(draw: st.DrawFn) -> tuple[Permutation, HubWorkload, Instance, RoutingContext]:
    """A random hub: asymmetric matrices, mixed windows, and a visit order over its stops."""
    n_stops = draw(st.integers(min_value=1, max_value=MAX_PROPERTY_STOPS))
    seed = draw(st.integers(min_value=0, max_value=2**16))
    dispatch_hour = draw(st.sampled_from((6.0, 8.0, 10.75, 16.9, 20.5)))
    windows = tuple(
        draw(
            st.one_of(
                st.none(),
                st.builds(
                    lambda start_s, length_s: TimeWindow(
                        Seconds(start_s), Seconds(start_s + length_s)
                    ),
                    st.floats(min_value=8 * 3600.0, max_value=18 * 3600.0),
                    st.floats(min_value=1800.0, max_value=4 * 3600.0),
                ),
            )
        )
        for _ in range(n_stops)
    )
    permutation = tuple(draw(st.permutations(range(n_stops))))
    instance = instance_with_windows(windows)
    workload = make_workload(instance, np.full(n_stops, PARCEL_KG, dtype=np.float64))
    routing = make_routing(asymmetric_matrices(instance.n_nodes, seed), dispatch_hour)
    return permutation, workload, instance, routing


@settings(max_examples=150, deadline=None)
@given(case=pricing_case(), cost_config=st.sampled_from(RATE_SETS))
def test_every_arc_weight_equals_the_shared_path(
    case: tuple[Permutation, HubWorkload, Instance, RoutingContext], cost_config: CostConfig
) -> None:
    """The price of admission: identical rupees, arc for arc, with no tolerance allowed.

    Priced under each rate set in turn, so distance, duration and lateness are each reachable on
    their own — a component that is right only because another compensates fails here.
    """
    permutation, workload, instance, routing = case
    arcs = priced_arcs(permutation, workload, instance, routing, cost_config)
    assert arcs, "a servable hub must have at least the singleton arcs"
    for (start, end), weight_inr in arcs.items():
        expected = oracle_weight(permutation[start:end], workload, instance, routing, cost_config)
        assert weight_inr == expected, (
            f"arc {start}->{end} priced {weight_inr}, expected {expected}"
        )


@settings(max_examples=50, deadline=None)
@given(case=pricing_case())
def test_the_arc_set_is_every_run_one_vehicle_can_carry(
    case: tuple[Permutation, HubWorkload, Instance, RoutingContext],
) -> None:
    """Capacity is structural: exactly the feasible runs exist, and no others.

    Demands are uniform parcels here, so the feasible run length is a count and the expected set
    can be written down without repeating the pricer's own summation.
    """
    permutation, workload, instance, routing = case
    per_vehicle = int(CAPACITY_KG // PARCEL_KG)
    expected = {
        (start, end)
        for start in range(len(permutation))
        for end in range(start + 1, min(start + per_vehicle, len(permutation)) + 1)
    }
    assert set(priced_arcs(permutation, workload, instance, routing, CostConfig())) == expected


# --------------------------------------------------------------------------------------------
# Pricing one whole tour, which is what the local search evaluates a move with
# --------------------------------------------------------------------------------------------


def whole_tour_weight(
    permutation: Permutation,
    workload: HubWorkload,
    instance: Instance,
    routing: RoutingContext,
    cost_config: CostConfig,
) -> Rupees:
    """Price the whole permutation as one vehicle's work."""
    return TourPricer(
        tour=ordered_tour(permutation, workload, hub_pricing(workload, instance), routing),
        routing=routing,
        cost_config=cost_config,
    ).whole_weight()


@settings(max_examples=100, deadline=None)
@given(case=pricing_case(), cost_config=st.sampled_from(RATE_SETS))
def test_whole_weight_equals_the_shared_path(
    case: tuple[Permutation, HubWorkload, Instance, RoutingContext], cost_config: CostConfig
) -> None:
    """Closing once must give exactly what closing at every stop and taking the last one gives.

    The local search prices hundreds of candidate reorderings per route, so it closes once rather
    than building a family of prefixes to read the end of. That shortcut is only safe while the two
    agree to the last bit, which is what this pins.
    """
    permutation, workload, instance, routing = case
    expected = oracle_weight(permutation, workload, instance, routing, cost_config)
    assert whole_tour_weight(permutation, workload, instance, routing, cost_config) == expected


def test_whole_weight_agrees_with_the_arc_that_covers_every_stop() -> None:
    """The whole tour is also an arc of the DAG, when one vehicle can carry the lot."""
    workload, instance, routing = single_stop_case((None, None, None), np.array([PARCEL_KG] * 3))
    permutation = (2, 0, 1)
    arcs = priced_arcs(permutation, workload, instance, routing, CostConfig())
    assert whole_tour_weight(permutation, workload, instance, routing, CostConfig()) == arcs[(0, 3)]


def test_whole_weight_of_a_single_stop_tour() -> None:
    """Hub, one stop, hub — the smallest thing the local search can be handed."""
    workload, instance, routing = single_stop_case((None,), np.array([PARCEL_KG]))
    expected = oracle_weight((0,), workload, instance, routing, CostConfig())
    assert whole_tour_weight((0,), workload, instance, routing, CostConfig()) == expected


def test_whole_weight_refuses_a_tour_with_no_stops() -> None:
    """A deployed vehicle carries something. ``Route`` refuses the same thing, loudly."""
    workload, instance, routing = single_stop_case((None,), np.array([PARCEL_KG]))
    with pytest.raises(InfeasibleSolutionError, match="not a tour"):
        whole_tour_weight((), workload, instance, routing, CostConfig())


# --------------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------------


def single_stop_case(
    windows: tuple[TimeWindow | None, ...],
    demand_kg: DemandArray,
    dispatch_hour: float = 8.0,
    seed: int = 3,
) -> tuple[HubWorkload, Instance, RoutingContext]:
    """A hub over ``windows``' customers, ready to price."""
    instance = instance_with_windows(windows)
    workload = make_workload(instance, demand_kg)
    return (
        workload,
        instance,
        make_routing(asymmetric_matrices(instance.n_nodes, seed), dispatch_hour),
    )


def test_a_single_stop_arc_matches_the_shared_path() -> None:
    """The smallest tour there is: hub, one stop, hub."""
    workload, instance, routing = single_stop_case((None,), np.array([PARCEL_KG]))
    arcs = priced_arcs((0,), workload, instance, routing, CostConfig())
    assert arcs == {(0, 1): oracle_weight((0,), workload, instance, routing, CostConfig())}


def test_an_empty_permutation_yields_no_arcs() -> None:
    """A hub with nothing to do has no tours, and asking for arcs is not an error."""
    workload, instance, routing = single_stop_case((None,), np.array([PARCEL_KG]))
    assert priced_arcs((), workload, instance, routing, CostConfig()) == {}


def test_a_load_exactly_at_capacity_is_priced_and_one_unit_over_is_not() -> None:
    """The capacity boundary is inclusive, and the arc past it is never created."""
    demand = np.array([CAPACITY_KG / 2.0, CAPACITY_KG / 2.0, PARCEL_KG])
    workload, instance, routing = single_stop_case((None, None, None), demand)
    arcs = priced_arcs((0, 1, 2), workload, instance, routing, CostConfig())
    assert (0, 2) in arcs, "two half-loads are exactly one vehicle"
    assert (0, 3) not in arcs, "a third stop puts the vehicle over"


def test_a_customer_with_no_window_is_never_late() -> None:
    """``None`` means available all day, exactly as the shared path reads it."""
    late_rates = CostConfig(
        variable_per_km=0.0, driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=1.0
    )
    workload, instance, routing = single_stop_case((None, None), np.array([PARCEL_KG] * 2))
    arcs = priced_arcs((0, 1), workload, instance, routing, late_rates)
    assert set(arcs.values()) == {Rupees(0.0)}


def test_lateness_accumulates_over_the_stops_the_arc_covers() -> None:
    """An arc's lateness is its own stops' and no others'.

    The window here closes before dispatch, so every stop it covers is late and the penalty has
    to grow with the arc rather than being charged once.
    """
    closed = TimeWindow(Seconds(6 * 3600.0), Seconds(7 * 3600.0))
    late_rates = CostConfig(
        variable_per_km=0.0, driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=1.0
    )
    workload, instance, routing = single_stop_case((closed, closed), np.array([PARCEL_KG] * 2))
    arcs = priced_arcs((0, 1), workload, instance, routing, late_rates)
    assert arcs[(0, 2)] > arcs[(0, 1)]
    for (start, end), weight in arcs.items():
        assert weight == oracle_weight((0, 1)[start:end], workload, instance, routing, late_rates)


@pytest.mark.parametrize("dispatch_hour", (7.9, 10.9, 16.9, 20.9, 23.5))
def test_a_leg_crossing_a_band_boundary_matches_the_shared_path(dispatch_hour: float) -> None:
    """Dispatch just before each band edge, so the first leg blends two multipliers.

    23:30 additionally pushes the tour past midnight, where the hour index wraps.
    """
    windows = (None, None, None)
    workload, instance, routing = single_stop_case(
        windows, np.array([PARCEL_KG] * 3), dispatch_hour=dispatch_hour
    )
    for (start, end), weight in priced_arcs(
        (0, 1, 2), workload, instance, routing, CostConfig()
    ).items():
        assert weight == oracle_weight(
            (0, 1, 2)[start:end], workload, instance, routing, CostConfig()
        )


def test_the_adaptive_penalty_is_a_rate_substitution() -> None:
    """The GA prices arcs at a scaled rate; the pricer must not know that has happened."""
    closes = TimeWindow(Seconds(8 * 3600.0), Seconds(9 * 3600.0))
    workload, instance, routing = single_stop_case((closes, None), np.array([PARCEL_KG] * 2))
    scaled = CostConfig(tw_penalty_per_hour=CostConfig().tw_penalty_per_hour * 16.0)
    for (start, end), weight in priced_arcs((0, 1), workload, instance, routing, scaled).items():
        assert weight == oracle_weight((0, 1)[start:end], workload, instance, routing, scaled)


# --------------------------------------------------------------------------------------------
# Failure modes and window resolution
# --------------------------------------------------------------------------------------------


def test_hub_pricing_gives_no_window_to_a_stop_that_is_not_a_customer() -> None:
    """Sources and hub visits have no window, which is how the shared path treats them.

    Stage 2 never prices a source. Agreeing with :func:`~src.scoring.route_window_outcome` anyway
    is cheaper than an equivalence argument that has to establish the case is unreachable.
    """
    instance = instance_with_windows((None,))
    workload = HubWorkload(
        hub_id=0,
        hub_node=NodeId(0),
        nodes=np.array([instance.source_node(0)], dtype=np.intp),
        demand_kg=np.array([PARCEL_KG]),
    )
    assert hub_pricing(workload, instance).windows == (None,)


def test_hub_pricing_keeps_the_hub_s_own_stop_order() -> None:
    """The window array is indexed by workload position, not by customer id."""
    early = TimeWindow(Seconds(8 * 3600.0), Seconds(9 * 3600.0))
    late = TimeWindow(Seconds(15 * 3600.0), Seconds(17 * 3600.0))
    instance = instance_with_windows((early, None, late))
    workload = make_workload(instance, np.array([PARCEL_KG] * 3))
    assert hub_pricing(workload, instance).windows == (early, None, late)


def test_ordered_tour_reorders_windows_with_the_permutation() -> None:
    """A permutation reorders the windows too, or lateness would be charged to the wrong stop."""
    early = TimeWindow(Seconds(8 * 3600.0), Seconds(9 * 3600.0))
    late = TimeWindow(Seconds(15 * 3600.0), Seconds(17 * 3600.0))
    instance = instance_with_windows((early, None, late))
    workload = make_workload(instance, np.array([PARCEL_KG] * 3))
    routing = make_routing(asymmetric_matrices(instance.n_nodes, 11), 8.0)
    tour = ordered_tour((2, 0, 1), workload, hub_pricing(workload, instance), routing)
    assert tour.windows == (late, early, None)
