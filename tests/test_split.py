"""Tests for the split procedure — the correctness linchpin of Stage 2.

A wrong split fails *silently*. It returns tours that close at their hub, carry a legal load and
cost more than they had to, and nothing raises. So this suite does not merely exercise
:func:`~src.stage2.split.split`; it proves the partition it returns is the cheapest one available,
against an independent oracle.

The oracle is :func:`partition_cost_inr`, which enumerates every capacity-feasible partition and
prices it through the **public** scoring fold rather than through anything in ``split.py``. If
``split`` and the enumeration ever disagree, one of them is wrong and the suite says which
direction.

Geometry is a line matrix with one metre worth a tenth of a second, reusing ``test_tour.py``'s
convention, and traffic is flat except where a test is about time windows — so a failure here is
never really a failure in the traffic model.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import (
    CAPACITY_TOLERANCE_KG,
    CostConfig,
    FleetConfig,
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
from src.exceptions import InfeasibleInstanceError, InfeasibleSolutionError
from src.scoring import stage_cost
from src.stage2.pricing import hub_pricing
from src.stage2.split import Permutation, SplitContext, split
from src.tour import RoutingContext, build_route
from src.units import DistanceMatrix, NodeId, Seconds
from src.workload import HubWorkload

CAPACITY_KG = 750.0
STOP_KG = 250.0
"""Three stops to a vehicle, exactly — so a capacity cut lands on a stop boundary."""

PARCEL_KG = 37.5
METRES_PER_SECOND = 10.0
EIGHT_AM_S = Seconds(8 * 3600.0)
SERVICE_S = Seconds(300.0)

FLAT = TrafficModel(hourly_multipliers=(1.0,) * 24)
"""A flat schedule, so a test about partitioning is not also a test about traffic."""

CLUSTER_M = (0.0, 0.0, 1_000.0, 2_000.0, 20_000.0, 21_000.0)
"""Hub and source at the origin, ``A1``/``A2`` beside them, ``B1``/``B2`` twenty kilometres out.

The fixture the greedy-filling regression turns on: two near stops, two far ones, and a vehicle
that holds exactly three.
"""

CLUSTER_MAX_RUN = 3
PROPERTY_STOPS = 5


def line_matrix(positions_m: Sequence[float]) -> DistanceMatrix:
    """Distances between nodes laid out on a line, so a leg length is readable off a position."""
    positions = np.asarray(positions_m, dtype=np.float64)
    matrix: DistanceMatrix = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
    return matrix


def instance_with_customers(n_customers: int, windows: tuple[TimeWindow | None, ...]) -> Instance:
    """One hub, one unused source, and ``n_customers`` customers with the given windows.

    The instance is here for the delivery windows and the node layout only. The mass under test
    lives in the :class:`~src.workload.HubWorkload`, which is what ``split`` actually reads — that
    separation is how a test can pose a stop heavier than any shipment an ``Instance`` would
    accept.
    """
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
    positions_m: Sequence[float],
    demand_kg: Sequence[float],
    windows: tuple[TimeWindow | None, ...] | None = None,
    cost_config: CostConfig | None = None,
) -> SplitContext:
    """Assemble a split context over a hand-written line matrix.

    ``positions_m`` holds one position per node of the flat space — hub, source, then the
    customers — so it is two entries longer than ``demand_kg``.
    """
    n_stops = len(demand_kg)
    instance = instance_with_customers(n_stops, windows or (None,) * n_stops)
    matrix = line_matrix(positions_m)
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


def scoring_instance(context: SplitContext) -> Instance:
    """Rebuild the instance a context was assembled over, for the oracle to price through.

    :class:`~src.stage2.split.SplitContext` stopped carrying an ``Instance`` when the arc pricer
    landed — it holds the hub's windows instead, so a per-hub GA worker can be handed one without
    the whole instance. The oracle still prices through :func:`~src.scoring.stage_cost`, which
    needs an instance, and these fixtures are a pure function of their stop count and windows.
    """
    return instance_with_customers(len(context.workload.nodes), context.pricing.windows)


def cluster_context(
    windows: tuple[TimeWindow | None, ...] | None = None,
    cost_config: CostConfig | None = None,
) -> SplitContext:
    """The two-cluster fixture: four stops of 250 kg against a 750 kg vehicle."""
    return make_context(CLUSTER_M, [STOP_KG] * 4, windows, cost_config)


def cut_sequences(n_stops: int, max_run: int) -> Iterator[tuple[int, ...]]:
    """Every partition of ``0..n_stops`` into contiguous runs of at most ``max_run`` stops.

    Yielded as cut points, so ``(0, 1, 4)`` means one vehicle takes stop 0 and another takes stops
    1 through 3. This is the enumeration ``split`` has to beat or match.
    """

    def walk(cuts: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if cuts[-1] == n_stops:
            yield cuts
            return
        for nxt in range(cuts[-1] + 1, min(cuts[-1] + max_run, n_stops) + 1):
            yield from walk((*cuts, nxt))

    yield from walk((0,))


def partition_cost_inr(
    cuts: tuple[int, ...], permutation: Permutation, context: SplitContext
) -> float:
    """Price one partition through the public scoring fold.

    Deliberately routed through :func:`~src.scoring.stage_cost` rather than through ``split``'s own
    arc pricer: an oracle that shared the code under test could only ever confirm it is
    self-consistent.
    """
    routes = tuple(
        build_route(context.workload, permutation[i:j], context.routing)
        for i, j in itertools.pairwise(cuts)
    )
    return float(
        stage_cost(routes, scoring_instance(context), context.cost_config).breakdown.total_inr
    )


def stop_nodes(routes: tuple[tuple[NodeId, ...], ...]) -> tuple[NodeId, ...]:
    """Concatenate the tours' interior nodes, in plan order."""
    return tuple(node for nodes in routes for node in nodes)


def visited(context: SplitContext, permutation: Permutation) -> tuple[NodeId, ...]:
    """The node ids the permutation asks for, in its own order."""
    return tuple(NodeId(int(context.workload.nodes[position])) for position in permutation)


# --------------------------------------------------------------------------------------------
# Optimality
# --------------------------------------------------------------------------------------------


def test_a_hand_computed_partition_is_found() -> None:
    """One near stop and two far ones: the vehicle driving out should serve the whole far pair.

    Three 375 kg stops against a 750 kg vehicle, so a tour holds at most two and there are exactly
    two two-vehicle partitions. Pairing the near stop with a far one (60 km + 62 km = 122 km) is
    far worse than sending it out alone (2 km + 62 km = 64 km).
    """
    context = make_context((0.0, 0.0, 1_000.0, 30_000.0, 31_000.0), [375.0] * 3)
    plan = split((0, 1, 2), context)
    assert tuple(route.nodes for route in plan.routes) == ((0, 2, 0), (0, 3, 4, 0))


def test_the_optimum_matches_a_brute_force_enumeration() -> None:
    """Against every feasible partition enumerated independently, split must find the cheapest."""
    context = cluster_context()
    permutation = (0, 1, 2, 3)
    costs = {
        cuts: partition_cost_inr(cuts, permutation, context)
        for cuts in cut_sequences(len(permutation), CLUSTER_MAX_RUN)
    }
    cheapest = min(costs, key=lambda cuts: costs[cuts])

    plan = split(permutation, context)
    assert plan.search_objective_inr == pytest.approx(costs[cheapest])
    assert tuple(route.n_stops for route in plan.routes) == tuple(
        j - i for i, j in itertools.pairwise(cheapest)
    )


def test_greedy_left_to_right_filling_is_not_optimal() -> None:
    """The canonical implementation error, and the reason this module gets a regression test.

    Filling each vehicle to the brim before opening the next takes ``(A1, A2, B1)`` then ``(B2)``
    — 40 km plus 42 km — because ``B1`` still fits. The optimum sends ``A1`` out alone and lets one
    vehicle sweep ``(A2, B1, B2)``: 2 km plus 42 km. Both deploy two vehicles, so the fixed charge
    is identical and every rupee of the gap is distance and driver time.

    An implementation that fills left to right passes every other test in this file and fails this
    one.
    """
    context = cluster_context()
    permutation = (0, 1, 2, 3)
    greedy_inr = partition_cost_inr((0, 3, 4), permutation, context)

    plan = split(permutation, context)
    assert tuple(route.nodes for route in plan.routes) == ((0, 2, 0), (0, 3, 4, 5, 0))
    assert plan.search_objective_inr < greedy_inr
    assert len(plan.routes) == 2, "the win must be a better partition, not a saved vehicle"


def test_two_splits_of_one_permutation_are_the_same_plan() -> None:
    """Not merely the same price: multi-seed comparability needs the identical plan back."""
    context = cluster_context()
    first, second = split((2, 0, 3, 1), context), split((2, 0, 3, 1), context)
    assert tuple(route.nodes for route in first.routes) == tuple(
        route.nodes for route in second.routes
    )
    assert first.search_objective_inr == second.search_objective_inr


# --------------------------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------------------------


def test_a_load_exactly_at_capacity_is_one_route() -> None:
    """Twenty 37.5 kg parcels is exactly one vehicle, and exactly full is legal.

    The float sum of twenty parcels can land a fraction of a microgram over 750.0, which is what
    ``CAPACITY_TOLERANCE_KG`` absorbs. Without it the arc spanning all twenty would not exist and
    this instance would buy a second vehicle for nothing.
    """
    positions = (0.0, 0.0, *(100.0 * index for index in range(1, 21)))
    plan = split(tuple(range(20)), make_context(positions, [PARCEL_KG] * 20))
    assert len(plan.routes) == 1
    assert plan.routes[0].n_stops == 20


def test_one_gram_over_capacity_forces_a_second_vehicle() -> None:
    """A gram past capacity makes the arc absent, not expensive — the constraint is structural."""
    positions = (0.0, 0.0, *(100.0 * index for index in range(1, 21)))
    demand = [PARCEL_KG] * 20
    demand[0] += 0.001
    plan = split(tuple(range(20)), make_context(positions, demand))
    assert len(plan.routes) == 2
    assert all(route.load_kg <= CAPACITY_KG + CAPACITY_TOLERANCE_KG for route in plan.routes)


def test_a_single_stop_permutation_is_one_route() -> None:
    """Hub, one stop, hub is the shortest plan a hub with work in it can have."""
    plan = split((0,), make_context((0.0, 0.0, 5_000.0), [STOP_KG]))
    assert tuple(route.nodes for route in plan.routes) == ((0, 2, 0),)


def test_an_empty_permutation_is_a_plan_with_no_routes() -> None:
    """A hub with nothing to deliver deploys nothing and costs nothing.

    ``group_by_hub`` drops empty hubs, so the pipeline never asks this; the DAG still has to answer
    it, because a node-zero-to-node-zero shortest path is the degenerate case of the recurrence.
    """
    context = replace(
        cluster_context(),
        workload=HubWorkload(
            hub_id=0,
            hub_node=NodeId(0),
            nodes=np.array([], dtype=np.intp),
            demand_kg=np.array([], dtype=np.float64),
        ),
    )
    plan = split((), context)
    assert plan.routes == ()
    assert plan.search_objective_inr == 0.0


def test_the_vehicle_count_never_falls_below_the_mass_floor() -> None:
    """1000 kg cannot leave in one 750 kg vehicle however the partition is drawn."""
    context = cluster_context()
    floor = math.ceil(float(context.workload.demand_kg.sum()) / CAPACITY_KG)
    assert len(split((0, 1, 2, 3), context).routes) >= floor


# --------------------------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------------------------


def test_a_stop_heavier_than_a_vehicle_is_infeasible() -> None:
    """The no-split rule, refused in Stage 2's own vocabulary.

    Posed by inflating the workload's demand directly, because ``Instance`` already refuses a
    shipment this heavy. The guard is still the right one and still load-bearing: the DAG reaches
    its terminal node only along single-stop arcs, and this is the stop whose arc would be
    missing.
    """
    context = cluster_context()
    context = replace(
        context,
        workload=replace(
            context.workload,
            demand_kg=np.array([STOP_KG, 900.0, STOP_KG, STOP_KG], dtype=np.float64),
        ),
    )
    with pytest.raises(InfeasibleInstanceError, match=r"1 customer\(s\) hold more than one"):
        split((0, 1, 2, 3), context)


def test_a_permutation_that_omits_a_stop_is_rejected() -> None:
    """A partial chromosome would deliver less and so price lower — it must not score at all."""
    with pytest.raises(InfeasibleSolutionError, match="exactly once"):
        split((0, 1, 2), cluster_context())


def test_a_permutation_that_repeats_a_stop_is_rejected() -> None:
    """A duplicated stop is a broken crossover, caught at the operator rather than at scoring."""
    with pytest.raises(InfeasibleSolutionError, match="exactly once"):
        split((0, 1, 2, 2), cluster_context())


def test_a_permutation_position_outside_the_workload_is_rejected() -> None:
    """An index past the hub's stop list would address another hub's customer."""
    with pytest.raises(InfeasibleSolutionError, match="exactly once"):
        split((0, 1, 2, 9), cluster_context())


# --------------------------------------------------------------------------------------------
# Time windows are soft
# --------------------------------------------------------------------------------------------


def test_a_missed_time_window_is_priced_but_not_forbidden() -> None:
    """No partition can reach the far cluster by 08:10, and a full plan still comes back.

    This is what "soft" buys: the alternative — treating a window as a hard arc filter — would
    make this hub unservable and return nothing at all.
    """
    unreachable = TimeWindow(start_s=EIGHT_AM_S, end_s=Seconds(EIGHT_AM_S + 600.0))
    permutation = (0, 1, 2, 3)
    late = split(permutation, cluster_context((None, None, unreachable, unreachable)))
    punctual = split(permutation, cluster_context())

    assert stop_nodes(tuple(route.interior_nodes for route in late.routes)) == visited(
        cluster_context(), permutation
    )
    assert late.search_objective_inr > punctual.search_objective_inr


def test_a_tight_time_window_moves_the_partition() -> None:
    """The window term is in the arc weight, so it changes which partition wins.

    Left alone, the optimum is ``(A1) (A2, B1, B2)``. ``B1``'s window closes at 08:33:20, which is
    exactly when a vehicle carrying only the far pair arrives and 300 s before one that served
    ``A2`` first does. Under a raised penalty rate that 300 s of lateness outweighs the ~₹23 the
    cheaper partition saves, and the plan becomes ``(A1, A2) (B1, B2)``.
    """
    closes = TimeWindow(start_s=EIGHT_AM_S, end_s=Seconds(EIGHT_AM_S + 2_000.0))
    urgent = CostConfig(tw_penalty_per_hour=2_500.0)
    plan = split((0, 1, 2, 3), cluster_context((None, None, closes, None), urgent))
    assert tuple(route.nodes for route in plan.routes) == ((0, 2, 3, 0), (0, 4, 5, 0))


# --------------------------------------------------------------------------------------------
# The search objective is not the reported cost
# --------------------------------------------------------------------------------------------


def test_the_search_objective_agrees_with_the_scoring_fold_at_the_configured_rate() -> None:
    """Priced at the configured rate, the DAG's total is the reported total. No second scorer."""
    context = cluster_context()
    plan = split((0, 1, 2, 3), context)
    folded = stage_cost(plan.routes, scoring_instance(context), context.cost_config)
    assert plan.search_objective_inr == pytest.approx(folded.breakdown.total_inr)


def test_a_scaled_penalty_rate_makes_the_search_objective_diverge_from_reported_cost() -> None:
    """Same routes, two numbers, on purpose — this is the GA's adaptive penalty, written down.

    The GA hands ``split`` a ``CostConfig`` whose window rate has been scaled up to guide the
    search. The plan it gets back is still reported at the configured rate. Anyone who "fixes" this
    discrepancy breaks one side or the other.
    """
    unreachable = TimeWindow(start_s=EIGHT_AM_S, end_s=Seconds(EIGHT_AM_S + 600.0))
    windows = (None, None, unreachable, unreachable)
    adaptive = CostConfig(tw_penalty_per_hour=CostConfig().tw_penalty_per_hour * 10.0)
    plan = split((0, 1, 2, 3), cluster_context(windows, adaptive))

    reported = stage_cost(plan.routes, scoring_instance(cluster_context(windows)), CostConfig())
    assert plan.search_objective_inr > reported.breakdown.total_inr


# --------------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------------

_PROPERTY_POSITIONS_M = (0.0, 0.0, 900.0, 4_100.0, 12_000.0, 12_600.0, 26_000.0)
_PROPERTY_DEMAND_KG = st.floats(
    min_value=50.0, max_value=CAPACITY_KG, allow_nan=False, allow_infinity=False
)


@given(
    permutation=st.permutations(range(PROPERTY_STOPS)),
    demand_kg=st.lists(_PROPERTY_DEMAND_KG, min_size=PROPERTY_STOPS, max_size=PROPERTY_STOPS),
)
@settings(deadline=None)
def test_every_returned_route_respects_capacity(
    permutation: list[int], demand_kg: list[float]
) -> None:
    """Capacity is enforced by construction, so no drawn permutation can produce a breach."""
    context = make_context(_PROPERTY_POSITIONS_M, demand_kg)
    plan = split(tuple(permutation), context)
    assert all(route.load_kg <= CAPACITY_KG + CAPACITY_TOLERANCE_KG for route in plan.routes)


@given(
    permutation=st.permutations(range(PROPERTY_STOPS)),
    demand_kg=st.lists(_PROPERTY_DEMAND_KG, min_size=PROPERTY_STOPS, max_size=PROPERTY_STOPS),
)
@settings(deadline=None)
def test_the_routes_concatenate_back_to_the_permutation_exactly(
    permutation: list[int], demand_kg: list[float]
) -> None:
    """Every stop served once, in the order given: no duplicates, no omissions, no reordering.

    The order matters as much as the set. ``split`` partitions a sequence; a version that sorted
    or resequenced its input would be doing the GA's job badly and invisibly.
    """
    context = make_context(_PROPERTY_POSITIONS_M, demand_kg)
    plan = split(tuple(permutation), context)
    served = stop_nodes(tuple(route.interior_nodes for route in plan.routes))
    assert served == visited(context, tuple(permutation))


@given(permutation=st.permutations(range(PROPERTY_STOPS)))
@settings(deadline=None)
def test_no_feasible_partition_beats_the_split(permutation: list[int]) -> None:
    """The guarantee the GA is built on: optimal for the order it was given, whatever that order.

    Uniform stop masses keep the enumeration's run length fixed at three, so the oracle covers
    exactly the arcs the DAG holds.
    """
    context = make_context(_PROPERTY_POSITIONS_M, [STOP_KG] * PROPERTY_STOPS)
    ordering = tuple(permutation)
    cheapest = min(
        partition_cost_inr(cuts, ordering, context)
        for cuts in cut_sequences(PROPERTY_STOPS, CLUSTER_MAX_RUN)
    )
    assert split(ordering, context).search_objective_inr == pytest.approx(cheapest)
