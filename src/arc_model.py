"""The cost model as an OR-Tools ``RoutingModel`` sees it, and the scaffolding around one.

Two solvers in this repository hand a problem to OR-Tools: the Stage 1 inbound CVRP
(:mod:`src.stage1.cvrp`) and the Stage 2 quality reference
(:mod:`src.stage2.ortools_reference`). Both need the same five things — a monetised arc, an
integer capacity dimension, a fleet size, search parameters, and a way to read tours back out of
a solved model — and §2.6 of CLAUDE.md forbids two copies of the arithmetic. This module owns
them once.

**This is neutral ground, in the sense :mod:`src.workload` and :mod:`src.tour` are.** Nothing here
imports either stage, and neither stage imports the other. It also holds no ``Instance`` and no
``Config``: every function takes the scalars and matrices it needs, which is what lets a frozen
per-hub task be built for a worker that may know nothing else (§1.1).

**Why the arc is money and not time.** A ``RoutingModel`` fixes arc costs before the search
starts, so the cumulative traffic model — which depends on *when* a leg is actually driven —
cannot live inside it. The proxy is the monetised arc:

    ``alpha × km  +  beta × hours × m(dispatch_hour)``

with ``gamma`` attached per vehicle. Duration *alone* would make the multiplier a no-op: a single
scalar multiple of the duration matrix has the same ``argmin``. The multiplier only changes a
decision once it is traded against ``₹/km``, which is why this returns rupees.

The proxy decides *which* tour is chosen and nothing else. Every distance, duration and arrival
time either solver reports comes from :func:`src.tour.build_route` over the real
:class:`~src.costs.traffic.TrafficModel`, band blending and all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.config import (
    CAPACITY_TOLERANCE_KG,
    COST_SCALE_MILLI_INR,
    GRAMS_PER_KG,
    METRES_PER_KM,
    SECONDS_PER_HOUR,
    CostConfig,
    FleetConfig,
)
from src.costs.traffic import TrafficModel
from src.units import DemandArray, DistanceMatrix, DurationMatrix

VisitOrders = tuple[tuple[int, ...], ...]
"""One tuple of stop positions per deployed vehicle, indexing into ``HubWorkload.nodes``."""

DEPOT = 0
"""Local node index of the hub in a per-hub model. Stops occupy 1..k."""

_MILLISECONDS_PER_SECOND = 1000


@dataclass(frozen=True, slots=True)
class ArcRates:
    """The cost model as OR-Tools sees it: money per arc, plus money per vehicle.

    ``traffic_multiplier`` is the static stand-in for the cumulative model — see the module
    docstring for why a static multiplier is the only kind an arc cost can carry, and why it
    still changes the answer once distance and time are priced against each other.
    """

    variable_per_km: float
    driver_per_hour: float
    fixed_per_vehicle: float
    traffic_multiplier: float


def arc_rates(cost: CostConfig, traffic: TrafficModel, dispatch_hour: float) -> ArcRates:
    """Bundle the cost rates with the traffic multiplier in force at dispatch.

    Takes the dispatch hour as a scalar rather than an :class:`~src.data.instance.Instance` so
    this module stays free of the instance model, and so a caller that has already sliced its
    problem down for a worker can still build the rates.

    Args:
        cost: Rates in INR.
        traffic: The multiplier schedule.
        dispatch_hour: Clock hour the fleet leaves at.

    Returns:
        The rates a ``RoutingModel`` will price arcs at.
    """
    return ArcRates(
        variable_per_km=cost.variable_per_km,
        driver_per_hour=cost.driver_per_hour,
        fixed_per_vehicle=cost.fixed_per_vehicle,
        traffic_multiplier=traffic.multiplier_for_hour(int(dispatch_hour)),
    )


def arc_cost_milli_inr(
    distance_m: DistanceMatrix, duration_s: DurationMatrix, rates: ArcRates
) -> npt.NDArray[np.int64]:
    """Price every arc of one hub's block in integer milli-rupees.

    Vectorised over the whole block rather than evaluated inside the transit callback: OR-Tools
    calls that callback tens of thousands of times per solve, and the arithmetic is identical for
    every call.

    Args:
        distance_m: This hub's ``(k+1, k+1)`` distance block, in metres.
        duration_s: The matching free-flow duration block, in seconds.
        rates: Rates and the static dispatch-hour multiplier.

    Returns:
        The same block, priced in integer milli-rupees.
    """
    inr = (
        rates.variable_per_km * distance_m / METRES_PER_KM
        + rates.driver_per_hour * rates.traffic_multiplier * duration_s / SECONDS_PER_HOUR
    )
    scaled: npt.NDArray[np.int64] = np.rint(inr * COST_SCALE_MILLI_INR).astype(np.int64)
    return scaled


def demand_grams(demand_kg: DemandArray) -> tuple[int, ...]:
    """One hub's stop demands as integer grams, with a leading zero for the depot.

    Grams rather than kilograms because a ``RoutingModel``'s capacity dimension is integer-valued
    and the default 37.5 kg shipment is not a whole number of kilograms: rounding to integer
    kilograms would drift by up to half a kilo per stop and could let a 20-stop tour appear to fit
    a vehicle it does not. See :data:`~src.config.GRAMS_PER_KG`.

    Args:
        demand_kg: Mass waiting at each stop, in ``HubWorkload.nodes`` order.

    Returns:
        Demands in grams, indexed by local node — the depot's zero first.
    """
    return (0, *(round(kg * GRAMS_PER_KG) for kg in demand_kg))


def capacity_grams(capacity_kg: float) -> int:
    """One vehicle's capacity in integer grams, matching :func:`demand_grams`.

    Separate from :func:`demand_grams` because the two are indexed differently — one per stop,
    one per vehicle — but they must use the same scale or the dimension compares grams against
    kilograms and every vehicle looks a thousand times too large.
    """
    return round(capacity_kg * GRAMS_PER_KG)


def vehicle_count(total_kg: float, fleet: FleetConfig) -> int:
    """Vehicles to offer one hub: its mass floor, times the configured slack, rounded up.

    The floor is reduced by the capacity tolerance before dividing, because a load summed in
    floating point can land a fraction of a microgram over an exact multiple of capacity and
    would otherwise buy a whole extra vehicle for no mass at all.

    Offering more vehicles than needed is safe and deliberate: unused ones stay at the depot and
    are dropped, and the fixed charge attached to each vehicle means the search prefers not to
    deploy them. Offering too few would make a solvable hub infeasible.

    Args:
        total_kg: Mass this hub must move.
        fleet: Supplies vehicle capacity and the slack factor.

    Returns:
        The vehicle count to offer, at least one.
    """
    floor = math.ceil((total_kg - CAPACITY_TOLERANCE_KG) / fleet.vehicle_capacity_kg)
    return max(1, math.ceil(floor * fleet.vehicle_slack_factor))


def search_parameters(
    time_limit_s: float, solution_limit: int
) -> pywrapcp.DefaultRoutingSearchParameters:
    """First solution by cheapest arc, then guided local search under the given limits.

    ``solution_limit`` of 1 returns the first-solution result and is independent of the time
    limit, which is what makes a seeded test reproducible.

    **Zero means unlimited, and the guard around the assignment is load-bearing.**
    ``DefaultRoutingSearchParameters()`` arrives with ``solution_limit`` pre-set to ``int64`` max —
    *not* to the protobuf default for the field, which is 0. So writing our zero straight through
    would cap the search at zero improved solutions rather than removing the cap, and the field's
    own default would not rescue it. Measured, not assumed: unset reads back as
    ``9223372036854775807``, and ``tests/test_arc_model.py`` pins that.

    Args:
        time_limit_s: Wall-clock budget for this solve.
        solution_limit: Improved solutions to accept before stopping; zero means unlimited.

    Returns:
        Search parameters ready to hand to ``SolveWithParameters``.
    """
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(int(time_limit_s * _MILLISECONDS_PER_SECOND))
    if solution_limit > 0:
        parameters.solution_limit = solution_limit
    parameters.log_search = False
    return parameters


def visit_orders(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    assignment: pywrapcp.Assignment,
    n_vehicles: int,
) -> VisitOrders:
    """Read each vehicle's tour out of a solved model as stop positions.

    Walked from the node *after* the start depot, so the returned positions index
    ``HubWorkload.nodes`` directly and a vehicle that never left reads as an empty walk. Local
    node ``i`` is stop ``i - 1``, since the hub occupies local index 0.

    Args:
        routing: The solved model.
        manager: Its index manager, for the index-to-node translation.
        assignment: The solution to read.
        n_vehicles: Vehicles the model was given.

    Returns:
        One tuple of stop positions per vehicle that was actually deployed, in visit order.
        Vehicles left at the depot are omitted, so the caller never builds an empty tour.
    """
    orders: list[tuple[int, ...]] = []
    for vehicle in range(n_vehicles):
        index = assignment.Value(routing.NextVar(routing.Start(vehicle)))
        order: list[int] = []
        while not routing.IsEnd(index):
            order.append(int(manager.IndexToNode(index)) - 1)
            index = assignment.Value(routing.NextVar(index))
        if order:
            orders.append(tuple(order))
    return tuple(orders)
