"""Solution representation and the single scoring path for the whole codebase.

:func:`evaluate_solution` is the only function that turns a :class:`Solution` into
:class:`Metrics`. The greedy baseline, the Stage 1 + Stage 2 pipeline, the ablation and the
OR-Tools reference all call it. A second scoring implementation — even a "quick" one inside a
fitness function — is a defect: the moment two exist, a reported improvement can come from the
scorer rather than the solver, and the headline number stops meaning anything.

Two consequences of that rule are worth stating explicitly:

*Routes do not cache their own cost.* A :class:`Route` carries physical facts — stops, load,
distance, duration, arrival times — produced by the routing layer, which owns traffic. Money is
derived here, on demand, by :func:`route_cost`. A cost field on ``Route`` would be a second
scoring path with a stale-value bug attached.

*Capacity never appears as a penalty.* It is a hard constraint enforced by construction upstream
(infeasible arcs are never created in the split DAG). This module therefore *asserts* capacity
rather than pricing it: a violation reaching here means a builder is broken, and it raises.

Time windows are soft, and this function charges them at the configured rate. The GA's adaptive
penalty multiplier is a search-guidance device that lives in the GA; it must not leak into the
reported cost, or two runs with different multipliers would be incomparable.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce

from src.config import (
    CAPACITY_TOLERANCE_KG,
    METRES_PER_KM,
    MIN_ROUTE_NODES,
    SECONDS_PER_HOUR,
    CostConfig,
)
from src.data.instance import Instance
from src.exceptions import InfeasibleSolutionError
from src.units import Metres, NodeId, Rupees, Seconds


def _require(condition: bool, message: str) -> None:
    """Raise :class:`InfeasibleSolutionError` unless ``condition`` holds."""
    if not condition:
        raise InfeasibleSolutionError(message)


@dataclass(frozen=True, slots=True)
class Route:
    """One vehicle tour: ``hub -> stops -> hub``, with its physical outcome.

    ``arrival_s`` is aligned element-wise with ``nodes`` and holds seconds from midnight,
    already traffic-adjusted by the cost layer. Carrying arrivals rather than recomputing them
    is what lets scoring stay free of any travel-time logic of its own.
    """

    hub_id: int
    nodes: tuple[NodeId, ...]
    load_kg: float
    distance_m: Metres
    duration_s: Seconds
    arrival_s: tuple[Seconds, ...]

    def __post_init__(self) -> None:
        _require(
            len(self.nodes) >= MIN_ROUTE_NODES, "a route must be hub -> at least one stop -> hub"
        )
        _require(self.nodes[0] == self.nodes[-1], "a route must start and end at its hub")
        _require(
            len(set(self.interior_nodes)) == len(self.interior_nodes),
            f"route from hub {self.hub_id} visits a stop twice",
        )
        _require(self.load_kg > 0.0, "a deployed vehicle must carry something")
        _require(self.distance_m >= 0.0, "distance_m must be non-negative")
        _require(self.duration_s > 0.0, "a route that takes no time is not a route")
        _require(
            len(self.arrival_s) == len(self.nodes),
            "arrival_s must have one entry per node, including both hub visits",
        )
        _require(
            all(a <= b for a, b in zip(self.arrival_s, self.arrival_s[1:], strict=False)),
            "arrival times must be non-decreasing along the route",
        )

    @property
    def interior_nodes(self) -> tuple[NodeId, ...]:
        """The stops, excluding the opening and closing hub visits."""
        return self.nodes[1:-1]

    @property
    def n_stops(self) -> int:
        """Number of served stops on this tour."""
        return len(self.nodes) - 2


@dataclass(frozen=True, slots=True)
class Solution:
    """A full plan: inbound consolidation tours and final-mile delivery tours.

    The two stages are held separately because they answer different questions — Stage 1 moves
    volume into hubs, Stage 2 moves it to customers — and because the headline KPI counts drops,
    which only Stage 2 makes.
    """

    stage1_routes: tuple[Route, ...]
    stage2_routes: tuple[Route, ...]

    @property
    def all_routes(self) -> tuple[Route, ...]:
        """Every tour in the plan, both stages."""
        return self.stage1_routes + self.stage2_routes

    @property
    def vehicles_used(self) -> int:
        """Vehicle-days deployed.

        One tour is charged as one vehicle-day: the fixed cost is levied per tour. A real
        operation might run a Stage 1 tour and a Stage 2 tour on the same asset, so this is an
        upper bound — applied identically to the baseline and the optimized plan, which is what
        keeps the comparison honest.
        """
        return len(self.all_routes)

    @property
    def total_distance_m(self) -> Metres:
        """Total distance driven across both stages."""
        return Metres(sum(route.distance_m for route in self.all_routes))

    @property
    def total_duration_s(self) -> Seconds:
        """Total vehicle time across both stages, including service time."""
        return Seconds(sum(route.duration_s for route in self.all_routes))

    @property
    def total_load_kg(self) -> float:
        """Total mass moved across both stages."""
        return sum(route.load_kg for route in self.all_routes)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """The cost model's four components, in INR. Sums with ``+`` so aggregation stays one-site."""

    variable_inr: Rupees
    driver_inr: Rupees
    fixed_inr: Rupees
    tw_penalty_inr: Rupees

    @classmethod
    def zero(cls) -> CostBreakdown:
        """The additive identity, for use as a fold seed."""
        return cls(Rupees(0.0), Rupees(0.0), Rupees(0.0), Rupees(0.0))

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        """Add two breakdowns component-wise."""
        return CostBreakdown(
            variable_inr=Rupees(self.variable_inr + other.variable_inr),
            driver_inr=Rupees(self.driver_inr + other.driver_inr),
            fixed_inr=Rupees(self.fixed_inr + other.fixed_inr),
            tw_penalty_inr=Rupees(self.tw_penalty_inr + other.tw_penalty_inr),
        )

    @property
    def total_inr(self) -> Rupees:
        """Sum of all four components."""
        return Rupees(self.variable_inr + self.driver_inr + self.fixed_inr + self.tw_penalty_inr)


@dataclass(frozen=True, slots=True)
class WindowOutcome:
    """How one tour fared against its customers' time windows."""

    lateness_s: Seconds
    violations: int

    @property
    def lateness_hr(self) -> float:
        """Total lateness in hours — the unit the penalty rate is quoted in."""
        return self.lateness_s / SECONDS_PER_HOUR


@dataclass(frozen=True, slots=True)
class Metrics:
    """The reported outcome of a plan.

    ``cost_per_drop_inr`` is the headline. Distance and duration sit beneath it as the drivers
    that explain it, and the secondary operational measures explain those in turn.
    """

    total_cost_inr: Rupees
    cost_per_drop_inr: Rupees
    total_distance_km: float
    total_duration_hr: float
    vehicles_used: int
    tw_violations: int
    tw_lateness_hr: float

    stops_per_hour: float
    """Stops per vehicle-hour across **both** stages — pickups and drops alike.

    Deliberately a different population from ``cost_per_drop_inr``, which divides by Stage 2
    deliveries only. This one answers "how productive is the fleet", so collection stops count;
    that one answers "what does a drop cost", so they do not. Do not multiply one by the other.
    """

    capacity_utilisation: float
    """Mean load factor over every vehicle-day deployed, both stages.

    The same freight is counted twice — once inbound to a hub, once outbound to a customer —
    because both legs consume a vehicle. That makes this a fleet-wide figure comparable across
    plans, not a per-leg one: a plan cannot improve it by shifting work between stages, only by
    filling vehicles better. Read a single stage's utilisation off its own routes instead.
    """

    breakdown: CostBreakdown


def route_window_outcome(route: Route, instance: Instance) -> WindowOutcome:
    """Measure ``route`` against the delivery windows of the customers it visits.

    Stops that are not customers — every Stage 1 source, and both hub visits — have no window
    and are skipped. Early arrival is not a violation; the vehicle waits, and the duration model
    has already charged for the wait.
    """
    lateness = 0.0
    violations = 0
    for node, arrival in zip(route.nodes, route.arrival_s, strict=True):
        if not instance.is_customer_node(node):
            continue
        window = instance.customer_at(node).window
        if window is None:
            continue
        late = window.lateness_s(arrival)
        if late > 0.0:
            lateness += late
            violations += 1
    return WindowOutcome(lateness_s=Seconds(lateness), violations=violations)


def route_cost(route: Route, cost_config: CostConfig, window: WindowOutcome) -> CostBreakdown:
    """Price one tour.

    The window outcome is passed in rather than recomputed so that scoring measures lateness
    exactly once per route, and so the arithmetic below is the only place in the codebase where
    rupees are produced.

    Args:
        route: The tour to price.
        cost_config: Rates in INR.
        window: Lateness for this tour, from :func:`route_window_outcome`.

    Returns:
        The four cost components for this tour.
    """
    return CostBreakdown(
        variable_inr=Rupees(cost_config.variable_per_km * route.distance_m / METRES_PER_KM),
        driver_inr=Rupees(cost_config.driver_per_hour * route.duration_s / SECONDS_PER_HOUR),
        fixed_inr=Rupees(cost_config.fixed_per_vehicle),
        tw_penalty_inr=Rupees(cost_config.tw_penalty_per_hour * window.lateness_hr),
    )


def evaluate_solution(solution: Solution, instance: Instance, cost_config: CostConfig) -> Metrics:
    """Score a plan. The single scoring path — every caller in the codebase comes through here.

    Args:
        solution: The plan to score.
        instance: The instance it was built for; supplies windows and vehicle capacity.
        cost_config: Rates in INR.

    Returns:
        The full :class:`Metrics` for the plan.

    Raises:
        InfeasibleSolutionError: If the plan breaks a hard constraint — a tour over capacity, a
            tour leaving from the wrong hub, a stop served twice, a stop served in the wrong
            stage, or a plan that leaves work undone (a customer undelivered, or a source with
            shipments waiting uncollected).
    """
    _validate_structure(solution, instance)

    outcomes = tuple(route_window_outcome(route, instance) for route in solution.all_routes)
    breakdown = reduce(
        operator.add,
        (
            route_cost(route, cost_config, outcome)
            for route, outcome in zip(solution.all_routes, outcomes, strict=True)
        ),
        CostBreakdown.zero(),
    )

    duration_hr = solution.total_duration_s / SECONDS_PER_HOUR
    stops = sum(route.n_stops for route in solution.all_routes)
    fleet_capacity_kg = solution.vehicles_used * instance.fleet.vehicle_capacity_kg

    return Metrics(
        total_cost_inr=breakdown.total_inr,
        # Through the instance, not through the plan's own stop count: validation has established
        # the two are equal, and dividing by the requirement says so out loud.
        cost_per_drop_inr=Rupees(breakdown.total_inr / instance.n_deliveries),
        total_distance_km=solution.total_distance_m / METRES_PER_KM,
        total_duration_hr=duration_hr,
        vehicles_used=solution.vehicles_used,
        tw_violations=sum(outcome.violations for outcome in outcomes),
        tw_lateness_hr=sum(outcome.lateness_hr for outcome in outcomes),
        stops_per_hour=stops / duration_hr,
        capacity_utilisation=solution.total_load_kg / fleet_capacity_kg,
        breakdown=breakdown,
    )


def _validate_structure(solution: Solution, instance: Instance) -> None:
    """Assert every hard constraint before any number is reported."""
    for route in solution.all_routes:
        _require(
            route.nodes[0] == instance.hub_node(route.hub_id),
            f"route claims hub {route.hub_id} but starts at node {route.nodes[0]}",
        )
        _require(
            route.load_kg <= instance.fleet.vehicle_capacity_kg + CAPACITY_TOLERANCE_KG,
            f"route from hub {route.hub_id} carries {route.load_kg} kg over capacity "
            f"{instance.fleet.vehicle_capacity_kg} kg",
        )
    _validate_stage(
        solution.stage1_routes,
        instance.is_source_node,
        "Stage 1",
        "source",
        _required_source_nodes(instance),
    )
    _validate_stage(
        solution.stage2_routes,
        instance.is_customer_node,
        "Stage 2",
        "customer",
        _required_customer_nodes(instance),
    )


def _required_source_nodes(instance: Instance) -> frozenset[NodeId]:
    """Source nodes Stage 1 must collect from: those at least one shipment originates at.

    Derived from the shipment list rather than the source list on purpose. A source with nothing
    waiting has nothing to collect, so a tour that skips it is correct rather than incomplete —
    and with origins drawn uniformly, some sources come out empty.
    """
    return frozenset(instance.source_node(shipment.source_id) for shipment in instance.shipments)


def _required_customer_nodes(instance: Instance) -> frozenset[NodeId]:
    """Customer nodes Stage 2 must deliver to: all of them.

    Every customer is the destination of exactly one shipment, enforced when the instance is
    built, so there is no such thing as a customer with nothing to receive.
    """
    return frozenset(
        instance.customer_node(customer.customer_id) for customer in instance.customers
    )


def _validate_stage(
    routes: tuple[Route, ...],
    is_legal_stop: Callable[[NodeId], bool],
    stage: str,
    stop_kind: str,
    required: frozenset[NodeId],
) -> None:
    """Check that a stage visits exactly the nodes it must, each exactly once.

    Completeness is checked here, in the scorer, because cost per drop is only comparable
    between plans that do the same work. Without it, omitting the most expensive customers
    lowers the numerator and improves the reported KPI — the cheapest plan would be the one that
    delivers least.
    """
    seen: set[NodeId] = set()
    for route in routes:
        for node in route.interior_nodes:
            _require(is_legal_stop(node), f"{stage} route stops at node {node}, not a {stop_kind}")
            _require(node not in seen, f"{stage} serves {stop_kind} node {node} more than once")
            seen.add(node)
    missing = sorted(required - seen)
    if missing:
        raise InfeasibleSolutionError(
            f"{stage} leaves {len(missing)} of {len(required)} {stop_kind}s unserved "
            f"(lowest unserved node {missing[0]})"
        )
