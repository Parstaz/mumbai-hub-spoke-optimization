"""The single scoring path for the whole codebase.

:func:`evaluate_solution` is the only function that turns a :class:`~src.solution.Solution` into
:class:`Metrics`. The greedy baseline, the Stage 1 + Stage 2 pipeline, the ablation and the
OR-Tools reference all call it. A second scoring implementation — even a "quick" one inside a
fitness function — is a defect: the moment two exist, a reported improvement can come from the
scorer rather than the solver, and the headline number stops meaning anything.

:func:`stage_cost` is the fold underneath it, exposed because a stage-level pipeline step has one
stage's tours and needs to report what they cost. It is not a second scoring path: it prices
routes through the same :func:`route_cost` and performs no validation, so it cannot be mistaken
for a verdict on a plan. :func:`evaluate_solution` calls it exactly once, over both stages
concatenated.

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
    SECONDS_PER_HOUR,
    CostConfig,
)
from src.data.instance import Instance
from src.exceptions import InfeasibleSolutionError
from src.solution import Route, Solution
from src.units import Metres, NodeId, Rupees, Seconds


def _require(condition: bool, message: str) -> None:
    """Raise :class:`InfeasibleSolutionError` unless ``condition`` holds."""
    if not condition:
        raise InfeasibleSolutionError(message)


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
class StageCost:
    """What one set of tours costs, and how it fared against its windows.

    Holds only the quantities that need the cost model and the instance to compute. Distance,
    duration, load and vehicle count are physical facts already owned by
    :class:`~src.solution.Solution`; a caller wanting them for a single stage builds a
    ``Solution`` over just those routes rather than having this type sum them a second time.
    """

    breakdown: CostBreakdown
    stops: int
    tw_violations: int
    tw_lateness_hr: float


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


def leg_cost(
    distance_m: Metres, duration_s: Seconds, lateness_s: Seconds, cost_config: CostConfig
) -> CostBreakdown:
    """Price one vehicle-day from its physical outcome. The only site that produces rupees.

    Takes scalars rather than a :class:`~src.solution.Route` because Stage 2's split DAG prices
    candidate tours it will never build a route for: of the thousands one split evaluates, three
    or four survive into the plan. Addressing the arithmetic by scalar is what lets
    :mod:`src.stage2.pricing` reuse this function instead of copying it — the difference between
    one cost model and two with a performance argument attached.

    Args:
        distance_m: Road distance driven, hub back to hub.
        duration_s: Wall-clock seconds from leaving the hub to returning, service included.
        lateness_s: Total seconds by which this tour missed its customers' windows.
        cost_config: Rates in INR.

    Returns:
        The four cost components for this vehicle-day.
    """
    return CostBreakdown(
        variable_inr=Rupees(cost_config.variable_per_km * distance_m / METRES_PER_KM),
        driver_inr=Rupees(cost_config.driver_per_hour * duration_s / SECONDS_PER_HOUR),
        fixed_inr=Rupees(cost_config.fixed_per_vehicle),
        tw_penalty_inr=Rupees(cost_config.tw_penalty_per_hour * (lateness_s / SECONDS_PER_HOUR)),
    )


def route_cost(route: Route, cost_config: CostConfig, window: WindowOutcome) -> CostBreakdown:
    """Price one tour, reading its physical outcome off the route.

    The window outcome is passed in rather than recomputed so that scoring measures lateness
    exactly once per route. The arithmetic itself lives in :func:`leg_cost`; this function is the
    route-shaped way in, and the two cannot drift because there is only one of them.

    Args:
        route: The tour to price.
        cost_config: Rates in INR.
        window: Lateness for this tour, from :func:`route_window_outcome`.

    Returns:
        The four cost components for this tour.
    """
    return leg_cost(route.distance_m, route.duration_s, window.lateness_s, cost_config)


def stage_cost(routes: tuple[Route, ...], instance: Instance, cost_config: CostConfig) -> StageCost:
    """Fold :func:`route_cost` and :func:`route_window_outcome` over a set of tours.

    The single aggregation site in the codebase, and the reason it takes a route tuple rather than
    a :class:`~src.solution.Solution`: a stage-level pipeline step has one stage's tours and no
    other, and the alternative to this function is that step adding up rupees itself. Splitting
    the fold out is what keeps :func:`evaluate_solution` the only *scoring* path while still
    letting a Stage 1 run report its inbound leg.

    No validation happens here. Completeness and capacity are properties of a whole plan, so they
    stay in :func:`evaluate_solution`; folding a partial plan is legitimate and must not be
    mistaken for scoring one.

    Args:
        routes: The tours to aggregate, in the order they should be summed.
        instance: Supplies the delivery windows lateness is measured against.
        cost_config: Rates in INR.

    Returns:
        The summed cost breakdown, stop count and window outcome for ``routes``.
    """
    outcomes = tuple(route_window_outcome(route, instance) for route in routes)
    breakdown = reduce(
        operator.add,
        (
            route_cost(route, cost_config, outcome)
            for route, outcome in zip(routes, outcomes, strict=True)
        ),
        CostBreakdown.zero(),
    )
    return StageCost(
        breakdown=breakdown,
        stops=sum(route.n_stops for route in routes),
        tw_violations=sum(outcome.violations for outcome in outcomes),
        tw_lateness_hr=sum(outcome.lateness_hr for outcome in outcomes),
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

    # One fold over both stages concatenated, not one fold per stage added together. Float
    # addition is not associative, so regrouping the sum could move the last bits of a reported
    # figure; this way the arithmetic is bit-identical to summing the routes in plan order.
    aggregate = stage_cost(solution.all_routes, instance, cost_config)
    breakdown = aggregate.breakdown

    duration_hr = solution.total_duration_s / SECONDS_PER_HOUR
    fleet_capacity_kg = solution.vehicles_used * instance.fleet.vehicle_capacity_kg

    return Metrics(
        total_cost_inr=breakdown.total_inr,
        # Through the instance, not through the plan's own stop count: validation has established
        # the two are equal, and dividing by the requirement says so out loud.
        cost_per_drop_inr=Rupees(breakdown.total_inr / instance.n_deliveries),
        total_distance_km=solution.total_distance_m / METRES_PER_KM,
        total_duration_hr=duration_hr,
        vehicles_used=solution.vehicles_used,
        tw_violations=aggregate.tw_violations,
        tw_lateness_hr=aggregate.tw_lateness_hr,
        stops_per_hour=aggregate.stops / duration_hr,
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
