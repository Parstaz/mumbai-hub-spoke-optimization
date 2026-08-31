"""Tests for the solution model and the single scoring path.

The expected costs here are worked out by hand from the cost model rather than captured from a
previous run, so the suite would catch a change to the rates or to the aggregation — which is
the whole point of having exactly one scoring function.

Reference plan used throughout (default rates: ₹9/km, ₹95/h, ₹1000/vehicle, ₹250/h late):

* Stage 1 — hub 0 → source (node 1) → hub 0: 5 km, 0.5 h
  ₹45 variable + ₹47.50 driver + ₹1000 fixed = ₹1092.50
* Stage 2 — hub 0 → customers (nodes 2, 3) → hub 0: 10 km, 1.0 h
  ₹90 variable + ₹95 driver + ₹1000 fixed = ₹1185
* Total ₹2277.50 over 2 drops = ₹1138.75 per drop
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import CostConfig, FleetConfig, GeoConfig, ScheduleConfig
from src.data.instance import (
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.exceptions import InfeasibleSolutionError, OptimizationError
from src.solution import (
    CostBreakdown,
    Metrics,
    Route,
    Solution,
    evaluate_solution,
    route_cost,
    route_window_outcome,
)
from src.units import Rupees, Seconds
from tests.conftest import build_instance, make_route

COSTS = CostConfig()

IDLE_SOURCE_GEO = GeoConfig(
    n_hubs=1,
    n_sources=2,
    n_customers=2,
    n_density_clusters=1,
    hub_candidate_pool=10,
)

STAGE1_ROUTE = make_route((0, 1, 0), load_kg=75.0, distance_m=5_000.0, duration_s=1_800.0)
STAGE2_ROUTE = make_route((0, 2, 3, 0), load_kg=75.0, distance_m=10_000.0, duration_s=3_600.0)


def _reference_solution() -> Solution:
    """The plan documented in the module docstring."""
    return Solution(stage1_routes=(STAGE1_ROUTE,), stage2_routes=(STAGE2_ROUTE,))


def _instance_with_idle_source() -> Instance:
    """Two sources, but both shipments originate at source 0 — source 1 has nothing waiting.

    Node layout: hub 0 is node 0, sources 0 and 1 are nodes 1 and 2, customers 0 and 1 are
    nodes 3 and 4.
    """
    return Instance(
        seed=0,
        geo=IDLE_SOURCE_GEO,
        fleet=FleetConfig(),
        schedule=ScheduleConfig(),
        hubs=(Hub(0, Coordinate(19.00, 72.90)),),
        sources=(Source(0, Coordinate(19.05, 72.95)), Source(1, Coordinate(19.06, 72.96))),
        customers=(
            Customer(0, Coordinate(19.10, 73.00), None),
            Customer(1, Coordinate(19.15, 73.05), None),
        ),
        shipments=(
            Shipment(0, source_id=0, customer_id=0, size_kg=37.5),
            Shipment(1, source_id=0, customer_id=1, size_kg=37.5),
        ),
    )


def _plan(stage2_route: Route) -> Solution:
    """A complete plan: the reference Stage 1 tour plus ``stage2_route``.

    The scorer only accepts plans that do all the work, so a test varying the delivery tour
    still has to ship the collection tour that feeds it.
    """
    return Solution(stage1_routes=(STAGE1_ROUTE,), stage2_routes=(stage2_route,))


def test_route_derived_properties() -> None:
    """Stops exclude both hub visits; interior nodes are what a stage validates."""
    assert STAGE2_ROUTE.n_stops == 2
    assert STAGE2_ROUTE.interior_nodes == (2, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"nodes": (0, 0)}, id="hub-to-hub-with-no-stops"),
        pytest.param({"nodes": (0, 2, 3)}, id="open-tour"),
        pytest.param({"nodes": (0, 2, 2, 0)}, id="stop-visited-twice"),
        pytest.param({"load_kg": 0.0}, id="empty-vehicle"),
        pytest.param({"duration_s": 0.0}, id="instant-route"),
        pytest.param({"distance_m": -1.0}, id="negative-distance"),
    ],
)
def test_route_rejects_malformed_input(kwargs: dict[str, object]) -> None:
    """Structural guards fire at construction, before a route can reach the scorer."""
    arguments: dict[str, object] = {
        "nodes": (0, 2, 3, 0),
        "load_kg": 75.0,
        "distance_m": 10_000.0,
        "duration_s": 3_600.0,
        **kwargs,
    }
    with pytest.raises(InfeasibleSolutionError):
        make_route(**arguments)  # type: ignore[arg-type]  # parametrised kwargs are mixed types


def test_route_rejects_arrival_times_that_do_not_match_the_stops() -> None:
    """One arrival per node, including both hub visits — otherwise windows misalign."""
    with pytest.raises(InfeasibleSolutionError):
        make_route((0, 2, 3, 0), arrival_s=(28_800.0, 29_000.0, 29_500.0))


def test_route_rejects_arrival_times_that_go_backwards() -> None:
    """A tour cannot travel back in time; this would silently absolve a late delivery."""
    with pytest.raises(InfeasibleSolutionError):
        make_route((0, 2, 3, 0), arrival_s=(28_800.0, 30_000.0, 29_000.0, 31_000.0))


def test_solution_aggregates_both_stages() -> None:
    """Derived totals span both stages, since both consume vehicles, distance and time."""
    solution = _reference_solution()
    assert solution.vehicles_used == 2
    assert solution.total_distance_m == 15_000.0
    assert solution.total_duration_s == 5_400.0
    assert solution.total_load_kg == 150.0


def test_cost_breakdown_arithmetic() -> None:
    """The fold seed is the additive identity and addition is component-wise."""
    left = CostBreakdown(Rupees(1.0), Rupees(2.0), Rupees(3.0), Rupees(4.0))
    assert left + CostBreakdown.zero() == left
    assert (left + left).total_inr == pytest.approx(20.0)


def test_route_cost_components(tiny_instance: Instance) -> None:
    """One route, priced by hand: ₹90 variable, ₹95 driver, ₹1000 fixed, no lateness."""
    outcome = route_window_outcome(STAGE2_ROUTE, tiny_instance)
    breakdown = route_cost(STAGE2_ROUTE, COSTS, outcome)
    assert breakdown.variable_inr == pytest.approx(90.0)
    assert breakdown.driver_inr == pytest.approx(95.0)
    assert breakdown.fixed_inr == pytest.approx(1000.0)
    assert breakdown.tw_penalty_inr == pytest.approx(0.0)
    assert breakdown.total_inr == pytest.approx(1185.0)


def test_evaluate_solution_matches_hand_computed_metrics(tiny_instance: Instance) -> None:
    """The headline number and every driver beneath it, computed by hand from the cost model."""
    metrics = evaluate_solution(_reference_solution(), tiny_instance, COSTS)
    assert metrics.total_cost_inr == pytest.approx(2277.50)
    assert metrics.cost_per_drop_inr == pytest.approx(1138.75)
    assert metrics.total_distance_km == pytest.approx(15.0)
    assert metrics.total_duration_hr == pytest.approx(1.5)
    assert metrics.vehicles_used == 2
    assert metrics.tw_violations == 0
    assert metrics.tw_lateness_hr == pytest.approx(0.0)
    assert metrics.stops_per_hour == pytest.approx(3 / 1.5)
    assert metrics.capacity_utilisation == pytest.approx(150.0 / 1500.0)
    assert metrics.breakdown.total_inr == pytest.approx(metrics.total_cost_inr)


def test_metrics_breakdown_sums_to_total(tiny_instance: Instance) -> None:
    """The breakdown is a decomposition of the reported total, not a parallel calculation."""
    metrics = evaluate_solution(_reference_solution(), tiny_instance, COSTS)
    components = (
        metrics.breakdown.variable_inr,
        metrics.breakdown.driver_inr,
        metrics.breakdown.fixed_inr,
        metrics.breakdown.tw_penalty_inr,
    )
    assert sum(components) == pytest.approx(metrics.total_cost_inr)


def test_driver_cost_makes_duration_matter(tiny_instance: Instance) -> None:
    """The coupling that lets the traffic model change the answer: slower is dearer.

    A route of identical length that takes an hour longer must cost ₹95 more. If this ever
    fails, the objective has become distance-only and traffic has stopped mattering.
    """
    slower = dataclasses.replace(
        STAGE2_ROUTE,
        duration_s=Seconds(7_200.0),
        arrival_s=(Seconds(28_800.0), Seconds(31_000.0), Seconds(34_000.0), Seconds(36_000.0)),
    )
    base = evaluate_solution(_plan(STAGE2_ROUTE), tiny_instance, COSTS)
    delayed = evaluate_solution(_plan(slower), tiny_instance, COSTS)
    assert delayed.total_cost_inr - base.total_cost_inr == pytest.approx(95.0)


def test_time_window_penalty_is_charged_once_per_late_stop() -> None:
    """Two hours late at one stop costs ₹500 and counts as a single violation."""
    instance = build_instance((TimeWindow(Seconds(28_800.0), Seconds(30_000.0)), None))
    route = make_route(
        (0, 2, 3, 0),
        arrival_s=(28_800.0, 37_200.0, 38_000.0, 39_000.0),  # customer 0 is 2 h late
    )
    metrics = evaluate_solution(_plan(route), instance, COSTS)
    assert metrics.tw_violations == 1
    assert metrics.tw_lateness_hr == pytest.approx(2.0)
    assert metrics.breakdown.tw_penalty_inr == pytest.approx(500.0)


def test_arrival_exactly_at_the_window_close_is_on_time() -> None:
    """The window boundary is inclusive; one second later is not."""
    instance = build_instance((TimeWindow(Seconds(28_800.0), Seconds(30_000.0)), None))
    on_time = make_route((0, 2, 3, 0), arrival_s=(28_800.0, 30_000.0, 30_500.0, 31_000.0))
    late = make_route((0, 2, 3, 0), arrival_s=(28_800.0, 30_001.0, 30_500.0, 31_000.0))
    assert evaluate_solution(_plan(on_time), instance, COSTS).tw_violations == 0
    assert evaluate_solution(_plan(late), instance, COSTS).tw_violations == 1


def test_all_day_customers_are_never_late(tiny_instance: Instance) -> None:
    """A customer with no window cannot be violated, however late the vehicle arrives."""
    route = make_route((0, 2, 3, 0), arrival_s=(28_800.0, 80_000.0, 85_000.0, 86_000.0))
    metrics = evaluate_solution(_plan(route), tiny_instance, COSTS)
    assert metrics.tw_violations == 0
    assert metrics.breakdown.tw_penalty_inr == pytest.approx(0.0)


def test_stage1_routes_are_not_counted_as_drops(tiny_instance: Instance) -> None:
    """Cost per drop divides by deliveries, not by stops: pickups are not drops."""
    metrics = evaluate_solution(_reference_solution(), tiny_instance, COSTS)
    assert metrics.cost_per_drop_inr == pytest.approx(metrics.total_cost_inr / 2)


def test_load_exactly_at_capacity_is_accepted(tiny_instance: Instance) -> None:
    """The capacity boundary is legal, including the float-summation edge at exactly 750 kg.

    Both tours are loaded to exactly capacity so that a utilisation of 1.0 still means what the
    name says — every vehicle full — rather than being diluted by a half-empty companion tour.
    """
    at_capacity = 20 * 37.5
    full_delivery = dataclasses.replace(STAGE2_ROUTE, load_kg=at_capacity)
    full_collection = dataclasses.replace(STAGE1_ROUTE, load_kg=at_capacity)
    metrics = evaluate_solution(
        Solution((full_collection,), (full_delivery,)), tiny_instance, COSTS
    )
    assert metrics.capacity_utilisation == pytest.approx(1.0)


def test_load_one_unit_over_capacity_is_rejected(tiny_instance: Instance) -> None:
    """Capacity is hard. Scoring asserts it rather than pricing it as a penalty."""
    overloaded = dataclasses.replace(STAGE2_ROUTE, load_kg=750.5)
    with pytest.raises(InfeasibleSolutionError, match="over capacity"):
        evaluate_solution(Solution((), (overloaded,)), tiny_instance, COSTS)


def test_route_leaving_from_the_wrong_hub_is_rejected(small_instance: Instance) -> None:
    """A route's declared hub must match the node it departs from.

    ``small_instance`` has two hubs (nodes 0 and 1), three sources (2–4) and four customers
    (5–8), so hub 1 exists and the mismatch is the only thing wrong with this route.
    """
    mismatched = dataclasses.replace(make_route((0, 5, 0)), hub_id=1)
    with pytest.raises(InfeasibleSolutionError, match="claims hub"):
        evaluate_solution(Solution((), (mismatched,)), small_instance, COSTS)


def test_route_claiming_a_nonexistent_hub_is_rejected(tiny_instance: Instance) -> None:
    """A hub id outside the instance fails as an instance lookup, still under the domain root."""
    mismatched = dataclasses.replace(STAGE2_ROUTE, hub_id=99)
    with pytest.raises(OptimizationError, match="unknown hub_id"):
        evaluate_solution(Solution((), (mismatched,)), tiny_instance, COSTS)


def test_stage1_route_stopping_at_a_customer_is_rejected(tiny_instance: Instance) -> None:
    """Stage 1 collects from sources. A customer stop there would break stage precedence."""
    wrong_stage = make_route((0, 2, 0))
    with pytest.raises(InfeasibleSolutionError, match="not a source"):
        evaluate_solution(Solution((wrong_stage,), (STAGE2_ROUTE,)), tiny_instance, COSTS)


def test_stage2_route_stopping_at_a_source_is_rejected(tiny_instance: Instance) -> None:
    """And the converse: final-mile tours serve customers only."""
    wrong_stage = make_route((0, 1, 0))
    with pytest.raises(InfeasibleSolutionError, match="not a customer"):
        evaluate_solution(Solution((STAGE1_ROUTE,), (wrong_stage,)), tiny_instance, COSTS)


def test_a_customer_served_by_two_routes_is_rejected(tiny_instance: Instance) -> None:
    """Double-serving a customer would understate cost per drop by inventing a drop."""
    first = make_route((0, 2, 3, 0))
    second = make_route((0, 3, 0))
    with pytest.raises(InfeasibleSolutionError, match="more than once"):
        evaluate_solution(Solution((STAGE1_ROUTE,), (first, second)), tiny_instance, COSTS)


def test_a_stage1_only_plan_is_rejected(tiny_instance: Instance) -> None:
    """Collecting without delivering leaves every customer unserved.

    Replaces an earlier check on a "delivers nothing" guard: completeness subsumes it, and the
    rejection now names how much work was left undone.
    """
    with pytest.raises(InfeasibleSolutionError, match="Stage 2 leaves 2 of 2 customers unserved"):
        evaluate_solution(Solution((STAGE1_ROUTE,), ()), tiny_instance, COSTS)


def test_a_complete_plan_is_accepted(tiny_instance: Instance) -> None:
    """A plan that collects from every stocked source and delivers to every customer scores."""
    metrics = evaluate_solution(_reference_solution(), tiny_instance, COSTS)
    assert metrics.vehicles_used == 2
    assert metrics.cost_per_drop_inr > 0.0


def test_a_plan_missing_one_customer_is_rejected(tiny_instance: Instance) -> None:
    """Omission is the cheapest way to look good, so the scorer refuses to price it.

    Skipping customer 1 removes its distance, duration and lateness from the numerator while
    the denominator stays at the instance's drop count. Left unchecked, the plan that delivers
    least would win.
    """
    short = make_route((0, 2, 0))
    with pytest.raises(InfeasibleSolutionError, match="Stage 2 leaves 1 of 2 customers unserved"):
        evaluate_solution(Solution((STAGE1_ROUTE,), (short,)), tiny_instance, COSTS)


def test_a_plan_missing_a_shipment_bearing_source_is_rejected(tiny_instance: Instance) -> None:
    """Freight left uncollected at a source is undone work, exactly like an undelivered drop."""
    with pytest.raises(InfeasibleSolutionError, match="Stage 1 leaves 1 of 1 sources unserved"):
        evaluate_solution(Solution((), (STAGE2_ROUTE,)), tiny_instance, COSTS)


def test_a_plan_skipping_a_source_with_no_shipments_is_accepted() -> None:
    """A source with nothing waiting need not be visited.

    Requirement comes from the shipment list, not the source list — with origins drawn
    uniformly some sources come out empty, and a tour that drives to one would be wasteful, not
    thorough.
    """
    instance = _instance_with_idle_source()
    collection = make_route((0, 1, 0))  # source 0 only; source 1 (node 2) has nothing waiting
    delivery = make_route((0, 3, 4, 0))
    metrics = evaluate_solution(Solution((collection,), (delivery,)), instance, COSTS)
    assert metrics.vehicles_used == 2
    assert metrics.cost_per_drop_inr > 0.0


def test_zero_rate_config_prices_a_plan_at_zero(tiny_instance: Instance) -> None:
    """An all-zero cost config is a legal ablation and produces a zero-cost plan."""
    free = CostConfig(
        variable_per_km=0.0, driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=0.0
    )
    metrics = evaluate_solution(_reference_solution(), tiny_instance, free)
    assert metrics.total_cost_inr == pytest.approx(0.0)
    assert metrics.cost_per_drop_inr == pytest.approx(0.0)


def test_metrics_is_frozen(tiny_instance: Instance) -> None:
    """Reported metrics cannot be edited after the fact."""
    metrics = evaluate_solution(_reference_solution(), tiny_instance, COSTS)
    assert isinstance(metrics, Metrics)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.total_cost_inr = Rupees(0.0)  # type: ignore[misc]  # frozen-ness is the assertion


def test_route_is_frozen() -> None:
    """Routes are immutable, so a scored plan cannot drift under the scorer."""
    assert isinstance(STAGE2_ROUTE, Route)
    with pytest.raises(dataclasses.FrozenInstanceError):
        STAGE2_ROUTE.load_kg = 1.0  # type: ignore[misc]  # frozen-ness is the assertion
