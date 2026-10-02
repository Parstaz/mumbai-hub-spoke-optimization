"""Tests for the Stage 2 OR-Tools quality reference.

Every case sets ``solution_limit=1``, stopping OR-Tools at the first-solution heuristic. That is
not a shortcut for speed: guided local search under a wall-clock limit returns whatever it had
reached when the clock ran out, so a test asserting on a GLS result would pass or fail according
to how busy the machine was. The exception is the starved-budget test, which is *about* the clock.

What is asserted is the set of properties that would let an unfair comparison look fair: that the
reference solves the same problem the GA does, that capacity stays hard and windows stay soft,
that the arrival timeline matches the GA's, and — most important — that a budget too short to find
a plan is reported rather than papered over.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.arc_model import DEPOT, ArcRates, search_parameters, vehicle_count
from src.config import (
    Config,
    GAConfig,
    GeoConfig,
    ReferenceConfig,
    RunConfig,
    Stage1Config,
)
from src.costs.matrix import CostMatrices, HaversineProvider
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.data.instance import Instance, TimeWindow
from src.exceptions import InfeasibleInstanceError
from src.scoring import evaluate_solution
from src.solution import Solution
from src.stage1.assignment import unconstrained
from src.stage1.cvrp import solve_stage1
from src.stage2.ortools_reference import (
    _TIME_DIMENSION,
    HubReference,
    MatchedRun,
    ReferencePlan,
    ReferenceTask,
    _add_arc_cost,
    _add_capacity,
    _add_time_windows,
    matched_run,
    solve_hub_reference,
    solve_reference,
    status_label,
)
from src.stage2.pricing import HubPricing
from src.stage2.solve import hub_of_source, solve_stage2
from src.units import Seconds
from src.workload import hub_of_customer

SMALL = Config(
    geo=GeoConfig(
        n_hubs=3, n_sources=8, n_customers=14, n_density_clusters=2, hub_candidate_pool=100
    ),
    ga=GAConfig(
        population_size=8,
        generations=4,
        tournament_k=3,
        elitism_count=1,
        seeded_individuals=1,
        penalty_adapt_interval=2,
    ),
    stage1=Stage1Config(cvrp_solution_limit=1, workers=1),
    reference=ReferenceConfig(solution_limit=1),
    run=RunConfig(seed=11, use_osrm=False),
)

DISPATCH_S = Seconds(8 * 3600.0)
SERVICE_S = Seconds(300.0)


def build(config: Config) -> tuple[Instance, CostMatrices, TrafficModel]:
    """The instance and matrices, haversine so no test touches the network."""
    instance = generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)
    distance_m, duration_s = HaversineProvider(
        circuity_factor=config.run.circuity_factor, speed_kmph=config.run.haversine_speed_kmph
    ).matrix(instance.coordinates())
    return (
        instance,
        CostMatrices(distance_m=distance_m, duration_s=duration_s),
        TrafficModel.from_config(config.traffic),
    )


BUDGET_S = 2.0
"""Per-hub budget for the tests that need a complete reference plan.

Deliberately **not** the GA's own clocks, and for a narrower reason than it might appear. On an
instance this small the GA converges in milliseconds — a one-stop hub stagnates in about 3 ms — and
the matched budget does in fact solve every hub at that, verified on seed 11. What it does not do
is solve them *dependably on any machine*: a few milliseconds leaves no margin, so a loaded CI box
could miss and turn an assertion about plan structure into a flaky one.

So the budget is pinned here to keep these tests about what a complete plan looks like. That is not
a dodge — ``solve_reference`` takes the budget as an argument, so this exercises its real contract —
and the matching itself is asserted separately against the GA's actual clocks by
:func:`test_every_hub_is_given_the_budget_its_ga_search_spent`, with
:func:`test_a_matched_budget_can_legitimately_be_too_short` covering both outcomes of a real one.
"""


def solved(config: Config = SMALL) -> tuple[Instance, Solution, ReferencePlan]:
    """Run both solvers over one instance, with the reference on a budget it can actually use.

    Both get the same inbound plan and therefore the same customer-to-hub mapping, which is the
    condition that makes the two columns comparable at all.
    """
    instance, matrices, traffic = build(config)
    inbound = solve_stage1(instance, matrices, traffic, unconstrained, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    ga = solve_stage2(instance, matrices, traffic, customer_hubs, config)
    generous = MatchedRun(
        hub_of_stop=customer_hubs,
        seconds_by_hub={run.outcome.hub_id: BUDGET_S for run in ga.runs},
    )
    reference = solve_reference(instance, matrices, traffic, generous, config)
    return instance, Solution(inbound, ga.routes), reference


def uniform_task(
    *,
    windows: tuple[TimeWindow | None, ...] = (None, None, None),
    demand_g: tuple[int, ...] = (0, 1000, 1000, 1000),
    budget: tuple[float, int] = (5.0, 1),
    multiplier: float = 1.0,
    n_vehicles: int = 1,
) -> ReferenceTask:
    """A hand-built four-node task: depot plus three stops, with distinct asymmetric legs.

    Durations are distinct and asymmetric on purpose, so an error in the timeline cannot cancel
    out the way it would on a symmetric or uniform matrix.

    ``budget`` pairs the wall clock with the solution limit because the two are only ever varied
    together here — a starved solve needs the limit lifted or the time limit is not what binds.
    """
    time_limit_s, solution_limit = budget
    duration_s = np.array(
        [
            [0.0, 600.0, 1200.0, 1800.0],
            [600.0, 0.0, 300.0, 900.0],
            [1200.0, 300.0, 0.0, 400.0],
            [1800.0, 900.0, 400.0, 0.0],
        ]
    )
    return ReferenceTask(
        hub_id=0,
        local_distance_m=duration_s * 5.0,
        local_duration_s=duration_s,
        demand_g=demand_g,
        capacity_g=750_000,
        n_vehicles=n_vehicles,
        rates=ArcRates(
            variable_per_km=9.0,
            driver_per_hour=95.0,
            fixed_per_vehicle=1000.0,
            traffic_multiplier=multiplier,
        ),
        pricing=HubPricing(windows=windows),
        dispatch_s=DISPATCH_S,
        service_s=SERVICE_S,
        lateness_per_second=69,
        time_limit_s=time_limit_s,
        solution_limit=solution_limit,
    )


@dataclasses.dataclass(frozen=True)
class Solved:
    """One hand-built task's solve, opened up enough to assert on the objective and the timeline."""

    orders: tuple[tuple[int, ...], ...]
    objective: int
    walk: tuple[tuple[int, int], ...]
    horizon: int


def objective_of(task: ReferenceTask) -> Solved:
    """Solve one task directly, returning its order, objective and arrival timeline.

    Goes through the same three model builders :func:`solve_hub_reference` uses, so what is
    asserted is the real model rather than a re-description of it. What it adds is access to the
    objective and the time dimension, which ``HubReference`` deliberately does not carry —
    nothing in the pipeline needs the proxy's own objective, and reporting it would invite
    comparing it against :func:`~src.scoring.evaluate_solution`'s figure.
    """
    manager = pywrapcp.RoutingIndexManager(len(task.demand_g), task.n_vehicles, DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    _add_arc_cost(routing, manager, task)
    _add_capacity(routing, manager, task)
    _add_time_windows(routing, manager, task)
    assignment = routing.SolveWithParameters(
        search_parameters(task.time_limit_s, task.solution_limit)
    )
    assert assignment is not None
    dimension = routing.GetDimensionOrDie(_TIME_DIMENSION)

    walk: list[tuple[int, int]] = []
    orders: list[tuple[int, ...]] = []
    for vehicle in range(task.n_vehicles):
        index = routing.Start(vehicle)
        order: list[int] = []
        while not routing.IsEnd(index):
            node = int(manager.IndexToNode(index))
            if vehicle == 0:
                walk.append((node, assignment.Value(dimension.CumulVar(index))))
            if node != DEPOT:
                order.append(node - 1)
            index = assignment.Value(routing.NextVar(index))
        if vehicle == 0:
            walk.append(
                (int(manager.IndexToNode(index)), assignment.Value(dimension.CumulVar(index)))
            )
        if order:
            orders.append(tuple(order))
    return Solved(
        orders=tuple(orders),
        objective=int(assignment.ObjectiveValue()),
        walk=tuple(walk),
        horizon=int(dimension.CumulVar(routing.End(0)).Max()),
    )


def arrival_times(task: ReferenceTask) -> list[tuple[int, int]]:
    """The first vehicle's node-and-arrival walk, for the timeline assertions."""
    return list(objective_of(task).walk)


# --------------------------------------------------------------------------------------------
# The plan it produces


def test_the_reference_plan_passes_the_shared_scorer() -> None:
    """The same validation the GA's plan passes: completeness, capacity, structure, right stage.

    Scored with the GA's own inbound leg, because the reference replaces Stage 2 and nothing else —
    which is also what makes the two columns differ in exactly one thing.
    """
    instance, ga_solution, reference = solved()
    assert reference.complete

    metrics = evaluate_solution(
        Solution(stage1_routes=ga_solution.stage1_routes, stage2_routes=reference.routes),
        instance,
        SMALL.cost,
    )

    assert metrics.total_cost_inr > 0.0
    assert metrics.vehicles_used == len(ga_solution.stage1_routes) + len(reference.routes)


def test_both_solvers_are_given_the_same_hubs_and_the_same_customers() -> None:
    """If the two partitioned the hubs differently, the gap would report that as solver quality."""
    _, ga_solution, reference = solved()

    ga_stops = {node for route in ga_solution.stage2_routes for node in route.interior_nodes}
    reference_stops = {node for route in reference.routes for node in route.interior_nodes}
    assert ga_stops == reference_stops

    ga_hubs = {route.hub_id for route in ga_solution.stage2_routes}
    assert {route.hub_id for route in reference.routes} == ga_hubs


def test_every_hub_is_given_the_budget_its_ga_search_spent() -> None:
    """The matched budget, asserted against the GA's own recorded clocks rather than a constant.

    This is the test that makes the comparison a matched-budget one. It deliberately uses the real
    clocks, however short, and asserts only the budgets — not that a plan came back.
    """
    instance, matrices, traffic = build(SMALL)
    inbound = solve_stage1(instance, matrices, traffic, unconstrained, SMALL)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    ga = solve_stage2(instance, matrices, traffic, customer_hubs, SMALL)

    reference = solve_reference(
        instance, matrices, traffic, matched_run(customer_hubs, ga.runs), SMALL
    )

    spent = {run.outcome.hub_id: run.elapsed_s for run in ga.runs}
    assert {hub.hub_id: hub.budget_s for hub in reference.hubs} == spent


def test_a_matched_budget_can_legitimately_be_too_short() -> None:
    """On a tiny instance the GA's own clocks are milliseconds, and the reference may find nothing.

    Not a defect in either solver, and not something the run is allowed to paper over: every hub
    still comes back with a status and a budget, and the plan simply is not complete. Asserted both
    ways so a faster machine cannot make this pass vacuously.
    """
    instance, matrices, traffic = build(SMALL)
    inbound = solve_stage1(instance, matrices, traffic, unconstrained, SMALL)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    ga = solve_stage2(instance, matrices, traffic, customer_hubs, SMALL)

    reference = solve_reference(
        instance, matrices, traffic, matched_run(customer_hubs, ga.runs), SMALL
    )

    assert len(reference.hubs) == len(ga.runs), "every hub is accounted for either way"
    assert all(hub.budget_s > 0.0 for hub in reference.hubs)
    if reference.complete:
        assert reference.unsolved == ()
    else:
        assert all(hub.orders is None for hub in reference.unsolved)
        assert all(hub.vehicles_deployed == 0 for hub in reference.unsolved)


def test_no_tour_exceeds_capacity() -> None:
    """Capacity is a hard dimension, never a penalty — §1.1, same as the GA's split DAG."""
    instance, _, reference = solved()

    assert all(
        route.load_kg <= instance.fleet.vehicle_capacity_kg + 1e-6 for route in reference.routes
    )


def test_no_empty_tour_is_ever_emitted() -> None:
    """Vehicles left at the depot must be dropped, not turned into zero-stop routes."""
    _, _, reference = solved()

    assert all(route.n_stops >= 1 for route in reference.routes)
    assert all(route.load_kg > 0.0 for route in reference.routes)


# --------------------------------------------------------------------------------------------
# The arrival timeline, which must match the GA's


def test_the_time_dimension_reproduces_the_ga_arrival_timeline() -> None:
    """Service rides the outgoing arc and is zero out of the depot, as ``TourPricer`` accumulates.

    Recomputed by hand along whatever tour the solver picked: dispatch, travel to the first stop,
    then service-plus-travel for every leg after it. If service were charged at the depot too, or
    on the incoming arc instead, every arrival after the first would be off by 300 s.
    """
    task = uniform_task()
    walk = arrival_times(task)

    clock = float(DISPATCH_S)
    for position, (node, cumul) in enumerate(walk):
        if position > 0:
            previous = walk[position - 1][0]
            if position > 1:
                clock += float(SERVICE_S)
            clock += float(task.local_duration_s[previous, node])
        assert cumul == pytest.approx(round(clock), abs=1)


def test_the_traffic_multiplier_stretches_the_timeline() -> None:
    """The model's timeline is static but not free-flow: it runs at the dispatch-hour multiplier.

    A reference that timed its tours at free flow while the GA timed them at ×1.6 would judge
    windows against a day that is 60% shorter than the one it is scored against.
    """
    free_flow = arrival_times(uniform_task(multiplier=1.0))
    peak = arrival_times(uniform_task(multiplier=1.6))

    free_flow_elapsed = free_flow[-1][1] - float(DISPATCH_S)
    peak_elapsed = peak[-1][1] - float(DISPATCH_S)
    assert peak_elapsed > free_flow_elapsed


def test_the_horizon_cannot_bind_on_the_longest_possible_tour() -> None:
    """A binding horizon would silently turn a soft window into a hard deadline.

    Checked against a single vehicle forced to visit every stop, which is the longest tour the
    model admits and therefore the case closest to the bound.
    """
    task = uniform_task(n_vehicles=1)
    walk = arrival_times(task)

    manager = pywrapcp.RoutingIndexManager(len(task.demand_g), 1, DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    _add_arc_cost(routing, manager, task)
    _add_capacity(routing, manager, task)
    _add_time_windows(routing, manager, task)
    horizon = routing.GetDimensionOrDie(_TIME_DIMENSION).CumulVar(routing.End(0)).Max()

    assert len(walk) == len(task.demand_g) + 1, "the fixture must force one tour over every stop"
    assert horizon > walk[-1][1]


def test_a_horizon_bounding_the_sum_rather_than_each_arc_would_reject_this_hub() -> None:
    """Regression: individually-rounded arcs can exceed the rounding of their sum.

    These are seed 11 hub 1's real numbers — one stop, two arcs of 6,985.66 s, service on the
    return only. Bounding the *total* gives ``round(6985.66 x 2 + 300) = 14,271`` while the two
    arcs actually cost ``round(6985.66) + round(6985.66 + 300) = 14,272``. The horizon then binds
    by one second and OR-Tools proves this trivially-servable hub infeasible.

    Asserted on a solve rather than on the arithmetic, because the arithmetic is only wrong in
    what it lets the model do.
    """
    leg_s = 4366.038077449 / 1.0
    duration_s = np.array([[0.0, leg_s], [leg_s, 0.0]])
    task = dataclasses.replace(
        uniform_task(windows=(None,), n_vehicles=2, demand_g=(0, 37_500)),
        local_distance_m=duration_s * 5.0,
        local_duration_s=duration_s,
        rates=ArcRates(
            variable_per_km=9.0,
            driver_per_hour=95.0,
            fixed_per_vehicle=1000.0,
            traffic_multiplier=1.6,
        ),
    )

    hub = solve_hub_reference(task)

    assert hub.solved, "a one-stop hub is servable by any vehicle; the horizon must not bind"
    assert hub.orders == ((0,),)


# --------------------------------------------------------------------------------------------
# Windows are soft


def test_an_unmeetable_window_still_returns_a_complete_plan() -> None:
    """Soft, not hard: §1.1. A hard bound would make a tight instance infeasible, not expensive."""
    impossible = TimeWindow(start_s=Seconds(0.0), end_s=Seconds(1.0))
    task = uniform_task(windows=(impossible, impossible, impossible))

    hub = solve_hub_reference(task)

    assert hub.solved
    assert sum(len(order) for order in hub.orders or ()) == 3


def test_lateness_enters_the_objective_at_the_configured_coefficient() -> None:
    """Behavioural proof the soft bound actually reaches the objective, priced as configured.

    Asserted on the objective rather than on the chosen order. A registered-but-unreached penalty
    would leave both identical, and every window figure in the comparison would then be measuring
    an unconstrained solve — but the *order* is a weak probe of that: these fixtures are symmetric,
    so reversing a tour costs the same and ``solution_limit=1`` stops at a first-solution
    heuristic that does not consult the bound at all.

    The objective is exact arithmetic instead. Holding the order fixed, adding a window the tour
    misses must raise the objective by precisely ``lateness_s x lateness_per_second``.
    """
    blind_walk = arrival_times(uniform_task())
    last_node, last_arrival = blind_walk[-2]
    deadline = float(last_arrival) - 500.0
    windows: list[TimeWindow | None] = [None, None, None]
    windows[last_node - 1] = TimeWindow(start_s=Seconds(0.0), end_s=Seconds(deadline))

    blind = objective_of(uniform_task())
    windowed = objective_of(uniform_task(windows=tuple(windows)))

    assert windowed.orders == blind.orders, "the fixture must hold the order fixed"
    assert windowed.objective - blind.objective == round(500.0) * 69


# --------------------------------------------------------------------------------------------
# A budget too short, and a broken model: the two must not be confused


def test_a_budget_too_short_to_find_a_plan_is_reported_not_substituted() -> None:
    """§3a of the step 8 plan: no retry outside the budget, no greedy fill-in, no silent drop.

    Asserted on the contract rather than on a specific status code, because which code a starved
    solve lands on is OR-Tools' business. The complementary branch is asserted too, so that a
    machine fast enough to solve even this cannot make the test pass vacuously.
    """
    hub = solve_hub_reference(uniform_task(budget=(1e-6, 0)))

    if hub.solved:
        assert hub.vehicles_deployed > 0
        assert not hub.budget_truncated or hub.orders is not None
    else:
        assert hub.orders is None
        assert hub.vehicles_deployed == 0
        assert hub.budget_s == 1e-6
        assert status_label(hub.status) != f"UNKNOWN({hub.status})"


def test_an_incomplete_reference_plan_is_not_complete_and_cannot_be_scored() -> None:
    """The consequence that matters: no complete plan means no cost per drop, by construction.

    Built directly rather than by starving a solve, so the branch is covered deterministically
    even on a machine where every hub does find a plan.
    """
    plan = ReferencePlan(
        hubs=(
            HubReference(
                hub_id=0,
                orders=((0,),),
                status=int(routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS),
                budget_s=1.0,
                elapsed_s=0.5,
                vehicles_offered=1,
                stops=1,
            ),
            HubReference(
                hub_id=1,
                orders=None,
                status=int(routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT),
                budget_s=0.001,
                elapsed_s=0.001,
                vehicles_offered=2,
                stops=5,
            ),
        ),
        routes=(),
    )

    assert not plan.complete
    assert [hub.hub_id for hub in plan.unsolved] == [1]
    assert plan.hubs[1].vehicles_deployed == 0
    assert not plan.hubs[1].at_fleet_ceiling


def test_a_model_that_admits_no_plan_raises_instead_of_reading_as_a_timeout() -> None:
    """A modelling bug must never be reportable as "the budget was too short".

    Reachable only by building the task directly: ``require_servable`` and ``vehicle_count``
    together make it impossible upstream, which is exactly why the two groups have to be
    distinguished by status rather than by assuming one cannot happen.
    """
    # One vehicle of 750 kg offered against 3,000 kg of demand: genuinely infeasible.
    task = uniform_task(demand_g=(0, 1_000_000, 1_000_000, 1_000_000), n_vehicles=1)

    with pytest.raises(InfeasibleInstanceError, match="model unsolvable"):
        solve_hub_reference(task)


def test_an_unrecognised_status_is_shown_rather_than_hidden() -> None:
    """A future OR-Tools code must surface as itself, not as silence."""
    assert status_label(10_000) == "UNKNOWN(10000)"
    assert status_label(int(routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS)) == (
        "ROUTING_SUCCESS"
    )


# --------------------------------------------------------------------------------------------
# Matching, and the worker payload


def test_a_hub_with_no_matched_budget_is_refused() -> None:
    """Defaulting to some arbitrary limit would unmatch the budget silently."""
    instance, matrices, traffic = build(SMALL)
    inbound = solve_stage1(instance, matrices, traffic, unconstrained, SMALL)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    ga = solve_stage2(instance, matrices, traffic, customer_hubs, SMALL)
    missing = matched_run(customer_hubs, ga.runs[:-1])

    with pytest.raises(InfeasibleInstanceError, match="no matched GA budget"):
        solve_reference(instance, matrices, traffic, missing, SMALL)


def test_two_runs_claiming_the_same_hub_are_refused() -> None:
    """A duplicate would drop one hub's clock in favour of another's, unnoticed."""
    instance, matrices, traffic = build(SMALL)
    inbound = solve_stage1(instance, matrices, traffic, unconstrained, SMALL)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    ga = solve_stage2(instance, matrices, traffic, customer_hubs, SMALL)

    with pytest.raises(InfeasibleInstanceError, match="both claim hub"):
        matched_run(customer_hubs, (ga.runs[0], ga.runs[0]))


def test_a_worker_payload_carries_no_shared_state() -> None:
    """CLAUDE.md §1.1, checked structurally because it has no behavioural symptom."""
    annotations = {str(field.type) for field in dataclasses.fields(ReferenceTask)}

    assert "Instance" not in annotations
    assert "Config" not in annotations
    assert not any("Generator" in annotation for annotation in annotations)
    assert ReferenceTask.__dataclass_params__.frozen


def test_the_fleet_offered_follows_the_rule_stage_one_uses() -> None:
    """Same ``vehicle_count``, so neither solver is handed a fleet the other was denied."""
    _, _, reference = solved()

    for hub in reference.hubs:
        expected = vehicle_count(hub.stops * SMALL.fleet.shipment_size_kg, SMALL.fleet)
        assert hub.vehicles_offered == expected
