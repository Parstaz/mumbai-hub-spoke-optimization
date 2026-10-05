"""A mature solver on the final mile, for one purpose: measuring how good the GA is.

**This is not part of the pipeline and must never be called from it.** It sits behind
``--reference`` on :mod:`src.cli.run_pipeline`, and ``tests/test_closure.py`` walks the import
closure of :mod:`src.stage2.solve` and :mod:`src.stage1.cvrp` to prove neither can reach it.
CLAUDE.md §1.1 requires that; the reason is that a reference the pipeline depended on would stop
being a reference and start being a component, at which point there is nothing left to compare.

**The GA is never tuned against this.** The measured gap is the deliverable, whichever way it
falls, on the same standard capacity-balanced assignment was reported under.

Why this is a fair fight, constraint by constraint:

* **Same problem.** The customer-to-hub grouping comes from
  :func:`~src.stage2.solve.hub_workloads` — the function the GA uses, not a copy — so both solvers
  partition the hubs identically. Same instance, same matrices, same windows, same
  :func:`~src.workload.require_servable` rule, and both plans are priced by the one
  :func:`~src.scoring.evaluate_solution`.
* **Same fleet.** :func:`~src.arc_model.vehicle_count`, the rule Stage 1 already uses.
* **Same arc.** :func:`~src.arc_model.arc_cost_milli_inr`, shared rather than reimplemented.
* **Same budget**, per hub, defined below.

**The budget is the wall clock that hub's GA actually spent.** Not the configured generation
budget: every hub on seed 42 stops on ``stagnation_limit`` well short of 600 generations, so
quoting the budget would describe a run that did not happen — and generations are not a currency
OR-Tools spends. Per hub rather than one aggregate, because the hubs are independent contests and
the 1,422 s headline is a *makespan* set by the largest hub; handing a 30-stop hub that figure
would give it two orders of magnitude more search than the GA had. The reference therefore runs in
the same process as the GA it is matched against, under the same pool contention: a budget
measured on an idle machine and spent on a busy one is not the same budget.

This is a matched-budget comparison and not a matched-to-convergence one. The GA stopped early by
choice, and the reference is given what the GA *spent*, not what it was *offered*. Reading it the
other way would turn the GA's own stopping rule into a handicap imposed on its opponent.

**Two proxies, both of which bias the reference and neither of which is hidden.** Arc costs in a
``RoutingModel`` are fixed before the search starts, so the cumulative band-blended traffic model
cannot live inside it: travel time here is static, at the dispatch-hour multiplier, exactly as in
Stage 1 (README limitation 7). The time dimension the windows are checked against is therefore
also a static timeline, so the reference optimises against a slightly different arrival profile
than the one it is later scored on. Both are proxies for *choosing* a tour. Once an ordering comes
back, :func:`~src.tour.build_route` rebuilds it over the global matrices and the real
:class:`~src.costs.traffic.TrafficModel`, so every reported figure is the cumulative one.

**A hub whose budget buys no plan is a result, not an error.** See :func:`solve_hub_reference`.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.arc_model import (
    DEPOT,
    ArcRates,
    VisitOrders,
    arc_cost_milli_inr,
    arc_rates,
    capacity_grams,
    demand_grams,
    search_parameters,
    vehicle_count,
    visit_orders,
)
from src.config import COST_SCALE_MILLI_INR, SECONDS_PER_HOUR, Config
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import Instance
from src.exceptions import InfeasibleInstanceError
from src.solution import Route
from src.stage2.pricing import HubPricing, hub_pricing
from src.stage2.solve import HubRun, hub_workloads
from src.tour import RoutingContext, build_route
from src.units import DistanceMatrix, DurationMatrix, NodeArray, Seconds
from src.workload import HubWorkload

logger = logging.getLogger(__name__)

_CAPACITY_DIMENSION = "Capacity"
_TIME_DIMENSION = "Time"

_Status = routing_enums_pb2.RoutingSearchStatus

_TRUNCATED_STATUSES = frozenset({_Status.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED})
"""Statuses meaning "a plan, but the search was still mid-descent when the clock stopped it".

Necessary but **not sufficient** for detecting budget truncation — see
:attr:`HubReference.clock_stopped`.
"""

_CLOCK_TOLERANCE_S = 0.5
"""Slack when deciding a solve spent its whole budget.

OR-Tools honours a time limit to within scheduling noise, so an exact equality test would read a
clock-stopped hub as having finished early. Half a second against budgets of 10 s and up is two
orders of magnitude below the quantity being judged.
"""

_BROKEN_MODEL_STATUSES = frozenset({_Status.ROUTING_INVALID, _Status.ROUTING_INFEASIBLE})
"""Statuses that are a *proof* the model is wrong — a bug here, never a budget outcome.

Held apart from the no-plan statuses deliberately. :func:`~src.workload.require_servable` has
passed and the vehicle count comes from the hub's own mass floor, so a feasible plan exists; a
solver *proving* otherwise is describing a broken model. Folding this in with "the budget was too
short" would let a modelling bug be reported as a finding about search time, which is the one
reading that would make the whole comparison worthless.

**``ROUTING_FAIL`` is deliberately absent, which an earlier draft of this module got wrong.** Its
meaning is "no solution found", which conflates *could not* with *had no time to* — and measured
on this repository's own instances, a hub whose matched budget is a few milliseconds returns
``ROUTING_FAIL`` or ``ROUTING_FAIL_TIMEOUT`` *nondeterministically* for the same task, which is
the wall-clock irreproducibility of §9 showing up in the status code. Raising on it turned a
one-stop hub with a 3.5 ms budget into a crash. Genuine infeasibility is reported as
``ROUTING_INFEASIBLE``, verified directly against both ways this model can be made infeasible
(demand beyond the offered fleet, and a single stop beyond one vehicle).

The residual case is a model that is *both* infeasible and starved: it is reported as no-plan
rather than raising. That is the right way round — the report names the hub and prints the status,
so it stays visible, and a real measurement gives every hub a budget long enough for infeasibility
to be proved rather than merely unreached.
"""


def status_label(status: int) -> str:
    """Name a ``RoutingSearchStatus`` for a report, falling back to the raw value.

    The fallback matters: a future OR-Tools may add a code this module has never seen, and a
    report that printed nothing would hide it. An unrecognised status is shown as its number
    rather than guessed at.
    """
    try:
        name: str = _Status.Value.Name(status)
    except ValueError:
        return f"UNKNOWN({status})"
    return name


@dataclass(frozen=True, slots=True)
class MatchedRun:
    """The GA run a reference solve is matched to: its grouping, and its per-hub budgets.

    Both halves have to travel together. The grouping decides *which* customers each hub serves
    and the budget decides how long its opponent gets, so pairing one run's mapping with another
    run's timings would silently compare two different experiments.
    """

    hub_of_stop: NodeArray
    seconds_by_hub: Mapping[int, float]


def matched_run(hub_of_stop: NodeArray, runs: tuple[HubRun, ...]) -> MatchedRun:
    """Build a :class:`MatchedRun` from the GA's own per-hub results.

    Args:
        hub_of_stop: The customer-to-hub mapping the GA solved under.
        runs: The GA's per-hub runs, from :attr:`~src.stage2.solve.Stage2Plan.runs`.

    Returns:
        The grouping and the per-hub wall clocks, keyed by hub id.

    Raises:
        InfeasibleInstanceError: If two runs claim the same hub. A duplicate would silently drop
            one hub's budget in favour of another's, and the reference would be measured against
            a search length that never happened.
    """
    seconds: dict[int, float] = {}
    for run in runs:
        hub_id = run.outcome.hub_id
        if hub_id in seconds:
            raise InfeasibleInstanceError(
                f"two GA runs both claim hub {hub_id}; a matched budget needs one clock per hub"
            )
        seconds[hub_id] = run.elapsed_s
    return MatchedRun(hub_of_stop=hub_of_stop, seconds_by_hub=seconds)


@dataclass(frozen=True, slots=True)
class ReferenceTask:
    """One hub's complete, self-contained final-mile problem — all a worker may know.

    Holds no ``Instance``, no ``Config`` and no ``Generator``, matching
    :class:`~src.stage2.solve.Stage2Task`. There is no seed because a ``RoutingModel`` search takes
    none; what makes a solve vary is the wall clock, which is why ``solution_limit`` exists.
    """

    hub_id: int
    local_distance_m: DistanceMatrix
    local_duration_s: DurationMatrix
    demand_g: tuple[int, ...]
    capacity_g: int
    n_vehicles: int
    rates: ArcRates
    pricing: HubPricing
    dispatch_s: Seconds
    service_s: Seconds
    lateness_per_second: int
    time_limit_s: float
    solution_limit: int


@dataclass(frozen=True, slots=True)
class _TaskInputs:
    """What every hub's task is built from; parent-side only, never crossing the pool.

    Exists so :func:`_hub_task` stays inside §2.2's five-parameter limit, and because these four
    genuinely do not vary between hubs: slicing a task out of them is the only per-hub work. The
    ``Instance`` and ``Config`` here are legal precisely because this type stays in the parent —
    :class:`ReferenceTask` is what a worker receives, and it holds neither.
    """

    instance: Instance
    matrices: CostMatrices
    rates: ArcRates
    config: Config


@dataclass(frozen=True, slots=True)
class HubReference:
    """What the reference managed on one hub, including the case where it managed nothing.

    ``orders`` is ``None`` when the matched budget bought no plan. That is reported rather than
    patched: a first-solution retry outside the budget, a greedy fill-in, or quietly dropping the
    hub would each put a number in the table for a solve that did not happen.
    """

    hub_id: int
    orders: VisitOrders | None
    status: int
    budget_s: float
    elapsed_s: float
    vehicles_offered: int
    stops: int
    solution_limit: int
    """Carried so the report can say the budget was not what actually stopped the search.

    Under ``--deterministic`` this is 1 and the solve returns its first-solution heuristic
    immediately, spending almost none of the matched budget. The resulting gap is then not a
    matched-budget measurement at all, and a table that did not say so would be the most
    misleading output this module could produce.
    """

    @property
    def solved(self) -> bool:
        """Whether this hub has a plan at all."""
        return self.orders is not None

    @property
    def first_solution_only(self) -> bool:
        """Whether the search was stopped at its first solution rather than by the clock."""
        return self.solution_limit == 1

    @property
    def vehicles_deployed(self) -> int:
        """Tours the reference actually deployed; zero when it found no plan."""
        return 0 if self.orders is None else len(self.orders)

    @property
    def at_fleet_ceiling(self) -> bool:
        """Whether every offered vehicle was used, so the fleet cap may have bound the search."""
        return self.solved and self.vehicles_deployed >= self.vehicles_offered

    @property
    def clock_stopped(self) -> bool:
        """Whether the budget, rather than the search itself, ended this solve.

        Measured on the clock and not on the status, because the status does not say this. On seed
        42 all 16 hubs returned ``ROUTING_SUCCESS`` while 13 of them had spent their budget to the
        tenth of a second: ``ROUTING_SUCCESS`` means the solver *holds a solution at a local
        optimum*, not that it had finished. Guided local search escapes local optima repeatedly, so
        sitting at one when the clock stops says nothing about whether more time would have helped.

        Keying the caveat on :data:`_TRUNCATED_STATUSES` alone therefore under-reports: it stayed
        silent on a run where 81% of hubs were cut off. CLAUDE.md §8.6 records the same mistake
        made the other way round for the GA's arms, and the rule it leaves is the one applied here
        — a truncation claim is made from what the run measured, never from what a status name
        suggests.
        """
        return self.solved and self.elapsed_s >= self.budget_s - _CLOCK_TOLERANCE_S

    @property
    def budget_truncated(self) -> bool:
        """Whether the budget ended this solve, by either signal.

        Either the solver said so (``PARTIAL_SUCCESS``: stopped mid-descent) or the clock shows it
        (:attr:`clock_stopped`). Both mean the reference's column is a lower bound on what it would
        reach given longer.
        """
        return self.solved and (self.status in _TRUNCATED_STATUSES or self.clock_stopped)


@dataclass(frozen=True, slots=True)
class ReferencePlan:
    """Every hub's reference result, and the tours of those that produced one.

    ``routes`` covers only the solved hubs, so it is a **complete plan only when**
    :attr:`complete` holds. :func:`~src.scoring.evaluate_solution` validates completeness and will
    refuse a partial plan, which is the correct outcome: there is no honest cost per drop for a
    plan that does not deliver to everyone.
    """

    hubs: tuple[HubReference, ...]
    routes: tuple[Route, ...]

    @property
    def complete(self) -> bool:
        """Whether every hub returned a plan, and the routes therefore form a scorable solution."""
        return all(hub.solved for hub in self.hubs)

    @property
    def unsolved(self) -> tuple[HubReference, ...]:
        """The hubs whose matched budget bought no plan, in hub order."""
        return tuple(hub for hub in self.hubs if not hub.solved)


def solve_reference(
    instance: Instance,
    matrices: CostMatrices,
    traffic: TrafficModel,
    matched: MatchedRun,
    config: Config,
) -> ReferencePlan:
    """Solve every hub's final mile with OR-Tools, under the GA's own per-hub budgets.

    Args:
        instance: The instance being solved; supplies windows and the customer node space.
        matrices: Instance-wide distance and duration matrices — the same ones the GA used.
        traffic: The multiplier schedule. Static at dispatch inside the model, cumulative in the
            tours this returns.
        matched: The GA run being matched: its hub grouping and its per-hub wall clocks.
        config: Rates, fleet and the reference's own search settings.

    Returns:
        Each hub's outcome, and the tours of the hubs that produced one.

    Raises:
        InfeasibleInstanceError: If a customer is owed more than one vehicle can carry, if a hub
            has no matched budget, or if a hub's model admits no solution.
    """
    workloads = hub_workloads(instance, matched.hub_of_stop, config.fleet.vehicle_capacity_kg)
    inputs = _TaskInputs(
        instance=instance,
        matrices=matrices,
        rates=arc_rates(config.cost, traffic, instance.schedule.dispatch_hour),
        config=config,
    )
    tasks = tuple(_hub_task(workload, inputs, matched) for workload in workloads)
    hubs = _solve_all(tasks, config.stage1.workers)

    routing = RoutingContext(
        matrices=matrices,
        traffic=traffic,
        start_time_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
    )
    routes = tuple(
        build_route(workload, positions, routing)
        for workload, hub in zip(workloads, hubs, strict=True)
        for positions in (hub.orders or ())
    )
    return ReferencePlan(hubs=hubs, routes=routes)


def _hub_task(workload: HubWorkload, inputs: _TaskInputs, matched: MatchedRun) -> ReferenceTask:
    """Slice one hub's problem out of the instance-wide matrices, and attach its matched budget.

    Raises:
        InfeasibleInstanceError: If this hub has no recorded GA wall clock. Defaulting to some
            arbitrary limit would quietly unmatch the budget on exactly the hub a reader would
            want to check.
    """
    if workload.hub_id not in matched.seconds_by_hub:
        raise InfeasibleInstanceError(
            f"hub {workload.hub_id} has no matched GA budget; the reference cannot be given a "
            f"search length the GA did not spend"
        )
    instance, config = inputs.instance, inputs.config
    local = np.concatenate(([workload.hub_node], workload.nodes))
    block = np.ix_(local, local)
    return ReferenceTask(
        hub_id=workload.hub_id,
        local_distance_m=inputs.matrices.distance_m[block],
        local_duration_s=inputs.matrices.duration_s[block],
        demand_g=demand_grams(workload.demand_kg),
        capacity_g=capacity_grams(config.fleet.vehicle_capacity_kg),
        n_vehicles=vehicle_count(float(workload.demand_kg.sum()), config.fleet),
        rates=inputs.rates,
        pricing=hub_pricing(workload, instance),
        dispatch_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_s=Seconds(instance.fleet.service_time_per_stop_s),
        lateness_per_second=_lateness_per_second(config),
        time_limit_s=matched.seconds_by_hub[workload.hub_id],
        solution_limit=config.reference.solution_limit,
    )


def _lateness_per_second(config: Config) -> int:
    """The window penalty as an integer milli-rupee charge per second late.

    ``RoutingModel`` soft bounds take an integer coefficient, so the configured ₹250/hour becomes
    69 milli-INR/s where the exact figure is 69.44 — the reference's proxy therefore under-prices
    lateness by 0.64%. It affects which tour the search picks and nothing it is scored at: the
    reported penalty comes from :func:`~src.scoring.leg_cost` at the configured rate.
    """
    return round(config.cost.tw_penalty_per_hour / SECONDS_PER_HOUR * COST_SCALE_MILLI_INR)


def _solve_all(tasks: tuple[ReferenceTask, ...], workers: int) -> tuple[HubReference, ...]:
    """Solve every hub, sequentially or across a spawn pool, and return results in hub order.

    Spawn for the same reason the other two pools use it: ``RoutingModel`` starts threads and
    forking a threaded process is undefined.
    """
    count = workers if workers > 0 else (os.cpu_count() or 1)
    if len(tasks) <= 1 or count == 1:
        finished = []
        for task in tasks:
            finished.append(solve_hub_reference(task))
            _log_finished(finished[-1], len(finished), len(tasks))
        return tuple(finished)

    with multiprocessing.get_context("spawn").Pool(processes=count) as pool:
        finished = []
        for hub in pool.imap_unordered(solve_hub_reference, tasks):
            finished.append(hub)
            _log_finished(hub, len(finished), len(tasks))
    return tuple(sorted(finished, key=lambda hub: hub.hub_id))


def _log_finished(hub: HubReference, done: int, total: int) -> None:
    """Report one hub as it lands, naming the status rather than only whether it worked."""
    logger.info(
        "reference hub %2d done (%2d/%d): %3d stops, %.1f s of %.1f s budget, %d/%d vehicles, %s",
        hub.hub_id,
        done,
        total,
        hub.stops,
        hub.elapsed_s,
        hub.budget_s,
        hub.vehicles_deployed,
        hub.vehicles_offered,
        status_label(hub.status),
    )


def solve_hub_reference(task: ReferenceTask) -> HubReference:
    """Solve one hub's final mile. The pool worker — module-level and pure, so it pickles.

    **A budget too short to find anything is a reported result, not a failure.** Some hubs
    stagnate fast, so a small hub may hand the reference only a few seconds. If the solver has no
    plan when the clock stops, this returns ``orders=None`` with the status that says so, and the
    report names the hub. Substituting anything — a retry outside the budget, a greedy tour — would
    put a figure in the table for a solve that did not happen, and silently dropping the hub would
    be worse still, because the remaining hubs are then the easy ones.

    A genuinely infeasible model is a different thing and raises. See
    :data:`_BROKEN_MODEL_STATUSES`.

    Args:
        task: This hub's self-contained problem, carrying its matched budget.

    Returns:
        This hub's reference result, with a plan when the budget bought one.

    Raises:
        InfeasibleInstanceError: If the model admits no solution. Servability has been checked and
            the vehicle count comes from the mass floor, so this means a bug in the model.
    """
    manager = pywrapcp.RoutingIndexManager(len(task.demand_g), task.n_vehicles, DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    _add_arc_cost(routing, manager, task)
    _add_capacity(routing, manager, task)
    _add_time_windows(routing, manager, task)

    started = time.perf_counter()
    assignment = routing.SolveWithParameters(
        search_parameters(task.time_limit_s, task.solution_limit)
    )
    elapsed_s = time.perf_counter() - started
    status = int(routing.status())

    if status in _BROKEN_MODEL_STATUSES:
        raise InfeasibleInstanceError(
            f"OR-Tools proves hub {task.hub_id}'s final-mile model unsolvable "
            f"({status_label(status)}): {len(task.demand_g) - 1} stops, {task.n_vehicles} "
            f"vehicles of {task.capacity_g} g. Servability passed and the vehicle count comes "
            f"from the mass floor, so this is a model bug, not a hard instance or a short budget"
        )
    return HubReference(
        hub_id=task.hub_id,
        orders=(
            visit_orders(routing, manager, assignment, task.n_vehicles)
            if assignment is not None
            else None
        ),
        status=status,
        budget_s=task.time_limit_s,
        elapsed_s=elapsed_s,
        vehicles_offered=task.n_vehicles,
        stops=len(task.demand_g) - 1,
        solution_limit=task.solution_limit,
    )


def _add_arc_cost(
    routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager, task: ReferenceTask
) -> None:
    """Price every arc in money, and charge the fixed vehicle cost per deployed vehicle.

    Travel only — service time is excluded on purpose. Every stop is served exactly once however
    the tours are partitioned, so total service is a constant and cannot move the ``argmin``;
    including it would add the same number to every candidate.
    """
    arc_cost = arc_cost_milli_inr(task.local_distance_m, task.local_duration_s, task.rates)

    def transit(from_index: int, to_index: int) -> int:
        return int(arc_cost[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(transit))
    routing.SetFixedCostOfAllVehicles(round(task.rates.fixed_per_vehicle * COST_SCALE_MILLI_INR))


def _add_capacity(
    routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager, task: ReferenceTask
) -> None:
    """Make capacity a hard dimension, never a penalty — §1.1, and what the GA's split DAG does."""

    def demand(from_index: int) -> int:
        return task.demand_g[int(manager.IndexToNode(from_index))]

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand),
        0,
        [task.capacity_g] * task.n_vehicles,
        True,
        _CAPACITY_DIMENSION,
    )


def _add_time_windows(
    routing: pywrapcp.RoutingModel, manager: pywrapcp.RoutingIndexManager, task: ReferenceTask
) -> None:
    """Add the arrival timeline and charge lateness softly, matching the GA's cost model.

    Three choices make this the same question the GA answers:

    * **Service rides on the outgoing arc, and is zero leaving the depot.** That makes the cumul at
      a node the *arrival* there, which is what
      :meth:`~src.stage2.pricing.TourPricer._advance` accumulates: dispatch, then travel to the
      first stop, then service-plus-travel for each leg after it.
    * **Slack is zero**, so a vehicle cannot wait to dodge a window. The GA models no waiting
      either — early arrival is simply not lateness.
    * **Windows are soft**, via a cumul soft upper bound, never a hard range. §1.1 makes time
      windows penalised rather than forbidden, and a hard range would also make a tight instance
      infeasible instead of expensive.

    The horizon must be a **provable** non-binding upper bound on any arrival, and it is built by
    bounding each arc separately rather than by bounding the total. That distinction is not
    cosmetic: ``elapsed`` rounds every arc individually, and a sum of individually-rounded arcs can
    exceed the rounding of their sum. Hub 1 of seed 11 is the worked case — two arcs of 6,985.66 s,
    one carrying service, give ``round(6985.66) + round(6985.66 + 300) = 14,272`` against
    ``round(6985.66 x 2 + 300) = 14,271``. One second, and a one-stop hub that any vehicle could
    serve came back ``ROUTING_INFEASIBLE``.

    So the bound is ``(k + 1)`` arcs — the most a tour over ``k`` stops can have — each at
    ``ceil(longest leg) + ceil(service)``, which dominates ``round(leg + service)`` for every arc
    termwise. A loose horizon costs nothing, since it is only a variable's upper bound; a tight one
    silently converts a soft window into a hard deadline, or the whole hub into an infeasible
    model. This is the one way this model could flatter the reference while looking correct.
    """
    travel_s = task.local_duration_s * task.rates.traffic_multiplier
    service_s = float(task.service_s)
    dispatch_s = round(float(task.dispatch_s))

    def elapsed(from_index: int, to_index: int) -> int:
        from_node = int(manager.IndexToNode(from_index))
        leg_s = float(travel_s[from_node, int(manager.IndexToNode(to_index))])
        return round(leg_s + (0.0 if from_node == DEPOT else service_s))

    n_stops = len(task.demand_g) - 1
    worst_arc_s = math.ceil(float(travel_s.max())) + math.ceil(service_s)
    horizon = dispatch_s + worst_arc_s * (n_stops + 1)
    routing.AddDimension(
        routing.RegisterTransitCallback(elapsed), 0, horizon, False, _TIME_DIMENSION
    )
    time_dimension = routing.GetDimensionOrDie(_TIME_DIMENSION)

    for vehicle in range(task.n_vehicles):
        time_dimension.CumulVar(routing.Start(vehicle)).SetRange(dispatch_s, dispatch_s)

    for position, window in enumerate(task.pricing.windows):
        if window is not None:
            time_dimension.SetCumulVarSoftUpperBound(
                manager.NodeToIndex(position + 1),
                round(float(window.end_s)),
                task.lateness_per_second,
            )
