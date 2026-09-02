"""Per-hub inbound CVRP: one independent OR-Tools model per hub, solved in parallel.

Hubs do not interact once the assignment is fixed — a vehicle leaving hub 3 never touches hub 7's
sources — so this is 16 small problems rather than one large one, and they run in a process pool.
Solving them separately is not only faster: a 285-stop single model would spend its time limit on
a search space that is mostly infeasible by construction.

**The worker is pure and sees almost nothing.** It receives one :class:`HubTask` holding that hub's
own ``(k+1, k+1)`` slice of the matrices and returns visit orders as local indices. It never sees
the full 1116×1116 matrices, the ``Config``, the ``Instance`` or a random ``Generator``. That is
what makes the pool correct rather than merely fast: there is no shared mutable state to be raced
over, and reproducibility does not depend on how the work was divided.

**OR-Tools optimises a static proxy; the reported cost is the real thing.** Arc costs in a
``RoutingModel`` are fixed before the search starts, so the cumulative traffic model — which
depends on when a leg is actually driven — cannot live inside it. The proxy is the monetised arc:

    ``alpha × km  +  beta × hours × m(dispatch_hour)``

with the fixed vehicle charge ``gamma`` attached to each vehicle, so the search minimises the same
four-component objective :mod:`src.scoring` reports rather than a distance or duration surrogate.
Note that duration *alone* would make the traffic multiplier a no-op: a single scalar multiple of
the duration matrix has exactly the same ``argmin``. The multiplier only changes a decision when
it is traded against ``₹/km``, which is why the arc cost is money and not time.

Once an ordering comes back, the ``Route`` is built by :func:`src.tour.build_route` from the
**global** matrices and the real :class:`~src.costs.traffic.TrafficModel`, so every distance,
duration and arrival time this module reports is the cumulative band-blended figure. The static
proxy influences which ordering is chosen and nothing else.

**Reproducibility caveat.** Guided local search under a wall-clock time limit returns whatever it
had reached when the clock ran out, so the same instance on a busier machine yields a different
plan. Set :attr:`~src.config.Stage1Config.cvrp_solution_limit` to 1 to stop at the first-solution
heuristic, which is deterministic and independent of the time limit; the test suite runs that way.
A real run leaves it unlimited and accepts the variation, which is why multi-seed evaluation
reports a spread rather than a single number.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
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
    Config,
    FleetConfig,
    Stage1Config,
)
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import Instance
from src.exceptions import InfeasibleInstanceError
from src.solution import Route
from src.stage1.assignment import AssignmentStrategy, max_stops_per_hub
from src.tour import RoutingContext, build_route
from src.units import DistanceMatrix, DurationMatrix, Seconds
from src.workload import (
    HubWorkload,
    group_by_hub,
    node_array,
    require_servable,
    stage_demands,
)

logger = logging.getLogger(__name__)

VisitOrders = tuple[tuple[int, ...], ...]
"""One tuple of stop positions per deployed vehicle, indexing into ``HubWorkload.nodes``."""

_DEPOT = 0
"""Local node index of the hub in a per-hub model. Stops occupy 1..k."""

_CAPACITY_DIMENSION = "Capacity"

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


@dataclass(frozen=True, slots=True)
class HubTask:
    """One hub's complete, self-contained CVRP. Everything a pool worker is allowed to know.

    Deliberately holds no ``Instance``, no ``Config`` and no ``Generator``: a worker that could
    reach the whole problem could also be handed a mutable view of it, and per-hub solves would
    stop being reproducible. The matrices here are this hub's own ``(k+1, k+1)`` slice with the
    hub at index 0.
    """

    hub_id: int
    local_distance_m: DistanceMatrix
    local_duration_s: DurationMatrix
    demand_g: tuple[int, ...]
    capacity_g: int
    n_vehicles: int
    rates: ArcRates
    time_limit_s: float
    solution_limit: int


def solve_stage1(
    instance: Instance,
    matrices: CostMatrices,
    traffic: TrafficModel,
    strategy: AssignmentStrategy,
    config: Config,
) -> tuple[Route, ...]:
    """Build the inbound plan: assign sources to hubs, then solve each hub's CVRP.

    Args:
        instance: The problem to solve; supplies the fleet, the operating day and the shipments
            the demand per source is read off.
        matrices: Road distances and free-flow durations, from
            :func:`~src.costs.matrix.build_matrices`.
        traffic: The multiplier schedule, integrated along each returned tour.
        strategy: Which hub collects from each source — :func:`~src.stage1.assignment.unconstrained`
            or :func:`~src.stage1.assignment.capacity_balanced`. The ablation's swap point.
        config: Rates, the balance slack, and the search and pool settings.

    Returns:
        The inbound tours, ready to place in a :class:`~src.solution.Solution` and score with
        :func:`~src.scoring.evaluate_solution`. Nothing here prices anything.

    Raises:
        InfeasibleInstanceError: If a source holds more than one vehicle can carry, if the balance
            cap cannot absorb every source, or if a hub's model admits no solution.
    """
    workloads = _assign(instance, matrices, strategy, config.stage1)
    rates = _arc_rates(instance, traffic, config)
    tasks = tuple(
        _hub_task(workload, matrices, rates, instance.fleet, config.stage1)
        for workload in workloads
    )
    orders = _solve_all(tasks, config.stage1.workers)

    context = RoutingContext(
        matrices=matrices,
        traffic=traffic,
        start_time_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
    )
    routes = tuple(
        build_route(workload, positions, context)
        for workload, hub_orders in zip(workloads, orders, strict=True)
        for positions in hub_orders
    )
    logger.info(
        "stage 1: %d inbound tours over %d sources at %d hubs, %d vehicles offered",
        len(routes),
        sum(len(workload.nodes) for workload in workloads),
        len(workloads),
        sum(task.n_vehicles for task in tasks),
    )
    return routes


def _assign(
    instance: Instance,
    matrices: CostMatrices,
    strategy: AssignmentStrategy,
    stage1: Stage1Config,
) -> tuple[HubWorkload, ...]:
    """Decide which hub collects from each source, and group the sources under their hub.

    Only sources with something waiting are offered to the strategy. Including the idle ones
    would inflate the balance cap's denominator and let a hub be capped on stops it would never
    have visited.
    """
    source_demand_kg, _ = stage_demands(instance)
    require_servable(source_demand_kg, instance.fleet.vehicle_capacity_kg, "source")

    hub_nodes = node_array(instance.hub_node(hub.hub_id) for hub in instance.hubs)
    active = np.flatnonzero(source_demand_kg > 0.0)
    stop_nodes = node_array(instance.source_node(int(index)) for index in active)
    cap = max_stops_per_hub(len(stop_nodes), len(hub_nodes), stage1.hub_balance_slack)

    hub_of_stop = strategy(hub_nodes, stop_nodes, matrices.distance_m, cap)
    return group_by_hub(hub_nodes, hub_of_stop, stop_nodes, source_demand_kg[active])


def _arc_rates(instance: Instance, traffic: TrafficModel, config: Config) -> ArcRates:
    """Bundle the cost rates with the traffic multiplier in force at dispatch."""
    return ArcRates(
        variable_per_km=config.cost.variable_per_km,
        driver_per_hour=config.cost.driver_per_hour,
        fixed_per_vehicle=config.cost.fixed_per_vehicle,
        traffic_multiplier=traffic.multiplier_for_hour(int(instance.schedule.dispatch_hour)),
    )


def _hub_task(
    workload: HubWorkload,
    matrices: CostMatrices,
    rates: ArcRates,
    fleet: FleetConfig,
    stage1: Stage1Config,
) -> HubTask:
    """Slice one hub's problem out of the instance-wide matrices.

    Slicing rather than passing the whole matrix is what keeps the pool payload proportional to a
    hub's own workload: a 20-stop hub ships a 21×21 block instead of a 20 MB pair of matrices.
    """
    local = np.concatenate(([workload.hub_node], workload.nodes))
    block = np.ix_(local, local)
    return HubTask(
        hub_id=workload.hub_id,
        local_distance_m=matrices.distance_m[block],
        local_duration_s=matrices.duration_s[block],
        demand_g=(0, *(round(kg * GRAMS_PER_KG) for kg in workload.demand_kg)),
        capacity_g=round(fleet.vehicle_capacity_kg * GRAMS_PER_KG),
        n_vehicles=_vehicle_count(workload, fleet),
        rates=rates,
        time_limit_s=stage1.cvrp_time_limit_s,
        solution_limit=stage1.cvrp_solution_limit,
    )


def _vehicle_count(workload: HubWorkload, fleet: FleetConfig) -> int:
    """Vehicles to offer this hub: its mass floor, times the configured slack, rounded up.

    The floor is subtracted by the capacity tolerance before dividing, because a load summed in
    floating point can land a fraction of a microgram over an exact multiple of capacity and
    would otherwise buy a whole extra vehicle for no mass at all.

    Offering more vehicles than needed is safe and deliberate: unused ones stay at the depot and
    are dropped, and the fixed charge attached to each vehicle means the search prefers not to
    deploy them. Offering too few would make a solvable hub infeasible.
    """
    total_kg = float(workload.demand_kg.sum())
    floor = math.ceil((total_kg - CAPACITY_TOLERANCE_KG) / fleet.vehicle_capacity_kg)
    return max(1, math.ceil(floor * fleet.vehicle_slack_factor))


def _solve_all(tasks: tuple[HubTask, ...], workers: int) -> tuple[VisitOrders, ...]:
    """Solve every hub, sequentially or across a process pool.

    A single hub, or a pool of one, takes the sequential path — the only one whose result a test
    can compare against a pool's, and the one that keeps a small run free of interpreter
    start-up. ``pool.map`` preserves input order, so the two paths return identically ordered
    results.

    The pool is explicitly a **spawn** pool. ``RoutingModel`` starts threads, and forking a
    process that holds threads is undefined; spawn also makes the platform default irrelevant, so
    a Linux CI run and a macOS laptop exercise the same code path.
    """
    count = _worker_count(workers)
    if len(tasks) <= 1 or count == 1:
        return tuple(solve_hub_cvrp(task) for task in tasks)
    with multiprocessing.get_context("spawn").Pool(processes=count) as pool:
        return tuple(pool.map(solve_hub_cvrp, tasks))


def _worker_count(workers: int) -> int:
    """Resolve the configured pool size, zero meaning one worker per CPU.

    Resolved here rather than as a config default so that ``os.cpu_count()`` is never read at
    import time — a config that changes with the machine it was imported on is not a config.
    """
    return workers if workers > 0 else (os.cpu_count() or 1)


def solve_hub_cvrp(task: HubTask) -> VisitOrders:
    """Solve one hub's CVRP. The pool worker — module-level and pure, so it pickles.

    Capacity is a hard OR-Tools dimension, never a penalty: an arc that would overfill a vehicle
    is not available to the search rather than expensive within it.

    Args:
        task: This hub's self-contained problem.

    Returns:
        One tuple of stop positions per vehicle that was actually deployed, in visit order.
        Vehicles left at the depot are omitted, so the caller never builds an empty tour.

    Raises:
        InfeasibleInstanceError: If the model admits no solution. With the vehicle count derived
            from the hub's own mass floor and every stop known to fit one vehicle, this means a
            bug in the model rather than a hard instance.
    """
    manager = pywrapcp.RoutingIndexManager(len(task.demand_g), task.n_vehicles, _DEPOT)
    routing = pywrapcp.RoutingModel(manager)
    arc_cost = _arc_cost_milli_inr(task)

    def transit(from_index: int, to_index: int) -> int:
        return int(arc_cost[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    def demand(from_index: int) -> int:
        return task.demand_g[int(manager.IndexToNode(from_index))]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(transit))
    routing.SetFixedCostOfAllVehicles(round(task.rates.fixed_per_vehicle * COST_SCALE_MILLI_INR))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand),
        0,
        [task.capacity_g] * task.n_vehicles,
        True,
        _CAPACITY_DIMENSION,
    )

    assignment = routing.SolveWithParameters(_search_parameters(task))
    if assignment is None:
        raise InfeasibleInstanceError(
            f"OR-Tools found no inbound plan for hub {task.hub_id}: "
            f"{len(task.demand_g) - 1} stops, {task.n_vehicles} vehicles of {task.capacity_g} g"
        )
    return _visit_orders(routing, manager, assignment, task.n_vehicles)


def _arc_cost_milli_inr(task: HubTask) -> npt.NDArray[np.int64]:
    """Price every arc of one hub's block in integer milli-rupees.

    Vectorised over the whole block rather than evaluated inside the transit callback: OR-Tools
    calls that callback tens of thousands of times per solve, and the arithmetic is identical for
    every call.
    """
    rates = task.rates
    inr = (
        rates.variable_per_km * task.local_distance_m / METRES_PER_KM
        + rates.driver_per_hour
        * rates.traffic_multiplier
        * task.local_duration_s
        / SECONDS_PER_HOUR
    )
    scaled: npt.NDArray[np.int64] = np.rint(inr * COST_SCALE_MILLI_INR).astype(np.int64)
    return scaled


def _search_parameters(task: HubTask) -> pywrapcp.DefaultRoutingSearchParameters:
    """First solution by cheapest arc, then guided local search under the configured limits.

    ``solution_limit`` of 1 returns the first-solution result and is independent of the time
    limit, which is what makes a seeded test reproducible. Zero means unlimited and is left
    unset, because OR-Tools treats the field's own default as no limit.
    """
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(int(task.time_limit_s * _MILLISECONDS_PER_SECOND))
    if task.solution_limit > 0:
        parameters.solution_limit = task.solution_limit
    parameters.log_search = False
    return parameters


def _visit_orders(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    assignment: pywrapcp.Assignment,
    n_vehicles: int,
) -> VisitOrders:
    """Read each vehicle's tour out of the solved model as stop positions.

    Walked from the node *after* the start depot, so the returned positions index
    ``HubWorkload.nodes`` directly and a vehicle that never left reads as an empty walk. Local
    node ``i`` is stop ``i - 1``, since the hub occupies local index 0.
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
