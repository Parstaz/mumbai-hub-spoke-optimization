"""Run every hub's GA, in parallel, and assemble the final-mile plan.

Hubs are independent once the customer-to-hub mapping is fixed, so this is pure speedup rather than
a decomposition with a cost. What it is not is free of rules — §1.1 of CLAUDE.md constrains exactly
what may cross the process boundary, and this module is where that is enforced.

**A worker receives its own sliced matrices and nothing else.** :class:`Stage2Task` holds no
``Instance``, no ``Config`` and no ``Generator``. The matrices are this hub's own ``(k+1, k+1)``
block with the hub at local index 0, so a 73-stop hub ships a 74×74 array rather than a 20 MB pair;
the windows come pre-resolved as a :class:`~src.stage2.pricing.HubPricing`, which is why
:class:`~src.stage2.split.SplitContext` stopped carrying an instance at all. The two config objects
it does carry are frozen value types holding scalars, in the same spirit as Stage 1's ``ArcRates``.

**Randomness is rebuilt, never shipped.** Each task carries two integers and the worker builds
``np.random.default_rng([seed, hub_id])`` from them. Passing a live ``Generator`` would share state
between hubs and destroy reproducibility silently — the run still succeeds, its numbers just stop
repeating. That is the failure mode CLAUDE.md's gotchas name, and the reason ``hub_id`` is in the
key as well as the payload: two hubs of identical shape must still search differently.

**Workers return chromosomes, not routes.** A worker's node ids are local to its slice, so a
``Route`` built there would carry indices meaningless to the scorer. The parent rebuilds each plan
over the instance-wide matrices, which keeps :func:`~src.scoring.evaluate_solution` looking at
global node ids and keeps route building on the one shared path, at a cost of one split per hub.

The pool is **spawn**, matching Stage 1: forking a threaded process is undefined, and the platform
default is not something a result should depend on.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from dataclasses import dataclass

import numpy as np

from src.config import SECONDS_PER_HOUR, Config, CostConfig, GAConfig
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import Instance
from src.exceptions import InfeasibleSolutionError
from src.solution import Route
from src.stage2.ga import HubOutcome, evolve
from src.stage2.pricing import HubPricing, hub_pricing
from src.stage2.split import SplitContext, split
from src.tour import RoutingContext
from src.units import DemandArray, DistanceMatrix, DurationMatrix, NodeArray, NodeId, Seconds
from src.workload import HubWorkload, group_by_hub, node_array, require_servable, stage_demands

logger = logging.getLogger(__name__)

_LOCAL_HUB_NODE = NodeId(0)
"""Every task's slice puts its own hub first, so a worker's hub is always node 0."""


@dataclass(frozen=True, slots=True)
class Stage2Plan:
    """The final-mile tours, and what each hub's search actually did to produce them.

    ``outcomes`` is returned rather than logged and dropped because the configured generation
    budget is not the budget spent — every hub on seed 42 stopped on ``stagnation_limit`` well
    short of it. A caller that reports "150x600" without saying what was used is describing a run
    that did not happen, so the entry point needs the figures, not just the routes.
    """

    routes: tuple[Route, ...]
    outcomes: tuple[HubOutcome, ...]


@dataclass(frozen=True, slots=True)
class Stage2Task:
    """One hub's complete, self-contained final-mile problem — all a worker may know."""

    hub_id: int
    local_distance_m: DistanceMatrix
    local_duration_s: DurationMatrix
    demand_kg: DemandArray
    pricing: HubPricing
    traffic: TrafficModel
    dispatch_s: Seconds
    service_s: Seconds
    capacity_kg: float
    cost: CostConfig
    ga: GAConfig
    seed: int


def solve_stage2(
    instance: Instance,
    matrices: CostMatrices,
    traffic: TrafficModel,
    hub_of_stop: NodeArray,
    config: Config,
) -> Stage2Plan:
    """Build the final-mile plan: one GA per hub, then the tours its answer implies.

    Args:
        instance: The instance being solved; supplies windows and the customer node space.
        matrices: Instance-wide distance and duration matrices.
        traffic: The multiplier schedule.
        hub_of_stop: Hub id per customer — the one Stage 1's assignment induces through each
            shipment, from :func:`~src.workload.hub_of_customer`. Never nearest-hub over customers:
            a parcel leaves from the hub it actually reached.
        config: The full run configuration. Sliced down before anything crosses to a worker.

    Returns:
        Every deployed tour across all hubs in hub order, with each hub's search outcome.
    """
    workloads = _workloads(instance, hub_of_stop, config)
    tasks = tuple(
        _hub_task(workload, instance, matrices, traffic, config) for workload in workloads
    )
    outcomes = _solve_all(tasks, config.stage1.workers)

    routing = RoutingContext(
        matrices=matrices,
        traffic=traffic,
        start_time_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
    )
    routes: list[Route] = []
    for workload, outcome in zip(workloads, outcomes, strict=True):
        context = SplitContext(
            workload=workload,
            routing=routing,
            pricing=hub_pricing(workload, instance),
            cost_config=config.cost,
        )
        if outcome.hub_id != workload.hub_id:
            raise InfeasibleSolutionError(
                f"hub {workload.hub_id}'s workload was paired with hub {outcome.hub_id}'s plan"
            )
        routes.extend(split(outcome.permutation, context).routes)
    return Stage2Plan(routes=tuple(routes), outcomes=outcomes)


def _log_finished(outcome: HubOutcome, done: int, total: int) -> None:
    """Report one hub the moment it lands, so a long solve is not silent while it runs."""
    logger.info(
        "hub %2d done (%2d/%d): %3d stops, %d generations, penalty x%.2f, %.0f INR",
        outcome.hub_id,
        done,
        total,
        len(outcome.permutation),
        outcome.generations_run,
        outcome.final_multiplier,
        outcome.objective_inr,
    )


def hub_of_source(instance: Instance, inbound_routes: tuple[Route, ...]) -> NodeArray:
    """Read Stage 1's source-to-hub assignment back off the tours it produced.

    Stage 1 returns routes and discards the assignment that produced them, so Stage 2 recovers it
    from where each source was actually collected. Reading the plan rather than re-deriving the
    assignment is the honest direction: a parcel leaves from the hub a vehicle really took it to,
    not from the hub a second call to the strategy would have chosen.

    Sources that no shipment originates at appear in no tour and keep the fill value. Nothing reads
    them — :func:`~src.workload.hub_of_customer` only looks up sources that a shipment names — and
    a source with nothing waiting is correctly skipped rather than visited.

    Args:
        instance: Supplies the node layout the source ids are recovered through.
        inbound_routes: Stage 1's tours, from :func:`~src.stage1.cvrp.solve_stage1` or the
            baseline's inbound leg.

    Returns:
        A hub id per source id, dense over every source.
    """
    hubs = np.full(len(instance.sources), -1, dtype=np.intp)
    for route in inbound_routes:
        for node in route.interior_nodes:
            hubs[int(node) - len(instance.hubs)] = route.hub_id
    return hubs


def _workloads(
    instance: Instance, hub_of_stop: NodeArray, config: Config
) -> tuple[HubWorkload, ...]:
    """Group customers by the hub their shipment reached, and refuse an unliftable stop first."""
    _, customer_kg = stage_demands(instance)
    require_servable(customer_kg, config.fleet.vehicle_capacity_kg, "customer")
    hub_nodes = node_array(instance.hub_node(hub.hub_id) for hub in instance.hubs)
    customer_nodes = node_array(
        instance.customer_node(customer.customer_id) for customer in instance.customers
    )
    return group_by_hub(hub_nodes, hub_of_stop, customer_nodes, customer_kg)


def _hub_task(
    workload: HubWorkload,
    instance: Instance,
    matrices: CostMatrices,
    traffic: TrafficModel,
    config: Config,
) -> Stage2Task:
    """Slice one hub's problem out of the instance-wide matrices.

    Slicing rather than passing the whole matrix is what keeps the pool payload proportional to a
    hub's own workload, and is also what makes the payload legal: the slice carries no information
    about any other hub.
    """
    local = np.concatenate(([workload.hub_node], workload.nodes))
    block = np.ix_(local, local)
    return Stage2Task(
        hub_id=workload.hub_id,
        local_distance_m=matrices.distance_m[block],
        local_duration_s=matrices.duration_s[block],
        demand_kg=workload.demand_kg,
        pricing=hub_pricing(workload, instance),
        traffic=traffic,
        dispatch_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
        cost=config.cost,
        ga=config.ga,
        seed=config.run.seed,
    )


def _solve_all(tasks: tuple[Stage2Task, ...], workers: int) -> tuple[HubOutcome, ...]:
    """Evolve every hub, sequentially or across a spawn pool.

    A single hub, or a pool of one, takes the sequential path — the only one whose result a test
    can compare against a pool's, and the one that keeps a small run free of interpreter start-up.
    Both paths return results in hub order and both log each hub as it lands; the per-hub seed
    makes the answer independent of which worker picked up which hub, and therefore of the order
    they come back in.
    """
    count = _worker_count(workers)
    if len(tasks) <= 1 or count == 1:
        finished = []
        for task in tasks:
            finished.append(solve_hub_ga(task))
            _log_finished(finished[-1], len(finished), len(tasks))
        return tuple(finished)

    with multiprocessing.get_context("spawn").Pool(processes=count) as pool:
        finished = []
        # imap_unordered rather than map: a result is yielded the moment its hub finishes, so a
        # long run reports progress instead of going silent until the slowest hub returns. Order
        # is restored by hub id afterwards, which is why HubOutcome carries one.
        for outcome in pool.imap_unordered(solve_hub_ga, tasks):
            finished.append(outcome)
            _log_finished(outcome, len(finished), len(tasks))
    return tuple(sorted(finished, key=lambda outcome: outcome.hub_id))


def _worker_count(workers: int) -> int:
    """Resolve the configured pool size, zero meaning one worker per CPU.

    Resolved here rather than as a config default so ``os.cpu_count()`` is never read at import
    time — a config that changes with the machine it was imported on is not a config.
    """
    return workers if workers > 0 else (os.cpu_count() or 1)


def solve_hub_ga(task: Stage2Task) -> HubOutcome:
    """Evolve one hub. The pool worker — module-level and pure, so it pickles.

    Rebuilds the hub's context against its *local* node space, where the hub is node 0 and its
    stops are 1..k in the order the parent sliced them. A returned chromosome is therefore
    positions into that same order, which is what lets the parent replay it against the
    instance-wide matrices without a translation table.

    Args:
        task: This hub's self-contained problem.

    Returns:
        The best chromosome found, priced at the configured rates.
    """
    n_stops = len(task.demand_kg)
    workload = HubWorkload(
        hub_id=task.hub_id,
        hub_node=_LOCAL_HUB_NODE,
        nodes=np.arange(1, n_stops + 1, dtype=np.intp),
        demand_kg=task.demand_kg,
    )
    context = SplitContext(
        workload=workload,
        routing=RoutingContext(
            matrices=CostMatrices(
                distance_m=task.local_distance_m, duration_s=task.local_duration_s
            ),
            traffic=task.traffic,
            start_time_s=task.dispatch_s,
            service_time_s=task.service_s,
            capacity_kg=task.capacity_kg,
        ),
        pricing=task.pricing,
        cost_config=task.cost,
    )
    return evolve(context, task.ga, np.random.default_rng([task.seed, task.hub_id]))
