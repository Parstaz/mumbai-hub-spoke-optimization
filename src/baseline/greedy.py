"""The greedy nearest-neighbour benchmark: the plan the optimized pipeline has to beat.

This module is deliberately unsophisticated, and every omission below is a decision rather than
an oversight:

* **Nearest hub, not best hub.** Each source goes to the hub closest to it by road. No
  consolidation view, no min-cost flow, no awareness that a hub is already oversubscribed.
* **Nearest neighbour, no improvement.** Tours are grown one nearest stop at a time and are never
  revisited. There is no 2-opt, no or-opt, no reinsertion.
* **No time-window awareness whatsoever.** Windows are neither sorted on, deferred for, nor
  penalised during construction. The resulting lateness is *recorded* — the arrival times this
  module produces are what :func:`~src.scoring.evaluate_solution` measures windows against — and
  a large violation count in the baseline column is the honest reading of a solver that ignores
  them.

What it is *not* allowed to be is a rigged straw man. It reads the same distance matrix and the
same :class:`~src.costs.traffic.TrafficModel` as the optimized pipeline, dispatches at the same
hour, and is scored by the same :func:`~src.scoring.evaluate_solution`. There is no scoring
arithmetic in this file at all: a benchmark that computed its own cost could be beaten by a
rounding difference rather than by better routing.

Capacity stays hard here as it does everywhere else — a vehicle is closed when nothing left to
serve still fits, never over-filled and penalised afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.config import CAPACITY_TOLERANCE_KG, SECONDS_PER_HOUR
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel, route_timeline
from src.data.instance import Instance
from src.exceptions import InfeasibleInstanceError
from src.solution import Route, Solution
from src.units import DistanceMatrix, Metres, NodeId, Seconds

logger = logging.getLogger(__name__)

NodeArray = npt.NDArray[np.intp]
"""Node ids as a NumPy index array, for indexing a matrix row in one operation."""

DemandArray = npt.NDArray[np.float64]
"""Mass in kilograms, aligned element-wise with a :data:`NodeArray`."""


@dataclass(frozen=True, slots=True)
class _HubWorkload:
    """The stops one hub must serve in one stage, with the mass waiting at each.

    Both stages reduce to this shape — a hub, a set of stops, a demand per stop — which is what
    lets one tour builder serve inbound collection and final-mile delivery without knowing which
    it is looking at.
    """

    hub_id: int
    hub_node: NodeId
    nodes: NodeArray
    demand_kg: DemandArray


@dataclass(frozen=True, slots=True)
class _RoutingContext:
    """The instance-wide inputs every tour needs, bundled so the builders stay small.

    Carrying the matrices and the traffic model through here rather than rebuilding either per
    hub is what guarantees the benchmark is measured over exactly the road network and the
    time-of-day multipliers the optimized pipeline sees.
    """

    matrices: CostMatrices
    traffic: TrafficModel
    start_time_s: Seconds
    service_time_s: Seconds
    capacity_kg: float


def solve_baseline(instance: Instance, matrices: CostMatrices, traffic: TrafficModel) -> Solution:
    """Build the greedy benchmark plan for ``instance``.

    Deterministic and free of randomness: every choice is an ``argmin`` over road distance, and
    ties fall to the lower node id. No seed is needed and none is accepted, so two runs on one
    instance are byte-identical plans.

    Both stages dispatch at :attr:`~src.config.ScheduleConfig.dispatch_hour`. The vehicles are
    treated as a parallel fleet leaving together rather than as a sequence of shifts, so every
    tour in the plan is exposed to the same morning traffic band — which is the exposure the
    optimized pipeline has to do something about.

    Args:
        instance: The problem to solve; also supplies the fleet, the operating day and the
            shipment-to-source-to-customer mapping the hub assignment follows.
        matrices: Road distances and free-flow durations, from
            :func:`~src.costs.matrix.build_matrices`.
        traffic: The time-of-day multiplier schedule to integrate along each tour.

    Returns:
        A complete :class:`~src.solution.Solution`: inbound tours collecting from every source
        that has something waiting, and final-mile tours delivering to every customer. Score it
        with :func:`~src.scoring.evaluate_solution`; nothing here prices anything.

    Raises:
        InfeasibleInstanceError: If a single source holds more mass than one vehicle can carry.
    """
    source_demand_kg, customer_demand_kg = _stage_demands(instance)
    _require_servable(source_demand_kg, instance.fleet.vehicle_capacity_kg)

    hub_nodes = _node_array(instance.hub_node(hub.hub_id) for hub in instance.hubs)
    source_nodes = _node_array(instance.source_node(src.source_id) for src in instance.sources)
    customer_nodes = _node_array(instance.customer_node(c.customer_id) for c in instance.customers)

    hub_of_source = _nearest_hub(hub_nodes, source_nodes, matrices.distance_m)
    hub_of_customer = _hub_holding_each_shipment(instance, hub_of_source)

    context = _RoutingContext(
        matrices=matrices,
        traffic=traffic,
        start_time_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
    )
    solution = Solution(
        stage1_routes=_stage_routes(
            _workloads(hub_nodes, hub_of_source, source_nodes, source_demand_kg), context
        ),
        stage2_routes=_stage_routes(
            _workloads(hub_nodes, hub_of_customer, customer_nodes, customer_demand_kg), context
        ),
    )
    logger.info(
        "greedy baseline: %d inbound tours over %d sources, %d final-mile tours over %d customers",
        len(solution.stage1_routes),
        int(np.count_nonzero(source_demand_kg)),
        len(solution.stage2_routes),
        len(instance.customers),
    )
    return solution


def _stage_demands(instance: Instance) -> tuple[DemandArray, DemandArray]:
    """Mass waiting at each source, and mass owed to each customer.

    Read off the shipments rather than the node lists because the two differ: a source may hold
    several shipments or none at all, while every customer is the destination of exactly one.
    """
    sizes = [shipment.size_kg for shipment in instance.shipments]
    return (
        _demand_kg(len(instance.sources), [s.source_id for s in instance.shipments], sizes),
        _demand_kg(len(instance.customers), [s.customer_id for s in instance.shipments], sizes),
    )


def _demand_kg(n_ids: int, ids: Sequence[int], sizes: Sequence[float]) -> DemandArray:
    """Scatter-add shipment masses into a dense vector indexed by ``ids``.

    Dense rather than a mapping so the tour builder can mask it against a candidate array in one
    vectorised comparison instead of a dictionary lookup per stop.
    """
    totals = np.zeros(n_ids, dtype=np.float64)
    np.add.at(totals, ids, sizes)
    return totals


def _require_servable(source_demand_kg: DemandArray, capacity_kg: float) -> None:
    """Reject an instance whose sources cannot be emptied one vehicle-load at a time.

    Only sources are checked. A customer's demand is a single shipment, and the instance already
    guarantees at construction both that every shipment fits a vehicle and that each customer is
    the destination of exactly one — so the final-mile equivalent of this check can never fire.
    """
    over = np.flatnonzero(source_demand_kg > capacity_kg + CAPACITY_TOLERANCE_KG)
    if over.size:
        raise InfeasibleInstanceError(
            f"{over.size} source(s) hold more than one vehicle can carry "
            f"(worst {source_demand_kg[over].max():.1f} kg against {capacity_kg:.1f} kg capacity); "
            f"the greedy baseline visits each stop once and never splits it across vehicles"
        )


def _node_array(nodes: Iterable[NodeId]) -> NodeArray:
    """Collect node ids into an index array.

    Built from the :class:`~src.data.instance.Instance` accessors rather than by arithmetic over
    the flat layout, so an off-by-one in the node space raises instead of quietly addressing a
    customer as a source.
    """
    return np.fromiter(nodes, dtype=np.intp)


def _nearest_hub(
    hub_nodes: NodeArray, stop_nodes: NodeArray, distance_m: DistanceMatrix
) -> NodeArray:
    """Hub id serving each stop: whichever hub is the shortest drive out to it.

    Measured hub-to-stop, the direction the vehicle leaves in. On an asymmetric road network the
    return leg can differ, and a benchmark that averaged the two would be doing a small
    optimization — which is precisely what this module must not do.
    """
    block = distance_m[np.ix_(hub_nodes, stop_nodes)]
    return np.asarray(np.argmin(block, axis=0), dtype=np.intp)


def _hub_holding_each_shipment(instance: Instance, hub_of_source: NodeArray) -> NodeArray:
    """Hub id serving each customer: the one its own shipment was consolidated at.

    Not the customer's nearest hub. Stage 1 has already carried the parcel somewhere, and the
    parcel has to leave from where it landed — that dependence is exactly the greedy decomposition
    the README lists as a known limitation, and the baseline has to display it rather than dodge
    it.
    """
    hubs = np.empty(len(instance.customers), dtype=np.intp)
    for shipment in instance.shipments:
        hubs[shipment.customer_id] = hub_of_source[shipment.source_id]
    return hubs


def _workloads(
    hub_nodes: NodeArray, hub_of_stop: NodeArray, stop_nodes: NodeArray, demand_kg: DemandArray
) -> tuple[_HubWorkload, ...]:
    """Group stops by their assigned hub, dropping anything with no work in it.

    A source that no shipment originates at is skipped rather than visited: there is nothing to
    collect there, so a tour through it would burn distance and a fixed vehicle charge to move
    zero kilograms and would make the benchmark artificially bad.
    """
    has_demand = demand_kg > 0.0
    workloads: list[_HubWorkload] = []
    for hub_id, hub_node in enumerate(hub_nodes):
        assigned = has_demand & (hub_of_stop == hub_id)
        if assigned.any():
            workloads.append(
                _HubWorkload(
                    hub_id=hub_id,
                    hub_node=NodeId(int(hub_node)),
                    nodes=stop_nodes[assigned],
                    demand_kg=demand_kg[assigned],
                )
            )
    return tuple(workloads)


def _nearest_neighbour_tours(
    workload: _HubWorkload, context: _RoutingContext
) -> tuple[tuple[int, ...], ...]:
    """Fill vehicles one at a time, each hopping to its nearest stop until nothing more fits.

    The candidate set is restricted to stops that still fit before the ``argmin`` runs, rather
    than closing the vehicle the moment the nearest stop is too heavy. With uniform shipment
    sizes the two rules are identical; where they differ, the second one abandons half-empty
    vehicles for reasons no dispatcher would accept, and an unfairly bad benchmark makes the
    reported improvement meaningless.

    Args:
        workload: One hub's stops and their demands.
        context: Supplies the distance matrix the hops are chosen on, and vehicle capacity.

    Returns:
        One tuple of positions into ``workload.nodes`` per vehicle, in visit order.
    """
    unvisited = np.ones(len(workload.nodes), dtype=np.bool_)
    tours: list[tuple[int, ...]] = []
    while unvisited.any():
        order: list[int] = []
        load_kg = 0.0
        current = workload.hub_node
        while True:
            room_kg = context.capacity_kg - load_kg + CAPACITY_TOLERANCE_KG
            eligible = unvisited & (workload.demand_kg <= room_kg)
            if not eligible.any():
                break
            legs_m = context.matrices.distance_m[current, workload.nodes]
            chosen = int(np.argmin(np.where(eligible, legs_m, np.inf)))
            order.append(chosen)
            unvisited[chosen] = False
            load_kg += float(workload.demand_kg[chosen])
            current = NodeId(int(workload.nodes[chosen]))
        tours.append(tuple(order))
    return tuple(tours)


def _stage_routes(
    workloads: tuple[_HubWorkload, ...], context: _RoutingContext
) -> tuple[Route, ...]:
    """Build every tour of one stage, hub by hub."""
    return tuple(
        _route(workload, positions, context)
        for workload in workloads
        for positions in _nearest_neighbour_tours(workload, context)
    )


def _route(workload: _HubWorkload, positions: tuple[int, ...], context: _RoutingContext) -> Route:
    """Turn one vehicle's visit order into a :class:`~src.solution.Route`.

    Arrival times come from :func:`~src.costs.traffic.route_timeline`, so the traffic multiplier
    accumulates along the tour exactly as it does for the optimized pipeline: a delayed stop
    pushes every later stop into whatever band it now lands in. The route's duration is read back
    off that timeline rather than summed independently, which keeps this module free of any
    travel-time arithmetic of its own.
    """
    stops = tuple(NodeId(int(workload.nodes[position])) for position in positions)
    nodes = (workload.hub_node, *stops, workload.hub_node)
    path = np.fromiter(nodes, dtype=np.intp, count=len(nodes))
    arrival_s = route_timeline(
        nodes,
        context.matrices.duration_s,
        context.start_time_s,
        context.traffic,
        context.service_time_s,
    )
    return Route(
        hub_id=workload.hub_id,
        nodes=nodes,
        load_kg=float(workload.demand_kg[list(positions)].sum()),
        distance_m=Metres(float(context.matrices.distance_m[path[:-1], path[1:]].sum())),
        duration_s=Seconds(arrival_s[-1] - context.start_time_s),
        arrival_s=arrival_s,
    )
