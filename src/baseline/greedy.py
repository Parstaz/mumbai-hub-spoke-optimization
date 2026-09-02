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

This module is the **control**, so its dependencies run one way only: it imports the shared
:mod:`src.workload` and :mod:`src.tour`, and never anything under ``src/stage1`` or
``src/stage2``. A control that imported the treatment would move whenever the treatment was
tuned, and the comparison would stop being one.
"""

from __future__ import annotations

import logging

import numpy as np

from src.config import CAPACITY_TOLERANCE_KG, SECONDS_PER_HOUR
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import Instance
from src.solution import Route, Solution
from src.tour import RoutingContext, build_route
from src.units import NodeArray, NodeId, Seconds
from src.workload import (
    HubWorkload,
    group_by_hub,
    nearest_hub,
    node_array,
    require_servable,
    stage_demands,
)

logger = logging.getLogger(__name__)


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
    source_demand_kg, customer_demand_kg = stage_demands(instance)
    require_servable(source_demand_kg, instance.fleet.vehicle_capacity_kg, "source")

    hub_nodes = node_array(instance.hub_node(hub.hub_id) for hub in instance.hubs)
    source_nodes = node_array(instance.source_node(src.source_id) for src in instance.sources)
    customer_nodes = node_array(instance.customer_node(c.customer_id) for c in instance.customers)

    hub_of_source = nearest_hub(hub_nodes, source_nodes, matrices.distance_m)
    hub_of_customer = _hub_holding_each_shipment(instance, hub_of_source)

    context = RoutingContext(
        matrices=matrices,
        traffic=traffic,
        start_time_s=Seconds(instance.schedule.dispatch_hour * SECONDS_PER_HOUR),
        service_time_s=Seconds(instance.fleet.service_time_per_stop_s),
        capacity_kg=instance.fleet.vehicle_capacity_kg,
    )
    solution = Solution(
        stage1_routes=_stage_routes(
            group_by_hub(hub_nodes, hub_of_source, source_nodes, source_demand_kg), context
        ),
        stage2_routes=_stage_routes(
            group_by_hub(hub_nodes, hub_of_customer, customer_nodes, customer_demand_kg), context
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


def _nearest_neighbour_tours(
    workload: HubWorkload, context: RoutingContext
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


def _stage_routes(workloads: tuple[HubWorkload, ...], context: RoutingContext) -> tuple[Route, ...]:
    """Build every tour of one stage, hub by hub."""
    return tuple(
        build_route(workload, positions, context)
        for workload in workloads
        for positions in _nearest_neighbour_tours(workload, context)
    )
