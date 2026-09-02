"""Turn a hub, a visit order and the road network into a :class:`~src.solution.Route`.

Every solver produces the same thing — an ordering of one hub's stops — and every solver then has
to convert that ordering into the physical facts a ``Route`` carries: nodes, load, distance,
duration, arrival times. This module is that conversion, and it exists so there is exactly one of
it. A second copy would let two solvers disagree about what the *same* ordering costs, which is
indistinguishable from one of them routing better.

Arrivals come from :func:`~src.costs.traffic.route_timeline`, so the traffic multiplier
accumulates along the tour — a delayed stop pushes every later stop into whatever band it now
lands in — and the route's duration is read back off that timeline rather than summed
independently. That keeps this module, and every solver above it, free of travel-time arithmetic
of its own.

Like :mod:`src.workload`, this is neutral ground: the baseline and the optimized stages both
import it and neither imports the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel, route_timeline
from src.solution import Route
from src.units import Metres, NodeId, Seconds
from src.workload import HubWorkload


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """The instance-wide inputs every tour needs, bundled so the builders stay small.

    Carrying the matrices and the traffic model through here rather than rebuilding either per
    hub is what guarantees every solver is measured over exactly the same road network and the
    same time-of-day multipliers.
    """

    matrices: CostMatrices
    traffic: TrafficModel
    start_time_s: Seconds
    service_time_s: Seconds
    capacity_kg: float


def build_route(
    workload: HubWorkload, positions: tuple[int, ...], context: RoutingContext
) -> Route:
    """Turn one vehicle's visit order into a :class:`~src.solution.Route`.

    Args:
        workload: The hub and its candidate stops; ``positions`` indexes into ``workload.nodes``.
        positions: This vehicle's stops, in visit order, as positions into ``workload.nodes``.
        context: The road network, the traffic schedule, and the dispatch and service times.

    Returns:
        The tour ``hub -> stops -> hub``, with its distance, duration and arrival times.
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
