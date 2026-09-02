"""Which hub collects from each source: the capacity-balanced alternative to nearest-hub.

Nearest-hub is the obvious rule and it is badly unbalanced. On the seed-42 instance it draws
8,850 kg to hub 9 while hubs 8 and 12 draw 300 kg each, because the density clusters the generator
places do not sit one per hub. That imbalance is not primarily a fleet-size problem — every hub
already deploys exactly its mass floor in vehicles, so redistributing mass only removes a vehicle
where it happens to remove a rounding remainder. It is a *distance* problem: an oversubscribed hub
serves sources far outside its natural catchment, and every one of those is a long leg on an
inbound tour.

:func:`capacity_balanced` is the treatment. It is a min-cost flow — this is where NetworkX does
real work rather than decoration — and the model is chosen for one reason above minimising
distance: **one indivisible unit of flow per source.** A source arc of capacity 1 makes splitting
a source across two hubs structurally impossible, which is the invariant every solver in this
repository shares. The cost of that guarantee is that the quantity capped is a source *count*
rather than a mass; a kilogram-denominated arc bound would balance mass exactly and split a source
the instant a cap bound, and there is no min-cost flow that does both. Mass balance here is
emergent — shipments per source is drawn from the same distribution everywhere, so capping counts
does move mass, but not to an exact bound. The Stage 1 CLI reports both spreads so the
approximation is visible rather than assumed.

:func:`nearest_hub` is deliberately **not** in this module. It lives in :mod:`src.workload`, which
the greedy baseline also imports. The baseline is the control; if it drew its hub assignment from
here then tuning this file would move the control's column too, and the reported improvement would
be measuring two changes at once.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

import networkx as nx
import numpy as np

from src.exceptions import InfeasibleInstanceError
from src.units import DistanceMatrix, NodeArray
from src.workload import nearest_hub

logger = logging.getLogger(__name__)

_SINK = "sink"
"""The flow network's single sink. Every hub drains into it under its throughput cap."""


class AssignmentStrategy(Protocol):
    """An interchangeable rule for deciding which hub collects from each stop.

    A ``Protocol`` rather than an ABC: the implementations are plain functions sharing a call
    shape and no state, and structural typing says exactly that. Two of them exist —
    :func:`unconstrained` and :func:`capacity_balanced` — which is what step 7's ablation needs
    and the only reason this abstraction is here at all.
    """

    def __call__(
        self,
        hub_nodes: NodeArray,
        stop_nodes: NodeArray,
        distance_m: DistanceMatrix,
        max_stops_per_hub: int,
    ) -> NodeArray:
        """Return a hub id per entry of ``stop_nodes``."""


def max_stops_per_hub(n_stops: int, n_hubs: int, balance_slack: float) -> int:
    """The cap :func:`capacity_balanced` may not exceed, from the even share and the slack.

    Rounded up, so the cap is never below the even share and the flow is always feasible for
    ``balance_slack >= 1.0``. On the default instance this is
    ``ceil(285 / 16 × 1.25) = 23`` sources per hub.

    Args:
        n_stops: Stops to assign — sources with something waiting at them.
        n_hubs: Hubs available to receive them.
        balance_slack: Multiplier on the even share, from
            :attr:`~src.config.Stage1Config.hub_balance_slack`.

    Returns:
        The maximum number of stops any one hub may be assigned.
    """
    return math.ceil(n_stops / n_hubs * balance_slack)


def unconstrained(
    hub_nodes: NodeArray,
    stop_nodes: NodeArray,
    distance_m: DistanceMatrix,
    _max_stops_per_hub: int,
) -> NodeArray:
    """Nearest-hub, adapted to the strategy signature by discarding the cap.

    The discarded argument *is* the ablation. Nearest-hub is exactly
    :func:`capacity_balanced` with no cap to bind — both minimise total hub-to-stop road distance,
    and the difference between the two columns is what enforcing the cap costs in distance and
    buys in balance. Keeping them behind one signature is what lets step 7 swap them without
    touching anything else.

    Args:
        hub_nodes: Hub node ids.
        stop_nodes: Stop node ids to assign.
        distance_m: Road distances for the whole instance.
        _max_stops_per_hub: Ignored. Present only to satisfy :class:`AssignmentStrategy`.

    Returns:
        A hub id per entry of ``stop_nodes``.
    """
    return nearest_hub(hub_nodes, stop_nodes, distance_m)


def capacity_balanced(
    hub_nodes: NodeArray,
    stop_nodes: NodeArray,
    distance_m: DistanceMatrix,
    max_stops_per_hub: int,
) -> NodeArray:
    """Assign stops to hubs by min-cost flow, under a per-hub cap on how many it may take.

    The network is bipartite with a single sink: one unit of supply per stop, a unit-capacity arc
    from each stop to each hub weighted by road distance, and a capped zero-weight arc from each
    hub to the sink. The optimum is therefore the cheapest assignment that respects the cap, and
    the unit arc capacity is what forbids splitting a stop.

    Weights and capacities are integers throughout. ``network_simplex`` is documented as
    unreliable on floating-point input, and metres are already a finer unit than the road network
    resolves, so rounding distance to whole metres costs nothing.

    Args:
        hub_nodes: Hub node ids, positionally indexed by hub id.
        stop_nodes: Stop node ids to assign; pass only stops with demand.
        distance_m: Road distances for the whole instance, indexed by node id.
        max_stops_per_hub: Cap on stops per hub, from :func:`max_stops_per_hub`.

    Returns:
        A hub id per entry of ``stop_nodes``, each stop assigned wholly to one hub.

    Raises:
        InfeasibleInstanceError: If the caps cannot absorb every stop, or if the flow is otherwise
            unsolvable. Never propagates a NetworkX exception: a caller cannot be expected to
            catch a foreign type to learn that its own configuration was too tight.
    """
    n_hubs, n_stops = len(hub_nodes), len(stop_nodes)
    if n_hubs * max_stops_per_hub < n_stops:
        raise InfeasibleInstanceError(
            f"{n_hubs} hubs capped at {max_stops_per_hub} stops each cannot absorb {n_stops} "
            f"stops; raise Stage1Config.hub_balance_slack above "
            f"{n_stops / (n_hubs * max(max_stops_per_hub, 1)):.2f}× its current value"
        )

    graph = _flow_network(hub_nodes, stop_nodes, distance_m, max_stops_per_hub)
    try:
        total_metres, flow = nx.network_simplex(graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded) as exc:
        raise InfeasibleInstanceError(
            f"no feasible hub assignment for {n_stops} stops over {n_hubs} hubs "
            f"capped at {max_stops_per_hub} each: {exc}"
        ) from exc

    logger.info(
        "min-cost flow assigned %d stops over %d hubs, cap %d, total hub-to-stop distance %.1f km",
        n_stops,
        n_hubs,
        max_stops_per_hub,
        int(total_metres) / 1000.0,
    )
    return _hub_of_stop(flow, n_hubs, n_stops)


def _flow_network(
    hub_nodes: NodeArray,
    stop_nodes: NodeArray,
    distance_m: DistanceMatrix,
    max_stops_per_hub: int,
) -> nx.DiGraph:
    """Build the bipartite flow network. See :func:`capacity_balanced` for the model.

    The hub-to-stop distance block is sliced once with :func:`numpy.ix_` and rounded vectorised;
    a Python loop computing ``distance_m[hub, stop]`` per arc would be an O(n²) scalar indexing
    pass over the matrix.
    """
    block = np.rint(distance_m[np.ix_(hub_nodes, stop_nodes)]).astype(np.int64)
    graph = nx.DiGraph()
    graph.add_node(_SINK, demand=len(stop_nodes))
    for stop_index in range(len(stop_nodes)):
        graph.add_node(("stop", stop_index), demand=-1)
        for hub_id in range(len(hub_nodes)):
            graph.add_edge(
                ("stop", stop_index),
                ("hub", hub_id),
                capacity=1,
                weight=int(block[hub_id, stop_index]),
            )
    for hub_id in range(len(hub_nodes)):
        graph.add_edge(("hub", hub_id), _SINK, capacity=max_stops_per_hub, weight=0)
    return graph


def _hub_of_stop(flow: dict[object, dict[object, int]], n_hubs: int, n_stops: int) -> NodeArray:
    """Read the assignment off the flow dictionary.

    Exactly one arc out of each stop carries flow — that is what unit supply and unit arc capacity
    guarantee — so the assignment is a lookup rather than a decision. The completeness check below
    is not defensive: it is the assertion that the guarantee held, and it fires as a bug rather
    than as an infeasibility.
    """
    hub_of_stop = np.full(n_stops, -1, dtype=np.intp)
    for stop_index in range(n_stops):
        for hub_id in range(n_hubs):
            if flow[("stop", stop_index)][("hub", hub_id)]:
                hub_of_stop[stop_index] = hub_id
                break
    unassigned = int(np.count_nonzero(hub_of_stop < 0))
    if unassigned:
        raise InfeasibleInstanceError(
            f"min-cost flow left {unassigned} of {n_stops} stops unassigned, which unit supply "
            f"per stop should have made impossible"
        )
    return hub_of_stop
