"""The work to be done, and which hub owns each piece of it.

Every solver in this repository starts from the same three questions: what mass is waiting where,
can the fleet lift it at all, and which hub is responsible for each stop. This module answers
them once, so the greedy benchmark, the Stage 1 CVRP and the Stage 2 GA cannot answer them
differently.

**This module is neutral ground, and that is load-bearing.** ``src/baseline`` and ``src/stage1``
both import from here; neither imports from the other, and nothing here imports either of them.
The baseline is the control the optimized pipeline is measured against, so if it drew
:func:`nearest_hub` from ``src/stage1`` then tuning the treatment would silently move the
control's numbers and the reported improvement would be measuring two changes at once.

:func:`require_servable` is the single expression of an architectural invariant: **no stop is ever
split across vehicles.** A stop holding more mass than one vehicle can carry makes the instance
infeasible rather than triggering a split, for every solver alike. Enforcing it in one function is
what makes the step 8 comparison between the GA and the OR-Tools reference a comparison under
identical constraints.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.config import CAPACITY_TOLERANCE_KG
from src.data.instance import Instance
from src.exceptions import InfeasibleInstanceError
from src.units import DemandArray, DistanceMatrix, NodeArray, NodeId


@dataclass(frozen=True, slots=True)
class HubWorkload:
    """The stops one hub must serve in one stage, with the mass waiting at each.

    Both stages reduce to this shape — a hub, a set of stops, a demand per stop — which is what
    lets one tour builder serve inbound collection and final-mile delivery without knowing which
    it is looking at.
    """

    hub_id: int
    hub_node: NodeId
    nodes: NodeArray
    demand_kg: DemandArray


def stage_demands(instance: Instance) -> tuple[DemandArray, DemandArray]:
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


def require_servable(source_demand_kg: DemandArray, capacity_kg: float) -> None:
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


def node_array(nodes: Iterable[NodeId]) -> NodeArray:
    """Collect node ids into an index array.

    Built from the :class:`~src.data.instance.Instance` accessors rather than by arithmetic over
    the flat layout, so an off-by-one in the node space raises instead of quietly addressing a
    customer as a source.
    """
    return np.fromiter(nodes, dtype=np.intp)


def nearest_hub(
    hub_nodes: NodeArray, stop_nodes: NodeArray, distance_m: DistanceMatrix
) -> NodeArray:
    """Hub id serving each stop: whichever hub is the shortest drive out to it.

    Measured hub-to-stop, the direction the vehicle leaves in. On an asymmetric road network the
    return leg can differ, and a benchmark that averaged the two would be doing a small
    optimization — which is precisely what the baseline must not do.
    """
    block = distance_m[np.ix_(hub_nodes, stop_nodes)]
    return np.asarray(np.argmin(block, axis=0), dtype=np.intp)


def group_by_hub(
    hub_nodes: NodeArray, hub_of_stop: NodeArray, stop_nodes: NodeArray, demand_kg: DemandArray
) -> tuple[HubWorkload, ...]:
    """Group stops by their assigned hub, dropping anything with no work in it.

    A source that no shipment originates at is skipped rather than visited: there is nothing to
    collect there, so a tour through it would burn distance and a fixed vehicle charge to move
    zero kilograms and would make the benchmark artificially bad.
    """
    has_demand = demand_kg > 0.0
    workloads: list[HubWorkload] = []
    for hub_id, hub_node in enumerate(hub_nodes):
        assigned = has_demand & (hub_of_stop == hub_id)
        if assigned.any():
            workloads.append(
                HubWorkload(
                    hub_id=hub_id,
                    hub_node=NodeId(int(hub_node)),
                    nodes=stop_nodes[assigned],
                    demand_kg=demand_kg[assigned],
                )
            )
    return tuple(workloads)
