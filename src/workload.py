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
from typing import Literal

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


def require_servable(
    demand_kg: DemandArray, capacity_kg: float, stop_kind: Literal["source", "customer"]
) -> None:
    """Reject an instance whose stops cannot be emptied one vehicle-load at a time.

    Both stages call this, over their own stops: Stage 1 over the sources it collects from, Stage 2
    over one hub's customers before :func:`~src.stage2.split.split` builds its DAG. Stage 2's call
    is load-bearing rather than belt-and-braces — the split DAG reaches its terminal node only
    because every single-stop arc exists, and a stop no vehicle can lift is exactly the arc that
    would be missing. Passing here is what guarantees the shortest path has a finite optimum.

    On a generated instance the customer side cannot fire: every shipment is checked against
    capacity when the instance is built, and each customer is the destination of exactly one. That
    is a property of the current generator and of uniform shipment sizes, not of the rule, which is
    why the check is made rather than assumed.

    ``stop_kind`` has no default on purpose. It is the noun in the refusal message, and a caller
    that omitted it would name the wrong kind of stop in the one place a reader looks to find out
    what was refused.

    Args:
        demand_kg: Mass waiting at each stop, aligned with the stop list under test.
        capacity_kg: What one vehicle can carry.
        stop_kind: What the stops in ``demand_kg`` are, for the message.

    Raises:
        InfeasibleInstanceError: If any stop holds more mass than one vehicle can carry.
    """
    over = np.flatnonzero(demand_kg > capacity_kg + CAPACITY_TOLERANCE_KG)
    if over.size:
        raise InfeasibleInstanceError(
            f"{over.size} {stop_kind}(s) hold more than one vehicle can carry "
            f"(worst {demand_kg[over].max():.1f} kg against {capacity_kg:.1f} kg capacity); "
            f"no stop is ever split across vehicles"
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
