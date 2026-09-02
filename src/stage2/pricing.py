"""Arc weights for the split DAG, priced from a prefix timeline instead of a built tour.

:func:`~src.stage2.split.split` asks one question over and over: what does it cost for one vehicle
to serve the contiguous run of stops ``i+1..j``? Answering it through
:func:`~src.tour.build_route` materialises a :class:`~src.solution.Route` for every candidate, and
a split keeps three or four of the thousands it evaluates.

**The redundancy this exploits.** Every vehicle departs at ``dispatch_hour``, so the arrival time
at stop ``k`` depends only on the stops before it — not on where the tour ends. Arrivals along
``i+1..j`` are therefore *the same numbers* as along ``i+1..j+1``, and so is the lateness
accumulated over them. One forward pass from ``i`` prices every arc leaving ``i``: extending ``j``
costs one leg of travel and one closing leg back to the hub, not a fresh timeline. That is what
makes this O(1) per arc where building a route is O(tour length).

Measured on seed 42's largest Stage 2 hub under OSRM matrices, 236 stops and 4,530 arcs:
**78.0 ms → 9.9 ms, a factor of 7.9, with all 4,530 arc weights bit-identical.** A prototype that
inlined the rate arithmetic and read window ends off a float array reached 8.3×; routing money
through :func:`~src.scoring.leg_cost` and lateness through
:meth:`~src.data.instance.TimeWindow.lateness_s` costs that difference and is worth it. The
alternative
considered was an arc-weight memo keyed by the stop subsequence. It was measured and rejected: hit
rates run 9.0% across 150 independent permutations, 52% for an OX child against both parents and
74% after a single or-opt move — 2–4×, for a 173k-key working set per hub. A memo exploits
redundancy *between* chromosomes, which depends on the population converging; this exploits
redundancy *inside* one split, which is structural.

**Why this is not a second cost model.** Nothing here computes rupees, distance or travel time of
its own. Money comes from :func:`~src.scoring.leg_cost`, the same arithmetic
:func:`~src.scoring.route_cost` uses. Lateness comes from
:meth:`~src.data.instance.TimeWindow.lateness_s`, so the window keeps owning what "late" means.
Travel time comes from :meth:`~src.costs.traffic.TrafficModel.travel_time_with_traffic`, band
blending and all. What this module contributes is the *order* those calls are made in, and
``tests/test_pricing.py`` is the price of admission for that: over random instances every arc
weight must equal the shared path's under ``==``, not ``approx``.

Two details are load-bearing for that equality and must not be "simplified":

* **Distance is summed as one array, never as a running total.** ``build_route`` sums the whole leg
  vector at once and NumPy's pairwise summation over more than eight elements is not left to
  right, so a prefix difference lands on different last bits. Legs are written into a buffer and
  summed with one ``.sum()`` per arc.
* **Lateness accumulates in stop order**, which is the order
  :func:`~src.scoring.route_window_outcome` accumulates in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.config import CostConfig
from src.data.instance import Instance, TimeWindow
from src.scoring import leg_cost
from src.tour import RoutingContext
from src.units import DemandArray, Metres, NodeId, Rupees, Seconds
from src.workload import HubWorkload

Legs = npt.NDArray[np.float64]
"""One value per leg of a tour, in visit order. Metres or seconds according to the field."""


@dataclass(frozen=True, slots=True)
class HubPricing:
    """The delivery windows of one hub's stops, in :attr:`~src.workload.HubWorkload.nodes` order.

    Resolved once per hub rather than per arc. :func:`~src.scoring.route_window_outcome` looks a
    window up through the :class:`~src.data.instance.Instance` on every stop of every candidate
    tour, which costs two dictionary-free but Python-level lookups per stop and dominated the
    profile once the timeline stopped doing so.

    Precomputing this is also what lets a per-hub GA worker be handed a frozen task with no
    ``Instance`` in it, which §1.1 of CLAUDE.md requires: a worker receives its own sliced
    matrices and nothing else.
    """

    windows: tuple[TimeWindow | None, ...]


def hub_pricing(workload: HubWorkload, instance: Instance) -> HubPricing:
    """Resolve one hub's delivery windows, aligned with its stop list.

    A stop that is not a customer gets ``None``, matching
    :func:`~src.scoring.route_window_outcome`, which skips sources and hub visits rather than
    treating them as unconstrained customers. Stage 2 never prices a source, but agreeing with the
    shared path on a case neither reaches is cheaper than an equivalence test that has to argue
    the case is unreachable.

    Args:
        workload: The hub and the stops it serves.
        instance: Supplies the windows.

    Returns:
        One window or ``None`` per entry of ``workload.nodes``.
    """
    return HubPricing(
        windows=tuple(
            instance.customer_at(NodeId(int(node))).window
            if instance.is_customer_node(NodeId(int(node)))
            else None
            for node in workload.nodes
        )
    )


@dataclass(frozen=True, slots=True)
class OrderedTour:
    """One permutation's legs, sliced out of the matrices in one vectorised pass each.

    Built once per split rather than per arc. The three slices are the only shapes a tour can
    need — out of the hub, along the chain of stops, and back to the hub — because a tour is
    ``hub -> stops -> hub`` and nothing else.
    """

    out_m: Legs
    chain_m: Legs
    back_m: Legs
    out_s: Legs
    chain_s: Legs
    back_s: Legs
    windows: tuple[TimeWindow | None, ...]


def ordered_tour(
    permutation: tuple[int, ...],
    workload: HubWorkload,
    pricing: HubPricing,
    routing: RoutingContext,
) -> OrderedTour:
    """Reorder the hub's legs and windows into the visit order under test.

    Fancy-indexed rather than looped: a Python loop pulling ``distance_m[a, b]`` per leg would be
    scalar indexing into an O(n²) matrix inside the GA's innermost loop, which §2.3 calls a defect.

    Args:
        permutation: Visit order, as positions into ``workload.nodes``.
        workload: The hub and its stops.
        pricing: The hub's windows, from :func:`hub_pricing`.
        routing: Supplies the distance and duration matrices.

    Returns:
        The permutation's legs and windows, ready to price arcs from.
    """
    nodes = np.asarray(workload.nodes, dtype=np.intp)[np.asarray(permutation, dtype=np.intp)]
    hub = int(workload.hub_node)
    distance_m, duration_s = routing.matrices.distance_m, routing.matrices.duration_s
    return OrderedTour(
        out_m=distance_m[hub, nodes],
        chain_m=distance_m[nodes[:-1], nodes[1:]],
        back_m=distance_m[nodes, hub],
        out_s=duration_s[hub, nodes],
        chain_s=duration_s[nodes[:-1], nodes[1:]],
        back_s=duration_s[nodes, hub],
        windows=tuple(pricing.windows[position] for position in permutation),
    )


@dataclass(frozen=True, slots=True)
class ArcPricer:
    """The arc factory for one permutation: which arcs exist, and what each one weighs.

    Capacity lives here because this is where arcs are created, and CLAUDE.md §1.1 makes capacity
    structural rather than penalised — an arc whose load exceeds one vehicle is not expensive, it
    is never yielded.
    """

    tour: OrderedTour
    prefix_kg: DemandArray
    room_kg: float
    routing: RoutingContext
    cost_config: CostConfig

    def weights_from(self, start: int) -> Iterator[tuple[int, Rupees]]:
        """Price every arc leaving DAG node ``start``, cheapest pass first.

        Yields ``(end, weight)`` for one vehicle serving permutation positions ``start..end-1``,
        with ``end`` ascending, stopping at the first load one vehicle cannot carry. Loads are
        non-negative, so every longer arc is over capacity too — the same reason
        :func:`~src.stage2.split.split`'s DAG is O(n × max_tour_length) rather than O(n²).

        Args:
            start: DAG node to leave from: the number of stops already served.

        Yields:
            The arc's far node and its weight in rupees.
        """
        travel = self.routing.traffic.travel_time_with_traffic
        service_s = float(self.routing.service_time_s)
        start_s = float(self.routing.start_time_s)
        tour = self.tour

        legs_m: Legs = np.empty(len(tour.windows) - start + 1, dtype=np.float64)
        legs_m[0] = tour.out_m[start]
        arrival_s = start_s + travel(Seconds(float(tour.out_s[start])), Seconds(start_s))
        lateness_s = 0.0

        for stop in range(start, len(tour.windows)):
            if self.prefix_kg[stop + 1] - self.prefix_kg[start] > self.room_kg:
                return
            if stop > start:
                departure_s = arrival_s + service_s
                leg_s = Seconds(float(tour.chain_s[stop - 1]))
                arrival_s = departure_s + travel(leg_s, Seconds(departure_s))
                legs_m[stop - start] = tour.chain_m[stop - 1]
            window = tour.windows[stop]
            if window is not None:
                lateness_s += window.lateness_s(Seconds(arrival_s))
            yield (
                stop + 1,
                self._closed_weight(legs_m, stop - start + 1, stop, arrival_s, lateness_s),
            )

    def _closed_weight(
        self, legs_m: Legs, served: int, stop: int, arrival_s: float, lateness_s: float
    ) -> Rupees:
        """Close the tour at ``stop`` and price it, without disturbing the running pass.

        The closing leg is written past the stops served so far, so the same buffer serves every
        arc leaving one ``start``: position ``served`` is overwritten on the next iteration by the
        chain leg that actually follows.
        """
        travel = self.routing.traffic.travel_time_with_traffic
        legs_m[served] = self.tour.back_m[stop]
        departure_s = arrival_s + float(self.routing.service_time_s)
        leg_s = Seconds(float(self.tour.back_s[stop]))
        closed_s = departure_s + travel(leg_s, Seconds(departure_s))
        return leg_cost(
            Metres(float(legs_m[: served + 1].sum())),
            Seconds(closed_s - float(self.routing.start_time_s)),
            Seconds(lateness_s),
            self.cost_config,
        ).total_inr
