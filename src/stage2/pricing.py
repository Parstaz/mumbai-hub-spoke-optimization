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
:meth:`~src.data.instance.TimeWindow.lateness_s` costs that difference and is worth it.

The alternative considered was an arc-weight memo keyed by the stop subsequence. It was measured
and rejected: hit rates run 9.0% across 150 independent permutations, 52% for an OX child against
both parents and 74% after a single or-opt move — 2–4×, for a 173k-key working set per hub. A memo
exploits redundancy *between* chromosomes, which depends on the population converging; this
exploits redundancy *inside* one split, which is structural.

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
from src.exceptions import InfeasibleSolutionError
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
class TourPricer:
    """Prices vehicle tours over one permutation, off a single forward timeline.

    Two callers ask two different questions of the same walk. The split DAG wants **every prefix**
    from a starting point — one arc per stop it might end at — so it closes the tour at each step.
    The memetic local search wants **one whole tour**, a candidate reordering of a route split
    already chose, so it closes once at the end. Sharing :meth:`_advance` between them is what
    keeps the second from becoming a third cost path: the local search runs inside a per-hub worker
    that has no ``Instance``, so it cannot fall back on
    :func:`~src.scoring.route_window_outcome` and would otherwise need its own lateness loop.

    Capacity is not a field here. It belongs to the DAG question — which arcs exist — and not to
    pricing a tour the caller already knows one vehicle can carry.
    """

    tour: OrderedTour
    routing: RoutingContext
    cost_config: CostConfig

    def weights_from(
        self, start: int, prefix_kg: DemandArray, room_kg: float
    ) -> Iterator[tuple[int, Rupees]]:
        """Price every arc leaving DAG node ``start``, ``end`` ascending.

        Yields ``(end, weight)`` for one vehicle serving permutation positions ``start..end-1``,
        stopping before the first load one vehicle cannot carry. Loads are non-negative, so every
        longer arc is over capacity too — the same reason
        :func:`~src.stage2.split.split`'s DAG is O(n × max_tour_length) rather than O(n²).

        The reachable end is found first, by scalar arithmetic over ``prefix_kg`` alone, so no leg
        of an infeasible arc is ever driven. CLAUDE.md §1.1 makes capacity structural: an
        over-capacity arc is not expensive here, it is never yielded.

        Args:
            start: DAG node to leave from: the number of stops already served.
            prefix_kg: Cumulative load along the permutation, with a leading zero.
            room_kg: What one vehicle can carry, plus the float tolerance.

        Yields:
            The arc's far node and its weight in rupees.
        """
        last = start
        while last < len(self.tour.windows) and prefix_kg[last + 1] - prefix_kg[start] <= room_kg:
            last += 1

        legs_m = self._leg_buffer(start)
        for stop, arrival_s, lateness_s in self._advance(start, last, legs_m):
            yield (
                stop + 1,
                self._closed_weight(legs_m, stop - start + 1, stop, arrival_s, lateness_s),
            )

    def whole_weight(self) -> Rupees:
        """Price the entire tour as one vehicle's work, closing once rather than at every stop.

        What the local search evaluates a candidate move with. Closing once matters: a 2-opt pass
        over a 20-stop route proposes 190 reorderings, and pricing each as a family of 20 prefixes
        to read the last would cost twenty times what the answer needs.

        Raises:
            InfeasibleSolutionError: If the tour has no stops. A deployed vehicle carries
                something — :class:`~src.solution.Route` refuses the same thing.
        """
        n_stops = len(self.tour.windows)
        if n_stops == 0:
            raise InfeasibleSolutionError("a tour with no stops is not a tour")

        legs_m = self._leg_buffer(0)
        stop, arrival_s, lateness_s = 0, 0.0, 0.0
        for stop, arrival_s, lateness_s in self._advance(0, n_stops, legs_m):  # noqa: B007
            pass
        return self._closed_weight(legs_m, n_stops, stop, arrival_s, lateness_s)

    def whole_lateness_s(self) -> Seconds:
        """Total seconds by which this tour misses its customers' windows.

        Falls out of the same walk that prices it, which is why it is here rather than computed
        again somewhere else. The adaptive penalty needs to know how much of the population is
        missing windows, and a per-hub worker has no :class:`~src.data.instance.Instance` to reach
        :func:`~src.scoring.route_window_outcome` through.

        A tour with no stops is not late, so this returns zero rather than raising — unlike
        :meth:`whole_weight`, which would have to invent a vehicle to charge for.
        """
        n_stops = len(self.tour.windows)
        if n_stops == 0:
            return Seconds(0.0)
        lateness_s = 0.0
        for _, _, lateness_s in self._advance(0, n_stops, self._leg_buffer(0)):  # noqa: B007
            pass
        return Seconds(lateness_s)

    def _leg_buffer(self, start: int) -> Legs:
        """A scratch vector for one pass's leg distances, with the hub-to-first leg already in it.

        Reused across every close in one pass, which is why the closing leg is written past the
        stops served so far: the next stop's chain leg overwrites it.
        """
        legs_m: Legs = np.empty(len(self.tour.windows) - start + 1, dtype=np.float64)
        legs_m[0] = self.tour.out_m[start]
        return legs_m

    def _advance(self, start: int, last: int, legs_m: Legs) -> Iterator[tuple[int, float, float]]:
        """Walk the timeline over stops ``start..last-1``, filling ``legs_m`` as it goes.

        Yields the running clock at each stop: the arrival there, and the lateness accumulated
        from ``start`` to there. Neither depends on where the tour eventually closes, which is the
        whole reason one pass can answer for every arc leaving ``start``.
        """
        travel = self.routing.traffic.travel_time_with_traffic
        service_s = float(self.routing.service_time_s)
        start_s = float(self.routing.start_time_s)
        tour = self.tour

        arrival_s = start_s + travel(Seconds(float(tour.out_s[start])), Seconds(start_s))
        lateness_s = 0.0
        for stop in range(start, last):
            if stop > start:
                departure_s = arrival_s + service_s
                leg_s = Seconds(float(tour.chain_s[stop - 1]))
                arrival_s = departure_s + travel(leg_s, Seconds(departure_s))
                legs_m[stop - start] = tour.chain_m[stop - 1]
            window = tour.windows[stop]
            if window is not None:
                lateness_s += window.lateness_s(Seconds(arrival_s))
            yield stop, arrival_s, lateness_s

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
