"""The split procedure: the optimal partition of a delivery order into vehicle tours.

Stage 2's chromosome is a plain permutation of one hub's customers with **no route delimiters**.
:func:`split` is what turns it into vehicle tours, by solving a shortest path over an auxiliary
DAG: node ``i`` means "the first ``i`` customers of the permutation have been served", and arc
``(i, j)`` means "one vehicle serves positions ``i+1..j`` as a single tour", weighted by what that
tour costs. The shortest path from ``0`` to ``n`` is the cheapest partition of that order. This is
route-first / cluster-second, after Prins (2004).

**Why this encoding rather than delimiters.** The obvious alternative writes vehicle boundaries
into the chromosome — ``[c3, c7, |, c1, c4]`` — and lets the operators move them. Crossover then
recombines the cut points as well as the order, so it constantly emits tours over capacity, which
needs a repair operator; and repair, not selection, ends up deciding what the population looks
like. Here the delimiters are not searched at all. They are *derived*, optimally, for whatever
order the GA proposes: every chromosome maps to a feasible plan, and to the best plan its order
admits. That is what lets the GA search sequence space alone, and why §1.1 of CLAUDE.md can call
capacity structural rather than penalised — there is no repair operator in this codebase because
this function makes one unnecessary.

**Capacity is hard by construction.** An arc whose load exceeds one vehicle is never created, so
an over-capacity tour is not expensive, it is unrepresentable. **Time windows are soft**: lateness
is priced into the arc weight, so the search prefers a time-feasible partition but still returns a
complete plan when no partition is time-feasible — which is the point of a soft constraint.

**Why the shortest path is exact.** Every vehicle departs at ``dispatch_hour``: the fleet leaves
together, which is the model the greedy baseline uses too. An arc's weight therefore depends only
on the contiguous run of stops it covers — not on which vehicle serves it, and not on what any
other vehicle did. That independence is what makes this a shortest-path problem rather than an
approximation of one. Staggered departures would break the formulation outright; they would need a
different algorithm, not a patch to this one.

**Why it always terminates with a finite answer.** :func:`~src.workload.require_servable` has
passed before the DAG is walked, so every single-stop arc ``(i, i + 1)`` exists and node ``n`` is
reachable from node ``0`` along the singleton chain. No unreachable node is possible, which is why
nothing below guards against one.

**Arc weights come from the shared paths and nowhere else.** A weight is
:func:`~src.tour.build_route` for the physical tour, then :func:`~src.scoring.route_window_outcome`
and :func:`~src.scoring.route_cost` for the money. This module contains no distance arithmetic, no
traffic arithmetic and no cost arithmetic of its own — the fixed vehicle charge that lets the DAG
trade one more vehicle against one longer tour is ``gamma`` inside ``route_cost``, not a term
added here.

**What that costs, measured.** A split is ``O(n x max_tour_length)`` arc weights, and each weight
is a whole ``build_route`` and a whole ``route_cost``. On seed 42's Stage 2 hubs under OSRM
matrices: **9.3 ms** per split at the median hub of 40 stops, **80 ms** at the largest of 250. At
:class:`~src.config.GAConfig`'s defaults that is 1.4 s and 12.0 s per generation, so 14 minutes and
2 hours respectively for one hub's 600 generations. The GA cannot run at those defaults against
this pricer. That is a step 6 problem and is deliberately not solved here: this is the module whose
errors are invisible, so it is written to be obviously right before it is written to be fast.

When the optimisation comes — an arc-weight memo keyed by the stop subsequence, or an incremental
prefix timeline exploiting that arrivals for ``hub -> p_i+1 .. p_j`` do not depend on what follows
— its **price of admission is an equivalence test**: over random permutations the fast path must
produce ``Route`` objects identical to :func:`~src.tour.build_route`'s and ``route_cost`` values
identical to the shared path's, trip for trip. Without that test the optimisation is a second cost
model with a performance argument attached, and §1.1's single scoring path is gone.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from src.config import CAPACITY_TOLERANCE_KG, CostConfig
from src.data.instance import Instance
from src.exceptions import InfeasibleSolutionError
from src.scoring import route_cost, route_window_outcome
from src.solution import Route
from src.tour import RoutingContext, build_route
from src.units import DemandArray, Rupees
from src.workload import HubWorkload, require_servable

Permutation = tuple[int, ...]
"""One hub's stops in visit order, as positions into ``HubWorkload.nodes``. No delimiters."""


@dataclass(frozen=True, slots=True)
class SplitContext:
    """Everything a split needs that does not vary between chromosomes.

    Bundled so :func:`split` takes two arguments and the GA builds this once per hub rather than
    threading five parameters through its generation loop.

    ``instance`` is here because :func:`~src.scoring.route_window_outcome` resolves delivery
    windows through it, and measuring lateness a second way — off a precomputed window array, say
    — would be exactly the duplicated logic :mod:`src.scoring` exists to prevent.
    """

    workload: HubWorkload
    routing: RoutingContext
    instance: Instance
    cost_config: CostConfig


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """The optimal partition of one permutation, and what the DAG scored it at.

    ``search_objective_inr`` is the shortest path's own total: the sum of the arc weights along
    the chosen partition. It exists to **rank permutations** inside the GA and for nothing else.
    It is never reported, and never compared against the baseline column.

    It will not generally equal :func:`~src.scoring.evaluate_solution`'s total over the same
    ``routes``, and that is deliberate. The GA guides its search with an adaptive time-window
    multiplier, supplied by handing this class's producer a ``CostConfig`` whose
    ``tw_penalty_per_hour`` has been scaled — see :func:`split`. Under that config the arc weights
    are priced at the search rate while the reported cost stays at the configured one. Same
    routes, two numbers, on purpose: reconciling them would either leak search guidance into the
    headline figure or take the adaptive penalty away from the search.
    """

    routes: tuple[Route, ...]
    search_objective_inr: Rupees


def split(permutation: Permutation, context: SplitContext) -> SplitPlan:
    """Partition ``permutation`` into the cheapest set of feasible vehicle tours.

    The returned partition is optimal for the given order — that is the guarantee the GA is built
    on. It is not the optimal set of tours for the hub, which would require reordering; finding a
    good order is the GA's job and this function's whole purpose is to relieve it of the rest.

    The GA applies its adaptive time-window penalty by passing a scaled ``cost_config``::

        replace(cost_config, tw_penalty_per_hour=cost_config.tw_penalty_per_hour * multiplier)

    Nothing here knows that adaptation exists; it prices arcs at whatever rate it is handed, which
    is what keeps the search's objective and the reported cost separable.

    Args:
        permutation: This hub's stops in visit order, as positions into
            ``context.workload.nodes`` — each position exactly once, no delimiters.
        context: The hub's workload, the road network, the instance and the rates to price at.

    Returns:
        The optimal partition as tours, with the shortest path's own total. An empty permutation
        yields no tours and a zero objective.

    Raises:
        InfeasibleSolutionError: If ``permutation`` is not each of the hub's stop positions
            exactly once. That is a broken crossover or mutation operator rather than a hard
            instance, which is what this exception means.
        InfeasibleInstanceError: If a stop holds more mass than one vehicle can carry.
    """
    _require_permutation(permutation, len(context.workload.nodes))
    require_servable(context.workload.demand_kg, context.routing.capacity_kg, "customer")

    prefix_kg = _prefix_load_kg(permutation, context.workload.demand_kg)
    best_inr, predecessor = _shortest_path(permutation, prefix_kg, context)
    return SplitPlan(
        routes=_tours(predecessor, permutation, context),
        search_objective_inr=Rupees(best_inr[-1]),
    )


def _require_permutation(permutation: Permutation, n_stops: int) -> None:
    """Reject a chromosome that is not a permutation of the hub's stops.

    Checked on every call rather than trusted, because a crossover that drops or duplicates a stop
    still produces tours that look entirely plausible: they close at their hub, they respect
    capacity, and they cost less than the correct answer because they deliver less. Scoring would
    catch it eventually, at the end of a run; this catches it at the operator that caused it.
    """
    if sorted(permutation) != list(range(n_stops)):
        raise InfeasibleSolutionError(
            f"a chromosome must hold each of the hub's {n_stops} stop positions exactly once; "
            f"got {len(permutation)} positions covering {len(set(permutation))} distinct stops"
        )


def _prefix_load_kg(permutation: Permutation, demand_kg: DemandArray) -> DemandArray:
    """Cumulative mass along the permutation, with a leading zero.

    Reduces an arc's load to one subtraction, ``prefix[j] - prefix[i]``, so the capacity test that
    decides whether an arc exists at all never sums a slice. Vectorised because the alternative is
    a Python loop over the hub's demands inside the GA's innermost loop.
    """
    ordered = demand_kg[np.asarray(permutation, dtype=np.intp)]
    prefix: DemandArray = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(ordered)))
    return prefix


def _shortest_path(
    permutation: Permutation, prefix_kg: DemandArray, context: SplitContext
) -> tuple[list[float], list[int]]:
    """Walk the DAG in topological order, which for this graph is just index order.

    One forward pass suffices — no Bellman-Ford relaxation rounds — because an arc only ever runs
    from a lower index to a higher one, so ``best_inr[i]`` is final before any arc leaving ``i``
    is considered.

    Args:
        permutation: The visit order being partitioned.
        prefix_kg: Cumulative load along it, from :func:`_prefix_load_kg`.
        context: Supplies vehicle capacity and everything an arc weight needs.

    Returns:
        The cheapest cost to reach each node, and the predecessor that achieved it.
    """
    n = len(permutation)
    best_inr = [0.0, *([math.inf] * n)]
    predecessor = [0] * (n + 1)
    room_kg = context.routing.capacity_kg + CAPACITY_TOLERANCE_KG

    for i in range(n):
        for j in range(i + 1, n + 1):
            # Loads are non-negative, so once one arc out of i is too heavy every longer one is
            # too. Breaking is what makes this O(n x max_tour_length) rather than O(n squared).
            if prefix_kg[j] - prefix_kg[i] > room_kg:
                break
            candidate_inr = best_inr[i] + _arc_weight_inr(permutation[i:j], context)
            # Strictly cheaper, with i ascending, so ties fall to the lowest predecessor and two
            # splits of one permutation are the same plan rather than merely the same price.
            if candidate_inr < best_inr[j]:
                best_inr[j] = candidate_inr
                predecessor[j] = i

    return best_inr, predecessor


def _arc_weight_inr(stops: Permutation, context: SplitContext) -> Rupees:
    """Price one candidate tour: the weight of the arc that serves ``stops`` with one vehicle.

    Every number here comes from the shared paths — the tour from :func:`~src.tour.build_route`
    under the cumulative traffic model, the money from :func:`~src.scoring.route_cost`. The
    lateness term is what makes the partition time-window aware, and the fixed vehicle charge
    inside ``route_cost`` is what makes one more vehicle a decision the DAG has to justify.
    """
    route = build_route(context.workload, stops, context.routing)
    window = route_window_outcome(route, context.instance)
    return route_cost(route, context.cost_config, window).total_inr


def _tours(
    predecessor: list[int], permutation: Permutation, context: SplitContext
) -> tuple[Route, ...]:
    """Rebuild the chosen partition's tours by walking the predecessors back from node ``n``.

    Rebuilt rather than cached during the pass: keeping the winning ``Route`` for every node would
    hold ``n`` tours alive to return the three or four that survive, and the rebuild is one
    :func:`~src.tour.build_route` per deployed vehicle against the thousands the pass already did.
    """
    cuts = [len(permutation)]
    while cuts[-1] > 0:
        cuts.append(predecessor[cuts[-1]])
    cuts.reverse()
    return tuple(
        build_route(context.workload, permutation[i:j], context.routing)
        for i, j in itertools.pairwise(cuts)
    )
