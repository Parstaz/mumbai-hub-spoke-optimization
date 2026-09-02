"""Memetic local search: refine each vehicle's own tour, then hand the order back to the GA.

The GA searches sequence space and :func:`~src.stage2.split.split` derives the vehicle boundaries.
That division leaves a gap: split finds the best *partition* of an order it is given, and never
reorders anything. Two stops adjacent in the chromosome but visited in the wrong order stay wrong
however the cuts fall. This module closes that gap by improving each route the split chose, and
writing the improvement back into the chromosome — Lamarckian, so the GA inherits it.

**Moves are evaluated inside one route, never across two.** A route holds at most 20 stops at the
default fleet size, so a full 2-opt neighbourhood is 190 candidates and each is priced by
:meth:`~src.stage2.pricing.TourPricer.whole_weight` in microseconds. The alternative — proposing
moves on the whole permutation and re-splitting to score each one — costs a full split per
candidate move, which is three orders of magnitude more per move and buys the ability to shift work
between vehicles that split is already free to do on the next generation anyway.

**No delta evaluation, deliberately.** Classic 2-opt scores a move by the four legs it changes.
That is unavailable here: ``beta`` is time-denominated and traffic accumulates along the route, so
reversing a segment changes the arrival time at every later stop and therefore the lateness of
every later window. A local move has a global effect on the cost, which is exactly the coupling
CLAUDE.md §1.2 says to preserve. Every candidate is priced in full.

**Why the write-back cannot make a chromosome worse.**

Concatenating the improved routes in plan order yields a permutation of exactly the same stops, and
the improved routes are themselves a contiguous partition of it. An arc weight depends *only* on
the contiguous run of stops it covers — that is step 5's departure-independence property, the same
one that makes the shortest path exact: the fleet leaves together, so what a run costs does not
depend on which vehicle drives it or on what any other vehicle did. Each improved route is
therefore exactly the weight of its own arc in the new order's DAG, and their sum is the weight of
one particular path through it.

:func:`~src.stage2.split.split` returns the *cheapest* path through that DAG. So::

    split(concatenated).search_objective_inr  <=  sum of the improved routes' weights

The re-split can only match or beat the figure local search arrived at. **That is why the fitness
stored for an improved individual is a fresh split of the concatenated chromosome, not the sum over
the improved routes.** The two differ whenever the new order admits a better set of cuts, and
storing the local search's own figure would have the GA rank on a number that re-splitting the
chromosome cannot reproduce — selection would prefer individuals whose recorded fitness the plan
does not actually deliver. One re-split per improved individual, not one per move: at the default
``local_search_pct`` that is 15 extra splits against the generation's own 150.

Anyone changing the write-back should keep both halves of that: the concatenation must preserve the
routes as a contiguous partition, and the fitness must come from splitting the result.
"""

from __future__ import annotations

from collections.abc import Iterator

from src.stage2.pricing import TourPricer, ordered_tour
from src.stage2.split import Permutation, SplitContext
from src.units import Rupees

_MIN_STOPS_TO_REORDER = 3
"""Below three stops a route has one non-trivial ordering, and reversing it changes nothing.

A two-stop tour ``hub -> a -> b -> hub`` and its reverse are different routes on an asymmetric
network, but 2-opt's only move on two stops *is* that reversal, and or-opt's only move produces it
too — so the neighbourhood is a single candidate either way and is left to the GA's operators.
"""


def refine(tours: tuple[Permutation, ...], context: SplitContext, max_passes: int) -> Permutation:
    """Improve each tour in place and concatenate the result into a chromosome.

    Args:
        tours: The partition to improve, from :attr:`~src.stage2.split.SplitPlan.tours`.
        context: The hub's workload, road network, windows and search rates.
        max_passes: Improvement passes per route before giving up, from
            :attr:`~src.config.GAConfig.local_search_max_passes`. A cap rather than convergence,
            because a route that keeps finding half-rupee improvements would otherwise spend the
            generation's whole budget on one vehicle.

    Returns:
        The improved order. Splitting it costs no more than the tours it was built from — see the
        module docstring for why that bound holds.
    """
    return tuple(stop for tour in tours for stop in _improve_tour(tour, context, max_passes))


def _improve_tour(tour: Permutation, context: SplitContext, max_passes: int) -> Permutation:
    """Run first-improvement over one route's neighbourhood until it stops paying.

    First improvement rather than best improvement: taking the first gain found costs one partial
    sweep where best improvement costs a full neighbourhood sweep for every move accepted, and on
    routed tours the two converge to solutions of much the same quality.
    """
    if len(tour) < _MIN_STOPS_TO_REORDER:
        return tour

    best = tour
    best_inr = _tour_weight(best, context)
    for _ in range(max_passes):
        improved = False
        for candidate in _neighbourhood(best):
            candidate_inr = _tour_weight(candidate, context)
            if candidate_inr < best_inr:
                best, best_inr, improved = candidate, candidate_inr, True
                break
        if not improved:
            return best
    return best


def _neighbourhood(tour: Permutation) -> Iterator[Permutation]:
    """Every 2-opt reversal and every single-stop relocation of ``tour``.

    Two move classes because they fix different faults. A 2-opt reversal removes a crossing in the
    route — the classic symptom of a nearest-neighbour construction — while a relocation moves one
    stop that simply sits on the wrong leg. Neither subsumes the other.
    """
    n = len(tour)
    for first in range(n - 1):
        for last in range(first + 1, n):
            yield (*tour[:first], *reversed(tour[first : last + 1]), *tour[last + 1 :])
    for origin in range(n):
        remainder = (*tour[:origin], *tour[origin + 1 :])
        for target in range(len(remainder) + 1):
            if target != origin:
                yield (*remainder[:target], tour[origin], *remainder[target:])


def _tour_weight(tour: Permutation, context: SplitContext) -> Rupees:
    """Price one candidate route through the same pricer the split DAG uses.

    Priced at ``context.cost_config``, which the GA has already scaled by its adaptive time-window
    multiplier. Local search therefore optimises the same objective selection ranks on; refining
    against the configured rate while selecting against the search rate would have the two pulling
    in different directions whenever the multiplier is away from 1.
    """
    return TourPricer(
        tour=ordered_tour(tour, context.workload, context.pricing, context.routing),
        routing=context.routing,
        cost_config=context.cost_config,
    ).whole_weight()
