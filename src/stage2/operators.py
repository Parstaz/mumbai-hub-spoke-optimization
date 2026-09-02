"""Crossover and mutation over a delimiter-free permutation.

Both operators here have one obligation above being good search moves: **whatever they return must
be a permutation of what they were given.** Stage 2's chromosome carries no vehicle boundaries, so
a dropped or duplicated stop does not produce an invalid-looking plan — it produces tours that
close at their hub, respect capacity, and cost *less* than the correct answer because they deliver
less. :func:`~src.stage2.split.split` refuses such a chromosome, but only after the operator that
caused it has already been selected from. That is why ``tests/test_operators.py`` carries
property-based tests on exactly this invariant, as CLAUDE.md §3 requires.

Neither operator knows about capacity, time windows or cost. They rearrange a sequence; the split
DAG decides what that sequence is worth. Keeping them that ignorant is what lets §1.1 call capacity
structural — an operator that tried to respect capacity would be choosing vehicle boundaries, which
is precisely the job the route-first/cluster-second encoding takes away from the GA.
"""

from __future__ import annotations

import numpy as np

from src.stage2.split import Permutation

_MIN_STOPS_TO_REARRANGE = 2
"""Below two stops there is only one ordering, so every move here is the identity."""


def order_crossover(
    first: Permutation, second: Permutation, rng: np.random.Generator
) -> Permutation:
    """Recombine two visit orders with OX, preserving a contiguous run from ``first``.

    OX is the right crossover for this encoding because it inherits *relative order* rather than
    absolute position. A child keeps one parent's contiguous segment intact and fills the rest with
    the other parent's stops in the sequence that parent visited them, which is what a tour cares
    about: two chromosomes that visit the same customers in the same order are the same plan
    however that order is offset.

    The alternative worth naming is a positional crossover such as PMX, which preserves absolute
    index and therefore recombines *where in the giant tour* a stop sits. That is meaningless here:
    position in the chromosome has no interpretation until :func:`~src.stage2.split.split` chooses
    the cuts, and it chooses them fresh for every chromosome.

    Args:
        first: The parent whose contiguous segment the child keeps.
        second: The parent supplying the remaining stops, in its own visit order.
        rng: Injected generator. Never the legacy global — see CLAUDE.md §2.3.

    Returns:
        A child holding exactly the stops of ``first``, each once.
    """
    n = len(first)
    if n < _MIN_STOPS_TO_REARRANGE:
        return first
    start, end = sorted(int(cut) for cut in rng.choice(n, size=2, replace=False))

    child: list[int] = [-1] * n
    child[start : end + 1] = first[start : end + 1]
    held = frozenset(first[start : end + 1])

    # Both the read from ``second`` and the write into the child start just past the segment and
    # wrap, which is what makes the inherited order relative rather than positional.
    donor = (second[(end + 1 + offset) % n] for offset in range(n))
    remaining = [stop for stop in donor if stop not in held]
    for offset, stop in enumerate(remaining):
        child[(end + 1 + offset) % n] = stop
    return tuple(child)


def or_opt_mutation(
    permutation: Permutation, max_segment_stops: int, rng: np.random.Generator
) -> Permutation:
    """Lift a short run of consecutive stops out and reinsert it elsewhere, order preserved.

    Or-opt rather than a swap because a swap of two stops changes four legs at once and almost
    always for the worse on a routed tour, so it behaves as noise. Moving a run of one to three
    stops changes three legs and expresses the move a planner actually makes — "these two drops
    belong on the other side of the round" — which is also the move the memetic local search
    refines within a route.

    The segment is reinserted **unreversed**. Reversal is 2-opt's move and belongs to
    :mod:`src.stage2.local_search`, where it is evaluated rather than gambled on.

    The move may land the segment back where it started, which is a no-op. Rejecting that would
    bias the position it lands in, and a mutation that occasionally does nothing costs one split
    it was going to spend anyway.

    Args:
        permutation: The order to perturb.
        max_segment_stops: Longest run that may be moved, from
            :attr:`~src.config.GAConfig.or_opt_max_segment_stops`.
        rng: Injected generator.

    Returns:
        A new order holding exactly the stops of ``permutation``, each once.
    """
    n = len(permutation)
    if n < _MIN_STOPS_TO_REARRANGE:
        return permutation

    # Capped at n - 1 so something is left to reinsert against: moving the whole tour is the
    # identity however far it is then shifted.
    length = int(rng.integers(1, min(max_segment_stops, n - 1) + 1))
    start = int(rng.integers(0, n - length + 1))
    segment = permutation[start : start + length]
    remainder = (*permutation[:start], *permutation[start + length :])
    insert_at = int(rng.integers(0, len(remainder) + 1))
    return (*remainder[:insert_at], *segment, *remainder[insert_at:])
