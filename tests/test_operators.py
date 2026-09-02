"""Tests for the GA's crossover and mutation.

The property tests here are the ones CLAUDE.md §3 makes mandatory, and they check one thing above
all: an operator returns a permutation of what it was handed. That invariant is load-bearing rather
than tidy. A chromosome carries no vehicle delimiters, so an operator that drops a stop yields a
plan that looks entirely legal — tours close at their hub, loads are under capacity — and costs
*less* than the correct answer, because it delivers less. Selection would then prefer it.

Determinism throughout: every generator is ``np.random.default_rng(seed)`` with the seed written
down, never the legacy global.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import GAConfig
from src.stage2.operators import or_opt_mutation, order_crossover
from src.stage2.split import Permutation

MAX_SEGMENT = GAConfig().or_opt_max_segment_stops
OR_OPT_MAX_BROKEN_LEGS = 3
"""Lifting a run breaks the tour where it left, and at both ends of where it lands."""


@st.composite
def permutation_pair(draw: st.DrawFn) -> tuple[Permutation, Permutation, int]:
    """Two orderings of the same stop set, plus a seed."""
    n_stops = draw(st.integers(min_value=0, max_value=30))
    first = tuple(draw(st.permutations(range(n_stops))))
    second = tuple(draw(st.permutations(range(n_stops))))
    return first, second, draw(st.integers(min_value=0, max_value=2**16))


def preserves_ox_structure(child: Permutation, first: Permutation, second: Permutation) -> bool:
    """Whether ``child`` is a legal OX child of these parents, for *some* pair of cut points.

    An oracle by verification rather than by construction: it does not rebuild the child, it looks
    for a contiguous run inherited from ``first`` such that everything outside that run appears in
    the relative order ``second`` visits it, read from just past the run and wrapping. The cut
    points are the operator's own random draw, so searching for them is what lets this assert OX's
    defining property without asserting on a generator's draw sequence.

    Rejects a positional fill, a fill that does not wrap, a reversed fill and a duplicated stop —
    :func:`test_the_structure_oracle_rejects_a_child_that_is_not_ox` is the proof it can fail.
    """
    n = len(child)
    if n == 0:
        return True
    for start in range(n):
        for end in range(start, n):
            if child[start : end + 1] != first[start : end + 1]:
                continue
            held = frozenset(first[start : end + 1])
            rotation = [(end + 1 + offset) % n for offset in range(n)]
            in_child = [child[index] for index in rotation if child[index] not in held]
            in_second = [second[index] for index in rotation if second[index] not in held]
            if in_child == in_second:
                return True
    return False


# --------------------------------------------------------------------------------------------
# The mandatory invariant
# --------------------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(case=permutation_pair())
def test_crossover_returns_a_permutation_of_its_parents(
    case: tuple[Permutation, Permutation, int],
) -> None:
    """Every stop exactly once — no duplicate, no omission, whatever the cut points."""
    first, second, seed = case
    child = order_crossover(first, second, np.random.default_rng(seed))
    assert sorted(child) == sorted(first)
    assert len(child) == len(first)


@settings(max_examples=300, deadline=None)
@given(case=permutation_pair())
def test_mutation_returns_a_permutation_of_its_input(
    case: tuple[Permutation, Permutation, int],
) -> None:
    """A relocated segment changes the order and nothing else."""
    permutation, _, seed = case
    mutated = or_opt_mutation(permutation, MAX_SEGMENT, np.random.default_rng(seed))
    assert sorted(mutated) == sorted(permutation)
    assert len(mutated) == len(permutation)


@settings(max_examples=200, deadline=None)
@given(case=permutation_pair())
def test_repeated_mutation_never_degrades_the_chromosome(
    case: tuple[Permutation, Permutation, int],
) -> None:
    """Mutation composed with itself is still a permutation.

    Worth its own test because the GA applies mutation to already-mutated children for hundreds of
    generations: an operator that leaked a stop once in a thousand calls would pass a single-shot
    check and still corrupt a run.
    """
    permutation, _, seed = case
    rng = np.random.default_rng(seed)
    mutated = permutation
    for _ in range(25):
        mutated = or_opt_mutation(mutated, MAX_SEGMENT, rng)
    assert sorted(mutated) == sorted(permutation)


# --------------------------------------------------------------------------------------------
# Order crossover
# --------------------------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(case=permutation_pair())
def test_crossover_inherits_the_second_parent_s_relative_order(
    case: tuple[Permutation, Permutation, int],
) -> None:
    """OX's defining property: sequence is inherited, not position.

    Everything outside the run kept from ``first`` must appear in the order ``second`` visits it.
    A positional crossover such as PMX would fail this, and positional inheritance is meaningless
    here — where a stop sits in the chromosome has no interpretation until
    :func:`~src.stage2.split.split` chooses the cuts, and it chooses them afresh every time.
    """
    first, second, seed = case
    child = order_crossover(first, second, np.random.default_rng(seed))
    assert preserves_ox_structure(child, first, second)


def test_the_structure_oracle_rejects_a_child_that_is_not_ox() -> None:
    """The negative control: a test that cannot fail proves nothing.

    Each of these is a permutation of the parents and each breaks OX in a different way, so the
    oracle above has to reject all three or it is not testing anything.
    """
    first = tuple(range(8))
    second = (5, 3, 7, 1, 0, 6, 2, 4)
    reversed_fill = order_crossover(first, tuple(reversed(second)), np.random.default_rng(6))
    assert not preserves_ox_structure(reversed_fill, first, second)
    assert not preserves_ox_structure(tuple(reversed(first)), first, second)
    assert not preserves_ox_structure(second, first, tuple(reversed(second)))


@pytest.mark.parametrize("permutation", [(), (0,)])
def test_crossover_on_a_degenerate_chromosome_is_the_identity(permutation: Permutation) -> None:
    """Nothing to recombine below two stops, and asking is not an error."""
    assert order_crossover(permutation, permutation, np.random.default_rng(0)) == permutation


def test_crossover_of_identical_parents_reproduces_them() -> None:
    """Two copies of one order can only produce that order, whatever the cuts.

    This is why the GA needs a diversity guard: a converged population produces clones for free.
    """
    parent = tuple(range(12))
    rng = np.random.default_rng(5)
    assert all(order_crossover(parent, parent, rng) == parent for _ in range(20))


# --------------------------------------------------------------------------------------------
# Or-opt mutation
# --------------------------------------------------------------------------------------------


def broken_legs(permutation: Permutation) -> int:
    """How many consecutive pairs are out of step, against a chromosome that started sorted.

    ``0, 1, 2, ...`` in order has none. Each break is one leg of the tour the move rearranged, so
    counting them is how a test sees the *shape* of an or-opt move without knowing its draw.
    """
    return sum(1 for a, b in itertools.pairwise(permutation) if b != a + 1)


@pytest.mark.parametrize("permutation", [(), (0,)])
def test_mutation_on_a_degenerate_chromosome_is_the_identity(permutation: Permutation) -> None:
    """One stop admits one ordering. Below two stops the move cannot express anything."""
    assert or_opt_mutation(permutation, MAX_SEGMENT, np.random.default_rng(0)) == permutation


def test_mutation_breaks_at_most_three_legs() -> None:
    """An or-opt move is local: it cuts the run out and splices it in, and nothing else.

    This is the property that separates or-opt from a shuffle. A swap of two distant stops would
    break four legs; a reversal would break two but invert everything between them, which is
    2-opt's move and belongs to the local search where it is evaluated rather than gambled on.
    """
    permutation = tuple(range(40))
    rng = np.random.default_rng(9)
    for _ in range(200):
        assert broken_legs(or_opt_mutation(permutation, MAX_SEGMENT, rng)) <= OR_OPT_MAX_BROKEN_LEGS


def test_mutation_with_a_cap_of_one_relocates_a_single_stop() -> None:
    """The smallest or-opt move: one stop lifted out, the rest left in their original order."""
    permutation = tuple(range(6))
    rng = np.random.default_rng(2)
    for _ in range(50):
        mutated = or_opt_mutation(permutation, 1, rng)
        movers = [
            stop
            for stop in permutation
            if list(rest := [other for other in mutated if other != stop]) == sorted(rest)
        ]
        assert movers, f"{mutated} cannot be reached by relocating one stop"


def test_mutation_can_reach_an_order_crossover_cannot() -> None:
    """Mutation is not decoration: it must actually move a converged population.

    Crossover of identical parents is the identity, so without mutation a converged population is
    a fixed point. This is the test that says mutation escapes it.
    """
    parent = tuple(range(12))
    rng = np.random.default_rng(3)
    assert any(or_opt_mutation(parent, MAX_SEGMENT, rng) != parent for _ in range(50))
