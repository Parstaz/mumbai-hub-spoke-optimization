"""Tests for the capacity-balanced hub assignment.

The flow model's guarantee is the thing under test, not its distance. Three properties are
asserted repeatedly because each is a way the model could be wrong while still looking
plausible: that **every stop lands wholly at one hub** (the no-split invariant, which unit arc
capacity is supposed to make structurally impossible), that the cap is never exceeded, and that
subject to the cap the assignment is genuinely cheapest — checked here against brute-force
enumeration on a case small enough to enumerate.

The last test in this module is not about assignment at all. It asserts the *import direction*:
that nothing the greedy baseline depends on reaches into ``src/stage1`` or ``src/stage2``. The
baseline is the control every improvement is quoted against, so if it drew code from the module
being optimized then tuning the treatment would move the control's column with it — silently, and
in the direction that flatters the treatment.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.exceptions import InfeasibleInstanceError
from src.stage1.assignment import (
    capacity_balanced,
    max_stops_per_hub,
    unconstrained,
)
from src.units import DistanceMatrix, NodeArray
from src.workload import nearest_hub, node_array
from tests.closure import SRC_ROOT, reachable_modules

TREATMENT_PACKAGES = ("src.stage1", "src.stage2")
"""The modules being optimized. The control must not reach into either."""


def nodes(count: int, offset: int = 0) -> NodeArray:
    """Node ids ``offset .. offset + count - 1``."""
    return node_array(range(offset, offset + count))  # type: ignore[arg-type]  # ints are NodeIds


def matrix_from_rows(rows: list[list[float]]) -> DistanceMatrix:
    """Embed a hub-by-stop distance block into a full square matrix.

    ``rows[h][s]`` is the drive from hub ``h`` to stop ``s``. Hubs occupy the low node ids and
    stops follow, matching the instance's flat layout. Everything not named is left at zero,
    which no assignment reads.
    """
    n_hubs, n_stops = len(rows), len(rows[0])
    matrix: DistanceMatrix = np.zeros((n_hubs + n_stops, n_hubs + n_stops), dtype=np.float64)
    for hub_id, row in enumerate(rows):
        for stop_index, metres in enumerate(row):
            matrix[hub_id, n_hubs + stop_index] = metres
    return matrix


def assign(rows: list[list[float]], cap: int) -> NodeArray:
    """Run the balanced assignment over a hub-by-stop block."""
    n_hubs, n_stops = len(rows), len(rows[0])
    return capacity_balanced(nodes(n_hubs), nodes(n_stops, n_hubs), matrix_from_rows(rows), cap)


def total_metres(rows: list[list[float]], hub_of_stop: NodeArray) -> float:
    """The hub-to-stop distance the assignment commits to."""
    return sum(rows[int(hub_id)][stop] for stop, hub_id in enumerate(hub_of_stop))


def cheapest_capped_assignment(rows: list[list[float]], cap: int) -> float:
    """Brute-force the optimum by enumerating every capped assignment.

    Only viable for a handful of stops, which is the point: it turns "the flow found a good
    assignment" into "the flow found *the* assignment" on a case a reader can check by hand.
    """
    n_hubs, n_stops = len(rows), len(rows[0])
    best = float("inf")
    for candidate in itertools.product(range(n_hubs), repeat=n_stops):
        counts = np.bincount(candidate, minlength=n_hubs)
        if counts.max() > cap:
            continue
        best = min(best, sum(rows[hub][stop] for stop, hub in enumerate(candidate)))
    return best


def test_unconstrained_is_exactly_nearest_hub() -> None:
    """The adapter must not become a second implementation of the control's rule."""
    rows = [[10.0, 90.0, 50.0], [80.0, 20.0, 40.0]]
    matrix = matrix_from_rows(rows)
    hub_nodes, stop_nodes = nodes(2), nodes(3, 2)
    assert unconstrained(hub_nodes, stop_nodes, matrix, 1).tolist() == (
        nearest_hub(hub_nodes, stop_nodes, matrix).tolist()
    )


def test_a_cap_that_cannot_bind_reproduces_nearest_hub() -> None:
    """With room for every stop at its nearest hub, the two strategies must agree.

    This is what makes the ablation interpretable: the balanced strategy is not a different
    objective, it is the same objective under a constraint. Where the constraint is slack, the
    columns have to coincide.
    """
    rows = [[10.0, 90.0, 50.0], [80.0, 20.0, 40.0]]
    assert (
        assign(rows, cap=3).tolist()
        == unconstrained(nodes(2), nodes(3, 2), matrix_from_rows(rows), 3).tolist()
    )


def test_the_cap_pushes_a_stop_off_its_nearest_hub() -> None:
    """All three stops prefer hub 0; a cap of 2 must send the cheapest-to-move one to hub 1.

    Stop 2 is the one to move: relocating it costs 30 m against 70 m for stop 0 and 60 m for
    stop 1. A model that moved the *farthest* stop, or an arbitrary one, would still respect the
    cap and would still look balanced.
    """
    rows = [[10.0, 20.0, 30.0], [80.0, 80.0, 60.0]]
    assert assign(rows, cap=2).tolist() == [0, 0, 1]


def test_the_assignment_is_the_cheapest_one_the_cap_allows() -> None:
    """Checked against brute-force enumeration over every capped assignment."""
    rows = [
        [10.0, 20.0, 30.0, 40.0, 95.0],
        [50.0, 15.0, 70.0, 25.0, 35.0],
        [90.0, 80.0, 12.0, 60.0, 55.0],
    ]
    for cap in (2, 3, 5):
        found = total_metres(rows, assign(rows, cap))
        assert found == pytest.approx(cheapest_capped_assignment(rows, cap))


def test_balancing_never_beats_nearest_hub_on_distance() -> None:
    """Nearest-hub is the uncapped optimum, so the cap can only cost radial distance.

    The whole premise of Stage 1 is that the cap buys back more on *tour* distance than it gives
    up here. If this assertion ever inverted, the flow would be solving a different problem than
    the one documented.
    """
    rows = [[10.0, 20.0, 30.0], [80.0, 80.0, 60.0]]
    capped = total_metres(rows, assign(rows, cap=2))
    uncapped = total_metres(rows, unconstrained(nodes(2), nodes(3, 2), matrix_from_rows(rows), 2))
    assert capped >= uncapped


def test_every_stop_is_assigned_wholly_to_exactly_one_hub() -> None:
    """The no-split invariant. Unit supply and unit arc capacity make this structural.

    A flow that split a stop would show up here as a hub id outside range or a length mismatch,
    because the reader takes the first arc carrying flow out of each stop.
    """
    rows = [[float(10 * hub + stop) for stop in range(9)] for hub in range(4)]
    hub_of_stop = assign(rows, cap=3)
    assert len(hub_of_stop) == 9
    assert set(hub_of_stop.tolist()) <= set(range(4))
    assert np.bincount(hub_of_stop, minlength=4).sum() == 9


def test_the_cap_is_respected_exactly() -> None:
    """Nine stops, four hubs, cap 3: at most three per hub, and none dropped."""
    rows = [[float(10 * hub + stop) for stop in range(9)] for hub in range(4)]
    counts = np.bincount(assign(rows, cap=3), minlength=4)
    assert counts.max() <= 3
    assert counts.sum() == 9


def test_a_cap_that_exactly_absorbs_every_stop_is_feasible() -> None:
    """The boundary: total capacity equal to the number of stops leaves no slack but is legal."""
    rows = [[float(10 * hub + stop) for stop in range(6)] for hub in range(3)]
    counts = np.bincount(assign(rows, cap=2), minlength=3)
    assert counts.tolist() == [2, 2, 2]


def test_one_stop_more_than_the_caps_can_absorb_is_rejected() -> None:
    """One over the boundary must name the knob that fixes it, not fail inside NetworkX."""
    rows = [[float(10 * hub + stop) for stop in range(7)] for hub in range(3)]
    with pytest.raises(InfeasibleInstanceError, match="hub_balance_slack"):
        assign(rows, cap=2)


def test_a_single_hub_takes_everything_it_is_capped_for() -> None:
    """The degenerate instance: one hub, and the assignment is forced."""
    assert assign([[5.0, 6.0]], cap=2).tolist() == [0, 0]


def test_the_assignment_is_reproducible() -> None:
    """Two calls on one input must be identical, or multi-seed runs are not comparable."""
    rows = [[float((7 * hub + 13 * stop) % 50) for stop in range(8)] for hub in range(3)]
    assert assign(rows, cap=3).tolist() == assign(rows, cap=3).tolist()


@pytest.mark.parametrize(
    ("n_stops", "n_hubs", "slack", "expected"),
    [
        (285, 16, 1.25, 23),  # the default instance
        (300, 16, 1.0, 19),  # an even split, rounded up off 18.75
        (16, 16, 1.0, 1),  # exactly one each
        (3, 2, 2.0, 3),  # slack wide enough that the cap cannot bind
    ],
)
def test_max_stops_per_hub_rounds_the_even_share_up(
    n_stops: int, n_hubs: int, slack: float, expected: int
) -> None:
    """Rounding up is what keeps the cap feasible for any slack at or above 1.0."""
    assert max_stops_per_hub(n_stops, n_hubs, slack) == expected


def test_nothing_the_baseline_depends_on_imports_the_optimized_stages() -> None:
    """The control must be independently frozen — transitively, not just at the top level.

    Walked as a closure rather than checked one file deep on purpose: the coupling this forbids
    would arrive by someone adding a Stage 1 import to ``src/workload.py`` or ``src/tour.py``,
    not to ``greedy.py`` where it would be obvious. If this fails, the baseline column can be
    moved by editing the module it is supposed to be measuring.
    """
    reached = reachable_modules((SRC_ROOT / "baseline" / "greedy.py",))

    offenders = sorted(m for m in reached if m.startswith(TREATMENT_PACKAGES))
    assert not offenders, (
        f"src/baseline reaches the optimized stages through {offenders}; the control must not "
        f"depend on the treatment"
    )
    # Guard against the walk silently finding nothing and passing vacuously.
    assert "src.workload" in reached
    assert "src.tour" in reached
