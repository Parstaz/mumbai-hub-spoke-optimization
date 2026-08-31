"""Tests for the plan model: what makes a route and a solution structurally well-formed.

Nothing here scores anything — no instance, no rates, no rupees. These are the invariants
``Route`` and ``Solution`` enforce on their own, which is precisely what lets the scorer assume
them. Scoring tests live in ``test_scoring.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.exceptions import InfeasibleSolutionError
from src.solution import Route, Solution
from tests.conftest import make_route

STAGE1_ROUTE = make_route((0, 1, 0), load_kg=75.0, distance_m=5_000.0, duration_s=1_800.0)
STAGE2_ROUTE = make_route((0, 2, 3, 0), load_kg=75.0, distance_m=10_000.0, duration_s=3_600.0)


def _reference_solution() -> Solution:
    """A two-stage plan: one collection tour and one delivery tour."""
    return Solution(stage1_routes=(STAGE1_ROUTE,), stage2_routes=(STAGE2_ROUTE,))


def test_route_derived_properties() -> None:
    """Stops exclude both hub visits; interior nodes are what a stage validates."""
    assert STAGE2_ROUTE.n_stops == 2
    assert STAGE2_ROUTE.interior_nodes == (2, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"nodes": (0, 0)}, id="hub-to-hub-with-no-stops"),
        pytest.param({"nodes": (0, 2, 3)}, id="open-tour"),
        pytest.param({"nodes": (0, 2, 2, 0)}, id="stop-visited-twice"),
        pytest.param({"load_kg": 0.0}, id="empty-vehicle"),
        pytest.param({"duration_s": 0.0}, id="instant-route"),
        pytest.param({"distance_m": -1.0}, id="negative-distance"),
    ],
)
def test_route_rejects_malformed_input(kwargs: dict[str, object]) -> None:
    """Structural guards fire at construction, before a route can reach the scorer."""
    arguments: dict[str, object] = {
        "nodes": (0, 2, 3, 0),
        "load_kg": 75.0,
        "distance_m": 10_000.0,
        "duration_s": 3_600.0,
        **kwargs,
    }
    with pytest.raises(InfeasibleSolutionError):
        make_route(**arguments)  # type: ignore[arg-type]  # parametrised kwargs are mixed types


def test_route_rejects_arrival_times_that_do_not_match_the_stops() -> None:
    """One arrival per node, including both hub visits — otherwise windows misalign."""
    with pytest.raises(InfeasibleSolutionError):
        make_route((0, 2, 3, 0), arrival_s=(28_800.0, 29_000.0, 29_500.0))


def test_route_rejects_arrival_times_that_go_backwards() -> None:
    """A tour cannot travel back in time; this would silently absolve a late delivery."""
    with pytest.raises(InfeasibleSolutionError):
        make_route((0, 2, 3, 0), arrival_s=(28_800.0, 30_000.0, 29_000.0, 31_000.0))


def test_solution_aggregates_both_stages() -> None:
    """Derived totals span both stages, since both consume vehicles, distance and time."""
    solution = _reference_solution()
    assert solution.vehicles_used == 2
    assert solution.total_distance_m == 15_000.0
    assert solution.total_duration_s == 5_400.0
    assert solution.total_load_kg == 150.0


def test_route_is_frozen() -> None:
    """Routes are immutable, so a scored plan cannot drift under the scorer."""
    assert isinstance(STAGE2_ROUTE, Route)
    with pytest.raises(dataclasses.FrozenInstanceError):
        STAGE2_ROUTE.load_kg = 1.0  # type: ignore[misc]  # frozen-ness is the assertion
