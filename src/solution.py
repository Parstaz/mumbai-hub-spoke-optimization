"""The plan: one vehicle tour, and a full two-stage solution built from tours.

A :class:`Route` carries physical facts — stops, load, distance, duration, arrival times —
produced by the routing layer, which owns traffic. It deliberately does not cache its own cost:
money is derived on demand by :mod:`src.scoring`, and a cost field here would be a second
scoring path with a stale-value bug attached.

Validation in this module is structural and instance-free — a tour closes at its hub, visits no
stop twice, and has one arrival time per stop. Anything that needs the instance to check
(capacity, whether a stop is the right kind for its stage, whether the plan leaves work undone)
lives in :mod:`src.scoring`, where the instance is in scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MIN_ROUTE_NODES
from src.exceptions import InfeasibleSolutionError
from src.units import Metres, NodeId, Seconds


def _require(condition: bool, message: str) -> None:
    """Raise :class:`InfeasibleSolutionError` unless ``condition`` holds."""
    if not condition:
        raise InfeasibleSolutionError(message)


@dataclass(frozen=True, slots=True)
class Route:
    """One vehicle tour: ``hub -> stops -> hub``, with its physical outcome.

    ``arrival_s`` is aligned element-wise with ``nodes`` and holds seconds from midnight,
    already traffic-adjusted by the cost layer. Carrying arrivals rather than recomputing them
    is what lets scoring stay free of any travel-time logic of its own.
    """

    hub_id: int
    nodes: tuple[NodeId, ...]
    load_kg: float
    distance_m: Metres
    duration_s: Seconds
    arrival_s: tuple[Seconds, ...]

    def __post_init__(self) -> None:
        _require(
            len(self.nodes) >= MIN_ROUTE_NODES, "a route must be hub -> at least one stop -> hub"
        )
        _require(self.nodes[0] == self.nodes[-1], "a route must start and end at its hub")
        _require(
            len(set(self.interior_nodes)) == len(self.interior_nodes),
            f"route from hub {self.hub_id} visits a stop twice",
        )
        _require(self.load_kg > 0.0, "a deployed vehicle must carry something")
        _require(self.distance_m >= 0.0, "distance_m must be non-negative")
        _require(self.duration_s > 0.0, "a route that takes no time is not a route")
        _require(
            len(self.arrival_s) == len(self.nodes),
            "arrival_s must have one entry per node, including both hub visits",
        )
        _require(
            all(a <= b for a, b in zip(self.arrival_s, self.arrival_s[1:], strict=False)),
            "arrival times must be non-decreasing along the route",
        )

    @property
    def interior_nodes(self) -> tuple[NodeId, ...]:
        """The stops, excluding the opening and closing hub visits."""
        return self.nodes[1:-1]

    @property
    def n_stops(self) -> int:
        """Number of served stops on this tour."""
        return len(self.nodes) - 2


@dataclass(frozen=True, slots=True)
class Solution:
    """A full plan: inbound consolidation tours and final-mile delivery tours.

    The two stages are held separately because they answer different questions — Stage 1 moves
    volume into hubs, Stage 2 moves it to customers — and because the headline KPI counts drops,
    which only Stage 2 makes.
    """

    stage1_routes: tuple[Route, ...]
    stage2_routes: tuple[Route, ...]

    @property
    def all_routes(self) -> tuple[Route, ...]:
        """Every tour in the plan, both stages."""
        return self.stage1_routes + self.stage2_routes

    @property
    def vehicles_used(self) -> int:
        """Vehicle-days deployed.

        One tour is charged as one vehicle-day: the fixed cost is levied per tour. A real
        operation might run a Stage 1 tour and a Stage 2 tour on the same asset, so this is an
        upper bound — applied identically to the baseline and the optimized plan, which is what
        keeps the comparison honest.
        """
        return len(self.all_routes)

    @property
    def total_distance_m(self) -> Metres:
        """Total distance driven across both stages."""
        return Metres(sum(route.distance_m for route in self.all_routes))

    @property
    def total_duration_s(self) -> Seconds:
        """Total vehicle time across both stages, including service time."""
        return Seconds(sum(route.duration_s for route in self.all_routes))

    @property
    def total_load_kg(self) -> float:
        """Total mass moved across both stages."""
        return sum(route.load_kg for route in self.all_routes)
