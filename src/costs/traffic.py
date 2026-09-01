"""Time-of-day traffic: the single owner of every travel-time multiplier in the codebase.

The matrices from :mod:`src.costs.matrix` are free-flow. This module is what turns them into
what a vehicle actually experiences, and because ``driver_per_hour`` is time-denominated in the
cost model, it is what makes *when* a route is driven change its cost rather than decorate it.

The load-bearing idea is that the multiplier is integrated **along** a leg, not sampled at its
departure. A leg leaving at 10:45 with 30 minutes of free-flow time does not finish inside the
08–11 band: the first quarter-hour is driven at 1.6× and the remainder at 1.2×, and it lands at
11:24, not at 11:33 (single multiplier 1.6) or 11:21 (single multiplier 1.2). Over a 20-stop
final-mile tour these differences compound into the peak-hour behaviour the whole experiment is
about, and the naive version quietly deletes it.

The integration treats the free-flow duration as a budget of *work* and the multiplier as a
rate: one wall-clock second inside a band of multiplier ``m`` retires ``1/m`` seconds of that
budget. Walking forward band by band and stopping when the budget is exhausted is exact, handles
a leg spanning any number of bands, and is the same arithmetic in both directions.

Traffic multipliers are synthetic and illustrative — they are not calibrated against observed
Mumbai traffic, and the README says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.config import HOURS_PER_DAY, SECONDS_PER_HOUR, TrafficConfig
from src.exceptions import ConfigurationError
from src.units import DurationMatrix, NodeId, Seconds


@dataclass(frozen=True, slots=True)
class TrafficModel:
    """The hourly multiplier schedule, expanded once and reused.

    Holds the 24-entry lookup rather than the band list. The GA evaluates millions of legs per
    run and each one needs a multiplier per band crossed; re-deriving the table from
    :class:`~src.config.TrafficConfig` on every lookup would put a loop over the bands inside the
    innermost loop of the search.
    """

    hourly_multipliers: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.hourly_multipliers) != HOURS_PER_DAY:
            raise ConfigurationError(
                f"hourly_multipliers must have {HOURS_PER_DAY} entries, one per clock hour, "
                f"got {len(self.hourly_multipliers)}"
            )
        if any(multiplier <= 0.0 for multiplier in self.hourly_multipliers):
            raise ConfigurationError("every hourly multiplier must be positive")

    @classmethod
    def from_config(cls, config: TrafficConfig) -> TrafficModel:
        """Build the model from the configured bands. The only sanctioned construction point."""
        return cls(hourly_multipliers=config.hourly_multipliers())

    def multiplier_for_hour(self, hour: int) -> float:
        """The travel-time multiplier in force during clock ``hour``.

        The hour is reduced modulo 24, so an hour index derived from an elapsed-seconds value
        that has run past midnight resolves to the right band instead of raising. Routes
        dispatched late in the day genuinely do cross midnight.
        """
        return self.hourly_multipliers[hour % HOURS_PER_DAY]

    def travel_time_with_traffic(
        self, base_duration_s: Seconds, departure_time_s: Seconds
    ) -> Seconds:
        """Wall-clock time to drive a leg of ``base_duration_s`` leaving at ``departure_time_s``.

        Integrates the multiplier across every band boundary the leg crosses rather than
        sampling it once at departure. See the module docstring for why that distinction is the
        point of this module.

        Args:
            base_duration_s: Free-flow travel time for the leg, from the duration matrix.
                Required to be non-negative, which :class:`~src.costs.matrix.CostMatrices`
                guarantees at construction; this is not re-checked here.
            departure_time_s: Seconds from midnight of the operating day at which the vehicle
                leaves. May exceed 86400 for a route that has crossed midnight.

        Returns:
            The elapsed wall-clock seconds. Always between ``base_duration_s × min(multiplier)``
            and ``base_duration_s × max(multiplier)``.
        """
        remaining_base_s = float(base_duration_s)
        clock_s = float(departure_time_s)

        # Stepping by clock hour rather than by band is the same answer — adjacent hours inside
        # one band share a multiplier — and removes the wrap-around bookkeeping a band walk needs.
        # Legs on this network are minutes long, so the loop runs once or twice.
        while remaining_base_s > 0.0:
            multiplier = self.multiplier_for_hour(int(clock_s // SECONDS_PER_HOUR))
            wall_left_in_hour_s = SECONDS_PER_HOUR - clock_s % SECONDS_PER_HOUR
            base_available_s = wall_left_in_hour_s / multiplier
            if base_available_s >= remaining_base_s:
                clock_s += remaining_base_s * multiplier
                break
            clock_s += wall_left_in_hour_s
            remaining_base_s -= base_available_s

        return Seconds(clock_s - departure_time_s)


def route_timeline(
    nodes: Sequence[NodeId],
    duration_s: DurationMatrix,
    start_time_s: Seconds,
    traffic: TrafficModel,
    service_time_s: Seconds,
) -> tuple[Seconds, ...]:
    """Arrival time at every node of a route, aligned element-wise with ``nodes``.

    This is what Stage 2 checks time windows against, and it is the reason traffic is *cumulative
    along a route* rather than per leg: each leg departs at the arrival time the previous legs
    produced, so an early delay pushes every later stop into a different band. Populating
    :attr:`src.solution.Route.arrival_s` from here is what lets :mod:`src.scoring` stay free of
    any travel-time logic of its own.

    Service time is charged on departure from every interior stop and not at either hub visit:
    the depot's own handling time is not part of a tour's cost model. A useful consequence is
    that ``timeline[-1] - start_time_s`` is the route's full duration including service, which is
    exactly what :attr:`src.solution.Route.duration_s` wants.

    Args:
        nodes: The tour, hub first and hub last, as indices into ``duration_s``.
        duration_s: Free-flow duration matrix for the instance.
        start_time_s: Seconds from midnight at which the vehicle leaves the hub.
        traffic: The multiplier schedule to integrate along the route.
        service_time_s: Dwell at each interior stop, from
            :attr:`src.config.FleetConfig.service_time_per_stop_s`.

    Returns:
        One arrival time per entry of ``nodes``, non-decreasing, starting at ``start_time_s``. A
        route of fewer than two nodes has no legs and yields its start time alone.
    """
    arrivals = [Seconds(float(start_time_s))]
    for position in range(1, len(nodes)):
        # Position 1 departs the hub, which is not serviced; every later leg departs a stop that
        # has just been served.
        dwell_s = 0.0 if position == 1 else float(service_time_s)
        departure_s = Seconds(arrivals[-1] + dwell_s)
        leg_base_s = Seconds(float(duration_s[nodes[position - 1], nodes[position]]))
        arrivals.append(
            Seconds(departure_s + traffic.travel_time_with_traffic(leg_base_s, departure_s))
        )
    return tuple(arrivals)
