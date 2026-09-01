"""Tests for the time-of-day traffic model.

The expected values below are worked out by hand from the band schedule, never read back from
the implementation. That is deliberate: the canonical error in a traffic model is to sample the
multiplier once at departure instead of integrating it along the leg, and that error produces a
perfectly plausible number. Only an independently derived expectation catches it.

Worked example, used by several tests below — a leg leaving at 10:45 with 30 minutes of
free-flow time, under bands ``08–11 → 1.6`` and ``11–17 → 1.2``:

* 10:45 → 11:00 is 900 wall-clock seconds at 1.6×, which retires 900 / 1.6 = 562.5 base seconds.
* 1800 - 562.5 = 1237.5 base seconds remain, driven at 1.2× for 1485 wall-clock seconds.
* Total 2385 s, arriving 11:24:45.

Sampling 1.6 at departure would give 2880 s (11:33); sampling 1.2 would give 2160 s (11:21).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config import SECONDS_PER_HOUR, TrafficBand, TrafficConfig
from src.costs.traffic import TrafficModel, route_timeline
from src.exceptions import ConfigurationError
from src.solution import Route
from src.units import DurationMatrix, Metres, NodeId, Seconds

MODEL = TrafficModel.from_config(TrafficConfig())
"""The default schedule: 08–11 → 1.6, 11–17 → 1.2, 17–21 → 1.8, 21–08 → 1.0."""

FLAT_MODEL = TrafficModel(hourly_multipliers=(1.0,) * 24)
"""A schedule with no time-of-day effect at all, for isolating what traffic contributes."""


def at(hour: float) -> Seconds:
    """Seconds from midnight for a fractional clock hour, e.g. ``at(10.75)`` is 10:45."""
    return Seconds(hour * SECONDS_PER_HOUR)


# --------------------------------------------------------------------------------------------
# Band lookup
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        pytest.param(7, 1.0, id="last-hour-of-the-overnight-band"),
        pytest.param(8, 1.6, id="morning-peak-opens"),
        pytest.param(10, 1.6, id="inside-the-morning-peak"),
        pytest.param(11, 1.2, id="midday-opens-the-hour-the-peak-closes"),
        pytest.param(16, 1.2, id="last-hour-of-midday"),
        pytest.param(17, 1.8, id="evening-peak-opens"),
        pytest.param(20, 1.8, id="last-hour-of-the-evening-peak"),
        pytest.param(21, 1.0, id="overnight-opens"),
        pytest.param(23, 1.0, id="before-midnight"),
        pytest.param(0, 1.0, id="after-midnight"),
    ],
)
def test_multiplier_for_hour_at_every_band_edge(hour: int, expected: float) -> None:
    """Bands are half-open ``[start, end)``: the hour a band ends belongs to the next one."""
    assert MODEL.multiplier_for_hour(hour) == expected


@pytest.mark.parametrize(("hour", "expected"), [(24, 1.0), (30, 1.0), (32, 1.6), (48, 1.0)])
def test_multiplier_for_hour_wraps_past_midnight(hour: int, expected: float) -> None:
    """An hour index taken from an elapsed-seconds clock that has passed midnight still resolves."""
    assert MODEL.multiplier_for_hour(hour) == expected


def test_traffic_model_rejects_a_wrong_sized_table() -> None:
    """The lookup is indexed by clock hour, so it must have exactly 24 entries."""
    with pytest.raises(ConfigurationError, match="24 entries"):
        TrafficModel(hourly_multipliers=(1.0,) * 23)


@pytest.mark.parametrize("multiplier", [0.0, -1.0])
def test_traffic_model_rejects_a_non_positive_multiplier(multiplier: float) -> None:
    """A zero multiplier would make a leg take no time; a negative one would run the clock back."""
    with pytest.raises(ConfigurationError, match="positive"):
        TrafficModel(hourly_multipliers=(multiplier,) + (1.0,) * 23)


def test_from_config_expands_the_configured_bands() -> None:
    """A non-default schedule must reach the model, not just the default one."""
    config = TrafficConfig(
        bands=(
            TrafficBand(start_hour=6, end_hour=18, multiplier=2.0),
            TrafficBand(start_hour=18, end_hour=6, multiplier=0.5),
        )
    )
    model = TrafficModel.from_config(config)
    assert model.multiplier_for_hour(6) == 2.0
    assert model.multiplier_for_hour(18) == 0.5
    assert model.multiplier_for_hour(5) == 0.5


# --------------------------------------------------------------------------------------------
# Cumulative travel time — the point of the module
# --------------------------------------------------------------------------------------------


def test_leg_inside_one_band_is_a_flat_scaling() -> None:
    """The simple case has to stay simple: 10 minutes at 12:00 is 10 × 1.2."""
    assert MODEL.travel_time_with_traffic(Seconds(600.0), at(12.0)) == pytest.approx(720.0)


def test_leg_spanning_a_band_boundary_blends_both_multipliers() -> None:
    """10:45 + 30 min free-flow lands at 11:24:45, not 11:33 (1.6) or 11:21 (1.2).

    The worked example from the module docstring, and the regression test for sampling the
    multiplier at departure instead of integrating along the leg.
    """
    elapsed = MODEL.travel_time_with_traffic(Seconds(1800.0), at(10.75))

    assert elapsed == pytest.approx(2385.0)
    assert at(10.75) + elapsed == pytest.approx(at(11.0) + 1485.0)
    # Both naive answers are excluded, in both directions.
    assert elapsed != pytest.approx(1800.0 * 1.6)
    assert elapsed != pytest.approx(1800.0 * 1.2)


def test_leg_ending_exactly_on_a_band_boundary_uses_only_the_first_band() -> None:
    """562.5 base seconds at 1.6 is exactly the 900 s from 10:45 to 11:00 — the closed edge."""
    elapsed = MODEL.travel_time_with_traffic(Seconds(562.5), at(10.75))
    assert elapsed == pytest.approx(900.0)


def test_leg_departing_exactly_on_a_band_boundary_uses_only_the_second_band() -> None:
    """Departing at 11:00:00 sharp is midday traffic, not a blend — the open edge."""
    elapsed = MODEL.travel_time_with_traffic(Seconds(600.0), at(11.0))
    assert elapsed == pytest.approx(720.0)


def test_leg_spanning_three_bands() -> None:
    """A leg crossing 17:00 and 21:00 must pick up all three multipliers in order.

    16:30 → 17:00: 1800 wall at 1.2 retires 1500 base.
    17:00 → 21:00: 14400 wall at 1.8 retires 8000 base, for 9500 so far.
    500 base remain, driven at 1.0 for 500 wall. Total 1800 + 14400 + 500 = 16700 s.
    """
    elapsed = MODEL.travel_time_with_traffic(Seconds(10_000.0), at(16.5))
    assert elapsed == pytest.approx(16_700.0)


def test_leg_crossing_midnight() -> None:
    """23:30 → 08:00 is 30600 wall at 1.0; the last 3600 base seconds are morning peak at 1.6.

    Total 30600 + 5760 = 36360 s. This is the test that fails if the hour lookup does not wrap.
    """
    elapsed = MODEL.travel_time_with_traffic(Seconds(34_200.0), at(23.5))
    assert elapsed == pytest.approx(36_360.0)


def test_zero_length_leg_takes_no_time() -> None:
    """The degenerate boundary: a self-loop in the matrix costs nothing at any hour."""
    assert MODEL.travel_time_with_traffic(Seconds(0.0), at(8.0)) == 0.0


def test_a_flat_schedule_is_the_identity() -> None:
    """With every multiplier at 1.0 the model must return the free-flow duration untouched."""
    assert FLAT_MODEL.travel_time_with_traffic(Seconds(4321.0), at(9.0)) == pytest.approx(4321.0)


@settings(max_examples=200)
@given(
    base_duration_s=st.floats(min_value=0.0, max_value=50_000.0),
    departure_time_s=st.floats(min_value=0.0, max_value=172_800.0),
)
def test_travel_time_stays_between_the_extreme_multipliers(
    base_duration_s: float, departure_time_s: float
) -> None:
    """However many bands a leg crosses, its blend is a weighted mean of the multipliers."""
    elapsed = MODEL.travel_time_with_traffic(Seconds(base_duration_s), Seconds(departure_time_s))
    slowest = max(MODEL.hourly_multipliers)
    fastest = min(MODEL.hourly_multipliers)
    assert base_duration_s * fastest - 1e-6 <= elapsed <= base_duration_s * slowest + 1e-6


@settings(max_examples=200)
@given(
    shorter_s=st.floats(min_value=0.0, max_value=20_000.0),
    extra_s=st.floats(min_value=0.0, max_value=20_000.0),
    departure_time_s=st.floats(min_value=0.0, max_value=86_400.0),
)
def test_travel_time_is_monotone_in_the_free_flow_duration(
    shorter_s: float, extra_s: float, departure_time_s: float
) -> None:
    """A longer leg from the same place at the same time can never arrive earlier."""
    departure = Seconds(departure_time_s)
    shorter = MODEL.travel_time_with_traffic(Seconds(shorter_s), departure)
    longer = MODEL.travel_time_with_traffic(Seconds(shorter_s + extra_s), departure)
    assert longer >= shorter - 1e-6


# --------------------------------------------------------------------------------------------
# route_timeline
# --------------------------------------------------------------------------------------------


def duration_matrix(leg_s: float, n_nodes: int = 3) -> DurationMatrix:
    """A matrix where every leg between distinct nodes takes ``leg_s`` free-flow seconds."""
    matrix = np.full((n_nodes, n_nodes), leg_s, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def nodes(*indices: int) -> tuple[NodeId, ...]:
    """Build a node sequence."""
    return tuple(NodeId(index) for index in indices)


def test_route_timeline_charges_traffic_and_service_at_the_right_stops() -> None:
    """A hand-computed two-leg tour: hub → customer → hub, dispatched into the morning peak.

    Leaving the hub at 08:00, a 600 s free-flow leg at 1.6 arrives at 08:16. The 300 s of
    service is charged on *departure*, so the return leg leaves at 08:21, is still inside the
    peak, and takes another 960 s. Total 2220 s = 960 + 300 + 960.
    """
    timeline = route_timeline(
        nodes(0, 1, 0), duration_matrix(600.0), at(8.0), MODEL, Seconds(300.0)
    )

    assert timeline == pytest.approx((28_800.0, 29_760.0, 31_020.0))
    assert timeline[-1] - at(8.0) == pytest.approx(2220.0)


def test_route_timeline_start_is_the_dispatch_time_and_the_hub_is_not_serviced() -> None:
    """The first entry is the departure from the hub, with no dwell charged there."""
    timeline = route_timeline(
        nodes(0, 1, 2, 0), duration_matrix(600.0, 3), at(8.0), FLAT_MODEL, Seconds(300.0)
    )
    assert timeline[0] == at(8.0)
    # Three legs of 600 s and two interior stops of 300 s: the closing hub adds no service.
    assert timeline[-1] - timeline[0] == pytest.approx(3 * 600.0 + 2 * 300.0)


def test_route_timeline_is_non_decreasing_and_fits_the_route_contract() -> None:
    """The output is exactly what ``Route.arrival_s`` requires, which is its only consumer."""
    tour = nodes(0, 1, 2, 0)
    timeline = route_timeline(tour, duration_matrix(900.0, 3), at(17.0), MODEL, Seconds(300.0))

    assert all(a <= b for a, b in pairwise(timeline))
    route = Route(
        hub_id=0,
        nodes=tour,
        load_kg=75.0,
        distance_m=Metres(10_000.0),
        duration_s=Seconds(timeline[-1] - timeline[0]),
        arrival_s=timeline,
    )
    assert len(route.arrival_s) == len(route.nodes)


def test_route_timeline_accumulates_traffic_along_the_route() -> None:
    """Later stops are pushed into later bands, which is what "cumulative" means here.

    Dispatching at 10:00 with 20-minute legs and no service time: the first leg is 1200 × 1.6 =
    1920 wall seconds, wholly inside the morning peak, arriving 10:32. The second departs 10:32
    and straddles 11:00 — 1680 wall at 1.6 retires 1050 base, and the remaining 150 base run at
    1.2 for 180 wall, giving 1860. A model taking one multiplier for the whole tour would report
    two identical legs.
    """
    timeline = route_timeline(
        nodes(0, 1, 0), duration_matrix(1200.0), at(10.0), MODEL, Seconds(0.0)
    )
    first_leg = timeline[1] - timeline[0]
    second_leg = timeline[2] - timeline[1]

    assert first_leg == pytest.approx(1200.0 * 1.6)
    assert second_leg == pytest.approx(1860.0)
    assert 1200.0 * 1.2 < second_leg < first_leg


def test_route_timeline_of_a_single_leg() -> None:
    """The shortest sequence with any travel in it."""
    timeline = route_timeline(nodes(0, 1), duration_matrix(600.0), at(12.0), MODEL, Seconds(300.0))
    assert timeline == pytest.approx((43_200.0, 43_920.0))


def test_route_timeline_without_legs_is_just_the_start_time() -> None:
    """The empty boundary: no legs means no travel, and the caller gets its own clock back."""
    assert route_timeline(nodes(0), duration_matrix(600.0), at(8.0), MODEL, Seconds(300.0)) == (
        at(8.0),
    )
    assert route_timeline((), duration_matrix(600.0), at(8.0), MODEL, Seconds(300.0)) == (at(8.0),)
