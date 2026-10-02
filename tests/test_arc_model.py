"""Tests for the shared OR-Tools arc model.

These functions used to be private to :mod:`src.stage1.cvrp` and were covered only through a full
CVRP solve. That is no longer good enough: :mod:`src.stage2.ortools_reference` prices its arcs with
the same code, so an error here would move *both* solvers' answers in the same direction and the
step 8 gap — the whole deliverable — would be the one number that failed to notice.

So the arithmetic is asserted directly, against figures worked out by hand from the configured
rates rather than against whatever the code happens to return.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.arc_model import (
    ArcRates,
    arc_cost_milli_inr,
    arc_rates,
    capacity_grams,
    demand_grams,
    search_parameters,
    vehicle_count,
)
from src.config import CostConfig, FleetConfig, TrafficConfig
from src.costs.traffic import TrafficModel

RATES = ArcRates(
    variable_per_km=9.0, driver_per_hour=95.0, fixed_per_vehicle=1000.0, traffic_multiplier=1.0
)


def test_the_arc_is_distance_money_plus_time_money() -> None:
    """Worked by hand: 10 km at ₹9/km is ₹90, 1 h at ₹95/h is ₹95, so 185,000 milli-rupees."""
    distance_m = np.array([[0.0, 10_000.0], [10_000.0, 0.0]])
    duration_s = np.array([[0.0, 3600.0], [3600.0, 0.0]])

    cost = arc_cost_milli_inr(distance_m, duration_s, RATES)

    assert cost[0, 1] == 90_000 + 95_000
    assert cost[0, 0] == 0


def test_the_traffic_multiplier_scales_time_money_and_leaves_distance_money_alone() -> None:
    """The coupling §1.2 calls load-bearing: if traffic scaled both, it could not change an argmin.

    Same 10 km / 1 h arc at ×1.6: distance money stays ₹90, driver money goes ₹95 → ₹152.
    """
    distance_m = np.array([[0.0, 10_000.0], [10_000.0, 0.0]])
    duration_s = np.array([[0.0, 3600.0], [3600.0, 0.0]])
    peak = ArcRates(
        variable_per_km=9.0,
        driver_per_hour=95.0,
        fixed_per_vehicle=1000.0,
        traffic_multiplier=1.6,
    )

    cost = arc_cost_milli_inr(distance_m, duration_s, peak)

    assert cost[0, 1] == 90_000 + 152_000


def test_arc_rates_reads_the_multiplier_in_force_at_dispatch() -> None:
    """08:00 sits in the 08–11 band, so an 08:00 dispatch prices driver time at ×1.6."""
    traffic = TrafficModel.from_config(TrafficConfig())

    rates = arc_rates(CostConfig(), traffic, dispatch_hour=8.0)

    assert rates.traffic_multiplier == 1.6
    assert rates.variable_per_km == 9.0
    assert rates.fixed_per_vehicle == 1000.0


def test_an_overnight_dispatch_is_priced_at_the_free_flow_multiplier() -> None:
    """The 21–08 band wraps midnight, which is the band a lookup is most likely to get wrong."""
    traffic = TrafficModel.from_config(TrafficConfig())

    assert arc_rates(CostConfig(), traffic, dispatch_hour=23.0).traffic_multiplier == 1.0


@pytest.mark.parametrize(
    ("total_kg", "expected"),
    [
        (0.0, 1),  # a hub with nothing waiting still gets a vehicle to be offered
        (1.0, 1),
        (750.0, 1),  # exactly one vehicle-load: the boundary a float sum can trip
        (750.000001, 1),  # inside CAPACITY_TOLERANCE_KG, so still one vehicle
        (751.0, 2),
        (1500.0, 2),  # two whole vehicle-loads, and this fixture applies no slack
    ],
)
def test_vehicle_count_is_the_mass_floor_times_slack(total_kg: float, expected: int) -> None:
    """The floor is mass, never stop count — and the tolerance keeps an exact load off the cliff.

    Slack is pinned to 1.0 here so these cases isolate the floor itself; the default 1.15 is
    exercised by the test below.
    """
    fleet = FleetConfig(vehicle_slack_factor=1.0)

    assert vehicle_count(total_kg, fleet) == expected


def test_the_slack_factor_rounds_up_rather_than_truncating() -> None:
    """Two vehicles at the default 1.15 slack is 2.3, which must buy three rather than two.

    Rounding down would make the slack factor a no-op for every hub below nine vehicles, which is
    most of them — the kind of bug that leaves the configured value looking honoured.
    """
    assert vehicle_count(1500.0, FleetConfig(vehicle_slack_factor=1.15)) == 3
    assert vehicle_count(750.0, FleetConfig(vehicle_slack_factor=1.15)) == 2


def test_demands_are_scaled_to_integer_grams_with_a_depot_zero() -> None:
    """37.5 kg is not a whole number of kilograms, which is the reason the scale is grams."""
    assert demand_grams(np.array([37.5, 75.0])) == (0, 37_500, 75_000)
    assert capacity_grams(750.0) == 750_000


def test_an_empty_hub_still_gets_its_depot_entry() -> None:
    """A model with no stops is still a model with a depot, and OR-Tools indexes from it."""
    assert demand_grams(np.array([])) == (0,)


def test_the_solution_limit_is_set_only_when_it_is_positive() -> None:
    """Zero must leave the pre-set sentinel alone rather than writing a literal zero.

    ``solution_limit`` is a proto3 scalar with no presence, so "unset" is not observable — what is
    observable is that ``DefaultRoutingSearchParameters()`` pre-fills it with ``int64`` max. The
    protobuf default for the field is 0, so a "simplification" that always assigned our zero would
    cap the search at zero improved solutions and read as an infeasible instance. Asserting the
    sentinel survives is what catches that.
    """
    limited = search_parameters(time_limit_s=1.0, solution_limit=1)
    unlimited = search_parameters(time_limit_s=1.0, solution_limit=0)

    assert limited.solution_limit == 1
    assert unlimited.solution_limit == 2**63 - 1


def test_the_time_limit_is_carried_through_in_milliseconds() -> None:
    """A fractional budget must survive the conversion; the Stage 2 reference passes small ones."""
    parameters = search_parameters(time_limit_s=0.25, solution_limit=0)

    assert parameters.time_limit.ToMilliseconds() == 250
