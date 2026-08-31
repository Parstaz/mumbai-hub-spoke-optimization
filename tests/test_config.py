"""Tests for the configuration layer.

Config validation is the first line of defence in this codebase: if an impossible run reaches a
solver, the failure surfaces as a strange number rather than an exception. Every guard therefore
gets a test, and every limit gets a test on both sides of it.
"""

import dataclasses

import pytest

from src.config import (
    HOURS_PER_DAY,
    Config,
    CostConfig,
    FleetConfig,
    GAConfig,
    GeoConfig,
    RunConfig,
    ScheduleConfig,
    TrafficBand,
    TrafficConfig,
)
from src.exceptions import ConfigurationError


def test_defaults_construct() -> None:
    """The default composite config is valid, which is what every entry point relies on."""
    config = Config()
    assert config.geo.n_nodes == 16 + 300 + 800
    assert config.cost.variable_per_km == 9.0
    assert config.run.seed == 42


def test_config_is_frozen() -> None:
    """Config objects cannot be mutated after construction, so workers cannot diverge."""
    config = GeoConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_hubs = 4  # type: ignore[misc]  # asserting frozen-ness is the point


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"lat_min": 19.5}, id="inverted-latitude-range"),
        pytest.param({"lon_min": 73.5}, id="inverted-longitude-range"),
        pytest.param({"n_hubs": 0}, id="no-hubs"),
        pytest.param({"n_sources": 0}, id="no-sources"),
        pytest.param({"n_customers": 0}, id="no-customers"),
        pytest.param({"n_density_clusters": 0}, id="no-density-clusters"),
        pytest.param({"clustered_fraction": 1.5}, id="fraction-above-one"),
        pytest.param({"clustered_fraction": -0.1}, id="negative-fraction"),
        pytest.param({"cluster_sigma_deg": 0.0}, id="zero-sigma"),
        pytest.param({"hub_candidate_pool": 4}, id="pool-smaller-than-hub-count"),
        pytest.param({"hub_kmeans_iterations": 0}, id="no-kmeans-iterations"),
    ],
)
def test_geo_config_rejects(kwargs: dict[str, float]) -> None:
    """Each geometry guard rejects its own failure mode."""
    with pytest.raises(ConfigurationError):
        GeoConfig(**kwargs)  # type: ignore[arg-type]  # parametrised kwargs are heterogeneous


def test_fleet_config_shipments_per_vehicle() -> None:
    """Default sizing gives 20 shipments per vehicle, inside the 15–25 design target."""
    assert FleetConfig().shipments_per_vehicle == 20


def test_fleet_config_allows_shipment_exactly_at_capacity() -> None:
    """A single shipment filling a whole vehicle is legal — the boundary is inclusive."""
    fleet = FleetConfig(vehicle_capacity_kg=750.0, shipment_size_kg=750.0)
    assert fleet.shipments_per_vehicle == 1


def test_fleet_config_rejects_shipment_over_capacity() -> None:
    """One unit over capacity is rejected: no vehicle could ever carry the parcel."""
    with pytest.raises(ConfigurationError):
        FleetConfig(vehicle_capacity_kg=750.0, shipment_size_kg=750.001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vehicle_capacity_kg": 0.0},
        {"shipment_size_kg": 0.0},
        {"vehicle_slack_factor": 0.99},
        {"service_time_per_stop_s": -1.0},
    ],
)
def test_fleet_config_rejects(kwargs: dict[str, float]) -> None:
    """Fleet sizing guards."""
    with pytest.raises(ConfigurationError):
        FleetConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variable_per_km": -1.0},
        {"driver_per_hour": -1.0},
        {"fixed_per_vehicle": -1.0},
        {"tw_penalty_per_hour": -1.0},
    ],
)
def test_cost_config_rejects_negative_rates(kwargs: dict[str, float]) -> None:
    """Negative rates are nonsense; zero is allowed so an ablation can switch a term off."""
    with pytest.raises(ConfigurationError):
        CostConfig(**kwargs)


def test_cost_config_allows_zero_rates() -> None:
    """A distance-only ablation is expressible."""
    cost = CostConfig(driver_per_hour=0.0, fixed_per_vehicle=0.0, tw_penalty_per_hour=0.0)
    assert cost.variable_per_km == 9.0


def test_traffic_bands_cover_the_whole_day() -> None:
    """The default schedule partitions the day, including the band that wraps midnight."""
    multipliers = TrafficConfig().hourly_multipliers()
    assert len(multipliers) == HOURS_PER_DAY
    assert all(value > 0.0 for value in multipliers)


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (0, 1.0),
        (7, 1.0),
        (8, 1.6),
        (10, 1.6),
        (11, 1.2),
        (16, 1.2),
        (17, 1.8),
        (20, 1.8),
        (21, 1.0),
        (23, 1.0),
    ],
)
def test_traffic_band_edges(hour: int, expected: float) -> None:
    """Bands are half-open: the first hour of a band belongs to it, the last hour does not.

    These are the exact boundaries a cumulative traffic model crosses mid-route, so an
    off-by-one here would silently mis-price the morning peak.
    """
    assert TrafficConfig().hourly_multipliers()[hour] == expected


def test_traffic_config_rejects_overlapping_bands() -> None:
    """An hour covered twice would make the multiplier ambiguous."""
    with pytest.raises(ConfigurationError):
        TrafficConfig(
            bands=(
                TrafficBand(0, 12, 1.0),
                TrafficBand(10, 24, 1.5),
            )
        )


def test_traffic_config_rejects_gaps() -> None:
    """An uncovered hour would leave a route crossing it with no multiplier at all."""
    with pytest.raises(ConfigurationError):
        TrafficConfig(bands=(TrafficBand(0, 12, 1.0), TrafficBand(13, 24, 1.5)))


def test_traffic_config_rejects_empty_band_list() -> None:
    """A schedule with no bands cannot price anything."""
    with pytest.raises(ConfigurationError):
        TrafficConfig(bands=())


@pytest.mark.parametrize(
    "args",
    [
        (24, 8, 1.0),  # start hour out of range
        (8, 25, 1.0),  # end hour out of range
        (8, 8, 1.0),  # zero-length band
        (8, 11, 0.0),  # non-positive multiplier
    ],
)
def test_traffic_band_rejects(args: tuple[int, int, float]) -> None:
    """Band-level guards."""
    with pytest.raises(ConfigurationError):
        TrafficBand(*args)


def test_wrapping_band_hours() -> None:
    """A band spanning midnight unrolls into evening hours followed by morning hours."""
    assert TrafficBand(21, 8, 1.0).hours == (21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7)


def test_schedule_day_length() -> None:
    """The default operating day is twelve hours."""
    assert ScheduleConfig().day_length_hours == 12.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"day_start_hour": 20.0, "day_end_hour": 8.0},
        {"dispatch_hour": 22.0},
        {"min_window_hours": 5.0, "max_window_hours": 4.0},
        {"min_window_hours": 0.0},
        {"max_window_hours": 13.0},
        {"all_day_fraction": 1.1},
        {"window_granularity_minutes": 0},
    ],
)
def test_schedule_config_rejects(kwargs: dict[str, float]) -> None:
    """Operating-day and window-shape guards."""
    with pytest.raises(ConfigurationError):
        ScheduleConfig(**kwargs)  # type: ignore[arg-type]  # parametrised kwargs are mixed types


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"generations": 0},
        {"tournament_k": 1},
        {"tournament_k": 1000},
        {"crossover_rate": 1.1},
        {"mutation_rate": -0.1},
        {"elitism_count": 150},
        {"local_search_pct": 1.5},
        {"stagnation_limit": 0},
    ],
)
def test_ga_config_rejects(kwargs: dict[str, float]) -> None:
    """GA hyperparameter guards, including elitism that would leave no room for children."""
    with pytest.raises(ConfigurationError):
        GAConfig(**kwargs)  # type: ignore[arg-type]  # parametrised kwargs are mixed types


def test_ga_config_allows_full_elitism_minus_one() -> None:
    """Elitism one below the population is legal; equal to it is not."""
    assert GAConfig(population_size=10, elitism_count=9).elitism_count == 9
    with pytest.raises(ConfigurationError):
        GAConfig(population_size=10, elitism_count=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"osrm_url": "   "},
        {"circuity_factor": 0.99},
    ],
)
def test_run_config_rejects(kwargs: dict[str, object]) -> None:
    """Run-level guards, including a circuity factor that would shorten road distance."""
    with pytest.raises(ConfigurationError):
        RunConfig(**kwargs)  # type: ignore[arg-type]  # parametrised kwargs are mixed types


def test_run_config_allows_unit_circuity() -> None:
    """A circuity factor of exactly one means straight-line roads: legal, if optimistic."""
    assert RunConfig(circuity_factor=1.0).circuity_factor == 1.0
