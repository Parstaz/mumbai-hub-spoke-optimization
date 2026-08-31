"""Tests for instance generation.

Two properties matter beyond "it runs": the generator is reproducible from its seed, and it
produces the *structure* the experiment depends on — hubs that are spread rather than
overlapping, and demand that is clustered rather than uniform. Both are asserted behaviourally,
against outcomes rather than against the sampling calls that produce them.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pytest

from src.config import (
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    FleetConfig,
    GeoConfig,
    ScheduleConfig,
)
from src.data.generate import generate_instance
from src.data.instance import Instance, Source
from src.exceptions import InstanceError
from tests.conftest import SMALL_GEO

DEFAULT_FLEET = FleetConfig()
DEFAULT_SCHEDULE = ScheduleConfig()


def _generate(geo: GeoConfig = SMALL_GEO, seed: int = 7) -> Instance:
    """Generate with default fleet and schedule, varying only what a test cares about."""
    return generate_instance(geo, DEFAULT_FLEET, DEFAULT_SCHEDULE, seed)


def _isotropic_xy(coords: npt.NDArray[np.float64], geo: GeoConfig) -> npt.NDArray[np.float64]:
    """Project lat/lon degrees into an isotropic plane so separations are comparable."""
    scaled = coords.copy()
    scaled[:, 1] *= math.cos(math.radians(0.5 * (geo.lat_min + geo.lat_max)))
    return scaled


def _nearest_neighbour_distances(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distance from each point to its closest other point."""
    deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    np.fill_diagonal(distances, np.inf)
    return np.asarray(distances.min(axis=1))


def test_same_seed_generates_an_identical_instance() -> None:
    """Reproducibility is the point of threading the seed through a single generator."""
    assert _generate(seed=11) == _generate(seed=11)


def test_different_seeds_generate_different_instances() -> None:
    """Distinct seeds must actually move the geometry, or multi-seed evaluation is theatre."""
    assert _generate(seed=11).coordinate_digest() != _generate(seed=12).coordinate_digest()


def test_generated_instance_matches_its_config() -> None:
    """Counts, ids and provenance all come out as configured."""
    instance = _generate()
    assert len(instance.hubs) == SMALL_GEO.n_hubs
    assert len(instance.sources) == SMALL_GEO.n_sources
    assert len(instance.customers) == SMALL_GEO.n_customers
    assert instance.geo == SMALL_GEO
    assert instance.seed == 7


def test_all_nodes_lie_inside_the_bounding_box() -> None:
    """Including hub centroids, which are derived rather than sampled."""
    instance = _generate(GeoConfig())
    coords = instance.coordinates()
    assert coords[:, 0].min() >= GeoConfig().lat_min
    assert coords[:, 0].max() <= GeoConfig().lat_max
    assert coords[:, 1].min() >= GeoConfig().lon_min
    assert coords[:, 1].max() <= GeoConfig().lon_max


def test_hubs_are_spread_not_overlapping() -> None:
    """The reason hubs are k-means centroids rather than uniform draws.

    Two hubs within a few hundred metres of one another would make hub assignment a formality;
    the separation floor here is roughly 2 km in the isotropic plane (0.02° ≈ 2.2 km).
    """
    geo = GeoConfig()
    instance = _generate(geo)
    hub_xy = _isotropic_xy(instance.coordinates()[: geo.n_hubs], geo)
    assert _nearest_neighbour_distances(hub_xy).min() > 0.02


def test_hubs_are_spread_more_evenly_than_uniform_draws() -> None:
    """Against the alternative the design rejects: uniform placement clusters by chance."""
    geo = GeoConfig()
    instance = _generate(geo)
    hub_xy = _isotropic_xy(instance.coordinates()[: geo.n_hubs], geo)

    rng = np.random.default_rng(0)
    uniform = rng.uniform(
        low=(geo.lat_min, geo.lon_min), high=(geo.lat_max, geo.lon_max), size=(geo.n_hubs, 2)
    )
    uniform_xy = _isotropic_xy(uniform, geo)

    assert (
        _nearest_neighbour_distances(hub_xy).min() > _nearest_neighbour_distances(uniform_xy).min()
    )


def test_clustered_demand_is_denser_than_uniform_demand() -> None:
    """Clustering is what gives consolidation something to exploit.

    Fully clustered demand must have measurably closer nearest neighbours than fully uniform
    demand over the same box and count.
    """
    geo = GeoConfig(n_customers=400, clustered_fraction=1.0)
    clustered = _generate(geo, seed=3).coordinates()[-geo.n_customers :]
    uniform_geo = GeoConfig(n_customers=400, clustered_fraction=0.0)
    uniform = _generate(uniform_geo, seed=3).coordinates()[-uniform_geo.n_customers :]

    clustered_median = float(np.median(_nearest_neighbour_distances(clustered)))
    uniform_median = float(np.median(_nearest_neighbour_distances(uniform)))
    assert clustered_median < uniform_median


def test_sources_have_no_time_windows() -> None:
    """Inbound pickup is unconstrained in time, so a source has nowhere to put a window."""
    assert {field.name for field in dataclasses.fields(Source)} == {"source_id", "coord"}


def test_time_windows_fall_inside_the_operating_day_on_the_configured_grid() -> None:
    """Windows are quarter-hour aligned and never straddle the end of the day."""
    schedule = ScheduleConfig()
    instance = generate_instance(GeoConfig(), DEFAULT_FLEET, schedule, seed=5)
    grid_s = schedule.window_granularity_minutes * SECONDS_PER_MINUTE
    day_start_s = schedule.day_start_hour * SECONDS_PER_HOUR
    day_end_s = schedule.day_end_hour * SECONDS_PER_HOUR

    windows = [c.window for c in instance.customers if c.window is not None]
    assert windows, "the default configuration should produce some windowed customers"
    for window in windows:
        assert day_start_s <= window.start_s < window.end_s <= day_end_s
        assert window.start_s % grid_s == 0.0
        assert window.end_s % grid_s == 0.0
        width_hr = (window.end_s - window.start_s) / SECONDS_PER_HOUR
        assert schedule.min_window_hours <= width_hr <= schedule.max_window_hours


def test_all_day_fraction_is_respected_approximately() -> None:
    """A quarter of customers should be all-day, within sampling noise at n=800."""
    instance = generate_instance(GeoConfig(), DEFAULT_FLEET, ScheduleConfig(), seed=5)
    all_day = sum(1 for customer in instance.customers if customer.window is None)
    assert all_day / len(instance.customers) == pytest.approx(0.25, abs=0.05)


def test_every_customer_has_exactly_one_shipment() -> None:
    """Keeps the KPI denominator unambiguous: one drop per customer."""
    instance = _generate(GeoConfig())
    served = [shipment.customer_id for shipment in instance.shipments]
    assert sorted(served) == list(range(instance.geo.n_customers))


def test_shipments_reference_valid_sources_and_default_size() -> None:
    """Origins are drawn from the source set; size comes from config, never a literal."""
    instance = _generate(GeoConfig())
    assert all(0 <= s.source_id < instance.geo.n_sources for s in instance.shipments)
    assert {s.size_kg for s in instance.shipments} == {DEFAULT_FLEET.shipment_size_kg}


def test_shipment_origins_are_spread_across_sources() -> None:
    """Uniform origin assignment is what makes Stage 1 a real consolidation problem."""
    instance = _generate(GeoConfig())
    used = {shipment.source_id for shipment in instance.shipments}
    # 800 uniform draws over 300 sources leave few untouched; a broken draw would collapse this.
    assert len(used) > 0.8 * instance.geo.n_sources


def test_generation_rejects_a_sigma_too_large_for_the_box() -> None:
    """Density centres must fit inside the box with room for their tails."""
    with pytest.raises(InstanceError):
        _generate(GeoConfig(cluster_sigma_deg=0.5))


def test_generation_rejects_windows_that_cannot_fit_the_granularity() -> None:
    """A coarse grid can make the configured window widths unrepresentable."""
    schedule = ScheduleConfig(window_granularity_minutes=300)
    with pytest.raises(InstanceError):
        generate_instance(SMALL_GEO, DEFAULT_FLEET, schedule, seed=1)


def test_single_hub_and_single_customer_are_legal() -> None:
    """The smallest meaningful instance generates without special-casing."""
    geo = GeoConfig(
        n_hubs=1, n_sources=1, n_customers=1, n_density_clusters=1, hub_candidate_pool=5
    )
    instance = _generate(geo, seed=2)
    assert instance.n_nodes == 3
    assert len(instance.shipments) == 1


def test_hub_count_is_honoured_when_the_pool_is_exactly_the_hub_count() -> None:
    """k-means with as many centroids as candidates still returns distinct hubs."""
    geo = GeoConfig(n_hubs=4, n_sources=2, n_customers=2, hub_candidate_pool=4)
    instance = _generate(geo, seed=4)
    hub_coords = {(hub.coord.lat, hub.coord.lon) for hub in instance.hubs}
    assert len(hub_coords) == geo.n_hubs
