"""Seeded generation of the synthetic Mumbai / Navi Mumbai instance.

Two generation choices are deliberate and load-bearing for the experiment:

*Density is clustered, not uniform.* A share of sources and customers is drawn from a handful of
Gaussian blobs standing in for commercial and residential concentrations. Uniform scatter gives
an unrealistically even workload in which consolidation buys almost nothing and every solver
looks equally good.

*Hubs are k-means centroids over a demand-shaped candidate pool*, not uniform draws. Uniform
placement puts hubs on top of one another and leaves whole quadrants unserved, which turns hub
assignment into a formality. Clustering also means the hub network follows demand, as a real one
would.

Every random draw goes through one ``np.random.Generator`` built from the run seed. That
generator is created here — the single sanctioned construction point — and the seed is recorded
on the returned :class:`~src.data.instance.Instance` so the data is reproducible from the
artefact alone.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from src.config import MINUTES_PER_HOUR, SECONDS_PER_HOUR, FleetConfig, GeoConfig, ScheduleConfig
from src.data.instance import (
    Coordinate,
    Customer,
    Hub,
    Instance,
    Shipment,
    Source,
    TimeWindow,
)
from src.exceptions import InstanceError
from src.units import Seconds

Points = npt.NDArray[np.float64]
"""An ``(n, 2)`` array of ``[latitude, longitude]`` in decimal degrees."""

# Cluster centres are held this many sigmas inside the bounding box so the Gaussian tails rarely
# need clipping; clipping would pile points onto the box edge and invent a coastline of demand.
_CENTRE_INSET_SIGMAS = 2.0


def generate_instance(
    geo: GeoConfig,
    fleet: FleetConfig,
    schedule: ScheduleConfig,
    seed: int,
) -> Instance:
    """Build a complete instance for ``seed``.

    Args:
        geo: Bounding box, node counts and density structure.
        fleet: Vehicle capacity and shipment size.
        schedule: Operating day and time-window shape.
        seed: Run seed; recorded on the instance and the only source of randomness.

    Returns:
        A validated :class:`~src.data.instance.Instance`.
    """
    rng = np.random.default_rng(seed)
    centres = _density_centres(geo, rng)

    hub_points = _place_hubs(geo, centres, rng)
    source_points = _sample_points(geo.n_sources, geo, centres, rng)
    customer_points = _sample_points(geo.n_customers, geo, centres, rng)
    windows = _delivery_windows(geo.n_customers, schedule, rng)

    return Instance(
        seed=seed,
        geo=geo,
        fleet=fleet,
        schedule=schedule,
        hubs=tuple(Hub(i, _coord(hub_points[i])) for i in range(geo.n_hubs)),
        sources=tuple(Source(i, _coord(source_points[i])) for i in range(geo.n_sources)),
        customers=tuple(
            Customer(i, _coord(customer_points[i]), windows[i]) for i in range(geo.n_customers)
        ),
        shipments=_link_shipments(geo, fleet, rng),
    )


def _coord(point: npt.NDArray[np.float64]) -> Coordinate:
    """Convert one ``[lat, lon]`` row to a :class:`~src.data.instance.Coordinate`."""
    return Coordinate(lat=float(point[0]), lon=float(point[1]))


def _density_centres(geo: GeoConfig, rng: np.random.Generator) -> Points:
    """Place the commercial/residential density centres, inset from the box edge."""
    inset = _CENTRE_INSET_SIGMAS * geo.cluster_sigma_deg
    low = (geo.lat_min + inset, geo.lon_min + inset)
    high = (geo.lat_max - inset, geo.lon_max - inset)
    if low[0] >= high[0] or low[1] >= high[1]:
        raise InstanceError("cluster_sigma_deg is too large for the configured bounding box")
    return rng.uniform(low=low, high=high, size=(geo.n_density_clusters, 2))


def _sample_points(n: int, geo: GeoConfig, centres: Points, rng: np.random.Generator) -> Points:
    """Draw ``n`` demand points: a clustered share around ``centres``, the rest uniform.

    The two groups are shuffled together so node id does not encode which group a point came
    from — otherwise any solver that happens to process ids in order gets a free spatial hint.
    """
    n_clustered = round(n * geo.clustered_fraction)
    assignments = rng.integers(0, len(centres), size=n_clustered)
    clustered = centres[assignments] + rng.normal(
        loc=0.0, scale=geo.cluster_sigma_deg, size=(n_clustered, 2)
    )
    scattered = rng.uniform(
        low=(geo.lat_min, geo.lon_min),
        high=(geo.lat_max, geo.lon_max),
        size=(n - n_clustered, 2),
    )
    points = np.vstack((clustered, scattered))
    rng.shuffle(points, axis=0)
    return _clip_to_box(points, geo)


def _clip_to_box(points: Points, geo: GeoConfig) -> Points:
    """Clamp points to the bounding box. A safety net for Gaussian tails, not a shaping step."""
    return np.clip(
        points,
        a_min=(geo.lat_min, geo.lon_min),
        a_max=(geo.lat_max, geo.lon_max),
    )


def _place_hubs(geo: GeoConfig, centres: Points, rng: np.random.Generator) -> Points:
    """Place hubs at k-means centroids of a demand-shaped candidate pool.

    Clustering runs in an isotropic plane: at 19°N a degree of longitude is ~5% shorter than a
    degree of latitude, and clustering in raw degrees would stretch the hub network east-west.

    Note that ``hub_candidate_pool`` consumes the generator stream before sources and customers
    are drawn, so changing it moves every downstream coordinate too — it reshapes the whole
    instance for a given seed, not just where the hubs land.
    """
    pool = _sample_points(geo.hub_candidate_pool, geo, centres, rng)
    lon_scale = math.cos(math.radians(0.5 * (geo.lat_min + geo.lat_max)))
    centroids = _kmeans(_scale_lon(pool, lon_scale), geo.n_hubs, geo.hub_kmeans_iterations, rng)
    return _clip_to_box(_scale_lon(centroids, 1.0 / lon_scale), geo)


def _scale_lon(points: Points, factor: float) -> Points:
    """Scale the longitude column, mapping between degree space and the isotropic plane."""
    scaled = points.copy()
    scaled[:, 1] *= factor
    return scaled


def _kmeans(points: Points, k: int, iterations: int, rng: np.random.Generator) -> Points:
    """Lloyd's algorithm with k-means++ seeding, vectorised over the candidate pool.

    Written out rather than pulled from scikit-learn: it is a dozen lines, it keeps the
    dependency list to the solvers that earn their place, and it lets the injected generator
    drive initialisation so hub placement is reproducible.
    """
    centroids = _kmeans_plusplus_init(points, k, rng)
    for _ in range(iterations):
        updated = _lloyd_step(points, centroids)
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return centroids


def _kmeans_plusplus_init(points: Points, k: int, rng: np.random.Generator) -> Points:
    """Seed centroids far apart, with probability proportional to squared distance.

    Arthur & Vassilvitskii (2007). Uniform seeding routinely starts two centroids inside one
    dense blob, and Lloyd's cannot recover from that — it produces exactly the overlapping hubs
    this generator exists to avoid.
    """
    centroids = np.empty((k, points.shape[1]), dtype=np.float64)
    centroids[0] = points[rng.integers(len(points))]
    nearest_sq = _squared_distances(points, centroids[:1]).ravel()
    for index in range(1, k):
        total = float(nearest_sq.sum())
        # Degenerate pool (every candidate coincident): fall back to a uniform draw.
        weights = nearest_sq / total if total > 0.0 else None
        centroids[index] = points[rng.choice(len(points), p=weights)]
        nearest_sq = np.minimum(
            nearest_sq, _squared_distances(points, centroids[index : index + 1]).ravel()
        )
    return centroids


def _lloyd_step(points: Points, centroids: Points) -> Points:
    """One assign-then-recentre pass.

    A centroid that loses every member is re-seeded onto the point its surviving siblings serve
    worst, rather than left to produce a NaN centroid that would poison every distance matrix
    downstream. Re-seeding happens one centroid at a time with the distances recomputed after
    each placement: choosing all re-seed points up front drops two centroids orphaned in the
    same pass onto the same candidate, whereupon ``np.allclose`` reports convergence and the run
    returns fewer distinct hubs than were configured.
    """
    labels = _squared_distances(points, centroids).argmin(axis=1)
    updated = centroids.copy()
    populated = {index for index in range(len(centroids)) if bool(np.any(labels == index))}
    for index in populated:
        updated[index] = points[labels == index].mean(axis=0)

    placed = sorted(populated)
    for index in range(len(centroids)):
        if index in populated:
            continue
        updated[index] = _worst_served_point(points, updated[placed])
        placed.append(index)
    return updated


def _worst_served_point(points: Points, centroids: Points) -> npt.NDArray[np.float64]:
    """The single ``[lat, lon]`` point farthest from every centroid — the best place to add one.

    Every point is assigned to some centroid, so ``centroids`` is never empty here: at least one
    cluster keeps a member.
    """
    farthest = int(_squared_distances(points, centroids).min(axis=1).argmax())
    # Indexing an ndarray with a scalar is untyped in the numpy stubs; the annotation narrows it.
    point: npt.NDArray[np.float64] = points[farthest]
    return point


def _squared_distances(points: Points, centroids: Points) -> npt.NDArray[np.float64]:
    """Squared Euclidean distance from every point to every centroid, shape ``(n, k)``."""
    deltas = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    # einsum is untyped in the numpy stubs; the annotation is the narrowing.
    squared: npt.NDArray[np.float64] = np.einsum("nkd,nkd->nk", deltas, deltas)
    return squared


def _delivery_windows(
    n: int, schedule: ScheduleConfig, rng: np.random.Generator
) -> tuple[TimeWindow | None, ...]:
    """Draw one delivery window per customer, or ``None`` for an all-day customer.

    Windows snap to a quarter-hour grid because dispatch commitments are made in round numbers,
    and the grid is worked in integer slots so a window can never straddle the end of the
    operating day.
    """
    slot_hours = schedule.window_granularity_minutes / MINUTES_PER_HOUR
    day_slots = round(schedule.day_length_hours / slot_hours)
    min_slots = math.ceil(schedule.min_window_hours / slot_hours)
    max_slots = math.floor(schedule.max_window_hours / slot_hours)
    if not 1 <= min_slots <= max_slots <= day_slots:
        raise InstanceError(
            "window bounds do not fit the operating day at the configured granularity"
        )

    widths = rng.integers(min_slots, max_slots + 1, size=n)
    starts = rng.integers(0, day_slots - widths + 1)
    all_day = rng.random(n) < schedule.all_day_fraction

    def window(index: int) -> TimeWindow | None:
        if all_day[index]:
            return None
        start_h = schedule.day_start_hour + float(starts[index]) * slot_hours
        end_h = start_h + float(widths[index]) * slot_hours
        return TimeWindow(
            start_s=Seconds(start_h * SECONDS_PER_HOUR),
            end_s=Seconds(end_h * SECONDS_PER_HOUR),
        )

    return tuple(window(index) for index in range(n))


def _link_shipments(
    geo: GeoConfig, fleet: FleetConfig, rng: np.random.Generator
) -> tuple[Shipment, ...]:
    """Create one shipment per customer, with an origin source drawn uniformly.

    One shipment per customer keeps the headline KPI unambiguous: drops equal customers, so cost
    per drop cannot be gamed by splitting an order. Origins are uniform rather than
    nearest-source, which is what makes Stage 1 a real consolidation problem — with
    geographically correlated origins the inbound answer is a foregone conclusion.
    """
    origins = rng.integers(0, geo.n_sources, size=geo.n_customers)
    return tuple(
        Shipment(
            shipment_id=customer_id,
            source_id=int(origins[customer_id]),
            customer_id=customer_id,
            size_kg=fleet.shipment_size_kg,
        )
        for customer_id in range(geo.n_customers)
    )
