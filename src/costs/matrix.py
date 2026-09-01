"""Distance and duration matrices: the two providers, and the cached entry point.

Every solver in this repository indexes the same pair of ``(n, n)`` matrices, laid out in the
flat ``hubs | sources | customers`` node space defined by :mod:`src.data.instance`. Building
them is the only place the codebase touches the network, and :func:`build_matrices` is the only
sanctioned way to get one — it owns provider selection, the fallback, and the cache.

Two providers implement :class:`DistanceProvider`:

*OSRM* is the real answer, and the reason the results table can claim road distances rather than
crow-flight estimates. Its ``/table`` service caps a single request at ``max-table-size²`` cells,
so a 1116-node matrix cannot be asked for in one go; :class:`OSRMProvider` walks a grid of square
blocks and reassembles them. Chunking is not an optimisation here, it is the only way the request
is legal at all.

*Haversine* is the fallback, and has no external dependencies of any kind. It must always work,
because a portfolio repository that cannot be run without a Docker image and a 400 MB OSM extract
is a repository nobody runs. It is great-circle distance scaled by a circuity factor, with
duration from a flat average speed.

Both return **free-flow** durations. Nothing here knows what time it is; the time-of-day
multiplier is applied along a route by :mod:`src.costs.traffic`, which is its single owner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import numpy.typing as npt
import requests

from src.config import EARTH_RADIUS_M, METRES_PER_KM, SECONDS_PER_HOUR, RunConfig
from src.costs.cache import MatrixCacheKey, read_cache, write_cache
from src.data.instance import Instance
from src.exceptions import MatrixCacheError, MatrixProviderError
from src.units import Coordinates, DistanceMatrix, DurationMatrix, MatrixPair

logger = logging.getLogger(__name__)

_MATRIX_NDIM = 2
"""A cost matrix is two-dimensional. Named so the shape guard reads as an assertion, not a
comparison against a stray literal."""

_OSRM_PROFILE = "driving"
_OSRM_COORD_PRECISION = 6
"""Decimal places in a URL coordinate. Six is ~0.1 m at this latitude — far below OSM's own
positional accuracy, and it keeps a 200-coordinate request URL near 4 kB."""


@dataclass(frozen=True, slots=True, eq=False)
class CostMatrices:
    """The validated pair of matrices every solver reads.

    ``eq=False`` because the members are NumPy arrays: a generated ``__eq__`` would compare them
    element-wise and then fail on the ambiguous truth value of the result. Tests compare the
    arrays directly, which is what they mean anyway.

    Validation lives here rather than in each provider so that both paths — and the cache reload
    — are held to one standard. Downstream code may then index freely: no solver checks whether
    a distance is finite.
    """

    distance_m: DistanceMatrix
    duration_s: DurationMatrix

    def __post_init__(self) -> None:
        for name, matrix in (("distance_m", self.distance_m), ("duration_s", self.duration_s)):
            square = matrix.ndim == _MATRIX_NDIM and matrix.shape[0] == matrix.shape[1]
            if not square or matrix.shape[0] == 0:
                raise MatrixProviderError(f"{name} must be a non-empty square matrix")
            if not np.isfinite(matrix).all():
                raise MatrixProviderError(f"{name} contains non-finite entries")
            if (matrix < 0.0).any():
                raise MatrixProviderError(f"{name} contains negative entries")
        if self.distance_m.shape != self.duration_s.shape:
            raise MatrixProviderError(
                f"distance {self.distance_m.shape} and duration {self.duration_s.shape} "
                f"matrices disagree on shape"
            )

    @property
    def n_nodes(self) -> int:
        """Side length of both matrices."""
        return int(self.distance_m.shape[0])


class DistanceProvider(Protocol):
    """Anything that can turn coordinates into a distance and a duration matrix.

    A ``Protocol`` rather than an ABC: the two implementations share no state and no behaviour,
    only a shape, and structural typing says exactly that. ``name`` is part of the contract
    because it is a component of the cache key — a haversine matrix and an OSRM matrix are
    different data and must never share a cache entry.
    """

    @property
    def name(self) -> str:
        """Short, filename-safe label identifying this provider in the cache key."""

    def matrix(self, coords: Coordinates) -> MatrixPair:
        """Build ``(distance_metres, duration_seconds)`` for every ordered pair of ``coords``.

        Args:
            coords: ``(n, 2)`` array of ``[latitude, longitude]`` in decimal degrees.

        Returns:
            Two ``(n, n)`` float64 matrices: road distance in metres, and free-flow travel time
            in seconds.

        Raises:
            MatrixProviderError: If the matrices could not be produced.
        """


@dataclass(frozen=True, slots=True)
class HaversineProvider:
    """Great-circle distance scaled to a road estimate, with duration from an average speed.

    The circuity factor is the whole model. The default 1.30 is measured against a live OSRM
    build of the Maharashtra extract: road distance over great-circle distance across eight
    Mumbai landmarks runs 1.13–1.40, mean 1.28. Re-measure with ``make providers``.

    It is calibrated against well-connected real places, so it does **not** reproduce what OSRM
    returns for this instance's own nodes — see :class:`~src.config.RunConfig` for why that
    figure is contaminated. This provider exists so a clean checkout runs at all; OSRM remains
    the reference for every number the repository reports, and any run that falls back says so
    at ``WARNING``.

    This provider is deliberately dependency-free. It is the reason ``make run`` works on a
    laptop with no Docker.
    """

    circuity_factor: float
    speed_kmph: float

    name: ClassVar[str] = "haversine"

    def matrix(self, coords: Coordinates) -> MatrixPair:
        """Build the matrices analytically; see :meth:`DistanceProvider.matrix`."""
        distance: DistanceMatrix = haversine_matrix(coords) * self.circuity_factor
        metres_per_second = self.speed_kmph * METRES_PER_KM / SECONDS_PER_HOUR
        duration: DurationMatrix = distance / metres_per_second
        return distance, duration


def haversine_matrix(coords: Coordinates) -> DistanceMatrix:
    """Great-circle distance in metres between every ordered pair of ``coords``.

    Fully vectorised: the trigonometry runs once over an ``(n, n)`` broadcast rather than in a
    Python loop, which for 1116 nodes is the difference between milliseconds and minutes.

    Args:
        coords: ``(n, 2)`` array of ``[latitude, longitude]`` in decimal degrees.

    Returns:
        An ``(n, n)`` symmetric matrix of metres with a zero diagonal.
    """
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    half_dlat = 0.5 * (lat[:, np.newaxis] - lat[np.newaxis, :])
    half_dlon = 0.5 * (lon[:, np.newaxis] - lon[np.newaxis, :])
    chord = (
        np.sin(half_dlat) ** 2
        + np.cos(lat)[:, np.newaxis] * np.cos(lat)[np.newaxis, :] * np.sin(half_dlon) ** 2
    )
    # Clip before arcsin: for coincident points the expression can land a few ULPs above 1.0,
    # which would produce NaN rather than the zero the diagonal must hold.
    distance: DistanceMatrix = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(chord, 1.0)))
    return distance


@dataclass(frozen=True, slots=True)
class OSRMProvider:
    """Assembles a full matrix from chunked queries against a self-hosted OSRM ``/table``.

    OSRM rejects a table request whose ``sources × destinations`` cell count exceeds
    ``max-table-size²``, so the matrix is walked as a grid of square blocks of side
    ``max_table_size``. Each request carries only that block's own coordinates — the union of its
    row and column nodes, at most ``2 × max_table_size`` points — rather than the full node list
    with index selectors, which for 1116 nodes would put a 22 kB coordinate string in the URL and
    exceed the server's request-line limit.

    Every failure mode raises :class:`~src.exceptions.MatrixProviderError`, because the caller's
    response to all of them is the same: warn loudly, and fall back to haversine.
    """

    base_url: str
    max_table_size: int
    timeout_s: float

    name: ClassVar[str] = "osrm"

    def matrix(self, coords: Coordinates) -> MatrixPair:
        """Assemble the matrices block by block; see :meth:`DistanceProvider.matrix`."""
        n_nodes = len(coords)
        distance: DistanceMatrix = np.empty((n_nodes, n_nodes), dtype=np.float64)
        duration: DurationMatrix = np.empty((n_nodes, n_nodes), dtype=np.float64)
        blocks = _block_bounds(n_nodes, self.max_table_size)

        with requests.Session() as session:
            for rows in blocks:
                for cols in blocks:
                    block_distance, block_duration = self._table_block(session, coords, rows, cols)
                    distance[rows.start : rows.stop, cols.start : cols.stop] = block_distance
                    duration[rows.start : rows.stop, cols.start : cols.stop] = block_duration

        logger.info(
            "assembled %d×%d OSRM matrices from %d blocks of at most %d×%d",
            n_nodes,
            n_nodes,
            len(blocks) ** 2,
            self.max_table_size,
            self.max_table_size,
        )
        return distance, duration

    def _table_block(
        self, session: requests.Session, coords: Coordinates, rows: range, cols: range
    ) -> MatrixPair:
        """Query one rectangular block of the matrix and return it as two arrays."""
        source_coords = coords[rows.start : rows.stop]
        dest_coords = coords[cols.start : cols.stop]
        payload = self._get(session, self._block_url(source_coords, dest_coords))
        shape = (len(rows), len(cols))
        return (
            _block_array(payload, "distances", shape),
            _block_array(payload, "durations", shape),
        )

    def _block_url(self, source_coords: Coordinates, dest_coords: Coordinates) -> str:
        """Build the ``/table`` URL for one block.

        The two coordinate groups are concatenated into the path and then addressed positionally
        by ``sources``/``destinations``, so a block on the diagonal sends its nodes twice. That
        costs a few hundred bytes of URL and removes the special case entirely.
        """
        n_sources = len(source_coords)
        points = ";".join(
            f"{lon:.{_OSRM_COORD_PRECISION}f},{lat:.{_OSRM_COORD_PRECISION}f}"
            for lat, lon in np.vstack((source_coords, dest_coords))
        )
        sources = ";".join(str(index) for index in range(n_sources))
        destinations = ";".join(str(n_sources + index) for index in range(len(dest_coords)))
        return (
            f"{self.base_url.rstrip('/')}/table/v1/{_OSRM_PROFILE}/{points}"
            f"?sources={sources}&destinations={destinations}&annotations=distance,duration"
        )

    def _get(self, session: requests.Session, url: str) -> dict[str, Any]:
        """Issue one request and return the decoded body.

        ``dict[str, Any]``: a decoded JSON object is genuinely untyped. Every field read from it
        is narrowed by :func:`_block_array` before it reaches a matrix.

        ``response.raise_for_status()`` is deliberately not used — OSRM reports ``TooBig`` as a
        400 with a JSON body, and that body is the single most useful diagnostic when chunking is
        misconfigured. The status code is folded into the message instead.
        """
        try:
            response = session.get(url, timeout=self.timeout_s)
            body: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MatrixProviderError(f"OSRM request to {self.base_url} failed: {exc}") from exc

        code = body.get("code")
        if code != "Ok":
            raise MatrixProviderError(
                f"OSRM at {self.base_url} returned HTTP {response.status_code} "
                f"code={code!r}: {body.get('message')!r}"
            )
        return body


def _block_bounds(n_nodes: int, chunk: int) -> tuple[range, ...]:
    """Split ``range(n_nodes)`` into consecutive blocks of at most ``chunk`` indices.

    Returned as ``range`` rather than ``slice`` so a block knows its own length, which is what
    the response-shape check compares against.
    """
    return tuple(range(start, min(start + chunk, n_nodes)) for start in range(0, n_nodes, chunk))


def _block_array(
    payload: dict[str, Any], key: str, shape: tuple[int, int]
) -> npt.NDArray[np.float64]:
    """Narrow one annotation of an OSRM response into a float64 block of the expected shape.

    Args:
        payload: Decoded ``/table`` response body.
        key: ``"distances"`` or ``"durations"``.
        shape: ``(n_sources, n_destinations)`` the block must have.

    Returns:
        The block as a float64 array.

    Raises:
        MatrixProviderError: If the annotation is absent, the wrong shape, or holds a ``null``
            for an unroutable pair. An unroutable pair is fatal rather than patched: an
            interpolated distance would silently become part of the reported cost.
    """
    if key not in payload:
        raise MatrixProviderError(
            f"OSRM response has no {key!r} annotation; the server must run the MLD pipeline "
            f"(osrm-partition + osrm-customize) for distance annotations to be available"
        )
    # OSRM encodes an unroutable pair as JSON null, which numpy renders as NaN under float64.
    block: npt.NDArray[np.float64] = np.array(payload[key], dtype=np.float64)
    if block.shape != shape:
        raise MatrixProviderError(f"OSRM returned a {block.shape} {key} block, expected {shape}")
    unroutable = int(np.isnan(block).sum())
    if unroutable:
        raise MatrixProviderError(
            f"OSRM could not route {unroutable} of {block.size} pairs in a {key} block; "
            f"the OSM extract probably does not cover the whole instance bounding box"
        )
    return block


def build_matrices(instance: Instance, run: RunConfig) -> CostMatrices:
    """Get the cost matrices for ``instance``: from cache, from OSRM, or from haversine.

    The single entry point to the cost layer. Provider choice follows ``run.use_osrm``, and an
    OSRM failure falls back to haversine with a ``WARNING`` — a fallback changes every distance,
    duration and rupee figure the run reports, so it is never allowed to be silent.

    Note the cache lookup sits *inside* each provider attempt, keyed by that provider's name. A
    fallback therefore consults the haversine cache rather than reusing an OSRM entry, and a
    damaged cache raises :class:`~src.exceptions.MatrixCacheError` rather than being mistaken for
    an OSRM outage and quietly downgrading the provider.

    Args:
        instance: Supplies the coordinates, the node count, the seed and the coordinate digest —
            every component of the cache key, from one object, so they cannot disagree.
        run: Provider selection, endpoint, fallback parameters and cache location.

    Returns:
        The validated matrices for this instance.

    Raises:
        MatrixCacheError: If a cache entry exists but cannot be trusted.
        MatrixProviderError: If the haversine fallback itself fails, which means a bug.
    """
    fallback = HaversineProvider(
        circuity_factor=run.circuity_factor, speed_kmph=run.haversine_speed_kmph
    )
    if not run.use_osrm:
        return _cached_matrices(fallback, instance, run.cache_dir)

    osrm = OSRMProvider(
        base_url=run.osrm_url,
        max_table_size=run.osrm_max_table_size,
        timeout_s=run.osrm_timeout_s,
    )
    try:
        return _cached_matrices(osrm, instance, run.cache_dir)
    except MatrixProviderError as exc:
        logger.warning(
            "OSRM at %s is unavailable (%s). Falling back to haversine × %.2f circuity at "
            "%.1f km/h: every distance, duration and cost this run reports is a great-circle "
            "estimate, not a road-network figure.",
            run.osrm_url,
            exc,
            run.circuity_factor,
            run.haversine_speed_kmph,
        )
        return _cached_matrices(fallback, instance, run.cache_dir)


def _cached_matrices(
    provider: DistanceProvider, instance: Instance, cache_dir: Path
) -> CostMatrices:
    """Return ``provider``'s matrices for ``instance``, building and caching them on a miss.

    Validation runs before the write, so a provider that returns nonsense fails the run instead
    of persisting nonsense for every later run to load.
    """
    key = MatrixCacheKey(
        seed=instance.seed,
        provider_name=provider.name,
        n_nodes=instance.n_nodes,
        coord_digest=instance.coordinate_digest(),
    )
    cached = read_cache(cache_dir, key)
    if cached is not None:
        try:
            return CostMatrices(distance_m=cached[0], duration_s=cached[1])
        except MatrixProviderError as exc:
            # Re-typed on purpose: a bad file on disk must not read as a provider outage, or the
            # fallback would silently swap OSRM for haversine over a corrupt cache entry.
            raise MatrixCacheError(f"cached matrix {key.filename} is invalid: {exc}") from exc

    matrices = provider.matrix(instance.coordinates())
    validated = CostMatrices(distance_m=matrices[0], duration_s=matrices[1])
    write_cache(cache_dir, key, matrices)
    return validated
