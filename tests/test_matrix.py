"""Tests for the distance/duration providers, the chunked OSRM path, and the cached entry point.

The OSRM tests run against a real loopback server (:mod:`tests.osrm_stub`) that enforces the same
``max_table_size²`` cell budget the production server does. Nothing in ``src`` is patched, so a
chunking regression fails here exactly as it would fail against a live OSRM.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import EARTH_RADIUS_M, METRES_PER_KM, SECONDS_PER_HOUR, RunConfig
from src.costs.cache import MatrixCacheKey, write_cache
from src.costs.matrix import (
    CostMatrices,
    HaversineProvider,
    OSRMProvider,
    build_matrices,
    haversine_matrix,
)
from src.data.instance import Instance
from src.exceptions import MatrixCacheError, MatrixProviderError
from src.units import Coordinates
from tests.osrm_stub import (
    Point,
    StubState,
    osrm_stub,
    truth_distance_m,
    truth_duration_s,
    unused_port,
)

# Six-decimal-exact so a coordinate survives the round trip through the OSRM URL unchanged and
# the assembled matrix can be compared for exact equality rather than approximate.
GRID_POINTS: tuple[Point, ...] = (
    (19.000, 72.800),
    (19.010, 72.815),
    (19.025, 72.830),
    (19.040, 72.845),
    (19.055, 72.860),
    (19.070, 72.875),
    (19.085, 72.890),
)


def grid_coords(n_points: int) -> Coordinates:
    """The first ``n_points`` grid points as a ``(n, 2)`` lat/lon array."""
    return np.array(GRID_POINTS[:n_points], dtype=np.float64)


def truth_matrices(coords: Coordinates) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """The full matrices the stub should produce, computed independently of any chunking."""
    points = [(float(lat), float(lon)) for lat, lon in coords]
    distance = np.array([[truth_distance_m(a, b) for b in points] for a in points])
    duration = np.array([[truth_duration_s(a, b) for b in points] for a in points])
    return distance, duration


def osrm_provider(base_url: str, max_table_size: int) -> OSRMProvider:
    """An OSRM provider pointed at a stub, with a short timeout so a hang fails fast."""
    return OSRMProvider(base_url=base_url, max_table_size=max_table_size, timeout_s=5.0)


# --------------------------------------------------------------------------------------------
# Haversine
# --------------------------------------------------------------------------------------------


def test_haversine_matches_one_degree_of_latitude() -> None:
    """One degree of latitude is a meridian arc of R × π/180 — the check that fixes the units."""
    coords = np.array([[19.0, 72.9], [20.0, 72.9]], dtype=np.float64)
    expected_m = EARTH_RADIUS_M * math.pi / 180.0
    assert haversine_matrix(coords)[0, 1] == pytest.approx(expected_m, abs=1.0)


def test_haversine_diagonal_is_exactly_zero() -> None:
    """Coincident points must give zero, not a NaN from arcsin of 1 + a few ULPs."""
    matrix = haversine_matrix(grid_coords(len(GRID_POINTS)))
    assert np.all(np.diagonal(matrix) == 0.0)
    assert np.isfinite(matrix).all()


def test_haversine_is_symmetric() -> None:
    """Great-circle distance has no direction, so the matrix is its own transpose."""
    matrix = haversine_matrix(grid_coords(5))
    assert np.array_equal(matrix, matrix.T)


def test_haversine_single_coordinate_gives_a_one_by_one_zero() -> None:
    """The degenerate boundary: one node is a legal, if pointless, instance."""
    distance, duration = HaversineProvider(circuity_factor=1.35, speed_kmph=24.0).matrix(
        np.array([[19.0, 72.9]], dtype=np.float64)
    )
    assert distance.shape == (1, 1)
    assert distance[0, 0] == 0.0
    assert duration[0, 0] == 0.0


def test_haversine_provider_applies_circuity_and_speed() -> None:
    """Distance is scaled great-circle; duration is that distance at the configured speed."""
    coords = grid_coords(4)
    provider = HaversineProvider(circuity_factor=1.35, speed_kmph=24.0)
    distance, duration = provider.matrix(coords)

    assert np.allclose(distance, haversine_matrix(coords) * 1.35)
    metres_per_second = 24.0 * METRES_PER_KM / SECONDS_PER_HOUR
    assert np.allclose(duration, distance / metres_per_second)


def test_haversine_circuity_of_one_leaves_distance_unscaled() -> None:
    """The boundary of the circuity guard: 1.0 is legal and means "road == great-circle"."""
    coords = grid_coords(3)
    distance, _ = HaversineProvider(circuity_factor=1.0, speed_kmph=24.0).matrix(coords)
    assert np.allclose(distance, haversine_matrix(coords))


# --------------------------------------------------------------------------------------------
# OSRM chunking
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_points", "max_table_size", "expected_requests"),
    [
        pytest.param(3, 3, 1, id="exactly-at-the-limit-is-one-request"),
        pytest.param(4, 3, 4, id="one-over-the-limit-splits-into-four-blocks"),
        pytest.param(7, 3, 9, id="three-by-three-block-grid"),
        pytest.param(7, 2, 16, id="four-by-four-block-grid"),
        pytest.param(1, 5, 1, id="single-node"),
    ],
)
def test_osrm_assembles_the_full_matrix_from_blocks(
    n_points: int, max_table_size: int, expected_requests: int
) -> None:
    """The reassembled matrix equals the ground truth cell for cell, at every chunk geometry.

    Exact equality is the point: a block written to the wrong offset, or a ``sources``/
    ``destinations`` index built against the wrong base, produces a matrix of the right shape
    holding the right values in the wrong places. Only a full comparison catches that.
    """
    coords = grid_coords(n_points)
    with osrm_stub(StubState(max_table_size=max_table_size)) as (base_url, state):
        distance, duration = osrm_provider(base_url, max_table_size).matrix(coords)

    expected_distance, expected_duration = truth_matrices(coords)
    assert np.array_equal(distance, expected_distance)
    assert np.array_equal(duration, expected_duration)
    assert len(state.requests) == expected_requests


def test_osrm_never_exceeds_the_server_cell_budget() -> None:
    """No single request asks for more than ``max_table_size²`` cells.

    The stub answers ``TooBig`` if one does, so the matrix call above would already have raised —
    this asserts the same invariant directly, because it is the constraint the whole chunking
    design exists to satisfy.
    """
    with osrm_stub(StubState(max_table_size=3)) as (base_url, state):
        osrm_provider(base_url, 3).matrix(grid_coords(7))

    assert state.requests
    assert all(request.cells <= 9 for request in state.requests)
    assert all(len(request.points) <= 6 for request in state.requests)


def test_osrm_sends_lon_lat_in_that_order_and_asks_for_both_annotations() -> None:
    """OSRM's URL grammar is ``lon,lat``; the codebase's is ``lat, lon``. The swap happens here."""
    coords = grid_coords(2)
    with osrm_stub(StubState(max_table_size=5)) as (base_url, state):
        osrm_provider(base_url, 5).matrix(coords)

    request = state.requests[0]
    # The stub parses back to (lat, lon); a swapped provider would round-trip to (lon, lat).
    assert request.points[: len(coords)] == tuple((float(a), float(b)) for a, b in coords)
    assert request.annotations == "distance,duration"


def test_osrm_indexes_destinations_after_sources() -> None:
    """Sources and destinations are addressed positionally in one concatenated coordinate list."""
    with osrm_stub(StubState(max_table_size=2)) as (base_url, state):
        osrm_provider(base_url, 2).matrix(grid_coords(4))

    for request in state.requests:
        assert request.sources == tuple(range(len(request.sources)))
        assert request.destinations == tuple(
            range(len(request.sources), len(request.sources) + len(request.destinations))
        )


# --------------------------------------------------------------------------------------------
# OSRM failure modes
# --------------------------------------------------------------------------------------------


def test_osrm_reports_a_too_big_response_verbatim() -> None:
    """A misconfigured chunk size must say ``TooBig``, not "something went wrong"."""
    with (
        osrm_stub(StubState(force_code="TooBig")) as (base_url, _),
        pytest.raises(MatrixProviderError, match="TooBig"),
    ):
        osrm_provider(base_url, 5).matrix(grid_coords(3))


def test_osrm_unroutable_pair_is_fatal() -> None:
    """A ``null`` cell raises rather than being interpolated into the reported cost."""
    with (
        osrm_stub(StubState(null_first_cell=True)) as (base_url, _),
        pytest.raises(MatrixProviderError, match="could not route"),
    ):
        osrm_provider(base_url, 5).matrix(grid_coords(3))


def test_osrm_missing_distance_annotation_names_the_mld_pipeline() -> None:
    """The CH pipeline serves durations but no distances; the error has to say so."""
    with (
        osrm_stub(StubState(omit_distances=True)) as (base_url, _),
        pytest.raises(MatrixProviderError, match="osrm-partition"),
    ):
        osrm_provider(base_url, 5).matrix(grid_coords(3))


def test_osrm_block_of_the_wrong_shape_raises() -> None:
    """A well-formed response that is not the block asked for must not be silently reshaped.

    NumPy would accept a ragged list as an object array rather than fail, so without this guard
    a short row becomes a matrix of the right shape holding shifted values.
    """
    with (
        osrm_stub(StubState(truncate_block=True)) as (base_url, _),
        pytest.raises(MatrixProviderError, match="expected"),
    ):
        osrm_provider(base_url, 5).matrix(grid_coords(3))


def test_osrm_unreachable_host_raises_a_provider_error() -> None:
    """A refused connection is a provider failure, not a stack trace out of ``requests``."""
    provider = osrm_provider(f"http://127.0.0.1:{unused_port()}", 5)
    with pytest.raises(MatrixProviderError, match="failed"):
        provider.matrix(grid_coords(2))


# --------------------------------------------------------------------------------------------
# CostMatrices validation
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distance", "duration", "expected"),
    [
        pytest.param(np.zeros((2, 3)), np.zeros((2, 3)), "square", id="rectangular"),
        pytest.param(np.zeros(4), np.zeros(4), "square", id="one-dimensional"),
        pytest.param(np.zeros((0, 0)), np.zeros((0, 0)), "non-empty", id="empty"),
        pytest.param(np.full((2, 2), np.nan), np.zeros((2, 2)), "non-finite", id="nan-distance"),
        pytest.param(
            np.zeros((2, 2)), np.full((2, 2), np.inf), "non-finite", id="infinite-duration"
        ),
        pytest.param(np.full((2, 2), -1.0), np.zeros((2, 2)), "negative", id="negative-distance"),
        pytest.param(
            np.zeros((2, 2)), np.zeros((3, 3)), "disagree on shape", id="mismatched-sides"
        ),
    ],
)
def test_cost_matrices_reject(
    distance: np.ndarray,  # type: ignore[type-arg]
    duration: np.ndarray,  # type: ignore[type-arg]
    expected: str,
) -> None:
    """Every validation guard rejects its own failure mode, before any number is reported."""
    with pytest.raises(MatrixProviderError, match=expected):
        CostMatrices(distance_m=distance, duration_s=duration)


def test_cost_matrices_accept_a_valid_pair() -> None:
    """The happy path, and the side length it reports."""
    matrices = CostMatrices(distance_m=np.zeros((5, 5)), duration_s=np.zeros((5, 5)))
    assert matrices.n_nodes == 5


# --------------------------------------------------------------------------------------------
# build_matrices: selection, cache, fallback
# --------------------------------------------------------------------------------------------


def run_config(tmp_path: Path, **overrides: object) -> RunConfig:
    """A RunConfig with the cache isolated to a temp directory."""
    return RunConfig(cache_dir=tmp_path, **overrides)  # type: ignore[arg-type]  # heterogeneous


def test_build_matrices_uses_haversine_when_osrm_is_disabled(
    tiny_instance: Instance, tmp_path: Path
) -> None:
    """``use_osrm=False`` must not open a socket at all, not merely fail over."""
    with osrm_stub() as (base_url, state):
        run = run_config(tmp_path, use_osrm=False, osrm_url=base_url)
        matrices = build_matrices(tiny_instance, run)

    assert state.requests == []
    expected = haversine_matrix(tiny_instance.coordinates()) * run.circuity_factor
    assert np.allclose(matrices.distance_m, expected)


def test_build_matrices_queries_osrm_when_enabled(tiny_instance: Instance, tmp_path: Path) -> None:
    """With OSRM reachable, the matrices are the server's, not the fallback's."""
    with osrm_stub(StubState(max_table_size=2)) as (base_url, state):
        run = run_config(tmp_path, osrm_url=base_url, osrm_max_table_size=2)
        matrices = build_matrices(tiny_instance, run)

    assert state.requests
    expected_distance, expected_duration = truth_matrices(tiny_instance.coordinates())
    assert np.array_equal(matrices.distance_m, expected_distance)
    assert np.array_equal(matrices.duration_s, expected_duration)


def test_build_matrices_serves_the_second_run_from_cache(
    tiny_instance: Instance, tmp_path: Path
) -> None:
    """The cache is what stops the GA re-querying OSRM on every fitness evaluation.

    Asserted behaviourally: the server is gone by the second call, yet the OSRM matrices come
    back. A cache miss here would fall back to haversine and the values would change.
    """
    with osrm_stub(StubState(max_table_size=2)) as (base_url, _):
        run = run_config(tmp_path, osrm_url=base_url, osrm_max_table_size=2)
        first = build_matrices(tiny_instance, run)

    second = build_matrices(tiny_instance, run)
    assert np.array_equal(first.distance_m, second.distance_m)
    assert np.array_equal(first.duration_s, second.duration_s)


def test_build_matrices_caches_providers_separately(
    tiny_instance: Instance, tmp_path: Path
) -> None:
    """A haversine run must never load a cache entry written by a road-network run."""
    with osrm_stub(StubState(max_table_size=4)) as (base_url, _):
        build_matrices(tiny_instance, run_config(tmp_path, osrm_url=base_url))
    build_matrices(tiny_instance, run_config(tmp_path, use_osrm=False))

    cached = sorted(path.name for path in tmp_path.glob("*.parquet"))
    assert len(cached) == 2
    assert any("_haversine_" in name for name in cached)
    assert any("_osrm_" in name for name in cached)


def test_build_matrices_falls_back_to_haversine_and_warns(
    tiny_instance: Instance, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreachable OSRM must change every number *loudly*, at WARNING."""
    run = run_config(tmp_path, osrm_url=f"http://127.0.0.1:{unused_port()}")
    with caplog.at_level(logging.WARNING, logger="src.costs.matrix"):
        matrices = build_matrices(tiny_instance, run)

    expected = haversine_matrix(tiny_instance.coordinates()) * run.circuity_factor
    assert np.allclose(matrices.distance_m, expected)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "haversine" in warnings[0].getMessage()
    assert "not a road-network figure" in warnings[0].getMessage()


def test_a_corrupt_cache_does_not_silently_downgrade_the_provider(
    tiny_instance: Instance, tmp_path: Path
) -> None:
    """A damaged cache file must surface as a cache error, not as an OSRM outage.

    This is the reason :class:`MatrixCacheError` is not a :class:`MatrixProviderError`. Were it
    one, the fallback would catch it and quietly swap road distances for great-circle estimates
    on the strength of a bad file.
    """
    key = MatrixCacheKey(
        seed=tiny_instance.seed,
        provider_name="osrm",
        n_nodes=tiny_instance.n_nodes,
        coord_digest=tiny_instance.coordinate_digest(),
    )
    key.path_in(tmp_path).write_bytes(b"not a parquet file")

    with (
        osrm_stub(StubState(max_table_size=4)) as (base_url, _),
        pytest.raises(MatrixCacheError, match="could not be read"),
    ):
        build_matrices(tiny_instance, run_config(tmp_path, osrm_url=base_url))


def test_a_readable_but_invalid_cache_entry_does_not_downgrade_the_provider(
    tiny_instance: Instance, tmp_path: Path
) -> None:
    """The other half of the same invariant: valid parquet holding an impossible matrix.

    The file reads back cleanly, so it is :class:`CostMatrices` validation that rejects it — and
    that raises :class:`MatrixProviderError`, which is exactly what the OSRM fallback catches.
    It has to be re-typed on the way out, or a bad cache file would swap road distances for
    great-circle estimates without a word.
    """
    key = MatrixCacheKey(
        seed=tiny_instance.seed,
        provider_name="osrm",
        n_nodes=tiny_instance.n_nodes,
        coord_digest=tiny_instance.coordinate_digest(),
    )
    negative = np.full((tiny_instance.n_nodes, tiny_instance.n_nodes), -1.0)
    write_cache(tmp_path, key, (negative, negative))

    with (
        osrm_stub(StubState(max_table_size=4)) as (base_url, _),
        pytest.raises(MatrixCacheError, match="is invalid"),
    ):
        build_matrices(tiny_instance, run_config(tmp_path, osrm_url=base_url))
