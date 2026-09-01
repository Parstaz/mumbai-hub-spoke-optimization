"""Tests for the parquet matrix cache.

Two things matter here. The round trip must be exact — a lossy cache would make a resumed run
disagree with a fresh one, and the repository's whole claim is that its numbers are reproducible.
And a damaged or mislabelled entry must be rejected loudly rather than rebuilt over in silence,
because a cache that quietly heals itself hides the bug that damaged it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.costs.cache import MatrixCacheKey, read_cache, write_cache
from src.exceptions import MatrixCacheError

KEY = MatrixCacheKey(seed=42, provider_name="osrm", n_nodes=3, coord_digest="deadbeefdeadbeef")


def sample_matrices(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """A distinguishable, non-symmetric pair so a transposed round trip would be visible."""
    distance = np.arange(n_nodes * n_nodes, dtype=np.float64).reshape(n_nodes, n_nodes)
    return distance, distance * 0.37


def test_cache_key_filename_carries_every_component() -> None:
    """The filename is the index: ``ls`` has to explain what an entry is."""
    assert KEY.filename == "matrix_seed42_osrm_n3_deadbeefdeadbeef.parquet"
    assert KEY.path_in(Path("data/cache")).parent == Path("data/cache")


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    """A cold start is the ordinary case, not an error."""
    assert read_cache(tmp_path, KEY) is None


def test_cache_round_trip_is_exact(tmp_path: Path) -> None:
    """Bit-for-bit, including the row-major layout that separates a matrix from its transpose."""
    distance, duration = sample_matrices(3)
    write_cache(tmp_path, KEY, (distance, duration))

    loaded = read_cache(tmp_path, KEY)
    assert loaded is not None
    assert np.array_equal(loaded[0], distance)
    assert np.array_equal(loaded[1], duration)
    assert loaded[0].shape == (3, 3)


def test_write_cache_creates_the_directory(tmp_path: Path) -> None:
    """A first run must not require the cache directory to exist already."""
    nested = tmp_path / "does" / "not" / "exist"
    path = write_cache(nested, KEY, sample_matrices(3))
    assert path.is_file()


def test_cache_keys_differ_on_every_component() -> None:
    """Seed, provider, node count and geometry each have to change the filename."""
    variants = {
        KEY.filename,
        MatrixCacheKey(43, "osrm", 3, "deadbeefdeadbeef").filename,
        MatrixCacheKey(42, "haversine", 3, "deadbeefdeadbeef").filename,
        MatrixCacheKey(42, "osrm", 4, "deadbeefdeadbeef").filename,
        MatrixCacheKey(42, "osrm", 3, "0123456789abcdef").filename,
    }
    assert len(variants) == 5


def test_cache_rejects_an_unreadable_file(tmp_path: Path) -> None:
    """Garbage on disk raises, so a corrupt cache cannot be mistaken for a cold start."""
    KEY.path_in(tmp_path).write_bytes(b"not a parquet file")
    with pytest.raises(MatrixCacheError, match="could not be read"):
        read_cache(tmp_path, KEY)


def test_cache_rejects_a_file_whose_size_contradicts_its_key(tmp_path: Path) -> None:
    """A truncated entry named for 3 nodes but holding 2 must not be reshaped into nonsense."""
    write_cache(tmp_path, KEY, sample_matrices(2))
    with pytest.raises(MatrixCacheError, match="holds 2 nodes but the key names 3"):
        read_cache(tmp_path, KEY)


def test_cache_rejects_a_non_square_cell_count(tmp_path: Path) -> None:
    """A row count that is not a perfect square cannot be a matrix at all."""
    table = pa.table({"distance_m": np.zeros(5), "duration_s": np.zeros(5)})
    pq.write_table(table, KEY.path_in(tmp_path))
    with pytest.raises(MatrixCacheError, match="not a positive square"):
        read_cache(tmp_path, KEY)


def test_cache_rejects_a_missing_column(tmp_path: Path) -> None:
    """Both matrices live in one file precisely so half a pair cannot be observed."""
    pq.write_table(pa.table({"distance_m": np.zeros(9)}), KEY.path_in(tmp_path))
    with pytest.raises(MatrixCacheError, match="missing column"):
        read_cache(tmp_path, KEY)


def test_cache_rejects_a_non_numeric_column(tmp_path: Path) -> None:
    """A column of the wrong type is a corrupt entry, not something to coerce."""
    text_key = MatrixCacheKey(42, "osrm", 2, "deadbeefdeadbeef")
    table = pa.table(
        {
            "distance_m": pa.array(["a", "b", "c", "d"], type=pa.string()),
            "duration_s": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        }
    )
    pq.write_table(table, text_key.path_in(tmp_path))
    with pytest.raises(MatrixCacheError, match="non-numeric"):
        read_cache(tmp_path, text_key)


def test_cache_rejects_nulls(tmp_path: Path) -> None:
    """A null cell would reload as NaN and poison every route that crossed it."""
    null_key = MatrixCacheKey(42, "osrm", 2, "deadbeefdeadbeef")
    table = pa.table(
        {
            "distance_m": pa.array([1.0, None, 3.0, 4.0], type=pa.float64()),
            "duration_s": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        }
    )
    pq.write_table(table, null_key.path_in(tmp_path))
    with pytest.raises(MatrixCacheError, match="nulls"):
        read_cache(tmp_path, null_key)
