"""On-disk persistence for an assembled cost matrix.

A 1116-node instance is a 1.25-million-cell table that takes ~125 chunked OSRM requests to
assemble. The genetic algorithm reads that table millions of times per run, so it has to be a
NumPy array in memory and it must be built exactly once. Without this cache, matrix I/O — not
search — dominates every runtime number the repository reports.

The key is ``(seed, provider, n_nodes, coordinate digest)``. The digest is what makes the cache
safe: two runs can share a seed and a node count while placing nodes differently (a changed
bounding box, a changed candidate pool), and reusing one's matrix for the other would be a wrong
answer rather than a slow one. The provider is in the key because a haversine matrix and an OSRM
matrix are different data, and a fallback run must never load a cache written by a real
road-network run.

Storage is parquet columnar: each matrix is flattened row-major into one column, and the side
length is recovered from the row count and checked against the key. Both columns live in one
file so a half-written pair cannot be observed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from src.exceptions import MatrixCacheError
from src.units import DistanceMatrix, DurationMatrix, MatrixPair

logger = logging.getLogger(__name__)

_DISTANCE_COLUMN = "distance_m"
_DURATION_COLUMN = "duration_s"
_COMPRESSION = "zstd"


@dataclass(frozen=True, slots=True)
class MatrixCacheKey:
    """Everything that has to match for a stored matrix to be the right one to reuse.

    Rendered into the filename rather than an index file: the cache is then self-describing,
    ``ls data/cache`` explains itself, and invalidating one entry is ``rm``.
    """

    seed: int
    provider_name: str
    n_nodes: int
    coord_digest: str

    @property
    def filename(self) -> str:
        """The single file this key addresses."""
        return (
            f"matrix_seed{self.seed}_{self.provider_name}"
            f"_n{self.n_nodes}_{self.coord_digest}.parquet"
        )

    def path_in(self, cache_dir: Path) -> Path:
        """Resolve this key against a cache directory."""
        return cache_dir / self.filename


def read_cache(cache_dir: Path, key: MatrixCacheKey) -> MatrixPair | None:
    """Load the matrices for ``key``, or ``None`` if they have not been built yet.

    A missing file is the ordinary cold-start case and is not an error. A file that exists but
    cannot be read, or whose contents contradict the key naming it, *is* an error: silently
    rebuilding over a corrupt entry would hide a real problem behind a slow run.

    Args:
        cache_dir: Directory the cache lives in; need not exist.
        key: Identity of the matrices wanted.

    Returns:
        ``(distance_metres, duration_seconds)``, or ``None`` on a cache miss.

    Raises:
        MatrixCacheError: If the file exists but is unreadable, malformed, or disagrees with
            ``key`` about the number of nodes.
    """
    path = key.path_in(cache_dir)
    if not path.is_file():
        return None

    table = _read_table(path)
    n_nodes = _side_length(path, table.num_rows)
    if n_nodes != key.n_nodes:
        raise MatrixCacheError(
            f"cached matrix {path} holds {n_nodes} nodes but the key names {key.n_nodes}; "
            f"delete the file to rebuild"
        )

    distance: DistanceMatrix = _column(path, table, _DISTANCE_COLUMN).reshape(n_nodes, n_nodes)
    duration: DurationMatrix = _column(path, table, _DURATION_COLUMN).reshape(n_nodes, n_nodes)
    logger.info("loaded %d×%d %s matrices from %s", n_nodes, n_nodes, key.provider_name, path)
    return distance, duration


def write_cache(cache_dir: Path, key: MatrixCacheKey, matrices: MatrixPair) -> Path:
    """Persist ``matrices`` under ``key`` and return the path written.

    Args:
        cache_dir: Directory to write into; created if absent.
        key: Identity to store the matrices under.
        matrices: ``(distance_metres, duration_seconds)``, both square and the same shape.

    Returns:
        The path written.
    """
    distance, duration = matrices
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = key.path_in(cache_dir)
    table = pa.table(
        {
            _DISTANCE_COLUMN: np.ravel(distance),
            _DURATION_COLUMN: np.ravel(duration),
        }
    )
    pq.write_table(table, path, compression=_COMPRESSION)
    logger.info("cached %s matrices to %s", key.provider_name, path)
    return path


def _read_table(path: Path) -> pa.Table:
    """Read the parquet file, translating any reader failure into a domain error.

    ``Exception`` is caught deliberately: pyarrow raises a family of ``ArrowInvalid`` /
    ``ArrowIOError`` types plus plain ``OSError`` depending on how the file is damaged, and the
    caller's response is the same for all of them. It is re-raised, never swallowed.
    """
    try:
        table: pa.Table = pq.read_table(path)
    except Exception as exc:
        raise MatrixCacheError(f"cached matrix {path} could not be read: {exc}") from exc
    missing = {_DISTANCE_COLUMN, _DURATION_COLUMN} - set(table.column_names)
    if missing:
        raise MatrixCacheError(f"cached matrix {path} is missing column(s) {sorted(missing)}")
    return table


def _side_length(path: Path, num_rows: int) -> int:
    """Recover the matrix side length from the flattened row count."""
    n_nodes = math.isqrt(num_rows)
    if n_nodes * n_nodes != num_rows or n_nodes == 0:
        raise MatrixCacheError(
            f"cached matrix {path} holds {num_rows} cells, which is not a positive square"
        )
    return n_nodes


def _column(path: Path, table: pa.Table, name: str) -> npt.NDArray[np.float64]:
    """Extract one column as a flat float64 array.

    Args:
        path: Source file, for the error message only.
        table: The parquet table read from ``path``.
        name: Column to extract.

    Returns:
        A flat ``float64`` array of the column's values.

    Raises:
        MatrixCacheError: If the column holds nulls or values outside the float domain.
    """
    column = table.column(name)
    if column.null_count:
        raise MatrixCacheError(f"cached matrix {path} has {column.null_count} nulls in {name!r}")
    try:
        values: npt.NDArray[np.float64] = np.asarray(
            column.to_numpy(zero_copy_only=False), dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise MatrixCacheError(f"cached matrix {path} has non-numeric {name!r}: {exc}") from exc
    return values
