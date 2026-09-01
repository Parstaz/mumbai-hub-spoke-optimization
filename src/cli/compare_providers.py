"""Print distances between known Mumbai landmarks under both providers, for eyeball validation.

This is the verification step for the cost layer. A distance matrix is the one artefact in the
pipeline that can be wrong by 30% without anything downstream complaining: every solver still
runs, every cost still balances, and the results table is quietly meaningless. Numbers between
places a reader knows are the cheapest check that the units, the coordinate order and the
circuity factor are all right.

Two things are worth looking at in the output. The **OSRM / crow** column is the circuity the
road network actually imposes; if it clusters far from ``RunConfig.circuity_factor``, the
fallback is miscalibrated and should be changed. And the **traffic** section at the bottom shows
one leg driven at four departure times, which is where a band multiplier that was never applied
shows up as four identical numbers.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import METRES_PER_KM, SECONDS_PER_HOUR, SECONDS_PER_MINUTE, RunConfig, TrafficConfig
from src.costs.matrix import HaversineProvider, OSRMProvider, haversine_matrix
from src.costs.traffic import TrafficModel
from src.data.instance import Coordinate
from src.exceptions import MatrixProviderError
from src.units import Coordinates, MatrixPair, Seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Landmark:
    """A recognisable point, used only as a reference for human judgement."""

    name: str
    coord: Coordinate


LANDMARKS: tuple[Landmark, ...] = (
    Landmark("Gateway of India", Coordinate(18.9220, 72.8347)),
    Landmark("CSMT station", Coordinate(18.9398, 72.8355)),
    Landmark("Bandra Kurla Complex", Coordinate(19.0662, 72.8685)),
    Landmark("Juhu Beach", Coordinate(19.0990, 72.8265)),
    Landmark("Airport Terminal 2", Coordinate(19.0896, 72.8656)),
    Landmark("Powai / IIT Bombay", Coordinate(19.1334, 72.9133)),
    Landmark("Vashi, Navi Mumbai", Coordinate(19.0771, 73.0169)),
    Landmark("Thane station", Coordinate(19.1860, 72.9750)),
)
"""All inside the configured bounding box, and spread across the island, the suburbs and the
mainland so the harbour crossings — where circuity is worst — are represented."""

PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),  # a two-kilometre hop: catches a units error immediately
    (1, 4),  # the classic airport run
    (2, 5),  # suburb to suburb, no water
    (3, 6),  # west coast to the mainland, across the harbour
    (4, 6),  # airport to Navi Mumbai
    (0, 7),  # the long diagonal, most of the bounding box
)

TRAFFIC_DEPARTURES: tuple[float, ...] = (7.0, 9.0, 10.75, 13.0, 18.0)
"""07:00 overnight, 09:00 peak, 10:45 straddling the 11:00 boundary, 13:00 midday, 18:00 peak."""

_NAME_WIDTH = 44
_ROW = "{name:<44}{crow:>9.2f}{hav_km:>9.2f}{hav_min:>9.1f}{osrm_km:>10}{osrm_min:>10}{ratio:>11}"
_HEADER = (
    f"{'landmark pair':<44}{'crow km':>9}{'hav km':>9}{'hav min':>9}"
    f"{'OSRM km':>10}{'OSRM min':>10}{'OSRM/crow':>11}"
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osrm-url", default=defaults.osrm_url, help="OSRM base URL")
    parser.add_argument(
        "--no-osrm", action="store_true", help="skip OSRM and report the fallback alone"
    )
    return parser.parse_args(argv)


def landmark_coords() -> Coordinates:
    """The landmarks as the ``(n, 2)`` lat/lon array both providers take."""
    return np.array([(mark.coord.lat, mark.coord.lon) for mark in LANDMARKS], dtype=np.float64)


def _osrm_matrices(coords: Coordinates, run: RunConfig) -> MatrixPair | None:
    """Query OSRM, or return ``None`` with an explanation if it is not answering.

    The comparison is still worth printing without OSRM — it validates the fallback's own units —
    so an outage degrades this script rather than failing it.
    """
    provider = OSRMProvider(
        base_url=run.osrm_url,
        max_table_size=run.osrm_max_table_size,
        timeout_s=run.osrm_timeout_s,
    )
    try:
        return provider.matrix(coords)
    except MatrixProviderError as exc:
        print(f"\n  OSRM unavailable, reporting haversine only: {exc}")
        print("  Start it with `make osrm`; see the README for the one-time extract build.\n")
        return None


def _row(pair: tuple[int, int], crow: float, haversine: MatrixPair, osrm: MatrixPair | None) -> str:
    """Format one landmark pair as a table row."""
    origin, destination = pair
    name = f"{LANDMARKS[origin].name} → {LANDMARKS[destination].name}"
    crow_km = crow / METRES_PER_KM
    return _ROW.format(
        name=name[: _NAME_WIDTH - 1],
        crow=crow_km,
        hav_km=haversine[0][origin, destination] / METRES_PER_KM,
        hav_min=haversine[1][origin, destination] / SECONDS_PER_MINUTE,
        osrm_km="-" if osrm is None else f"{osrm[0][origin, destination] / METRES_PER_KM:.2f}",
        osrm_min="-"
        if osrm is None
        else f"{osrm[1][origin, destination] / SECONDS_PER_MINUTE:.1f}",
        ratio="-" if osrm is None else f"{osrm[0][origin, destination] / crow:.2f}",
    )


def _traffic_lines(base_duration_s: Seconds, traffic: TrafficModel) -> list[str]:
    """Show one leg driven at several departure times, so the bands are visible as numbers."""
    lines = []
    for hour in TRAFFIC_DEPARTURES:
        departure = Seconds(hour * SECONDS_PER_HOUR)
        elapsed = traffic.travel_time_with_traffic(base_duration_s, departure)
        arrival_h = (departure + elapsed) / SECONDS_PER_HOUR
        lines.append(
            f"  depart {int(hour):02d}:{round(hour % 1 * 60):02d}  "
            f"multiplier at departure {traffic.multiplier_for_hour(int(hour)):.1f}  "
            f"→ {elapsed / SECONDS_PER_MINUTE:6.1f} min  "
            f"(effective {elapsed / base_duration_s:.2f}×, arrives {arrival_h:05.2f}h)"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print the provider comparison and the traffic demonstration. Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    run = RunConfig(osrm_url=args.osrm_url, use_osrm=not args.no_osrm)

    coords = landmark_coords()
    crow = haversine_matrix(coords)
    haversine = HaversineProvider(
        circuity_factor=run.circuity_factor, speed_kmph=run.haversine_speed_kmph
    ).matrix(coords)
    osrm = None if args.no_osrm else _osrm_matrices(coords, run)

    print(
        f"\n  haversine fallback: × {run.circuity_factor} circuity "
        f"at {run.haversine_speed_kmph} km/h"
    )
    print(f"  OSRM endpoint:      {run.osrm_url if not args.no_osrm else 'skipped'}\n")
    print(f"  {_HEADER}")
    print(f"  {'-' * len(_HEADER)}")
    for pair in PAIRS:
        print(f"  {_row(pair, float(crow[pair]), haversine, osrm)}")

    if osrm is not None:
        ratios = [osrm[0][pair] / crow[pair] for pair in PAIRS]
        print(
            f"\n  observed circuity: min {min(ratios):.2f}, mean {sum(ratios) / len(ratios):.2f}, "
            f"max {max(ratios):.2f} — compare against circuity_factor = {run.circuity_factor}"
        )

    print("\n  traffic, applied to a 30-minute free-flow leg:")
    for line in _traffic_lines(Seconds(1800.0), TrafficModel.from_config(TrafficConfig())):
        print(line)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
