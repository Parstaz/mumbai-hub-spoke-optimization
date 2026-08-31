"""Generate a seeded instance, report its shape, and render the geography.

This is the verification step for the foundation layer: it exercises generation, JSON
round-tripping and plotting, and prints the numbers a reader needs to judge whether the
synthetic instance is plausible before any solver touches it.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from src.config import (
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    FleetConfig,
    GeoConfig,
    RunConfig,
    ScheduleConfig,
)
from src.data.generate import generate_instance
from src.data.instance import Instance
from src.exceptions import InstanceError
from src.plots.instance_scatter import THEMES, plot_instance

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="run seed")
    parser.add_argument("--out", type=Path, default=None, help="instance JSON path")
    parser.add_argument(
        "--theme",
        choices=("light", "dark", "both"),
        default="light",
        help="colour theme for the scatter plot",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip figure rendering")
    return parser.parse_args(argv)


def summarise(instance: Instance) -> list[str]:
    """Build the human-readable summary of an instance, one line per fact."""
    geo, fleet, schedule = instance.geo, instance.fleet, instance.schedule
    windowed = [c.window for c in instance.customers if c.window is not None]
    widths_hr = [(w.end_s - w.start_s) / SECONDS_PER_HOUR for w in windowed]
    per_source = Counter(shipment.source_id for shipment in instance.shipments)
    load_kg = sum(shipment.size_kg for shipment in instance.shipments)
    hub_lats = [hub.coord.lat for hub in instance.hubs]
    hub_lons = [hub.coord.lon for hub in instance.hubs]

    return [
        f"seed                  {instance.seed}",
        f"digest                {instance.coordinate_digest()}",
        f"bounding box          lat {geo.lat_min:.2f}–{geo.lat_max:.2f}, "
        f"lon {geo.lon_min:.2f}–{geo.lon_max:.2f}",
        f"nodes                 {instance.n_nodes} "
        f"({geo.n_hubs} hubs, {geo.n_sources} sources, {geo.n_customers} customers)",
        f"hub extent            lat {min(hub_lats):.3f}–{max(hub_lats):.3f}, "
        f"lon {min(hub_lons):.3f}–{max(hub_lons):.3f}",
        f"shipments             {len(instance.shipments)} × {fleet.shipment_size_kg:g} kg "
        f"= {load_kg:,.0f} kg",
        f"per source            min {min(per_source.values())}, "
        f"mean {len(instance.shipments) / geo.n_sources:.2f}, max {max(per_source.values())} "
        f"({geo.n_sources - len(per_source)} unused)",
        f"vehicle               {fleet.vehicle_capacity_kg:g} kg capacity, "
        f"{fleet.shipments_per_vehicle} shipments, "
        f"{fleet.service_time_per_stop_s / SECONDS_PER_MINUTE:g} min per stop",
        f"fleet lower bound     {math.ceil(load_kg / fleet.vehicle_capacity_kg)} vehicles "
        f"(capacity only, ignores geography and time)",
        f"time windows          {len(windowed)} windowed, "
        f"{geo.n_customers - len(windowed)} all-day "
        f"(target {schedule.all_day_fraction:.0%} all-day)",
        f"window width          mean {sum(widths_hr) / len(widths_hr):.2f} h "
        f"over {schedule.day_start_hour:g}:00–{schedule.day_end_hour:g}:00",
    ]


def _render_plots(instance: Instance, theme_name: str, figure_dir: Path) -> list[Path]:
    """Render the requested theme variants and return the paths written."""
    names = tuple(THEMES) if theme_name == "both" else (theme_name,)
    return [
        plot_instance(
            instance,
            figure_dir / f"instance_seed{instance.seed}_{name}.png",
            THEMES[name],
        )
        for name in names
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate, persist, verify and plot one instance. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    run = RunConfig(seed=args.seed)

    instance = generate_instance(GeoConfig(), FleetConfig(), ScheduleConfig(), args.seed)
    destination = args.out or run.data_dir / f"instance_seed{args.seed}.json"
    instance.write_json(destination)

    # The instance on disk is the artefact a results table is defensible against, so verify the
    # round trip here rather than trusting it.
    if Instance.read_json(destination) != instance:
        raise InstanceError(f"instance at {destination} does not round-trip through JSON")

    print(f"\nInstance written to {destination} (JSON round-trip verified)\n")
    for line in summarise(instance):
        print(f"  {line}")

    if not args.no_plot:
        for path in _render_plots(instance, args.theme, run.figure_dir):
            print(f"\nFigure written to {path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
