"""Run the greedy nearest-neighbour benchmark on a seeded instance and print its metrics.

This is the verification step for the baseline. The numbers it prints are the column every later
step is compared against, so the point of running it by hand is to decide whether they are
*plausible* before anything claims to improve on them.

What to look at, on the default instance: cost per drop in the low hundreds of rupees, and a
non-zero window violation count — a window-blind solver that is never late has a broken clock, not
a good plan. Capacity utilisation is the misleading one. It reads high (~83%) and that is not the
baseline doing well: with uniform 37.5 kg shipments, twenty to a vehicle, filling a vehicle is
trivial and every solver will do it. The greedy's waste is in *distance* — customers are delivered
from whichever hub their shipment happened to reach, so its final-mile tours criss-cross the map —
and that is where the optimized pipeline has to find its improvement.

Every figure here comes from :func:`~src.scoring.evaluate_solution`. Nothing in this file or in
:mod:`src.baseline.greedy` computes a cost.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from src.baseline.greedy import solve_baseline
from src.config import MINUTES_PER_HOUR, SECONDS_PER_MINUTE, Config, CostConfig, RunConfig
from src.costs.matrix import build_matrices
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.data.instance import Instance
from src.scoring import Metrics, evaluate_solution
from src.solution import Solution

logger = logging.getLogger(__name__)

_LABEL_WIDTH = 24
_MONEY_WIDTH = 13


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface."""
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=defaults.seed, help="run seed")
    parser.add_argument(
        "--instance",
        type=Path,
        default=None,
        help="instance JSON to score; by default the seed's instance is regenerated in memory",
    )
    parser.add_argument("--osrm-url", default=defaults.osrm_url, help="OSRM base URL")
    parser.add_argument(
        "--no-osrm", action="store_true", help="skip OSRM and use the haversine fallback"
    )
    return parser.parse_args(argv)


def _load_instance(path: Path | None, config: Config) -> Instance:
    """Read the instance from ``path``, or regenerate the seed's instance in memory.

    Regenerating is the default because generation is deterministic in the seed: it produces
    exactly the instance ``make data`` writes, and it works on a clean checkout where nothing has
    been written yet. ``--instance`` is for scoring a specific artefact on disk.
    """
    if path is not None:
        return Instance.read_json(path)
    return generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)


def _money(amount: float) -> str:
    """Format rupees to two decimal places in a fixed-width column."""
    return f"₹{amount:>{_MONEY_WIDTH},.2f}"


def _component_line(label: str, amount: float, total: float) -> str:
    """One cost component, indented under the total it contributes to, with its share."""
    return f"    {label:<{_LABEL_WIDTH - 4}}{_money(amount)}{amount / total:>9.1%}"


def _clock(hour: float) -> str:
    """Render a fractional clock hour as ``HH:MM``."""
    return f"{int(hour):02d}:{round(hour % 1 * MINUTES_PER_HOUR):02d}"


def context_lines(instance: Instance, run: RunConfig) -> list[str]:
    """Describe what was solved and with what, so a printed table can be traced back to it."""
    geo, fleet, schedule = instance.geo, instance.fleet, instance.schedule
    provider = "haversine fallback (--no-osrm)" if not run.use_osrm else f"OSRM at {run.osrm_url}"
    return [
        f"{'instance':<{_LABEL_WIDTH}}seed {instance.seed}, digest {instance.coordinate_digest()}",
        f"{'nodes':<{_LABEL_WIDTH}}{instance.n_nodes} "
        f"({geo.n_hubs} hubs, {geo.n_sources} sources, {geo.n_customers} customers), "
        f"{len(instance.shipments)} shipments",
        f"{'distances':<{_LABEL_WIDTH}}{provider}",
        f"{'fleet':<{_LABEL_WIDTH}}{fleet.vehicle_capacity_kg:g} kg vehicle, "
        f"{fleet.service_time_per_stop_s / SECONDS_PER_MINUTE:g} min per stop, "
        f"dispatch {_clock(schedule.dispatch_hour)}",
    ]


def metric_lines(metrics: Metrics, solution: Solution, cost: CostConfig) -> list[str]:
    """Render the full :class:`~src.scoring.Metrics` table, headline first.

    Cost per drop leads because it is the KPI the whole repository reports; the four cost
    components sit under the total that produced it, and the operational measures follow as the
    explanation for why the total is what it is.
    """
    total = metrics.total_cost_inr
    breakdown = metrics.breakdown
    drops = sum(route.n_stops for route in solution.stage2_routes)
    return [
        f"{'cost per drop':<{_LABEL_WIDTH}}{_money(metrics.cost_per_drop_inr)}   ← headline KPI",
        f"{'total cost':<{_LABEL_WIDTH}}{_money(total)}",
        _component_line(f"variable ₹{cost.variable_per_km:g}/km", breakdown.variable_inr, total),
        _component_line(f"driver ₹{cost.driver_per_hour:g}/h", breakdown.driver_inr, total),
        _component_line(f"fixed ₹{cost.fixed_per_vehicle:g}/veh", breakdown.fixed_inr, total),
        _component_line(f"late ₹{cost.tw_penalty_per_hour:g}/h", breakdown.tw_penalty_inr, total),
        "",
        f"{'distance':<{_LABEL_WIDTH}}{metrics.total_distance_km:>{_MONEY_WIDTH + 1},.1f} km",
        f"{'duration':<{_LABEL_WIDTH}}{metrics.total_duration_hr:>{_MONEY_WIDTH + 1},.1f} h",
        f"{'vehicle-days':<{_LABEL_WIDTH}}{metrics.vehicles_used:>{_MONEY_WIDTH + 1},d}"
        f"    ({len(solution.stage1_routes)} inbound, {len(solution.stage2_routes)} final-mile)",
        f"{'stops per hour':<{_LABEL_WIDTH}}{metrics.stops_per_hour:>{_MONEY_WIDTH + 1},.2f}",
        f"{'capacity utilisation':<{_LABEL_WIDTH}}"
        f"{metrics.capacity_utilisation:>{_MONEY_WIDTH + 1}.1%}",
        f"{'window violations':<{_LABEL_WIDTH}}{metrics.tw_violations:>{_MONEY_WIDTH + 1},d}"
        f"    of {drops} drops, {metrics.tw_lateness_hr:,.1f} h late in total",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Solve the baseline and print its metrics. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    config = Config(
        run=RunConfig(seed=args.seed, osrm_url=args.osrm_url, use_osrm=not args.no_osrm)
    )

    instance = _load_instance(args.instance, config)
    matrices = build_matrices(instance, config.run)
    traffic = TrafficModel.from_config(config.traffic)

    solution = solve_baseline(instance, matrices, traffic)
    metrics = evaluate_solution(solution, instance, config.cost)

    print("\n  Greedy nearest-neighbour baseline — no consolidation, no local search, no time")
    print("  window awareness. This is the column the optimized pipeline has to beat.\n")
    for line in context_lines(instance, config.run):
        print(f"  {line}")
    print()
    for line in metric_lines(metrics, solution, config.cost):
        print(f"  {line}" if line else "")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
