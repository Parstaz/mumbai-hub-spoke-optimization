"""Compare Stage 1's inbound leg against the greedy baseline's, under both assignment strategies.

This is the verification step for Stage 1, and it is deliberately a three-column table rather
than a single number, because two separate changes are being measured and they must not be
allowed to hide each other:

* **baseline → nearest** is the CVRP's contribution. Same hub assignment as the greedy benchmark,
  better tours.
* **nearest → balanced** is the assignment's contribution. Same solver, hubs capped so no one of
  them draws far more than its share.

What to look at. The hub spread rows answer whether the imbalance is gone: nearest-hub draws
8,850 kg to hub 9 on the default instance while two hubs draw 300 kg each. The distance and cost
rows answer whether removing that imbalance was worth anything — a question this table is allowed
to answer in the negative, and on the default instance it does. Note also that the cap is an upper
bound and not a lower one: it can stop a hub being oversubscribed but cannot conjure work for a
hub with nothing near it, so the minimum stays low however tight the cap.

Every rupee here comes from :func:`~src.scoring.stage_cost`, the same fold
:func:`~src.scoring.evaluate_solution` uses. Nothing in this file prices anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Sequence

import numpy as np

from src.baseline.greedy import solve_baseline
from src.config import (
    METRES_PER_KM,
    SECONDS_PER_HOUR,
    Config,
    RunConfig,
    Stage1Config,
)
from src.costs.matrix import build_matrices
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.data.instance import Instance
from src.scoring import stage_cost
from src.solution import Route
from src.stage1.assignment import AssignmentStrategy, capacity_balanced, unconstrained
from src.stage1.cvrp import solve_stage1
from src.units import DemandArray

logger = logging.getLogger(__name__)

_LABEL_WIDTH = 22
_COLUMN_WIDTH = 18
"""Wide enough for the longest cell, ``300/8,850 σ2,004``, plus a separating space."""

STRATEGIES: dict[str, AssignmentStrategy] = {
    "nearest": unconstrained,
    "balanced": capacity_balanced,
}
"""The ablation's two arms, by CLI name.

Kept here rather than in :mod:`src.stage1.assignment` on purpose: a registry inside the library
for two functions is speculative abstraction, and the only thing that needs to look a strategy up
by string is an argument parser.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class InboundSummary:
    """One column of the table: what an inbound plan cost and how it loaded the hubs."""

    label: str
    tours: int
    distance_km: float
    duration_hr: float
    cost_inr: float
    stops: int
    hub_stops: DemandArray
    hub_kg: DemandArray


def summarise(
    label: str, routes: tuple[Route, ...], instance: Instance, config: Config
) -> InboundSummary:
    """Reduce one inbound plan to the figures the table reports.

    Distance and duration are summed off the routes because they are physical facts the routing
    layer already produced. The money comes from :func:`~src.scoring.stage_cost` and from nowhere
    else.
    """
    cost = stage_cost(routes, instance, config.cost)
    n_hubs = len(instance.hubs)
    hub_stops = np.zeros(n_hubs, dtype=np.float64)
    hub_kg = np.zeros(n_hubs, dtype=np.float64)
    for route in routes:
        hub_stops[route.hub_id] += route.n_stops
        hub_kg[route.hub_id] += route.load_kg
    return InboundSummary(
        label=label,
        tours=len(routes),
        distance_km=sum(route.distance_m for route in routes) / METRES_PER_KM,
        duration_hr=sum(route.duration_s for route in routes) / SECONDS_PER_HOUR,
        cost_inr=cost.breakdown.total_inr,
        stops=cost.stops,
        hub_stops=hub_stops,
        hub_kg=hub_kg,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface."""
    run, stage1 = RunConfig(), Stage1Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=run.seed, help="run seed")
    parser.add_argument("--osrm-url", default=run.osrm_url, help="OSRM base URL")
    parser.add_argument(
        "--no-osrm", action="store_true", help="skip OSRM and use the haversine fallback"
    )
    parser.add_argument(
        "--strategy",
        choices=(*STRATEGIES, "both"),
        default="both",
        help="hub assignment strategy to solve with; 'both' reports the full ablation",
    )
    parser.add_argument(
        "--slack",
        type=float,
        default=stage1.hub_balance_slack,
        help="hub_balance_slack: multiplier on the even share of sources per hub",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=stage1.cvrp_time_limit_s,
        help="OR-Tools time limit per hub, in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=stage1.workers,
        help="per-hub process pool size; 0 means one per CPU",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="stop at the first-solution heuristic, making the solve reproducible",
    )
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> Config:
    """Assemble the run's configuration from the parsed arguments."""
    return Config(
        run=RunConfig(seed=args.seed, osrm_url=args.osrm_url, use_osrm=not args.no_osrm),
        stage1=Stage1Config(
            hub_balance_slack=args.slack,
            cvrp_time_limit_s=args.time_limit,
            cvrp_solution_limit=1 if args.deterministic else 0,
            workers=args.workers,
        ),
    )


def _spread(values: DemandArray) -> str:
    """Render the min, max and standard deviation of a per-hub quantity.

    The minimum is taken over hubs that received work, so a hub the assignment gave nothing to
    does not read as a hub loaded with zero. The deviation is over all of them, because an idle
    hub is part of how uneven the plan is.
    """
    active = values[values > 0.0]
    if not active.size:
        return "—"
    return f"{active.min():,.0f}/{active.max():,.0f} σ{values.std():,.0f}"


def _row(label: str, cells: list[str]) -> str:
    """One table row: a left-aligned label followed by right-aligned columns."""
    body = "".join(f"{cell:>{_COLUMN_WIDTH}}" for cell in cells)
    return f"{label:<{_LABEL_WIDTH}}{body}"


def table_lines(summaries: list[InboundSummary]) -> list[str]:
    """Render the comparison table, hub spread first and cost last.

    The spread rows lead because they are what the assignment strategy is *for*; the cost row is
    last because it is the verdict on whether that was worth paying for.
    """
    return [
        _row("", [summary.label for summary in summaries]),
        _row("sources per hub", [_spread(s.hub_stops) for s in summaries]),
        _row("kg per hub", [_spread(s.hub_kg) for s in summaries]),
        "",
        _row("inbound tours", [f"{s.tours:,d}" for s in summaries]),
        _row("sources collected", [f"{s.stops:,d}" for s in summaries]),
        _row("inbound distance", [f"{s.distance_km:,.1f} km" for s in summaries]),
        _row("inbound duration", [f"{s.duration_hr:,.1f} h" for s in summaries]),
        _row("inbound cost", [f"₹{s.cost_inr:,.0f}" for s in summaries]),
        _row("vs baseline", [_delta(s, summaries[0]) for s in summaries]),
    ]


def _delta(summary: InboundSummary, reference: InboundSummary) -> str:
    """This column's inbound cost as a percentage change against the baseline column."""
    if summary is reference:
        return "—"
    change = summary.cost_inr / reference.cost_inr - 1.0
    return f"{change:+.1%}"


def verdict_lines(summaries: list[InboundSummary]) -> list[str]:
    """State what the table shows, read off the numbers rather than assumed.

    Written to be capable of reporting that the capacity-balanced assignment lost. It is a
    measurement, not a thesis, and the README's limitations depend on this printing the truth.
    """
    by_label = {summary.label: summary for summary in summaries}
    nearest, balanced = by_label.get("nearest"), by_label.get("balanced")
    if nearest is None or balanced is None:
        return []
    change = balanced.cost_inr / nearest.cost_inr - 1.0
    direction = "cheaper" if change < 0.0 else "dearer"
    return [
        f"Capping hub intake took the worst hub from {nearest.hub_kg.max():,.0f} kg to "
        f"{balanced.hub_kg.max():,.0f} kg,",
        f"and made the inbound leg {abs(change):.1%} {direction}: "
        f"{balanced.distance_km - nearest.distance_km:+,.0f} km, "
        f"{balanced.tours - nearest.tours:+d} vehicles.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Solve Stage 1 under the requested strategies and print the comparison. Returns exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    config = _config(args)

    instance = generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)
    matrices = build_matrices(instance, config.run)
    traffic = TrafficModel.from_config(config.traffic)

    summaries = [
        summarise(
            "baseline", solve_baseline(instance, matrices, traffic).stage1_routes, instance, config
        )
    ]
    chosen = tuple(STRATEGIES) if args.strategy == "both" else (args.strategy,)
    for name in chosen:
        routes = solve_stage1(instance, matrices, traffic, STRATEGIES[name], config)
        summaries.append(summarise(name, routes, instance, config))

    print("\n  Stage 1 inbound leg: greedy benchmark, then the CVRP under each hub assignment.")
    print("  baseline → nearest isolates the solver; nearest → balanced isolates the assignment.\n")
    print(
        f"  {'instance':<{_LABEL_WIDTH}}seed {instance.seed}, digest {instance.coordinate_digest()}"
    )
    provider = "haversine fallback" if not config.run.use_osrm else f"OSRM at {config.run.osrm_url}"
    print(f"  {'distances':<{_LABEL_WIDTH}}{provider}")
    print(
        f"  {'search':<{_LABEL_WIDTH}}{config.stage1.cvrp_time_limit_s:g}s per hub, "
        f"slack {config.stage1.hub_balance_slack:g}"
        f"{', first solution only' if args.deterministic else ', guided local search'}"
    )
    print()
    for line in table_lines(summaries):
        print(f"  {line}" if line else "")
    print()
    for line in verdict_lines(summaries):
        print(f"  {line}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
