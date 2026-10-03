"""Run the full Stage 1 + Stage 2 pipeline on one seed and print it beside the baseline.

This is step 6's verification step, and it is deliberately a two-column table rather than a single
number. The greedy benchmark is re-solved in the same process, over the same instance and the same
matrices, so the comparison cannot drift: both columns come from
:func:`~src.scoring.evaluate_solution`, and nothing in this file prices anything.

What to look at. **Cost per drop** is the headline. Beneath it, distance and lateness are where the
improvement is expected to come from — step 4 measured the baseline at 36.4% distance and 9.6%
lateness — while **vehicle-days** is the one to read sceptically. Both stages already sit at their
per-hub mass floors, so there is very little fleet to save, and an optimized plan that deploys
*more* vehicles than the benchmark is not necessarily wrong: the split DAG trades a fixed vehicle
charge against distance and lateness, and buying a vehicle to avoid a long late tour can be
correct. It is reported rather than suppressed.

The ``--strategy`` flag reaches Stage 2, not just Stage 1. Which hub assignment runs inbound
decides how the customers distribute across hubs on the way out — 236 stops at the largest hub
under ``nearest`` against 73 under ``balanced`` on seed 42 — and whether that trade pays *overall*
is step 7's question. This entry point exists to let step 7 measure it rather than rediscover it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from src.baseline.greedy import solve_baseline
from src.cli.ablation import Problem
from src.cli.format import delta, generations_used
from src.cli.reference import report_lines
from src.cli.run_baseline import context_lines
from src.cli.run_stage1 import STRATEGIES
from src.config import (
    Config,
    CostConfig,
    GAConfig,
    ReferenceConfig,
    RunConfig,
    Stage1Config,
)
from src.costs.matrix import CostMatrices, build_matrices
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance
from src.data.instance import Instance
from src.scoring import Metrics, evaluate_solution
from src.solution import Solution
from src.stage1.assignment import AssignmentStrategy
from src.stage1.cvrp import solve_stage1
from src.stage2.ga import HubOutcome
from src.stage2.ortools_reference import matched_run, solve_reference
from src.stage2.solve import Stage2Plan, hub_of_source, solve_stage2
from src.workload import hub_of_customer

logger = logging.getLogger(__name__)

_LABEL_WIDTH = 24
_COLUMN_WIDTH = 16


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface."""
    run, stage1, ga = RunConfig(), Stage1Config(), GAConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=run.seed, help="run seed")
    parser.add_argument("--osrm-url", default=run.osrm_url, help="OSRM base URL")
    parser.add_argument(
        "--no-osrm", action="store_true", help="skip OSRM and use the haversine fallback"
    )
    parser.add_argument(
        "--strategy",
        choices=tuple(STRATEGIES),
        default="nearest",
        help="hub assignment strategy, applied to both stages",
    )
    parser.add_argument(
        "--generations", type=int, default=ga.generations, help="GA generations per hub"
    )
    parser.add_argument(
        "--population", type=int, default=ga.population_size, help="GA population per hub"
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help="disable the memetic local search: step 7's ablation arm",
    )
    parser.add_argument(
        "--workers", type=int, default=stage1.workers, help="per-hub pool size; 0 means one per CPU"
    )
    parser.add_argument(
        "--trace-generations",
        action="store_true",
        help="log per generation whether the incumbent moved and whether anything could have "
        "moved it; roughly doubles the run and does not change its answer",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="stop Stage 1 at the first-solution heuristic, making the inbound leg reproducible",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="after the pipeline, re-solve Stage 2 with OR-Tools under the GA's own per-hub wall "
        "clock and report the gap; step 8's quality benchmark, never part of the pipeline",
    )
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> Config:
    """Assemble the run configuration from the parsed arguments, once, at the entry point."""
    defaults = GAConfig()
    return Config(
        run=RunConfig(seed=args.seed, osrm_url=args.osrm_url, use_osrm=not args.no_osrm),
        stage1=Stage1Config(
            cvrp_solution_limit=1 if args.deterministic else 0, workers=args.workers
        ),
        ga=GAConfig(
            population_size=args.population,
            generations=args.generations,
            local_search_pct=0.0 if args.no_local_search else defaults.local_search_pct,
            trace_generations=args.trace_generations,
        ),
        # --deterministic reaches the reference for the same reason it reaches Stage 1: guided
        # local search under a wall clock is not reproducible, and a run asking for repeatability
        # has to get it from both OR-Tools models or from neither.
        reference=ReferenceConfig(solution_limit=1 if args.deterministic else 0),
    )


def solve_pipeline(
    instance: Instance,
    matrices: CostMatrices,
    traffic: TrafficModel,
    strategy: AssignmentStrategy,
    config: Config,
) -> tuple[Solution, Stage2Plan]:
    """Solve both legs in order and compose them into one plan.

    The composition is the whole point and is one line: Stage 2 delivers from the hub each
    shipment actually reached, read back off Stage 1's tours. That sequential dependence is the
    greedy decomposition the README lists as a known limitation — the assignment optimal for the
    inbound leg need not be optimal for the outbound one — and it is displayed here rather than
    dodged.

    The whole :class:`~src.stage2.solve.Stage2Plan` is returned rather than just its outcomes
    because ``--reference`` needs the per-hub wall clocks of **this** run. Re-solving Stage 2 to
    recover them would match the reference's budget to a different search than the one whose
    routes it is compared against, and would double the runtime on the largest hub.
    """
    inbound = solve_stage1(instance, matrices, traffic, strategy, config)
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, inbound))
    outbound = solve_stage2(instance, matrices, traffic, customer_hubs, config)
    return Solution(stage1_routes=inbound, stage2_routes=outbound.routes), outbound


def budget_lines(hubs: tuple[HubOutcome, ...], ga: GAConfig) -> list[str]:
    """Report the generations actually used against the budget configured.

    Printed rather than left to a limitation further down, because this header is the run's
    self-description and step 9 will quote it. A hub stops when ``stagnation_limit`` generations
    pass without improving its incumbent, and on seed 42 every hub stopped that way well short of
    the budget — so a header claiming the budget was spent would be describing a run that did not
    happen.
    """
    summary = generations_used(hubs, ga)
    if summary is None:
        return []
    lines = [
        f"{'generations used':<{_LABEL_WIDTH}}median {summary.median}, "
        f"range {summary.lowest}-{summary.highest} of {ga.generations} budget",
    ]
    if summary.stopped_early:
        lines.append(
            f"{'':<{_LABEL_WIDTH}}{summary.stopped_early} of {summary.hubs} hubs stopped early on "
            f"stagnation_limit={ga.stagnation_limit}"
        )
    return lines


def comparison_lines(baseline: Metrics, optimized: Metrics, cost: CostConfig) -> list[str]:
    """Render the two columns and the delta between them, headline first.

    The four cost components sit directly under the total because *where* the improvement came
    from is the question this table exists to answer. Step 4 measured the baseline's pools as
    distance 36.4% and lateness 9.6%, so a pipeline winning almost entirely on lateness is
    attacking the smaller one and is worth knowing about — the components say which it is, and the
    headline alone does not.
    """
    greedy, pipeline = baseline.breakdown, optimized.breakdown
    rows = (
        ("cost per drop ₹", baseline.cost_per_drop_inr, optimized.cost_per_drop_inr, ",.2f"),
        ("total cost ₹", baseline.total_cost_inr, optimized.total_cost_inr, ",.0f"),
        (
            f"  variable ₹{cost.variable_per_km:g}/km",
            greedy.variable_inr,
            pipeline.variable_inr,
            ",.0f",
        ),
        (f"  driver ₹{cost.driver_per_hour:g}/h", greedy.driver_inr, pipeline.driver_inr, ",.0f"),
        (f"  fixed ₹{cost.fixed_per_vehicle:g}/veh", greedy.fixed_inr, pipeline.fixed_inr, ",.0f"),
        (
            f"  late ₹{cost.tw_penalty_per_hour:g}/h",
            greedy.tw_penalty_inr,
            pipeline.tw_penalty_inr,
            ",.0f",
        ),
        ("distance km", baseline.total_distance_km, optimized.total_distance_km, ",.1f"),
        ("duration h", baseline.total_duration_hr, optimized.total_duration_hr, ",.1f"),
        ("vehicle-days", float(baseline.vehicles_used), float(optimized.vehicles_used), ",.0f"),
        (
            "window violations",
            float(baseline.tw_violations),
            float(optimized.tw_violations),
            ",.0f",
        ),
        ("lateness h", baseline.tw_lateness_hr, optimized.tw_lateness_hr, ",.1f"),
    )
    header = (
        f"{'':<{_LABEL_WIDTH}}{'greedy':>{_COLUMN_WIDTH}}{'pipeline':>{_COLUMN_WIDTH}}"
        f"{'change':>{_COLUMN_WIDTH}}"
    )
    lines = [header, "-" * (_LABEL_WIDTH + 3 * _COLUMN_WIDTH)]
    for label, before, after, spec in rows:
        lines.append(
            f"{label:<{_LABEL_WIDTH}}{before:>{_COLUMN_WIDTH}{spec}}"
            f"{after:>{_COLUMN_WIDTH}{spec}}{delta(before, after):>{_COLUMN_WIDTH}}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Solve both legs, score them against the baseline, and print the table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    config = _config(args)

    instance = generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)
    matrices = build_matrices(instance, config.run)
    traffic = TrafficModel.from_config(config.traffic)

    baseline = evaluate_solution(solve_baseline(instance, matrices, traffic), instance, config.cost)
    started = time.perf_counter()
    solution, plan = solve_pipeline(instance, matrices, traffic, STRATEGIES[args.strategy], config)
    elapsed_s = time.perf_counter() - started
    optimized = evaluate_solution(solution, instance, config.cost)

    print("\n  Stage 1 CVRP + Stage 2 genetic algorithm, against the greedy benchmark.")
    print("  Both columns are scored by evaluate_solution over the same instance.\n")
    for line in context_lines(instance, config.run):
        print(f"  {line}")
    print(
        f"  {'assignment':<{_LABEL_WIDTH}}{args.strategy}"
        f", population {config.ga.population_size}, generation budget {config.ga.generations}"
        f"{'' if config.ga.local_search_pct else ', no local search'}"
    )
    for line in budget_lines(plan.outcomes, config.ga):
        print(f"  {line}")
    print(f"  {'solve time':<{_LABEL_WIDTH}}{elapsed_s:,.1f} s\n")
    for line in comparison_lines(baseline, optimized, config.cost):
        print(f"  {line}")
    print()

    if args.reference:
        _print_reference(
            Problem(instance=instance, matrices=matrices, traffic=traffic),
            solution,
            plan,
            config,
        )
    return 0


def _print_reference(
    problem: Problem, solution: Solution, plan: Stage2Plan, config: Config
) -> None:
    """Re-solve the final mile with OR-Tools under the GA's budgets, and report the gap.

    Called from :func:`main` and never from :func:`solve_pipeline` — §1.1 keeps the reference
    outside the pipeline, and ``tests/test_closure.py`` proves the solve path cannot reach it. The
    main table is already printed and flushed by the time this starts, so a long reference solve
    does not withhold the result a reader came for.

    ``plan`` is the Stage 2 result whose routes are in the table above, so the budgets handed over
    are the clocks that produced *those* tours. The mapping is read back off the inbound tours the
    GA actually delivered from, the same direction :func:`~src.cli.ablation.inbound_leg` composes
    it, so both solvers serve each customer from the hub its own parcel reached.
    """
    sys.stdout.flush()
    instance = problem.instance
    customer_hubs = hub_of_customer(instance, hub_of_source(instance, solution.stage1_routes))
    started = time.perf_counter()
    reference = solve_reference(
        instance,
        problem.matrices,
        problem.traffic,
        matched_run(customer_hubs, plan.runs),
        config,
    )
    elapsed_s = time.perf_counter() - started

    print("\n  Stage 2 genetic algorithm against an OR-Tools reference, matched budget per hub.")
    print("  Both columns are scored by evaluate_solution over the same inbound plan.\n")
    print(f"  {'reference solve time':<{_LABEL_WIDTH}}{elapsed_s:,.1f} s\n")
    for line in report_lines(
        instance, solution.stage1_routes, solution.stage2_routes, reference, config.cost
    ):
        print(f"  {line}" if line else "")


if __name__ == "__main__":
    raise SystemExit(main())
