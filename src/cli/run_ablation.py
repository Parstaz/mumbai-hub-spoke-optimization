"""Report step 7's ablation: memetic local search on/off × nearest/balanced hub assignment.

:mod:`src.cli.ablation` runs the four arms; this is the entry point that prints them. Nothing here
solves or prices anything — every column comes from :func:`~src.scoring.evaluate_solution`.

**Read the total cost per drop, not the two legs separately.** That is the whole reason the four
arms are one table. Step 4 measured capacity-balanced assignment on the inbound leg alone, found it
1.7% dearer, and could not see that balancing also cuts Stage 2's largest hub from 236 stops to 73.
The per-leg rows here exist to *explain* the total, and are indented with the cost components to
say so; the verdict is computed from the total and from nothing else.

**Generations actually run are reported for every arm.** Every arm stops on ``stagnation_limit``
far short of the 600-generation budget, and a figure quoted against the budget is not a figure
about that configuration. The truncation caveat is printed by the run itself rather than left to
the write-up, because the table is what gets copied out of a terminal and the caveat has to travel
with it.

**That caveat does not assume it is symmetric.** Stagnation truncation is — every arm stops by the
same mechanism, so it biases the absolute level and not the differences. Budget truncation is not
guaranteed to be, and on seed 42 it was not: hub 9 exhausted the 600-generation ceiling in both
no-local-search arms and in neither local-search arm. An arm with hubs cut off at the ceiling was
stopped rather than converged, so a difference measured against it is an *upper bound* on the other
arm's benefit. :func:`truncation_lines` therefore counts the exhausted hubs per arm and only claims
symmetry when the counts agree — an earlier version asserted it unconditionally and was false about
the very run that printed it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from src.baseline.greedy import solve_baseline
from src.cli.ablation import Arm, Priced, Problem, inbound_leg, priced, probe_arms, solve_arms
from src.cli.run_baseline import context_lines
from src.cli.run_pipeline import delta, generations_used
from src.cli.run_stage1 import STRATEGIES
from src.config import Config, CostConfig, GAConfig, RunConfig, Stage1Config
from src.costs.matrix import build_matrices
from src.costs.traffic import TrafficModel
from src.data.generate import generate_instance

logger = logging.getLogger(__name__)

_LABEL_WIDTH = 22
_COLUMN_WIDTH = 14
"""Five columns of this width plus the label stay inside a 100-column terminal."""

_DEFAULT_PROBE_ARM = "nearest"
"""Which arm the noise probe repeats unless ``--probe-arm`` says otherwise.

``nearest`` with local search is the shipping configuration and the arm holding the 236-stop hub,
so it has the largest search space and the most room to vary. Sizing the noise floor on the
noisiest arm bounds it for the other three rather than flattering them.

It is selectable because bounding is not always what is wanted. Establishing that a *difference*
between two arms survives reseeding needs both of them reseeded and compared at matched seeds —
holding one arm at a single seed while varying the other compares a reseeded figure against an
unreseeded one, which invents a gap out of the unreseeded arm's own seed variation.
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define the CLI surface.

    Deliberately offers neither ``--strategy`` nor ``--no-local-search``: running all four arms is
    what this module is for, and a flag that let one be skipped would let the 2×2 be reported with
    a cell missing.
    """
    run, stage1, ga = RunConfig(), Stage1Config(), GAConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=run.seed, help="run seed")
    parser.add_argument("--osrm-url", default=run.osrm_url, help="OSRM base URL")
    parser.add_argument(
        "--no-osrm", action="store_true", help="skip OSRM and use the haversine fallback"
    )
    parser.add_argument(
        "--generations", type=int, default=ga.generations, help="GA generations per hub"
    )
    parser.add_argument(
        "--population", type=int, default=ga.population_size, help="GA population per hub"
    )
    parser.add_argument(
        "--workers", type=int, default=stage1.workers, help="per-hub pool size; 0 means one per CPU"
    )
    parser.add_argument(
        "--noise-probe",
        type=int,
        default=2,
        help="repeats of one arm at other GA seeds, to size the noise floor; 0 skips it",
    )
    parser.add_argument(
        "--probe-arm",
        choices=tuple(STRATEGIES),
        default=_DEFAULT_PROBE_ARM,
        help="which assignment's local-search arm the noise probe reseeds",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="stop Stage 1 at the first-solution heuristic, making the inbound leg reproducible",
    )
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> Config:
    """Assemble the shipping configuration, once, at the entry point.

    ``penalty_warmup_generations`` is left at its default of 0 rather than set here: the four arms
    measure local search and hub assignment, and turning on a second experimental knob would make
    the result attributable to either.
    """
    return Config(
        run=RunConfig(seed=args.seed, osrm_url=args.osrm_url, use_osrm=not args.no_osrm),
        stage1=Stage1Config(
            cvrp_solution_limit=1 if args.deterministic else 0, workers=args.workers
        ),
        ga=GAConfig(population_size=args.population, generations=args.generations),
    )


_Cell = Callable[[Priced], float]

_HEADLINE: tuple[str, _Cell, str] = (
    "cost per drop ₹",
    lambda p: p.metrics.cost_per_drop_inr,
    ",.2f",
)
"""The headline KPI, held apart from the rest so the ``vs greedy`` row can follow it directly."""


def _rows(cost: CostConfig) -> tuple[tuple[str, _Cell, str], ...]:
    """The rows beneath the headline: cost components, then the two legs, then the drivers.

    Built per call rather than as a constant because four of the labels quote the configured rates,
    and a label naming a rate the run did not use would be worse than no label. The two leg rows
    are indented with the components to mark them as decomposition, not as figures to compare.
    """
    breakdown = (
        (f"  variable ₹{cost.variable_per_km:g}/km", "variable_inr"),
        (f"  driver ₹{cost.driver_per_hour:g}/h", "driver_inr"),
        (f"  fixed ₹{cost.fixed_per_vehicle:g}/veh", "fixed_inr"),
        (f"  late ₹{cost.tw_penalty_per_hour:g}/h", "tw_penalty_inr"),
    )
    return (
        ("total cost ₹", lambda p: p.metrics.total_cost_inr, ",.0f"),
        *((label, _component(field), ",.0f") for label, field in breakdown),
        ("  stage 1 inbound ₹", lambda p: p.stage1_inr, ",.0f"),
        ("  stage 2 final mile ₹", lambda p: p.stage2_inr, ",.0f"),
        ("distance km", lambda p: p.metrics.total_distance_km, ",.1f"),
        ("duration h", lambda p: p.metrics.total_duration_hr, ",.1f"),
        ("vehicle-days", lambda p: float(p.metrics.vehicles_used), ",.0f"),
        ("window violations", lambda p: float(p.metrics.tw_violations), ",.0f"),
        ("lateness h", lambda p: p.metrics.tw_lateness_hr, ",.1f"),
    )


def _component(field: str) -> _Cell:
    """Read one named component off a column's cost breakdown.

    A closure over the field name rather than four near-identical lambdas: the four component rows
    differ only in which attribute they pull, and spelling that difference once is what keeps the
    row table readable as a table.
    """
    return lambda p: float(getattr(p.metrics.breakdown, field))


def table_lines(benchmark: Priced, arms: Sequence[Arm], cost: CostConfig) -> list[str]:
    """Render the benchmark and every arm side by side.

    The ``vs greedy`` row sits directly under the headline because it is the number a reader will
    quote, and putting it anywhere else invites quoting the absolute figure instead — which the
    truncation caveat says is a lower bound.
    """
    columns = [benchmark] + [arm.priced for arm in arms]
    label, headline, spec = _HEADLINE
    lines = [
        _row("", ["greedy"] + [arm.strategy for arm in arms]),
        _row("", ["benchmark"] + [arm.search_label for arm in arms]),
        "-" * (_LABEL_WIDTH + _COLUMN_WIDTH * len(columns)),
        _row(label, [f"{headline(column):{spec}}" for column in columns]),
        _row(
            "vs greedy",
            ["—"] + [delta(headline(benchmark), headline(column)) for column in columns[1:]],
        ),
    ]
    lines.extend(
        _row(row_label, [f"{cell(column):{row_spec}}" for column in columns])
        for row_label, cell, row_spec in _rows(cost)
    )
    return lines


def _row(label: str, cells: Sequence[str]) -> str:
    """One table row: a left-aligned label followed by right-aligned columns."""
    return f"{label:<{_LABEL_WIDTH}}" + "".join(f"{cell:>{_COLUMN_WIDTH}}" for cell in cells)


def generation_lines(arms: Sequence[Arm], ga: GAConfig) -> list[str]:
    """Report, per arm, how much of the generation budget was actually spent.

    One line per arm rather than a single aggregate: the arms are the comparison, and an ablation
    whose arms ran for materially different lengths is reporting something other than the knob it
    thinks it changed.
    """
    lines = [f"generations used, against the {ga.generations} budget:"]
    for arm in arms:
        summary = generations_used(arm.outcomes, ga)
        if summary is None:
            continue
        lines.append(
            f"  {arm.strategy:<10}{arm.search_label:<11}"
            f"median {summary.median:>3}, range {summary.lowest:>3}-{summary.highest:<3}, "
            f"{summary.stopped_early} of {summary.hubs} stopped early, {arm.elapsed_s:,.0f} s"
        )
    return lines


def truncation_lines(arms: Sequence[Arm], ga: GAConfig) -> list[str]:
    """State what the early stopping does and does not invalidate, in the run's own output.

    Stagnation truncation biases the absolute level and not the differences, because every arm
    stops by the same mechanism. **Budget truncation need not be symmetric**, and on seed 42 it was
    not: hub 9 exhausted the 600-generation ceiling in both no-local-search arms and in neither
    local-search arm. An arm with hubs cut off at the ceiling was stopped rather than converged, so
    a difference measured against it overstates the other arm's benefit. Asserting symmetry without
    checking for it is how a caveat becomes false, so the symmetric claim is made only when the
    per-arm counts agree.

    Silent when no arm stopped early, so a run that did spend its budget does not carry a caveat
    that no longer applies to it.
    """
    counts: list[tuple[str, int, int]] = []
    stagnated = False
    for arm in arms:
        summary = generations_used(arm.outcomes, ga)
        if summary is None:
            continue
        stagnated = stagnated or bool(summary.stopped_early)
        counts.append(
            (
                f"{arm.strategy} {arm.search_label}",
                summary.hubs - summary.stopped_early,
                summary.hubs,
            )
        )
    if not stagnated:
        return []
    lines = [
        f"Arms stop on stagnation_limit={ga.stagnation_limit} rather than exhausting the budget,",
        "so the absolute cost per drop above is a lower bound on what this configuration reaches.",
    ]
    if len({exhausted for _, exhausted, _ in counts}) == 1:
        return [
            *lines,
            "Truncation applies to all arms identically: it biases the level, not the differences",
            "this ablation reports.",
        ]
    return [*lines, *_asymmetry_lines(counts, ga)]


def _asymmetry_lines(counts: Sequence[tuple[str, int, int]], ga: GAConfig) -> list[str]:
    """Report unequal budget truncation, and which way it biases the comparison.

    Named per arm rather than summarised, because *which* arm was cut off is what decides the
    direction: a difference measured against a truncated arm is an upper bound on the untruncated
    arm's benefit, not an estimate of it.
    """
    tally = ", ".join(f"{label} {exhausted} of {hubs}" for label, exhausted, hubs in counts)
    return [
        f"Arms were NOT truncated identically. Hubs that exhausted the {ga.generations}-generation",
        f"budget rather than converging: {tally}.",
        "An arm with more exhausted hubs was cut off rather than converged, so a difference",
        "measured against it is an upper bound on the other arm's benefit, not an estimate of it.",
    ]


def verdict_lines(arms: Sequence[Arm]) -> list[str]:
    """State what the four arms show, read off the numbers rather than assumed.

    Written to be capable of reporting that local search lost, or changed nothing. The sign
    agreement between the two strategies is called out because a knob that helps under one
    assignment and hurts under the other has not been shown to do anything.
    """
    by_arm = {(arm.strategy, arm.local_search): arm for arm in arms}
    lines: list[str] = []
    effects: list[float] = []
    for strategy in STRATEGIES:
        with_ls, without_ls = by_arm.get((strategy, True)), by_arm.get((strategy, False))
        if with_ls is None or without_ls is None:
            continue
        after, before = with_ls.priced.metrics, without_ls.priced.metrics
        effects.append(after.cost_per_drop_inr - before.cost_per_drop_inr)
        lines.append(
            f"Local search on {strategy}: ₹{before.cost_per_drop_inr:,.2f} without → "
            f"₹{after.cost_per_drop_inr:,.2f} with, "
            f"{delta(before.cost_per_drop_inr, after.cost_per_drop_inr)} per drop."
        )
    if len(effects) == len(STRATEGIES):
        agree = effects[0] * effects[1] > 0.0
        lines.append(
            f"The two assignments {'agree' if agree else 'disagree'} on the sign, so the effect "
            f"{'holds under both' if agree else 'is not separable from the assignment'}."
        )
    return lines + _assignment_verdict(by_arm)


def _assignment_verdict(by_arm: dict[tuple[str, bool], Arm]) -> list[str]:
    """Compare the two assignments end to end, then name what each leg did.

    Every figure comes from this run rather than from step 4's recorded 1.7%, so the contrast is
    between numbers measured under the same conditions. This is the comparison step 4 could not
    make, and reading only the inbound row is what made its verdict incomplete.

    The final-mile direction is spelled out because the total and the inbound figure can land on
    the same percentage by coincidence — they did on seed 42, at +1.7% each — and a reader who sees
    only those two concludes Stage 2 was neutral. It was not: it moved the same way by 1.78%. Step
    6's hypothesis was specifically that a flatter distribution buys a *better final mile*, so this
    is the row that confirms or refutes it, and it has to be printed rather than inferred.
    """
    nearest, balanced = by_arm.get(("nearest", True)), by_arm.get(("balanced", True))
    if nearest is None or balanced is None:
        return []
    total = delta(
        nearest.priced.metrics.cost_per_drop_inr, balanced.priced.metrics.cost_per_drop_inr
    )
    inbound = delta(nearest.priced.stage1_inr, balanced.priced.stage1_inr)
    outbound = delta(nearest.priced.stage2_inr, balanced.priced.stage2_inr)
    direction = "cheaper" if balanced.priced.stage2_inr < nearest.priced.stage2_inr else "dearer"
    return [
        f"Balanced against nearest, both with local search: {total} on total cost per drop.",
        f"  By leg: inbound {inbound}, final mile {outbound}. Balancing made the final mile "
        f"{direction},",
        "  so the total is not the inbound figure passed through, whatever the percentages read.",
    ]


def probe_lines(reference: Arm, samples: Sequence[Arm], arms: Sequence[Arm]) -> list[str]:
    """Report the GA-seed spread on one arm, and whether the measured effects clear it.

    The spread is the honest error bar on a single-seed 2×2. An effect smaller than it has not
    been measured, however clean the table above looks.
    """
    drops = [reference.priced.metrics.cost_per_drop_inr] + [
        arm.priced.metrics.cost_per_drop_inr for arm in samples
    ]
    spread = max(drops) - min(drops)
    lines = [
        f"Noise probe: {reference.strategy} {reference.search_label}, same inbound plan, "
        f"GA seed varied over {len(drops)} runs.",
        f"  {f'GA seed {reference.ga_seed} (reported)':<26}₹{drops[0]:,.2f}",
    ]
    lines.extend(
        f"  {f'GA seed {arm.ga_seed}':<26}₹{arm.priced.metrics.cost_per_drop_inr:,.2f}"
        for arm in samples
    )
    lines.append(f"  {'spread':<26}₹{spread:,.2f}")
    lines.extend(_clearance_lines(arms, spread, reference.strategy))
    return lines


def _clearance_lines(arms: Sequence[Arm], spread: float, source: str) -> list[str]:
    """Say, per strategy, whether local search moved the answer by more than seed choice does.

    ``source`` names the arm the floor was measured on, and is in the line rather than implied by
    the header above it: the probe reseeds one arm, so a clearance verdict on the *other* strategy
    is being judged against a floor borrowed from this one. That is worth reporting and worth
    labelling, and an unlabelled line reads as though each effect had its own floor.
    """
    by_arm = {(arm.strategy, arm.local_search): arm for arm in arms}
    lines: list[str] = []
    for strategy in STRATEGIES:
        with_ls, without_ls = by_arm.get((strategy, True)), by_arm.get((strategy, False))
        if with_ls is None or without_ls is None:
            continue
        effect = abs(
            with_ls.priced.metrics.cost_per_drop_inr - without_ls.priced.metrics.cost_per_drop_inr
        )
        verb = "clears" if effect > spread else "does not clear"
        lines.append(f"  local search on {strategy}: ₹{effect:,.2f}, {verb} the {source} floor.")
    return lines


def header_lines(problem: Problem, config: Config) -> list[str]:
    """Describe the run before its numbers, so the table cannot be read out of context."""
    return [
        *context_lines(problem.instance, config.run),
        f"{'search':<{_LABEL_WIDTH}}population {config.ga.population_size}, "
        f"budget {config.ga.generations} generations, "
        f"penalty_warmup_generations={config.ga.penalty_warmup_generations}",
        f"{'stage 1':<{_LABEL_WIDTH}}solved once per assignment, shared by both of its arms",
    ]


def _print_report(problem: Problem, benchmark: Priced, arms: Sequence[Arm], config: Config) -> None:
    """Emit the 2×2 and flush it, so a slow noise probe does not withhold the main result."""
    print("\n  Step 7 ablation: memetic local search on/off x nearest/balanced hub assignment.")
    print("  Every column is scored by evaluate_solution over the same instance.\n")
    for block in (
        header_lines(problem, config),
        table_lines(benchmark, arms, config.cost),
        generation_lines(arms, config.ga),
        truncation_lines(arms, config.ga),
        verdict_lines(arms),
    ):
        for line in block:
            print(f"  {line}")
        print()
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 2×2, print it, then size the noise floor beneath it. Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    config = _config(args)

    instance = generate_instance(config.geo, config.fleet, config.schedule, config.run.seed)
    problem = Problem(
        instance=instance,
        matrices=build_matrices(instance, config.run),
        traffic=TrafficModel.from_config(config.traffic),
    )
    benchmark = priced(
        solve_baseline(problem.instance, problem.matrices, problem.traffic), instance, config.cost
    )

    legs = tuple(inbound_leg(name, problem, config) for name in STRATEGIES)
    arms = solve_arms(legs, problem, config)
    _print_report(problem, benchmark, arms, config)

    if args.noise_probe > 0:
        probe_leg = next(leg for leg in legs if leg.strategy == args.probe_arm)
        reference = next(arm for arm in arms if arm.strategy == args.probe_arm and arm.local_search)
        samples = probe_arms(probe_leg, problem, config, args.noise_probe)
        for line in probe_lines(reference, samples, arms):
            print(f"  {line}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
