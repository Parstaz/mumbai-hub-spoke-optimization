"""Render the GA against the OR-Tools reference: the step 8 comparison.

:mod:`src.stage2.ortools_reference` solves; this renders. Nothing here prices anything — both
columns come from :func:`~src.scoring.evaluate_solution`, and the gap column reuses
:func:`~src.cli.format.delta` so the sign convention cannot diverge from the rest of the
repository's output.

**What the gap means, and what it does not.** The two columns differ in exactly one thing: which
solver ordered the final-mile stops. Same instance, same matrices, same inbound plan, same
customer-to-hub mapping, same fleet rule, same per-hub wall clock. What they do *not* share is how
that budget is spent, which is the point.

Three caveats travel with the table rather than being left to the write-up, because the table is
what gets copied out of a terminal:

* the reference optimises a **static** traffic proxy and a static arrival timeline, so its windows
  are judged against a slightly different day than the one it is scored on;
* its lateness coefficient is rounded to an integer milli-rupee per second, 0.64% under the
  configured rate, in the proxy only;
* OR-Tools' guided local search under a wall clock is **not reproducible** (§9), so a second run
  of the same seed gives a slightly different reference column.

**A hub the reference could not solve is printed, not hidden.** If any hub came back without a
plan there is no reference cost per drop at all — :func:`~src.scoring.evaluate_solution` refuses an
incomplete plan, correctly — and :func:`status_lines` reports which hubs and why. Per-hub figures
for the hubs that *did* solve are deliberately not aggregated: the hubs that solved are the easy
ones, so a partial total would be selecting on the outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.cli.format import delta
from src.config import CostConfig
from src.data.instance import Instance
from src.scoring import Metrics, evaluate_solution, stage_cost
from src.solution import Route, Solution
from src.stage2.ortools_reference import HubReference, ReferencePlan, status_label
from src.units import Rupees

_LABEL_WIDTH = 24
_COLUMN_WIDTH = 16


@dataclass(frozen=True, slots=True)
class Comparison:
    """The two scored columns, and the reference result behind one of them.

    Built once by :func:`compare` so each column is scored exactly once however many blocks of the
    report read it. Two renderers each calling :func:`~src.scoring.evaluate_solution` would be two
    chances for the headline and the verdict sentence to disagree about the same run.

    The ``*_stage2_inr`` figures come from :func:`~src.scoring.stage_cost` and are a
    *decomposition*: they must never be added to the inbound leg to recover the total, which
    ``evaluate_solution`` folds in one pass precisely so float regrouping cannot move it.
    """

    ga: Metrics
    reference: Metrics
    ga_stage2_inr: Rupees
    reference_stage2_inr: Rupees


def compare(
    instance: Instance,
    inbound: tuple[Route, ...],
    ga: tuple[Route, ...],
    plan: ReferencePlan,
    cost: CostConfig,
) -> Comparison | None:
    """Score both plans over the shared inbound leg, or ``None`` if there is nothing to score.

    ``None`` rather than a partial column: an incomplete reference plan has no honest cost per
    drop, and a caller that rendered one anyway would be comparing the GA's whole final mile
    against a subset of the reference's.

    Args:
        instance: The instance both plans were built for.
        inbound: Stage 1's tours, shared by both columns so every difference shown is Stage 2's.
        ga: The GA's final-mile tours.
        plan: The reference's result.
        cost: The run's configured rates — the same object the rest of the run is scored at.

    Returns:
        Both scored columns, or ``None`` when the reference has no complete plan.
    """
    if not plan.complete:
        return None
    return Comparison(
        ga=evaluate_solution(Solution(inbound, ga), instance, cost),
        reference=evaluate_solution(Solution(inbound, plan.routes), instance, cost),
        ga_stage2_inr=stage_cost(ga, instance, cost).breakdown.total_inr,
        reference_stage2_inr=stage_cost(plan.routes, instance, cost).breakdown.total_inr,
    )


def comparison_lines(comparison: Comparison, cost: CostConfig) -> list[str]:
    """Render the GA and the reference side by side, with the gap between them.

    The final-mile row sits with the components rather than at the top because the *total* is what
    a reader should quote — the same discipline the step 7 ablation's table follows.
    """
    before, after = comparison.ga, comparison.reference
    rows = (
        ("cost per drop ₹", before.cost_per_drop_inr, after.cost_per_drop_inr, ",.2f"),
        ("total cost ₹", before.total_cost_inr, after.total_cost_inr, ",.0f"),
        (
            "  final mile ₹",
            comparison.ga_stage2_inr,
            comparison.reference_stage2_inr,
            ",.0f",
        ),
        (
            f"  variable ₹{cost.variable_per_km:g}/km",
            before.breakdown.variable_inr,
            after.breakdown.variable_inr,
            ",.0f",
        ),
        (
            f"  driver ₹{cost.driver_per_hour:g}/h",
            before.breakdown.driver_inr,
            after.breakdown.driver_inr,
            ",.0f",
        ),
        (
            f"  fixed ₹{cost.fixed_per_vehicle:g}/veh",
            before.breakdown.fixed_inr,
            after.breakdown.fixed_inr,
            ",.0f",
        ),
        (
            f"  late ₹{cost.tw_penalty_per_hour:g}/h",
            before.breakdown.tw_penalty_inr,
            after.breakdown.tw_penalty_inr,
            ",.0f",
        ),
        ("distance km", before.total_distance_km, after.total_distance_km, ",.1f"),
        ("duration h", before.total_duration_hr, after.total_duration_hr, ",.1f"),
        ("vehicle-days", float(before.vehicles_used), float(after.vehicles_used), ",.0f"),
        ("window violations", float(before.tw_violations), float(after.tw_violations), ",.0f"),
        ("lateness h", before.tw_lateness_hr, after.tw_lateness_hr, ",.1f"),
    )
    header = (
        f"{'':<{_LABEL_WIDTH}}{'GA':>{_COLUMN_WIDTH}}{'OR-Tools':>{_COLUMN_WIDTH}}"
        f"{'gap':>{_COLUMN_WIDTH}}"
    )
    lines = [header, "-" * (_LABEL_WIDTH + 3 * _COLUMN_WIDTH)]
    lines.extend(
        f"{label:<{_LABEL_WIDTH}}{first:>{_COLUMN_WIDTH}{spec}}"
        f"{second:>{_COLUMN_WIDTH}{spec}}{delta(first, second):>{_COLUMN_WIDTH}}"
        for label, first, second, spec in rows
    )
    return lines


def verdict_lines(comparison: Comparison | None) -> list[str]:
    """State the measured gap in a sentence, read off the numbers rather than assumed.

    Written to be capable of reporting that the GA won, that it lost, or that the two tied. The
    gap is the deliverable either way — CLAUDE.md §1.1 forbids tuning the GA to close it — so this
    names the direction explicitly rather than leaving a signed percentage to be interpreted.
    """
    if comparison is None:
        return [
            "No gap can be reported: the reference has no complete plan, so it has no cost per",
            "drop. See the per-hub status above.",
        ]
    before = comparison.ga.cost_per_drop_inr
    after = comparison.reference.cost_per_drop_inr
    if before == after:
        return [f"The GA and the reference tie at ₹{before:,.2f} per drop."]
    leader, margin = ("GA", after - before) if before < after else ("reference", before - after)
    return [
        f"GA ₹{before:,.2f} per drop against the reference's ₹{after:,.2f}: "
        f"the {leader} is ahead by ₹{abs(margin):,.2f},",
        f"{delta(before, after)} on the reference's column. The GA is not tuned against this "
        f"figure; the gap is the result.",
    ]


def budget_lines(plan: ReferencePlan) -> list[str]:
    """Report what budget each solver got and what the reference did with it.

    The totals are printed beside the per-hub spread because a reader's first question about a
    matched-budget claim is whether it was matched in aggregate or per subproblem. It is both, and
    showing one without the other invites the assumption that it was only the weaker of the two.
    """
    if not plan.hubs:
        return []
    budgets = sorted(hub.budget_s for hub in plan.hubs)
    used = sum(hub.elapsed_s for hub in plan.hubs)
    lines = [
        "Budget: each hub got the wall clock its own GA search spent, not the configured",
        "generation budget — every hub stops on stagnation_limit well short of it.",
        f"  {'per-hub budget':<22}median {budgets[len(budgets) // 2]:,.1f} s, "
        f"range {budgets[0]:,.1f}-{budgets[-1]:,.1f} s over {len(budgets)} hubs",
        f"  {'reference total':<22}{used:,.2f} s against the GA's {sum(budgets):,.2f} s",
        "This is matched-budget, not matched-to-convergence: the GA stopped early by choice and",
        "the reference is given what the GA spent, not what it was offered.",
    ]
    return lines + _first_solution_lines(plan)


def _first_solution_lines(plan: ReferencePlan) -> list[str]:
    """Warn when the reference was stopped at its first solution rather than by the clock.

    ``--deterministic`` sets ``solution_limit=1`` so the inbound CVRP is reproducible, and it
    reaches the reference for the same reason. The side effect is that the reference returns its
    first-solution heuristic and spends almost none of the matched budget — so the gap shown is
    *not* a matched-budget measurement, it is the GA against an unoptimised construction
    heuristic. Printing the table without saying so would be the most misleading output this
    module could produce, which is why the warning is emitted rather than left to the reader to
    infer from a suspiciously small elapsed figure.
    """
    capped = [hub for hub in plan.hubs if hub.first_solution_only]
    if not capped:
        return []
    return [
        "",
        f"WARNING: {len(capped)} of {len(plan.hubs)} hubs ran with solution_limit=1, so the",
        "reference stopped at its first-solution heuristic and did not use its budget. This is",
        "what --deterministic buys, and the gap below is NOT the measured step 8 gap: it is the",
        "GA against an unoptimised construction heuristic. Drop --deterministic to measure.",
    ]


def status_lines(plan: ReferencePlan) -> list[str]:
    """Report every hub's solver status, and what an unsolved or truncated one does to the result.

    Printed in all cases, including the all-solved one. ``ROUTING_SUCCESS`` on every hub is itself
    the evidence that the budget was sufficient, and it should be visible rather than inferred
    from the absence of a warning.
    """
    if not plan.hubs:
        return []
    lines = ["Per-hub solver status:"]
    lines.extend(
        f"  hub {hub.hub_id:<3}{hub.stops:>4} stops  {hub.budget_s:>8,.1f} s  "
        f"{hub.vehicles_deployed}/{hub.vehicles_offered} veh  {status_label(hub.status)}"
        for hub in plan.hubs
    )
    return lines + _unsolved_lines(plan.unsolved, len(plan.hubs)) + _truncation_lines(plan)


def _unsolved_lines(unsolved: Sequence[HubReference], total: int) -> list[str]:
    """Say plainly that there is no comparable figure, and why that is a result not a failure."""
    if not unsolved:
        return []
    named = ", ".join(str(hub.hub_id) for hub in unsolved)
    return [
        "",
        f"The reference found no solution on {len(unsolved)} of {total} hubs within the matched",
        f"budget (hub {named}). It therefore has no complete plan and no cost per drop, and",
        "nothing has been substituted for the missing tours: a retry outside the budget or a",
        "greedy fill-in would report a figure for a solve that did not happen. A matched-budget",
        "comparison can end this way — it is a finding about the budget, not a failed run.",
    ]


def _truncation_lines(plan: ReferencePlan) -> list[str]:
    """Name the hubs whose search was still improving when the clock stopped it.

    The mirror of the caveat :func:`~src.cli.run_ablation.truncation_lines` prints for the GA's
    arms, and the same standard: a solver cut off mid-descent makes the gap an upper bound on its
    quality, so a run that was truncated has to say so rather than report the figure flat.
    """
    truncated = [hub for hub in plan.hubs if hub.budget_truncated]
    if not truncated:
        return []
    ceilings = [hub for hub in plan.hubs if hub.at_fleet_ceiling]
    lines = [
        "",
        f"{len(truncated)} of {len(plan.hubs)} hubs were stopped by the budget rather than by",
        "their own search — they spent the whole matched clock, or reported a descent still in",
        "progress. ROUTING_SUCCESS does not contradict that: it means a local optimum was held",
        "when the clock stopped, and guided local search leaves local optima routinely.",
        "So the reference's column is a lower bound on what it would reach given longer, which",
        "makes the gap an upper bound on the GA's advantage, or a lower bound on its deficit.",
    ]
    if ceilings:
        named = ", ".join(str(hub.hub_id) for hub in ceilings)
        lines += [
            f"{len(ceilings)} hubs deployed every vehicle offered (hub {named}), so the fleet cap",
            "may have bound the reference rather than the search.",
        ]
    return lines


def proxy_lines() -> list[str]:
    """State the two modelling proxies and the reproducibility caveat, beneath the table.

    Beneath rather than above: they qualify the figure, and a reader who has not yet seen the
    figure has nothing to attach them to.
    """
    return [
        "Caveats, all three of which bias the reference's column and none of which are hidden:",
        "  - arc costs and the arrival timeline use a static dispatch-hour traffic multiplier,",
        "    because a RoutingModel fixes arc costs before searching. Reported figures are the",
        "    cumulative band-blended ones, via src/tour.py — the proxy chooses the tour only.",
        "  - the window penalty is rounded to an integer milli-rupee per second, 0.64% under the",
        "    configured rate, in the proxy only.",
        "  - guided local search under a wall clock is not reproducible, so a second run of this",
        "    seed gives a slightly different reference column.",
    ]


def report_lines(
    instance: Instance,
    inbound: tuple[Route, ...],
    ga: tuple[Route, ...],
    plan: ReferencePlan,
    cost: CostConfig,
) -> list[str]:
    """Assemble the whole block, blank-line separated. The one entry point a caller needs.

    Args:
        instance: The instance both plans were built for.
        inbound: Stage 1's tours, shared by both columns.
        ga: The GA's final-mile tours.
        plan: The reference's result, complete or not.
        cost: The run's configured rates.

    Returns:
        Every line of the report, in order.
    """
    comparison = compare(instance, inbound, ga, plan, cost)
    blocks = [
        comparison_lines(comparison, cost) if comparison is not None else [],
        verdict_lines(comparison),
        budget_lines(plan),
        status_lines(plan),
        proxy_lines(),
    ]
    lines: list[str] = []
    for block in blocks:
        if block:
            lines.extend(block)
            lines.append("")
    return lines
