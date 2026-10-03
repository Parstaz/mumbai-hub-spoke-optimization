"""Reporting helpers shared by the entry points, on neutral ground.

These live here rather than in whichever entry point happened to need them first, for two
reasons that both turned out to be structural.

The sign convention is the one thing about :func:`delta` a reader has to be able to trust without
checking, and ``generations_used`` decides what "used" means for every table that reports a
budget — so there must be exactly one of each.

And they cannot live in :mod:`src.cli.run_pipeline`. That module now imports
:mod:`src.cli.reference` to render ``--reference``, so anything importing it transitively reaches
the OR-Tools reference solve; ``make ablation`` did, until these moved. ``tests/test_closure.py``
is what caught it, and keeping the shared helpers here is what keeps that test passing honestly
rather than by being relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import GAConfig
from src.stage2.ga import HubOutcome


def delta(before: float, after: float) -> str:
    """Percentage change, signed so an improvement reads negative.

    Shared by the pipeline table, the step 7 ablation and the step 8 reference comparison. A
    second formatter would be free to disagree about the sign convention, which is exactly the
    kind of difference a reader would not think to check for.

    Args:
        before: The reference figure.
        after: The figure being compared against it.

    Returns:
        A signed percentage, or an em dash when ``before`` is zero and the change is undefined.
    """
    if before == 0.0:
        return "—"
    return f"{(after - before) / before:+.1%}"


@dataclass(frozen=True, slots=True)
class GenerationsUsed:
    """How much of the generation budget a run's hubs actually spent.

    Separated from its rendering because two entry points need the same figures in different
    shapes: this one prints a two-line block for a single run, the step 7 ablation prints one line
    per arm. Computing it twice would be two chances to disagree about what "used" means.
    """

    median: int
    lowest: int
    highest: int
    stopped_early: int
    hubs: int


def generations_used(hubs: tuple[HubOutcome, ...], ga: GAConfig) -> GenerationsUsed | None:
    """Summarise the generations each hub ran, or ``None`` when no hub ran at all.

    ``None`` rather than a zeroed summary: an instance with nothing to deliver has no budget
    story to tell, and a caller that rendered "median 0 of 600" would be describing a search that
    never started.

    Args:
        hubs: Every hub's outcome, in any order.
        ga: The configuration the run was launched with, for the budget and the stagnation limit.

    Returns:
        The median, range and early-stop count across hubs, or ``None`` if ``hubs`` is empty.
    """
    if not hubs:
        return None
    used = sorted(outcome.generations_run for outcome in hubs)
    return GenerationsUsed(
        median=used[len(used) // 2],
        lowest=used[0],
        highest=used[-1],
        stopped_early=sum(1 for value in used if value < ga.generations),
        hubs=len(used),
    )
