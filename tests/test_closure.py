"""The pipeline must not be able to reach the OR-Tools reference.

CLAUDE.md §1.1: ``src/stage2/ortools_reference.py`` is a quality benchmark behind a ``--reference``
flag, not part of the pipeline. This is the test that makes that enforceable rather than merely
stated.

There is no behavioural symptom to catch instead. A pipeline that called the reference would
produce a complete, correctly-scored, entirely plausible plan — it would just no longer be
measuring a hand-written GA against an independent solver, because the two would share a code
path. By the time anyone noticed, every number in the results table would be suspect.

The arrow is deliberately one-way: the reference imports
:func:`~src.stage2.solve.hub_workloads` *from* the pipeline, which is required — both solvers must
group customers by the same function or they are solving different problems. What is forbidden is
the reverse.
"""

from __future__ import annotations

from tests.closure import SRC_ROOT, reachable_modules

REFERENCE_MODULE = "src.stage2.ortools_reference"

PIPELINE_ROOTS = (
    SRC_ROOT / "stage2" / "solve.py",
    SRC_ROOT / "stage1" / "cvrp.py",
    SRC_ROOT / "stage2" / "ga.py",
    SRC_ROOT / "baseline" / "greedy.py",
)
"""Every entry into the solve path, plus the baseline. None may reach the reference."""


def test_the_pipeline_cannot_reach_the_ortools_reference() -> None:
    """Walked transitively: the import that would break this would not be added to solve.py."""
    reached = reachable_modules(PIPELINE_ROOTS)

    assert REFERENCE_MODULE not in reached, (
        f"the pipeline reaches {REFERENCE_MODULE}; it is a quality benchmark behind --reference, "
        f"and a pipeline that depends on it is no longer measured by it"
    )
    # Guard against the walk silently finding nothing and passing vacuously.
    assert "src.stage2.split" in reached
    assert "src.arc_model" in reached


def test_the_reference_does_reach_the_pipelines_grouping() -> None:
    """The permitted direction, asserted so the one-way arrow is documented by a test.

    If this ever fails, the reference has grown its own customer-to-hub grouping and the two
    solvers may no longer be partitioning the hubs identically — which would show up as solver
    quality in the reported gap.
    """
    reached = reachable_modules((SRC_ROOT / "stage2" / "ortools_reference.py",))

    assert "src.stage2.solve" in reached
    assert "src.arc_model" in reached


def test_the_reference_is_not_imported_by_any_cli_except_the_pipeline_entry_point() -> None:
    """Only ``run_pipeline`` may offer ``--reference``; no other entry point may pull it in.

    ``make ablation`` and ``make baseline`` report numbers the README quotes. A reference solve
    appearing in either would change what those commands cost and how long they take, without
    changing what they claim to measure.
    """
    cli = SRC_ROOT / "cli"
    offenders = sorted(
        path.name
        for path in cli.glob("*.py")
        if path.name not in {"run_pipeline.py", "reference.py"}
        and REFERENCE_MODULE in reachable_modules((path,))
    )

    assert not offenders, f"these entry points reach the reference solve: {offenders}"
