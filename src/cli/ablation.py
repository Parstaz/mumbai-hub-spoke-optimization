"""Step 7's experiment: run the 2×2 of memetic local search × hub assignment.

This module runs the four arms and prices them. It renders nothing — :mod:`src.cli.run_ablation`
is the entry point that reports what it produced.

**Stage 1 is solved once per strategy, and shared by that strategy's two arms.** Its CVRP runs
guided local search under a wall-clock limit, so it is not reproducible (README limitation 9). Four
independent pipeline runs would hand the two local-search arms of a strategy different inbound
plans *and* a different customer-to-hub mapping, confounding the very effect being measured.
Sharing makes the local-search contrast exact; the nearest-versus-balanced contrast still carries
one draw of Stage 1's noise, which the report says out loud rather than hiding. Dropping Stage 1 to
``cvrp_solution_limit=1`` would remove that draw at the price of measuring a deliberately weaker
inbound plan than the one the pipeline ships, which is a worse trade.

**The noise probe** re-runs one arm at other GA seeds against that same inbound plan. A 2×2 on one
seed measures each effect against a single draw and cannot say whether it is real; the probe's
spread is the error bar the effects have to clear.

Nothing here tunes anything. ``penalty_warmup_generations`` stays at its default of 0 in all four
arms — whether it generalises across hubs and seeds is step 9's question. If local search does not
pay, that is the finding, and it ships the way capacity-balanced assignment not paying shipped.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from src.cli.run_stage1 import STRATEGIES
from src.config import Config, CostConfig
from src.costs.matrix import CostMatrices
from src.costs.traffic import TrafficModel
from src.data.instance import Instance
from src.scoring import Metrics, evaluate_solution, stage_cost
from src.solution import Route, Solution
from src.stage1.cvrp import solve_stage1
from src.stage2.ga import HubOutcome
from src.stage2.solve import hub_of_source, solve_stage2
from src.units import NodeArray, Rupees
from src.workload import hub_of_customer


@dataclass(frozen=True, slots=True)
class Problem:
    """The seeded instance and the matrices priced against it — identical for every arm.

    Bundled so the solve functions stay inside §2.2's five-parameter limit, and because these
    three genuinely vary together: all of them change when the seed does, none of them change
    between arms. Building them once is also what makes the columns comparable at all.
    """

    instance: Instance
    matrices: CostMatrices
    traffic: TrafficModel


@dataclass(frozen=True, slots=True)
class Priced:
    """A scored plan, with the two legs it decomposes into.

    ``metrics`` is the verdict and comes from :func:`~src.scoring.evaluate_solution`. The two leg
    figures come from :func:`~src.scoring.stage_cost` and are a *decomposition*: they must never be
    added back together to obtain the total. ``evaluate_solution`` deliberately folds both stages
    concatenated exactly once so that float regrouping cannot move a reported figure, and summing
    the legs here would reintroduce precisely the regrouping it avoids.
    """

    metrics: Metrics
    stage1_inr: Rupees
    stage2_inr: Rupees


@dataclass(frozen=True, slots=True)
class StrategyLeg:
    """One hub assignment's inbound plan, solved once and shared by both of its arms."""

    strategy: str
    inbound: tuple[Route, ...]
    customer_hubs: NodeArray


@dataclass(frozen=True, slots=True)
class Arm:
    """One cell of the 2×2 — or one repeat of it, when the noise probe varies ``ga_seed``."""

    strategy: str
    local_search: bool
    ga_seed: int
    priced: Priced
    outcomes: tuple[HubOutcome, ...]
    elapsed_s: float

    @property
    def search_label(self) -> str:
        """The arm's local-search state, as the report's second header line spells it."""
        return "with l.s." if self.local_search else "no l.s."


def priced(solution: Solution, instance: Instance, cost: CostConfig) -> Priced:
    """Score a plan and decompose it by stage. The single site producing a reportable column."""
    return Priced(
        metrics=evaluate_solution(solution, instance, cost),
        stage1_inr=stage_cost(solution.stage1_routes, instance, cost).breakdown.total_inr,
        stage2_inr=stage_cost(solution.stage2_routes, instance, cost).breakdown.total_inr,
    )


def inbound_leg(strategy: str, problem: Problem, config: Config) -> StrategyLeg:
    """Solve Stage 1 under one hub assignment, and compose the mapping Stage 2 will inherit.

    Called once per strategy rather than once per arm. See this module's docstring: Stage 1's
    guided local search is not reproducible, so re-solving it per arm would make the local-search
    comparison a comparison across two different inbound plans.

    Args:
        strategy: A key of :data:`~src.cli.run_stage1.STRATEGIES`.
        problem: The instance and matrices, fixed across every arm.
        config: The run configuration; only Stage 1's fields are read here.

    Returns:
        The inbound tours and the customer-to-hub mapping they induce.
    """
    inbound = solve_stage1(
        problem.instance, problem.matrices, problem.traffic, STRATEGIES[strategy], config
    )
    return StrategyLeg(
        strategy=strategy,
        inbound=inbound,
        customer_hubs=hub_of_customer(problem.instance, hub_of_source(problem.instance, inbound)),
    )


def outbound_arm(leg: StrategyLeg, problem: Problem, config: Config) -> Arm:
    """Solve Stage 2 over an existing inbound plan and score the two legs together.

    The arm's identity is read off ``config`` rather than passed alongside it, so a caller cannot
    label an arm one way and configure it another.

    Args:
        leg: The shared inbound plan and mapping this arm delivers from.
        problem: The instance and matrices, fixed across every arm.
        config: This arm's configuration — ``local_search_pct`` and ``run.seed`` are what vary.

    Returns:
        The scored arm, tagged with the configuration that produced it.
    """
    started = time.perf_counter()
    outbound = solve_stage2(
        problem.instance, problem.matrices, problem.traffic, leg.customer_hubs, config
    )
    elapsed_s = time.perf_counter() - started
    solution = Solution(stage1_routes=leg.inbound, stage2_routes=outbound.routes)
    return Arm(
        strategy=leg.strategy,
        local_search=bool(config.ga.local_search_pct),
        ga_seed=config.run.seed,
        priced=priced(solution, problem.instance, config.cost),
        outcomes=outbound.outcomes,
        elapsed_s=elapsed_s,
    )


def solve_arms(legs: Sequence[StrategyLeg], problem: Problem, config: Config) -> tuple[Arm, ...]:
    """Run every cell of the 2×2, local search varying inside each strategy.

    Ordered strategy-major so the two arms a reader most wants to compare — same inbound plan,
    local search the only difference — end up adjacent in the report.

    Args:
        legs: One solved inbound plan per assignment strategy.
        problem: The instance and matrices, fixed across every arm.
        config: The shipping configuration, with local search on.

    Returns:
        Two arms per leg: local search on, then off.
    """
    arms: list[Arm] = []
    for leg in legs:
        for arm_config in (config, without_local_search(config)):
            arms.append(outbound_arm(leg, problem, arm_config))
    return tuple(arms)


def without_local_search(config: Config) -> Config:
    """The same configuration with the memetic step switched off.

    ``local_search_pct = 0`` makes :func:`~src.stage2.ga._refine_some` return its population
    untouched, so nothing else about the search changes — which is what makes this an ablation
    rather than a second configuration.
    """
    return replace(config, ga=replace(config.ga, local_search_pct=0.0))


def probe_arms(leg: StrategyLeg, problem: Problem, config: Config, repeats: int) -> tuple[Arm, ...]:
    """Re-run one arm at other GA seeds, holding the inbound plan and everything else fixed.

    Only ``run.seed`` moves, and :func:`~src.stage2.solve.solve_stage2` reads ``run`` for nothing
    else — the instance and matrices arrive as arguments and stay on the reported seed. So this
    reseeds the genetic algorithm without regenerating the problem, and the spread it produces is
    Stage 2 search noise and nothing else.

    Args:
        leg: The same inbound plan the reference arm used.
        problem: The instance and matrices, fixed across every arm.
        config: The reference arm's configuration; its seed is the one varied from.
        repeats: How many further seeds to run.

    Returns:
        One arm per additional seed, in ascending seed order.
    """
    return tuple(
        outbound_arm(leg, problem, replace(config, run=replace(config.run, seed=seed)))
        for seed in range(config.run.seed + 1, config.run.seed + 1 + repeats)
    )
