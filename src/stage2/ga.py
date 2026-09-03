"""The evolution loop for one hub.

Everything else in ``src/stage2`` is a piece this assembles: :func:`~src.stage2.split.split` turns
an order into the best plan that order admits, :mod:`~src.stage2.operators` proposes new orders,
:mod:`~src.stage2.local_search` refines them, :mod:`~src.stage2.penalty` decides what lateness is
worth while the search runs. This module is the generation loop and the bookkeeping around it, and
holds no routing, cost or traffic logic of its own.

**Two objectives, deliberately.** Selection, local search and the diversity guard all rank on the
*search* objective — arc weights priced at the adaptive multiplier. The incumbent returned at the
end is tracked on the **configured** objective, at multiplier 1.0, because that is what
:func:`~src.scoring.evaluate_solution` will charge and it is the only number that means anything
outside this loop. Keeping them apart is what lets the penalty adapt freely without the answer
depending on where the multiplier happened to be when the run stopped.

That separation also removes a comparability trap. When the multiplier moves, every stored search
objective was computed at the old rate, so the population is re-scored — but the incumbent never
needs re-scoring, because it was never held at a rate that moves.

**Why a diversity guard.** :func:`~src.stage2.operators.order_crossover` of two identical parents
returns that parent, so a converged population reproduces itself for free and the run quietly
becomes a very slow local search on one chromosome. A child already present is mutated until it is
not, which costs a mutation rather than a generation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.config import GAConfig
from src.stage2.local_search import refine
from src.stage2.operators import or_opt_mutation, order_crossover
from src.stage2.penalty import AdaptivePenalty
from src.stage2.population import initial_population
from src.stage2.pricing import TourPricer, ordered_tour
from src.stage2.split import Permutation, SplitContext, split
from src.units import Rupees

_DIVERSITY_RETRIES = 8
"""Mutations attempted to make a duplicate child distinct before it is accepted anyway.

Bounded rather than unbounded: on a hub of three stops there are six orders in total, so a
population that has genuinely enumerated the space would otherwise spin here forever.
"""


@dataclass(frozen=True, slots=True)
class Individual:
    """One chromosome, with what it scored and how the split partitioned it.

    ``tours`` is carried rather than recomputed because two consumers need it — the local search
    refines it, and the penalty measures lateness over it — and re-deriving it would mean a second
    split of a chromosome that has already been split.

    ``objective_inr`` is the *search* objective, priced at whatever multiplier was in force when
    this individual was scored. It is comparable only with others scored under the same penalty.
    """

    permutation: Permutation
    tours: tuple[Permutation, ...]
    objective_inr: Rupees


@dataclass(frozen=True, slots=True)
class Incumbent:
    """The best order found so far, priced at the **configured** rates.

    A separate type from :class:`Individual` because the two carry incomparable numbers, and the
    single most plausible bug in this module is comparing one against the other. An incumbent is
    never re-scored: the rate it was priced at does not move.
    """

    permutation: Permutation
    objective_inr: Rupees


@dataclass(frozen=True, slots=True)
class HubOutcome:
    """What one hub's run produced, and enough context to report on the run itself.

    ``hub_id`` is carried rather than inferred from position because results come back from the
    pool in completion order, not hub order — see :func:`~src.stage2.solve._solve_all`. Without it
    a plan could be reassembled against the wrong hub's workload, which produces a complete,
    capacity-legal plan that delivers to the wrong customers.
    """

    hub_id: int
    permutation: Permutation
    objective_inr: Rupees
    """Priced at the configured rates, so this is comparable across runs and across hubs."""

    generations_run: int
    final_multiplier: float


@dataclass(frozen=True, slots=True)
class Session:
    """The per-hub constants of one run. ``context`` holds the *configured* rates, never scaled."""

    context: SplitContext
    ga: GAConfig
    rng: np.random.Generator


def evolve(context: SplitContext, ga: GAConfig, rng: np.random.Generator) -> HubOutcome:
    """Search one hub's visit orders, and return the best one found.

    Args:
        context: The hub's workload, road network, windows and **configured** cost rates.
        ga: Hyperparameters. ``generations`` is a budget rather than a tuned value — see
            :class:`~src.config.GAConfig`.
        rng: Injected generator, built by the worker from its seed and hub id.

    Returns:
        The best order found, priced at the configured rates.
    """
    session = Session(context=context, ga=ga, rng=rng)
    penalty = AdaptivePenalty.neutral()
    population = tuple(
        _score(order, session, penalty) for order in initial_population(context, ga, rng)
    )
    best = _configured_best(population, session)
    stagnant = 0

    for generation in range(1, ga.generations + 1):
        population = _refine_some(_repopulate(population, session, penalty), session, penalty)
        champion = _configured_best(population, session)
        if champion.objective_inr < best.objective_inr:
            best, stagnant = champion, 0
        else:
            stagnant += 1
        if generation % ga.penalty_adapt_interval == 0:
            penalty = penalty.adapt(_violating_fraction(population, session), ga)
            population = tuple(_score(one.permutation, session, penalty) for one in population)
        if stagnant >= ga.stagnation_limit:
            return _outcome(session, best, generation, penalty)

    return _outcome(session, best, ga.generations, penalty)


def _outcome(
    session: Session, best: Incumbent, generations: int, penalty: AdaptivePenalty
) -> HubOutcome:
    """Package a finished run, tagged with the hub it belongs to."""
    return HubOutcome(
        hub_id=session.context.workload.hub_id,
        permutation=best.permutation,
        objective_inr=best.objective_inr,
        generations_run=generations,
        final_multiplier=penalty.multiplier,
    )


def _rates(session: Session, penalty: AdaptivePenalty) -> SplitContext:
    """The hub's context with lateness priced at the search rate rather than the configured one."""
    return replace(session.context, cost_config=penalty.rates(session.context.cost_config))


def _score(permutation: Permutation, session: Session, penalty: AdaptivePenalty) -> Individual:
    """Split a chromosome at the search rate and record what it cost."""
    plan = split(permutation, _rates(session, penalty))
    return Individual(
        permutation=permutation, tours=plan.tours, objective_inr=plan.search_objective_inr
    )


def _configured_best(population: tuple[Individual, ...], session: Session) -> Incumbent:
    """Re-price the population's champion at the configured rates, as an incumbent.

    Only the champion, not the population: one extra split per generation against the generation's
    own ``population_size``. The champion by search objective is not always the champion by
    configured objective when the multiplier is away from 1.0 — but it is the individual the search
    is actually pursuing, and re-pricing all of them to find out would double the run.
    """
    champion = min(population, key=lambda one: one.objective_inr)
    plan = split(champion.permutation, session.context)
    return Incumbent(permutation=champion.permutation, objective_inr=plan.search_objective_inr)


def _violating_fraction(population: tuple[Individual, ...], session: Session) -> float:
    """Share of the population whose plan misses at least one window.

    Measured every :attr:`~src.config.GAConfig.penalty_adapt_interval` generations rather than
    every one, because it walks every tour of every individual a second time.
    """
    late = sum(1 for one in population if _is_late(one, session))
    return late / len(population)


def _is_late(individual: Individual, session: Session) -> bool:
    """Whether any vehicle in this individual's plan arrives after a window closes."""
    return any(
        TourPricer(
            tour=ordered_tour(
                tour, session.context.workload, session.context.pricing, session.context.routing
            ),
            routing=session.context.routing,
            cost_config=session.context.cost_config,
        ).whole_lateness_s()
        > 0.0
        for tour in individual.tours
    )


def _repopulate(
    population: tuple[Individual, ...], session: Session, penalty: AdaptivePenalty
) -> tuple[Individual, ...]:
    """Carry the elites through unchanged and breed the rest, rejecting clones.

    Elites are carried by chromosome rather than re-scored, so their recorded objective stays the
    one selection compared them on. A penalty change re-scores the whole population including them.
    """
    ranked = sorted(population, key=lambda one: one.objective_inr)
    survivors = list(ranked[: session.ga.elitism_count])
    seen = {one.permutation for one in survivors}

    while len(survivors) < session.ga.population_size:
        child = _breed(population, session)
        for _ in range(_DIVERSITY_RETRIES):
            if child not in seen:
                break
            child = or_opt_mutation(child, session.ga.or_opt_max_segment_stops, session.rng)
        seen.add(child)
        survivors.append(_score(child, session, penalty))
    return tuple(survivors)


def _breed(population: tuple[Individual, ...], session: Session) -> Permutation:
    """Two tournament winners, recombined and perturbed at the configured rates."""
    first = _select(population, session)
    second = _select(population, session)
    child = first.permutation
    if session.rng.random() < session.ga.crossover_rate:
        child = order_crossover(first.permutation, second.permutation, session.rng)
    if session.rng.random() < session.ga.mutation_rate:
        child = or_opt_mutation(child, session.ga.or_opt_max_segment_stops, session.rng)
    return child


def _select(population: tuple[Individual, ...], session: Session) -> Individual:
    """Tournament selection: the cheapest of ``tournament_k`` drawn without replacement.

    Without replacement so a tournament cannot be won by an individual competing against itself,
    which would make the effective selection pressure depend on the population size.
    """
    contenders = session.rng.choice(len(population), size=session.ga.tournament_k, replace=False)
    return min((population[int(index)] for index in contenders), key=lambda one: one.objective_inr)


def _refine_some(
    population: tuple[Individual, ...], session: Session, penalty: AdaptivePenalty
) -> tuple[Individual, ...]:
    """Run the memetic local search on a random share of the population.

    Random rather than the best few. Refining the elites every generation would spend a full
    neighbourhood sweep per generation rediscovering that they are already locally optimal, while a
    random sample keeps reaching individuals that are not. ``local_search_pct = 0`` disables the
    whole mechanism, which is step 7's ablation switch.
    """
    count = round(len(population) * session.ga.local_search_pct)
    if count == 0:
        return population
    chosen = {
        int(index) for index in session.rng.choice(len(population), size=count, replace=False)
    }
    return tuple(
        _refine_one(one, session, penalty) if index in chosen else one
        for index, one in enumerate(population)
    )


def _refine_one(individual: Individual, session: Session, penalty: AdaptivePenalty) -> Individual:
    """Improve each of an individual's tours, then **re-split** the concatenated result.

    The fitness stored is the fresh split's, never the sum over the improved tours. See
    :mod:`src.stage2.local_search` for why that is required and why it cannot lose.
    """
    scaled = _rates(session, penalty)
    improved = refine(individual.tours, scaled, session.ga.local_search_max_passes)
    return _score(improved, session, penalty)
