"""How a hub's first generation is built.

A population of pure random permutations is a poor start at 236 stops: the expected cost of a
random order is enormous, every individual is equally bad, and the first hundred generations go on
recovering structure that a two-line heuristic supplies for free. Seeding a few nearest-neighbour
orders gives selection something to work with immediately, and leaving the rest random is what
keeps the population diverse enough to escape the seeds' shared blind spots.

**Why three seeds and not thirty.** A nearest-neighbour order is deterministic given its starting
stop — :func:`numpy.argmin` breaks ties by lowest index — so distinct starts do give distinct
tours, and the starts are drawn without replacement to keep them distinct. But those tours are
strongly *correlated*: they agree wherever the greedy choice is unambiguous, which on a clustered
instance is most of the route. Each extra seed therefore buys less than a random individual's worth
of diversity while occupying the slot a random individual would have had.
:attr:`~src.config.GAConfig.seeded_individuals` is set to hedge against a single unlucky start, not
to fill the population with variations on one tour.

**The seeds are written here, not imported from the baseline.** ``src/baseline/greedy.py`` builds
nearest-neighbour tours too, and reaching for it would be the natural thing to do. It is forbidden
by CLAUDE.md §1.1 in the other direction — the control must not import the treatment — and doing it
this way round would still be wrong in spirit: the baseline is the column this stage is measured
against, and a shared constructor makes "we improved on the benchmark" partly a statement about
code both arms run. The two are similar by construction and independent by intent.

Seeding does not flatter the comparison. The GA reports what
:func:`~src.scoring.evaluate_solution` says about its final plan, and if it never improved on its
own seed that would show up as a result of zero, not as a hidden advantage.
"""

from __future__ import annotations

import numpy as np

from src.config import GAConfig
from src.stage2.split import Permutation, SplitContext
from src.tour import RoutingContext
from src.workload import HubWorkload


def nearest_neighbour_order(
    workload: HubWorkload, routing: RoutingContext, first: int
) -> Permutation:
    """Visit the nearest unvisited stop each time, starting from ``first``.

    Distances are read hub-to-stop and stop-to-stop off the same matrices every solver uses, and
    the row is masked and minimised in one vectorised pass rather than scanned — the alternative
    is a Python loop over an O(n²) matrix, which §2.3 calls a defect.

    Args:
        workload: The hub and the stops it serves.
        routing: Supplies the distance matrix.
        first: Position into ``workload.nodes`` to start from. Varying it is how several distinct
            seeds come out of one deterministic construction.

    Returns:
        A visit order over every position of ``workload.nodes``.
    """
    nodes = np.asarray(workload.nodes, dtype=np.intp)
    distance_m = routing.matrices.distance_m
    unvisited = np.ones(len(nodes), dtype=bool)
    unvisited[first] = False
    order = [first]
    for _ in range(len(nodes) - 1):
        row = distance_m[nodes[order[-1]], nodes]
        nearest = int(np.argmin(np.where(unvisited, row, np.inf)))
        unvisited[nearest] = False
        order.append(nearest)
    return tuple(order)


def initial_population(
    context: SplitContext, ga: GAConfig, rng: np.random.Generator
) -> tuple[Permutation, ...]:
    """Build generation zero: a few nearest-neighbour seeds, the rest random.

    Seed starting points are drawn without replacement, so the seeds are genuinely different tours
    rather than one tour repeated — a nearest-neighbour construction is deterministic given its
    start, and duplicates would occupy population slots that contribute nothing to selection.

    Args:
        context: The hub's workload and road network.
        ga: Supplies the population size and how many seeds to plant.
        rng: Injected generator.

    Returns:
        Exactly ``ga.population_size`` visit orders.
    """
    n_stops = len(context.workload.nodes)
    n_seeded = min(ga.seeded_individuals, ga.population_size, n_stops)
    starts = rng.choice(n_stops, size=n_seeded, replace=False)
    seeded = [
        nearest_neighbour_order(context.workload, context.routing, int(start)) for start in starts
    ]
    random_orders = [
        tuple(int(position) for position in rng.permutation(n_stops))
        for _ in range(ga.population_size - n_seeded)
    ]
    return tuple(seeded + random_orders)
