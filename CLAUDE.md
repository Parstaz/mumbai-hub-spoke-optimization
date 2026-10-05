# mumbai-hub-spoke-optimization

Two-stage hub-and-spoke pickup and delivery optimization over a synthetic Mumbai / Navi Mumbai
network. This is a portfolio repository: the code is read by humans as an artifact in its own right.
Treat every file as though it were going through review by a senior engineer.

---

## 1. Domain

| Stage | Scope | Solver |
|---|---|---|
| 1 — inbound | Collect from ~300 sources into 16 hubs. Tours: hub → sources → hub. | OR-Tools CVRP + NetworkX min-cost flow |
| 2 — final mile | Deliver from hubs to ~800 customers under capacity and time windows. Tours: hub → customers → hub. | Custom genetic algorithm |

Benchmark: greedy nearest-neighbour per hub — no consolidation optimization, no local search, no
time-window awareness.

### 1.1 Architectural invariants

These are settled. Do not redesign them, and do not propose alternatives mid-task.

- Stage 2 is a **hand-written genetic algorithm**. It is the reason this repository exists.
  **Never replace it with OR-Tools, and never route around it.**
- Pickup-before-delivery precedence is enforced **structurally**, by stage ordering. There is no
  repair operator in this codebase and none is to be introduced.
- `src/stage2/ortools_reference.py` is a quality benchmark behind a `--reference` flag. It is not
  part of the pipeline and must not be invoked from it.
- **Single scoring path.** `evaluate_solution()` in `src/scoring.py` is the only function that
  scores a `Solution`. Baseline and optimized pipeline both call it. A second scoring
  implementation is a defect, not a convenience. `stage_cost()` in the same module is the fold
  underneath it, for pipeline steps that hold one stage's tours; it does no validation and
  `evaluate_solution()` calls it exactly once, over both stages concatenated, so the two cannot
  disagree by a floating-point regrouping.
- **The baseline is the frozen control.** `src/baseline/` and the shared modules it depends on
  (`src/workload.py`, `src/tour.py`) must never import `src/stage1/` or `src/stage2/`. If the
  control drew code from the treatment, tuning the treatment would move the control's column with
  it — silently, and in the direction that flatters the treatment. Shared helpers go to neutral
  ground; only treatment-specific strategies live under a stage. Enforced by a test that walks
  the import closure from `greedy.py`.
- **No stop is ever split across vehicles.** A stop holding more mass than one vehicle can carry
  makes the instance infeasible: `require_servable()` in `src/workload.py` raises
  `InfeasibleInstanceError`, and it is the *one* expression of the rule — the greedy baseline,
  the Stage 1 CVRP, the Stage 2 GA and the OR-Tools reference all call it. Two copies would let
  step 8 compare solvers under different constraint sets. This is also why Stage 1's min-cost
  flow carries one indivisible unit per source and therefore caps a source *count* rather than a
  mass; a kilogram-denominated arc bound would split a source the instant a cap bound.
- **Per-hub solves run under `multiprocessing`.** Anything shared across hubs must be immutable or
  passed by value. A worker receives its own sliced matrices in a frozen dataclass and nothing
  else — no `Config`, no `Instance`, no `Generator`. Use a **spawn** pool: OR-Tools starts
  threads, and forking a threaded process is undefined.
- **Capacity is hard, enforced by construction** — infeasible arcs are never created in the split
  DAG. It never appears as a fitness penalty.
- **Time windows are soft** — penalised in fitness with an adaptive multiplier.
- Traffic is a static time-of-day multiplier applied **cumulatively along a route**. No live traffic
  API. A route crossing a band boundary mid-leg blends multipliers; a single per-route multiplier is
  incorrect.
- Vehicle capacity and shipment size are **configuration variables**, never literals. Mixed fleet
  and mixed shipment sizes are planned extensions.
- **Never tune the GA to beat the OR-Tools reference.** The measured gap is the deliverable.

### 1.2 Cost model (INR)

| Symbol | Component | Unit | Scales with |
|---|---|---|---|
| `alpha` | variable running cost | ₹/km | distance |
| `beta` | driver / labour cost | ₹/hour | duration |
| `gamma` | fixed vehicle cost | ₹/vehicle/day | fleet size deployed |

Defaults — order-of-magnitude, Tata Ace class SCV, 750 kg payload, ~21 kmpl: `variable_per_km=9.0`,
`driver_per_hour=95.0`, `fixed_per_vehicle=1000.0`, `tw_penalty_per_hour=250.0`.

`beta` being time-denominated is what makes the traffic model change the answer rather than decorate
it. Preserve that coupling.

Traffic bands: `08–11 → 1.6` · `11–17 → 1.2` · `17–21 → 1.8` · `21–08 → 1.0`. Synthetic; labelled as
such in the README.

Headline KPI: **cost per drop (₹/delivery)**. Distance and duration report beneath it as drivers.
Secondary: stops per hour, capacity utilisation.

---

## 2. Engineering standards

### 2.1 Language and typing

- Python 3.11+. Use modern syntax: `X | None`, `list[T]`, `match` where it genuinely reads better.
- `mypy --strict` must pass. No `Any` without an adjacent comment justifying it.
- No `# type: ignore` without an error code and a reason:
  `# type: ignore[arg-type]  # OR-Tools stubs are incomplete`.
- Public functions and all dataclass fields are annotated. Inference is not sufficient at API
  boundaries.
- Use `Protocol` for interfaces with more than one implementation (e.g. `DistanceProvider`). Do not
  use ABCs for this.
- Domain scalars get `NewType` where confusion is plausible: `Metres`, `Seconds`, `Rupees`.

### 2.2 Structure and size

- Functions: **≤ 50 lines**, **≤ 5 parameters**, cyclomatic complexity **≤ 10**. Exceeding any of
  these means extracting a helper, not adding a comment.
- Modules: **≤ 400 lines**. Beyond that, split by responsibility.
- One public concept per module. If a module needs "and" to describe it, it is two modules.
- Prefer pure functions. Confine mutation to explicitly named builders.
- No class with a single method and no state — that is a function.

### 2.3 Data and state

- Config and value objects are `@dataclass(frozen=True, slots=True)`.
- No mutable default arguments, ever.
- **No module-level config reads and no global mutable state.** Config is passed in explicitly. This
  is what makes multi-seed runs and parallel hub solves correct.
- Randomness goes through an injected `np.random.Generator` from `np.random.default_rng(seed)`.
  Never `random.*` and never the legacy global `np.random.*`.
- Matrix work is vectorised NumPy. A Python loop over an O(n²) matrix is a defect.

### 2.4 Errors and logging

- Fail loudly and early. Validate inputs at construction; do not defensively coerce downstream.
- Define a small exception hierarchy in `src/exceptions.py`, rooted at `OptimizationError`. Raise
  domain exceptions (`InfeasibleInstanceError`, `MatrixProviderError`), not bare `Exception`.
- Never `except:` or `except Exception:` without re-raising or a comment stating why swallowing is
  correct.
- Standard-library `logging`, module-level `logger = logging.getLogger(__name__)`.
- **`print()` appears only in CLI entry points.** Never in library code.
- Log the OSRM-to-haversine fallback at `WARNING` — a silent fallback that changes every number in
  the results table is unacceptable.

### 2.5 Documentation in code

- Google-style docstrings on every public function, class and module.
- The docstring answers **why**, not what. `split()` explains why route-first/cluster-second beats
  delimiter encoding; it does not narrate its own loop.
- Non-obvious algorithmic choices carry a one-line rationale and, where applicable, a citation
  (`Prins (2004)`).
- **No commented-out code.** Git holds history.
- No `TODO` without an owner and a tracking issue. Otherwise delete it.

### 2.6 Prohibited

- Duplicate scoring, distance, or traffic logic anywhere outside its owning module.
- Magic numbers in module bodies. All constants live in `config.py`.
- Broad `from x import *`.
- Mocking code we own in tests. Mock only the network boundary.
- Speculative abstraction: no plugin systems, registries, or factories for a single implementation.
  Extension points exist only where the roadmap names one (fleet heterogeneity, shipment size).
- Renaming, reformatting, or "tidying" files outside the scope of the current task.

---

## 3. Testing

- `pytest`. Tests mirror the source tree: `tests/test_<module>.py`.
- Every test is deterministic and seeded. **No network access in tests** — OSRM is stubbed at the
  provider boundary.
- Target ≥ 90% line coverage on `src/stage2/`, `src/stage1/`, `src/costs/`, `src/baseline/`, and
  the shared `src/workload.py`, `src/tour.py`, `src/scoring.py`. Entry
  points and plotting are exempt. The baseline is in the list because it produces the comparison
  column: an untested benchmark makes every improvement claim unfalsifiable.
- Property-based tests (`hypothesis`) are mandatory for:
  - `split()` — every returned route respects capacity; the union of routes equals the input
    permutation exactly, with no duplicates or omissions.
  - Crossover and mutation — output is always a valid permutation of the input set.
- `split()` additionally requires a regression test where left-to-right greedy filling is provably
  **not** optimal. This is the canonical implementation error; the suite must catch it.
- Boundary tests are required wherever a limit exists: load exactly at capacity (legal), one unit
  over (rejected), empty and single-element inputs, traffic-band edges.
- Assert on behaviour, not implementation. Do not assert call counts on our own code.

---

## 4. Tooling

```bash
ruff check . --fix     # lint
ruff format .          # format, line length 100
mypy --strict src/     # types
pytest -q              # tests
```

All four pass before any step is considered complete. `pre-commit` runs `ruff` and `mypy` on staged
files.

### Make targets

```bash
make osrm      # one-time OSRM Docker setup (extract, partition, customize, routed)
make osrm-up   # start the routing server (port 5001); osrm-down to stop it
make data      # generate seeded synthetic instance
make providers # landmark distances under both providers + traffic bands on one leg
make baseline  # greedy baseline, print metrics
make stage1    # inbound leg: baseline vs CVRP under each hub assignment
make run       # full Stage 1 + Stage 2 pipeline, one seed
make ablation  # step 7's 2x2: local search on/off x nearest/balanced, on total cost per drop
make eval      # multi-seed evaluation → results CSV
make test      # ruff + mypy + pytest — the gate
make cov       # line coverage against the ≥ 90% standard (see §3 for the measured set)
```

Extra flags go through `ARGS`. `make` claims a bare `--flag` on its own command line as one of its
options and exits before Python sees it, so `make run --strategy balanced` fails with
`unrecognized option`; both files documented that broken form until step 7.

```bash
make run ARGS="--strategy balanced"
make run ARGS="--reference"            # step 8: the GA against OR-Tools, matched budget per hub
make ablation ARGS="--probe-arm balanced"
```

`--reference` is a measurement, so it must not be combined with `--deterministic`: that caps the
reference at `solution_limit=1`, which stops it at its first-solution heuristic and spends almost
none of the matched budget. The run prints a `WARNING` when it happens, because the resulting table
looks like a measurement and is not one.

---

## 5. Version control

- Conventional Commits: `feat(stage2): add adaptive time-window penalty`.
  Scopes: `config`, `data`, `costs`, `stage1`, `stage2`, `solution`, `baseline`, `eval`, `docs`, `ci`.
- One logical change per commit. Formatting-only changes are committed separately.
- Imperative mood, lower case, no trailing period.

---

## 6. Definition of done

A step is complete only when all of the following hold:

1. `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest` all pass.
2. New public functions have Google-style docstrings stating rationale.
3. New constants are in `config.py`.
4. Tests cover the happy path, the boundaries, and at least one failure mode.
5. The step's verification command runs and its output has been inspected.
6. No file outside the step's scope has been modified.
7. The build-status box below is ticked.

---

## 7. Build status

- [x] 1 — config, synthetic data, `Solution` + `evaluate_solution()`
- [x] 2 — cost layer: chunked OSRM matrix, parquet cache, cumulative traffic bands
- [x] 3 — greedy baseline
- [x] 4 — Stage 1: hub assignment (nearest / min-cost-flow) + per-hub CVRP
- [x] 5 — split procedure + property tests
- [x] 6 — Stage 2 GA: OX, or-opt, adaptive penalty, memetic 2-opt
- [x] 7 — ablation: with vs without local search × nearest/balanced, on total cost per drop
- [x] 8 — OR-Tools reference solve
- [ ] 9 — multi-seed evaluation, notebook, README

---

## 8. Known limitations

Stated plainly in the README. Do not soften or omit them.

1. Sequential Stage 1 → Stage 2 is a **greedy decomposition**. The hub assignment optimal for
   inbound is not necessarily optimal for outbound. Co-optimization is out of scope.
2. All data is synthetic.
3. Traffic multipliers are illustrative, not calibrated against observed Mumbai traffic.
4. **Capacity-balanced hub assignment does not pay on the default instance.** It removes the
   imbalance it targets — worst hub 8,850 kg → 2,738 kg at `hub_balance_slack=1.25` — and costs
   1.7% more on the inbound leg (+82 km, no vehicle saved). Tightening the cap makes it worse
   monotonically. The mechanism: the vehicle floor is set by mass, and mass is not what makes an
   inbound tour expensive — geography is, so relocating a source to a less-loaded hub buys a
   longer radial leg for nothing. Reported, not tuned away. `make stage1` prints the column.

   **Step 6 suspected that verdict was incomplete. Step 7 measured it and the suspicion was
   wrong.** Composing the assignment through each shipment — a customer is served from the hub its
   parcel reached — balancing cuts Stage 2's largest hub from **236 stops to 73**, median 36 → 58.
   Step 6 argued a flatter distribution is a smaller search space per stop, so equal GA effort
   would buy more and might repay the 1.7%. **It does not: balancing makes the final mile dearer
   too**, +1.78% (₹145,443 → ₹148,031 with local search), on the leg the argument said it would
   help. Total +1.74%, ₹264.63 → ₹269.25 per drop.

   The result is robust in direction. The sign holds without local search (+0.93%) and at all three
   GA seeds tried, balanced above nearest by ₹4.61 / ₹5.15 / ₹2.34 per drop at matched seeds.
   Balanced was not starved of search: median 210 generations against nearest's 97.

   **The mechanism is real but delivers variance reduction, not a better mean.** Across GA seeds the
   balanced arm is **3.8× steadier** (range ₹0.61 against nearest's ₹2.34). A smaller search space
   per stop buys consistency, not quality. The reason it loses is the same as on the inbound leg:
   balancing buys window compliance (7 violations / 0.7 h against 9 / 1.6 h) and pays **205 km** for
   it, and at ₹9/km plus driver time distance dominates.

   Read the *total* cost per drop, not the two legs separately. On seed 42 the total (+1.74%) and
   the inbound leg (+1.67%) both round to +1.7%, which invites the conclusion that Stage 2 was
   neutral; it moved the same way by 1.78%. `make ablation` prints all four arms.
5. **Hub 9 stops early against a distorted objective. Localised, not solved.** On seed 42 every
   hub ends on `stagnation_limit=75` rather than `generations=600` (76–463, median ~95). On hub 9
   that early stop costs real money, and four schedule variants say so:

   | hub 9, seed 42 | generations | last improvement | configured ₹ | lateness ₹ |
   |---|---|---|---|---|
   | adaptive (shipping default) | 95 | 20 | 29,172.6 | 158.6 |
   | penalty pinned at x1.00 | 313 | 238 | 28,516.5 | 16.0 |
   | `penalty_warmup_generations=50` | 486 | 411 | **28,477.9** | **4.5** |
   | `penalty_warmup_generations=150` | 368 | 293 | 28,521.7 | 4.5 |

   Any of the three interventions recovers ~₹650–695. Warm-up is better than pinning on **both**
   cost and lateness, and still ends at x16 — so the adaptive schedule earns its keep, it just must
   not engage before the population has committed to a basin.

   **The load-bearing observation is hub 0, not hub 9.** Traced per generation, hub 0 has a
   *bit-identical* multiplier trajectory to hub 9 — x1.00 through generation 10, x2.00 at 11, x4 at
   21, x8 at 31, x16 at 41 — and keeps improving to generation 388, where hub 9's improvements are
   dead by generation 20 with x2.00 in force. **Identical schedule, opposite outcome.** That rules
   out the schedule *by construction* as a sufficient explanation: whatever makes hub 9 fragile is a
   property of hub 9 interacting with the schedule, not of the schedule. It also locates the damage
   at the **first adaptation step**, x2.00 at generation 11 — not at a high multiplier, which is
   where an earlier draft of this entry wrongly put it.

   **Ruled out, each by measurement rather than argument.** Penalty/stagnation correlation
   (Pearson +0.21, wrong sign, and the one x1.00 hub ran shortest). Incumbent-tracking blindness
   (0 of 95 flagged generations on hub 9, 6 of 463 on hub 0 and all six outside the deciding
   window). Population collapse (zero duplicate children on either hub). Incumbent re-injection as a
   fix (drift 73/95 → 8/95, answer unchanged to the rupee — the incumbent plateaus at generation 20
   and drift only begins at 23, so eviction is downstream). The cheaper plans are not less feasible
   ones: all arms sit at the 12-vehicle mass floor with zero over-capacity routes, and the recovered
   plans carry *less* lateness, not more.

   **Not investigated, and why stopping here was a decision rather than an omission.** The remaining
   candidates for hub 9's fragility are population size, tournament pressure, or-opt segment length,
   local-search share, the diversity guard's interaction with all of these, and hub geometry itself
   (236 stops against hub 0's 85). Each is another 30–60 minute run and **there is no principled
   ordering between them** — nothing in the evidence favours one over another, which is what makes
   the search unbounded. The mechanism is localised, the fix is implemented behind
   `penalty_warmup_generations` (default 0, the schedule as step 1 wrote it), and it stays off.
   Step 7 did **not** measure it — its four arms held it at 0 so the ablation measured local search
   and hub assignment, not a third knob. **Step 9** tests it across hubs and seeds. Recorded as a
   limitation, not solved.

   One datum step 7 produced on `local-search share`, which the list above calls uninvestigated:
   with local search off, hub 9 runs to the 600-generation ceiling instead of stopping early. That
   does not explain the early stop — one hub, one seed, and the off arm is the dearer one — but the
   memetic step is entangled with hub 9's convergence and is not the neutral candidate implied.

   **Consequences for what is reported.** The −14.4% headline is a **lower bound**: 15 of 16 hubs
   ran at x8 or above. The claim that step 7's ablation was "unaffected because truncation hits both
   arms identically" was **wrong, and step 7 disproved it** — see §8.6. Stagnation truncation is
   symmetric; budget truncation was not.

6. **Memetic local search pays, but −2.84% is an upper bound rather than an estimate.** Switching it
   off costs 2.84% per drop under `nearest` and 2.27% under `balanced` (₹7.73, ₹6.26), same sign
   under both, clearing the GA-seed noise floor at 3.30× and 2.68×. It is also faster in wall clock
   on nearest, 1,422 s against 2,216 s, converging in a median 97 generations against 185.

   **The arms were not truncated identically, and the asymmetry flatters local search.** Hub 9
   exhausted the 600-generation ceiling in *both* no-local-search arms and *neither* local-search arm
   — 15 of 16 hubs stopping early against 16 of 16 — so the control was cut off rather than
   converged on its largest hub: ₹28,640 at 195 generations with local search, ₹30,235 at the 600
   ceiling without. That hub carries **₹1,595 of the ₹6,188 gain on nearest, 26% of it.** Read
   −2.84% as "no worse than 2.84% better".

   `truncation_lines()` in `src/cli/run_ablation.py` counts exhausted hubs per arm and only claims
   symmetry when the counts agree. An earlier version asserted it unconditionally and was false
   about the run printing it. **Do not restore the unconditional claim.**

7. OR-Tools optimises a **static** arc cost using the dispatch-hour traffic multiplier, because a
   `RoutingModel` fixes arc costs before searching. The cumulative band-blended model still
   produces every reported figure, via `src/tour.py`. The proxy affects which tour is chosen, not
   what it is then said to cost.

   This applies to **both** OR-Tools models — Stage 1's CVRP and Stage 2's reference — since they
   share `src/arc_model.py`. For the reference it extends to the arrival timeline the time
   dimension carries, so its windows are judged on a static day and scored on a cumulative one.
   See §8.10 for the direction that biases the result in.

10. **The OR-Tools reference beats the hand-written GA by 2.9% per drop on seed 42, and the margin
    is a lower bound.** Step 8's deliverable. Matched per-hub wall clock, same instance, same
    matrices, same inbound plan, same customer-to-hub mapping, same `require_servable()` rule, both
    columns scored by `evaluate_solution()`:

    | seed 42, nearest, local search on | greedy | GA | OR-Tools |
    |---|---|---|---|
    | cost per drop ₹ | 309.01 | 264.63 | **257.02** |
    | vs greedy | — | −14.4% | **−16.8%** |
    | final mile ₹ | — | 145,443 | **139,354** (−4.2%) |
    | distance km | 9,988.4 | 8,889.8 | **8,346.1** (−6.1%) |
    | duration h | 396.7 | 371.4 | 360.3 (−3.0%) |
    | vehicle-days | 96 | 96 | 96 (±0) |
    | window violations | 68 | 9 | 3 |
    | lateness h | 94.5 | 1.6 | 1.1 |

    **Where the ₹6,089 comes from**, read off the run's own component rows rather than derived:

    | component | GA | OR-Tools | delta | share of gap |
    |---|---|---|---|---|
    | variable ₹9/km | 80,008 | 75,115 | **−4,893** | 80.4% |
    | driver ₹95/h | 35,286 | 34,230 | −1,056 | 17.3% |
    | fixed ₹1,000/veh | 96,000 | 96,000 | 0 | 0.0% |
    | late ₹250/h | 410 | 270 | −140 | 2.3% |

    The four sum to the ₹6,089 final-mile delta exactly. **No part of the gap is fleet sizing** —
    vehicle-days are identical at 96 because both solvers sit at the per-hub mass floor — so this is
    routing quality throughout. It is mostly distance (544 km, four fifths of the gap) with the
    driver time that distance drags along behind it (a sixth), and the window penalty is a rounding
    error on the total at 2.3%. Worth noting anyway: the reference more than halved violations, 9 to
    3, while *under-pricing* lateness in its own objective — the next point.

    **Three things bias this measurement, all of them against the reference.** That is why −2.9% is
    a floor on OR-Tools' advantage rather than an estimate of it, and it is part of the result
    rather than a footnote to it:

    - **Static traffic** (§8.7). The reference chooses tours under a dispatch-hour-constant arc
      cost and a static arrival timeline, then gets scored under the cumulative band-blended model.
      It is optimising a slightly wrong objective and still wins.
    - **Rounded lateness coefficient.** `SetCumulVarSoftUpperBound` takes an integer, so ₹250/hour
      becomes 69 milli-INR/s where the exact figure is 69.44 — the proxy under-prices lateness by
      0.64%. It cut violations from 9 to 3 anyway.
    - **13 of 16 hubs were stopped by the clock**, having spent their matched budget to the tenth of
      a second. Only hubs 0, 9 and 10 finished inside it. So the reference's column is itself a
      lower bound on what it reaches given longer.

    **What the GA's 2.9% deficit does and does not mean.** It does not mean the hand-written solver
    is redundant: that is the wrong question, and §1.1 forbids tuning it to close the gap. What step
    8 establishes is that a from-scratch GA — route-first/cluster-second encoding with an exact
    split, no repair operator anywhere, capacity structural rather than penalised, an adaptive
    window penalty, a memetic 2-opt — lands within 2.9% of a mature constraint solver on the same
    budget and the same problem, and beats the greedy control by 14.4% doing it. The architecture is
    the deliverable and it is now demonstrated against something, rather than asserted. The 2.9% is
    the price of that demonstration, stated.

    **Budget, and what "matched" bought.** Per-hub median 237.3 s, range 10.1–1,388.6 s, set by each
    hub's own GA search; the reference consumed 5,354 s of the 5,875 s the GA spent. Matched
    *per hub* rather than in aggregate, because the hubs are independent contests and the 1,410 s
    headline is a makespan set by hub 0 — one aggregate figure would have handed an 8-stop hub two
    orders of magnitude more search than the GA gave it. This is matched-budget, **not**
    matched-to-convergence: the GA stopped itself on `stagnation_limit` (median 97 generations of
    600) and the reference was given what the GA *spent*, not what it was *offered*.

    All 16 hubs returned `ROUTING_SUCCESS`, none hit its fleet ceiling, and the §8.11 no-plan branch
    never fired. Single instance, single GA seed, and the reference is not reproducible (§9) — so
    this is one point estimate, which is step 9's problem.

11. **The reference's no-plan branch is right, but the evidence that motivated it was not.** A hub
    whose matched budget buys no plan is reported — `orders=None`, the solver status printed, no
    cost per drop for the whole run, and nothing substituted for the missing tours. That behaviour
    is correct and stays: a matched budget genuinely can be too short, and the alternatives (a retry
    outside the budget, a greedy fill-in, silently dropping the hub) all put a figure in the table
    for a solve that did not happen, with the dropped hubs being the hard ones.

    **But it was built partly on a misreading.** During implementation a one-stop hub came back with
    no plan on a 3.5 ms budget, and that was recorded as a budget-scarcity finding. It was not: the
    time dimension's horizon bounded the *rounded sum* of arc transits rather than the *sum of
    rounded* transits, and individually-rounded arcs exceeded it by one second — so a hub any single
    vehicle could serve was proved `ROUTING_INFEASIBLE` by its own horizon. With the horizon fixed
    to `(k+1)` arcs at `ceil(longest leg) + ceil(service)`, that hub solves in 3.6 ms, and on seed 42
    the branch never fires at all.

    Recorded because the two are easy to conflate and the conflation is self-serving: a modelling
    bug that presents as "the budget was too short" is a bug that gets written up as a finding.
    `ROUTING_FAIL` is the same trap — it means "no solution found", not "infeasible", and at
    millisecond budgets the same task returns `ROUTING_FAIL` or `ROUTING_FAIL_TIMEOUT`
    nondeterministically. Only `ROUTING_INFEASIBLE` and `ROUTING_INVALID` raise.

    A third instance of the same family: `ROUTING_SUCCESS` on every hub while 13 of 16 had spent
    their whole budget. It means "holds a local optimum", not "finished", so keying the truncation
    caveat on the status alone left it silent on the run that needed it. `HubReference.clock_stopped`
    measures it from the clock instead. Same rule as §8.6: a truncation claim comes from what the
    run measured, never from what a status name suggests.

8. **Stage 1's guided local search is not bit-reproducible, but does not propagate.** It returns
   whatever it reached when the clock ran out; `--deterministic` stops at the first-solution
   heuristic and the test suite uses it. Step 7 measured the consequence: three independent full
   runs on seed 42 agree to **₹0.01 per drop**, two of them bit-identical on every arm and every
   generation figure. Stage 1's search perturbs tour *ordering* within a hub (₹1–5 of ₹66,000) but
   does not change the source-to-hub assignment, so the customer-to-hub mapping is identical and an
   identical mapping with an identical per-hub GA seed gives an identical Stage 2.

   For step 9 this splits. Against Stage 1 noise, one run per seed is a sound point estimate.
   Against the **GA** seed it is not: `Stage2Task.seed` is `RunConfig.seed`, so a multi-seed run
   varies instance and GA draw together and cannot separate them. The GA draw alone spans ₹2.34 per
   drop on seed 42's nearest arm, against effects of ₹4.6–7.7. Report a per-seed spread, not a mean.

9. **Step 7's result is single-instance.** Three GA seeds on seed 42's geography. Everything in §8.4
   and §8.6 is a statement about that instance.

---

## 9. Gotchas

Append a line when the same mistake occurs twice. Do not add entries speculatively.

- OSRM `/table` enforces `max_table_size` (100 on the public demo server). Matrices here are
  ~1100×1100; requests must be chunked via `sources=` / `destinations=` and reassembled. The
  limit is a **cell budget of `max_table_size²`**, not a coordinate count, so square blocks of
  side `max_table_size` are legal. Send only each block's own coordinates: the full node list in
  the URL is a 22 kB request line the server rejects.
- OSRM's host port is **5001**, not 5000 — macOS binds 5000 to the AirPlay Receiver, so 5000
  fails on a fresh clone on every Mac. `RunConfig.osrm_url` and `docker-compose.yml` must agree;
  a mismatch falls back to haversine and reads as an outage rather than a misconfiguration.
- `urlparse` splits a trailing `;`-separated group off the last path segment as RFC 2396
  "params", silently discarding every OSRM coordinate after the first. Use `urlsplit`.
- The assembled matrix must be cached to parquet, keyed by `(seed, provider, n_nodes, coord_hash)`.
  Without it, re-querying during fitness evaluation dominates runtime.
- A shared `Generator` or mutable config across per-hub workers silently destroys
  reproducibility — the run still succeeds, and its numbers stop being repeatable. §1.1 has the
  rule; this is the symptom to recognise.
- `ortools` is built with SWIG, whose `SwigPyPacked`, `SwigPyObject` and `swigvarlink` types carry
  no `__module__` attribute, which Python 3.14 deprecates. Under pytest's
  `filterwarnings = ["error"]` that is raised inside `_pywrapcp`'s module init where it cannot
  propagate, and the interpreter **segfaults during collection** — it does not fail a test.
  `pyproject.toml` ignores that exact message and nothing broader.
- OR-Tools guided local search under a wall-clock time limit is not reproducible: it returns
  whatever it reached when the clock ran out, so a busier machine yields a different plan. Tests
  must set `cvrp_solution_limit=1`, which stops at the first-solution heuristic and is
  independent of the time limit.
- A tunable's **default and its justification get written in different files and drift**.
  `penalty_min_multiplier` and `seeded_individuals` both landed in `config.py` with a value the
  consuming module's docstring did not support — and in one case the docstring gave no reasoning
  at all, so there was nothing for the default to contradict. When adding a field to a config
  dataclass, write the number's argument in the module that *reads* it, and check the two agree
  before committing. A default nobody can justify is a magic number with a longer name.
