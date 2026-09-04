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
make eval      # multi-seed evaluation → results CSV
make test      # ruff + mypy + pytest — the gate
make cov       # line coverage against the ≥ 90% standard (see §3 for the measured set)
```

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
- [ ] 7 — ablation: with vs without local search
- [ ] 8 — OR-Tools reference solve
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

   **What that measurement could not see.** It is a statement about the *inbound* leg alone, and
   it remains true as written. Composing the assignment through each shipment — a customer is
   served from the hub its parcel reached — balancing also cuts Stage 2's largest hub from **236
   stops to 73**, median 36 → 58. A flatter distribution is not merely faster to price: for a
   fixed generation budget it is a smaller search space per stop, so equal GA effort buys more.
   Whether that pays back the 1.7% is **step 7's question**, deliberately unresolved here;
   `make run --strategy` is plumbed through both stages so the ablation can measure it end to end
   rather than rediscover it. Read the *total* cost per drop, not the two legs separately —
   reading them separately is what hid this.
5. **The GA stops on `stagnation_limit`, and whether that is convergence or truncation is
   per hub.** On seed 42 every hub ended on `stagnation_limit=75` rather than on
   `generations=600`: 76 to 463 generations, median ~95. The header reports generations used
   against the budget, so a run describes itself honestly.

   **Two hubs traced at full settings with `--trace-generations`, and they disagree.**

   * **Hub 9** (236 stops, stopped at 95) had **converged**. 13 incumbent improvements, all before
     generation 27, then 68 consecutive dry generations. `stagnation_limit` stopped a search that
     had nothing left to find.
   * **Hub 0** (85 stops, stopped at 463) was **still improving**. 44 improvements, the last at
     generation 388, at a declining but nonzero rate — 19 in the first 57 generations, still 5 in
     generations 342–399 — with a longest dry spell of 52, repeatedly approaching the threshold and
     recovering. It stopped with 137 generations of budget unspent, so what truncated it was
     `stagnation_limit`, not `generations`.

   So raising the budget buys nothing on a hub like 9 and might buy something on a hub like 0, and
   the difference is not predictable from size alone — the *larger* hub is the converged one.

   **Both proposed causes of early stopping are dead, on both hubs.** The theory that
   `_configured_best` loses improvements by re-pricing only the search champion: hub 9 flagged 0 of
   95 generations, hub 0 flagged 6 of 463 — and all six fell between generations 11 and 49, none in
   the final 75, which is the window that decides the stop. Real, rare, and immaterial. The theory
   that the population collapses: **zero duplicate children on either hub**, with 104–115 of 147
   still fresh in the final ten generations. Re-pricing top *k* would fix nothing, and the guard is
   not the problem.

   **For step 7, the caveat is per hub, not global.** "Local search buys X%" is a clean claim on
   hubs that converge, and a **lower bound** on hubs still improving when they stop. Trace both
   arms rather than assuming either hub generalises, and report which hubs were in which state.

6. OR-Tools optimises a **static** arc cost using the dispatch-hour traffic multiplier, because a
   `RoutingModel` fixes arc costs before searching. The cumulative band-blended model still
   produces every reported figure, via `src/tour.py`. The proxy affects which tour is chosen, not
   what it is then said to cost.

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
