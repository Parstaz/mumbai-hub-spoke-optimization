# mumbai-hub-spoke-optimization

Two-stage hub-and-spoke pickup and delivery optimization over a synthetic Mumbai / Navi Mumbai
network. Stage 1 consolidates ~300 sources into 16 hubs (OR-Tools CVRP + min-cost flow); Stage 2
runs a hand-written genetic algorithm to deliver from those hubs to ~800 customers under capacity
and time windows. The headline KPI is **cost per drop (₹/delivery)**.

> **Build status.** Both stages are built (step 6 of 9). `make run` solves one seed end to end and
> prints it against the greedy benchmark; on seed 42 under OSRM that is **₹309.01 → ₹264.63 per
> drop, −14.4%**, at an unchanged 96 vehicle-days. The ablation (step 7), the OR-Tools reference
> (step 8) and the multi-seed evaluation with its figures and full write-up (step 9) are still to
> come, so treat that as one seed rather than a result — a single run has no error bar, and the
> figure it is compared against comes from a solver whose inbound leg is not bit-reproducible.
> This README currently documents the cost layer, the traffic model and the baseline in depth;
> the two solvers are documented in their own modules until step 9.

---

## Running it without any setup

The cost layer has a great-circle fallback with no external dependencies, so the pipeline runs on
a clean checkout with no Docker and no OSM data:

```bash
make data                                   # seeded synthetic instance + scatter plot
.venv/bin/python -m src.cli.compare_providers --no-osrm
.venv/bin/python -m src.cli.run_baseline --no-osrm    # greedy benchmark + metrics table
```

Distances are then haversine × `RunConfig.circuity_factor` (1.30, measured — see below), with
durations at `haversine_speed_kmph` (24 km/h). Those are estimates, not road-network figures, and
every run that uses them says so at `WARNING`.

---

## One-time OSRM setup

Real road distances need a self-hosted [OSRM](https://project-osrm.org/) instance. The whole
sequence is `make osrm`, which takes 15–30 minutes and around 4 GB of RAM. What it does, and how
to do it by hand:

### 1. Download the extract

```bash
mkdir -p data/osrm
curl -fL -o data/osrm/maharashtra-latest.osm.pbf \
  https://download.openstreetmap.fr/extracts/asia/india/maharashtra-latest.osm.pbf
head -c 32 data/osrm/maharashtra-latest.osm.pbf | grep -aq OSMHeader || echo "not a PBF"
```

Maharashtra (~165 MB) rather than all of India (~1.2 GB): it covers the whole bounding box in
`GeoConfig` with room to spare, and preprocesses in minutes instead of hours.

The mirror is openstreetmap.fr because Geofabrik publishes India only as six multi-state zones,
with no per-state Maharashtra extract. Check the magic bytes as above — a mirror that answers an
unknown path with a redirect to its index page hands you a 9 kB HTML file, and `osrm-extract`
reports that twenty minutes later as `invalid BlobHeader size` rather than as a failed download.
`make osrm` does this check for you and downloads to a temporary name so a bad fetch cannot be
mistaken for a good one on the next run.

### 2. Preprocess — extract, partition, customize

```bash
docker compose --profile build run --rm osrm-extract
docker compose --profile build run --rm osrm-partition
docker compose --profile build run --rm osrm-customize
```

These write the `.osrm.*` graph files next to the `.pbf`, and only need re-running when the
extract changes.

**Use MLD, not CH.** `osrm-partition` + `osrm-customize` is the multi-level Dijkstra pipeline. The
contraction-hierarchies alternative (`osrm-contract`) answers `/table` with durations but **no
distances**, and this codebase needs both — `src/costs/matrix.py` rejects a response missing the
`distances` annotation and names this as the likely cause.

### 3. Run the server

```bash
make osrm-up      # or: docker compose up -d osrm
curl "http://127.0.0.1:5001/table/v1/driving/72.8347,18.9220;72.8355,18.9398?annotations=distance,duration"
```

The container runs `osrm-routed --algorithm mld --max-table-size 100`.

**The published port is 5001, not OSRM's usual 5000.** macOS binds 5000 to the AirPlay Receiver
by default, so 5000 fails on a fresh clone on every Mac. `RunConfig.osrm_url` defaults to the
same 5001. To use 5000 instead — after freeing it in *System Settings › General › AirDrop &
Handoff* — set `OSRM_PORT=5000` and change `RunConfig.osrm_url` to match. Change both: a
mismatch makes the pipeline fall back to haversine, which looks like an OSRM outage rather than
a misconfiguration.

### 4. Sanity-check the numbers

```bash
make providers
```

Prints road distances between eight Mumbai landmarks under both providers, side by side with the
great-circle distance, plus the circuity each pair implies. Two things to look at:

- **OSRM / crow** is where `circuity_factor` comes from. Measured against a live MLD build of the
  Maharashtra extract, the eight landmarks give **1.13–1.40, mean 1.28**, which is why the
  default is 1.30. Long trunk routes sit at the bottom of that range (Gateway → Thane, 1.13) and
  short suburban hops at the top (BKC → Powai, 1.40) — circuity always rises as legs shorten.
- The **traffic** section applies the bands to a single 30-minute leg at five departure times.
  A leg leaving at 10:45 must come out at an effective 1.32× — between the 1.6 of the morning
  peak and the 1.2 of midday — because it crosses the 11:00 boundary. Five identical numbers
  there would mean the multipliers are not being applied at all.

---

## Why the matrix is chunked and cached

Two constraints shape `src/costs/matrix.py`, and both are load-bearing rather than incidental:

**OSRM caps a `/table` request** at `max-table-size²` cells — 10,000 on the demo server's default
of 100. The instance here is ~1116 nodes, or 1.25 million cells, so the matrix is walked as a grid
of square blocks and reassembled. Each request carries only its own block's coordinates, because
putting all 1116 in the URL with index selectors produces a 22 kB request line the server will not
accept.

**The assembled matrix is cached to parquet** under
`data/cache/matrix_seed{seed}_{provider}_n{nodes}_{digest}.parquet`. The genetic algorithm reads
the matrix millions of times per run; without the cache, matrix I/O rather than search would
dominate every runtime figure the repository reports. The coordinate digest is in the key because
two runs can share a seed and a node count while placing nodes differently, and the provider is in
the key because a fallback run must never load an entry written by a road-network run.

---

## Traffic

A static time-of-day multiplier, applied **cumulatively along a route**:

| Band | Multiplier |
|---|---|
| 08–11 | 1.6 |
| 11–17 | 1.2 |
| 17–21 | 1.8 |
| 21–08 | 1.0 |

A leg is integrated across every band boundary it crosses, not scaled by the multiplier in force
at its departure. Leaving at 10:45 with 30 minutes of free-flow time, the first quarter-hour is
driven at 1.6× and the remainder at 1.2×, arriving at 11:24:45 rather than 11:33 or 11:21. Because
`driver_per_hour` is time-denominated, this is what makes *when* a route is driven change its cost
rather than decorate it.

---

## Known limitations

Stated plainly, and not softened anywhere else in the repository:

1. **Sequential Stage 1 → Stage 2 is a greedy decomposition.** The hub assignment that is optimal
   for inbound consolidation is not necessarily optimal for outbound delivery. Co-optimization is
   out of scope.
2. **All data is synthetic.** Sources, customers, shipments and time windows are generated from a
   seed; no real order book is involved.
3. **Traffic multipliers are illustrative, not calibrated** against observed Mumbai traffic. They
   are plausible round numbers chosen to make the peak-hour trade-off visible, not measurements.

4. **About a quarter of generated nodes are not on the road network.** The bounding box is a
   rectangle over a city that is not one: it covers the Arabian Sea west of the coast, Thane
   Creek, the harbour and Sanjay Gandhi National Park. Measured against OSRM's `/nearest`, 25%
   of nodes sit more than 500 m from the nearest routable road and 9% more than 1 km, the worst
   4.8 km out. The rate is uniform across hubs, sources and customers — hubs are not
   disproportionately affected, despite being k-means centroids.

   The consequence: OSRM routes between *snapped* positions, so every distance carries a
   displacement error on top of the true road distance. It inflates absolute distance, duration
   and therefore cost per drop. On this instance the realized ratio of road to great-circle
   distance over tour legs is ~1.9, against a true landmark-measured circuity of 1.28; the gap
   is the artefact, not the city. It is visible as per-leg ratios below 1.0, which no real road
   network can produce.

   **It does not bias the comparison.** The same displacement applies identically to the greedy
   baseline, the GA and the OR-Tools reference, since all three route over the same matrix. The
   headline claim is a *relative* improvement over the baseline, and that is unaffected. Absolute
   ₹/drop should be read as inflated.

   Fixing it properly means constraining node placement to the network, which would make
   `generate_instance` depend on a running OSRM — giving up both offline `make data` and the
   guarantee that a seed alone determines the instance, which `coordinate_digest` and the matrix
   cache key rely on. That trade is not worth it for a bias that cancels.

5. **The haversine fallback is not a substitute for OSRM.** `circuity_factor = 1.30` is
   calibrated against real landmarks (see above), so it models road circuity honestly but does
   *not* reproduce OSRM's totals on this instance. Every reported number comes from OSRM; any run
   that falls back logs it at `WARNING`, and results from the two providers must not be compared.

6. **Capacity-balanced hub assignment does not pay here, and it is reported rather than hidden.**
   Nearest-hub assignment is badly lopsided: on seed 42 hub 9 draws 8,850 kg while two hubs draw
   300 kg each. The min-cost flow fixes exactly that — worst hub down to 2,738 kg at the default
   `hub_balance_slack = 1.25` — and the inbound leg gets **1.7% more expensive**: +82 km, and not
   one vehicle saved. Tightening the cap makes it worse, monotonically; a perfectly even split
   (`--slack 1.0`) costs 7.4% more than nearest-hub *and* adds a vehicle.

   The mechanism is that the vehicle floor is set by mass, and mass is not what makes an inbound
   tour expensive. A hub drawing 8,850 kg simply dispatches twelve vehicles, and their tours stay
   inside its own dense catchment. Moving a source to a less-loaded hub buys a long radial leg
   and saves nothing, because the vehicle it would have shared was going to be full either way.
   Balance only starts to pay where it happens to shed a whole vehicle, and at ₹1,000/vehicle
   that is a lumpy, seed-dependent effect rather than a trend.

   So Stage 1's improvement over the baseline — **−4.4%** on the inbound leg, 1,477 → 1,229 km at
   an unchanged 48 tours — is the **CVRP's**, not the assignment's. Both strategies ship, because
   the negative result is the interesting half of the ablation. `make stage1` prints all three
   columns side by side and states the verdict from the numbers.

   **That verdict is about the inbound leg alone, and step 6 found something it could not see.** A
   customer is served from the hub its parcel reached, so the assignment propagates: balancing
   also cuts Stage 2's largest hub from **236 stops to 73** (median 36 → 58). That is not just
   cheaper to price. For a fixed generation budget a flatter distribution is a smaller search
   space per stop, so equal GA effort buys more optimisation. Whether it pays back the 1.7% is
   **step 7's question and is not answered here** — `make run --strategy {nearest,balanced}`
   carries the flag through both stages so the ablation can measure it end to end. Read the total
   cost per drop, not the two legs separately: reading them separately is what hid this.

7. **OR-Tools optimises a static arc cost.** A `RoutingModel` fixes arc costs before the search
   begins, so the cumulative traffic model cannot live inside it; Stage 1's arc cost uses the
   dispatch-hour multiplier as a stand-in. Every *reported* distance, duration and arrival time
   still comes from the cumulative band-blended model in `src/tour.py`. The proxy affects which
   tour is chosen, never what that tour is then said to cost.

8. **The GA stops on its stagnation limit, and whether that means "converged" or "cut off"
   depends on the hub.** On seed 42 every hub ended on `stagnation_limit = 75` rather than on
   `generations = 600` — between 76 and 463 generations, median around 95. The run header reports
   generations used against the budget, so the figure never needs correcting further down.

   Three traces at full settings with `--trace-generations`, and the first reading of them was
   wrong. **Hub 9** (236 stops, stopped at generation 95, penalty ended at ×32) looked converged —
   13 incumbent improvements, all before generation 27, then 68 generations that found nothing.
   Re-running the same hub on the same seed with the penalty **pinned at ×1.00** ran **313
   generations and reached ₹28,516.5 against ₹29,172.6 — ₹656 better**, with 41 improvements and
   the last at generation 238. Its convergence was an artefact of the penalty, not a property of
   the problem. **Hub 0** (85 stops, stopped at 463, penalty ended at ×16) was genuinely still
   improving: 44 improvements tapering from 19 in the first 57 generations to 5 across 342–399,
   the last at 388, and it stopped with 137 generations of budget unspent.

   The mechanism is visible in the traces. Selection ranks plans at the adaptive rate while the
   incumbent is tracked at the configured one, and at a high multiplier those are different enough
   that elitism preserves the individual that is best *for the search* rather than the one that is
   best *as reported* — so the best-known plan is evicted from the population. The trace shows it
   as `population min` drifting above the incumbent: **73 of 95 generations on hub 9 at ×32, and 0
   of 313 once the penalty is pinned, 0 of 463 on hub 0 at ×16.** With the best plan gone from the
   population nothing can improve on it, and the stagnation counter expires on a search that had
   not finished. Somewhere between ×16 and ×32 the guidance stops guiding and starts discarding
   the answer.

   That is one hub on one seed, and the multipliers stay as configured until step 7 measures this
   across hubs and seeds. The transferable part is narrower: an early stop cannot be read as
   "converged" without checking whether the best plan was still in the population when it happened.

   The same traces killed both proposed explanations for stopping early. That the GA loses
   improvements by re-pricing only its search champion: hub 9 flagged **0 of 95** generations, hub 0
   **6 of 463**, and all six of those fell between generations 11 and 49 — none in the final 75,
   which is the window that actually decides when a run stops. Real, rare, and beside the point *for
   stopping* — though those six are genuine generations in which a cheaper plan sat in the
   population uncaptured. That the population collapses: **no duplicate children on either hub at
   all**, with 104–115 of 147 still novel in the closing ten generations.

   Whether capturing those missed plans would have helped is checkable rather than arguable,
   because the incumbent never feeds back into selection: the population evolves identically either
   way, so re-pricing every individual would simply take the running minimum of the `population
   min` the trace already records. On both hubs that comes to the final cost exactly — ₹29,172.6
   and ₹14,986.9. Hub 0's six missed plans were at most ₹71.1 better than the incumbent of the day,
   and the cheapest was ₹15,633 against its final ₹14,986.9. That is a measurement on two hubs, not
   a guarantee: a missed plan changes the answer whenever it beats everything the run later
   reaches.

9. **Guided local search under a wall-clock limit is not bit-reproducible.** It returns whatever
   it had reached when the clock ran out, so the same seed on a busier machine can yield a
   different plan. Run `--deterministic` to stop at the first-solution heuristic, which is
   reproducible; the test suite does. Step 9's multi-seed evaluation reports a spread for this
   reason.

---

## Development

```bash
make test      # ruff check + ruff format --check + mypy --strict + pytest
make data      # regenerate the instance
make providers # landmark distance comparison
make baseline  # greedy nearest-neighbour benchmark, print its metrics
make stage1    # inbound leg: baseline vs the CVRP under each hub assignment
make osrm      # one-time OSRM setup, then start the server
make osrm-down # stop it
```

No test touches the network. The OSRM provider is exercised against a loopback stub
(`tests/osrm_stub.py`) that speaks OSRM's real URL grammar and enforces the same
`max_table_size²` cell budget the production server does, so a chunking regression fails the suite
exactly as it would fail against a live instance.
