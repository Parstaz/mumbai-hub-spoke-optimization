# mumbai-hub-spoke-optimization

Two-stage hub-and-spoke pickup and delivery optimization over a synthetic Mumbai / Navi Mumbai
network. Stage 1 consolidates ~300 sources into 16 hubs (OR-Tools CVRP + min-cost flow); Stage 2
runs a hand-written genetic algorithm to deliver from those hubs to ~800 customers under capacity
and time windows. The headline KPI is **cost per drop (₹/delivery)**.

> **Build status.** This README currently documents the cost layer only — the distance/duration
> matrices and the traffic model (step 2 of 9). The full write-up, results tables and figures land
> with step 9.

---

## Running it without any setup

The cost layer has a great-circle fallback with no external dependencies, so the pipeline runs on
a clean checkout with no Docker and no OSM data:

```bash
make data                                   # seeded synthetic instance + scatter plot
.venv/bin/python -m src.cli.compare_providers --no-osrm
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

---

## Development

```bash
make test      # ruff check + ruff format --check + mypy --strict + pytest
make data      # regenerate the instance
make providers # landmark distance comparison
make osrm      # one-time OSRM setup, then start the server
make osrm-down # stop it
```

No test touches the network. The OSRM provider is exercised against a loopback stub
(`tests/osrm_stub.py`) that speaks OSRM's real URL grammar and enforces the same
`max_table_size²` cell budget the production server does, so a chunking regression fails the suite
exactly as it would fail against a live instance.
