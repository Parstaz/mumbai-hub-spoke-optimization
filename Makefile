PY := .venv/bin/python
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy
PYTEST := .venv/bin/pytest

SEED ?= 42

# Extra flags for the run targets: `make run ARGS="--strategy balanced"`. Needed because make
# claims any bare `--flag` on its own command line as one of its own options and exits, so
# `make run --strategy balanced` fails with "unrecognized option" rather than reaching Python.
ARGS ?=

# 5001 rather than OSRM's conventional 5000: macOS holds 5000 with the AirPlay Receiver unless
# it is disabled in System Settings > General > AirDrop & Handoff, so 5000 fails on a fresh
# clone on every Mac. Override to get it back: `OSRM_PORT=5000 make osrm-up providers`.
# RunConfig.osrm_url defaults to the same port; change both together or the pipeline points at
# nothing and falls back to haversine, which reads as an outage rather than a mismatch.
OSRM_PORT ?= 5001
export OSRM_PORT

OSRM_DIR := data/osrm
OSRM_PBF := $(OSRM_DIR)/maharashtra-latest.osm.pbf
# openstreetmap.fr rather than Geofabrik: Geofabrik does not publish a per-state India extract,
# only six multi-state zones, and its 302 to the index page for a non-existent path is
# indistinguishable from a download to anything that does not check what it received.
OSRM_EXTRACT_URL := https://download.openstreetmap.fr/extracts/asia/india/maharashtra-latest.osm.pbf

.PHONY: data providers baseline stage1 run ablation osrm osrm-up osrm-down lint format check types unit cov test

## generate a seeded synthetic instance, print summary stats, render the scatter plot
data:
	$(PY) -m src.cli.generate_data --seed $(SEED) --theme both

## landmark distances under both providers, plus the traffic bands applied to one leg
providers:
	$(PY) -m src.cli.compare_providers --osrm-url http://127.0.0.1:$(OSRM_PORT)

## greedy nearest-neighbour benchmark on the seeded instance, with its full metrics table
baseline:
	$(PY) -m src.cli.run_baseline --seed $(SEED) --osrm-url http://127.0.0.1:$(OSRM_PORT)

## stage 1 inbound leg: greedy baseline vs the CVRP under each hub assignment strategy
stage1:
	$(PY) -m src.cli.run_stage1 --seed $(SEED) --osrm-url http://127.0.0.1:$(OSRM_PORT) $(ARGS)

## full Stage 1 + Stage 2 pipeline on one seed, printed against the greedy baseline
run:
	$(PY) -m src.cli.run_pipeline --seed $(SEED) --osrm-url http://127.0.0.1:$(OSRM_PORT) $(ARGS)

## step 7's 2x2: memetic local search on/off x nearest/balanced, read on total cost per drop
ablation:
	$(PY) -m src.cli.run_ablation --seed $(SEED) --osrm-url http://127.0.0.1:$(OSRM_PORT) $(ARGS)

## one-time OSRM setup: download the extract, then extract -> partition -> customize -> routed.
## Takes 15-30 minutes and roughly 4 GB of RAM; the artefacts persist in $(OSRM_DIR).
osrm: $(OSRM_PBF)
	docker compose --profile build run --rm osrm-extract
	docker compose --profile build run --rm osrm-partition
	docker compose --profile build run --rm osrm-customize
	$(MAKE) osrm-up

# Downloaded to .part and renamed only once it is verified to be a PBF. A mirror that answers a
# bad path with an HTML page otherwise leaves a file make considers up to date, and osrm-extract
# fails 20 minutes later with "invalid BlobHeader size" instead of "that is not an extract".
$(OSRM_PBF):
	mkdir -p $(OSRM_DIR)
	curl -fL --retry 3 -o $@.part $(OSRM_EXTRACT_URL)
	@head -c 32 $@.part | grep -aq OSMHeader \
	  || { echo "ERROR: $(OSRM_EXTRACT_URL) returned $$(file -b $@.part), not an OSM PBF"; \
	       rm -f $@.part; exit 1; }
	mv $@.part $@

osrm-up:
	docker compose up -d osrm

osrm-down:
	docker compose down

lint:
	$(RUFF) check . --fix

format:
	$(RUFF) format .

check:
	$(RUFF) check .
	$(RUFF) format --check .

types:
	$(MYPY) --strict src/

unit:
	$(PYTEST) -q

## line coverage against the >= 90% standard. Deliberately not part of `test`: the gate stays
## the four checks the definition of done names, and coverage is read, not enforced by a number.
# src.workload, src.tour, src.scoring and src.arc_model are named explicitly: they hold logic
# extracted out of src/baseline and src/stage1, and a package-path target would have silently
# dropped it from the measurement at the moment it stopped living under a measured directory.
cov:
	$(PYTEST) -q --cov=src/baseline --cov=src/costs --cov=src/stage1 --cov=src/stage2 \
	  --cov=src.workload --cov=src.tour --cov=src.scoring --cov=src.arc_model \
	  --cov-report=term-missing

## the gate: everything that must pass before a step is considered complete
test: check types unit
