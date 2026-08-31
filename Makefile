PY := .venv/bin/python
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy
PYTEST := .venv/bin/pytest

SEED ?= 42

.PHONY: data lint format check types unit test

## generate a seeded synthetic instance, print summary stats, render the scatter plot
data:
	$(PY) -m src.cli.generate_data --seed $(SEED) --theme both

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

## the gate: everything that must pass before a step is considered complete
test: check types unit
