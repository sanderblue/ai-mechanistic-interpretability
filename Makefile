# Developer entry points. `PY` lets you point at a specific interpreter, e.g.
#   make e01 PY=.venv/bin/python
PY ?= python

.PHONY: help setup lint format test test-integration e01 e02 clean

help:
	@echo "setup            install the package + both backends + dev tools (editable)"
	@echo "lint             ruff check + format --check"
	@echo "format           ruff format + ruff check --fix"
	@echo "test             fast, hermetic unit tests (no model downloads)"
	@echo "test-integration model-backed tests (downloads weights, runs forward passes)"
	@echo "e01              logit-lens experiment on GPT-2"
	@echo "e02              causal-tracing experiment on GPT-2"
	@echo "clean            remove caches and experiment outputs"

setup:
	$(PY) -m pip install -e ".[tl,nnsight,dev]"

lint:
	ruff check interp experiments tests
	ruff format --check interp experiments tests

format:
	ruff format interp experiments tests
	ruff check --fix interp experiments tests

test:
	$(PY) -m pytest

test-integration:
	$(PY) -m pytest -m integration

e01:
	$(PY) -m interp.run logit_lens --config configs/logit_lens_gpt2.yaml

e02:
	$(PY) -m interp.run causal_tracing --config configs/causal_tracing_gpt2.yaml

clean:
	rm -rf outputs/* .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
