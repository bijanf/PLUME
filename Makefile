# Reproduce the results and figures of the paper.  `make help` lists the targets.

PYTHON ?= python

# Each experiment reads only the results/ files written by those listed before it.
# The environment is recorded first, while the working tree is still clean.
EXPERIMENTS := environment stratification constants forward_checks identifiability \
               nesca clast_shape vent_shape heat_budget source_discrimination \
               heat_context kernel_maps modes nesca_maps shape_sensitivity \
               shape_space source_evidence_shape synthetic vent_misfit

# In the order the figures appear in the paper.
FIGURES := kernels svd nesca vent shape sources

.DEFAULT_GOAL := help
.PHONY: help install data data-all test lint format results figures all clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*## "} {printf "  %-9s %s\n", $$1, $$2}'

install: ## Install the package and its development tools
	$(PYTHON) -m pip install -e ".[dev]"

data: ## Fetch and verify the NESCA workbooks (0.4 MB)
	$(PYTHON) scripts/fetch_data.py

data-all: ## Fetch the NESCA workbooks and the World Ocean Atlas files (162 MB)
	$(PYTHON) scripts/fetch_data.py --woa

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check linting and formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply lint fixes and formatting
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

results: data-all ## Recompute every file in results/
	@set -e; for e in $(EXPERIMENTS); do \
	  echo "==> experiments/$$e"; $(PYTHON) experiments/$$e/run.py; \
	done
	$(PYTHON) experiments/synthetic/sbc.py
	$(PYTHON) experiments/manifest/run.py

figures: ## Draw the figures of the paper from results/ into figures/
	@set -e; for f in $(FIGURES); do $(PYTHON) figures/src/fig_$$f.py; done

all: results figures ## Recompute every result, then draw the figures

clean: ## Remove drawn figures, coverage reports and tool caches
	rm -rf figures/*.pdf figures/*.png .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
