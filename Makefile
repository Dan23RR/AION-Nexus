# AION-NEXUS — common operations
# Run `make help` for the catalog of targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

CHECKPOINT ?= checkpoints/aion_nexus_v1.pth
PORT ?= 8080

.PHONY: help install install-dev install-onnx test test-fast test-smoke test-edge test-determinism test-concurrency test-api lint format type-check verify benchmark smoke audit onnx quantize manifest-write manifest-check docker-build docker-run docker-stop clean

help:  ## Show this help
	@echo "AION-NEXUS Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- Setup ---------------------------------------------------------------

install:  ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@command -v pre-commit >/dev/null && pre-commit install || echo "pre-commit not installed; skipping hook setup"

install-onnx:  ## Install ONNX export dependencies
	$(PIP) install onnx onnxruntime

# ---- Testing -------------------------------------------------------------

test:  ## Run full test suite (excluding tests requiring real data)
	$(PYTEST) tests/ -v --tb=short

test-fast:  ## Run only fast smoke tests (~30 seconds)
	$(PYTEST) tests/test_smoke.py -v --tb=short

test-smoke:  ## Alias for test-fast
	$(MAKE) test-fast

test-edge:  ## Run preprocessing edge-case tests
	$(PYTEST) tests/test_preprocessing_edge_cases.py -v --tb=short

test-determinism:  ## Run determinism / reproducibility tests
	$(PYTEST) tests/test_determinism.py -v --tb=short

test-concurrency:  ## Run thread-safety / concurrency tests
	$(PYTEST) tests/test_concurrency.py -v --tb=short

test-api:  ## Run API integration tests (requires fastapi + httpx test client)
	$(PYTEST) tests/test_api_integration.py -v --tb=short

# ---- Code quality -------------------------------------------------------

lint:  ## Run ruff linter
	@command -v ruff >/dev/null || { echo "ruff not installed: pip install ruff"; exit 1; }
	ruff check aion_nexus server tests scripts examples

format:  ## Auto-format with ruff
	@command -v ruff >/dev/null || { echo "ruff not installed: pip install ruff"; exit 1; }
	ruff format aion_nexus server tests scripts examples
	ruff check --fix aion_nexus server tests scripts examples

type-check:  ## Run mypy type checker
	@command -v mypy >/dev/null || { echo "mypy not installed: pip install mypy"; exit 1; }
	mypy aion_nexus server scripts

# ---- Operations ---------------------------------------------------------

verify:  ## Run F1 verification on FEMTO + MFPT (requires checkpoint + data)
	$(PYTHON) -m scripts.verify_checkpoint --checkpoint $(CHECKPOINT)

benchmark:  ## Run inference latency benchmark
	$(PYTHON) -m scripts.benchmark_inference --batch-size 1
	$(PYTHON) -m scripts.benchmark_inference --batch-size 32

smoke:  ## Quick sanity: model loads + 1 inference (no test framework)
	@$(PYTHON) -c "from aion_nexus import InferenceEngine, create_aion_nexus, NUM_CHANNELS, SIGNAL_LENGTH; \
	  import numpy as np; \
	  engine = InferenceEngine(create_aion_nexus()); \
	  sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32); \
	  r = engine.predict(sig); \
	  print(f'OK: predicted={r.predicted_class_name} conf={r.confidence:.3f} latency={r.latency_ms:.1f}ms')"

audit:  ## Run supply-chain audit (CVEs + license check)
	$(PYTHON) -m scripts.audit_supply_chain

onnx:  ## Export checkpoint to ONNX
	$(PYTHON) -m scripts.export_onnx --checkpoint $(CHECKPOINT) --dynamic-batch

quantize:  ## Quantize checkpoint to INT8 (with --validate to verify parity)
	$(PYTHON) -m scripts.quantize --in $(CHECKPOINT) --out checkpoints/aion_nexus_v1_int8.pth --validate

manifest-write:  ## Generate SHA-256 manifest of all package files
	$(PYTHON) -m scripts.generate_manifest

manifest-check:  ## Verify package files match manifest
	$(PYTHON) -m scripts.generate_manifest --check

# ---- Docker -------------------------------------------------------------

docker-build:  ## Build production container image
	docker build -t aion-nexus:1.0.1 .
	docker tag aion-nexus:1.0.1 aion-nexus:latest

docker-run:  ## Run container locally on $(PORT) with checkpoint mounted
	docker run -d --rm --name aion-nexus \
	  -p $(PORT):8080 \
	  -v $$(pwd)/checkpoints:/app/checkpoints:ro \
	  aion-nexus:latest
	@echo "Started AION-NEXUS on http://localhost:$(PORT)"
	@echo "Health: curl http://localhost:$(PORT)/health"

docker-stop:  ## Stop and remove the running container
	-docker stop aion-nexus
	-docker rm aion-nexus

# ---- Cleanup ------------------------------------------------------------

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
	@echo "Cleaned."
