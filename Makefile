# =============================================================================
# QSFL Makefile  (target: Linux x86_64; see README for Windows notes)
#
#   make setup        CPU install from the pinned lock (smoke-test ready, no GPU,
#                     no native build) — uses pure-Python kyber-py KEM.
#   make setup-gpu    install CUDA torch wheels for the H100 (cu124).
#   make lock         recompile requirements.txt from requirements.in.
#   make test         fast CPU-only end-to-end smoke test + unit tests (minutes).
#   make demo         run the headline single-model demo (configs/single_model).
#   make train        run federated training (configs/federated).
#   make eval         run the attack-evaluation suite.
#   make lint         ruff check.
#   make clean        remove caches + intermediate checkpoints (keep disk lean).
# =============================================================================

PY ?= python3
PIP ?= $(PY) -m pip
# H100 wheels. Adjust the CUDA tag if your toolkit differs (cu121/cu124/cu126).
TORCH_CUDA_INDEX ?= https://download.pytorch.org/whl/cu124
TORCH_VERSION ?= 2.6.0
TORCHVISION_VERSION ?= 0.21.0

.PHONY: setup setup-gpu setup-liboqs lock test test-fast demo train eval lint clean help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}' || \
	sed -n 's/^#   //p' $(MAKEFILE_LIST)

setup: ## CPU install from pinned lock (smoke-test ready)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo ">> CPU setup complete. Run 'make test' for the smoke test."

setup-gpu: ## Install CUDA torch wheels for the H100 (cu124)
	$(PIP) install torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION) --index-url $(TORCH_CUDA_INDEX)
	@echo ">> GPU torch installed from $(TORCH_CUDA_INDEX)."

setup-liboqs: ## Install the liboqs-python PRIMARY KEM backend (needs liboqs; prefer conda env)
	$(PIP) install liboqs-python==0.10.0
	@echo ">> liboqs-python installed. Set aggregation.kem.backend=liboqs to use it."

lock: ## Recompile requirements.txt from requirements.in (uv or pip-tools)
	@if command -v uv >/dev/null 2>&1; then \
		uv pip compile requirements.in -o requirements.txt ; \
	else \
		$(PY) -m piptools compile requirements.in -o requirements.txt ; \
	fi
	@echo ">> requirements.txt regenerated."

test: ## Fast CPU-only smoke test + unit tests
	$(PY) -m pytest -m "not slow and not gpu and not liboqs" -q
	$(PY) scripts/run_pipeline.py --config configs/smoke.yaml

test-fast: ## Unit tests only (no end-to-end pipeline run)
	$(PY) -m pytest -m "not slow and not gpu and not liboqs" -q

demo: ## Headline single-model demo (Diagram A)
	$(PY) scripts/run_pipeline.py --config configs/single_model.yaml

train: ## Federated training (Diagram B)
	$(PY) scripts/run_pipeline.py --config configs/federated.yaml

eval: ## Attack-evaluation suite
	$(PY) scripts/run_eval.py --config configs/federated.yaml

lint: ## ruff check
	$(PY) -m ruff check src tests scripts

clean: ## Remove caches + intermediate checkpoints
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ __pycache__
	rm -rf checkpoints/*.tmp results/_intermediate
	@echo ">> cleaned."
