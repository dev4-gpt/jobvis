.DEFAULT_GOAL := help

# Resolve every command relative to the checkout that owns this Makefile. This
# matters when a user runs `make -f .../Makefile app` from the parent Projects
# directory: `PYTHONPATH=src` must never accidentally point at the parent.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Keep the src/ layout explicit so local commands do not depend on the
# platform-specific behavior of uv/hatch editable installs.
UV_RUN := cd "$(REPO_ROOT)" && PYTHONPATH=src uv run
# Override with `make CONSOLE_PORT=8001 WIZARD_PORT=7861 app`, or export the
# corresponding JOBVIS_* variables. This lets Jobvis coexist with unrelated
# local services on either default port.
CONSOLE_PORT ?= $(if $(JOBVIS_CONSOLE_PORT),$(JOBVIS_CONSOLE_PORT),8000)
WIZARD_PORT ?= $(if $(JOBVIS_WIZARD_PORT),$(JOBVIS_WIZARD_PORT),7860)
JOBVIS_CONSOLE_ENV := $(if $(JOBVIS_CONSOLE_PORT),JOBVIS_CONSOLE_PORT=$(JOBVIS_CONSOLE_PORT),$(if $(filter-out 8000,$(CONSOLE_PORT)),JOBVIS_CONSOLE_PORT=$(CONSOLE_PORT),))
JOBVIS_WIZARD_ENV := $(if $(JOBVIS_WIZARD_PORT),JOBVIS_WIZARD_PORT=$(JOBVIS_WIZARD_PORT),$(if $(filter-out 7860,$(WIZARD_PORT)),JOBVIS_WIZARD_PORT=$(WIZARD_PORT),))

# Self-documenting help: any target with a `## comment` is listed.
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install deps and pre-commit hooks
	uv sync --all-groups --no-editable
	uv run pre-commit install

.PHONY: app
app: preflight port-check ## Launch both surfaces: wizard on configured port and console on configured port
	cd "$(REPO_ROOT)" && $(JOBVIS_WIZARD_ENV) $(JOBVIS_CONSOLE_ENV) PYTHONPATH=src uv run python -m job_scout.app

.PHONY: preflight
preflight: source-check web-check ## Validate source, console build, imports, and model configuration before launch
	$(UV_RUN) python -m job_scout.health

.PHONY: source-check
source-check: ## Verify this checkout contains the current import-safe runtime
	$(UV_RUN) python scripts/runtime_source_check.py

.PHONY: web-check
web-check: ## Fail clearly when the static voice-console build is missing
	@test -f "$(REPO_ROOT)/web/out/index.html" || (echo "Jobvis console is not built at $(REPO_ROOT)/web/out. Run 'make web-build' once, then run 'make app' again."; exit 1)

.PHONY: port-check
port-check: ## Fail clearly if either configured Jobvis port is already occupied
	@for port in $(WIZARD_PORT) $(CONSOLE_PORT); do \
		pids="$$(lsof -tiTCP:$$port -sTCP:LISTEN 2>/dev/null || true)"; \
		if [ -n "$$pids" ]; then \
			echo "Jobvis cannot start: port :$$port is already in use (PID(s): $$pids). Run 'make WIZARD_PORT=$(WIZARD_PORT) CONSOLE_PORT=$(CONSOLE_PORT) stop' or choose different ports."; \
			exit 1; \
		fi; \
	done

.PHONY: stop
stop: ## Stop only local listeners occupying the configured Jobvis ports
	@for port in $(WIZARD_PORT) $(CONSOLE_PORT); do \
		pids="$$(lsof -tiTCP:$$port -sTCP:LISTEN 2>/dev/null || true)"; \
		for pid in $$pids; do \
			command="$$(ps -p $$pid -o command= 2>/dev/null || true)"; \
			case "$$command" in \
				*job_scout.app*) echo "Stopping Jobvis listener on :$$port (PID $$pid)"; kill $$pid ;; \
				*) echo "Leaving non-Jobvis process on :$$port (PID $$pid): $$command" ;; \
			esac; \
		done; \
		if [ -z "$$pids" ]; then \
			echo "No listener on :$$port"; \
		fi; \
	done

.PHONY: jobvis-api
jobvis-api: ## API only, no wizard — for frontend work with `make web-dev` (empty session)
	uv run python -m job_scout.api

.PHONY: web-build
web-build: ## Build the Jobvis console into web/out (static export served by `make jobvis`)
	cd "$(REPO_ROOT)/web" && npm ci && npm run build

.PHONY: web-dev
web-dev: ## Next dev server on :3000 against the API on :8000 (run `make jobvis` too)
	cd "$(REPO_ROOT)/web" && npm run dev

.PHONY: web-assets
web-assets: ## Vendor the MediaPipe hand-tracking assets (only needed for gesture control)
	@mkdir -p "$(REPO_ROOT)/web/public/mediapipe/wasm"
	cp "$(REPO_ROOT)"/web/node_modules/@mediapipe/tasks-vision/wasm/* "$(REPO_ROOT)/web/public/mediapipe/wasm/"
	curl -fL -o "$(REPO_ROOT)/web/public/mediapipe/hand_landmarker.task" \
		https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
	@echo "gestures ready — set NEXT_PUBLIC_JOBVIS_GESTURES=1 in web/.env.local and rebuild"

.PHONY: batch
batch: ## Run the baseline batch (prompts for --yes cost confirmation)
	$(UV_RUN) python scripts/run_batch.py

.PHONY: snapshot
snapshot: ## Rebuild data/cached_jobs.json from live sources
	$(UV_RUN) python scripts/snapshot_jobs.py

.PHONY: fixtures
fixtures: ## Regenerate the synthetic fixture CV PDFs + LinkedIn export ZIPs
	$(UV_RUN) python scripts/generate_fixture_cvs.py
	$(UV_RUN) python scripts/generate_fixture_linkedin.py

.PHONY: tailor-batch
tailor-batch: ## Run the Phase 2 tailoring batch (prompts for --yes cost confirmation)
	$(UV_RUN) python scripts/run_tailor_batch.py

.PHONY: eval-datasets
eval-datasets: ## Push ranking + tailoring datasets to Opik from traces
	$(UV_RUN) python scripts/build_eval_dataset.py --kind ranking --push
	$(UV_RUN) python scripts/build_eval_dataset.py --kind tailoring --push

.PHONY: evals
evals: ## Show the eval harness usage (each suite prompts for --yes)
	$(UV_RUN) python scripts/run_evals.py --help

.PHONY: backtest
backtest: ## Run the deterministic application-pack backtest (CV= PACK= JOB=)
	@test -n "$(CV)" || (echo "Usage: make backtest CV=/path/to/cv.pdf PACK=/path/to/pack.json JOB=/path/to/job.txt"; exit 2)
	@test -n "$(PACK)" || (echo "Usage: make backtest CV=/path/to/cv.pdf PACK=/path/to/pack.json JOB=/path/to/job.txt"; exit 2)
	@test -n "$(JOB)" || (echo "Usage: make backtest CV=/path/to/cv.pdf PACK=/path/to/pack.json JOB=/path/to/job.txt"; exit 2)
	$(UV_RUN) python scripts/backtest_pack.py --cv "$(CV)" --pack "$(PACK)" --job-description "$(JOB)" $(if $(OUTPUT),--output "$(OUTPUT)",)

.PHONY: queue
queue: ## Create the Opik annotation queue + feedback definitions
	$(UV_RUN) python scripts/setup_annotation_queue.py --queue

.PHONY: jobvis-agent
jobvis-agent: ## Create/update the Jobvis ElevenLabs agent (prints the agent id)
	$(UV_RUN) python scripts/setup_jobvis_agent.py

.PHONY: test
test: ## Run the test suite
	$(UV_RUN) pytest

.PHONY: lint
lint: ## Lint with ruff
	$(UV_RUN) ruff check .

.PHONY: health
health: source-check ## Run the keyless import/config health check
	$(UV_RUN) python -m job_scout.health

.PHONY: ci
ci: ## Run the keyless checks used by pull-request CI
	$(UV_RUN) ruff format --check .
	$(UV_RUN) ruff check .
	$(UV_RUN) pyright
	$(UV_RUN) pytest
	$(UV_RUN) pytest gates/ -v

.PHONY: format
format: ## Format with ruff
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

.PHONY: gates
gates: ## Deterministic eval regression gate (Opik dataset access, zero LLM calls)
	$(UV_RUN) pytest gates/ -v

.PHONY: search-bench
search-bench: ## Paired soft-deadline benchmark (live job APIs, no LLM calls)
	$(UV_RUN) python scripts/bench_search.py
