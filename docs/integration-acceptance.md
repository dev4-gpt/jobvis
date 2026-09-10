# Jobvis integration acceptance

Verified on 2026-09-10 in the active `observable-job-agent` checkout. The older
detached checkout and separately installed legacy worker were not modified.

## Runtime diagnosis

The earlier “35 tests passed” claim was not used as acceptance evidence. Bounded
Python subprocess checks with faulthandler reproduced stalls while importlib was
reading existing cached files. A clean Python 3.12.12 environment was installed
from the lockfile with the application extra, and a separate bytecode cache made
the full API import and test collection work. This identifies the observed stall
location, not a proven underlying filesystem cause. Existing environments/caches
were preserved.

Validation environment: `/private/tmp/jobvis-integration-venv`, bytecode cache
`/private/tmp/jobvis-integration-pycache`. Frontend dependencies were installed with
`npm ci` from the existing lockfile in `/private/tmp/jobvis-web-check.IU4tWv` after
the original dependency tree also stalled. These temporary directories are not
required application paths and can be recreated.

## Checks

| Check | Result |
| --- | --- |
| Offline Python suite | 428 passed; four PDF-compilation tests deselected |
| Real Tectonic PDF compilation | Four passed |
| Ruff: source, tests, changed verification scripts | Passed |
| Frontend TypeScript, ESLint, production build | Passed |
| Chromium on the final static build | Expansion, memory save/confirm/delete, snapshot approval and export passed with API fixtures |

Backend tests use temporary candidate/application stores, mocked providers and a
test encryption key. Coverage includes registry overrides, single-flight board
caching, candidate policy, conservative liveness/deadlines, manual-only Apify,
configuration and provider failures, cancellation, deduplication and ranking caps,
encrypted restart/isolation/deletion, confirmation and legacy imports, current
corpus grounding, queue schema/score boundaries, stable IDs and capacity, stale
approvals, audit hashes, missing/tampered artifacts, repeat exports and tracker dates.

The console smoke script is `scripts/check_integration_console.py`. It serves a
static build on loopback, mocks API responses and blocks non-loopback requests.
Backend API tests separately exercise the real routes with temporary stores.
The browser fixture and backend tests are complementary checks, not a claim of
one live provider-to-worker run. Built output was copied into ignored `web/out`
for the local application.

Reproduction commands (from the repository, with a healthy Python 3.12 environment):

```sh
uv sync --frozen --extra application
uv run pytest -q -m 'not integration and not compile'
uv run pytest -q -m compile
uv run ruff check src tests scripts/check_integration_console.py scripts/verify_local_pack.py
cd web
npm ci
npm run typecheck
npm run lint
npm run build
cd ..
uv run python scripts/check_integration_console.py --web-dir web/out
```

Playwright needs Chromium installed. Tectonic is required for the compile tests.
One existing Starlette warning recommends its newer test-client transport;
it does not fail these tests.

## Connected versus live-verified

All five integrations have console/backend wiring and fixture coverage. Curated
boards honor the direct-source switch; Apify remains disabled without valid task
configuration, and memory persistence fails closed without encryption.

Not verified live in this run: every curated employer board, an actual paid Apify
task and its dataset mapping/cancellation, the user's macOS Keychain permissions,
or AIHawk accepting and running an export bundle. No provider charges were incurred
by the verification fixtures, no applications were submitted, and no outreach,
external memory synchronization, reminders, or worker dispatch were started.

Legacy plaintext memory is imported only by explicit action, remains unconfirmed,
and its original file is preserved. The UI warns that this plaintext copy still
exists. Memory is encrypted local keyword retrieval, not vector search or Synaptic
protocol interoperability. Export bundles contain local PDFs protected by filesystem
permissions; unlike memory, these worker-readable artifacts are not encrypted.
