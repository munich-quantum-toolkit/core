# Independent native multi-program adoption

Status: extracted from Core PR #2226; local validation complete.

## Scope and decisions

Core issue #2362 owns Client APIs, bindings, bundled-device adaptation, and
indexed results for QDMI PR #509. The program-format enum remains unchanged.
Metadata removal, replaceable drivers, and payload capabilities are independent.
This targets Core 4.1 with released QDMI 1.4 before publication.

Retain existing single-program and calibration APIs, optional shot counts,
byte-exact binary payloads, session ownership and current concurrent execution.
DDSIM supports one program per native job for now; the model-only SC device does
not execute jobs. Neither device claims unsupported aggregate semantics. An
isolated test provider exercises multiple programs and indexed retrieval.

## Validation

Run the release build and CTest suite, generated stubs, Python QDMI/SDK tests,
repository lint, C++ lint, and the documented DDSIM example. Cover atomic
setters, indexed ordering and invalid indices, deep copies, binary bytes,
optional shots, retrieval, failure/cancellation and the single-program path.
Preserve current target inference and simulator concurrency regressions.

Local results: 3,879 native tests passed with one existing skip; 413 Python
QDMI/SDK tests passed; minimum-dependency testing passed 412 tests with one
expected skip. Generated stubs, repository lint and C++ lint passed. Hosted CI
remains a separate publication gate.

## Follow-ups

Core issue #2359 coordinates independent SDK and provider consumers. Do not
implement their fallback policy here or equate concurrent single submissions
with a native aggregate job. No payload descriptors or execution-capability
properties are introduced by this extraction.
