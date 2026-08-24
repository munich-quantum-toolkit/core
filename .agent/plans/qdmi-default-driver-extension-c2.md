# Optional packaged QDMI driver extension

Status: independently rebased and validated locally.

## Motivation and settled boundary

Installed providers need catalogue discovery and targeted stable-ID opening
without importing vendor Python code or copying libraries beside the driver.
Core's packaged driver provides an optional private extension for this purpose.
The standard Client ABI remains usable with drivers that lack that extension.
Standardizing discovery/configuration is QDMI v2 work.

This is Core #2230 on #2229, targeting Core 4.1 / QDMI 1.4. It does not depend
on metadata removal, batching, payload capabilities, or compiler changes.

## Implementation

- Load the two optional private symbols in the Client wrapper. Staging a
  manifest does not select the process-wide driver; successful raw targeted
  allocation does, even when subsequent initialization fails.
- Stage trusted manifests transactionally at lowest precedence, freeze the
  registry after successful construction, and keep canonical paths idempotent.
- Open exactly one stable ID with strict per-call overrides. Preserve sized
  custom values, reject malformed paths/IDs, and propagate provider errors.
- Discover Python manifests using entry-point metadata and wheel RECORD paths,
  never provider imports. Invalid automatic entries warn and are skipped;
  explicit staging remains strict.
- Retain initialized provider libraries across independent sessions. Complete
  initialization before moving the session owner, including on Windows.

The entry points are qdmi::default_driver::addManifest/openDevice and the Python
default_driver submodule. No public QDMI C header changes are needed.
Installed-consumer packaging is the separate follow-up #2231.

## Acceptance

Run the independent release build and CTest suite, generated stubs, QDMI/SDK
Python tests, repository lint, and C++ lint. Check absent optional symbols,
selection timing, freeze rollback, idempotent paths, strict overrides, valid
warning outputs, malformed JSON, UTF-8 paths, and session lifetime. Discovery
tests must reject missing RECORD, ambiguous/off-anchor paths and traversal while
proving that provider code is not imported. Retain current optional-device
configurations, concurrency, and compiler rules.

The release suite passed 3,873 tests with one existing skip. All 467 selected
Python tests passed. Stub generation, repository lint and C++ lint passed.

## Recovery and non-goals

Keep useful commits, human attribution and review threads. Use guarded pushes;
do not create archive branches or request reviews. There is no new registry API,
public discovery standard, scheduler policy, or payload contract here.
