# QDMI adapter performance

Status: complete.

## Supported behavior and ownership

`DDSIMDevice` scopes an implicit session to its declared PennyLane wire count.
Explicit `session_parameters["custom1"]` takes precedence; omitted wires retain
65,535 sites. Multi-controlled gates use the declared sites. Deferred
measurements still require spare declared wires.

Qiskit job construction validates every compiled program before submission.
`prepareProgramSubmission` in `mlir/lib/Compiler/QDMIAdapter.cpp` owns the
compatibility check and calibration-free destination snapshot. Private Python
bindings group programs by format and retain one destination per format. A mixed
group containing a foreign artifact takes a fresh snapshot. Submission reuses
`trySubmitPrograms` without Python payload copies. Native work releases the GIL;
Python attribute access and conversion hold it. Public APIs and Batch ordering,
attempts, cancellation, fallback, and tracker semantics are unchanged.

Provider name filtering opens devices and checks their names before constructing
backends. Nonmatching devices therefore cannot emit snapshot-failure warnings;
device-open failures still emit sanitized warnings. Qiskit target construction
materializes the site sequence once without retaining another cache.

## Evidence and exclusions

Measurements use at least five runs with `time.perf_counter`. The initial
baseline is `7a3e5f0`; runtime readout probes include merged #2953
(`dcaa0883e6`). Alternating comparisons on ARM64 show lower costs for scoped
PennyLane sessions, foreign-artifact preparation, provider name lookup, and
full-width Qiskit target construction. The host is shared; absolute timings are
not portable throughput claims. Harnesses, raw measurements, and environment
details remain outside the repository.

The implementation incorporates #2954 shared decoding and reuses #2955 native
submission. Artifact-preparation timings include work from #2955; they do not
establish an incremental speedup over that implementation.

Compilation remains the main adapter cost after session scoping. No compiler
pipeline, cache, scheduling, public sampler, or validation-ownership changes are
included. Shot-vector copies retain separate recovery entries. Compatibility
checks at four sites are too small to justify an identity fast path. A distinct
bitstring decode cache helped repeated outcomes but regressed high-entropy
samples, so runtime decoding is unchanged. SC artifact execution could not be
measured because the registered SC model has no executable compiler format.

## Validation

The plugin, QDMI, and MLIR Python suites pass (1,140 tests), including Catalyst.
Native CTest passes (4,346 tests). Stub generation, C++ lint, and repository
lint pass. Regression probes detect the old wire width, mixed-artifact
stale-snapshot acceptance, and unnecessary provider target construction. Native
tests check one snapshot per prepared group and no ordinary calibration reads;
Python tests cover alternate formats and multi-program admission fallback.
