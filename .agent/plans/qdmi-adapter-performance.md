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
bindings use LLVM small containers to group programs and return one Python
destination per format. A mixed group containing a foreign artifact takes a
fresh snapshot. Submission reuses `trySubmitPrograms` without Python payload
copies. Native work releases the GIL; Python attribute access and conversion
hold it. Public APIs and Batch ordering, attempts, cancellation, fallback, and
tracker semantics are unchanged.

Provider name filtering opens devices and checks their names before constructing
backends. Nonmatching devices therefore cannot emit snapshot-failure warnings;
device-open failures still emit sanitized warnings. Qiskit target construction
materializes the site sequence once without retaining another cache.

Catalyst readout retains the shared decoder's MSB-first strings. Consumers
select wire order directly; numeric sample expansion fills its checked
final-size array without reversing strings or appending one element at a time.

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
checks at four sites are too small to justify an identity fast path. Direct
shared decoding and bounded-cache prototypes improved high-entropy samples but
regressed repeated or intermediate distributions. The shared decoder is
unchanged; the retained runtime conversion removes a redundant pass and improves
all measured distributions. SC artifact execution could not be measured because
the registered SC model has no executable compiler format.

## Validation

The plugin, QDMI, and MLIR Python suites pass (1,141 tests), including Catalyst.
Native CTest passes (4,346 tests). Stub generation, C++ lint, and repository
lint pass. Regression probes detect the old wire width, mixed-artifact
stale-snapshot acceptance, and unnecessary provider target construction. Native
tests check one snapshot per prepared group and no ordinary calibration reads;
Python tests cover alternate formats and multi-program admission fallback.

## Follow-up conclusions

Seven alternating scoped-W1 measurements on four performance cores reduce
compile-only time from 35.94 ms to 10.79 ms and complete gradients from 69.49 ms
to 43.83 ms using an external four-worker prototype. The current all-to-all,
single-function workload does not exploit inner compiler parallelism. A native
batch helper should reuse the existing LLVM pool, preserve indexed errors, and
fall back to serial for shared MLIR contexts. Program copies share contexts;
independently imported tapes do not. General input-consumption semantics and
explicit-connectivity workloads need more proof. Estimated implementation:
100–145 production lines plus tests. Catalyst's individual synchronous calls do
not expose a batch and would not benefit.

An external import-once sampler prototype preserves all 200 W5 payloads while
reducing median end-to-end time from 659.42 ms to 533.14 ms across five
alternating runs. It retains Python validation. A production implementation must
preserve PUB shapes, registers, metadata, per-PUB shots, preprocessing
overrides, and Batch recovery. An internal sampler subclass plus a private
native binding is estimated at 120–180 production lines plus 100–180 test lines.
The relevant private Qiskit hooks match in supported versions 2.5.0 and 2.6.1,
but future compatibility remains a cost. Neither parallel compilation nor the
native sampler is implemented in this PR.
