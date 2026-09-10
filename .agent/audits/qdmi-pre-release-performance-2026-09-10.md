# QDMI pre-release performance and determinism audit

Date: 2026-09-10. Status: both confirmed findings applied; final re-audit
complete.

Audit and measurement baseline: `ad74680f1ef380456a1b89a810ef33ee8d218f69`.
Publication base: `9d6526f48`, after a clean rebase over three upstream commits.
Those commits do not change the three production/test files in this diff. Exact
measured source hashes and reproduction commands are retained in the
[benchmark record](../benchmarks/qdmi-pre-release-performance/README.md).

## Result

No further actionable finding or v4 blocker was established in this bounded
re-audit. Both measured P2 findings are applied: native parsing and compilation
release the Python GIL, and scalar site-property queries no longer construct an
unused size-error string.

The complexity review retained the production changes and regression cases. Git
replaces the duplicate patch file; unused benchmark setup was removed.

Scope follows
[#2253](https://github.com/munich-quantum-toolkit/core/issues/2253): concrete
copies, allocations, scaling problems, and observable nondeterminism in the QDMI
component's MLIR-facing paths. This is not a sign-off for every MLIR pass or the
unmerged QDMI 1.4 stack.

## Applied findings

### 1. Release the GIL at owned native parsing and compilation boundaries

`bindings/mlir/register_mlir.cpp` now releases the GIL in `programFromInput`
after strings and paths have become owned C++ values. `compileProgram` and both
explicit-target branches of `compileProgramForTarget` also release it around the
native pipeline. `withQCOProgram` reuses `compileProgram` with the same default
QCO pipeline as before.

The boundary review checked every caller of `programFromInput`, including device
compilation, submission, and simulation preparation. Python casts, path-protocol
execution, Qiskit import, and result construction retain the GIL. Native errors
unwind through the scoped guard before Python exception translation.
`takeResult` only constructs C++ exceptions. Target environments own their
target snapshot; native input is copied or consumed under the same ownership
rules as before. Typed input cloning remains under the GIL. The final simulation
consumer runs after reacquisition, preserving NumPy/DD ownership.

With approximately 10,000 RX/CX/RZ gates on two qubits, the median maximum
heartbeat gap across three runs changed as follows:

| Input and target                   |     Before | Applied change |
| ---------------------------------- | ---------: | -------------: |
| Source string, explicit target     | 547.912 ms |       1.061 ms |
| Typed QCO input, explicit target   | 378.974 ms |      12.992 ms |
| Source string, open DDSIM device   |  48.822 ms |       1.149 ms |
| Typed QCO input, open DDSIM device |  13.623 ms |      14.039 ms |

The last case retains its cloning pause and has overlapping sample ranges.
Source/explicit total compilation took 547.163 ms before and 482.963 ms after.
Responsiveness is the demonstrated benefit; this small sample does not justify a
general compiler throughput claim.

`test_native_compilation_releases_gil` retains five regression cases: targetless
compilation, both explicit-target result forms, source parsing, and path
parsing. A background thread must make progress during the native call, with
ordinary Python interpreter switching disabled for the check. The parsing cases
fail before the compilation scope so a compiler-only fix cannot satisfy them.
All five cases failed at their intended assertion on the baseline and pass with
the fix. Existing simulation and Qiskit-import tests also pass.

### 2. Skip unused size diagnostics for scalar site properties

`Site::queryProperty` in `include/mqt-core/qdmi/Client.hpp` constructs the size
diagnostic only for strings. Scalar queries pass the existing value diagnostic
to the shared decoder, whose scalar branch does not read the size diagnostic.

The review traced required and optional scalar results, unsupported properties,
and both string size/data calls. Error text, query counts, and string validation
remain unchanged. Device and operation queries already reuse their diagnostic;
the session helper needs its size message for its array protocol. No new query
helper or provider cache is needed.

| Native workload                       | Before median | Applied median | Allocations before → after |
| ------------------------------------- | ------------: | -------------: | -------------------------: |
| 100,000 DDSIM index queries           |      1.376 ms |       0.644 ms |                100,000 → 0 |
| 100,000 static-provider index queries |      1.346 ms |       0.604 ms |                100,000 → 0 |
| Full DDSIM compiler-target snapshot   |     16.525 ms |      12.417 ms |          262,613 → 131,543 |
| Full 100-site static snapshot         |      0.156 ms |       0.150 ms |              3,271 → 3,071 |

Each timing is the median of five runs. Counts are specific to the measured
libstdc++ build. The DDSIM snapshot removes 131,070 allocations and 4,063,170
cumulative allocated bytes while retaining 327,675 site-property calls and 487
operation-property calls. This is not a peak-memory measurement. Existing client
tests retain the values, diagnostics, and optional-property checks; the
allocation probe provides the runnable performance check without adding a
production allocator hook.

## Determinism and retained contracts

The final inspection found no new output-order dependency in these paths:

- The adapter's `DenseMap<QDMI_Site, SiteId>` serves lookup; provider-declared
  numeric IDs and site order determine observable mappings.
- Static-provider lookup tuples sort by site ID, while query results retain
  configured tuple order. Operand order within each tuple remains significant.
- Registry enumeration uses the registration-order vector, not opened-device or
  library-handle maps. DDSIM histogram output uses ordered keys.
- Five fresh Python processes, with distinct hash seeds and allocation padding,
  reproduced the baseline hashes of OpenQASM 3, Base QIR bitcode, and Adaptive
  QIR bitcode. This sample does not prove determinism for all compiler inputs.

The merged device-directed compiler API from PR #2495 and prior performance
fixes from PR #2481 remain covered. Dense-result caching and allocation-free
size checks, PennyLane sample decoding/name lookup, and Qiskit duration
snapshots are unchanged and their tests pass. Source submission still takes one
target snapshot; artifact submission checks a fresh destination contract and
skips ordinary calibration reads. Target copies share immutable storage and
topology distances remain lazy.

## Deferred candidates and exclusions

- `snapshotOperations` passes an owned operation vector to the view-taking
  `NativeOperations::fromOperations` factory, which copies tuple storage. The
  unchanged 100,000-tuple control costs about 1.5 ms and 100,001 allocations;
  1,000 tuples cost about 0.016 ms. These measurements do not justify a new
  ownership API before release. Borrowed inputs still require an owning copy.
- Operation/capability permutation checks can be quadratic, but same-order
  inputs use a linear fast path and inspected vocabularies are small. No
  critical workload justifies speculative indexes. Tuple reordering already uses
  sorted views.
- Full DDSIM snapshots still inspect 65,535 sites. Their remaining linear work
  preserves the contract. Skipping fresh compatibility checks or caching by
  device ID alone would weaken it.
- Registry indexing, IQM timeout changes, and lazy child initialization remain
  excluded by earlier decisions. General routing/DD/QIR algorithm audits and the
  unmerged PR #2226/#2227/#2233/#2373 stack remain separate work.

## Validation

Environment: ARM64 DGX Spark, GCC 13.3/libstdc++, Release with IPO, LLVM/MLIR
23.1.0, GIL-enabled CPython 3.14.7. Publication checks below were rerun after
the rebase. Timing samples above remain tied to the measurement baseline.

- Release build: 212 compiler, 241 client, and 72 DDSIM tests pass (525 native).
- Final Nox `tests-3.14`: 833 tests pass across `test_mlir.py`,
  `test_qco_dd.py`, `test_mlir_qiskit_translation.py`, QDMI compilation/client
  tests, and the three focused Qiskit/PennyLane frontend files, including the
  five new GIL cases.
- `uvx nox -s stubs` passes after the final binding change; generated stubs have
  no diff.
- Full `uvx nox -s lint` and `uvx nox -s cpp-lint -- 9d6526f48` pass. The latter
  checks the whole changed binding file against the publication base.
- The repository's C++ linter excludes ordinary public headers. A supplemental
  whole-header clang-tidy check for `Client.hpp` reports five existing naming
  and implicit-conversion warnings. The baseline header, supplied through a VFS
  overlay, produces the same diagnostics; no new warning was introduced.
- Native probes, Python probes, five-process output comparison, and explicit
  experiment syntax/format checks pass. The generated plot was inspected.

Reproduction commands, exact source revisions, raw samples, spread, neutral
results, and before/after plots are in the linked benchmark record. Detailed
test, baseline-failure, and lint logs are in `/tmp/qdmi-release-evidence`.

No cloud execution, hardware jobs, Windows, ThreadSanitizer, free-threaded
Python, or hosted CI validation was performed. Concurrent mutation of one
Python-owned program is not a new supported contract.
