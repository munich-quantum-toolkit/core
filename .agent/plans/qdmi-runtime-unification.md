# QDMI runtime unification

Status: implemented. Shared execution ownership and pure result decoding live in
the runtime. Broad paired measurements support indexed native readout; the
public expanded-string decoder retains its existing implementation.

## Contracts and ownership

Baseline: `07f2a288ff` (merged PR #2987). The runtime owns provider sessions,
retained target environments, submission configuration, and logical result
processing. Catalyst retains its registry/factory lifetime adapter. Qiskit keeps
per-run parameters, metadata, strict SHOTS memory, and alternate-format
preflight. Python Batch remains responsible for attempts, recovery, and cached
frontend results. Every program must be checked before the first submission.

Pure result decoding shares the compiler's ResultLayout contract without linking
the compiler transformation pipeline. Existing public decoding signatures stay
available. Counts fallback remains distinct from malformed-result errors.

The ordered readout owns distinct logical outcomes with frequencies and an index
for every shot. Normalization and projection merge equal outcomes; indices
preserve order and correlations. Native consumers avoid per-shot logical string
copies. Histogram results retain frequencies without claiming shot order. No
adaptive policy, worker pool, protocol change, or new dependency is introduced.

## Evidence and completion

- Build an exact baseline with the same compiler, release settings, and Python
  dependencies as the candidate; compare matching native and Python adapters.
- Use MQT-generated GHZ variants, W, Bernstein-Vazirani, QFT variants, QPE, and
  QAOA across widths 4, 8, 12, 16, and 20 where practical, with 1,000, 100,000,
  and selected one-million-shot cases. Preserve seeds and input fingerprints.
- Measure completed-job result processing for Qiskit memory/counts and PennyLane
  samples/aggregates. Check the legacy expanded-string decoder too. Report
  wall/CPU time and retained RSS separately from full-execution probes.
- Alternate baseline/candidate runs, compare every output fingerprint, and
  report family/size regressions rather than only pooled means. Run performance
  measurements without concurrent local builds or tests; shared-host results
  remain diagnostic.
- Reuse existing layout, width, malformed-input, native multi-program admission,
  registry lifetime, and recovery tests. Regenerate stubs and run release native
  tests, Python integration tests, whole-file C++ lint, and repository lint.

## Implementation and validation

The shared native `Context` owns execution state. Catalyst adds only its
registry identity and capacity; both Python integrations use the shared binding.
Per-run Qiskit parameters borrow storage only during submission, while retained
PennyLane parameters own their bytes.

`MQTResultReadout` owns pure layout decoding and links only LLVM Support.
`MQTQDMIExecution` owns compiler-aware submission, and `MQTQDMIRuntime` owns
provider readout. Qiskit formats distinct logical outcomes as hexadecimal in its
own result type; runtime outcomes remain binary.

Completed checks: 61 native compiler-adapter/Catalyst tests; 1,782 Python tests
with two optional Catalyst skips; 344 plugin tests rerun after final native
cleanup; generated stubs; whole-file C++ lint; repository lint. Existing tests
cover layout projection, malformed input, parameter ownership, preflight,
registry lifetime, and recovery. No new test scaffolding was added.

## Broad paired measurements

The matrix completed 218 configurations with three alternating pairs each (1,308
fresh jobs), covering eight MQT benchmark families: GHZ, W, Bernstein-Vazirani,
QFT, QPE, QAOA, Grover, and QFT adder. GHZ and QFT each include two
output-distribution variants. Widths span 4–20 qubits and shot counts span
1,000–1,000,000. QPE stops at 16 qubits and Grover at 12. Every input and output
fingerprint matched between revisions.

Two of the planned 220 configurations—PennyLane samples and aggregates for the
20-qubit QFT adder—are excluded after baseline execution hit the 120-second
bound before readout. Worker cleanup was verified, and all other measurements
resumed without discarding completed pairs.

Both revisions use Clang 23.1.6, Release/ThinLTO, mold, Python 3.14, seed 42,
and one DDSIM shot worker on the DGX Spark ARM64 host. Runs alternate on CPUs
16–19 without concurrent local builds/tests. Processes reset by family/width;
each measurement uses a fresh public job after a 100-shot warm-up. Readout
starts after native completion. Total time includes submission/compilation,
waiting, and frontend results, excluding device and circuit construction.
Retained RSS includes allocator history; process peaks are cumulative within
each group and cannot establish allocation volume or worker memory.

Completed-job readout improved in 122 of 125 ordered configurations. The median
candidate/baseline ratios were 0.723 for Qiskit memory and 0.683 for PennyLane
samples. Counts and aggregate controls had median ratios of 1.004–1.005. The
three ordered median regressions were at most 0.213 ms, with paired ratios
crossing one. These shared-host measurements are diagnostic, not isolated
throughput claims.

Five fresh-process pairs resolved the apparent GHZ-X-20 counts regression:
Qiskit counts measured 21.960 → 21.999 ms, and PennyLane aggregates measured
16.557 → 16.049 ms. All 14 separate wide-output probes through 129 bits matched.

The first implementation routed the public expanded-string decoder through
indexed readout. Focused probes confirmed a small concentrated-outcome cost
(BV-8: 4.169 → 4.597 ms; million-shot GHZ-16: 70.153 → 73.286 ms). A direct LLVM
string-cache prototype improved uniform results but retained a small BV slowdown
(4.382 → 4.710 ms) and extra retained memory. Neither legacy change is needed
for native readout, so the final expanded decoder keeps main's exact
implementation. Both paths share sample validation and layout projection; native
consumers alone use indexed readout. There is no adaptive policy.

Representative completed-job medians from the broad matrix (milliseconds):

| Workload                   |     Shots | Qiskit memory, main → candidate | PennyLane samples, main → candidate |
| -------------------------- | --------: | ------------------------------: | ----------------------------------: |
| GHZ, 16 qubits             | 1,000,000 |                   62.76 → 49.65 |                       66.92 → 57.03 |
| W, 16 qubits               | 1,000,000 |                   66.53 → 49.77 |                       67.55 → 57.37 |
| Half-period QFT, 20 qubits |   100,000 |                     9.88 → 6.79 |                         8.98 → 6.80 |
| Uniform QFT, 20 qubits     |   100,000 |                   48.66 → 39.82 |                       32.61 → 28.22 |
| Uniform QFT, 16 qubits     | 1,000,000 |                  150.94 → 97.87 |                      132.23 → 84.04 |
| QAOA, 16 qubits            | 1,000,000 |                  128.61 → 65.66 |                      119.74 → 70.43 |
| Grover, 12 qubits          |   100,000 |                     7.53 → 3.60 |                         7.17 → 3.34 |

These gains concern completed-job result processing. Sums of per-cell median
total times are essentially unchanged because simulation dominates the larger
cases. No overall throughput or peak-memory reduction is claimed.

Raw rows, harnesses, build logs, exact extension hashes, and the full report are
under `/tmp/qdmi-runtime-unification/`. The broad matrix measures the initial
indexed implementation; subsequent focused measurements check the final build.
Native ordered decoding is unchanged across these builds. Final-build focused
checks and fingerprint are recorded in the external report.

The saved native extension SHA-256 identifiers are:

- Main: `1df0b161ea2177228a5ccc80fe17562fb60eab03d030c57d281843ed007489d0`.
- Broad matrix:
  `60d7e636635a2b6f05c22677d7019e568ffb9085983c3ec71bb305d322b97f80`.
- Final: `96e9eacf0016f4f748be0181f5645be269b23fc19199375cc0d10e948b06f208`.

No Windows, hardware-provider, or free-threaded Python checks were run. Python
Catalyst execution remains unavailable; the native Catalyst runtime suite
passes.
