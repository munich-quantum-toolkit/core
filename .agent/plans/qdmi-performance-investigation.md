# QDMI performance and efficiency investigation

Status: implemented and validated against `b6d06d45fc`. Three local changes
remain: lazy state reconstruction, shared-deadline shutdown, and hinted
histogram insertion. Shared readout on main supplies item 3. Catalog opening and
idle-worker retention remain unchanged.

## Scope and baseline

Integration baseline: `b6d06d45fc`. Paired measurement baseline: `92b9fef855`.
Historical measurement baseline: `328b426c17`. Preserve the QDMI Client
interface, result formats, and DDSIM sampling and state-result availability. The
approved failure-timing change is limited to invalid retained states: valid
samples remain available from a completed program, and state/probability queries
return `QDMI_ERROR_FATAL`.

The investigation covers DDSIM result ownership and worker shutdown, the shared
Catalyst/PennyLane readout, C++ sparse-result construction, catalog opening, and
idle-worker retention. It excludes external providers, simulation/JIT
algorithms, MLIR passes, new public parameters, and container return-type
changes.

Earlier decisions remain in force:

- [Driver performance audit](../audits/qdmi-driver-performance-2026-09-08.md):
  no registry index, job pool, metadata cache, or extra loader cache.
- [Result performance audit](../audits/qdmi-main-performance-2026-09-08.md):
  retain dense-result bounds and allocation-free size queries where applicable.
- [Pre-release audit](../audits/qdmi-pre-release-performance-2026-09-10.md):
  full target snapshots still enumerate sites; session width bounds this work.
- [Python concurrency audit](../audits/qdmi-python-gil.md): retain eager child
  initialization and eager client-catalog opening.

## Current integration contracts

- Shared `decodeShots` and `decodeCounts` own result validation and layout
  projection across Qiskit, Catalyst, and the runner (#2954). Catalyst
  aggregates prefer counts and reuse the same job if ordered samples are later
  requested (#2953). Samples stay MSB-first until consumption, and expansion
  writes into a checked, pre-sized array (#2960). Item 3 needs no local frontend
  loop.
- DDSIM's requested logical shot streams are independent of granted execution
  threads for seeded jobs (#2958). QCO automatic worker selection belongs to the
  sampler (#2965). These contracts remain unchanged by lazy result ownership and
  idle-process shutdown.
- Worker response serialization and the parent state-reconstruction path still
  carry the same retained-state contract. The worker pool and the C++ sparse
  result parser are unchanged upstream, so items 1, 2, and 4 remain applicable.
- PennyLane scopes implicit DDSIM sessions to declared wires (#2960). Full
  65,535-site snapshots remain relevant to unscoped or explicitly full-width
  sessions, rather than the normal fixed-wire PennyLane path. Catalog opening
  itself remains eager and sequential.
- Native Qiskit submission (#2955), target-aware exports (#2957), tensor
  cancellation (#2966), and reduced QIR export preparation (#2967) affect
  surrounding work. Historical end-to-end timings and seeded fingerprints must
  not be treated as incremental results against this integration baseline.
  Parallel compilation and import-once sampling remain separate follow-ups in
  [the adapter plan](qdmi-adapter-performance.md), outside this task.

## Current-main comparison

A focused paired check uses the current worker executable for both variants,
with the device library built from `92b9fef855` or the local implementation.
Five alternating pairs per workload use 100 OpenQASM 3 programs, 12 qubits,
1,000 shots per program, and an eight-worker budget. Circuit construction
matches the historical method below. All runs created eight workers and left no
child processes after finalization.

| Workload | Main parent RSS | Local parent RSS | Main finalization | Local finalization |
| -------- | --------------: | ---------------: | ----------------: | -----------------: |
| GHZ      |        67.8 MiB |         15.6 MiB |           80.6 ms |            10.2 ms |
| Random   |       124.6 MiB |         48.6 MiB |           80.6 ms |            10.2 ms |

Values are medians; RSS is sampled while the completed job owns its results.
This confirms that the memory reduction and shared shutdown still apply after
integration. It does not establish an isolated end-to-end throughput gain. Raw
measurements are in `/tmp/qdmi-performance/rebase-92b9fef855/paired.jsonl`.

## Historical measurement method

The measurements below compare `328b426c17` with the original implementation;
they explain the retained design choices, not current-main speedups. Native
measurements use the DGX Spark's ARM64 CPU, Clang 23.1.6, LLVM/MLIR 23.1.0,
Release, ThinLTO, and mold. The host is shared. Wall-clock runs overlap other
work, so they establish workload behavior and memory use, not an isolated
end-to-end speedup. Report medians of five runs unless explicitly marked as a
single pilot. Harnesses and raw measurements remain outside the repository.

The maintainer approved a bounded matrix: 100-program GHZ batches at 12, 16, 20,
and 22 qubits; random 12-qubit batches of 100 and 16-qubit batches of eight;
OpenQASM 3 and Base QIR; 1,000 shots per program. Random circuits use seed 42,
eight layers of per-qubit RY/RZ rotations and shuffled CX pairs. A single
100-program random 16-qubit pilot supplements the repeated smaller rows. Full
100-program random 20/22-qubit rows are excluded from this bounded run.

## 1. Lazy retained-state reconstruction

`ProgramResult` owns the serialized state and its declared width. All state and
probability queries pass through one `std::call_once` before accessing the
package, state edge, or cached status. The initializer builds a local package,
validates the root width, retains its root, and publishes it. Explicit failures
are cached, their LLVM errors consumed, and unusable serialized bytes released.
An allocation exception leaves the bytes available for a later retry.

Sampling-only queries never construct a result package. State queries retain the
same dense/sparse values and availability. The approved deferred-error behavior
is documented in [the DDSIM guide](../../docs/qdmi/ddsim_device.md).

Five-run parent RSS medians, while the completed job still owns its results:

| OpenQASM workload              |  Baseline | Lazy state |
| ------------------------------ | --------: | ---------: |
| 100 GHZ programs, 12 qubits    |  67.1 MiB |   15.5 MiB |
| 100 GHZ programs, 22 qubits    |  67.9 MiB |   15.7 MiB |
| 100 random programs, 12 qubits | 124.0 MiB |   48.8 MiB |
| 8 random programs, 16 qubits   | 144.2 MiB |   72.4 MiB |

The single 100-program random 16-qubit pilot fell from 1,035.6 to 420.2 MiB. The
five-run Base QIR random 12-qubit batches fell from 126.2 to 50.6 MiB. An
instrumented baseline compared normal workers with a measurement-only prototype
that omits state serialization. For 100 random 12-qubit programs, the retained
parent RSS increase fell from 109.15 to 7.00 MiB, independently meeting the
100-MiB action threshold. Each state was 237,534 bytes; summed worker
serialization CPU time was 163.65 ms per batch. GHZ states were 1,358 bytes
each, with 9.54 ms summed serialization CPU. These rows do not justify a new
skip-state parameter. The measured benefit is memory reduction; no isolated
end-to-end latency gain is claimed. A separate five-run random 12-qubit batch
comparison recorded Linux `ru_maxrss` medians of 123.3 MiB before and 47.6 MiB
after. These high-water diagnostics use the kernel's peak accounting; the table
above uses retained `/proc` RSS snapshots. Transport read/decode time was not
isolated from deserialization, so no separate transport-speed claim is made.

Single-program random 20/22-qubit pilots hit a 120-second bound on the loaded
host and were canceled through the public API. They provide no latency result.

Retain the decoder's separate state-byte copy. Moving the entire response frame
would also retain its capacity for output records and histogram keys,
potentially keeping a large frame for a tiny state. The lazy-package change does
not require that ownership tradeoff or a new worker protocol representation.

External instrumentation verifies zero deserializations for counts-only use and
one per result for eight cold concurrent-read cases, including terminal
sampling. An adversarial worker returns valid counts with corrupt state bytes:
the job completes, counts remain readable, and repeated state queries return a
cached fatal error. The native concurrent-read regression covers both extraction
and terminal sampling.

## 2. Close idle workers before waiting

`WorkerPool::shutdown` requests closure from every idle worker, then waits using
one shared one-second deadline. Each wait first checks for exit; an expired
deadline does not introduce another sleep. Remaining processes are terminated
and reaped. Active-worker termination and the `accepting_` contract are
unchanged. The single-worker shutdown path uses the same helpers with its own
deadline.

Five-run interleaved finalization medians were 15/19 ms before/after with one
worker, 85/14 ms with eight, and 220/18 ms with twenty. Every run observed the
requested number of idle workers and no children after finalization. The Linux
regression holds programs at a file barrier to ensure multiple workers start,
reads child PIDs as streams from procfs, finalizes through the public C API, and
checks that those processes no longer exist. It imposes no wall-clock
performance assertion and respects the configured worker budget.

## 3. Use the shared readout on main

`mlir::decodeShots` validates and decodes each distinct raw outcome once,
keeping shot order and correlations. `mlir::decodeCounts` validates totals and
merges outcomes projected onto the same logical bits. The runtime consumes these
results directly, without a second reversal or normalization pass.

Adjacent compaction in the frontend would run after the shared decoder and no
longer avoid the original repeated decoding work. No additional cache, grouped
representation, or shared-decoder rewrite is justified by the historical
measurements. Keep the shared decoder and the runtime counts-first policy.

The compiler adapter's ordered-shot/layout tests and Catalyst's
`AggregateReadoutKeepsCountsAndSamplesFromOneJob` test cover layout projection,
malformed input, nonadjacent repeated outcomes, and counts-to-samples reuse.

## 4. Hint sparse-result insertion

Use `std::map::insert_or_assign(end(), key, value)`. Sorted unique DDSIM keys
use the end hint; the standard library handles unsorted input and duplicate
keys. Duplicate keys keep their last value, matching the existing contract.
There is no sorting scan, extra container, or public return-type change.

Five-run parent-CPU medians for C++ `Job::getCounts`, after one warm-up, using
uniform-superposition circuits and one million shots:

| Qubits | Outcomes |   Baseline | Hinted insertion |
| ------ | -------: | ---------: | ---------------: |
| 16     |   65,536 |  12.394 ms |         7.279 ms |
| 20     |  644,497 | 143.925 ms |        75.734 ms |
| 24     |  970,693 | 166.447 ms |       111.997 ms |

Provider-copy timings were 1.5–23 ms; parsing and map construction dominate
these native rows. Public Python `get_counts()` CPU medians, measured separately
on CPU 19 with the same driver for both variants, fell from 16.17 to 13.17 ms at
16 qubits, 262.24 to 237.77 ms at 20, and 449.25 to 376.90 ms at 24. Full result
fingerprints and million-shot totals agree. The existing indexed-histogram
driver fixture checks unsorted and duplicate keys through `Job::getCounts`.

## 5. Keep eager sequential catalog opening

With one fixture library adding 200 ms to each session initialization, first
catalog access took median 0.211, 0.814, and 1.623 seconds for 1, 4, and 8
registrations. Concurrent opening would still serialize these registrations on
the library's module mutex. The maintainer retained eager sequential opening; no
targeted-session routing or concurrency is added.

## 6. Keep idle-worker reuse

After GHZ programs at 4, 16, and 22 qubits, each idle worker retained about 46.7
MiB. Five-run cold execution medians were 13.6–15.8 ms, versus 4.5–7.3 ms for a
warm worker. A wide pool therefore retains substantial aggregate memory but
avoids repeated startup. The maintainer retained the current policy: no idle
cap, timer, or configuration parameter is added.

## Publication review

Lazy reconstruction belongs to the result owner: samples avoid creating a DD
package, while state consumers pay reconstruction once. Worker serialization
remains necessary for later state queries. Root registration can allocate, so
serialized bytes remain owned until registration and publication succeed. Pool
shutdown owns the common deadline; standalone worker cleanup retains its bounded
wait. Histogram insertion retains arbitrary input ordering and
last-duplicate-wins behavior through the standard container fallback. No new
public API, protocol variant, or dependency is needed.

## Validation

Current integration validation passed:

- Clang 23 release build with ThinLTO and matching Python 3.14 editable install.
- 98 DDSIM, 316 C++ client, 8 Catalyst runtime, and 432 compiler tests,
  including shared readout decoding, plus 8 QDMI configuration checks.
- 3 focused QCO sampling tests covering seeded worker grants, disabled contexts,
  and adaptive workers.
- 2,046 Python QDMI, plugin, MLIR, and QCO tests; 2 optional Catalyst tests
  skipped.
- Whole-file C++ lint and repository lint.
- The paired retained-memory and worker-cleanup comparison above (at
  `92b9fef855`).
- A publication-head fault-injection probe throws `std::bad_alloc` immediately
  before root registration. The first query returns `QDMI_ERROR_OUTOFMEM`; a
  retry returns the correct GHZ amplitudes. Moving byte release before root
  registration makes the same probe fail with `QDMI_ERROR_FATAL`.

The historical implementation passed its native and Python suites and both
lints. External probes established zero counts-only deserializations, one decode
per result under concurrent state queries, cached corruption errors, and a
cleanup regression that detects omitted finalization. These are historical
checks until repeated on the integration baseline.

No public binding signature changes are required. Windows, hardware providers,
free-threaded Python, and the full high-width random matrix remain outside this
validation run. Shared-host timing does not establish an isolated end-to-end
speedup.
