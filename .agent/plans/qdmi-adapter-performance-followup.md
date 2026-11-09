# QDMI result efficiency and adapter follow-up

Status: implemented locally on `58a6541655c74eeaeeac64df94d9cd7d13c74af9`. The
scope is adapter items 5, 7, and 6.

## Outcome and ownership

### Aggregate-first PennyLane readout (item 5)

`PennyLaneJob` selects counts for standard expectation, variance, probability,
and count measurements when every measurement on the preprocessed tape supports
that path. Mixed sample/aggregate tapes and custom measurement subclasses keep
sample processing. Empty-wire constant observables and locally completed tapes
also retain sample arrays.

The native helper reuses Catalyst's counts-first readout and the shared result
validators. Retrieval failure or an empty result permits SHOTS fallback;
malformed decoded results fail without fallback. The helper retains the decoded
map directly and aggregates SHOTS fallback entries only when needed. The shared
decoder consumes owned map nodes, validates and projects their keys, and merges
collisions through hinted insertion. Arbitrary projection order and repeated
outcomes remain supported. The binding extracts nodes and reverses their keys in
place into PennyLane wire order while building the final Python dictionary.

Provider results for supported aggregate tapes are retained as count
dictionaries in `entries` and `collect()`. Other entries retain shot-by-wire
arrays. Ordinary QNode return shapes, shot partitions, attempts, cancellation,
recovery, and tracking remain unchanged. Unqualified counts/probabilities use
the effective tape wires; custom sample processors are not replaced by inherited
counts processors. No new readout policy option or density heuristic is added.

Concentrated distributions benefit substantially. The retained counts
representation scales with distinct outcomes, while the sample representation
scales with shots and wires. Dense probability outputs and wide-observable
processing can still allocate exponentially sized arrays. The measured uniform
single-observable case also benefits after removing redundant map allocations;
this does not guarantee gains for every distribution or measurement.

### Qiskit memory conversion (item 7)

A private MLIR/QDMI binding retrieves and decodes native shots without building
an intermediate Python list. Native work releases the GIL. Shared `decodeShots`
is unchanged, including shot ordering, correlations, validation, and layout
projection. A private decoding exception distinguishes malformed output from
provider retrieval errors so the Qiskit collector retains its error boundary and
retrieval note. Python hex encoding, counting, and sampler post-processing
remain unchanged. No sampler subclass or encoding cache is introduced.

### Full-width Qiskit ISA circuits (item 6)

The native Qiskit importer creates static qubits only for referenced physical
sites. It collects ordinary instruction operands and control-flow qubit maps in
the entry block, preserving dominance for nested-only sites and physical indices
without renumbering. Collection stops once all sites are referenced, so small
dense circuits avoid a full extra instruction scan. Quantum/classical register
validation and result layouts remain intact.

Existing exporters already eliminate unused static qubits; target-aware QPY
export derives its addressable width from the target. No public import option or
width metadata was added. The backend and DDSIM guides use scoped sessions and
explain that explicit Qiskit layouts can expand circuits to target width. The
65,535-site default and explicit `custom1` behavior are unchanged.

## Evidence

The baseline is the current main revision named above. Python 3.14.10, PennyLane
0.45.1, Qiskit 2.6.2, Clang 23.1.6, LLVM/MLIR 23.1.0, Release, ThinLTO, and mold
were used on ARM64. Five alternating pairs ran on CPUs 16–19 of a shared host.
These are workload diagnostics, not isolated or portable throughput claims.
Harnesses and raw data remain outside the repository.

PennyLane used 20-qubit GHZ and uniform-superposition circuits, one million
shots, seed 42, and a single Z expectation. The comparison selects the unchanged
sample path or the new counts path on the same native build. Timings cover
`job.result()` after submission, including any remaining execution wait. RSS is
a retained parent `/proc` snapshot after collection, not worker memory or peak
RSS. Kernel high-water readings were unsuitable for this comparison because some
processes inherited a larger pre-exec high-water mark.

| Workload            |   Samples | Previous counts | Final counts | Samples parent RSS | Final counts parent RSS |
| ------------------- | --------: | --------------: | -----------: | -----------------: | ----------------------: |
| GHZ expectation     |  91.22 ms |         5.55 ms |      5.20 ms |          227.5 MiB |               175.7 MiB |
| Uniform expectation | 662.60 ms |       817.08 ms |    629.20 ms |          385.4 MiB |               369.4 MiB |

These five alternating three-way comparisons include a saved native build of the
previous counts implementation. Uniform readout is approximately 5% faster than
samples and retains 4% less parent memory in this workload. GHZ retains its
large gain. The ownership changes remove the measured uniform regression without
a runtime threshold, alternate readout policy, or narrowed result
representation. Counts dictionaries have identical full fingerprints across the
previous and final builds; sample and count expectations agree within `1e-15`.
Raw data and harnesses are in `/tmp/qdmi-uniform-followup/paired-final.jsonl`
and its directory.

Qiskit memory comparisons alternate the original `get_shots` plus `decode_shots`
calls and the new native helper on the same completed job. All one-million-shot
output sequences match exactly. Retrieval-plus-decoding medians were 144.74 to
69.95 ms for GHZ and 398.74 to 300.44 ms for uniform outcomes. Hex
encoding/counting remained approximately 45 and 594 ms, respectively, in these
runs. These numbers exclude Qiskit's sampler post-processing and do not
establish a complete sampler speedup.

The actual full-width DDSIM backend kept a Bell circuit at two qubits with the
default preset layout and expanded it to 65,535 with `initial_layout=[0, 1]`. A
`custom1="2"` backend stayed at two qubits in both cases. Five alternating
native-extension comparisons used the saved baseline extension and candidate
with the same Python adapter code:

| Serialization workload         | Baseline | Candidate |
| ------------------------------ | -------: | --------: |
| 65,535-qubit ISA Bell circuit  | 54.66 ms |  15.52 ms |
| Two-qubit, 10,000-gate circuit | 72.43 ms |  71.39 ms |

Payload hashes matched across all pairs. The importer improvement does not
remove full-width target construction or Qiskit's own transpilation costs.

## Completed foundation and exclusions

Lazy DDSIM retained-state reconstruction, shared-deadline shutdown, and hinted
sparse-result insertion already landed in #2970. They are inherited benefits,
not gains from this follow-up. Shared readout validation and Catalyst's
counts-first policy also predate this change. The
[completed DDSIM investigation][device-plan] retains their historical evidence;
the [original adapter proposal][adapter-plan] records the initial hypotheses.

Keep eager catalog opening and child initialization, idle-worker reuse, seeded
logical shot streams, lazy state availability/error behavior, dense result
bounds, sequential submission, and preflight before submission. Add no
registry/cache framework, skip-state option, protocol change, or worker policy.
Compiler parallelism, fixed compile overhead, validation-ownership changes,
import-once sampling, and submission payload-copy work remain out of scope.

## Validation

- Release build and matching editable Python package.
- The uniform-readout follow-up passed 431 focused Python tests (two optional
  Catalyst tests skipped) and the 61 compiler-adapter/Catalyst tests. It adds no
  tests; existing cases cover projected collisions, unordered outcomes, repeated
  SHOTS fallback, malformed results, and count/sample wire semantics.
- 1,782 Python plugin, QDMI, MLIR, Qiskit translation, and target tests passed;
  two optional Catalyst tests skipped.
- All 4,386 native tests passed before the final ordered-map refinement; the 61
  affected compiler-adapter/Catalyst tests were rerun afterward and passed.
- Generated stubs, repository lint, and whole-file C++ lint passed. C++ lint
  selected all changed native files from the staged working changes, with
  changed-line filtering disabled.
- Existing native readout coverage was extended for histogram preference, layout
  order, SHOTS fallback, and malformed histograms. Existing recovery coverage
  now exercises aggregate and mixed measurements. Three focused Python tests
  cover aggregate wire/partition semantics, custom measurement dispatch, and
  nested-only physical sites in wide circuits.
- Selecting the old sample path makes the aggregate regression fail. The saved
  baseline creates 65,535 static operations for the two-site nested circuit; the
  new regression requires only its two referenced sites. An intermediate missing
  control-flow-map implementation crashed on the nested-only case; the corrected
  implementation passes that regression and existing native
  preflight/control-flow tests.
- An external native-memory probe verified exact layout projection and distinct
  retrieval versus decoding errors. Paired memory probes verified full ordered
  sample equality. Specialist review found no remaining correctness blockers.

Windows, hardware providers, free-threaded Python, hosted CI, and rendered
documentation previews were not checked during local validation. No remote
provider jobs were run. Performance outside the measured workloads remains
uncharacterized.

[adapter-plan]: https://github.com/munich-quantum-toolkit/core/blob/875c12f5a21bc2661cdbe915e6a848416ea6898a/.agent/plans/qdmi-adapter-performance-followup.md
[device-plan]: https://github.com/munich-quantum-toolkit/core/blob/727fc5ed941279046318ef3a3438208c951e28ab/.agent/plans/qdmi-performance-investigation.md
