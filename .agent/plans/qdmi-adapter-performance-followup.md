# QDMI native readout and ISA import efficiency

Scope: adapter items 5, 7, and 6 in PR #2987. Preserve provider interfaces,
result validation, ordered samples, seeded streams, and batch recovery. No new
public configuration or adaptive readout policy is needed.

## Ownership

The compiler produces `ResultLayout`, which maps provider bits to logical
outputs. Shared decoders validate binary data, widths, layouts, and shot totals.
Frontend-specific measurement and output formatting belong at runtime.

### PennyLane and Catalyst

`mlir/lib/Runtime/QDMIReadout` owns the concrete weighted readout shared by
PennyLane and Catalyst. It prefers histograms for aggregates and SHOTS for
ordered samples. Retrieval failure or an empty response permits fallback;
malformed decoded results fail without fallback. The QDMI client retains its
last-raw-duplicate-wins parser contract. Logical projection merges counts.

The PennyLane binding retrieves once for each tape/shot partition and computes
requested wire marginals natively. It returns probability arrays for
probability, expectation, and variance measurements, and binary dictionaries for
count measurements. Integer frequencies are accumulated before normalization.
PennyLane retains eigenvalue processing, scalar coefficients, all-outcomes
handling, and return shapes. Expectation eigenvalue types are preserved;
variance follows PennyLane's float64 conversion. Exact measurement-class
dispatch keeps custom processors and mixed sample tapes on the ordered sample
path.

`Batch` caches completed per-tape measurement results. `entries` and `collect()`
retain a scalar, array, counts dictionary, or measurement tuple for each
preprocessed tape and shot partition. `result()` assembles those cached values;
it does not repeat measurement processing. Locally completed tapes use the same
sample measurement conversion before their results are retained. QNode transform
postprocessing remains outside the batch, as before.

Catalyst shares the readout loader and logical bit indexing. Its small output
loops retain support for strided `DataView` buffers without a temporary dense
array or an indirect callback for every outcome. Its retained job still supports
aggregate-to-ordered-sample reuse without another execution.

### Qiskit results

The private `_qiskit.readout` binding retrieves, validates, and formats final
hexadecimal counts and optional memory. It supports arbitrary classical widths,
including projected outputs above 64 bits. Native encoding retains first-shot
order and shares one Python hexadecimal string per distinct memory outcome. A
private exception distinguishes decoding failures from provider retrieval
errors. Qiskit's result models, register metadata, and sampler interface stay
unchanged.

Memory always requires SHOTS; it does not inherit PennyLane/Catalyst histogram
fallback. Counts-only results use native histogram decoding and avoid a
Python/native dictionary roundtrip. Zero-width counts-only results remain empty
without querying provider results.

### Full-width ISA circuits

The Qiskit importer collects physical references during its existing root
validation traversal. Ordinary root qargs and root control-flow operand maps
cover nested users; nested local and gate-formal indices are not collected as
physical sites. Only referenced static qubits are created, in the entry block,
without renumbering. There is no separate operand-discovery pass, import option,
or added width metadata. Existing exporters and target-aware QPY behavior remain
unchanged.

## Measurements

Main baseline: `58a6541655c74eeaeeac64df94d9cd7d13c74af9`. A second comparison
uses the initially published PR head `8b28f8ce480ffeff9ab91e68a2a8074fcbc5cd2a`.
Each baseline uses its saved native extension and matching Python job code.

Measurements use Python 3.14.10, PennyLane 0.45.1, Qiskit 2.6.2, Clang 23.1.6,
LLVM/MLIR 23.1.0, Release, ThinLTO, and mold on ARM64. Five alternating pairs
run on CPUs 16–19 of a shared host. These are workload diagnostics, not isolated
or portable throughput claims. Harnesses and raw measurements remain outside the
repository in `/tmp/qdmi-native-redesign/`.

Readout workloads use 20 qubits, one million shots, and DDSIM seed 42. Timing
starts after the native job completes and covers `job.result()`, including
retrieval, decoding, frontend processing, and result assembly. It excludes
compilation and provider execution. Qiskit timings include hexadecimal memory
and count construction, but exclude sampler post-processing.

| Completed-job result processing          |      Main | Native readout |
| ---------------------------------------- | --------: | -------------: |
| PennyLane GHZ, one Z expectation         |  86.52 ms |        0.18 ms |
| PennyLane uniform, one Z expectation     | 332.07 ms |      106.31 ms |
| PennyLane GHZ, twenty Z expectations     | 103.30 ms |        0.39 ms |
| PennyLane uniform, twenty Z expectations | 345.59 ms |      155.07 ms |
| Qiskit GHZ memory and counts             | 215.03 ms |       75.77 ms |
| Qiskit uniform memory and counts         | 928.28 ms |      506.80 ms |

Uniform retained parent RSS after result collection fell from 378.7 to 328.0 MiB
for PennyLane and 390.8 to 352.9 MiB for Qiskit. These are `/proc` snapshots,
not peak RSS or worker memory. All paired ordered-memory fingerprints match;
PennyLane expectation results agree within `1e-15`.

Against the initially published PR, uniform PennyLane readout fell from 305.74
to 106.86 ms for one observable and 2446.32 to 157.96 ms for twenty. Qiskit
uniform memory/count construction fell from 853.36 to 509.65 ms. These separate
paired runs confirm that moving reduction and final formatting across the Python
boundary removes work beyond the earlier container optimizations.

Five paired serialization measurements against main:

| Serialization workload         |     Main | Referenced-site import |
| ------------------------------ | -------: | ---------------------: |
| 65,535-qubit ISA Bell circuit  | 55.11 ms |               15.77 ms |
| Two-qubit, 10,000-gate circuit | 72.71 ms |               72.69 ms |

Payload hashes match. Folding reference collection into validation leaves the
published PR's timing approximately unchanged while removing an extra traversal.
Target construction and Qiskit's own transpilation costs remain outside these
serialization timings.

## Validation and limits

- Release build and matching editable package; generated Python stubs.
- All 61 compiler-adapter and Catalyst native tests passed, covering projection,
  duplicates, malformed readout, fallback, and reuse of one job for counts and
  samples.
- All 1,784 Python plugin, QDMI, compiler, translation, and target tests passed,
  covering measurement semantics, shot partitions, recovery, and nested-only
  physical sites.
- The existing aggregate regression includes complex expectation eigenvalues; an
  injected float64 conversion makes it fail.
- One added public-backend regression checks counts and memory with an 81-bit
  classical output; an injected 64-bit truncation makes both cases fail.
- Independent native result checks matched Python hexadecimal conversion for 34
  sparse-output cases through 129 bits.
- Generated stubs, whole-file C++ lint, and repository lint passed.

Windows, hardware providers, free-threaded Python, and rendered documentation
previews are outside this local validation. Two optional Catalyst Python tests
require an unavailable dependency. Dense probabilities and eigenvalue arrays
still scale exponentially with the requested measurement width.

Lazy DDSIM state reconstruction, shared-deadline shutdown, and hinted client
histogram insertion already landed in #2970. They are inherited benefits. Keep
eager catalog opening and idle-worker reuse. Compiler parallelism, import-once
sampling, provider protocols, and public QDMI container changes remain outside
this work.
