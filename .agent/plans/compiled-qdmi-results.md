# Native QDMI payloads and logical results

Status: complete for Core. Provider transport changes and pinned consumer checks
live in the companion PRs.

## Contract

Compilation is explicit and precedes submission. Core owns native format
serialization and an optional terminal result mapping. Provider libraries own
service syntax and raw column decoding. Qiskit and PennyLane restore their
frontend output order without recompiling during submission or recovery.

Core exposes no provider-named profiles. `terminal_readout=None` retains the
format's ordinary semantics. `MEASURED` canonicalizes selected terminal
measurements; `ALL` also measures gaps in the addressed range. OpenQASM's
`physical_qubits` selects physical references or virtual indices. Provider
helpers select these controls and, where necessary, an explicit program format.

Terminal normalization supports direct measurement stores, permutations,
subsets, and untouched zero holes. It coalesces repeated terminal measurements
and keeps the final write to each classical slot. It rejects feedback, gates on
already measured qubits, and classical computation. This restriction does not
apply to the default dynamic path. A result map contains only raw indices or
zero holes.

IQM targets retain physical site IDs and calibrated-locus filtering. Core's
Qiskit backend compilation preserves the full TranspileLayout on sparse sites.
Braket QPUs submit native gates verbatim, while SV1 retains virtual contiguous
indices. Braket's device accepts the documented flat native subset or existing
verbatim source. IBM preserves its QPY-first path and adapts bare native QASM at
the C++ transport boundary using its existing tokenizer.

Operation CUSTOM2 retains the temporary `mqt.operation-parameters.v1` contract.
One C++ parser serves compiler and Qiskit targets; Qiskit's own Target validator
checks native placement and parameter variants. Test doubles expose real custom
property bytes; the Qiskit conftest shim is removed.

## Evidence-led changes

Core's standard OpenQASM declarations belong in the exporter. Native aliases
remain receiver-defined; standard names must retain their standard semantics.
Braket owns removal of its unsupported standard include at the wire boundary. Do
not generate or strip arbitrary native gate definitions: opaque calibrated
operations have no portable definition, and Braket rejects custom definitions.
IBM's narrow declaration repair also supports direct bare QDMI submissions.

Qiskit's Layout constructor accepts both dictionary orientations. Constructing
integer-to-qubit dictionaries avoids its failed first type check for each entry,
without changing the full layout or observable mapping. During Qiskit target
import, a global operation fills the active-site set only when its size differs
from the address width. Valid sites then already cover the range; invalid sites
remain subject to the unchanged final range check.

Paired public-entry measurements on Linux ARM64, LLVM 23, and Qiskit 2.6.1 used
Bernstein-Vazirani (8 bits), QAOA (12 qubits, depth 4), and partial readout. All
retain DDSIM's full 65,535-site Qiskit layout. Six steady samples per revision
reduced median end-to-end time from 220/265/220 ms to 142/186/141 ms. The
compile phase fell from 153–156 ms to 74–76 ms; submission and results stayed
similar. This shared-host measurement does not predict cloud latency. Allocation
traffic, peak live allocations, and process RSS were measured separately;
increasing RSS before cyclic garbage collection is not evidence of a leak.

Reject implicit target caching: explicit snapshot reuse saves only a few
milliseconds during serialization, while destination validation remains
necessary. Preserve the public snapshot lifetime and complete result layouts.

## Validation

Use asymmetric deterministic circuits and inspect complete samples as well as
counts. Cover measured subsets, zero holes, sparse site IDs, routing layouts,
parameter restrictions, and default feedback/reset/reused-bit execution. Reject
unsupported terminal forms before any network submission. Verify actual provider
payload consumers; Braket LocalSimulator is not a cloud syntax oracle.

Previous live evidence covers SV1, Rigetti, and AQT (16 shots each), plus an IQM
mock. Those runs predate this simplification and are not fresh live validation.
No IBM hardware run has been performed. Fresh local validation passes 770 native
compiler/OpenQASM tests and 1,676 installed-package Python tests. Strict-header
and native-name regressions fail on the previous exporter and pass with the fix.
Generated stubs, whole-file C++ lint, and strict executable documentation pass;
the changed guide was inspected in Chrome. Provider consumers must run their
offline tests against the final shared revision. No new live jobs are required
for these changes.
