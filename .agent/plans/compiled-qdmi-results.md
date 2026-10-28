# Native QDMI payloads and logical results

Status: simplifying Core #2886 and its provider/PennyLane consumers on current
main. The previous published integration is the comparison baseline; the
simplification is undergoing local validation and independent review.

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

Terminal normalization supports one measurement per selected qubit, direct
writes to distinct classical slots, permutations, subsets, and untouched zero
holes. It rejects feedback, repeated measurement, overwritten destinations, and
classical computation. This restriction does not apply to the default dynamic
path. A result map contains only raw indices or zero holes.

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

## Remaining work

- Finish shared/provider/PennyLane local validation and generated stubs.
- Inspect rendered shortened docs and resolve independent review findings.
- Update signed Core and PennyLane commits, then provider dependency pins and PR
  descriptions. Preserve IBM's existing cleanup-series base.
- Remove stale experiment scaffolding from worktrees; retain evidence outside
  the repositories. Hosted CI is not being monitored in this task.

## Validation

Use asymmetric deterministic circuits and inspect complete samples as well as
counts. Cover measured subsets, zero holes, sparse site IDs, routing layouts,
parameter restrictions, and default feedback/reset/reused-bit execution. Reject
unsupported terminal forms before any network submission. Verify actual provider
payload consumers; Braket LocalSimulator is not a cloud syntax oracle.

Previous live evidence covers SV1, Rigetti, and AQT (16 shots each), plus an IQM
mock. Those runs predate this simplification and are not fresh live validation.
No IBM hardware run has been performed. Final local checks are recorded in
/home/nvidia/evaluations/qdmi-results-2026-10-28/.
