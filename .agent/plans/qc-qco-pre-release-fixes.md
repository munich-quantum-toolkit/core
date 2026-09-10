# QC/QCO pre-release fixes

Baseline: `ad74680f1ef380456a1b89a810ef33ee8d218f69`.

Resolve the three findings in [the audit](../audits/qc-qco-pre-release-2253.md).
The QCO builder retains hash lookup for tracking, emits tensors in register
creation order, and orders scalar disposal and reinsertion by SSA definitions.
Argument preparation groups live qubits once, excluding explicitly carried
scalars. The reset canonicalizer scans from allocations instead of repeatedly
searching backward from resets. Unknown indices and unsupported tensor users
stop the proof; only first accesses in the allocation block can lose resets.

The allocation rewrite replaces its root while preserving attributes. This
follows MLIR's pattern contract without storing analysis over mutable IR.

A separate re-audit finding concerns region-local indices retained by scalar
results after a branch exits. Fixing it requires branch provenance agreement;
constant hoisting alone can choose the wrong slot on another branch. Preserve
this as a separate correctness finding, outside the three applied fixes.

The three original fixes are applied in
`37bda5bddf073039f90122d882f6f2eba98a7a38`. The final audit retains the
additional index-provenance defect as open. Direct QCO export and overlapping PR
findings remain outside this change.

Validation: 1,371 native tests, full-file C++ lint, repository lint, MLIR
documentation generation, and benchmark assertions pass. Matched measurements
show 869.81 to 26.02 ms for 1,024 used slots and 51.616 to 0.983 ms for 4,096
registers. Fresh-slot canonicalization regresses by 7.7% at 1,024 slots; this
tradeoff is recorded. Both reproducibility probes now produce one output across
24 processes. See the audit and benchmark directory for raw evidence and limits.
