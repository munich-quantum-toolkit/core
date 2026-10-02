# QC/QCO pre-release fixes

Baseline: `ad74680f1ef380456a1b89a810ef33ee8d218f69`.

Resolve the four findings in [the audit](../audits/qc-qco-pre-release-2253.md).
The QCO builder retains hash lookup for tracking, emits tensors in register
creation order, and orders scalar disposal and reinsertion by SSA definitions.
Argument preparation groups live qubits once, excluding explicitly carried
scalars. The reset canonicalizer scans from allocations instead of repeatedly
searching backward from resets. Unknown indices and unsupported tensor users
stop the proof; only first accesses in the allocation block can lose resets.

The allocation rewrite replaces its root while preserving attributes. This
follows MLIR's pattern contract without storing analysis over mutable IR.

Structured callbacks preserve input types and tensor register IDs by result
position. Scalar qubit outputs may permute the input qubits while preserving the
set of extracted slots. Results use the input slots by position. Equal constants
reuse the dominating input index; dynamic indices must be the same SSA value.
Unsupported changes produce a usage diagnostic. Carry full tensors when changing
the set of extracted slots. This prevents region-local indices from escaping
without hoisting a branch-specific index.

Scalar if/switch overloads delegate to their range counterparts. The Ponytail
review removed 46 lines of duplicated region construction and the unused
single-argument preparation helper. The register snapshot is local to each
structured operation; ordinary gate tracking remains unchanged.

All four findings are implemented. Direct QCO export and overlapping PR findings
remain outside this change.

The audit summarizes validation, measured results, regressions, and sample
spread at their original source revisions.
