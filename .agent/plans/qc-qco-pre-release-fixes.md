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

Structured callbacks preserve each input's type, tensor register, and extracted
slot by result position. Equal constants reuse the dominating input index;
dynamic indices must be the same SSA value. Other associations produce a usage
diagnostic. Carry full tensors when changing slot associations. This fixes the
additional region-local index escape without hoisting a branch-specific index.

Scalar if/switch overloads delegate to their range counterparts. The Ponytail
review removed 46 lines of duplicated region construction and the unused
single-argument preparation helper. The register snapshot is local to each
structured operation; ordinary gate tracking remains unchanged.

All four findings are implemented. Direct QCO export and overlapping PR findings
remain outside this change. Main was integrated with a signed merge to retain
the exact source commits referenced by the historical measurements.

Final validation and measurements are recorded in the audit and benchmark
directory. Historical performance results are retained at their original source
revisions, including the fresh-slot regression and sample spread.
