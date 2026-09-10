# QC/QCO pre-release performance and determinism audit

Date: 2026-09-10. Upstream baseline: `ad74680f1ef380456a1b89a810ef33ee8d218f69`.
Applied fixes: `37bda5bddf073039f90122d882f6f2eba98a7a38`. A final fetch
confirmed that upstream main still matches the baseline. Scope: QC/QCO builders,
conversion, QTensor canonicalization, and compiler/export boundaries, following
[issue #2253](https://github.com/munich-quantum-toolkit/core/issues/2253).

## Result

The three original findings are fixed. No additional performance or
reproducibility defect was confirmed in the changed paths. The final audit found
one additional, pre-existing **P2 correctness defect** in QCO builder index
tracking across region exits. It remains open and prevents an unconditional
readiness verdict for the builder.

[Harness, raw measurements, plots, settings, and reproduction commands](../benchmarks/qc-qco-pre-release/README.md).
No remote state was changed.

## Open finding: region-local tensor indices escape through scalar results

Priority: P2. Confidence: high. Disposition: newly confirmed, not implemented.

`mlir/lib/Dialect/QCO/Builder/QCOProgramBuilder.cpp::qtensorExtract` records an
SSA index in the qubit's `regIndex`. `updateQubitTracking` copies that index
when structured builders map a branch result to an enclosing operation result.
`insertExtractedQubits` later uses the copied value during preparation or
finalization, even when the index was defined inside the completed branch.

The [minimal reproducer](../benchmarks/qc-qco-pre-release/escaping-index.cpp)
allocates one tensor slot, carries its tensor and extracted qubit through
`qcoIf`, reinserts and extracts that qubit inside the then branch, and
finalizes. The builder emits an outer `qtensor.insert` whose index belongs to
the then region. Verification fails with
`operand #2 does not dominate this use`.

The exact baseline builder source/header and the fixed builder both reproduce
this failure and emit identical modules. Their builds use the same dialect
libraries, and no transformations run. The normal-form ordering changes do not
cause this defect. The relevant extraction and tracking code is unchanged.

Fix provenance at structured-region boundaries. Track the association between
each yielded scalar, its register, and its slot across all possible branches;
ensure any index used outside a region dominates that use. Diagnose combinations
that the builder cannot represent. Do not merely hoist a constant: if branches
return different slots, one branch's index is not a valid unconditional index
for the merged result. A complete fix also needs loop and switch coverage.

Until then, reinserting within each branch and carrying the full tensor across
the boundary avoids this reproducer. The additional correctness work is separate
from the three performance and determinism changes applied here.

## Applied finding 1: deterministic live-value disposal and reinsertion

Original priority: P1, a determinism blocker under #2253. Disposition: applied.

`QCOProgramBuilder` retains DenseSets for membership checks. It now emits
tensors in register creation order and orders scalar sinks and tensor
reinsertion by SSA definitions, including block-argument and operation-result
positions. Sorting occurs only at emission boundaries; gate updates keep
constant-time hash lookup. No pointer address or hash-table traversal determines
emitted order.

Before the fix, identical scalar and tensor builder calls each produced 24
serialized modules from 24 fresh processes. After the fix, each workload
produces exactly **one** serialized module across 24 processes. All outputs
verify and pass QCO linearity. This checks artifact reproducibility; it does not
claim a change in quantum probabilities.

Durable tests cover scalar disposal, same-operation result ordering, tensor
creation order, insertion order, explicit scalar arguments, and nested block
arguments. The nested ordering test uses indices that dominate their uses; the
separate escaping-index defect is retained as an open reproducer above.

## Applied finding 2: allocation-rooted freshness discovery

Original priority: P2, high measured impact. Disposition: applied.

The canonicalization pattern now belongs to `qtensor.alloc` and scans its linear
tensor chain once per match attempt. It tracks accessed constant slots and
removes resets only on their first extraction. Unknown indices, unsupported
users, and block boundaries stop the proof. Required resets remain intact. This
removes the backward search repeated at every unsuccessful reset match.

Successful rewriting replaces the allocation with an attribute-preserving clone,
satisfying the
[MLIR root-rewrite contract](https://mlir.llvm.org/docs/PatternRewriter/#restrictions).
No persistent cache over mutable IR is introduced. Root replacement can cause a
second linear scan, which contributes to the measured fresh-slot overhead.

| Used slots | Upstream median | Fixed median |
| ---------- | --------------: | -----------: |
| 128        |        15.53 ms |      3.10 ms |
| 256        |        58.11 ms |      6.28 ms |
| 512        |       220.57 ms |     12.62 ms |
| 1,024      |       869.81 ms |     26.02 ms |

At 1,024 used slots, sample ranges are 865.55–982.91 ms before and 25.67–63.86
ms after. Every required reset remains, and both versions emit byte-identical
canonical IR at every measured size.

Fresh-slot canonicalization remains linear and removes every eligible reset. At
1,024 fresh slots it changes from **12.65 to 13.62 ms**, a **7.7% regression**.
Sample ranges are 12.51–12.84 ms before and 13.41–13.79 ms after. This is a
measured tradeoff, not a whole-compiler speedup claim. The useful fresh-slot
fold is preserved; deleting it was only an earlier diagnostic experiment.

## Applied finding 3: group extracted qubits once per argument preparation

Original priority: P2, lower release urgency. Disposition: applied.

`prepareInitArgs` groups live qubits by the requested registers once. It
excludes explicit scalar arguments, validates consumed values, and delegates
insertion to the same ordered helper used by finalization. The single-argument
overload uses the same path. If no qubits are live, validation returns the
arguments without allocating the grouping map.

This removes the roughly N(N+1)/2 live-set visits for N one-slot registers.
Grouping is linear in the live values and arguments, followed by sorting within
each register. It does not introduce linear erasure on ordinary gate updates.

| Registers with one extracted qubit each | Upstream median | Fixed median |
| --------------------------------------- | --------------: | -----------: |
| 256                                     |        0.260 ms |     0.062 ms |
| 1,024                                   |        3.403 ms |     0.242 ms |
| 4,096                                   |       51.616 ms |     0.983 ms |

At 4,096 registers, sample ranges are 51.54–51.70 ms before and 0.970–1.052 ms
after. The no-extraction workload changes from 0.447 to 0.429 ms. These measure
`qcoIf` construction, excluding initial program construction and finalization.
They do not represent typical single-register circuits.

## Final audit coverage and limits

The final pass traced tracking producers and consumers, structured-argument
preparation, finalization and helper-function disposal, the allocation/extract/
insert/reset chain, and neighboring conversion and QTensor emission paths. It
checked pointer-order independence, failed rewrite behavior, linearity, explicit
scalar exclusions, and the reset proof's stop conditions. The negative used-slot
workload and the unknown-index and same-index regressions pass.

The earlier OpenQASM double-preparation fix remains in main. Direct QCO export
remains deferred. QC-to-QCO sink emission retains its ordered MapVector; QTensor
branch scalarization sorts accessed indices before emission. Examined cache
invalidation traversals do not themselves emit operations.

PR #2502 remains open and owns QIR builder output ordering and the redundant
QC-to-QCO modifier-validation walk. PR #2505 remains open and owns modifier
contracts, redundant conversion checks, and QTensor helper/shrinking cleanup.
Those are not counted as new findings. This report does not clear the whole MLIR
issue or replace the outstanding contract work.

## Validation

- Rebuilt Release/Clang 23.1.1 targets against LLVM/MLIR 23.1.0. All
  **1,371 tests** passed: QCO IR 583, QTensor IR 42, QC-to-QCO 178, QCO-to-QC
  153, QC/QCO round trip 6, QCO utilities 192, QTensor utilities 4, QTensor
  transforms 2, compiler
  211. Counts and exit statuses are retained in `tests-fixed.json`.
- Full-file C++ lint on the committed diff from `ad74680f1` selected all five
  changed C++ source/test files and reported zero findings. An earlier run on
  uncommitted files selected none and is not counted as validation.
- Repository lint, `git diff --check`, and MLIR documentation generation passed.
- Harness syntax, native probe compilation, output/hash assertions, and plot
  generation passed. The new correctness reproducer fails verification as
  expected on both builder versions and remains an open finding.
- Native measurements alternate five process pairs, with two warmups and three
  samples per process, pinned to CPU 18. Raw samples include scheduler outliers.
  The preceding complete run with larger outliers is retained separately.
- The implementation commit is signed and verified. No hosted CI, sanitizers,
  hardware execution, or arbitrary-workload performance guarantees are claimed.
