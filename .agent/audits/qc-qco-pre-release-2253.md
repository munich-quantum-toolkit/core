# QC/QCO pre-release performance and determinism audit

Date: 2026-09-10. Audit baseline: `ad74680f1ef380456a1b89a810ef33ee8d218f69`.
Local validation base: `9d6526f4827ed96f7a16880325c61fe725f2f737`. The original
performance fixes are recorded at `37bda5bddf073039f90122d882f6f2eba98a7a38`;
the final source is `f0b995a4450e7c93d4ef9141861cda6a1b674fd0`, including the
index-provenance fix and main above. Scope: QC/QCO builders, conversion, QTensor
canonicalization, and compiler/export boundaries, following
[issue #2253](https://github.com/munich-quantum-toolkit/core/issues/2253).

## Result

All four confirmed findings are fixed, including the additional index-provenance
bug found during the final audit. The final review confirmed no further defect
in these changed paths. This does not clear the whole MLIR issue or extend the
builder's supported control-flow subset.

## Applied finding 4: preserve tensor indices across structured results

Original priority: P2. Confidence: high. Disposition: applied.

`qtensorExtract` records an SSA index in the qubit's `regIndex`. Previously,
structured builders copied a callback's index to the enclosing result, even when
the index was defined inside that callback's region. Later automatic reinsertion
then emitted an operand that did not dominate its use.

The builder now saves each input's type, register ID, and extracted slot before
constructing a structured operation. Each callback preserves input types and
tensor register IDs by result position. Scalar qubit outputs may permute the
input qubits but must preserve the set of extracted tensor slots. Results use
the input slots and dominating indices by position. Equal constant indices are
accepted; dynamic indices must use the same SSA value. Changed tensor register
IDs, changed sets of slots, and unprovable dynamic equivalence terminate with a
usage diagnostic. Carry the full tensor and reinsert inside the callback when
changing the set of extracted slots.

The check covers both `qcoIf` branches, `scfFor`, both `scfWhile` regions, and
every `qcoIndexSwitch` case and default. Scalar overloads delegate to the range
overloads. The API documents these builder limits; general QCO IR is not
restricted by this builder check. No index hoisting or persistent analysis cache
is introduced.

The local reproducer failed with the exact upstream builder and verified with
the final builder. The
[regression tests](../../mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp) cover
nested reinsertion, equivalent constants, shared dynamic indices, changed slots
in every callback, changed registers, and region-local dynamic indices.

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
arguments. The nested test also covers re-extraction with region-local
constants.

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

These historical measurements compare `ad74680f1` with `37bda5bdd`:

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

These historical measurements compare `ad74680f1` with `37bda5bdd`:

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

## Ponytail review

The review found two duplicated scalar implementations and their now-unused
single-argument preparation helper. Both findings were applied:

- `QCOProgramBuilder::qcoIndexSwitch: shrink:` duplicated scalar construction.
  Delegate to the range overload with locally owned callback adapters.
- `QCOProgramBuilder::qcoIf: shrink:` duplicated scalar construction. Delegate
  to the range overload and delete the unused `prepareInitArg` helper.

net: -46 lines possible.

## Performance measurements

The measured source `f0b995a44` was compared with the same `ad74680f1` baseline
using five alternating process pairs and three timed samples after two warmups.
The table summarizes the medians and complete sample ranges.

| Workload                              | Baseline median (range), ms | Final median (range), ms |
| ------------------------------------- | --------------------------: | -----------------------: |
| 1,024 used slots                      |   871.313 (866.157–881.210) |   26.061 (25.811–26.307) |
| 1,024 fresh slots                     |      12.659 (12.498–12.743) |   13.648 (13.421–13.795) |
| 4,096 registers with extracted qubits |     51.707 (51.617–262.692) |      1.123 (1.065–4.378) |
| 4,096 registers without extractions   |         0.444 (0.439–0.492) |      0.524 (0.519–0.531) |

The used-slot and extracted-register improvements remain substantial. The
fresh-slot regression is **7.8%**. Boundary validation adds work to argument
preparation: the no-extraction case is **18.0%** slower than baseline, an
absolute increase of **0.080 ms** at 4,096 registers. These costs are retained
with the correctness checks and conservative reset proof.

Both canonicalization variants emit byte-identical IR with the required reset
counts at every size. Both final-source reproducibility workloads produce one
serialized module across 24 fresh processes, with verification and linearity
checks. The host had concurrent LLVM/Core builds; CPU 18 affinity does not
provide isolation. The complete sample spread includes scheduler outliers. These
are microbenchmarks, not whole-compiler speedup guarantees.

## Validation

- Full native validation of the permutation fix used Release/Clang 23.1.1 and
  LLVM/MLIR 23.1.0: **3,464 tests passed**, including all 2,595 MLIR tests;
  `ScQDMIJobSpecificationTest.QueryJobId` skipped. This includes the three
  branching-GHZ mapping tests that exposed the restrictive slot check in CI. A
  builder regression covers permutations across two registers for if, for,
  while, and switch, with scalar-only and mixed tensor/scalar arguments. It
  checks final reinsertion slots, ordinary verification, and QCO linearity.
- Initial validation used Release/Clang 23.1.1 and LLVM/MLIR 23.1.0. All
  **1,376 selected native tests** passed: QCO IR 587, QTensor IR 42, QC-to-QCO
  178, QCO-to-QC 153, QC/QCO round trip 6, QCO utilities 192, QTensor utilities
  4, QTensor transforms 2, and compiler 212.
- Full-file C++ lint from the fixed PR base `9d6526f48` selected all five
  changed source/test files and reported zero findings. Repository lint,
  `git diff --check`, and MLIR documentation generation passed.
- The standalone index reproducer exits 1 with the baseline builder and 0 with
  the final builder. Its final output verifies and passes QCO linearity.
- Native probe compilation, benchmark assertions, Python syntax checks, and plot
  generation passed during the investigation.
- All commits are signed and verified. These are local checks; hosted CI is
  reported separately. No sanitizer or hardware-execution result is claimed.
