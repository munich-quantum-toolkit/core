# SpecAudit: global-phase normalization

Scope:

- production: `mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp`;
- focused tests:
  `mlir/unittests/Dialect/Utils/test_global_phase_normalization.cpp`;
- transitive assertion helper: `mlir/unittests/ExactUnitaryTest.h`;
- focused target: `mqt-core-mlir-unittests-dialect-utils`;
- GoogleTest filter: `GlobalPhaseNormalizationTest.*`;
- baseline: `cb5cf0103bd9841726c8ec6c5abb725758afea58`;
- audit date: 2026-08-19.

Every repository citation below was read at the pinned baseline. The initial and
final audit worktrees and baseline binaries were clean. No production or test
change survives. This audit adds only this ledger. It does not adjudicate or
apply a verdict.

## Role registry

The audit used these isolated roles:

- persistent scope steward;
- four valid fresh cartographers: machine/external, published, requested, and
  public surface;
- census;
- two independent prosecutors;
- provenance;
- defender;
- serialized executor plus two supplemental fresh executors;
- unlock analyst;
- architecture analyst;
- fresh red team;
- fresh final editor.

One earlier requested-behavior cartographer was discarded. A broad pull-request
timeline exposed test diffs and violated cartographer isolation. Its work did
not enter this ledger.

After the native agent-tree lifetime cap was reached, role isolation used
separate fresh sessions and detached disposable worktrees. Each role received
only the inputs needed for its wave.

## Spec ledger

**S1, rung 1.** `NormalizeGlobalPhases` is a `ModuleOp` pass. It depends on
Arith, QC, and QCO, and its stated scope is QC and QCO phase normalization
(`mlir/include/mlir/Dialect/Utils/Transforms/Passes.td:14-17`).

**S2, rung 1.** The pass combines direct `qc.gphase` and `qco.gphase` operations
into at most one per basic block, immediately before the terminator. Nested
regions normalize independently
(`mlir/include/mlir/Dialect/Utils/Transforms/Passes.td:18-21`).

**S3, rung 1.** Extraction from `inv` negates the angle. Extraction from `pow`
requires a finite compile-time integral exponent. Extraction from `ctrl`
produces `p` for one control and a smaller controlled `p` for more controls. The
pass moves a local angle slice only when the complete slice is pure and
independent of modifier block arguments
(`mlir/include/mlir/Dialect/Utils/Transforms/Passes.td:23-28`).

**S4, rung 1.** Function, CFG, structured-control-flow, and unknown-region
boundaries stay intact. Traversal is linear in visited operations and avoids
module rescans from individual rewrite patterns
(`mlir/include/mlir/Dialect/Utils/Transforms/Passes.td:30-33`).

**S5, rung 1.** QC and QCO `gphase` are zero-target, one-`f64`-parameter unitary
operations with `MemWrite` (`mlir/include/mlir/Dialect/QC/IR/QCOps.td:138-160`,
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:142-164`). Direct constants must be
finite with magnitude at most 10000 radians. Dynamic angles have the same
runtime precondition (`mlir/include/mlir/Dialect/QC/IR/QCOps.td:161-183`,
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:165-190`). Both verifiers enforce
the constant case
(`mlir/lib/Dialect/QC/IR/Operations/StandardGates/GPhaseOp.cpp:30-37`,
`mlir/lib/Dialect/QCO/IR/Operations/StandardGates/GPhaseOp.cpp:59-66`). The
exported utility is finite and `abs(theta) <= 10000`
(`mlir/include/mlir/Dialect/Utils/Utils.h:77-86`). By application of S5, every
`gphase` emitted by this pass must meet the same precondition.

**S6, rung 1.** QC and QCO modifiers have one-block implicit-yield regions.
Their bodies may contain classical SSA work and may capture classical SSA values
(`mlir/include/mlir/Dialect/QC/IR/QCOps.td:971-988`,
`mlir/include/mlir/Dialect/QC/IR/QCOps.td:1038-1051`,
`mlir/include/mlir/Dialect/QC/IR/QCOps.td:1096-1114`,
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:1167-1185`,
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:1256-1270`,
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:1332-1353`). QC rejects CBit
allocation/load/store, QC allocation/deallocation/static/measure/reset, MemRef
load/store, and qubit capture
(`mlir/lib/Dialect/QC/IR/Modifiers/ModifierUtils.cpp:34-59`). QCO rejects CBit
allocation/load/store, QCO allocation/sink/static/measure/reset, QTensor
extract/insert, and qubit capture
(`mlir/lib/Dialect/QCO/IR/Modifiers/ModifierUtils.cpp:34-59`). These are
distinct verifier lists. They do not promise that every unknown classical
operation is accepted or pure.

**S7, rung 1.** Target aliases map modifier operands to qubit block arguments
structurally (`mlir/include/mlir/Dialect/Utils/Utils.h:232-300`). QCO modifiers
also carry matching operand/result shape traits and explicit input/output groups
(`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:1167-1201`). A QCO yield must have
the parent-defined count, order, and types
(`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td:1141-1164`,
`mlir/lib/Dialect/QCO/IR/QCOOps.cpp:211-240`). The modifier verifiers also check
target block-argument and yield counts
(`mlir/lib/Dialect/QCO/IR/Modifiers/CtrlOp.cpp:305-350`,
`mlir/lib/Dialect/QCO/IR/Modifiers/InvOp.cpp:476-508`,
`mlir/lib/Dialect/QCO/IR/Modifiers/PowOp.cpp:829-860`).

**S8, rung 1, external.** OpenQASM defines `gphase(gamma) = exp(i gamma) I` in
the scope that contains it. Products of same-scope phases therefore add their
angles modulo `2*pi`. The addition and periodicity statement is a direct
inference from the specified unitary, not a choice of representative. See the
official
[OpenQASM 3.1 gate specification](https://openqasm.com/versions/3.1/language/gates.html#gate-gphase).

**S9, rung 1, external.** OpenQASM defines the `inv`, `pow`, and positive `ctrl`
semantics. It states that `inv @ gphase(a)` is `gphase(-a)` and that one
positive control of `gphase(a)` is `p(a)`. See the official
[modifier definitions](https://openqasm.com/versions/3.1/language/gates.html#quantum-gate-modifiers)
and
[`p` definition](https://openqasm.com/versions/3.1/language/standard_library.html#p).
MQT's zero-control form is valid variadic IR. It is not an OpenQASM input
promise.

**S10, rung 2.** The changelog announces modifier and global-phase
normalization, but does not promise spelling or representation detail
(`CHANGELOG.md:54-64`).

**S11, rung 2.** The public surface has a direct module helper and in-place QC
and QCO C++ methods
(`mlir/include/mlir/Dialect/Utils/Transforms/GlobalPhaseNormalization.h:19-22`,
`mlir/include/mlir/Compiler/Programs.h:176-180`,
`mlir/include/mlir/Compiler/Programs.h:210-214`). The methods call the same
helper (`mlir/lib/Compiler/Programs.cpp:323-330`,
`mlir/lib/Compiler/Programs.cpp:394-401`).

**S12, rung 2.** The generated pass declaration and registration surface is
exported (`mlir/include/mlir/Dialect/Utils/Transforms/Passes.h:16-22`). The
public compiler registration entry point registers this pass
(`mlir/include/mlir/Support/Passes.h:31-32`,
`mlir/lib/Support/Passes.cpp:51-64`).

**S13, rung 2.** The QC and QCO Python methods take no arguments and return
`None` (`python/mqt/core/mlir.pyi:350-357`, `python/mqt/core/mlir.pyi:398-405`).
The bindings adapt a C++ `false` result to `RuntimeError`
(`bindings/mlir/register_mlir.cpp:70-75`,
`bindings/mlir/register_mlir.cpp:96-103`) and bind both methods
(`bindings/mlir/register_mlir.cpp:678-684`,
`bindings/mlir/register_mlir.cpp:741-748`). No source promises a stable error
message.

**S14, rung 2.** The exported validity utility accepts exactly finite angles
with magnitude at most 10000. It does not select a normalized representative
(`mlir/include/mlir/Dialect/Utils/Utils.h:77-86`).

**S15, rung 3.**
[Issue `#1641`](https://github.com/munich-quantum-toolkit/core/issues/1641)
requests merging multiple `GPhaseOp` operations.

**S16, rung 3.** The same issue requests merging by adding the angles.

**S17, rung 3.** The same issue requests cancellation of opposite angles.

**S18, rung 3 for purpose only.** The Purpose of
`.agent/plans/global-phase-normalization.md` requires preservation of complete
unitaries and direct pass, C++, Python, and cleanup APIs
(`.agent/plans/global-phase-normalization.md:10-27`). Later choices about exact
dynamic operand order, the last control as the relative-phase target, and
byte-identical printing are rung 4 plan details, not promises
(`.agent/plans/global-phase-normalization.md:270-291`,
`.agent/plans/global-phase-normalization.md:351-356`).

## Explicit non-promises

No rung 1 to 3 source promises:

- an exact SSA tree or operand order;
- a retained source location;
- exact diagnostic text;
- one modulo-`2*pi` representative;
- byte-stable printing;
- mixed QC/QCO rejection;
- zero-control phase placement;
- exact cleanup ordering.

## GitHub drift

The live refresh immediately before red team found 47 open issues and 18 open
pull requests. A resolution refresh on 2026-08-20 found 47 open issues and 27
open pull requests. The remote default ref still resolved to the pinned baseline
in both refreshes. Every open item was scanned for a relationship to this scope,
and the changed file list of every open pull request was checked. No open pull
request directly changes the normalizer, the focused tests, `Passes.td`, or
`GlobalPhaseNormalization.h`.

| Item                                                                                                                                              | Disposition                                                                                                |
| :------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| [Open issues](https://github.com/munich-quantum-toolkit/core/issues?q=is%3Aissue%20state%3Aopen)                                                  | 47 scanned; no direct baseline blocker.                                                                    |
| [Open pull requests](https://github.com/munich-quantum-toolkit/core/pulls?q=is%3Apr%20state%3Aopen)                                               | 18 scanned during audit; 27 during resolution reconciliation, including every changed file.                |
| [PR `#2150`](https://github.com/munich-quantum-toolkit/core/pull/2150)                                                                            | Its only audited-scope touch is shared `Utils.h`; symbolic-Arith downstream overlap, not a direct blocker. |
| [PR `#2178`](https://github.com/munich-quantum-toolkit/core/pull/2178)                                                                            | Also touches shared `Utils.h`; Qiskit parameter provenance is not a direct blocker.                        |
| [PR `#2062`](https://github.com/munich-quantum-toolkit/core/pull/2062)                                                                            | Gate-angle quantization is a downstream validation consumer.                                               |
| [PR `#2080`](https://github.com/munich-quantum-toolkit/core/pull/2080)                                                                            | DD density simulation is a downstream validation consumer.                                                 |
| [Issue `#1641`](https://github.com/munich-quantum-toolkit/core/issues/1641)                                                                       | Closed request anchor for S15-S17.                                                                         |
| [PR `#1986`](https://github.com/munich-quantum-toolkit/core/pull/1986) and [PR `#1995`](https://github.com/munich-quantum-toolkit/core/pull/1995) | Merged provenance anchors.                                                                                 |

Unrelated open items were excluded only after the full scan.

## Human decision

On 2026-08-20, the maintainer accepted verdicts 1-6, deferred verdict 7, and
kept verdicts 8-9. The accepted remedies will share one pull request but remain
separate commits. Verdict 7 requires a later ownership decision across global
phase normalization, control canonicalization, and QIR conversion. This audit
does not mark a verdict applied until its resolution merges.

## Assertion census

The focused file contains exactly 115 literal assertion sites. Five are in its
local helpers. `ExactUnitaryTest.h` contributes exactly five transitive sites at
lines 46, 47, 51, 53, and 58. The total is 120.

For stable IDs, `#N` is the assertion's source-order ordinal within its named
GoogleTest or helper. Ranges include every ordinal. The table enumerates every
site exactly once. `B0` and `B1` are the clean baseline runs below. Mutation IDs
name the more specific fault evidence. Parse guards, pass-success guards,
verification guards, QC-to-QCO pass guards, result/count/position guards, and
matrix-construction guards are included in the Anchored count.

| Stable ID                                                                                    | Baseline line(s)             | Class          | Promise and evidence               |
| :------------------------------------------------------------------------------------------- | :--------------------------- | :------------- | :--------------------------------- |
| `expectNormalizedUnitary#1-#2`                                                               | 76, 77                       | Anchored       | S1, S5-S7, S11; B0, B1, M5-M7, M10 |
| `expectNormalizedQCUnitary#1-#3`                                                             | 85, 90, 91                   | Anchored       | S1, S5-S7, S18; B0, B1             |
| `GlobalPhaseNormalizationTest.CombinesQCOConstantsAtBlockExit#1-#4`                          | 112, 113, 117, 118           | Anchored       | S1, S2, S11; B0, B1, M1, M8        |
| `GlobalPhaseNormalizationTest.CombinesQCOConstantsAtBlockExit#5-#6`                          | 122, 123                     | Over-specified | S2, S8, S16; M1, M8                |
| `GlobalPhaseNormalizationTest.FoldsMulDerivedPhasesWithinPracticalAngleLimit#1-#5`           | 153, 154, 157, 160, 162      | Anchored       | S2, S5, S14, S18; B0, B1, M1, M5   |
| `GlobalPhaseNormalizationTest.FoldsMulDerivedPhasesWithinPracticalAngleLimit#6`              | 163                          | Over-specified | S5, S8; M1                         |
| `GlobalPhaseNormalizationTest.FoldsMulDerivedPhasesWithinPracticalAngleLimit#7`              | 164                          | Redundant      | S5, S14; M5                        |
| `GlobalPhaseNormalizationTest.FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit#1-#5`     | 197, 198, 201, 204, 206      | Anchored       | S2, S5, S14, S18; B0, B1, M1, M10  |
| `GlobalPhaseNormalizationTest.FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit#6`        | 207                          | Over-specified | S5, S8; M1                         |
| `GlobalPhaseNormalizationTest.FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit#7`        | 208                          | Redundant      | S5, S14; M10                       |
| `GlobalPhaseNormalizationTest.QCControlledExtractionPreservesFullUnitaryUnderOuterControl#1` | 227                          | Anchored       | S3, S9, S18; B0, B1                |
| `GlobalPhaseNormalizationTest.QCInverseAndIntegralPowerPreserveFullUnitary#1`                | 247                          | Anchored       | S3, S9, S18; B0, B1                |
| `GlobalPhaseNormalizationTest.PreservesDynamicOrderAndIsIdempotent#1-#3`                     | 262, 263, 267                | Anchored       | S1, S2, S11; B0, B1, M2, M3, M9    |
| `GlobalPhaseNormalizationTest.PreservesDynamicOrderAndIsIdempotent#4-#6`                     | 269-271                      | Over-specified | S2, S4, S5, S16; M2, M3, M9        |
| `GlobalPhaseNormalizationTest.PreservesDynamicOrderAndIsIdempotent#7`                        | 276                          | Anchored       | S1, S11; B0, B1, M3                |
| `GlobalPhaseNormalizationTest.PreservesDynamicOrderAndIsIdempotent#8`                        | 280                          | Contract-free  | No promise; M3                     |
| `GlobalPhaseNormalizationTest.KeepsSCFStyleRegionsIndependent#1-#5`                          | 302, 303, 307-309            | Anchored       | S2, S4; B0, B1                     |
| `GlobalPhaseNormalizationTest.FactorsInverseAndIntegralPower#1-#5`                           | 333, 334, 339-341            | Anchored       | S2, S3, S8, S9; B0, B1             |
| `GlobalPhaseNormalizationTest.FractionalPowerRemainsBoundary#1-#3`                           | 359, 360, 364                | Anchored       | S3, S9; B0, B1                     |
| `GlobalPhaseNormalizationTest.DynamicPowerRemainsBoundary#1-#4`                              | 381, 382, 386, 387           | Anchored       | S3, S9; B0, B1                     |
| `GlobalPhaseNormalizationTest.NonFinitePowerExponentsRemainBoundaries#1-#4`                  | 412-415                      | Anchored       | S3, S5; B0, B1                     |
| `GlobalPhaseNormalizationTest.FactorsControlledPhaseOntoControl#1-#6`                        | 436, 437, 441, 442, 445, 446 | Anchored       | S3, S7, S9; B0, B1                 |
| `GlobalPhaseNormalizationTest.ControlledExtractionPreservesFullUnitaryUnderOuterControl#1`   | 475                          | Anchored       | S3, S7, S9, S18; B0, B1            |
| `GlobalPhaseNormalizationTest.ThreeControlsPreserveFullUnitary#1-#4`                         | 499, 504-506                 | Anchored       | S3, S7, S9, S18; B0, B1, M7        |
| `GlobalPhaseNormalizationTest.ReorderedQCOControlsThreadCorrectResults#1-#4`                 | 528-530, 534                 | Anchored       | S3, S5, S7; B0, B1, M7             |
| `GlobalPhaseNormalizationTest.ReorderedQCOControlsThreadCorrectResults#5-#7`                 | 536-538                      | Over-specified | S3, S7, S9; M7                     |
| `GlobalPhaseNormalizationTest.ReorderedQCOControlsThreadCorrectResults#8`                    | 539                          | Anchored       | S7; B0, B1, M7                     |
| `GlobalPhaseNormalizationTest.MultipleTargetsPreserveFullUnitary#1`                          | 562                          | Anchored       | S3, S7, S18; B0, B1                |
| `GlobalPhaseNormalizationTest.IntegralPowersPreserveFullUnitaryAndReleasePhase#1-#2`         | 585, 589                     | Anchored       | S3, S9, S18; B0, B1                |
| `GlobalPhaseNormalizationTest.NestedInverseAndPowerOrdersPreserveFullUnitary#1-#3`           | 622, 626, 629                | Anchored       | S3, S9, S18; B0, B1                |
| `GlobalPhaseNormalizationTest.ZeroControlsReleaseAnUnchangedPhase#1-#2`                      | 653, 654                     | Contract-free  | No promise; M6                     |
| `GlobalPhaseNormalizationTest.MemoryDependentAngleRemainsInsideModifier#1-#5`                | 674-676, 680, 681            | Anchored       | S3, S5, S6; B0, B1                 |
| `GlobalPhaseNormalizationTest.CFGBlocksRemainIndependentScopes#1-#8`                         | 704-706, 713-717             | Anchored       | S2, S4, S5; B0, B1                 |
| `GlobalPhaseNormalizationTest.FunctionsRemainIndependentScopes#1-#4`                         | 739, 740, 743, 745           | Anchored       | S2, S4; B0, B1                     |
| `GlobalPhaseNormalizationTest.IndexSwitchRegionsRemainIndependentScopes#1-#5`                | 770-772, 777, 779            | Anchored       | S2, S4, S5, S7; B0, B1             |
| `GlobalPhaseNormalizationTest.SCFLoopRegionRemainsAnIndependentScope#1-#5`                   | 799-801, 805, 806            | Anchored       | S2, S4, S5; B0, B1                 |
| `GlobalPhaseNormalizationTest.ExactSpecialConstantsCancelWithoutTolerance#1-#2`              | 826, 827                     | Anchored       | S8, S17; B0, B1, M4                |
| `GlobalPhaseNormalizationTest.PracticalAngleBoundaryPreservesFullUnitaryUnderControl#1`      | 850                          | Anchored       | S3, S5, S9, S18; B0, B1            |
| `GlobalPhaseNormalizationTest.VerifiesPracticalConstantAngleRange#1-#2`                      | 878, 886                     | Anchored       | S5, S14; B0, B1, M5, M10           |
| `expectFullUnitaryEqual#1-#2`                                                                | 46, 47                       | Anchored       | S18; B0, B1, M1, M6, M7            |
| `expectFullUnitaryEqual#3-#4`                                                                | 51, 53                       | Anchored       | S18 and matrix safety; B0, B1      |
| `expectFullUnitaryEqual#5`                                                                   | 58                           | Anchored       | S8, S9, S18; B0, B1, M1, M6, M7    |

Focused-file arithmetic after red team:

| Class           | Count | Baseline lines outside the 100 Anchored sites |
| :-------------- | ----: | :-------------------------------------------- |
| Anchored        |   100 | Enumerated above                              |
| Over-specified  |    10 | 122, 123, 163, 207, 269-271, 536-538          |
| Redundant       |     2 | 164, 208                                      |
| Contract-free   |     3 | 280, 653, 654                                 |
| Coverage-driven |     0 | None                                          |

The five `ExactUnitaryTest.h` sites are Anchored. Thus the full arithmetic is
`100 + 10 + 2 + 3 + 5 = 120`.

## Summary

Ranked by complexity removed per unit of risk. This pilot found test narrowing
and one correctness-enabling representation freedom. It did not earn a
production deletion or measure a performance gain.

| Rank | Verdict                           | Class          | Main remedy                        | Risk                      |
| ---: | :-------------------------------- | :------------- | :--------------------------------- | :------------------------ |
|    1 | 3, dynamic dependency shape       | Over-specified | Semantic two-input oracle          | Medium; enables S5 repair |
|    2 | 1, folded QCO shape/value         | Over-specified | Safe foldable value oracle         | Low                       |
|    3 | 4, symmetric control mapping      | Over-specified | Wire-identity or unitary oracle    | Medium                    |
|    4 | 6, printed idempotence            | Contract-free  | Remove byte equality               | Low                       |
|    5 | 7, zero-control placement         | Contract-free  | Keep semantics, drop placement     | Medium ownership question |
|    6 | 2, exact constant representatives | Over-specified | Direct constant plus modulo oracle | Low                       |
|    7 | 5, duplicate validity checks      | Redundant      | Remove two assertions              | Low                       |
|    8 | 8, exact cancellation             | Anchored       | Keep                               | Low                       |
|    9 | 9, unitary helper guards          | Anchored       | Keep                               | Low                       |

## Verdicts and remedies

### 1. The first folded QCO test requires one exact constant shape and value

`GlobalPhaseNormalizationTest.CombinesQCOConstantsAtBlockExit#5-#6` at
`mlir/unittests/Dialect/Utils/test_global_phase_normalization.cpp:122-123` are
**Over-specified** against S2, S8, and S16. The promises require one phase with
the summed value modulo `2*pi`. They do not require an `arith.constant` or the
representative `0.75`.

M1 changed only the representative and failed line 123. M8 returned the same
`0.75` through `arith.addf(0.25, 0.5)`. The target built, but this test received
`SIGSEGV` at line 121 in `ConstantOp::getValue` because it dereferenced an
unchecked failed `getDefiningOp<arith::ConstantOp>()`. The other 26 tests
passed.

**Remedy.** Retain a value oracle, but accept any foldable, verifier-valid
expression with the correct value modulo `2*pi`. Call
`utils::valueToConstantDouble` safely, assert that it returned a value, assert
`utils::isValidGlobalPhaseAngle`, and compare the result modulo `2*pi` with
`0.75`. Do not dereference an unchecked `ConstantOp`.

### 2. Two defect regressions pin one normalized numeric representative

`FoldsMulDerivedPhasesWithinPracticalAngleLimit#6` at line 163 and
`FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit#6` at line 207 are
**Over-specified** against S5 and S8. M1 added `2*pi` to materialized constants.
Only lines 123, 163, and 207 failed; full matrices and verification passed.

These tests came from
[PR `#1995`](https://github.com/munich-quantum-toolkit/core/pull/1995), which
fixed real later-folding and verifier failures. The defect regression must
survive: the pass must emit a direct verifier-valid `arith.constant` and the
aggregate must be correct modulo `2*pi`. Preserve the direct-constant assertions
at lines 160 and 204. Replace exact representative equality with a semantic
modulo oracle. Do not weaken the test to accept arbitrary zero.

### 3. Dynamic phases pin the root opcode and operand order

`PreservesDynamicOrderAndIsIdempotent#4-#6` at lines 269-271 are
**Over-specified** against S2, S4, S5, and S16.

- M2 reversed the `arith.addf` operands. Only lines 270-271 failed. Line 269 and
  the printed-idempotence assertion passed.
- M3 retained a neutral `+0` around each phase. Only lines 270-271 and 280
  failed.
- M9 materialized the sum as `lhs - (-rhs)`. Exactly line 269 failed. The other
  26 tests passed.

**Remedy.** Replace the root/op-order checks with either an unordered transitive
dependency oracle or an evaluated semantic oracle that proves both dynamic
inputs contribute. The narrowed test must still fail if either input is omitted.
It may allow a validity-preserving runtime modulo representation.

This freedom is a correctness unlock. The sum of two individually valid dynamic
angles can exceed 10000, so S5 can require a bounded runtime representation
without breaking an incidental tree test. It is not evidence that production
source can be deleted.

### 4. Reordered controls pin one symmetric relative-phase target

`ReorderedQCOControlsThreadCorrectResults#5-#7` at lines 536-538 are
**Over-specified** against S3, S7, and S9. M7 selected the first rather than
last symmetric control as the relative-phase target and restored the result
vector. Only lines 536-538 failed. Line 539, module verification, and the other
matrix tests passed. The reordered fixture itself has no matrix oracle.

**Remedy.** Keep line 539. Replace lines 536-538 with a representation-neutral
mapping from each original input wire to its final output, or make the wires
distinguishable in downstream operations and compare the complete unitary. The
oracle must protect QCO yield order and wire identity, not one legal choice of
symmetric target.

### 5. Two explicit validity checks duplicate stronger checks

`FoldsMulDerivedPhasesWithinPracticalAngleLimit#7` at line 164 and
`FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit#7` at line 208 are
**Redundant**. Each follows successful whole-module verification of the same
direct constant at lines 154 and 198. The independent boundary test at lines 878
and 886 checks both dialect verifiers.

M5 emitted 10001 for the Mul-derived result. The test stopped at module
verification on line 154 before line 164. M10 did the same for the
SIToFP-derived result; it stopped at line 198 before line 208. The independent
verifier test passed in both runs. Remove lines 164 and 208 only.

### 6. Byte-identical printing after a second pass has no contract

`PreservesDynamicOrderAndIsIdempotent#8` at line 280 is **Contract-free**. S11
promises a callable in-place API. No rung 1 to 3 source promises byte-stable
printing or exact second-run SSA spelling. M3 changed only lines 270-271 and
280.

**Remedy.** Remove the byte equality. If repeat-run coverage remains, assert
successful verification, one phase, stable semantics or dependencies, and
bounded IR growth. Do not claim that this removal permits deletion of the
normalizer's idempotence fast path. No experiment ties that production candidate
to line 280.

### 7. Zero-control phase placement has no contract

`ZeroControlsReleaseAnUnchangedPhase#1-#2` at lines 653-654 are
**Contract-free**. MQT permits a zero-control variadic modifier, but S9 makes no
OpenQASM input promise for it and no rung 1 to 3 source chooses phase placement.
M6 left the phase local. Only lines 653-654 failed; full-unitary comparison and
verification passed.

**Remedy.** Keep the full-unitary and verification oracle. Drop the exact local
versus parent placement checks.

No safe production deletion follows now. The QC and QCO `CtrlOp` canonicalizers
already inline simple zero-control bodies
(`mlir/lib/Dialect/QC/IR/Modifiers/CtrlOp.cpp:90-109`,
`mlir/lib/Dialect/QCO/IR/Modifiers/CtrlOp.cpp:112-143`). Direct QIR conversions
invoke normalization before conversion
(`mlir/lib/Conversion/QCToQIR/QIRBase/QCToQIRBase.cpp:462-468`,
`mlir/lib/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.cpp:685-691`), and QIR
rejects a nested `gphase`
(`mlir/lib/Conversion/QCToQIR/QIRCommon/QIRCommon.cpp:292-303`). Pipeline
ordering or a generalized canonicalizer is the ownership question.

### 8. Exact cancellation is a requested regression

`ExactSpecialConstantsCancelWithoutTolerance#2` at line 827 is **Anchored** by
S17. M4 retained the zero aggregate rather than erasing it. Only line 827
failed. Keep it.

### 9. The full-unitary helper guards are safety and semantic anchors

All five assertions in `ExactUnitaryTest.h` are **Anchored**. Lines 46 and 47
guard matrix construction. Lines 51 and 53 guard matrix dimensions before
indexed access. Line 58 checks every complete-unitary entry. S18 requires the
complete-unitary oracle. In particular, lines 51 and 53 remain necessary safety
checks; they are not redundant value assertions.

## Executed evidence

The executors used clean detached worktrees at the pinned baseline. For release
runs, the reproducible command sequence is:

```sh
cmake --preset release
cmake --build --preset release --target \
  mqt-core-mlir-unittests-dialect-utils
./build/release/mlir/unittests/Dialect/Utils/\
mqt-core-mlir-unittests-dialect-utils \
  --gtest_filter='GlobalPhaseNormalizationTest.*'
```

For a mutation, apply only its diff, rebuild the same target, run the same
filter, then restore the diff before the next mutation. The executors also ran
the discovered CTest cases when collecting failure lists.

**B0, release baseline.** The focused run passed 27 of 27 tests.

**C0, fresh coverage.** Coverage used the coverage preset and `gcovr` resolved
from `PATH`, never a tool-cache path:

```sh
cmake --preset coverage
find build/coverage -name '*.gcda' -delete
.agent/audit-probe.sh t1 --lang cpp \
  --source mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp \
  --target mqt-core-mlir-unittests-dialect-utils \
  --ctest '^GlobalPhaseNormalizationTest\.'
gcovr --root . \
  --filter mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp \
  --print-summary
```

Fresh coverage data were deleted before C0. The result was 87.0% line coverage,
241 of 277 instrumented lines.

**M1, add `2*pi` to constant materialization.** The relevant fault was:

```diff
+#include <numbers>
+
- return utils::constantFromScalar(rewriter, loc,
-                                  utils::normalizeAngle(*constant));
+ return utils::constantFromScalar(
+     rewriter, loc,
+     utils::normalizeAngle(*constant) + 2.0 * std::numbers::pi);
```

Only lines 123, 163, and 207 failed. Matrices and the verifier passed.

**M2, reverse dynamic addition operands.**

```diff
- stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs));
+ stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, rhs, lhs));
```

Only lines 270-271 failed. Lines 269 and 280 passed.

**M3, retain a neutral `+0` around each collected phase.** The executor wrapped
each dynamic leaf in `arith.addf(value, 0.0)` and disabled the
already-normalized dynamic fast return so the second run exercised the
representation again:

```diff
- stack.push_back(*value);
+ auto zero = utils::constantFromScalar(rewriter, loc, 0.0);
+ stack.push_back(rewriter.create<arith::AddFOp>(loc, *value, zero));
...
- if (!constant ||
-     (utils::normalizeAngle(*constant) == *constant && *constant != 0.0)) {
+ if (constant && utils::normalizeAngle(*constant) == *constant &&
+     *constant != 0.0) {
    return std::nullopt;
  }
```

Only lines 270-271 and 280 failed.

**M4, retain a zero aggregate instead of erasing it.**

```diff
- if (aggregate->expression.isZero()) {
-   return std::nullopt;
- }
+ // Fault injection: materialize a zero aggregate.
```

Only line 827 failed.

**M5, emit 10001 for the Mul-derived normalized result.** The executor inserted
this fault before the normal constant return:

```diff
+ if (*constant == utils::normalizeAngle(12000.0)) {
+   return utils::constantFromScalar(rewriter, loc, 10001.0);
+ }
```

The target built. The affected test stopped at module verification on line 154,
before line 164. The independent verifier test passed.

**M6, leave a zero-control phase local.** The QCO control path normalized the
zero-control body as its own region and returned no contribution:

```diff
+ if (op.getNumControls() == 0) {
+   normalizeRegion(op->getRegion(0));
+   return std::nullopt;
+ }
  auto phase = normalizeBlock(*op.getBody(), op);
- if (!phase || op.getNumControls() == 0) {
+ if (!phase) {
    return phase;
  }
```

Only lines 653-654 failed. Full-unitary comparison and verification passed.

**M7, choose the first symmetric control as target.** The result vector was
restored to original-input order:

```diff
- rewriter, phase->loc, ValueRange(oldControls).drop_back(),
- oldControls.back(), [&](Value target) {
+ rewriter, phase->loc, ValueRange(oldControls).drop_front(),
+ oldControls.front(), [&](Value target) {
    return qco::POp::create(rewriter, phase->loc, target, angle)
        .getOutputTarget(0);
  });
- llvm::append_range(newControls, relative.getOutputQubits());
+ newControls.push_back(relative.getOutputTarget(0));
+ llvm::append_range(newControls, relative.getOutputControls());
```

Only lines 536-538 failed. Line 539, verification, and all other matrix tests
passed. The reordered fixture itself has no matrix oracle.

**M8, return the specific `0.75` result as `arith.addf(0.25, 0.5)`.** The fault
kept that constant pair symbolic, used a non-folding `AddFOp`, and returned that
result rather than re-folding it:

```diff
- if (lhs && rhs) {
+ if (lhs && rhs && !(*lhs == 0.25 && *rhs == 0.5)) {
...
- stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs));
+ stack.push_back(rewriter.create<arith::AddFOp>(loc, lhs, rhs));
...
- if (const auto constant = utils::valueToConstantDouble(result)) {
+ if (const auto constant = utils::valueToConstantDouble(result);
+     constant && *constant != 0.75) {
```

The target built. `CombinesQCOConstantsAtBlockExit` alone received `SIGSEGV` at
test line 121 in `ConstantOp::getValue` due to the unchecked direct-constant
cast. The other 26 tests passed.

**M9, materialize addition as `lhs - (-rhs)`.**

```diff
- stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs));
+ const auto negRhs = rewriter.createOrFold<arith::NegFOp>(loc, rhs);
+ stack.push_back(rewriter.createOrFold<arith::SubFOp>(loc, lhs, negRhs));
```

Exactly line 269 failed. The other 26 tests passed.

**M10, emit 10001 for the SIToFP-derived normalized result.**

```diff
+ if (*constant == utils::normalizeAngle(24000.0)) {
+   return utils::constantFromScalar(rewriter, loc, 10001.0);
+ }
```

Only that test failed, at module verification on line 198. Line 208 never ran.
The other 26 tests passed.

**B1, final restored focused coverage run.** After restoring all mutations, the
focused coverage run passed 27 of 27 tests and the worktree was clean.

Some mutation runs printed a non-fatal gcov checksum warning because rebuilt
objects met older coverage data. C0 deleted coverage data first and is the
authoritative coverage number. The audit-probe CTest failure-count parser also
reported zero for some failing GoogleTest runs because its output pattern did
not match. Direct binary and CTest exit status and the failure lists above are
authoritative. This is probe-tooling debt, not a test verdict.

## Provenance

[Commit `1f9fd7eb`](https://github.com/munich-quantum-toolkit/core/commit/1f9fd7ebfa085bd57e7224caac8ab3a73df36981)
from [PR `#1986`](https://github.com/munich-quantum-toolkit/core/pull/1986)
co-introduced the implementation, focused test file, `ExactUnitaryTest.h`, and
the plan. AI-assistance trailers exist in the commit. Blame shows that the exact
representation, order, and placement details at lines 122-123, 269-271, 536-538,
653-654, and 280 were born there and remain unchanged.

[Commit `696f13d6`](https://github.com/munich-quantum-toolkit/core/commit/696f13d6a8faaef80587daf08238f623a4474373)
from [PR `#1995`](https://github.com/munich-quantum-toolkit/core/pull/1995)
added recursive constant folding and both derived-angle tests to fix real
later-folding and verifier failures. Preserve the defect regression: a direct
verifier-valid constant and the correct aggregate modulo `2*pi`.

[Commit `157668c80`](https://github.com/munich-quantum-toolkit/core/commit/157668c8064d8aec51da35c54a84c453fc4600bb)
from [PR `#2006`](https://github.com/munich-quantum-toolkit/core/pull/2006)
removed the flaky nested-power scaling test. S4's linearity promise now has no
focused performance assertion. This is residual performance risk, not
Coverage-driven debt.

## Unlock and architecture analysis

No production deletion is earned. Keep `PhaseExpression` and the recursive scope
traversal. Deferred materialization and the scope walk protect S3 and S4 and
avoid the previously observed quadratic nested-modifier path
(`mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp:55-58`,
`mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp:255-308`,
`mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp:373-450`).

Verdict 3 removes one representation constraint. It permits a runtime
validity-preserving sum without first changing a test that requires one
`arith.addf` root in encounter order. This is a correctness-enabling freedom,
not proven source deletion.

The following are ordinary follow-ups, not audit unlocks:

- the idempotence fast path may be simpler, but no experiment ties it to line
  280;
- shared constant-fold or hoist caches are benchmark-driven optimizations, and
  no audited assertion blocks them.

## Red-team revisions and residual risks

1. **High-priority S5 correctness follow-up.** Runtime addition of two
   individually valid dynamic angles can exceed 10000. Integral-power scaling
   can also exceed 10000 or overflow for a large finite integral exponent. The
   current pass normalizes only values proven constant. Verdict 3 is the reason
   a runtime validity representation can now be considered without breaking an
   incidental SSA-shape test. This audit does not implement it.

2. **Mixed-dialect design risk.** Mixed QC and QCO contributions in one block
   are undocumented. A debug build asserts; a release build follows the first
   contribution's dialect
   (`mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp:184-195`,
   `mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp:381-448`). This
   is a residual design risk, not a current verdict or a promise of rejection.

3. **Performance risk.** S4 promises linear traversal, but the focused scaling
   oracle was removed by PR `#2006` after it proved flaky. No stable focused
   performance oracle replaced it.

4. **Pipeline ownership risk.** Zero-control placement spans normalization,
   `CtrlOp` canonicalization, and QIR conversion order. Verdict 7 removes an
   unsupported placement promise but does not choose the owning layer.

5. **Downstream drift risk.** Re-run PRs `#2062`, `#2080`, and `#2150` when a
   selected remedy changes numeric or dynamic expression representation.

## Deliberately not touched

- Production code and tests were not edited.
- Line 827 remains the requested exact-cancellation regression.
- All five `ExactUnitaryTest.h` assertions remain, including the dimension
  guards at lines 51 and 53.
- The 100 focused-file Anchored sites remain individually recorded above.
- No remote state changed.

## Progress

- [x] (2026-08-19) Pin and verify the clean detached baseline.
- [x] (2026-08-19) Complete four valid isolated spec-cartography waves.
- [x] (2026-08-19) Discard the contaminated requested-behavior cartography.
- [x] (2026-08-19) Census all 120 direct and transitive assertion sites.
- [x] (2026-08-19) Complete two independent prosecutions and provenance.
- [x] (2026-08-19) Complete defence and serialized mutation execution.
- [x] (2026-08-19) Complete supplemental fresh execution.
- [x] (2026-08-19) Complete unlock and architecture analysis.
- [x] (2026-08-19) Complete fresh red team and incorporate its revisions.
- [x] (2026-08-19) Complete fresh final editorial reconciliation.

All audit waves are complete. The human decision is recorded above. Status:
verdicts 1-6 selected for resolution, verdict 7 deferred, and verdicts 8-9
retained. The audit stops at the ledger. The selected remedies land as separate
commits in one later pull request.
