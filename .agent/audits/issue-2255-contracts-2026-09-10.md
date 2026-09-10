🤖 *AI text below* 🤖 <!-- rumdl-disable-line MD041 -->

# MLIR contract and simplification audit — issue [#2255]

Status: awaiting maintainer triage; findings proposed, not accepted or
implemented. Date: 2026-09-10. Baseline:
`d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1` (main, including [#2478], [#2477],
[#2498], [#2499], and [#2500]). No in-scope tracked modifications.

Scope:
[issue #2255](https://github.com/munich-quantum-toolkit/core/issues/2255), its
parent [#2250], and the repository's MLIR development and audit policies. This
report requests scope decisions before implementation. For each finding, please
indicate address, defer, or reject; no proposed fix is pre-approved.

The report-only branch is based on `4faf68e3ae5b31bc5dbbd434e5e3ce5795096088`.
Runtime evidence remains tied to the tested baseline above; source line numbers
also refer to that baseline.

## Result

The strongest findings are semantic or contract defects, not opportunities to
add blanket IR verification. Prioritize these small, independently reviewable
fix families:

1. Preserve classical side-effect order across mapping.
2. Preserve modifier wire correspondence in standalone QCO-to-jeff conversion.
3. Stabilize XX±YY matrix construction for large finite angles.
4. Remove the assumption that measured qubits always have defining operations.
5. Ensure full loop unrolling actually makes progress.
6. Enforce mapping's documented flat tensor-chain subset before placement.
7. Infer QIR resource capacity from operand roles, not the first producer use.
8. Make QIR loop classification total and include the latch.
9. Align attributed `qco.index_switch` parsing and printing.
10. Complete modifier signature checks in the owning verifiers.

Separately, three worthwhile cleanup families remove a redundant conversion
walk, stale QTensor utilities, and non-local child verifier checks. A fourth
small allocation reduction remains optional. No speedup was measured or is
claimed. Existing tests passing did not catch the new counterexamples.

## Ranked findings

### 1. [P1, reproduced] Mapping can reverse observable classical calls

**Where:** `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`,
`validateRoutingOperations` at line 152 and sorting calls at lines 640 and 1699;
`mlir/lib/Dialect/QCO/Utils/Sorting.cpp:53`.

The validation permits classical `func.call` operations without quantum
operands. The sorter deliberately does not order unknown or value-less effects
(its header explicitly says so). Mapping nevertheless invokes it on these
operations. A valid two-qubit input containing `@first(%computed)` followed by
`@second()` returns successfully with `@second()` before `@first`. Both input
and output pass ordinary verification and QCO linearity checks.

**Evidence:** `issue-2255-evidence/inputs/mapping-side-effects.mlir`, mode
`mapping`, exit 0; the raw output records the reversed calls. The direct helper
probe independently produces the same ordering.

**Smallest direction:** define and enforce the supported classical-effect subset
at mapping's preflight, or conservatively preserve ordering around unknown
effects. Do not build general alias analysis merely to retain a narrow mapping
contract. Preserve current deterministic scheduling and CBit effect ordering.
This establishes a mapping-pass defect; the complete target pipeline was not
tested with this input and may reject surviving external calls later.

### 2. [P1, reproduced] QCO-to-jeff drops modifier wire correspondence

**Where:** `mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp`, `handleResult` near
line 261, inverse conversion near line 1620, and yield conversion near 1722.

The conversion stores the inner gate's results as all modifier outputs instead
of preserving the body's yielded SSA mapping. A valid, linear two-target inverse
containing S on the first target and an untouched second target aborts with
`incorrect # of replacement values`. A two-target inverse SWAP yielding its
results in reversed order succeeds, but a following measurement uses the first
inner SWAP result rather than the yielded second result.

**Evidence:** `jeff-inverse-unused-wire.mlir` (exit -6) and
`jeff-inverse-permuted-yields.mlir` (exit 0 with incorrect output operand),
under `issue-2255-evidence/inputs/`, mode `qco-jeff`.

**Smallest direction:** preserve actual yield correspondence, or diagnose the
unsupported normal form before inlining and require the existing modifier
unroller. Do not change the QCO dialect to outlaw otherwise valid bodies. Cover
untouched targets, permutations, and nested inv/pow/ctrl interactions. The
public default `intoJeff` pipeline unrolls modifiers first; the demonstrated
failure is in the separately exposed conversion pass.

### 3. [P1, reproduced] XX±YY factories produce non-unitary matrices

**Where:** `mlir/lib/Dialect/QCO/IR/Operations/StandardGates/XXPlusYYOp.cpp:62`
and `XXMinusYYOp.cpp:62`.

The off-diagonal formula embeds `beta - pi/2` and `-beta - pi/2` in separate
complex exponentials. At the valid finite value beta = 1e16, subtracting pi/2
rounds independently and destroys the required phase relationship. Calling the
actual linked factories with theta = 1 gives a maximum entrywise error in
`U * U.adjoint() - I` of **0.3501755** for each gate.

**Evidence:** `issue-2255-evidence/inputs/large-beta-xx-plus-minus.mlir`, mode
`large-beta`, exit 0 with two `MAX_UNITARITY_ERROR=3.501755e-01` records. The
driver calls production C++ factories; this is not merely a copied Python
formula. DD adapters and other matrix consumers share these factories.

**Smallest direction:** factor out `-i`, evaluate `exp(i*beta)` once, and obtain
the partner by conjugation with the appropriate sign. Preserve the established
phase convention and add unitarity checks across large finite values. This also
removes one exponential evaluation, but there is no timing result. The earlier
[#2477] correction to R/U/U2 does not cover these gates.

### 4. [P1, reproduced] Two lifting passes assume a defining operation exists

**Where:**
`mlir/lib/Dialect/QCO/Transforms/Optimizations/MeasurementLifting.cpp` at lines
96, 139, 177 and `HadamardLifting.cpp` at lines 147, 155.

A function may directly measure a qubit block argument. Both passes use
`dyn_cast` on the nullable result of `getDefiningOp()`. The verified linear
input aborts in LLVM Casting.h with `dyn_cast on a non-existent value`.

**Evidence:** `issue-2255-evidence/inputs/measure-block-argument.mlir`, modes
`measurement-lifting` and `hadamard-lifting`, both exit -6.

**Smallest direction:** use `Value::getDefiningOp<ExpectedOp>()` in the matching
code. This expresses the producer requirement directly and removes the brittle
assumption; it is not a reason to add module verification before each pass.

### 5. [P1, reproduced] Full quantum unrolling can retry forever

**Where:**
`mlir/lib/Dialect/QCO/Transforms/Optimizations/QuantumLoopUnroll.cpp`, lines
109–121.

A three-iteration loop whose body contains only a yield that reverses two qubits
is valid and linear. It is not covered by the identity-only special case. MLIR's
full unroller returns success without changing this empty body, and the wrapper
unconditionally sets `changed = true`, repeating forever.

**Evidence:** `issue-2255-evidence/inputs/quantum-yield-permutation.mlir`, mode
`unroll-full`, killed by the diagnostic runner's five-second timeout after
`INPUT_VERIFIED_AND_LINEAR`. Source analysis establishes the non-progressing
cycle; the timeout alone is not the proof. Existing permutation coverage uses
only one iteration.

**Smallest direction:** share the existing empty-body handling used by payload
control-flow legalization, or explicitly compose the yield permutation. Keep
zero-trip, identity, and odd/even permutation semantics. Do not equate a
utility's success result with actual progress.

### 6. [P1, reproduced] Placement erases a still-used tensor allocation

**Where:** `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`,
`discoverComputation` at lines 193–240 and `applyPlacement` at line 311.

Discovery promises to reject inputs outside its flat allocate/extract/compute/
insert/deallocate structure without mutation. It instead traverses the more
general `TensorIterator` and records control-flow operations it will not later
replace. A tensor passed through a valid `qco.if` reaches allocation erasure
while that operation still uses it.

**Evidence:** `issue-2255-evidence/inputs/mapping-tensor-conditional.mlir`, mode
`mapping`, exit -6: `expected 'op' to have no uses`, after input verification
and linearity succeeded.

**Smallest direction:** directly enforce the documented flat chain during
discovery and diagnose unsupported users before mutation. A restricted
one-use-chain traversal may replace the overly general iterator here. Do not add
tensor-region lowering or a late assertion solely for this fix. These are valid
QCO programs outside the mapping subset, not invalid-IR pass tests.

### 7. [P1, reproduced] QIR capacity depends on producer/use-list shape

**Where:** `mlir/lib/Dialect/QIR/Transforms/AttachQIRAttributes.cpp:283–319`.

Qubit counting chooses only the first integer-to-pointer user of an ID constant,
then treats most `__quantum__qis` calls as qubit uses. Two verified examples
expose both directions of the error:

| Input                                                   | Emitted capacities (qubits/results) | Required capacities |
| ------------------------------------------------------- | ----------------------------------- | ------------------- |
| ID 7 shared by separate qubit/result pointer operations | 0 / 8                               | 8 / 8               |
| Measure qubit 0 into result 7, then read result 7       | 8 / 8                               | 1 / 8               |

**Evidence:** `issue-2255-evidence/inputs/qir-shared-static-id.mlir`, mode
`attach-base`; `qir-result-not-qubit.mlir`, mode `attach-adaptive`. Both exit 0
with verified output and the incorrect metadata above.

**Smallest direction:** infer resource roles from the known QIR call operands,
as result counting already does. Iterating every `inttoptr` alone does not fix
the result-read misclassification. Preserve sparse-ID capacity and overflow
handling. No resource-runtime execution was performed.

### 8. [P2, source-confirmed] QIR loop classification is not total

**Where:** `mlir/lib/Dialect/QIR/Transforms/AttachQIRAttributes.cpp:368–403` and
`420–430`.

Natural-loop construction initially includes the header but pushes the latch
without inserting it. A simple header-to-conditional-latch loop therefore loses
its only condition. `classifyLoop` also reaches the end of a bool-returning
function when no conditional branch exists. Finally, selecting the first
conditional in a pointer set is not a defined classification of all exits.

**Evidence:** the unconditional and conditional-latch probes both verify and
return successfully in this build, emitting `backwards_branching = 1`. No crash
or nondeterminism was observed. The missing return and omitted latch are direct
source evidence, independent of that observed result.

**Smallest direction:** make the classification rule total and deterministic,
include the latch, and consider MLIR's existing loop analysis instead of
maintaining bespoke discovery. An unconditional loop and a measurement-based
exit need explicit semantics. Do not claim sanitizer confirmation: none ran.

### 9. [P2, reproduced] Attributed index switches do not round-trip

**Where:** `mlir/lib/Dialect/QCO/IR/QCOOps.cpp:285` and `460`.

The parser consumes a bare optional attribute dictionary while the printer emits
the `attributes` keyword. A valid linear switch with `{audit.marker}` prints
successfully, then fails to parse at that keyword (`expected 'default'`).

**Evidence:** `issue-2255-evidence/inputs/switch-attribute.mlir`, mode
`roundtrip`, exit 5.

**Smallest direction:** align parser and printer syntax with a round-trip
regression containing a discardable attribute. No new abstraction is needed.

### 10. [P2, reproduced] Modifier signatures are incompletely verified

**Where:** `mlir/lib/Dialect/QC/IR/Modifiers/ModifierUtils.cpp:32`, the QC
Ctrl/Inv/Pow verifiers, and `mlir/lib/Dialect/QCO/IR/Modifiers/CtrlOp.cpp:289`.

Generic QC inverse syntax with one operand and two body arguments is accepted. A
QCO control operation with one input control but no output control is also
accepted by both normal verification and the separate linearity check.

**Evidence:** `qc-modifier-extra-argument.mlir` and
`qco-control-missing-output.mlir` under `issue-2255-evidence/inputs/`, mode
`verify`, both exit 0.

**Smallest direction:** establish body argument count/type correspondence and
control input/output arity in the owning operation verifiers. These examples are
semantically invalid IR. They must not be counted as valid-input pass bugs or
motivate repeated defensive checks in downstream passes.

## Simplifications worth keeping separate from correctness fixes

### Remove QC-to-QCO's duplicate modifier-validation walk

`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp:684–694` performs an extra module walk
re-verifying QC Ctrl/Inv/Pow before conversion; its sole call is near line 1994.
The `TypeSwitch` include is used only by this helper. Valid-input conversion
does not need a second operation-verification layer.

Three conversion-test families near lines 1401, 1501, and 1583 of
`test_qc_to_qco.cpp` disable verification and build invalid QC programs. Their
builders and cases occupy roughly 245 lines. Removing the duplicate walk should
move any unique register-backed capture coverage to the QC verifier suite, where
forbidden body/capture matrices already exist. Do not blindly delete the entire
test region: check the retained input classes first.

Benefit: one full module traversal and roughly a dozen production lines removed,
plus potentially substantial test consolidation. Timing benefit unmeasured.

### Retire the stale QTensor index/chain helper header

`mlir/include/mqt/Dialect/QTensor/IR/QTensorUtils.h` contains 75 lines with no
production includes or callers on this baseline. Four chain helpers have no
callers; `areEquivalentIndices` survives only in three direct helper tests. The
last production index-helper use was removed by [#2477]'s allocation lookup
rewrite.

The behavior-level QTensor tests for equivalent constant/dynamic indices and
reset provenance must remain. Header plus obsolete direct tests/include amount
to roughly 100 lines, not a measured performance result. The header is in the
installed include tree: confirm the unreleased API removal is acceptable;
absence of in-tree callers does not prove absence of external users.

### Let the modifier verifier own forbidden QTensor bodies

`mlir/lib/Dialect/QTensor/IR/Operations/ExtractOp.cpp:142–146` and
`InsertOp.cpp:79–83` inspect ancestors to reject use inside modifiers. The
owning QCO modifier verifier already rejects these operations and nested
regions. These child checks duplicate policy and make verification non-local.

Remove the duplicate ancestor checks (roughly a dozen lines plus now-unused
includes). Change the test that verifies only a child inside invalid modifier IR
to verify the owner/module instead. Retain local extract/insert bounds checks
and their tests. This reduces invalid-IR checking inside unrelated ops without
weakening the module contract.

### Optional: remove redundant ShrinkRegisters bookkeeping

`mlir/lib/Dialect/QTensor/Transforms/ShrinkRegisters.cpp:125–149` constructs a
second mapped-index vector after collecting all accesses into the live-index set
and building the complete mapping. Successful collection already proves the
deallocation and lookup invariants.

Direct lookups during emission can remove the redundant vector and impossible
failure branches, saving O(number of accesses) temporary storage. Keep planning
before mutation and existing alias/index behavior. This is source-proven modest
allocation reduction, not a benchmarked speedup or a release blocker.

## Explicit support decisions and unresolved candidates

- **QC-to-QIR function subset:** Base `ensureBlocks` near lines 411–448,
  Adaptive near 654–674, and shared result-pointer handling assume an
  argument-free entry and inlined measurement-bearing helpers. Verified entry
  arguments and helper measurements produce verifier-detected conversion failure
  (four probes, exit 6), not successful wrong output. Prefer a clear
  supported-subset diagnostic or forwarding supported entry arguments over
  inventing per-function resource management. The default pipeline inlines
  helpers; do not reopen previously closed broad QIR resource-management work.
- **Unregistered operations at program import:** `Programs.cpp:134` dereferences
  `operation->getDialect()` in `moduleUsesDialect`. With an explicitly
  permissive caller-owned context, a verified unknown operation causes both
  QC/QCO `fromModule` probes to exit -11. Default parsing does not allow this.
  Decide whether that external-context usage is supported; if so, inspect the
  operation-name namespace or diagnose at import. Do not present this as a
  default compiler-input failure.
- **Hadamard controlled-X shape:** the matcher near
  `HadamardLifting.cpp:159–183` does not state its nonzero-control and
  single-target requirements. The zero-control and multi-target diagnostic
  inputs in this audit have a custom-syntax error and were rejected before the
  pass. This remains source-level follow-up, not a reproduced finding. Next
  check: correct those inputs and verify observable measurement semantics.
- **Controlled-SWAP yielded permutation:**
  `DecomposeMultiControlled.cpp:1219–1224` substitutes CSWAP based on the sole
  inner gate without an explicit positional-yield check. Existing modifier
  helpers/tests preserve non-positional yield semantics. The supplied probe also
  has a syntax error, so this is not included in the confirmed findings. Next
  check: corrected valid input and an independent wire-order oracle; production
  DD construction intentionally has narrower modifier support.
- **QCO-to-QC multi-block linear arguments:** a probe fails with unresolved
  materialization (exit 6). This establishes a support gap only; no crash or
  silently invalid result was observed.

## Rejected or intentionally retained candidates

- No reintroduction of guards for nonlinear/otherwise invalid QCO IR in passes.
  Invalid modifier signatures belong in verifiers, as above.
- No blanket deletion of traits, arity tests, phase corrections, wire-order
  handling, or exact-output tests. Their semantic consumers still matter.
- No finding for QuantumLoopUnroll omitting a direct Arith dependency: the QCO
  dialect loads Arith transitively.
- No replacement of the custom sorter with generic sorting without measuring the
  known performance tradeoff and preserving effect ordering.
- No removal of recent decomposition caches, sparse register plans, or the
  measured-qubit routing cache: their complexity or correctness purposes remain.
- No unbounded-recursion finding for phase-hoisting or Pauli twirling without a
  concrete non-progressing rewrite. Prior closed feedback remains relevant.
- No mass test deletion or new timing claim inferred from line counts.

## Coverage and related work

Reviewed the transform implementations across QCO, QC, QTensor, QIR, and MQT;
all eight conversion implementation families; QC/QCO/QTensor/CBit/MQT IR
contracts, modifiers and standard-gate factories; relevant TableGen pass
dependencies, callers and tests; compiler program/pipeline/target boundaries;
and shared sorting, wire/tensor traversal, constant, and rewrite utilities.
Depth varied: targeted contract and semantic analysis, not a formal proof or
fresh numerical derivation of every decomposition algorithm.

Historical overlap checked: merged [#2290], [#2291], [#2293]–[#2296],
[#2300]–[#2302], [#2304], [#2307], [#2308], [#2320], [#2322]; closed [#2287],
[#2303], [#2305], [#2306], [#2309], [#2318], [#2319], [#2321]. [#2501]
(notation) is now merged. Current adjacent work includes [#2495]
(device-directed compilation), [#2280] (constant propagation), [#2260] (QC
inspection), [#2196]–[#2201] (function/interprocedural infrastructure), [#2032]
(CUDA-Q interop), and [#1955] (mapping benchmarks). This is the status checked
during the audit, not a guarantee against subsequent changes.

## Validation and reproducibility

[Evidence and reproduction instructions](issue-2255-evidence/README.md),
[19 reduced inputs](issue-2255-evidence/inputs/), and
[23 recorded probe outcomes](issue-2255-evidence/results-d994fe683.json)
are included in this PR. The JSON records contain exit codes, stdout, stderr,
and timeout flags; only local path prefixes were normalized. Build artifacts,
personal paths, full generated test logs, and syntax-invalid exploratory inputs
are not included.

Toolchain: AppleClang 21, LLVM/MLIR 23.1.0, Release with project-enabled
assertions, thin LTO, arm64. Pinned dependency source trees were reused from an
earlier checkout; dependency definitions were checked unchanged. Built `mqt-cc`
and the following four test targets at `d994fe683`. All **1,173 tests passed**:

| Test executable                        | Tests | Failures |
| -------------------------------------- | ----: | -------: |
| `mqt-core-mlir-unittest-qco-utils`     |   192 |        0 |
| `mqt-core-mlir-unittest-optimizations` |   198 |        0 |
| `mqt-core-mlir-unittest-qco-ir`        |   579 |        0 |
| `mqt-core-mlir-unittests-compiler`     |   204 |        0 |

These are selected existing suites, not the complete repository test matrix.
Sanitizers, broad fuzzing, and new performance benchmarks were not run.

The diagnostic driver verifies the module and QCO linearity before executing the
selected operation. For verifier-omission inputs, passing those checks is the
defect, not evidence of semantic validity. Negative exit codes are signals, not
successful tests. Zero exit codes require checking the output semantics. The
runner records raw outcomes and does not infer verdicts from text.

The reduced publication harness and current-main validation are recorded in the
PR description. Historical output is never overwritten by reproduction.

No production source, regular test target, dependency, or build configuration is
changed by this PR. Accepted behavior changes should receive focused regressions
in the owning test suite. Unresolved candidates do not count as confirmed
findings.

[#1955]: https://github.com/munich-quantum-toolkit/core/pull/1955
[#2032]: https://github.com/munich-quantum-toolkit/core/pull/2032
[#2196]: https://github.com/munich-quantum-toolkit/core/pull/2196
[#2201]: https://github.com/munich-quantum-toolkit/core/pull/2201
[#2250]: https://github.com/munich-quantum-toolkit/core/issues/2250
[#2255]: https://github.com/munich-quantum-toolkit/core/issues/2255
[#2260]: https://github.com/munich-quantum-toolkit/core/pull/2260
[#2280]: https://github.com/munich-quantum-toolkit/core/pull/2280
[#2287]: https://github.com/munich-quantum-toolkit/core/pull/2287
[#2290]: https://github.com/munich-quantum-toolkit/core/pull/2290
[#2291]: https://github.com/munich-quantum-toolkit/core/pull/2291
[#2293]: https://github.com/munich-quantum-toolkit/core/pull/2293
[#2296]: https://github.com/munich-quantum-toolkit/core/pull/2296
[#2300]: https://github.com/munich-quantum-toolkit/core/pull/2300
[#2302]: https://github.com/munich-quantum-toolkit/core/pull/2302
[#2303]: https://github.com/munich-quantum-toolkit/core/pull/2303
[#2304]: https://github.com/munich-quantum-toolkit/core/pull/2304
[#2305]: https://github.com/munich-quantum-toolkit/core/pull/2305
[#2306]: https://github.com/munich-quantum-toolkit/core/pull/2306
[#2307]: https://github.com/munich-quantum-toolkit/core/pull/2307
[#2308]: https://github.com/munich-quantum-toolkit/core/pull/2308
[#2309]: https://github.com/munich-quantum-toolkit/core/pull/2309
[#2318]: https://github.com/munich-quantum-toolkit/core/pull/2318
[#2319]: https://github.com/munich-quantum-toolkit/core/pull/2319
[#2320]: https://github.com/munich-quantum-toolkit/core/pull/2320
[#2321]: https://github.com/munich-quantum-toolkit/core/pull/2321
[#2322]: https://github.com/munich-quantum-toolkit/core/pull/2322
[#2477]: https://github.com/munich-quantum-toolkit/core/pull/2477
[#2478]: https://github.com/munich-quantum-toolkit/core/pull/2478
[#2495]: https://github.com/munich-quantum-toolkit/core/pull/2495
[#2498]: https://github.com/munich-quantum-toolkit/core/pull/2498
[#2499]: https://github.com/munich-quantum-toolkit/core/pull/2499
[#2500]: https://github.com/munich-quantum-toolkit/core/pull/2500
[#2501]: https://github.com/munich-quantum-toolkit/core/pull/2501
