# Contract audit: global-phase normalization

Status: historical audit and decision record. Baseline:
`cb5cf0103bd9841726c8ec6c5abb725758afea58` (2026-08-19). The maintainer accepted
findings 1-6 on 2026-08-20, deferred 7, and retained 8-9. Acceptance is recorded
here; implementation and present-day status were not revalidated during document
cleanup.

Source: `mlir/lib/Dialect/Utils/Transforms/NormalizeGlobalPhases.cpp`. Tests:
`mlir/unittests/Dialect/Utils/test_global_phase_normalization.cpp` and
`mlir/unittests/ExactUnitaryTest.h`. All line references and experiment results
below refer to the baseline, not the current checkout.

## Result

The audit supported narrower representation checks while retaining full-unitary,
verification, wire-order, and reported-defect coverage. It did not demonstrate a
safe production deletion or a performance improvement.

| ID  | Decision                                                                                  | Disposition |
| --- | ----------------------------------------------------------------------------------------- | ----------- |
| 1   | Safely inspect foldable phase values instead of assuming a constant producer              | Accepted    |
| 2   | Compare phase representatives modulo `2*pi`, retaining direct-constant regressions        | Accepted    |
| 3   | Test both dynamic inputs' contribution instead of one SSA tree                            | Accepted    |
| 4   | Test input/output wire identity instead of one symmetric control target                   | Accepted    |
| 5   | Remove two validity checks already enforced by earlier verification of the same constants | Accepted    |
| 6   | Replace printed equality with semantic repeat-run checks                                  | Accepted    |
| 7   | Reconsider zero-control phase placement only after resolving QIR pipeline ownership       | Deferred    |
| 8   | Keep exact cancellation of opposite phases                                                | Retained    |
| 9   | Keep full-unitary helper checks, including matrix dimensions                              | Retained    |

## Contract and reasoning

`mlir/include/mlir/Dialect/Utils/Transforms/Passes.td` describes block-local
normalization, independent nested scopes, modifier extraction, and linear
traversal. QC/QCO gate definitions and their custom verifiers require finite
phase angles with magnitude at most 10000; dynamic inputs have the same runtime
precondition. A transformed program must retain the complete unitary and valid
QCO linear wiring. Operation declarations alone do not establish every pipeline
postcondition.

The
[OpenQASM gate semantics](https://openqasm.com/versions/3.1/language/gates.html#gate-gphase)
and [issue `#1641`](https://github.com/munich-quantum-toolkit/core/issues/1641)
support phase addition and exact cancellation. They do not choose one numeric
representative or symmetric control target.

Findings 1-2 concern `CombinesQCOConstantsAtBlockExit` and the two
`Folds*DerivedPhasesWithinPracticalAngleLimit` tests. M1 and M8 below exposed
representation assumptions, including an unchecked failed `ConstantOp` lookup.
The derived-angle tests reproduce the later-folding/verifier defect fixed by
[PR `#1995`](https://github.com/munich-quantum-toolkit/core/pull/1995).
They must still require a direct verifier-valid constant and the correct value
modulo `2*pi`; only the first test may accept another foldable expression.

Finding 3 concerns `PreservesDynamicOrderAndIsIdempotent`. M2 and M9 changed
operand order or the root opcode without losing either input. A replacement
oracle must detect omission of either input. A dependency check alone cannot
establish arithmetic correctness for arbitrary replacement expressions.

Finding 4 concerns `ReorderedQCOControlsThreadCorrectResults`. M7 changed the
symmetric relative-phase target and restored output order. The original fixture
had no matrix oracle, so relaxing target checks requires a wire-identity oracle
or distinguishable downstream uses and complete-unitary comparison.

Finding 5 is supported by the specific verifier implication, not merely two
tests catching one fault. M5 and M10 failed at whole-module verification before
the later validity assertions for those same direct constants could run. The
independent angle-boundary verifier tests remain.

Finding 6 concerns byte equality after a second normalization. M3 showed
sensitivity to a neutral expression, but also disabled a fast path and could
increase IR on repeated runs. It does not justify losing convergence or bounded
growth checks, or deleting the production fast path. Keep verification, phase
count, semantics, and a suitable repeat-run bound.

Finding 7 remains an ownership question. M6 preserved the unitary while leaving
a zero-control phase nested. However, direct QIR conversion normalizes before
conversion and rejects nested global phases. Check the QC/QCO `CtrlOp`
canonicalizers and QIR conversion order before changing placement requirements.
A locally equivalent unitary is not sufficient proof for this pipeline change.

M4 confirmed that `ExactSpecialConstantsCancelWithoutTolerance` catches failure
to erase a zero aggregate. The five checks in `ExactUnitaryTest.h` guard matrix
construction, dimensions before indexing, and every complete-unitary entry. Keep
those regression and safety checks.

## Unresolved questions at the baseline

- Dynamic addition and integral-power scaling can exceed the phase-angle bound
  even when the original angles satisfy it. This is a correctness question,
  separate from permission to change tests.
- Mixed QC/QCO contributions in one block had different debug and release
  behavior. Establish ownership and supported behavior before changing it.
- The linear-traversal requirement lacked a stable focused performance check
  after the timing test was removed in PR `#2006`.
- Changes to numeric representation need downstream checks in quantization,
  synthesis, QIR, and DD execution. Historical PR overlap lists are not current
  dependency evidence.

## Historical executed evidence

The following records the original experiments; they were not rerun for this
documentation change. Mutations are described against the baseline source and
are examples for reproducing the evidence, not proposed production patches.
Direct binary/CTest exit status and diagnostics were used: the retired helper
misparsed some failing CTest runs as zero failures. Its counts are not evidence.

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
+ /// Fault injection: materialize a zero aggregate.
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
+ auto negRhs = rewriter.createOrFold<arith::NegFOp>(loc, rhs);
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
