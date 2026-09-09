# Contract audit: remaining canonicalization fixes

Status: complete; remaining fixes implemented and validated locally. Base:
`7e2a2679fd6c48397f2d7ca5e3d2018d841a69a2`, including PR #2464. Date:
2026-09-09.

The comprehensive audit, original registration inventory, and historical
validation remain available at
[commit 79e9c347b](https://github.com/munich-quantum-toolkit/core/blob/79e9c347b30b6c6639b6422a206fdc78dba6bd16/.agent/audits/dialect-canonicalization.md).
The Arith dependency declarations moved to
[PR #2484](https://github.com/munich-quantum-toolkit/core/pull/2484), modifier
and gate wire mappings to
[PR #2485](https://github.com/munich-quantum-toolkit/core/pull/2485), and
bounded angle arithmetic to
[PR #2486](https://github.com/munich-quantum-toolkit/core/pull/2486), which
targets PR #2485. This record covers only the fixes that remain in PR #2477.

## Result and scope

The remaining changes stabilize R/U/U2 matrices and U2 inversion, simplify
bounded U-power reconstruction, replace identity rewrites with folds, and remove
redundant work in CBit, QCO If, and QTensor canonicalization. Direct pair tests
protect immediate QCO/QTensor linearity. Larger QTensor normalization and
provenance changes remain deferred.

The scope includes these canonicalizers and the shared matrix helpers that
establish their semantics. Transformation and conversion patterns remain outside
the audit and implementation scope. Their tests establish relevant consumer
contracts only. Preserve verified IR, full unitaries including global phase,
wire identity, and exactly-one-use quantum values after each rewrite. Failed
matches leave IR unchanged. The owning policy is
[MLIR development](../../docs/mlir/development.md).

## Findings and decisions

### 1. Preserve large-angle matrix semantics

Impact: high for affected inputs. Confidence: concrete numerical counterexamples
and independent matrix oracles. Disposition: implemented and validated locally.

`ROp::unitaryMatrix` formerly formed `exp(i*(+/-phi-pi/2))`. At axis angles
`1e16` and `1e308`, rounding can erase the quarter turn and produce a nonunitary
matrix. Its off-diagonal entries now use `-i*sin(theta/2)*exp(+/-i*phi)`. Direct
axis-matrix and unitarity tests cover the owning helper; R merge and power tests
compare untouched input and rewritten output.

U/U2 matrices had the same problem with `lambda+pi`, while `phi+lambda` could
absorb a phase or overflow. `computeUMatrix` now multiplies separate phase
factors and applies the off-diagonal minus sign directly. QCO U/U2 and the
shared U-power helper use this implementation. Independent
`P(phi) * RY(theta) * P(lambda)` tests cover large, cancelling, and overflowing
angle sums. `UToU2PreservesLargeEulerAngles` checks the canonicalization
consumer.

QC and QCO invert U2 as `U(-pi/2,-lambda,-phi)`. Exact sign changes avoid the
former unbounded additions and subtractions of pi. Constant and dynamic tests
retain ordinary and controlled full-unitary behavior. Their former U2-only
output assertions are replaced by untouched-reference matrix checks. Keep direct
U2 patterns: routing U2 through U would cycle with U-to-U2 canonicalization.

### 2. Extract U powers from the matrix already computed

Impact: medium maintenance benefit. Confidence: independent sequential-product
oracles. Disposition: implemented and validated locally.

`powerUParameters` extracts Euler angles and global phase from its bounded
binary matrix power. This removes a second quaternion calculation and angle
reduction. Retain finite inputs, positive integral exponents no greater than
1024, and the `5e-13` full-matrix reconstruction bound. Diagonal, anti-diagonal,
near-gimbal, and large-phase tests cover extraction boundaries. The exponent
limit does not promise that every input below it folds: reconstruction can still
exceed the error bound.

Historical sensitive U-power cases remain under control and retain full-matrix
checks. They no longer require a surviving Pow when the new extraction meets the
same error bound. The helper's error and input limits remain unchanged.

### 3. Fold only identities that can remove their own root

Impact: medium maintenance benefit. Confidence: source ownership and direct
rewrite-boundary tests. Disposition: implemented and validated locally.

QCO Id and exact QCO Unitary identity use fold hooks. Preserve Unitary's exact
matrix predicate and operand order. `-I` and small nonzero phases, including
`1e-16`, must survive; verifier tolerance does not authorize erasing phase. The
fixed-gate fusion consumer now accepts the single RX left after its Id prefix
folds, while retaining full-matrix and nontrivial-prefix fusion checks.

H/X/Y/Z and QTensor Insert pair cancellation retain two-operation rewrites.
Migrating these cases to `Involution` or a root-only fold is rejected: the fold
cannot erase the producer, so forwarding introduces a second use until later
DCE. A historical pipeline test missed that violation. Direct
`PatternApplicator` tests verify IR and linearity immediately after
cancellation, including a QTensor equal-dynamic-SSA case and an uncertain-alias
nonmatch.

### 4. Remove redundant local analysis work

Impact: low to medium. Confidence: reproduced CBit missed optimization and
source invariants. Disposition: implemented and validated locally.

- CBit load forwarding skips read-only whole-register snapshots. A store of true
  followed by a live snapshot and scalar load formerly retained the load.
  Whole-register writes, ambiguous indices, region operations, and unknown
  register users remain barriers. The unread zero-initialization boolean is
  removed; nullable `value` retains its existing meaning.
- `ForwardClassicalResults` maps ordered branch-yield pairs to their earliest
  result. Matching is expected O(N), replacing N(N-1)/2 pair comparisons for N
  distinct pairs. Output follows result order, never map traversal. Preserve
  reversed-pair distinctions, the equal-value branch case, result-segment
  updates, and the linear suffix. Both branch yields reuse the unused-result
  erase mask.
- QTensor reset provenance decodes the target constant once and each traversed
  index once. The match set remains unchanged. Keep the installed
  `areEquivalentIndices` API and its tests; a lack of other in-tree callers does
  not justify deleting a public utility.

The [evidence appendix](dialect-canonicalization-probes.md) names the permanent
regressions and distinguishes historical measurements from current validation.

## Deferred QTensor work

Adjacent insert/extract commuting requires N(N-1)/2 successful rewrites for N
distinct constant-index accesses. Baseline counts at N=16/64/256 were
120/2,016/32,640. Reset provenance also revisits quadratic predecessor chains;
decoding once reduces repeated work without changing that bound.

Mapping and QTensor branch scalarization consume the all-extracts-before-inserts
form. History #1987 records failures at the default greedy iteration limit, so
production cleanup runs to convergence. Do not remove normalization or add an
arbitrary cap without preserving that consumer contract. A batch algorithm must
preserve same-block/direct-SSA boundaries, dynamic and same-index barriers, wire
identity, and deterministic order. A provenance cache needs an invalidation
owner. Neither redesign is implemented here.

`ScalarizeQTensorInputs` retains static tensors, distinct constant indices,
complete reinsertion, and positional tensor yielding. Branch-specific accesses
and sparse extraction remain supported. Full-tensor expansion or general alias
analysis would broaden the supported subset and its cost. Keep the direct pair
and same-index barriers; nonlocal cancellation was removed in #1987 because it
bypassed structured updates.

## Validation

The narrowed diff was rebuilt with the release preset and LLVM/MLIR 23.1.0 on
2026-09-09. These results apply to the remaining changes, independently of the
combined change at `79e9c347b`:

- The six focused binaries passed 1,150 tests: QC 350, QCO 525, CBit 13, QTensor
  39, MQT utilities 27, and QCO optimizations 196. XML result counts confirmed
  that each binary ran tests.
- `cmake --build --preset release -j8 --target all mlir-doc` passed.
- `ctest --preset release --output-on-failure -j8` passed 3,224 tests, with
  `ScQDMIJobSpecificationTest.QueryJobId` skipped because the device does not
  support optional job IDs. No tests failed.
- `uvx nox -s cpp-lint -- 7e2a2679f` passed with LLVM 23 tooling and no
  findings.
- `uvx nox -s lint` passed.

Transformation and conversion tests validate matrix and identity-folding
consumers; their production patterns were not changed. Hosted CI has not yet run
on this narrowed diff.
