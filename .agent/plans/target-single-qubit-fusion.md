# Target fusion and bounded one-qubit angles

Status: implemented. Compact H/RZ synthesis and the PR-local Python reductions
are complete. Broader consolidation belongs in the follow-up to PR #2559, “Unify
one-qubit synthesis and clarify test ownership.”

## Scope and ownership

Target-native synthesis owns optional one-qubit fusion. Its selected Euler basis
is a sufficient lowering basis, not the target's complete native gate set.
Constant native runs require a strict gate-count reduction. Symbolic native runs
remain intact; other symbolic runs use direct Euler identities when available
and otherwise retain individual gate lowering. Standalone fusion continues to
support general runtime quaternion composition. Controlled bodies remain with
the native-control lowering owner.

The shared one-qubit synthesis implementation in
`mlir/lib/Dialect/QCO/Transforms/Optimizations/MergeSingleQubitRotationGates.cpp`
normalizes evaluated gate operands before adding offsets or phase contributions.
All named gates handled there have a common exact period of `4*pi` in each
parameter. This yields operands in `[-2*pi, 2*pi]` and preserves controlled
phase. A uniform `2*pi` reduction would be incorrect for Pauli rotations.

The implementation folds known constants and uses `4*atan(tan(angle/4))` for
runtime values. Qiskit and jeff already support those scalar operations. Scaling
by four is exact in the normal binary64 range and avoids reduction by rounded
pi. Direct Euler formulas then add only bounded offsets and at most two bounded
operands. The resulting gate and phase angles remain bounded; they need not all
use the same principal interval. General quaternion composition also reduces its
accumulated phase after each gate, so its bound is independent of run size.

Normalization applies after evaluating each gate expression, not to a shared
free symbol: `rz(a/2)` and `rz(a)` must reduce independently. General scalar
values, power exponents, and custom gate arguments have no universal period. The
public finite-parameter contract remains unchanged. This change establishes the
bound at the synthesis consumer; it does not add a global bounded-input
precondition or a new angle type, verifier, frontend attribute, or exporter.

## Validation

H/RZ pairs in either order use Hadamard conjugation and the existing Euler
emitters: one U gate or at most three gates in the other bases. Tests check
exact phase under control and late binding after Qiskit and jeff export. Native
synthesis restores controlled U2 to supported U after canonicalization. The U
pipeline guard remains until native synthesis handles dynamic controlled bodies
and isolated gates with the same contracts.

Validation for this update:

- The optimized native build passed. All 329 focused Euler/fusion,
  rotation-merge, native-synthesis, and compiler-pipeline CTests passed.
- All 97 Python MLIR tests passed with Qiskit and jeff available. This PR's
  additions now contain seven consumer cases; numerical sweeps live in C++.
- Whole-file C++ lint passed with no findings. Repository lint checks the
  complete update before publication.

The focused CTest filter is
`Euler|ZSXXShortcut|FuseSingleQubitUnitaryRuns|MergeSingleQubitRotationGatesTest|TargetSynthesisTest|CompilerPipelineTest`.
The Python entry point is `pytest test/python/test_mlir.py`, with the bundled
DDSIM and SC device manifests configured. Routine lint commands are documented
in the root agent guide.

Controlled matrix comparisons cover wrap boundaries and multiple turns in all
seven synthesis bases. Target tests cover preserving native H and symbolic runs,
retaining profitable constant fusion, and portable individual lowering when a
direct symbolic identity does not apply.
