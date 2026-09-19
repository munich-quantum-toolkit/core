# Target fusion and bounded one-qubit angles

Status: complete.

## Scope and ownership

Target-native synthesis owns optional one-qubit fusion. Its selected Euler basis
is a sufficient lowering basis, not the target's complete native gate set.
Constant native runs require a strict gate-count reduction. Symbolic native
runs remain intact; other symbolic runs use direct Euler identities when
available and otherwise retain individual gate lowering. Standalone fusion
continues to support general runtime quaternion composition.
Controlled bodies remain with the native-control lowering owner.

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
use the same principal interval. General quaternion composition also reduces
its accumulated phase after each gate, so its bound is independent of run size.

Normalization applies after evaluating each gate expression, not to a shared
free symbol: `rz(a/2)` and `rz(a)` must reduce independently. General scalar
values, power exponents, and custom gate arguments have no universal period.
The public finite-parameter contract remains unchanged. This change establishes
the bound at the synthesis consumer; it does not add a global bounded-input
precondition or a new angle type, verifier, frontend attribute, or exporter.

## Validation

- The optimized native build passed. All 260 focused compiler, target-synthesis,
  and rotation-merge CTests passed.
- All 109 Python MLIR tests passed, including Qiskit late binding and jeff
  export. Six independent numerical and export reproducers also passed.
- Repository lint and whole-file C++ lint passed for the final changes.

The focused CTest filter is
`MergeSingleQubitRotationGatesTest|TargetSynthesisTest|CompilerPipelineTest`.
The Python entry point is `pytest test/python/test_mlir.py`, with the bundled
DDSIM and SC device manifests configured. Routine lint commands are documented
in the root agent guide.

Controlled matrix comparisons cover wrap boundaries and multiple turns in all
seven synthesis bases. Target tests cover preserving native H and symbolic
runs, retaining profitable constant fusion, and portable individual lowering
when a direct symbolic identity does not apply.
