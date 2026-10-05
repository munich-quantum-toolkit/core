# Shared scalar and SSA parameter arithmetic

Status: complete.

## Outcome and scope

`mlir/include/mqt/Dialect/MQT/Utils/Parameters.h` owns floating-point parameter
representation, materialization, and arithmetic. Euler synthesis, rotation
merging, Pauli synthesis, Z-frame propagation, Qiskit import, and the QC, QCO,
and QIR program builders reuse these facilities.

Supported operands are host doubles and existing f64 SSA values. SSA operands
must dominate the builder's current insertion point. The borrowed builder must
outlive each `FloatExpression`; assertions check builder, context, and operand
types. Gate verifiers retain finite-constant validation.

## Decisions

- `FloatParameter` preserves the existing scalar-or-SSA argument type and the
  Euler `RotationParameter` alias. `ConstantFolding.h` owns recursive constant
  evaluation. Rotation addition resolves operands once before its policy checks;
  mixed arithmetic reuses known constant values.
- `FloatExpression` uses MLIR `createOrFold`, including `math.powf`, to fold
  operations during emission. Upstream `ArithBuilder` creates operations without
  folding and does not meet this contract. The wrapper adds no expression trees,
  caches, reassociation, or symbolic evaluator.
- `variantToValue` shares scalar-or-SSA dispatch. Its default materializer emits
  arithmetic constants; QIR supplies its LLVM constant builder and retains
  entry-block placement.
- QCO owns angle wrapping, approximate zero removal, SU(2) cancellation,
  balanced rotation sums, and controlled phase corrections. General arithmetic
  preserves signed zeros and does not discard unknown operands or normalize gate
  angles.

## Validation

Run the shared parameter tests from the repository root:

```sh
ctest --preset release -R ParametersTest
```

The six tests in `mlir/unittests/Dialect/MQT/Utils/test_parameters.cpp` pass
locally. They cover host and SSA arithmetic, direct emission-time folding, mixed
constants, SSA identity and dominance, signed zero, nonfinite propagation, and
caller-owned materialization. See the root
[agent guide](../../AGENTS.md#build-and-validation) for routine build and lint.
