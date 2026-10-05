# Shared scalar and SSA parameter arithmetic

Status: complete; implementation and local validation are complete.

## Goal and scope

Resolve #2673 by sharing floating-point parameter representation and arithmetic
between Euler synthesis, rotation merging, and the QC, QCO, and QIR program
builders. `mqt/Dialect/MQT/Utils/Parameters.h` owns these facilities;
`ConstantFolding.h` remains the owner of recursive constant evaluation.

The baseline is `726c1adf4` (the merged native synthesis change). The working
tree was clean. The supported operands are host doubles and existing f64 SSA
values. SSA operands must dominate the builder insertion point. Gate verifiers
retain finite-constant validation; arithmetic does not validate or normalize
gate angles.

## Decisions

- Keep the scalar-or-SSA representation as a type alias, preserving existing C++
  argument types and the Euler `RotationParameter` alias.
- Move the existing SSA arithmetic wrapper into Parameters and reuse it for
  mixed-parameter arithmetic. Use MLIR `createOrFold`; do not add expression
  trees, caches, reassociation, or a symbolic evaluator. Upstream `ArithBuilder`
  creates operations without folding, so it does not retain the emission-time
  folding contract. The shared wrapper uses the operation folders directly.
- Keep angle wrapping, approximate zero removal, SU(2) cancellation, balanced
  rotation sums, and controlled phase corrections in QCO. General arithmetic
  must preserve signed zeros and must not discard unknown operands.
- Share scalar-or-SSA dispatch through `variantToValue`. Its default
  materializer emits arithmetic constants; QIR supplies its existing LLVM
  constant builder and retains its entry-block insertion policy.

## Validation

The release build used LLVM/MLIR 23.1.0. All 2,008 tests passed across the MQT
utility, QCO optimization, native synthesis, decomposition, QC/QCO/QIR IR, and
compiler binaries. This includes direct regressions for host and SSA arithmetic,
SSA identity and dominance, signed zero, nonfinite propagation, and caller-owned
materialization.

After rebuilding the editable package in Release, the following command passed
all 791 Python tests:

```sh
uv run --no-sync pytest test/python/test_mlir_qiskit_target.py \
  test/python/test_mlir_qiskit_translation.py test/python/test_mlir.py -q
```

The existing workloads include 1,024-angle symbolic rotation runs and 100-qubit
symbolic SU2 circuits, bindable Qiskit exports, jeff serialization, controlled
phases, and angles up to 1e300. Their expression and gate-count bounds passed.
`uvx nox -s lint` and whole-file `uvx nox -s cpp-lint` passed. C++ lint used
clang-tidy 23.1.2 and checked all seven changed translation units, including
their headers. Its local build cache was refreshed after a compiler upgrade. No
hosted CI result is claimed.

## Outcome

`FloatParameter` preserves the builders' scalar-or-SSA argument type;
`parameterToConstantDouble` reuses the constant evaluator; `addParameters` and
`scaleParameter` use the same `FloatExpression` operations as runtime rotation
merging. Euler keeps its `RotationParameter` alias and rotation-specific zero
identity policy. QIR keeps LLVM constant emission and entry-block placement.
