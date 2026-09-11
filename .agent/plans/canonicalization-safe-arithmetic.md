# Safe arithmetic in gate canonicalization

Status: complete and validated locally.

## Goal and scope

Keep canonicalization from overflowing finite gate angles or discarding a
rotation through binary64 rounding. The shared angle helpers in
`mlir/include/mlir/Dialect/MQT/Utils/Angles.h` and
`mlir/lib/Dialect/MQT/Utils/Angles.cpp` own the arithmetic bound. QCO gate
merges and QC/QCO power rewrites use those helpers before changing IR. Fixed
named gates reduce their exponents by their exact periods before computing
angles.

This change covers canonicalization and its tests. Matrix evaluation, U-gate
power synthesis, wire correspondence, and transformation implementations are
outside its scope.

## Decisions

- Accept constant sums and products only when their absolute rounding error does
  not exceed `PARAMETER_COMPARISON_TOLERANCE`; retain dynamic arithmetic because
  its error has no known bound. This restricts rewrites, not valid input angles.
  Keep the existing global-phase range and principal-branch limits.
- Use TwoSum for addition and `std::fma` for multiplication error. Check every
  failure condition before hoisting supporting operations from modifier bodies.
- Share phase-gate classification across QC and QCO. Validate named phases
  against sine and cosine to reject inaccurate reduction of large angles.
- Preserve consumer matrix checks when updating expectations for short dynamic
  rotation runs. Retain a nontrivial branch-cut negative when squared Pauli
  gates simplify directly to identity.

## Validation

With LLVM/MLIR 23.1.0, the release build and complete CTest run pass: 3,353
entries passed and one skipped because the SC device does not support job IDs.
The focused MQT utility, QC/QCO IR, and decomposition suites also pass. Tests
retain nested powers with a dynamic inner exponent and keep the exponent
computation in the outer body.

Full-file C++ lint against the main base `46f98eabe` covers both the wire and
arithmetic changes and passes. Repository lint and final diff checks pass.
Hosted CI has not tested this extracted change.
