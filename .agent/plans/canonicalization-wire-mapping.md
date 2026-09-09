# Preserve wire mapping in canonicalization

Status: complete and validated locally.

## Scope and decisions

QC and QCO canonicalization must preserve gate targets and the complete action
of modifier bodies. The changes cover `CtrlOp`, `InvOp`, and `PowOp` under
`mlir/lib/Dialect/{QC,QCO}/IR/Modifiers/`, plus the XXPlusYY merge in
`mlir/include/mlir/Dialect/QCO/QCOUtils.h`. Numerical arithmetic, matrix
evaluation, folders, dialect dependencies, and transformation and conversion
patterns are outside this scope.

Valid QCO bodies can yield a permutation of their wires. The shared
`hasPositionalBodyYields` helper follows the unitary interface's input/output
correspondence. Algebraic rewrites require positional outer yields because an
extra permutation changes the body unitary. They keep nested bodies intact;
requiring positional yields recursively would reject safe rewrites.

Valid bodies with fewer than two wires need no correspondence traversal. Stop
following each wire as soon as it reaches its expected argument, including
direct pass-throughs. Current callers establish empty or single-unitary bodies;
no shared cache is needed. Power folding rejects unsupported gate types before
checking correspondence.

Zero controls and power one inline the complete body, including its yields.
Power zero erases the complete action. Double inversion inlines each region
separately to preserve both mappings. These rewrites accept multi-gate bodies.
Nested controls can hoist supported classical operations after all match checks
succeed. Controlled global phase acts on the last control; unused original
targets do not change that target.

QC U powering maps the inner gate's actual target to the outer operand. XXPlusYY
merging requires ordered wires because reversing the wires negates beta.
XXMinusYY retains its symmetric-wire merge.

## Validation

Regression tests are in the QC and QCO `IR/test_*_modifier_canonicalization.cpp`
files, QCO `IR/test_qco_modifier_yield_canonicalization.cpp`, and QCO
`IR/test_qco_wire_canonicalization.cpp`. They verify input and output IR, check
QCO linearity, retain unsupported yield permutations, and check safe inlining
and ordered gate outputs. XXPlusYY tests compare the full unitary against the
uncanonicalized input.

With LLVM/MLIR 23.1.0, the release build passes. The QC and QCO IR binaries pass
all 357 and 549 tests, respectively. The complete release CTest run reports
3,341 passed entries and one skip for the SC device's unsupported job-ID
property. Full-file C++ lint against the main base `46f98eabe` and repository
lint pass. These results describe local validation.
