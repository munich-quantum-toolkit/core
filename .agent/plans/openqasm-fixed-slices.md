# OpenQASM slices with known lengths

Status: complete.

## Scope and decisions

Use the existing affine analysis to prove slice bounds, a nonzero constant step,
and a constant selection length. Nonconstant indices such as `q[i:i]` and
`q[i:i+1]` in bounded loops remain supported. Measurement-selected ranges and
unproved lengths fail during semantic analysis. Register slices retain register
broadcast rules, including one-element selections.

Quantum selections expand into existing proven qubit references. Classical
selections use existing bit references, loads, stores, and fixed-width integer
operations. Each resolver shares reference construction between scalar and slice
operands. The shared index proof handles scalar aliases, negative indices of
proven sign, and explicit 64-bit casts used by OpenQASM export. Initialization is
tracked per constant bit; nonconstant reads require full initialization.
Overlapping assignments snapshot the source before writing. Whole-register
selections use one register read or write, preserving emission budgets.

No runtime slice assertions, runtime bit-vector widths, or slice dialect
operations remain. Compound slice assignments are rejected consistently with
indexed compounds. Existing assertions for unproved scalar classical indices,
runtime zero steps, and integer power preconditions are unchanged.

Inclusive loop arithmetic uses 64-bit values and unsigned remaining distances,
including signed limits, unsigned endpoints, negative steps, break, and continue.
Signed integer power uses the standard extended multiply operation to detect
overflow. The frontend emits no explicit i128 arithmetic. Compile-time angle
quantization retains its existing APInt working precision for exact modular
products and shifts.

The shared QTensor canonicalization matcher declines transient unused loop
arguments; pipeline boundaries still verify linearity. The jeff register
read/write conversion reads raw operands after region conversion remaps their
types, matching the existing scalar load/store conversion.

## Validation

The Clang 23 release preset with ThinLTO and mold passes:

- 207 OpenQASM frontend and emitter tests.
- 256 compiler tests, including DD and QIR execution of slices, overlapping
  assignments, affine indices, and inclusive ranges at integer limits, plus
  QIR execution of signed powers near the machine limits.
- 46 QTensor tests. Restoring the previous assertion makes the nested singleton
  loop regression abort; the fix passes.
- 212 QC translation tests and 153 jeff round-trip tests.
- All 31 existing benchmark generation and simulation tests.

Six external interchange cases cover static, reversed, partially initialized,
singleton loop, affine quantum broadcast, and affine classical selections. All
reach QC, QCO, QIR, and jeff. Classical selections export to OpenQASM and reimport.
Qiskit accepts the fully initialized static and reversed selections; this check
used installed bindings whose Qiskit exporter matches the PR baseline.

Existing exporter limitations remain: OpenQASM and Qiskit reject nonconstant
quantum indices; Qiskit also rejects undefined returned classical bits and
classical index loop bodies. These do not prevent native DD or QIR execution.

## Final checks

Repository lint passes. Full-file C++ lint covered all 13 changed C++ files; the
final compiler test-file rerun is clean. The only remaining diagnostics are 18
pre-existing parser warnings, independently reproduced on the unchanged PR
baseline. No generated files are included.
