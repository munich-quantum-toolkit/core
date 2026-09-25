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
proven sign, and explicit 64-bit casts used by OpenQASM export. Initialization
is tracked per constant bit; nonconstant reads require full initialization.
Overlapping assignments snapshot the source before writing. Whole-register
selections use one register read or write, preserving emission budgets.

No runtime slice assertions, runtime bit-vector widths, or slice dialect
operations remain. Compound slice assignments are rejected consistently with
indexed compounds. Upstream scalar indexing and arithmetic semantics remain
unchanged.

Inclusive loop arithmetic uses 64-bit values and unsigned remaining distances,
including signed limits, unsigned endpoints, negative steps, break, and
continue. The frontend emits no explicit i128 arithmetic. Compile-time angle
quantization retains its existing APInt working precision for exact modular
products and shifts.

The shared QTensor canonicalization matcher declines transient unused loop
arguments; pipeline boundaries still verify linearity. The jeff register
read/write conversion reads raw operands after region conversion remaps their
types, matching the existing scalar load/store conversion.

## Validation

Execution regressions live in the QIR JIT suite, which already links the
runtime. They check asymmetric bitstrings, affine quantum and classical
selections, overlapping assignments, fixed-width expressions, and representative
integer-limit ranges. Compiler tests retain the jeff interchange check without
linking the JIT. Export tests cover OpenQASM round trips.

The latest release build passes 493 tests: QIR JIT (57), compiler (233), and
OpenQASM frontend/emitter (203). Bit-string literals preserve leading zeros and
put bit zero on the right; QIR execution exposes the corresponding recording
order. Reading unwritten bits remains invalid, independently of the separate
complete-output work in #2626.

CLI checks compile static, affine quantum, and affine classical selections to
QCO, QIR, jeff, and OpenQASM, then reimport the exported OpenQASM to QIR.
Repository lint and full-file C++ lint pass.
