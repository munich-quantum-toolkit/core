# OpenQASM slices with known lengths

Status: in progress. The final complexity review remains.

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

## Work remaining

- Complete the complexity review of both PR diffs.

## Validation

The release build passes all 920 tests across the OpenQASM (206), compiler
(259), QTensor (46), QC translation (219), jeff round-trip (153), and benchmark
(37) test binaries. Compiler regressions cover overlapping selections, affine
indices, integer-limit ranges, and OpenQASM/jeff round trips.

CLI checks compile static, affine quantum, and affine classical selections to
QCO, QIR, jeff, and OpenQASM, then reimport the exported OpenQASM to QIR.
Repository lint passes. Full-file C++ lint covers the complete PR diff; the
corrected emitter test include is checked in a focused follow-up.
