# Import Qiskit classical-expression captures

Status: historical implementation record.

## Goal and scope

Qiskit 2.5 control-flow expressions can read a `Clbit` or `ClassicalRegister`
from the containing circuit. These values can be block captures, but a condition
or switch target can also be their only use. The current importer handles
literal expression trees but rejects these variable leaves. After this change,
`QCProgram.from_qiskit` can import Boolean and unsigned-integer conditions that
read classical bits and registers, including nested control flow and
expression-valued switch targets. The imported MLIR reads the existing
first-class CBit registers, so each expression refers to the same classical
state as the source circuit.

This plan covers import only. It does not add Qiskit writer APIs or construct
Qiskit control-flow operations during export.

## Constraints

- The current branch already contains structured-control import, CBit register
  storage, and the scalar symbolic `Parameter` tree. The older full
  implementation therefore cannot be cherry-picked safely. Evidence:
  `QiskitImport.cpp` already emits `scf.if`, `scf.while`, and
  `scf.index_switch`, while the parent scalar-symbol commit adds the independent
  parameter work.

- Qiskit 2.5 native switch-target accessors are not safe for an
  expression-valued target. The public Python `SwitchCaseOp.target` expression
  tree must be used for that case. Evidence: the earlier implementation records
  that the native C accessors abort when the target is an expression.

- `CircuitInstruction.clbits` contains the bits passed to the control-flow
  blocks, not every bit read by the condition or switch target. An explicit body
  can have zero classical operands while its expression reads a Clbit from the
  containing circuit. Evidence: an explicit `if_test` with an empty `clbits`
  argument is valid Qiskit, but the initial resolver rejected it because both
  the instruction and its block had zero Clbits.

- A nested expression bit must first be resolved in its containing Python
  circuit. A lookup in the root Python circuit can confuse equal Clbit objects
  from similar local registers. Evidence: the nested regression maps local Clbit
  zero to root Clbit one and observes a load from root index one.

- Qiskit's native legacy Clbit-condition accessor returns an index in the
  containing nested circuit. Using that number as a root index reads the wrong
  CBit register element. Evidence: a tuple condition on root Clbit one inside a
  context-managed `for` loop initially emitted `cbit.load` at index zero;
  resolving the Python condition bit through the enclosing map emits index one.

- Qiskit's public control-flow condition setter updates the Python operation
  while the native control-flow view can retain the tree recorded at insertion
  time. Evidence: mutating a public condition from logical AND to OR leaves the
  native operator as AND, so combining native operators with Python leaves
  silently imports a mixed, stale expression.

- Qiskit accepts a public `expr.Value(3, Uint(1))` switch target, so the
  importer must reject a value that does not fit its declared width before
  constructing an LLVM `APInt`. Without the preflight, LLVM aborts the Python
  process instead of reporting a recoverable import error.

- Public Qiskit Boolean `Value` nodes expose their value as Python integer zero
  or one. A strict nanobind conversion to C++ `bool` rejects both, so
  normalization must accept only the integer range `[0, 1]` and convert it
  explicitly.

- A Qiskit cast to `Bool` tests whether the complete source value is nonzero.
  Truncating a packed register to `i1` inspects only its least significant bit;
  for example, `0b10` must be true rather than false.

- Qiskit's low-level public expression constructors and public condition setter
  permit a node whose declared result type conflicts with its operator and
  operands. The Python-authoritative path therefore needs its own recursive type
  preflight rather than relying on constructor helpers having produced every
  tree.

## Decisions

- Add `ClassicalBit` and `ClassicalRegister` to `ExpressionKind`, with a global
  bit index or a normalized register payload on `Expression`. Rationale: The
  normalized tree then owns stable capture identity and stays independent of
  Python object lifetimes.

- Keep the scalar `Parameter` model and `Loop::parameter` unchanged. Rationale:
  Scalar symbols and classical captures have different identity and typing
  rules. This branch must remain composable with the reviewed scalar slice.

- Retain the Python instruction and containing circuit in each
  `NativeControlFlowReader`, but let the top-level `NativeCircuitReader` own the
  root circuit for the synchronous traversal. Resolve a classical bit in the
  containing circuit and compose its local index with the enclosing native map.
  Rationale: The recursive call stack already keeps the root reader alive, so
  copying the root Python object through every nested reader adds no lifetime
  protection.

- Parse expression-valued switch targets from the public Python expression tree.
  Continue to use native metadata for cases and block maps. Rationale: This
  avoids the unsafe Qiskit 2.5 native accessor while keeping the established
  native control-flow reader for supported metadata.

- Parse the complete public Python condition for expressions, Clbits, registers,
  and comparison values; retain native metadata only for blocks, capture maps,
  loops, and switch cases. Rationale: one authoritative tree prevents stale
  native operators or values from being combined with current Python capture
  identities.

- Range-check every unsigned literal against its normalized width and lower
  casts to `Bool` with integer or unordered floating-point comparisons against
  zero. Rationale: malformed public inputs must raise a runtime error, and
  Boolean conversion must inspect the whole value, including NaN for Qiskit
  floating-point expressions.

- Validate operator, operand, and result-type compatibility on the normalized
  expression before emitting MLIR. Rationale: malformed public trees must fail
  deterministically during preflight instead of producing ill-typed semantics or
  partially constructing a program.

- Document circuit Clbit and ClassicalRegister expression variables separately
  from standalone runtime variables. Rationale: circuit-owned bits resolve to
  existing CBit state whether or not a block captures them; the importer still
  rejects Qiskit runtime inputs, and export remains outside this slice.

## Outcome and validation

Import preserves Clbit and ClassicalRegister identity using the containing
circuit and native root maps. Expression leaves use CBit loads and little-
endian packing; preflight rejects malformed captures. Public expression trees
supply conditions and switch targets even when their bits are absent from block
operands. Standalone runtime inputs remain unsupported.

The final rebuilt binding passed 22 focused tests, all 175 translation tests,
and lint. Export-side writer construction was outside this slice.

## Code and ownership

`bindings/mlir/qiskit/QiskitTranslation.h` contains version-neutral normalized
data passed between the Qiskit version adapter and the MLIR importer.
`bindings/mlir/qiskit/Qiskit2_5.cpp` reads Qiskit 2.5 through its native C API
and selected public Python objects. `NativeControlFlowReader` supplies one
normalized `ClassicalTarget` for an if, while, or switch operation.
`bindings/mlir/qiskit/QiskitImport.cpp` lowers that target to MLIR. It already
stores classical state in `!cbit.reg<N>` values and provides `loadClassicalBit`
and `packRegister` helpers.

A block capture is a Clbit used by a control-flow block that comes from its
enclosing circuit. Qiskit exposes Python objects in `CircuitInstruction.clbits`
in block-capture order. Its native control-flow object exposes a map from that
local order to root-circuit Clbit indices. A condition or switch target can also
read a bit that no block uses. The importer therefore retains the containing
Python circuit to find the local bit and uses the enclosing native map to reach
its root index when the circuit is nested. The top-level reader owns the root
Python circuit throughout the synchronous traversal.

The current scalar `Parameter` tree represents numeric gate and loop
expressions. It is unrelated to Qiskit's typed classical-expression tree and
must not be refactored in this task.

## Acceptance

Acceptance requires that a Qiskit if or while condition containing
`expr.lift(circuit.clbits[i])` imports to an MLIR `cbit.load` from the matching
register element. A register expression must load and pack its members in
Qiskit's little-endian register order. An inner control-flow instruction must
resolve its own `CircuitInstruction.clbits` capture order and reach the same
root CBit elements. An expression-valued Qiskit switch must import without
calling a native switch-expression target accessor and must produce
`scf.index_switch`. Explicit if and switch bodies with empty classical operand
lists must still read condition-only and target-only bits from the containing
circuit. A nested condition-only bit must follow the enclosing block's
local-to-root permutation. A nested legacy tuple condition on root Clbit one
must emit a `cbit.load` at index one even when that bit is local index zero in
the enclosing block. Publicly mutating either an expression or tuple condition
must import the current Python operator, target, and comparison value. A packed
register containing `0b10` must cast to true. An unsigned literal outside its
declared width must raise `RuntimeError` rather than aborting the process. A
public expression whose declared result type conflicts with its operator and
operands must fail during preflight.

Malformed block-capture lists and variables absent from the containing circuit
must fail during validation with a clear runtime error. Existing literal
expression, structured-control, CBit, and symbolic parameter tests must continue
to pass. The final tree must have no exporter or writer control-flow
construction changes.

## Interfaces

At completion, `ExpressionKind` in `bindings/mlir/qiskit/QiskitTranslation.h`
has `ClassicalBit` and `ClassicalRegister` cases. `Expression` has
`uint32_t bit` and `Register reg` payloads. `NativeControlFlowReader` in
`Qiskit2_5.cpp` owns the full Python instruction, its operation, its containing
circuit. Its Python-authoritative condition and expression normalization
resolves all classical leaves and legacy Clbit or register conditions to
root-circuit indices through the containing-circuit and parent-map path.
`QiskitImport.cpp` passes the classical-bit state and root map directly into the
expression emitter, which calls `loadClassicalBit` and `packRegister`.

This work depends only on Qiskit 2.5's existing native extension table,
nanobind's public Python object access, MLIR's arithmetic and structured-control
dialects, and MQT Core's CBit builder methods. It introduces no new dependency.
