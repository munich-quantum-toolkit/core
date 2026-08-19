# Import Qiskit classical-expression captures

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-08-19 14:46Z) Read the repository instructions and compare the
      current scalar/CBit branch with the earlier full control-flow
      implementation.
- [x] (2026-08-19 14:52Z) Extend the normalized expression model with captured
  bit and register leaves without changing the scalar `Parameter` model.
- [x] (2026-08-19 14:53Z) Normalize Qiskit expression variables through public
      Python bit identity and Qiskit's native local-to-root Clbit maps.
- [x] (2026-08-19 14:54Z) Materialize and validate captured leaves through the
  existing CBit load and register-packing helpers.
- [x] (2026-08-19 14:56Z) Add focused bit, register, nested-capture, malformed
  capture, and switch-expression import tests.
- [x] (2026-08-19 15:03Z) Build, run the full Qiskit translation test file and
  repository lint session, inspect the final diff, and prepare the completed
  import slice for a local commit.
- [x] (2026-08-19 15:10Z) Reproduce the valid explicit-body case in which a
      condition reads a root Clbit absent from all block operands.
- [x] (2026-08-19 15:19Z) Retain the Python circuit hierarchy, add a
      containing-circuit resolver with parent-map composition, add focused if,
      switch, and nested-map regressions, rebuild, and pass all eight focused
      capture tests.
- [x] (2026-08-19 15:23Z) Pass all 165 Qiskit translation tests and the complete
      repository lint session, inspect the final diff, and prepare the existing
      local import commit for amendment.
- [x] (2026-08-19 16:12Z) Reproduce a nested legacy tuple condition that reads
      root Clbit one as local index zero, route it through the public Python bit
      resolver, add the exact `for`-then-`if` regression, rebuild, and pass all
      nine focused capture and condition tests.
- [x] (2026-08-19 16:14Z) Pass all 166 Qiskit translation tests, rerun the
      complete repository lint session, inspect the final diff, and prepare the
      existing local import commit for amendment.
- [x] (2026-08-19 19:45Z) Rebase the focused import commit onto the scalar
      commit after first-class CBit support merged, rebuild the release MLIR
      binding, and pass all 166 Qiskit translation tests again.
- [x] (2026-08-19 20:08Z) Restack onto the audited scalar parent, update the
      recorded parent identity, rebuild the release binding, and pass all 167
      Qiskit translation tests.

## Surprises & Discoveries

- Observation: The current branch already contains structured-control import,
  CBit register storage, and the scalar symbolic `Parameter` tree. The older
  full implementation therefore cannot be cherry-picked safely. Evidence:
  `QiskitImport.cpp` already emits `scf.if`, `scf.while`, and
  `scf.index_switch`, while the parent scalar-symbol commit adds the independent
  parameter work.

- Observation: Qiskit 2.5 native switch-target accessors are not safe for an
  expression-valued target. The public Python `SwitchCaseOp.target` expression
  tree must be used for that case. Evidence: the earlier implementation records
  that the native C accessors abort when the target is an expression.

- Observation: A full test run must inject the worktree-built extension into
  child Python processes as well as the pytest process. Evidence: one existing
  isolation test launches `sys.executable`; after using a temporary
  `sitecustomize.py`, all 162 tests exercised the local binding and passed. The
  temporary harness was removed after validation.

- Observation: `CircuitInstruction.clbits` contains the bits passed to the
  control-flow blocks, not every bit read by the condition or switch target. An
  explicit body can have zero classical operands while its expression reads a
  Clbit from the containing circuit. Evidence: an explicit `if_test` with an
  empty `clbits` argument is valid Qiskit, but the initial resolver rejected it
  because both the instruction and its block had zero Clbits.

- Observation: A nested expression bit must first be resolved in its containing
  Python circuit. A lookup in the root Python circuit can confuse equal Clbit
  objects from similar local registers. Evidence: the nested regression maps
  local Clbit zero to root Clbit one and observes a load from root index one.

- Observation: Qiskit's native legacy Clbit-condition accessor returns an index
  in the containing nested circuit. Using that number as a root index reads the
  wrong CBit register element. Evidence: a tuple condition on root Clbit one
  inside a context-managed `for` loop initially emitted `cbit.load` at index
  zero; resolving the Python condition bit through the enclosing map emits index
  one.

## Decision Log

- Decision: Add `ClassicalBit` and `ClassicalRegister` to `ExpressionKind`, with
  a global bit index or a normalized register payload on `Expression`.
  Rationale: The normalized tree then owns stable capture identity and stays
  independent of Python object lifetimes. Date/Author: 2026-08-19 / Codex.

- Decision: Keep `ParameterKind`, `Parameter`, and `Loop::parameter` unchanged.
  Rationale: Scalar symbols and classical captures have different identity and
  typing rules. This branch must remain composable with the reviewed scalar
  slice. Date/Author: 2026-08-19 / Codex.

- Decision: Retain the full Python `CircuitInstruction`, the containing Python
  circuit, and the root Python circuit in `NativeControlFlowReader`. Resolve a
  classical bit in the containing circuit and compose its local index with the
  enclosing native capture map when the circuit is nested. Use the current
  native block map only to validate the instruction structure. Apply this rule
  to expression leaves, switch targets, and legacy tuple conditions. Rationale:
  `CircuitInstruction.clbits` describes block operands only, native condition
  indices can remain local, and direct root lookup is ambiguous for nested local
  registers. Date/Author: 2026-08-19 / Codex.

- Decision: Parse expression-valued switch targets from the public Python
  expression tree. Continue to use native metadata for cases and block maps.
  Rationale: This avoids the unsafe Qiskit 2.5 native accessor while keeping the
  established native control-flow reader for supported metadata. Date/Author:
  2026-08-19 / Codex.

- Decision: Document circuit Clbit and ClassicalRegister expression variables
  separately from standalone runtime variables. Rationale: circuit-owned bits
  resolve to existing CBit state whether or not a block captures them; the
  importer still rejects Qiskit runtime inputs, and export remains outside this
  slice. Date/Author: 2026-08-19 / Codex.

## Outcomes & Retrospective

The import slice now preserves Clbit and ClassicalRegister identity through the
containing Python circuit and Qiskit's native root maps. It lowers variable
leaves through the existing CBit load and little-endian register pack paths,
preflights malformed captures, and reads expression-valued switch targets only
through the public Python expression tree. Conditions and switch targets also
work when their classical bits are absent from every block operand. The public
support table distinguishes these supported circuit values from rejected
standalone runtime inputs.

The release MLIR binding built successfully. The complete Qiskit translation
test file passed with 167 tests against that local extension, including the
subprocess isolation test, the condition-only regressions, and the nested legacy
Clbit condition. `uvx nox -s lint`, `git diff --check`, Clang format, Ruff,
Rumdl, Prettier, and `ty` all passed. Export-side writer construction remains
deliberately out of scope.

## Context and Orientation

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
its root index when the circuit is nested. The retained root Python circuit owns
the complete object hierarchy while the reader traverses nested blocks.

The current scalar `Parameter` tree represents numeric gate and loop
expressions. It is unrelated to Qiskit's typed classical-expression tree and
must not be refactored in this task.

## Plan of Work

First, extend `ExpressionKind` and `Expression` in
`bindings/mlir/qiskit/QiskitTranslation.h` with bit and register leaves. Keep
all scalar parameter declarations byte-for-byte unchanged.

Next, update `bindings/mlir/qiskit/Qiskit2_5.cpp`. Make the native expression
normalizer walk the matching public Python expression node beside each native
node. Resolve a `Var` leaf by inspecting its public `var` object. For a Clbit,
find the bit in the containing Python circuit and compose the local index
through the enclosing native capture map. Use the same resolver for a legacy
tuple condition's Clbit instead of trusting its native local index. For a
classical register, apply the same mapping to each member in register order.
Reject malformed captures, duplicate or invalid types, standalone variables, and
widths outside the existing 64-bit limit. Keep a Python-only expression walker
for switch targets so no unsafe native switch-expression accessor is called.

Then update `bindings/mlir/qiskit/QiskitImport.cpp`. Pass callbacks into the
recursive expression emitter. A bit leaf calls `loadClassicalBit`; a register
leaf calls `packRegister` and extends it to the normalized expression width.
Extend preflight validation to check leaf types, bit bounds, register size,
unique register bits, and expression widths before MLIR construction begins.

Finally, add tests to `test/python/test_mlir_qiskit_translation.py`. Cover one
captured Clbit expression, one captured register expression, nested control flow
whose inner expression uses outer captures, and an expression-valued switch
target. Also cover explicit if and switch bodies whose expression bits are
absent from every block operand, plus a nested permutation that proves
parent-map composition. Verify the expected CBit loads and
arithmetic/control-flow ops, and re-import the produced program or source
circuit where export is outside this slice.

## Concrete Steps

Run all commands from the repository root.

Inspect the focused diff and formatting:

    git diff --check
    clang-format --dry-run --Werror bindings/mlir/qiskit/Qiskit2_5.cpp \
      bindings/mlir/qiskit/QiskitImport.cpp \
      bindings/mlir/qiskit/QiskitTranslation.h
    uvx ruff check test/python/test_mlir_qiskit_translation.py

Build the Qiskit binding with the configured release tree. If the isolated
worktree has no compatible build tree yet, configure it with the repository's
release preset first:

    cmake --build build/release --parallel 8

Run the focused tests:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py \
      -k 'classical_expression or condition_only or switch_expression'

Run the complete Qiskit translation test file after the focused tests pass:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py

Run the repository lint session before handoff:

    uvx nox -s lint

## Validation and Acceptance

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
the enclosing block.

Malformed block-capture lists and variables absent from the containing circuit
must fail during validation with a clear runtime error. Existing literal
expression, structured-control, CBit, and symbolic parameter tests must continue
to pass. The final tree must have no exporter or writer control-flow
construction changes.

## Idempotence and Recovery

All build, format-check, and test commands are repeatable. Source changes are
limited to the version-neutral normalized model, the Qiskit 2.5 reader, the MLIR
importer, one Python test file, and this plan. Do not reset or overwrite
unrelated work. If a test exposes a Qiskit API difference, inspect the installed
2.5 objects from the test environment and adjust only the version-specific
reader. Do not add a private exporter fallback.

## Artifacts and Notes

The source branch begins at the focused scalar-symbol parent, which already
includes CBit and symbolic scalar support. Native expression nodes do not carry
sufficient public Clbit identity by themselves. `CircuitInstruction.clbits`
supplies identity for block operands, while the containing Python circuit
supplies identity for bits used only by a condition or switch target.

## Interfaces and Dependencies

At completion, `ExpressionKind` in `bindings/mlir/qiskit/QiskitTranslation.h`
has `ClassicalBit` and `ClassicalRegister` cases. `Expression` has
`uint32_t bit` and `Register reg` payloads. `NativeControlFlowReader` in
`Qiskit2_5.cpp` owns the full Python instruction, its operation, its containing
circuit, and the root Python circuit. Its expression normalization resolves all
classical leaves and legacy Clbit conditions to root-circuit indices through the
containing-circuit and parent-map path. `QiskitImport.cpp` accepts expression
leaves only through callbacks backed by `loadClassicalBit` and `packRegister`.

This work depends only on Qiskit 2.5's existing native extension table,
nanobind's public Python object access, MLIR's arithmetic and structured-control
dialects, and MQT Core's CBit builder methods. It introduces no new dependency.

Revision note: Created the initial self-contained plan after comparing the
current scalar/CBit branch with the earlier combined implementation. Updated it
after implementation and final validation to record the public documentation
decision, subprocess-aware test setup, and successful results. Updated it again
after the final audit found valid condition-only and target-only bits outside
the block-capture list; the plan now records the containing-circuit resolver and
nested parent-map regression. Updated it once more after the nested legacy
Clbit-condition accessor exposed its containing-circuit index rather than a root
index.
