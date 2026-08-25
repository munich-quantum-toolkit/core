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
- [x] (2026-08-21 12:30Z) Rebase the capture-only change onto current `main`
      after symbolic Qiskit parameter support merged, dropping the superseded
      parent commit while preserving the focused six-file feature delta.
- [x] (2026-08-21 13:05Z) Reproduce three review findings: stale native
      condition operators after public Python mutation, an aborting out-of-range
      `Uint` switch literal, and low-bit truncation for `Uint`-to-`Bool` casts.
- [x] (2026-08-21 13:25Z) Make conditions Python-authoritative, validate literal
      widths, lower Boolean casts as nonzero comparisons, remove the dead hybrid
      native-expression walker, normalize integer-backed Boolean values, and
      preflight operator/type compatibility.
- [x] (2026-08-21 13:35Z) Rebuild and refresh the editable MLIR binding, pass
      all 21 focused capture and corrective cases, and pass all 174 Qiskit
      translation tests against the updated extension.
- [x] (2026-08-21 13:40Z) Pass the complete repository lint session, pinned
      formatting hooks, and `git diff --check`.
- [x] (2026-08-21 13:45Z) Inspect the final diff, create separate gitmoji
      implementation and documentation commits, and push only PR #2175.
- [x] (2026-08-21 22:29Z) Apply the complexity review findings as five focused
      commits, rebuild the binding, pass all 22 focused tests and all 175 Qiskit
      translation tests, and pass the complete repository lint session.

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

- Observation: Qiskit's public control-flow condition setter updates the Python
  operation while the native control-flow view can retain the tree recorded at
  insertion time. Evidence: mutating a public condition from logical AND to OR
  leaves the native operator as AND, so combining native operators with Python
  leaves silently imports a mixed, stale expression.

- Observation: Qiskit accepts a public `expr.Value(3, Uint(1))` switch target,
  so the importer must reject a value that does not fit its declared width
  before constructing an LLVM `APInt`. Without the preflight, LLVM aborts the
  Python process instead of reporting a recoverable import error.

- Observation: Public Qiskit Boolean `Value` nodes expose their value as Python
  integer zero or one. A strict nanobind conversion to C++ `bool` rejects both,
  so normalization must accept only the integer range `[0, 1]` and convert it
  explicitly.

- Observation: A Qiskit cast to `Bool` tests whether the complete source value
  is nonzero. Truncating a packed register to `i1` inspects only its least
  significant bit; for example, `0b10` must be true rather than false.

- Observation: Qiskit's low-level public expression constructors and public
  condition setter permit a node whose declared result type conflicts with its
  operator and operands. The Python-authoritative path therefore needs its own
  recursive type preflight rather than relying on constructor helpers having
  produced every tree.

## Decision Log

- Decision: Add `ClassicalBit` and `ClassicalRegister` to `ExpressionKind`, with
  a global bit index or a normalized register payload on `Expression`.
  Rationale: The normalized tree then owns stable capture identity and stays
  independent of Python object lifetimes. Date/Author: 2026-08-19 / Codex.

- Decision: Keep the scalar `Parameter` model and `Loop::parameter` unchanged.
  Rationale: Scalar symbols and classical captures have different identity and
  typing rules. This branch must remain composable with the reviewed scalar
  slice. Date/Author: 2026-08-19 / Codex.

- Decision: Initially retain the full Python `CircuitInstruction`, the
  containing Python circuit, and the root Python circuit in
  `NativeControlFlowReader`. Rationale: `CircuitInstruction.clbits` describes
  block operands only, native condition indices can remain local, and direct
  root lookup is ambiguous for nested local registers. This decision was
  superseded after the final complexity review. Date/Author: 2026-08-19 / Codex.

- Decision: Retain the Python instruction and containing circuit in each
  `NativeControlFlowReader`, but let the top-level `NativeCircuitReader` own the
  root circuit for the synchronous traversal. Resolve a classical bit in the
  containing circuit and compose its local index with the enclosing native map.
  Rationale: The recursive call stack already keeps the root reader alive, so
  copying the root Python object through every nested reader adds no lifetime
  protection. Date/Author: 2026-08-21 / Codex.

- Decision: Parse expression-valued switch targets from the public Python
  expression tree. Continue to use native metadata for cases and block maps.
  Rationale: This avoids the unsafe Qiskit 2.5 native accessor while keeping the
  established native control-flow reader for supported metadata. Date/Author:
  2026-08-19 / Codex.

- Decision: Parse the complete public Python condition for expressions, Clbits,
  registers, and comparison values; retain native metadata only for blocks,
  capture maps, loops, and switch cases. Rationale: one authoritative tree
  prevents stale native operators or values from being combined with current
  Python capture identities. Date/Author: 2026-08-21 / Codex.

- Decision: Range-check every unsigned literal against its normalized width and
  lower casts to `Bool` with integer or unordered floating-point comparisons
  against zero. Rationale: malformed public inputs must raise a runtime error,
  and Boolean conversion must inspect the whole value, including NaN for Qiskit
  floating-point expressions. Date/Author: 2026-08-21 / Codex.

- Decision: Validate operator, operand, and result-type compatibility on the
  normalized expression before emitting MLIR. Rationale: malformed public trees
  must fail deterministically during preflight instead of producing ill-typed
  semantics or partially constructing a program. Date/Author: 2026-08-21 /
  Codex.

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

After rebasing onto current `main`, the corrective review pass rebuilt and
refreshed the MLIR binding, passed 21 focused cases covering current public
conditions, bounded literals, Boolean casts, and malformed expression typing,
and passed all 174 tests in the complete Qiskit translation file. The complete
repository lint session, pinned Clang and Python formatting, Rumdl, Ruff, `ty`,
targeted Clang-Tidy 21.1.1, and `git diff --check` all pass. Export-side writer
construction remains deliberately out of scope.

The final complexity pass shared public target normalization, replaced manual
operation lookup with `llvm::StringSwitch`, removed duplicate root ownership and
expression callbacks, and kept one representative MLIR text round trip. The
rebuilt binding passed all 22 focused tests, all 175 Qiskit translation tests,
and the complete repository lint session.

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
its root index when the circuit is nested. The top-level reader owns the root
Python circuit throughout the synchronous traversal.

The current scalar `Parameter` tree represents numeric gate and loop
expressions. It is unrelated to Qiskit's typed classical-expression tree and
must not be refactored in this task.

## Plan of Work

First, extend `ExpressionKind` and `Expression` in
`bindings/mlir/qiskit/QiskitTranslation.h` with bit and register leaves. Keep
all scalar parameter declarations byte-for-byte unchanged.

Next, update `bindings/mlir/qiskit/Qiskit2_5.cpp`. Normalize condition and
switch expression trees entirely from the current public Python operation.
Resolve a `Var` leaf by inspecting its public `var` object. For a Clbit, find
the bit in the containing Python circuit and compose the local index through the
enclosing native capture map. Use the same resolver for a legacy tuple
condition's Clbit instead of trusting its native local index. For a classical
register, apply the same mapping to each member in register order. Reject
malformed captures, duplicate or invalid types, standalone variables, and widths
outside the existing 64-bit limit, and reject unsigned literals that do not fit
their declared width before MLIR construction.

Then update `bindings/mlir/qiskit/QiskitImport.cpp`. Pass the classical-bit
state and root map directly into the recursive expression emitter. A bit leaf
calls `loadClassicalBit`; a register leaf calls `packRegister` and extends it to
the normalized expression width. Lower casts to `Bool` as nonzero comparisons
instead of integer truncation or floating-point conversion. Extend preflight
validation to check leaf types, bit bounds, register size, unique register bits,
and expression widths before MLIR construction begins.

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

Build the Qiskit binding with the configured Python release tree. If the
worktree has no compatible build tree yet, refresh the editable installation
through the repository's standard `uv` workflow first:

    cmake --build build/python/Release --target mqt-core-mlir-bindings --parallel 8
    uv sync --inexact --no-dev --no-build-isolation-package mqt-core

Run the focused tests:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py \
      -k 'classical_expression or condition_only or switch_expression or bool_uint_and_float or boolean_expression or cast_to_bool or condition_mutation or narrow_uint or malformed_public_expression'

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

## Idempotence and Recovery

All build, format-check, and test commands are repeatable. Source changes are
limited to the version-neutral normalized model, the Qiskit 2.5 reader, the MLIR
importer, one Python test file, and this plan. Do not reset or overwrite
unrelated work. If a test exposes a Qiskit API difference, inspect the installed
2.5 objects from the test environment and adjust only the version-specific
reader. Do not add a private exporter fallback.

## Artifacts and Notes

The focused capture branch is based directly on current `main`, which already
includes CBit and symbolic scalar support. Native expression nodes do not carry
sufficient public Clbit identity by themselves. `CircuitInstruction.clbits`
supplies identity for block operands, while the containing Python circuit
supplies identity for bits used only by a condition or switch target.

## Interfaces and Dependencies

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

Revision note: Created the initial self-contained plan after comparing the
current scalar/CBit branch with the earlier combined implementation. Updated it
after implementation and final validation to record the public documentation
decision, subprocess-aware test setup, and successful results. Updated it again
after the final audit found valid condition-only and target-only bits outside
the block-capture list; the plan now records the containing-circuit resolver and
nested parent-map regression. Updated it once more after the nested legacy
Clbit-condition accessor exposed its containing-circuit index rather than a root
index. Updated it after rebasing onto current `main` and addressing the final
review findings to record Python-authoritative conditions, bounded unsigned
literals, nonzero Boolean casts, and their focused regressions. Updated it after
the complexity pass to record the shared target normalization, direct lowering
state, root lifetime ownership, reduced round-trip coverage, and final
validation results.
