# Support compact Qiskit classical initialization

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Qiskit defines every classical bit as zero before a circuit runs. OpenQASM 3
leaves a declared bit undefined until a statement assigns it. The Qiskit plugin
currently preserves Qiskit behavior by exporting one assignment for every bit.
After this change, it can initialize each `ClassicalRegister` with one compact
`register = 0;` statement, while loose classical bits still use scalar
assignments. MQT Core accepts that spelling only as a narrow Qiskit
compatibility extension, and native OpenQASM declarations without assignments
remain undefined.

The behavior is visible in the Qiskit serializer output and in direct OpenQASM
frontend tests. A partially measured multi-bit register has one register-wide
zero assignment before the measurement, and untouched bits lower to constant
false values.

## Progress

- [x] (2026-09-03 02:40 CEST) Read issue #2332, its originating #2288 review
      discussion, repository policy, and the frontend and serializer paths.
- [x] (2026-09-03 02:40 CEST) Checked OpenQASM 3.1 assignment and cast rules and
      selected an explicit zero-only Qiskit compatibility extension.
- [x] (2026-09-03 02:40 CEST) Checked related PRs #2181 and #2297 and found no
      design dependency; both touch nearby frontend files and require a later
      rebase check.
- [x] (2026-09-03 03:25 CEST) Added zero-only whole-register assignment analysis
      and direct semantic and lowering tests.
- [x] (2026-09-03 03:25 CEST) Changed Qiskit serialization to initialize each
      classical register once, retained loose-bit initialization, and tested the
      serialized order and shape with Qiskit 1.1 and 2.5.
- [x] (2026-09-03 03:40 CEST) Documented the compatibility form and completed
      focused validation and the changed-file C++ lint. Repository lint passes
      for the changed files.
- [x] (2026-09-03 04:46 CEST) Rebased onto `main` after #2337 fixed its stale
      sampling call and confirmed that the complete repository lint suite
      passes.

## Surprises & Discoveries

- Observation: OpenQASM 3.1 does not permit implicit assignment from an integer
  to `bit[n]`. Assignment operands must have the same type, and an
  integer-to-bit cast needs an explicit matching width. Qiskit nevertheless
  exports `ClassicalRegister` stores as `register = 0;`.
- Observation: The semantic frontend already accepts `bit[1] value; value = 0;`
  through its scalar-bit path. Only registers wider than one reach the current
  whole-register diagnostic.
- Observation: The frontend already represents and lowers one-bit assignments
  through `BitAssignmentStatement`. Expanding the compatibility form into those
  existing statements avoids a new bit-vector expression kind and keeps the QC
  emitter unchanged.

## Decision Log

- Decision: Treat only an exact integer literal zero assigned to an unindexed
  bit register as a Qiskit compatibility extension. Reject nonzero literals,
  negative values, named integers, and computed expressions. Rationale: This is
  the only form needed for Qiskit zero initialization and does not imply a
  general nonstandard conversion rule. Date/Author: 2026-09-03 / Codex.
- Decision: Expand the accepted register assignment into one existing
  `BitAssignmentStatement` per target bit. Rationale: The current frontend
  already tracks initialization and lowers these statements correctly; a new
  public expression or statement type would add unused flexibility. Date/Author:
  2026-09-03 / Codex.
- Decision: Work from `main` rather than stack on #2297. Rationale: #2297
  implements the opposite standards-compliant bit-register-to-integer cast and
  is not a semantic prerequisite. Avoiding its new types keeps this extension
  independent, although overlapping test and documentation files must be checked
  when either PR changes. Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The frontend now expands an exact integer-literal-zero assignment to an
unindexed `bit[n]` register into the existing per-bit assignment model. The
Qiskit serializer emits one such assignment per `ClassicalRegister`, while loose
`Clbit` instances retain scalar false assignments. This preserves Qiskit
initialization semantics without adding a general integer conversion or new
frontend representation.

The complete OpenQASM target suite passes 177 tests, and the complete Qiskit
mock-backend module passes 21 tests. The serializer shape was also checked with
the minimum supported Qiskit 1.1 and current Qiskit 2.5. Ruff and whitespace
checks pass for all changed files. Cpp-linter reports zero clang-format and zero
clang-tidy findings after a complete non-unity build. The complete repository
lint suite also passes after rebasing onto `main` at #2337.

## Context and Orientation

`python/mqt/core/plugins/qiskit/backend.py` prepares an empty circuit before
OpenQASM export. Its `_serialize_to_qasm3` function currently inserts a Qiskit
`store(false)` instruction for every classical bit. Qiskit exports a store of
integer zero to a complete `ClassicalRegister` as `register = 0;`.

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` turns parsed OpenQASM syntax
into the typed frontend model declared in
`mlir/include/mlir/Target/OpenQASM/Frontend.h`. Its `analyzeAssignment` function
accepts whole-register bit-vector values, then otherwise resolves a single
target bit and analyzes the right-hand side as a Boolean value. The existing
typed `BitAssignmentStatement` records a target bit and a condition.
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` lowers each such
statement to a `cbit.store` operation.

The semantic tests are in
`mlir/unittests/Target/OpenQASM/test_openqasm_semantics.cpp`, and end-to-end QC
lowering tests are in
`mlir/unittests/Target/OpenQASM/test_openqasm_emitter.cpp`. Qiskit plugin tests
are in `test/python/plugins/qiskit/test_mock_backend.py`. User-facing OpenQASM
input behavior is documented in `docs/mlir/OpenQASM.md`.

## Plan of Work

First, update `analyzeAssignment` after it resolves the target register. When an
unindexed register has more than one target bit, accept the right-hand side only
when its syntax node is the non-wide integer literal zero. Analyze that literal
once as a false condition. Mark every target bit initialized, add one
`BitAssignmentStatement` for each bit, and return success. Keep the current
whole-register diagnostic for every other scalar right-hand side.

Add semantic tests that accept zero for one- and multi-bit registers and reject
nonzero, negative, and computed zero forms. Add an end-to-end lowering test for
a three-bit output initialized to zero and then partially measured. The two
untouched bits must be constant false, and the measured bit must retain the
measurement result.

Second, change `_serialize_to_qasm3` to store zero once into every
`ClassicalRegister`. Then initialize only classical bits that belong to no
register with scalar `false` stores. Update the existing backend test to use a
one-bit register, a multi-bit register, a loose classical bit, and a partial
measurement. Check that register assignments replace registered per-bit stores
and that all initialization precedes measurement.

Finally, document `register = 0;` as a literal-zero-only Qiskit compatibility
form. State that it is not a general integer-to-bit conversion and that bare
OpenQASM 3 declarations remain undefined. Do not add a standalone changelog
entry because this changes unreleased v4 frontend behavior.

## Concrete Steps

Run all commands from the repository root. Build and run the OpenQASM target
tests with:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittest-openqasm-target
    ./build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target

Run the focused Python test with:

    uv run --no-sync pytest test/python/plugins/qiskit/test_mock_backend.py \
      -k qasm3_zero_initializes_classical_bits

At completion, run:

    git diff --check
    uvx nox -s cpp-lint
    uvx nox -s lint

## Validation and Acceptance

The semantic frontend must accept literal zero for an unindexed bit register of
any positive width. It must reject `1`, `-1`, `0 + 0`, and a scalar variable in
the same position. Existing whole-register bit-vector assignments must remain
unchanged. OpenQASM 3 output bits with no assignment must still fail the
initialized-output check.

The translated three-bit partial-measurement program must verify as QC IR. Its
untouched output bits must be constant false, and its measured output bit must
come from `qc.measure`. The Qiskit serializer must emit one zero store per
classical register and one false store per loose classical bit, all before any
measurement.

The complete OpenQASM target test binary, focused Python test, C++ lint, and
repository lint must pass.

## Idempotence and Recovery

All edits, builds, and tests are repeatable. The work is isolated in its own
worktree and does not modify the branches for #2330, #2331, #2181, or #2297. If
one of the related OpenQASM PRs changes, fetch it and compare the semantic,
test, and documentation hunks before rebasing; do not overwrite another task's
branch.
