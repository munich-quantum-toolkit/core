# Support symbolic Qiskit parameter expressions

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
stay current as the implementation changes.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Users can import Qiskit circuits whose gates and global phase use free
parameters or real-valued parameter expressions. The compiler represents each
free parameter as a named `f64` function input and represents arithmetic with
frontend-neutral Arith and Math dialect operations. Users can export the
program, bind the reconstructed Qiskit parameters, and obtain the same numeric
circuit. Lexically bound `for`-loop values remain distinct from free parameters,
even when their displayed names match.

This work completes issue #2067. It extends the Qiskit circuit translation
introduced by #2031 and builds on the CBit representation from #2158. It must
not weaken the existing preflight checks, mutate input circuits, or expose a
partially constructed output circuit after a failure. Exact
`ParameterVectorElement` provenance is a separate follow-up.

## Progress

- [x] (2026-08-18 11:26Z) Stack the direct-symbol implementation on the current
      #2136 branch and preserve both changes through the overlapping Qiskit
      files.
- [x] (2026-08-18 11:26Z) Confirm that Qiskit 2.5 records parameter expressions
  as a postfix `_qpy_replay` sequence and provides C constructors for the
  supported arithmetic operations.
- [x] (2026-08-18 14:45Z) Normalize Qiskit numbers, symbols, and supported
  expressions into one bounded, frontend-neutral C++ tree in the
  version-specific translation.
- [x] (2026-08-18 14:45Z) Materialize normalized expressions as `f64` Arith and
      Math SSA values on import, with symbol lookup by identity rather than
      name.
- [x] (2026-08-18 14:45Z) Reconstruct normalized expressions from supported
  compiler SSA on export and materialize shared Qiskit parameter objects
  through the Qiskit C API.
- [x] (2026-08-18 14:45Z) Permit parameterized custom definitions when all
      symbols resolve, and preserve lexical identity through nested control
      flow.
- [x] (2026-08-19 13:40Z) Split exact `ParameterVectorElement` provenance into a
      follow-up and reject vector elements explicitly in this scalar-symbol
      layer.
- [x] (2026-08-18 14:45Z) Add contract tests, update the support table, and
  update this plan.
- [x] (2026-08-18 15:22Z) Fold pull request #2150 into the existing Qiskit
  changelog entry without changing its wording.
- [x] (2026-08-18 13:51Z) Rebase on the current #2136 head and run the Release,
  C++, Python 3.13, Qiskit 2.5.0 stable-ABI, documentation, stub, and lint
  validation.
- [x] (2026-08-18 14:59Z) Rebase the symbolic commit onto the `main` merge of
      #2136. Retain the merged controlled-unitary helper and the symbolic
      parameter reconstruction path in the only conflicting file.
- [x] (2026-08-19 14:15Z) Port the scalar-symbol layer onto the exact current
  #2158 head and rerun the focused Python and complete compiler suites.
- [x] (2026-08-19 19:50Z) Rebase the validated scalar-symbol commit onto `main`
      after #2158 merged, rebuild the Python bindings, rerun all 157 Qiskit
      translation tests, and pass the focused formatting and diff checks.
- [x] (2026-08-19 20:01Z) Reject named `f64` inputs that do not occur in any
  exported parameter tree, add the source-unchanged regression, rebuild, and
  pass all 158 Qiskit translation tests.

## Surprises & Discoveries

- Observation: Qiskit 2.5 has no public expression-tree reader that works
  without an optional SymPy installation. Its own parameter-expression code
  records a stable postfix replay sequence in `_qpy_replay`. Evidence: nested
  expressions expose `OPReplay` records with `op`, `lhs`, and `rhs`; reverse
  subtraction, division, and power use distinct opcodes.
- Observation: Qiskit rejects two free parameters with the same name in one
  circuit, but it permits a lexically bound loop parameter and a distinct free
  parameter to share a name. Evidence: the existing name-keyed local map
  incorrectly captured the free parameter in such a loop body.
- Observation: Qiskit can construct parameter objects that share a UUID but
  disagree on their name. The importer must compare canonical scalar symbol
  metadata and reject such aliases before creating a module.
- Observation: a custom gate's definition already contains the actual symbols or
  expressions supplied at its call site. The importer does not need a separate
  formal-parameter substitution scheme. It must validate the definition against
  the current global and lexical identities.
- Observation: an expression can convert to a number while still tracking free
  parameters. The version-specific reader must inspect `parameters` before it
  treats a value as a numeric constant.
- Observation: treating `ParameterVectorElement` as an ordinary standalone
  symbol changes positional binding order. This layer therefore rejects vector
  elements instead of inferring semantics from names such as `theta[10]`.
- Observation: Merely collecting a named function argument does not preserve it
  in Qiskit. The writer only creates parameters reached from emitted gate or
  global-phase expression trees, so an unused input would otherwise disappear.

## Decision Log

- Decision: Use one immutable, copyable scalar expression tree at the generic
  reader/writer boundary. Rationale: Qiskit-specific replay objects remain in
  `Qiskit2_5.cpp`, while import and export share one frontend-neutral contract.
  Date/Author: 2026-08-18 / Codex.
- Decision: Support finite numbers, symbols, add, subtract, multiply, divide,
  power, negate, sine, cosine, tangent, inverse sine, inverse cosine, inverse
  tangent, exponential, logarithm, absolute value, and real conjugation.
  Rationale: Arith, Math, and Qiskit's 2.5 C API represent this real-valued
  subset directly. Operations without matching compiler semantics fail with a
  precise diagnostic. Date/Author: 2026-08-18 / Codex.
- Decision: Key all parameters by their Qiskit identity during import and by
  their compiler SSA value during export. Use `mqt.input_name` for the public
  scalar name. Rationale: identity prevents lexical capture without storing a
  frontend object in MLIR. Date/Author: 2026-08-18 / Codex.
- Decision: Preserve symbol sharing but do not preserve Qiskit's original UUID
  across a round trip. Rationale: the compiler input is the frontend-neutral
  identity. The writer creates exactly one Qiskit symbol for each input and
  reuses it throughout gates and global phase. Date/Author: 2026-08-18 / Codex.
- Decision: Bound normalized expression depth and node count before compiler or
  circuit construction. Rationale: the existing definition and control-flow
  readers are bounded, and parameter replay must have the same fail-closed
  behavior for adversarial input. Date/Author: 2026-08-18 / Codex.
- Decision: Reject `ParameterVectorElement` input in this PR and implement exact
  vector provenance as a stacked follow-up. Rationale: scalar symbols complete
  issue #2067, while vector identity, allocation bounds, sparse indices, and
  vector-level binding form an independently reviewable contract. Date/Author:
  2026-08-19 / Codex.
- Decision: Require every named `f64` input identity to occur in the normalized
  parameter trees that will be emitted. Rationale: Qiskit circuits cannot
  declare an otherwise unused parameter, so failing before writer allocation
  avoids silently changing the public parameter set. Date/Author: 2026-08-19 /
  Codex.

## Outcomes & Retrospective

The scalar implementation is complete. Shared direct symbols, bounded real
expression trees, parameterized definitions, identity-safe loop bindings, and
global phase passed the original focused validation. Unused named inputs now
fail before writer allocation rather than disappearing. The split branch builds
and passes all 158 Qiskit translation tests after #2158 merged.

## Context and Orientation

`bindings/mlir/qiskit/QiskitTranslation.h` defines the normalized objects shared
by the generic translation and one Qiskit-version adapter.
`bindings/mlir/qiskit/Qiskit2_5.cpp` is the only file that reads Python
parameter objects, `_qpy_replay`, or calls Qiskit's `qk_param_*` C functions.

`bindings/mlir/qiskit/QiskitImport.cpp` validates a complete source circuit,
creates a QC program, inserts one named `f64` entry argument per free symbol,
and lowers normalized expressions to SSA values.
`bindings/mlir/qiskit/QiskitExport.cpp` performs the reverse preflight: it
recognizes a supported `f64` SSA expression graph, builds normalized
expressions, and only then asks a version-specific writer to allocate a Qiskit
circuit.

The importer uses `mqt.input_name`, declared in
`mlir/include/mlir/Dialect/Utils/Utils.h`, for the stable public name of each
compiler input. The compiler representation uses `arith.addf`, `arith.subf`,
`arith.mulf`, `arith.divf`, and `arith.negf`, plus matching real-valued Math
dialect operations. A local `for` induction parameter is a temporary SSA value
keyed by the loop parameter's Qiskit identity. It is not a function input.

## Plan of Work

First, replace the number-or-symbol `Parameter` value in `QiskitTranslation.h`
with an immutable expression node. Keep the node copyable because instructions,
modifiers, and global phase own values. In `Qiskit2_5.cpp`, normalize a number
or direct symbol immediately. For a parameter expression, replay `_qpy_replay`
into a bounded stack. Normalize reverse binary opcodes by swapping their
operands. Reject malformed stacks, non-finite constants, unsupported functions,
excessive depth, and excessive node count before returning to generic import.
Read a `for` parameter through the public control-flow operation so its UUID is
preserved.

Next, change `QiskitImport.cpp` to validate every tree leaf by identity and to
emit each supported node as an `f64` Arith or Math value. Register the Math
dialect in the import context. Key both local and global parameter maps by
identity. Remove the numeric-only custom-definition check in the version
adapter; the existing recursive definition preflight then validates its actual
symbols and expressions against the same maps.

Then change `QiskitExport.cpp` to recognize compiler inputs, finite constants,
and the supported Arith and Math operations recursively. Cache each SSA result
so a shared compiler subexpression remains shared in the normalized tree.
Represent inverse angles through expression negation and combine all global
phase contributions through expression addition. Complete this preflight before
the writer allocates a destination circuit. In `Qiskit2_5.cpp`, recursively
construct `QkParam` values and reuse one cached Qiskit symbol for each compiler
input identity.

Finally, add focused Python regressions for direct and shared symbols, nested
binary and unary expressions, reverse operators, partial binding, global phase,
parameterized custom definitions, lexical name collisions, supported manual MLIR
expression export, explicit vector-element rejection, and fail-closed
unsupported input. Update only the support table and concise surrounding text.
Mark the prior numeric-only decision in
`.agent/plans/qiskit-circuit-translation.md` as superseded by this plan. Keep
changelog prose unchanged and add pull request #2150 to the existing Qiskit
translation entry.

## Concrete Steps

Run all commands from the repository root. Build the changed binding after each
production batch:

    cmake --build build/release --target mqt-core-mlir-bindings --parallel 2

Run the focused translation tests in a synchronized environment that builds and
installs the current worktree for parent and child processes:

    uvx nox -s tests-3.13 -- -q -o addopts= test/python/test_mlir_qiskit_translation.py

Build the MLIR reference documentation and the complete Sphinx documentation:

    cmake --build --preset release --target mlir-doc
    uvx nox --non-interactive -s docs

Finish with generated-stub verification, repository lint, and whitespace
validation:

    uvx nox -s stubs
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

Import a Qiskit circuit with two shared free symbols in nested arithmetic, gate
arguments, and global phase. The QC entry function must have one named `f64`
argument per symbol and must contain the matching Arith and Math operations.
Export it, bind the parameters, and compare its numeric operator and global
phase with the source circuit.

Import partially bound expressions and a parameterized custom gate. Both must
resolve the remaining symbols without source mutation. Import a `for` loop whose
binder has the same displayed name as a distinct free symbol used in its body.
The gate must use the free function argument, not the loop induction value.

Export hand-written QC with supported `f64` Arith and Math expressions. The
result must contain shared Qiskit parameters and bind to the same numeric
values. Duplicate or unused named inputs, unsupported SSA operations,
unsupported Qiskit functions, non-finite constants, malformed trees, and
excessive expressions must fail during preflight.

Reject a `ParameterVectorElement` before module construction and leave the
source circuit unchanged. Continue to accept standalone scalar parameters whose
names contain brackets without inferring vector semantics.

## Idempotence and Recovery

All build and test commands are repeatable. Build artifacts remain under
`build/` and are not committed.

If `main` advances before publication, rebase this scalar commit first. Preserve
the CBit resource model and the symbolic-expression fields and paths in
overlapping Qiskit translation files, then restack each dependent Qiskit commit.

Do not push, open a pull request, edit issue text, or post review replies
without fresh human authorization. Preserve unrelated worktree changes.

## Artifacts and Notes

The Qiskit 2.5 replay opcodes required by this implementation are addition,
subtraction, multiplication, division, power, their reverse forms, sine, cosine,
tangent, inverse sine, inverse cosine, inverse tangent, exponential, logarithm,
absolute value, and conjugation. Reverse subtraction, division, and power swap
the replay operands before creating the generic tree. Real conjugation is an
identity operation. Other replay opcodes fail with their operation name in the
diagnostic.

The Release compiler suite passed all 133 tests before the final rebase. A fresh
nanobind 2.15.0 and Qiskit 2.5.2 build passed all 158 focused scalar-symbol
Qiskit translation tests after #2158 merged. Rebasing onto `cb5cf0103` after
pull request 2173 only relocated the changelog entry. The production source tree
did not change. The focused Clang format, Ruff, Rumdl, and committed-diff checks
also pass. Stub generation, the warnings-as-errors documentation build,
repository lint, and focused clang-tidy checks remain part of publication
validation.

## Interfaces and Dependencies

`Parameter` in `QiskitTranslation.h` is a copyable immutable tree with a kind,
finite numeric value or symbol name and identity, and zero, one, or two child
pointers. `Loop::parameter` is `std::optional<Parameter>` and must contain a
symbol when present. `CircuitReader` returns normalized trees for instruction
parameters and global phase. `CircuitWriter` accepts the same tree and
reconstructs Qiskit parameters with the version-specific C and public Python
APIs.

No SymPy dependency is added. No Qiskit object or expression string is stored in
MLIR. The supported compiler operations remain frontend-neutral Arith and Math
dialect operations on `f64` values.

Revision note (2026-08-19): Split exact vector provenance into a separate
follow-up and aligned this plan with the scalar-symbol contract on #2158.
