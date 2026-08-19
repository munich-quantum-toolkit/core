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
introduced by #2031 and builds on the dense-unitary support merged in #2136. It
must not weaken the existing preflight checks, mutate input circuits, or expose
a partially constructed output circuit after a failure.

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
- [x] (2026-08-18 14:45Z) Preserve `ParameterVector` group identity, size, and
  numeric index without inferring groups from bracketed standalone names.
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
  disagree on their name or vector metadata. The importer must compare all
  canonical symbol metadata and reject such aliases before creating a module.
- Observation: a custom gate's definition already contains the actual symbols or
  expressions supplied at its call site. The importer does not need a separate
  formal-parameter substitution scheme. It must validate the definition against
  the current global and lexical identities.
- Observation: an expression can convert to a number while still tracking free
  parameters. The version-specific reader must inspect `parameters` before it
  treats a value as a numeric constant.
- Observation: ordinary names such as `theta[10]` do not retain Qiskit
  `ParameterVector` ordering, while parsing brackets would misclassify a valid
  standalone `Parameter("theta[10]")`. Genuine vector elements therefore carry
  explicit group identity, name, size, and numeric index metadata.

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
  scalar name. Genuine vector elements additionally use `mqt.input_group`,
  `mqt.input_group_name`, `mqt.input_group_index`, and `mqt.input_group_size`;
  never infer grouping from a displayed name. Rationale: identity prevents
  lexical capture, and explicit provenance preserves vector ordering and binding
  without misclassifying standalone symbols. Date/Author: 2026-08-18 / Codex.
- Decision: Preserve symbol sharing but do not preserve Qiskit's original UUID
  across a round trip. Rationale: the compiler input is the frontend-neutral
  identity. The writer creates exactly one Qiskit symbol for each input and
  reuses it throughout gates and global phase. Date/Author: 2026-08-18 / Codex.
- Decision: Bound normalized expression depth and node count before compiler or
  circuit construction. Rationale: the existing definition and control-flow
  readers are bounded, and parameter replay must have the same fail-closed
  behavior for adversarial input. Date/Author: 2026-08-18 / Codex.
- Decision: Reconstruct vectors with their recorded size but create every used
  `ParameterVectorElement` directly instead of indexing the vector. Rationale:
  direct construction preserves vector-level and positional binding and does not
  enlarge the vector when a valid sparse element lies beyond its recorded size.
  Date/Author: 2026-08-18 / Codex.
- Decision: Reject parameter-vector groups larger than 65,536 elements during
  import and export preflight, and reject translations whose distinct groups
  declare more than 65,536 elements in total. Rationale: the individual and
  aggregate bounds prevent crafted metadata from requesting unbounded vector
  allocation. Date/Author: 2026-08-18 / Codex.

## Outcomes & Retrospective

The implementation and local validation are complete. Shared direct symbols,
bounded real expression trees, parameterized definitions, identity-safe loop
bindings, global phase, and `ParameterVector` provenance pass their focused
tests. The branch is ready for review on the current `main` branch.

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
compiler input. Vector elements carry the optional, all-or-none group identity,
name, index, and size attributes listed in the Decision Log. The compiler
representation uses `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, and
`arith.negf`, plus matching real-valued Math dialect operations. A local `for`
induction parameter is a temporary SSA value keyed by the loop parameter's
Qiskit identity. It is not a function input.

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
input identity. After native circuit creation, replace grouped symbols with
public `ParameterVectorElement` objects belonging to a shared `ParameterVector`
of the recorded size.

Finally, add focused Python regressions for direct and shared symbols, nested
binary and unary expressions, reverse operators, partial binding, global phase,
parameterized custom definitions, lexical name collisions, supported manual MLIR
expression export, vector ordering and vector-level binding, and fail-closed
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
values. Duplicate input names, unsupported SSA operations, unsupported Qiskit
functions, non-finite constants, malformed trees, and excessive expressions must
fail during preflight.

Round-trip a vector containing at least 12 elements, a sparse vector, two
distinct same-name vectors, and a standalone bracketed parameter name. Both
positional circuit binding and `{ParameterVector: values}` binding must remain
equivalent. Sparse elements must preserve the recorded vector size and must not
enlarge that size from their index.

## Idempotence and Recovery

All build and test commands are repeatable. Build artifacts remain under
`build/` and are not committed.

If `main` changes before review, rebase the single symbolic commit onto the new
head. Preserve both dense-unitary and symbolic-expression fields and paths in
overlapping Qiskit translation files.

Do not push, open a pull request, edit issue text, or post review replies
without fresh human authorization. Preserve unrelated worktree changes.

### Artifacts and Notes

The Qiskit 2.5 replay opcodes required by this implementation are addition,
subtraction, multiplication, division, power, their reverse forms, sine, cosine,
tangent, inverse sine, inverse cosine, inverse tangent, exponential, logarithm,
absolute value, and conjugation. Reverse subtraction, division, and power swap
the replay operands before creating the generic tree. Real conjugation is an
identity operation. Other replay opcodes fail with their operation name in the
diagnostic.

The final Release build passed together with 133 compiler tests. A fresh
nanobind 2.15.0 and Qiskit 2.5.2 Nox build passed all 165 focused Qiskit
translation tests. Stub generation, the warnings-as-errors documentation build,
repository lint, focused clang-tidy checks, and `git diff --check` also passed.

### Interfaces and Dependencies

`Parameter` in `QiskitTranslation.h` is a copyable immutable tree with a kind,
finite numeric value or symbol name and identity, optional source-level group
identity/name/index/size, and zero, one, or two child pointers.
`Loop::parameter` is `std::optional<Parameter>` and must contain a symbol when
present. `CircuitReader` returns normalized trees for instruction parameters and
global phase. `CircuitWriter` accepts the same tree and reconstructs Qiskit
parameters with the version-specific C and public Python APIs.

No SymPy dependency is added. No Qiskit object or expression string is stored in
MLIR. The supported compiler operations remain frontend-neutral Arith and Math
dialect operations on `f64` values.

Revision note (2026-08-18): Recorded complete expression, vector, and lexical
behavior plus final functional acceptance evidence.
