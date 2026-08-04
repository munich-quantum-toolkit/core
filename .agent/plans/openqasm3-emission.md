# Emit OpenQASM 3 from QC compiler programs

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core can currently translate OpenQASM 3 into the MLIR QC dialect, optimize
that program through QCO, and emit QIR, but it cannot turn the resulting QC
program back into a standardized circuit language. After this work, C++ and
Python clients can request an `OpenQASMProgram`, and `mqt-cc` can write OpenQASM
text with `--emit=openqasm3`. A user can demonstrate the feature by compiling a
Bell-state program to OpenQASM, parsing the emitted source with the existing
strict frontend, and compiling the reparsed program onward.

The exporter is deliberately independent of the legacy `include/mqt-core/qasm3`,
`src/qasm3`, `include/mqt-core/ir`, and `src/ir` components. It reads the QC,
arith, math, memref, func, and SCF operations already present in an MLIR module.
Programs whose meaning relies on dynamic-index safety checks are rejected
instead of reverse-engineering the importer machinery.

## Progress

- [x] (2026-08-04 11:29Z) Refreshed `origin/main`, allocated a clean isolated
      worktree, and read the workspace, repository, AI-use, and ExecPlan
      policies.
- [x] (2026-08-04 11:29Z) Inspected the compiler program API, default pipeline,
      Python binding entry point, CLI driver, QC operation interfaces, and
      existing OpenQASM frontend translation target.
- [x] (2026-08-04 13:16Z) Implemented the buffered QC-to-OpenQASM translator and
      focused unit tests for gates, structured control, outputs, and explicit
      failure boundaries.
- [x] (2026-08-04 13:16Z) Added `OpenQASMProgram`, direct QC export, and the
  optimized default-pipeline output stage.
- [x] (2026-08-04 13:16Z) Bound the output artifact in Python and exposed
  `--emit=openqasm3` in `mqt-cc`.
- [x] (2026-08-04 13:16Z) Added strict round-trip, matrix-equivalence, Python,
  CLI, buffered-failure, and file-output coverage.
- [x] (2026-08-04 13:16Z) Documented the feature, updated the changelog,
  regenerated bindings, and completed release, test, documentation, and lint
  validation.
- [x] (2026-08-04 13:16Z) Refreshed `origin/main`, confirmed it remains at the
  implementation base, and completed self-review without modifying remote
  state.
- [x] (2026-08-04 16:35Z) Remediated PR feedback: replaced nonstandard gate
      helpers with `inv @ sx`, `u2`, and `U`; implemented native OpenQASM switch
      parsing, semantic analysis, QC lowering, and emission; and rewrote the
      user documentation with input before output.
- [x] (2026-08-04 16:35Z) Cleared changed-file clang-tidy findings and raised
      focused translator line coverage from 75% to 89.1% with tests for
      structured results, loop-carried state, type inference, casts, and
      explicit rejection boundaries.
- [x] (2026-08-04 16:35Z) Rebuilt release and passed 154 OpenQASM target tests,
      284 QC translation tests, 219 compiler tests, and 40 Python MLIR tests.
- [x] (2026-08-04 18:20Z) Addressed the independent review's switch-budget and
      changelog findings, rebased onto `origin/main` at `2e0778f9d`, and passed
      the complete release build plus focused release and coverage suites.
- [x] (2026-08-04 18:20Z) Revalidated all changed translation units with
      clang-tidy, regenerated Python stubs without a diff, and completed the
      repository lint and warnings-as-errors documentation builds.
- [x] (2026-08-04 19:35Z) Added targeted emitter boundary tests after the
      separate Codecov C++ patch check reported 88.1%, raising focused
      translator line coverage to 92.1% and covering structured-control
      rejection propagation without broadening the supported subset.
- [x] (2026-08-04 20:44Z) Fixed the Windows-only iterator deduction failure and
      ignored-result warnings, simplified ECR to one controlled-X plus local
      corrections, and removed the redundant exporter-specific dead-expression
      traversal in favor of the existing QC cleanup pipeline.
- [x] (2026-08-04 20:44Z) Made `OpenQASMProgram` a direct compiler input, moved
      shared output-attribute names to the existing dialect utilities, and
      reduced imported type metadata to the scalar-bit and unsigned-integer
      distinctions that QC otherwise erases.
- [x] (2026-08-04 20:56Z) Rebuilt release and coverage configurations; passed
      287 QC translation tests, 219 compiler tests, the 15 affected Python
      emission/input and gate-matrix tests, changed-source clang-tidy, complete
      repository lint, and the warnings-as-errors documentation build.
- [x] (2026-08-04 21:10Z) Addressed the independent PR review's two
      documentation findings, rebased onto `origin/main` at `a132638c8`, and
      revalidated the 22 focused emitter tests, compiler integration test, and
      15 affected Python tests against the rebased release build.

## Surprises & Discoveries

- Observation: The compiler API and `mqt-cc` currently implement the same
  output-stage decisions separately. Evidence: `runDefaultPipeline` in
  `mlir/lib/Compiler/Programs.cpp` converts optimized QCO back to QC before QIR,
  while `mlir/tools/mqt-cc/mqt-cc.cpp` constructs the equivalent pass sequence
  directly.
- Observation: QC modifiers already implement `qc::UnitaryOpInterface` and their
  regions are verified to contain only the small set of operations valid in a
  unitary body. This permits a uniform exporter without depending on the legacy
  circuit classes.
- Observation: OpenQASM helper-gate formals need ordinary identifiers such as
  `p0`; reusing SSA-derived names made strict reparsing unnecessarily fragile.
  Full-matrix Qiskit comparisons also showed that OpenQASM's built-in `U` and
  the QC `u`/`u2` operations agree without an extra global phase.
- Observation: The current strict frontend accepts nested `else { if (...) }`
  control but not explicit scalar-cast syntax. The exporter therefore emits
  standard OpenQASM casts and documents cast-containing programs outside the
  current strict round-trip subset.
- Observation: Importing an integer `while` condition introduces checked
  arithmetic and safety operations. Direct safety-free, type-preserving
  `scf.while` is exportable, while the importer-produced checked form correctly
  reaches the documented unsupported boundary.
- Observation: Removing returned measurements during QC-to-QIR changed a
  function's result arity without realigning result attributes. Preserving
  OpenQASM output hints exposed the latent Func-to-LLVM assertion. Realigning
  the ordinary MLIR result-attribute array at that existing arity-changing
  operation fixes the issue without a metadata side channel or preservation
  pass.
- Observation: A documentation-only top-level CMake configuration currently
  assumes binding targets exist. The final documentation validation generated
  MLIR reference pages with bindings enabled, then ran Sphinx in nitpicky,
  warnings-as-errors mode.
- Observation: The OpenQASM standard library defines `sx` and the compatibility
  alias `u2`, but not `sxdg`; the language-level `U` gate covers QC `u`.
  Emitting `inv @ sx`, `u2`, and `U` removes three unnecessary helper
  definitions.
- Observation: OpenQASM switch cases permit multiple constant integer labels and
  do not fall through. Representing them directly as `scf.index_switch`
  preserves structured control and eliminates the exporter's nested-if
  reconstruction.
- Observation: Lowering a source case with several labels creates one
  `scf.index_switch` region per label. The importer's projected-emission
  preflight must therefore multiply the case body cost by its label count; doing
  so keeps the existing operation budget effective before constructing any IR.
- Observation: MSVC models `llvm::find_if` over `std::array` as an array
  iterator, not a pointer. Removing the ECR helper dependency lookup eliminated
  both the non-portable pointer deduction and an unnecessary two-RZX
  decomposition.
- Observation: The importer only erases two output-type distinctions needed by
  the exporter: scalar `bit` versus `bit[1]`, and `uint` versus signless `i64`.
  Boolean, signed integer, floating-point, and bit-array outputs are inferred
  directly from QC types and operations.

## Decision Log

- Decision: Emit OpenQASM 3.1 with `stdgates.inc`, using standard gate names
  where possible and self-contained helper gate declarations for the remaining
  QC gates. Rationale: This keeps emitted programs portable and parseable in
  strict mode without MQT-specific language extensions. Date/Author: 2026-08-04
  / Codex.
- Decision: Buffer the complete translation before writing to a caller stream.
  Rationale: Unsupported operations must not leave a syntactically truncated
  output file. Date/Author: 2026-08-04 / Codex.
- Decision: Treat dynamic memory indices, `cf.assert`, live `ub.poison`, and
  checked-index scaffolding as unsupported. Rationale: The user explicitly
  prefers a focused practical exporter over reconstruction of importer safety
  machinery. Date/Author: 2026-08-04 / Codex.
- Decision: Make `OpenQASMProgram` an owned textual value rather than an MLIR
  `Program` subclass, and accept that value directly as compiler input.
  Rationale: OpenQASM owns source text rather than an MLIR context and module;
  reparsing it at the compiler boundary is straightforward and keeps the value
  reusable. Date/Author: 2026-08-04 / Codex.
- Decision: Export from optimized QC in the coordinated pipeline but expose a
  non-consuming direct method on `QCProgram`. Rationale: The compiler output
  should reflect normal optimization, while callers inspecting frontend QC need
  a predictable direct path. Date/Author: 2026-08-04 / Codex.
- Decision: Preserve output names and add output-kind metadata only for scalar
  `bit` and `uint`. Rationale: QC types and defining operations infer all other
  supported output kinds. Standard MLIR result attributes retain the two
  genuinely erased distinctions without an exporter-specific side table or
  preservation pass. Date/Author: 2026-08-04 / Codex.
- Decision: Run the existing QC cleanup pipeline on a copy before direct
  `QCProgram` export. Rationale: Dead importer scaffolding should be removed by
  MLIR passes, not by a recursive exporter heuristic, while the caller's QC
  program and the low-level translator's strict validation contract remain
  unchanged. Date/Author: 2026-08-04 / Codex.
- Decision: Split the modern OpenQASM translations into
  `MLIRQCOpenQASMTranslation` and retain the legacy circuit translations in
  `MLIRQCTranslation`. Rationale: Compiler clients can use either OpenQASM
  direction without linking `MQT::CoreIR`, while existing users retain the
  aggregate target. Date/Author: 2026-08-04 / Codex.
- Decision: Parse and emit native `switch`/`case`/`default` statements and map
  them directly to `scf.index_switch`. Rationale: This follows the language
  construct, supports carried state, and is simpler than synthesizing nested
  conditionals. Date/Author: 2026-08-04 / Codex.

## Outcomes & Retrospective

The implementation now provides a deterministic OpenQASM 3.1 boundary format for
practical QC programs through the translation API, compiler artifact, Python
bindings, and `mqt-cc`. It handles static logical and physical qubits,
measurement and classical outputs, the QC gate set and nested modifiers,
printable scalar expressions, and structured SCF control with simultaneous
state-update semantics. Extended gates are emitted as focused private
definitions only when used.

The importer retains output names and only the scalar-bit and unsigned-integer
kind distinctions that QC otherwise erases. Ordinary MLIR result-attribute
transport carries them through QC/QCO; the exporter infers the remaining
supported kinds. Dynamic indices, runtime safety machinery, arbitrary CFGs, and
other deliberately unsupported categories fail with location-based diagnostics
before buffered output is committed.

Validation completed after rebasing onto `origin/main` at `2e0778f9d`:

- the complete release build succeeded;
- all 155 OpenQASM frontend tests, 287 QC translation tests, and 219 compiler
  tests passed in both release and coverage builds;
- all 40 Python MLIR tests passed, including 14 full-matrix helper-gate
  comparisons and compiler round trips;
- generated Python stubs completed successfully without a diff;
- CLI file emission followed by strict re-import completed successfully;
- Sphinx completed in nitpicky warnings-as-errors mode after generating the MLIR
  reference pages;
- all changed translation units completed clang-tidy without findings;
- focused translator line coverage is 91.7% (993 of 1083 lines), up from 75%;
- the repository-wide lint session and `git diff --check` passed;
- an independent `$mqt-pr-review` pass found no remaining correctness, API,
  documentation, C++20, MLIR-style, or scope findings after its three
  publication blockers were resolved.

## Context and Orientation

`mlir/include/mlir/Compiler/Programs.h` defines typed compiler artifacts and the
`ProgramFormat` enum. `mlir/lib/Compiler/Programs.cpp` imports OpenQASM,
performs QC/QCO/QIR conversions, and coordinates the default pipeline. The new
textual artifact and pipeline branch belong there.

`mlir/include/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h` and
`mlir/lib/Dialect/QC/Translation/` contain the modern OpenQASM frontend and its
QC emitter. The reverse translator will live beside them under the separate
public header `TranslateQCToOpenQASM3.h` and will be compiled into
`MLIRQCOpenQASMTranslation`. It may use MLIR and QC libraries already linked by
that target but must not include or link the legacy QASM or circuit libraries.
The existing `MLIRQCTranslation` target remains an aggregate for legacy users.

`bindings/mlir/register_mlir.cpp` exposes compiler artifacts through nanobind.
`bindings/patterns.txt` supplies handwritten overloads to the stub generator.
Generated `.pyi` files are updated only by `uvx nox -s stubs`.

`mlir/tools/mqt-cc/mqt-cc.cpp` is the standalone driver. It imports QASM or
MLIR, runs QC-to-QCO optimization, converts requested outputs, and writes
through LLVM output streams. OpenQASM must be treated as a textual QC-stage
output rather than MLIR bytecode.

The QC dialect represents logical scalar qubits with `qc.alloc`, physical qubits
with `qc.static`, and registers as rank-one memrefs containing `!qc.qubit`.
Gates implement `qc::UnitaryOpInterface`, which exposes parameters, targets, and
controls. Classical structured control uses MLIR SCF operations. The exporter
maps SSA values to deterministic generated OpenQASM variables.

## Plan of Work

First, add the translation header and a buffered implementation. Validate that
the module contains one defined, argument-free function with a single entry
block and only supported nested SCF regions. Pre-assign deterministic names for
qubit allocations, physical qubits, classical storage, SSA temporaries, and
outputs. Render expressions with explicit precedence and render statements
through an indented LLVM stream.

Emit direct standard gates and portable helper declarations for non-standard QC
gates. Print modifier operations using OpenQASM `ctrl`, `inv`, and `pow` syntax.
If a modifier body contains more than one unitary operation, create a
deterministically named helper gate whose parameters and qubits are its explicit
arguments. Add matrix-oriented and strict-frontend round-trip tests for the gate
surface.

Render statically indexed qubit and bit storage, measurements, reset, barrier,
arithmetic, comparisons, scalar conversions, and math expressions. Render
single-block `scf.if`, constant-bound `scf.for`, expression-only `scf.while`,
and `scf.index_switch`. Declare result variables before structured statements
and assign yielded values through fresh next-state temporaries before updating
loop-carried variables.

Add optional function-result attributes `qc.openqasm.output_name` and
`qc.openqasm.output_kind` in the frontend so output names and the
`bit`/`bool`/signedness distinction survive conversion. The exporter must also
handle QC without these hints, using operation provenance where unambiguous and
diagnosing user-visible ambiguity.

Add `OpenQASMProgram`, direct QC export, the program format enum entry, and the
default pipeline branch. Then add the nanobind class, enum member, stub
overloads, and the `mqt-cc` text writer. Tests must prove string and file output
and that default compilation reaches the optimized-QC output stage.

Finally, expand `docs/mlir/OpenQASM.md`, the compiler collection page, the MLIR
overview, and `CHANGELOG.md`. Document exact supported constructs and clear
failure boundaries, especially dynamic indexing and importer safety machinery.

## Concrete Steps

All commands run from the repository root through `.agent/run.sh` when they
create build or tool caches.

Configure and build the focused targets:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-mlir-unittest-qc-translation \
      mqt-core-mlir-unittests-compiler mqt-cc

Run the focused native tests:

    ./build/release/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Run Python, generated-artifact, documentation, and lint validation:

    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check

Exercise the CLI with a temporary Bell-state QASM file, emit text to stdout and
to a second temporary file, then feed that file back to `mqt-cc` with
`--input-format=qasm`. Successful commands exit zero; the emitted file begins
with `OPENQASM 3.1;`.

## Validation and Acceptance

The public translator succeeds for a representative QC module and returns
strictly parseable OpenQASM 3.1. Passing its text to `translateQASM3ToQC`
produces a verified QC module. Gate tests cover each standard or generated gate
and modifier nesting; helper definitions are compared against the QC unitary,
including global phase.

Programs with measurement, output values, arithmetic gate parameters, reset,
barrier, nested conditionals, constant-range loops, while loops, switches, and
carried scalar/bit state emit structured OpenQASM rather than unrolled or
flattened control flow. Tests compare observable operations and result kinds
rather than exact incidental temporary names, while a separate determinism test
checks byte-identical repeated emission.

A module with `cf.assert`, live poison, dynamic memref indexing, an unsupported
type, arbitrary CFG, multiple functions, calls, or an unknown operation fails
with a location-based diagnostic. The destination stream remains empty.

`QCProgram::toOpenQASM3()` runs the existing QC cleanup pipeline on a copy and
returns an `OpenQASMProgram` without consuming or mutating the QC program.
`runDefaultPipeline(..., ProgramFormat::OpenQASM3)` returns the textual program
after the normal QCO optimization round trip. An `OpenQASMProgram` may also be
passed directly as compiler input and remains reusable. Python exposes the same
behaviors and `mqt-cc --emit=openqasm3` writes plain text to stdout and files.

The focused native and Python suites pass, generated stubs match the binding,
documentation builds, lint passes, and `git diff --check` reports no whitespace
errors.

## Idempotence and Recovery

Source edits and tests are additive and can be rerun. CMake configuration and
build commands are idempotent in the worktree-local `build/release` directory.
Stub generation is repeatable and is the only supported way to update generated
Python interface files.

If translation work exposes a malformed module, preserve the failing test and
return a diagnostic rather than weakening validation. If a broad check fails
because dependencies or network access are unavailable, retain the focused
native evidence, record the exact failure here, and do not alter source merely
to make an environmental check green. Never reset, clean, or modify another task
worktree.

## Artifacts and Notes

The initial implementation base is commit
`47a25e76087f1c44cf2c622c2b628c1b57e2f7a6`, whose subject is
`✨ Compose the compiler target pipeline (#1999)`.

The expected minimal emitted Bell program has this shape:

    OPENQASM 3.1;
    include "stdgates.inc";

    qubit[2] _mqt_q;
    bit[2] _mqt_c;
    h _mqt_q[0];
    cx _mqt_q[0], _mqt_q[1];
    _mqt_c[0] = measure _mqt_q[0];
    _mqt_c[1] = measure _mqt_q[1];

Exact generated names may differ, but they must be deterministic and
collision-safe.

## Interfaces and Dependencies

The final C++ translation interface is:

    namespace mlir::qc {
    LogicalResult translateQCToOpenQASM3(ModuleOp moduleOp,
                                         llvm::raw_ostream& output);
    FailureOr<std::string> translateQCToOpenQASM3(ModuleOp moduleOp);
    }

`OpenQASMProgram` owns a `std::string`, provides `source()` and `write(path)`,
and appears in both `CompilerProgram` and `CompilerInput`. `QCProgram` provides
`std::optional<OpenQASMProgram> toOpenQASM3() const`.

The translation target may link MLIR Arith, ControlFlow, Func, Math, MemRef,
SCF, UB, and QC libraries, plus LLVM Support. It must not link the legacy
`mqt-core-qasm`, `mqt-core-ir`, or circuit targets.

Revision note (2026-08-04): Created the initial self-contained implementation
plan after inspecting the exact compiler and dialect architecture.
