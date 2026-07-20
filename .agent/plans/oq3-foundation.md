# Complete a direct, specification-driven OpenQASM-to-QC frontend

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept current as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core must accept as many valid OpenQASM programs as its compiler can
faithfully process, diagnose unsupported language features at the QC emission
boundary, and carry every accepted program through the complete compiler. After
this work, a user can give OpenQASM to `QCProgram::fromQASMString`, receive
verified QC directly, optimize it in QCO, optionally serialize and deserialize
it through `jeff`, convert it back to QC, and obtain valid QIR. The standard
acceptance path is QC to QCO to reconstructed QC to QIR; `jeff` compatibility is
tracked independently and never blocks valid QC or QIR. The unit tests make that
path observable stage by stage.

The parser and semantic analyzer remain independent of MLIR and continue to
recognize valid source even when the selected compiler target lacks a concept.
The gate `pow @` modifier is the defining example: parsing and semantic analysis
succeed, but the direct QC emitter reports a source-located unsupported-feature
diagnostic until QC gains power semantics. There is no OQ3 MLIR dialect or
OQ3-to-QC conversion. An intermediate dialect that cannot proceed through the
compiler adds maintenance and test surface without user value.

The scope is the staged frontend under `mlir/include/mlir/Target/OpenQASM` and
`mlir/lib/Target/OpenQASM`, direct QC translation under
`mlir/lib/Dialect/QC/Translation`, the OpenQASM fixture corpus and compiler
tests, and only those existing conversion files for which a full-chain fixture
demonstrates a real defect. Preserve the legacy `QuantumComputation` parser and
unrelated behavior. Do not push or publish GitHub text under this plan.

## Progress

- [x] (2026-07-15) Replaced the legacy stream adapter with an LLVM-native lexer,
      grammar-only parser, persistent syntax tree, and separate semantic
      analyzer.
- [x] (2026-07-16) Implemented source locations and includes, lexical scope,
  types, definite initialization, assignments, expressions, custom gates,
  broadcasting, dynamic indices, measurement, `if`, `for`, `while`, and
  loop-carried scalar and bit state in the staged frontend.
- [x] (2026-07-16) Ported 27 healthy behavior fixtures from the earlier OpenQASM
      implementation and completed a cleanup and clean-build validation of that
      implementation.
- [x] (2026-07-16) Re-evaluated the architecture after review and concluded that
      the reduced OQ3 dialect is unnecessary. Inspected the public compiler
      program APIs and identified the compiler paths that require separate
      acceptance contracts.
- [x] (2026-07-16) Revised this plan after critical review: isolate direct
      emission in a private emitter, build the full-chain harness before
      changing conversions, and require evidence-backed minimal regression
      fixes.
- [x] (2026-07-16) Removed the OQ3 dialect, OQ3-to-QC conversion, registrations,
      documentation, and dialect-specific tests, while relocating the gate
      catalog to the frontend.
- [x] (2026-07-16) Implemented direct typed-program-to-QC emission behind
      private `OpenQASMToQCEmitter` files and kept `TranslateQASM3ToQC.cpp` as a
      small public adapter.
- [x] (2026-07-16) Converted target tests from OQ3 inspection to direct QC
  behavior and precise target diagnostics, including rejection of `pow @`.
- [x] (2026-07-16) Defined a shared `{name, source}` OpenQASM compiler corpus
      and added public-API full-chain tests, including both direct composition
      and `runDefaultPipeline`.
- [x] (2026-07-16) Used failing full-chain stages to isolate the `jeff`
      entry-point round-trip defect and added a parser-independent native
      regression. Retained the structured QC-to-QCO changes with their existing
      native regressions.
- [x] (2026-07-16) Minimized the complete diff against `origin/main`, removing
      superseded OQ3 code, duplicated dispatch data, stale registrations, and
      iteration artifacts. Retained only the QC-to-QCO structured-state and
      `jeff` entry-point corrections backed by native regressions.
- [x] (2026-07-16) Added maintained parser, semantic, QC-emission, standard
      Adaptive, `jeff`, and Base support matrices.
- [x] (2026-07-16) Ran the affected frontend, translation, conversion, compiler,
      QIR, and legacy-parser tests; warning-as-error documentation; repository
      lint; diff checks; and sequential coverage. The substantive frontend and
      emitter surface reached 90.7 percent line coverage.
- [x] (2026-07-16) Incorporated the first post-implementation review: tightened
      the accepted-input contract for runtime indices, made structured custom
      gate capability checks transitive, added native result-bearing SCF
      conversion regressions and verified result types across `jeff`.
- [x] (2026-07-16) Ran the complete affected validation and repository checks
      after the first review fixes. Clean sequential coverage reached 91.0
      percent lines over the substantive frontend and emitter sources.
- [x] (2026-07-16) Incorporated the final review: replaced Boolean index
      resolvability with a small constant lattice, made literal branches and
      equal-constant joins precise, rejected multi-iteration induction indices,
      and collapsed dispatch and custom-gate expansion into one overflow-safe
      projected-emission budget.
- [x] (2026-07-16) Added a mutable floating-point `for`/`while` fixture to the
      initial `jeff` corpus, exact QIR output-recording assertions, and stronger
      native result-bearing `if`/`while` conversion semantics.
- [x] (2026-07-16) Closed the checked-integer acceptance gap by rejecting
      non-folded checked integer arithmetic and ranges at the QC boundary with a
      source-located diagnostic while preserving frontend support.
- [x] (2026-07-16) Ran final documentation, lint, architecture, affected unit,
      legacy-parser, and clean sequential coverage validation. Coverage is 89.9
      percent (4117/4579); the remaining lines do not justify test-only padding.
- [x] (2026-07-16) Re-read the complete effective branch diff after the final
      implementation commits. Removed the unused gate-policy field from the
      resolved program model and corrected stale plan claims; retained the
      production and regression surface required by the demonstrated contracts.
- [x] (2026-07-16) Repaired CI portability after Windows MSVC builds rejected
      Clang/GCC-only signed-overflow builtins in semantic constant evaluation.
      The evaluator now uses LLVM's portable overflow helpers. Restored the
      non-inheriting MLIR clang-tidy configuration after the rebase accidentally
      enabled the repository-wide checks and produced 1,194 unrelated reports.
- [x] (2026-07-16) Began the post-review correction by fixing the ten remaining
      clang-tidy diagnostics and moving the frontend's implementation headers
      from the library source directory to
      `mlir/include/mlir/Target/OpenQASM/Detail`. The owning containers remain
      standard vectors, while local bounded working sets continue to use LLVM
      small vectors.
- [x] (2026-07-16) Removed `jeff`-derived QC-emission rejection of runtime
      indices, varying induction indices, and non-folded integer expressions.
      Implemented faithful runtime signed and unsigned integer arithmetic,
      retained dynamic index bounds assertions, and replaced division-based
      range trip counts with constant structured bounds or comparison-driven
      inclusive loops.
- [x] (2026-07-16) Split compiler contracts into a 17-program standard
      QC-to-QCO-to-QC-to-QIR corpus, a 6-program `jeff`-compatible round-trip
      corpus, and four explicit `jeff`-boundary failure tests. Added MLIR's
      canonical `cf.assert`-to-LLVM lowering to QIR with a parser-independent
      regression so runtime source checks survive the standard path.
- [x] (2026-07-16) Rebuilt and ran all affected frontend, translation,
      conversion, compiler, QIR, and legacy-parser tests. The warning-as-error
      documentation session, repository lint, and diff checks pass.

## Surprises & Discoveries

- Observation: the old MLIR entry point copied the `SourceMgr` main buffer into
  `std::istringstream`, losing the source manager's include and location model.
  Evidence: the replacement parser consumes LLVM buffers directly and its
  persistent program retains included source identity.

- Observation: the legacy scanner recognized `for` and `while`, but its parser
  could not construct those statements. Evidence: the staged frontend now has
  source fixtures that produce and exercise standard SCF regions.

- Observation: valid source and target support are distinct, but a dialect is
  not required to preserve that distinction. Evidence: the typed semantic
  program already retains modifiers and source locations, so a QC emitter can
  reject `pow @` before creating target IR.

- Observation: the OQ3 dialect has shrunk to gate declarations, applications,
  and modifiers while classical computation and control flow already use
  standard MLIR. Its conversion mostly expands typed custom gates and maps a
  gate catalog to QC, work that can be performed directly from the typed model.

- Observation: the initial QC-to-QCO changes for structured classical and
  quantum state lacked independent evidence. The retained implementation is now
  justified by four parser-independent conversion regressions distilled from
  full-chain failures, rather than by the mere existence of structured OpenQASM.

- Observation: `runDefaultPipeline` covers QC to QCO optimization, QCO back to
  QC, and QC to QIR, but intentionally does not include a `jeff` round trip.
  Therefore the standard path is the acceptance contract; `jeff` has separate
  positive and explicit boundary-failure suites.

- Observation: structured control flow generally requires the Adaptive QIR
  profile, while straight-line circuits can exercise both Base and Adaptive
  profiles. Encoding expected failures as fixture flags would hide unsupported
  behavior, so the corpus contains only names and sources and profile grouping
  is expressed by the test suites that select it.

- Observation: `jeff` round trips preserved entry functions with observable bit
  results, but `JeffToQCO` restored the `entry_point` marker only for
  result-less functions. Evidence: the first explicit chain reached
  reconstructed QC but Adaptive QIR reported that no entry point existed. The
  native `jeff` regression now proves that nonempty result types and the marker
  survive together.

- Observation: the `jeff` representation cannot preserve the frontend's runtime
  `cf.assert` bounds checks. Evidence: genuinely runtime-dynamic indexing
  reaches verified QC, optimized QCO, reconstructed QC, and QIR, while
  QCO-to-`jeff` rejects `cf.assert`. The limitation is now reported at
  `intoJeff()` rather than by the source emitter.

- Observation: `scf.for` is an automatic allocation scope, so selecting the
  nearest such scope left result-bearing `scf.if` scratch storage inside a loop.
  Evidence: the nested native regression found the alloca in the loop body.
  Hoisting to the enclosing function allocates each conditional's storage once.

- Observation: successful final QIR alone does not prove that observable entry
  results survived intermediate formats. Evidence: the full-chain tests now
  compare entry result types in QC, optimized QCO, `jeff` bytes and restored
  `jeff`, reconstructed QCO, and reconstructed QC before checking the QIR status
  signature and output-recording calls.

- Observation: static loop bounds do not make a multi-iteration induction value
  static at each source use, but this is not a QC limitation. Evidence:
  `for uint i in [0:2] { x q[i]; }` now reaches QIR with runtime bounds checks
  and fails only at QCO-to-`jeff`.

- Observation: separate dynamic-dispatch and custom-gate expansion limits can
  each pass while their composition is excessive. Evidence: 4096 dispatch leaves
  applying a 25-operation custom gate project 102400 primitive emissions.

- Observation: non-folded checked integer expressions emit i128 arithmetic and
  `cf.assert` operations. MLIR's control-flow-to-LLVM conversion requires the
  assert pattern to be registered separately. Once registered, checked integer
  state reaches QIR; `jeff` still rejects it at its conversion boundary.

## Decision Log

- Decision: remove the OQ3 MLIR dialect and emit QC directly from the typed
  frontend program. Rationale: OpenQASM is compiler input, and successful import
  should mean that the program can enter the compiler's supported dialects.
  Unsupported target concepts remain diagnosable from the typed source model.
  Date/Author: 2026-07-16 / Codex.

- Decision: retain the staged lexer, syntax, and semantic design. Rationale:
  parsing, source-language validity, and target emission have different
  responsibilities, and includes, scope, and precise diagnostics already rely on
  that separation. Date/Author: 2026-07-16 / Codex.

- Decision: keep the existing `oq3::frontend` namespace unless a narrow rename
  is independently justified. Rationale: the namespace denotes the OpenQASM 3
  language frontend and is not itself an MLIR dialect; renaming every frontend
  type would add churn without changing behavior. Only dialect-specific
  identifiers must disappear. Date/Author: 2026-07-16 / Codex.

- Decision: put direct emission in private `OpenQASMToQCEmitter.h` and
  `OpenQASMToQCEmitter.cpp`, leaving `TranslateQASM3ToQC.cpp` as a small public
  adapter. Rationale: a large emitter should not obscure the stable translation
  entry points, and private files avoid exposing a second public API.
  Date/Author: 2026-07-16 / Codex.

- Decision: rename the frontend library target to `MLIROpenQASMFrontend`.
  Rationale: after emission moves to QC translation, the target contains only
  lexing, parsing, persistent syntax, semantic analysis, and the gate catalog;
  the name should describe that boundary. Date/Author: 2026-07-16 / Codex.

- Decision: keep custom-gate expansion limits and QC capability preflight in the
  emitter. Rationale: semantics validates source legality, including recursion,
  while expansion cost and target representability depend on the selected
  output. Date/Author: 2026-07-16 / Codex.

- Decision: reject unsupported gate modifiers before emitting any part of the
  affected application. Rationale: target failure must be precise and cannot
  silently alter or partially lower source semantics. Date/Author: 2026-07-16 /
  Codex.

- Decision: build the full compiler-chain corpus before altering downstream
  conversions and do not blanket-revert QC-to-QCO. Rationale: the current branch
  may contain both necessary and speculative hunks. Stage-specific failures and
  minimized native-IR regressions provide the evidence needed to retain,
  simplify, or remove each change safely. Date/Author: 2026-07-16 / Codex.

- Decision: conversion unit tests remain parser-independent. Rationale: a QC,
  QCO, `jeff`, or QIR conversion regression should construct or parse the
  smallest native MLIR that demonstrates the conversion invariant. OpenQASM
  belongs only in translation and compiler integration tests. Date/Author:
  2026-07-16 / Codex.

- Decision: share at most `{name, source}` across OpenQASM compiler fixtures.
  Rationale: per-fixture expected-failure or capability flags turn gaps into
  accepted behavior. Separate positive suites select the source subset they are
  required to support. Date/Author: 2026-07-16 / Codex.

- Decision: preserve observable `jeff` entry-point results when restoring QCO.
  Rationale: `jeff` serialization already retains those results; replacing them
  with a synthetic status code discarded program output and prevented the
  reconstructed QC module from reaching QIR. Result-less legacy entry points
  still receive the historical i64 status result. Date/Author: 2026-07-16 /
  Codex.

- Decision: accept runtime dynamic indices at the QC boundary and retain their
  bounds assertions. Rationale: QC and QIR represent the checks faithfully;
  inability to serialize them through `jeff` is an optional-format limitation
  reported by `intoJeff()`, not a reason to reject OpenQASM. Date/Author:
  2026-07-16 / Codex.

- Decision: retain the second structured-terminator conversion phase in
  QC-to-QCO. Rationale: result-bearing `if`, `for`, and `while` need the final
  region-local QCO value maps, and converting terminators in the first worklist
  makes correctness depend on traversal order. Four native regressions now cover
  the parent and terminator contracts. Date/Author: 2026-07-16 / Codex.

- Decision: compute structured-control capability transitively and memoize it
  per reachable custom gate. Rationale: modifiers on a wrapper around a looped
  gate are just as unsupported as modifiers on the looped gate itself, while
  unused definitions must have no effect on accepted source. Date/Author:
  2026-07-16 / Codex.

- Decision: lower all valid induction-variable indices with runtime bounds
  checks. Rationale: varying indices are first-class QC and QIR behavior;
  constant simplification is an optimization rather than an acceptance rule.
  Date/Author: 2026-07-16 / Codex.

- Decision: enforce one 100000-operation projected-emission budget that composes
  custom-gate expansion and register dispatch with overflow-safe multiplication.
  Rationale: emitted work, not either mechanism independently, is the relevant
  safety bound. Date/Author: 2026-07-16 / Codex.

- Decision: emit non-folded signed integer arithmetic with overflow assertions,
  unsigned arithmetic with 64-bit wrap semantics, and inclusive ranges without
  division-based trip counts. Rationale: these operations traverse the standard
  QC-to-QIR pipeline faithfully. Unsupported `jeff` serialization is tested at
  that boundary. Date/Author: 2026-07-16 / Codex.

- Decision: use constant `scf.for` bounds when all range components are known
  and a comparison-driven `scf.while` otherwise. Rationale: positive, negative,
  empty, singleton, non-divisible, and boundary ranges avoid endpoint overflow,
  while dynamic zero steps remain guarded by `cf.assert`. Date/Author:
  2026-07-16 / Codex.

- Decision: accept the measured 89.9 percent frontend and emitter line coverage
  without adding tests whose only purpose is crossing a round-number threshold.
  Rationale: final review identified no missing behavior-driven regression, and
  line-only tests would not improve evidence for the supported contracts.
  Date/Author: 2026-07-16 / Codex.

## Outcomes & Retrospective

The completed frontend groundwork is retained: the native parser and semantic
analyzer cover the source-language behavior needed by the compiler. The earlier
OQ3 target architecture has been removed in favor of direct QC emission.

The direct architecture and end-to-end behavior are implemented. Seventeen
OpenQASM fixtures traverse direct QC, QCO cleanup and optimization,
reconstructed QC, and Adaptive QIR; the same fixtures pass `runDefaultPipeline`.
Four straight-line fixtures also reach Base QIR. A separate six-program corpus
round-trips through `jeff`, while four explicitly tracked cases reach optimized
QCO and then fail at `intoJeff()`. The standard corpus includes runtime and
induction-variable indexing, checked integer state, and dynamic ranges in
addition to custom gates and structured control flow.

The downstream production corrections are constrained to demonstrated conversion
invariants. QC-to-QCO preserves classical results alongside linear quantum state
through `if`, `for`, and `while`, converts their terminators after region
contents, and allocates conditional scratch storage once per function.
`JeffToQCO` restores entry-point markers without losing observable results. Both
areas have parser-independent native regressions.

Runtime-dynamic indices, multi-iteration induction indices, and non-folded
integer expressions now produce verified QC and reach QIR. Signed operations
assert overflow and invalid division, unsigned operations wrap at 64 bits, and
dynamic ranges use comparison-driven structured control flow. MLIR's canonical
`cf.assert` conversion preserves these checks in QIR. Cases that `jeff` cannot
represent fail at its conversion boundary instead of reducing source support.

## Context and Orientation

`mlir/lib/Target/OpenQASM/Frontend.cpp` owns source buffers and orchestrates
parsing. Implementation headers live under
`mlir/include/mlir/Target/OpenQASM/Detail`; their corresponding sources under
`mlir/lib/Target/OpenQASM` implement tokenization, grammar, recovery, persistent
syntax, and semantic analysis. `OpenQASMSemantics.cpp` resolves syntax into the
`TypedProgram` declared in `mlir/include/mlir/Target/OpenQASM/Frontend.h`. These
files use LLVM support but do not require an `MLIRContext`. A `TypedProgram` is
a compact resolved representation containing expressions, conditions,
declarations, statements, gate definitions, source locations, and output
registers.

Direct QC construction lives in the private
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`. The reusable gate
metadata lives in `mlir/include/mlir/Target/OpenQASM/GateCatalog.h` and
`mlir/lib/Target/OpenQASM/GateCatalog.cpp`, where semantic analysis and target
emission share one authoritative catalog.

The stable user-facing translation functions are declared in
`mlir/include/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h` and implemented
in `mlir/lib/Dialect/QC/Translation/TranslateQASM3ToQC.cpp`. They accept either
an LLVM source manager or source text and return an owning reference to an MLIR
module. `QCProgram::fromQASMString` in `mlir/lib/Compiler/Programs.cpp` calls
this API. The translation source must stay small; a new private emitter beside
it owns all typed-program-to-QC construction.

QC uses reference-like qubits. QCO is the optimizer dialect and uses linear SSA
values, meaning each quantum operation returns the next value representing its
qubit. QC-to-QCO and QCO-to-QC bridge those models. `jeff` is a serializable
exchange representation reached from QCO. QIR is LLVM-based output reached from
QC. The compiler program wrappers in `mlir/include/mlir/Compiler/Programs.h`
provide ownership-safe transitions between these representations.

`mlir/unittests/programs/qasm_programs.cpp` and its header contain reusable
OpenQASM source fixtures. Translation equivalence tests live in
`mlir/unittests/Dialect/QC/Translation/test_qasm3_translation.cpp`. The complete
public compiler path belongs in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`. Tests directly attached to
QC-to-QCO, QCO-to-QC, `jeff`, and QIR must use their dialect-native builders or
small MLIR strings, not invoke the OpenQASM parser.

## Plan of Work

### Milestone 1: remove the intermediate dialect and establish direct emission

Delete `mlir/include/mlir/Dialect/OQ3`, `mlir/lib/Dialect/OQ3`,
`mlir/include/mlir/Conversion/OQ3ToQC`, `mlir/lib/Conversion/OQ3ToQC`, and
`mlir/unittests/Dialect/OQ3`. Remove their `add_subdirectory` entries, generated
operation dependencies, tool dialect registrations, unit-test registration, and
link libraries from the adjacent CMake files. Delete `docs/mlir/OQ3.md` and
remove its navigation entries. Do not remove `oq3::frontend` merely because its
name contains `oq3`; it is language code rather than a dialect identifier.

Move the gate catalog to `mlir/include/mlir/Target/OpenQASM/GateCatalog.h` and
`mlir/lib/Target/OpenQASM/GateCatalog.cpp`, retaining one authoritative table
for language gates, standard-library gates, compatibility aliases, canonical QC
primitives, parameter counts, control counts, target counts, variadic controls,
and inverse aliases. Update semantic includes and namespaces without duplicating
the table.

Rename the CMake library in `mlir/lib/Target/OpenQASM/CMakeLists.txt` from
`MLIROpenQASMTarget` to `MLIROpenQASMFrontend`. It contains `Frontend.cpp`, the
lexer, syntax, semantics, and `GateCatalog.cpp`, and links only what those
stages use. Remove `mlir/include/mlir/Target/OpenQASM/OpenQASM.h` and the old
emitter source after direct emission has replaced their behavior.

Add private `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.h` and
`OpenQASMToQCEmitter.cpp`. The header declares only a translation-internal
function that accepts a resolved `oq3::frontend::TypedProgram` and an
`MLIRContext` and returns `OwningOpRef<ModuleOp>`. `TranslateQASM3ToQC.cpp`
parses, analyzes, prints collected source diagnostics on failure, invokes this
private function, verifies the returned QC module, and contains no lowering
implementation.

The emitter reuses standard `arith`, `cf`, `func`, `math`, `memref`, `scf`, and
`ub` operations for classical behavior and emits QC operations directly. Port
only the target logic from OQ3-to-QC: catalog-to-primitive dispatch, implicit
and variadic controls, inverse aliases, the four-parameter `cu` phase behavior,
ordered inverse/positive-control/negative-control modifiers, and recursive
inlining of typed custom-gate bodies. Semantic analysis continues to reject
source-illegal recursion. The emitter preflights reachable custom-gate expansion
cost, target support, modifier operands, and structured custom-gate limitations
before creating each affected application. A `pow @` modifier produces a
source-located error and a null translation result; scalar exponentiation and
the scalar `pow()` function remain supported.

Acceptance for this milestone is a clean build with no OQ3 dialect or conversion
target and direct QC translation for existing supported sources. Repository
searches for `OQ3Dialect`, OQ3 operation class names, `createOQ3ToQCPass`, and
`MLIROQ3` must be empty. A search for `oq3::frontend` is not an acceptance
failure.

### Milestone 2: convert target tests to direct behavior

Refactor `mlir/unittests/Target/OpenQASM/test_openqasm.cpp` so parser tests
inspect parse results, semantic tests inspect `TypedProgram`, and target tests
inspect verified QC or emitted diagnostics. Remove all tests whose only purpose
is OQ3 operation verification. Preserve behavior tests for source ownership,
recovery, includes, scope, initialization, expressions, broadcasting, dynamic
dispatch, control flow, recursion, and cost bounds.

Add a positive direct-emission test for representative primitive and custom
gates and a negative test proving that a valid `pow @` program parses and
analyzes but `qc::translateQASM3ToQC` returns null with a source-located message
stating that QC power support is unavailable. Add equivalent focused cases for
every other frontend-accepted feature that the emitter rejects. Update
`mlir/unittests/Target/OpenQASM/CMakeLists.txt` to link `MLIROpenQASMFrontend`,
`MLIRQCTranslation`, and only directly used test libraries.

Keep exact QC equivalence tests in
`mlir/unittests/Dialect/QC/Translation/test_qasm3_translation.cpp`. They compare
canonicalized direct translation against QC builder references and should cover
catalog aliases, controls, inverses, custom-gate expansion, expressions,
broadcasting, measurement, and structured control flow where a stable reference
is practical.

### Milestone 3: build the full compiler-chain corpus

In `mlir/unittests/programs/qasm_programs.h` and `qasm_programs.cpp`, expose a
small shared corpus whose descriptors contain only a stable name and source.
Keep the sources themselves as the existing named constants where useful. Do not
attach capability flags to descriptors. Expose separate source groups for the
broad standard QC-to-QCO-to-QC-to-QIR path, the smaller `jeff` round trip, known
`jeff`-boundary failures, and the straight-line Base subset. Include nested
control flow, loop-carried state, runtime and induction-variable indexing,
checked integer arithmetic, dynamic ranges, gates, reset, barrier, and
observable outputs.

Add a parameterized integration suite to
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`. For every source in the
standard group, use the public APIs in exactly this order:

    QCProgram::fromQASMString
    QCProgram::intoQCO
    QCOProgram::cleanup
    QCOProgram::runPassPipeline("mqt-qco-default")
    QCOProgram::cleanup
    QCOProgram::intoQC
    QCProgram::cleanup
    QCProgram::intoQIR(QIRProfile::Adaptive)
    QIRProgram::llvmIR and QIRProgram::toBitcode

The `jeff`-positive group inserts serialization and deserialization between
optimized QCO and reconstructed QC. The incompatible group requires every stage
through optimized QCO to succeed and `intoJeff()` to fail. Retain copies at
ownership boundaries, require nonempty LLVM IR and valid bitcode, and run the
straight-line subset through Base and Adaptive QIR.

Add a separate parameterized call to `runDefaultPipeline` for every broad corpus
source, requesting Adaptive QIR and checking its LLVM IR and bitcode. This
proves the production default path independently; it does not replace the
explicit chain. Test failure messages must include the source name and stage.

### Milestone 4: isolate and fix demonstrated downstream defects

Run the full-chain corpus against the current conversion implementations. For
each failure, save the smallest native QC, QCO, or `jeff` MLIR that reproduces
the stage failure. Add that reduced program to the appropriate conversion unit
test using existing program builders when they express it cleanly, otherwise a
small MLIR string. Do not make these unit tests parse OpenQASM.

Inspect the branch diff in `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` hunk by
hunk. Retain a change only when a reduced regression proves that it is needed,
and simplify it to the smallest dialect-native correction. Pay particular
attention to SCF operands and results, region arguments, `scf.yield`,
`scf.condition`, measurement results, and the distinction between classical
state and linear quantum state. Apply the same evidence rule to
`mlir/lib/Conversion/QCOToQC`, `jeff` conversions, and QC-to-QIR. Do not edit a
downstream conversion merely because it was named in this plan.

After each fix, run its focused native conversion test first, then the failing
full-chain fixture, then the entire corpus. If a feature cannot be represented
faithfully by the current pipeline, move its failure to the direct QC emitter
only when the limitation is intrinsic to accepted compiler dialects rather than
a correctable conversion bug, and document the diagnostic and matrix status.
Never add a fixture flag that makes the integration test accept failure.

### Milestone 5: minimize, document, and validate

Inspect the effective diff against `origin/main`, including all commits and
unstaged files. Delete obsolete OQ3 concepts, duplicate gate dispatch,
superseded tests, stale target names, temporary compatibility wrappers,
iteration comments, and downstream conversion hunks lacking native regression
evidence. Keep `TranslateQASM3ToQC.cpp` small and keep production dependencies
pointing in one direction: QC translation depends on the OpenQASM frontend, not
the reverse.

Create `docs/mlir/OpenQASM.md` and link it from `docs/mlir/index.md` and the
relevant translation overview. It contains two maintained feature tables but
does not duplicate the language specification. The first covers parser and
semantic behavior. The second has columns for feature, Parse, Semantics, QC,
standard Adaptive QIR, `jeff`, Base, restriction or rejection reason, and the
representative test. Use precise statuses such as supported, recognized and
rejected semantically, or accepted by the frontend and rejected by QC. Mark
structured fixtures Adaptive-only and record Base support only for the tested
straight-line subset. List `pow @` as parsed and semantically valid but rejected
by QC. Update `CHANGELOG.md` to describe direct OpenQASM import without an OQ3
dialect claim.

Run formatting, all affected unit binaries, the legacy parser regression,
warning-as-error documentation, coverage, and repository lint after cleanup.
Record the final evidence in this plan's progress, discoveries, outcomes, and
artifacts sections.

## Concrete Steps

Run all commands from the repository root. Preserve unrelated changes and
inspect status before editing:

    git status --short --branch
    git diff --stat origin/main...HEAD
    git diff --stat origin/main

Configure a clean debug build using an installed MLIR 22.1 CMake package. The
path is supplied by the environment and must not be committed to this plan:

    MLIR_DIR=/path/to/mlir/lib/cmake/mlir cmake --preset debug

Build the direct frontend, translation, conversion, and compiler tests:

    cmake --build build/debug --target \
      mqt-core-mlir-unittest-openqasm-target \
      mqt-core-mlir-unittest-qc-translation \
      mqt-core-mlir-unittest-qc-to-qco \
      mqt-core-mlir-unittest-qco-to-qc \
      mqt-core-mlir-unittest-jeff-round-trip \
      mqt-core-mlir-unittests-compiler -j4

Run the binaries directly so stage failures are visible:

    ./build/debug/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
    ./build/debug/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation
    ./build/debug/mlir/unittests/Conversion/QCToQCO/mqt-core-mlir-unittest-qc-to-qco
    ./build/debug/mlir/unittests/Conversion/QCOToQC/mqt-core-mlir-unittest-qco-to-qc
    ./build/debug/mlir/unittests/Conversion/JeffRoundTrip/mqt-core-mlir-unittest-jeff-round-trip
    ./build/debug/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Build and run the QC-to-QIR Base and Adaptive test targets discovered under
`mlir/unittests/Conversion/QCToQIR`, and run all configured MLIR unit tests to
catch target-name or registration omissions:

    cmake --build build/debug --target mqt-core-mlir-unittests -j4
    ctest --test-dir build/debug --output-on-failure -L mqt-mlir-unittests

Build and run the unaffected legacy parser regression:

    cmake --build build/debug --target mqt-core-ir-test -j4
    (cd build/debug/test/ir && ./mqt-core-ir-test --gtest_filter='Qasm3ParserTest.*')

Check the architecture after deletion. These searches are deliberately limited
to dialect-specific identifiers and must return no matches:

    rg 'OQ3Dialect|OQ3Ops|ApplyGateOp|GateDeclOp|createOQ3ToQCPass|MLIROQ3' mlir docs
    rg 'add_subdirectory\(OQ3\)|OQ3ToQC' mlir

Build documentation and run repository policy checks:

    MLIR_DIR=/path/to/mlir/lib/cmake/mlir uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check origin/main
    git status --short --branch

For coverage, use the coverage preset, delete only ignored stale coverage
counters, and run the affected binaries sequentially because concurrent runs can
corrupt shared counters. Report line and branch coverage for
`mlir/lib/Target/OpenQASM` and the private direct emitter. Keep generated output
under `build/coverage` and record the final command and summary here when run.

## Validation and Acceptance

The frontend is accepted when parsing and semantic analysis require no MLIR
context, included buffers retain accurate source locations, valid programs
produce a resolved typed program, and invalid source returns collected
diagnostics at the owning stage.

Direct emission is accepted when supported programs return verified modules
containing QC and standard MLIR dialects only. Primitive aliases, custom gates,
broadcasting, controls, inverse and negative controls, expressions, dynamic
indices, measurements, reset, barrier, and structured control flow must retain
their tested behavior. A valid `pow @` program must parse and analyze, then fail
QC translation with a precise source-located message and no fallback IR.

The complete compiler is accepted when every broad corpus fixture passes the
explicit public API chain through optimized QCO, reconstructed QC, Adaptive QIR,
LLVM IR, and bitcode, and also passes `runDefaultPipeline`. Every source in the
straight-line subset must additionally produce Base QIR. The smaller `jeff`
corpus must round-trip, while tracked incompatible programs must succeed through
optimized QCO and fail specifically at `intoJeff()`.

Every retained downstream conversion change is accepted only with a focused
parser-independent native-IR regression that fails without the change and passes
with it. The related full-chain OpenQASM fixture must also pass. No conversion
test may link the OpenQASM frontend solely to construct its input.

The architecture is accepted when there is no OQ3 dialect, OQ3 operation,
OQ3-to-QC pass, generated OQ3 target, tool registration, or dialect test. The
`oq3::frontend` namespace may remain. There is one gate catalog, the frontend
library is named `MLIROpenQASMFrontend`, the public translation adapter is
small, and direct emission is private to QC translation.

Final acceptance requires all affected and full MLIR unit tests, the legacy
parser regression, documentation with warnings treated as errors, measured
coverage of the substantive newly added frontend/emitter lines,
`uvx nox -s lint`, and `git diff --check origin/main` to pass. Coverage tests
must exercise useful behavior rather than pad a numeric threshold. The final
diff must contain no build output, generated documentation, temporary
workaround, or unjustified production conversion change.

## Idempotence and Recovery

Configuration, compilation, unit tests, documentation, lint, and diff checks are
repeatable and write only to ignored build directories. If CMake retains deleted
OQ3 targets, remove the ignored `build` and `docs/_build` directories and
configure again; do not add source-tree cleanup workarounds.

Make the architecture transition in coherent local commits when useful, but do
not push. Before removing an old source, ensure its required direct-emission
behavior has moved into the private emitter and its tests pass. If a downstream
fixture fails, preserve the failing source, reduce it to native IR, and repair
the owning conversion instead of introducing a parser-side special case.

Never discard unrelated user changes or edit another task worktree. This plan
does not authorize pushing, changing pull request state, resolving review
threads, or publishing comments. Any later public action requires explicit human
authorization and the disclosure required by `docs/ai_usage.md`.

## Artifacts and Notes

The completed groundwork before this revision comprised an LLVM-native staged
frontend, 27 imported behavior fixtures, source control flow and carried state,
and clean focused validation. It also comprised an OQ3 dialect and OQ3-to-QC
pass that this plan now deliberately removes. Earlier OQ3-specific test counts
are historical evidence, not revised acceptance evidence.

The target-boundary proof after implementation must read:

    analyzeOpenQASM(pow-source) succeeds.
    translateQASM3ToQC(pow-source) fails at the pow modifier location.
    No OQ3 module is constructed.

The standard-chain proof must record a representative structured fixture
reaching:

    OpenQASM -> QC -> QCO -> optimized QCO -> QC -> Adaptive QIR
    -> LLVM IR and bitcode

The final corpus contains seventeen standard programs, six `jeff` round-trip
programs, four `jeff`-incompatible programs, and four Base programs. One native
`JeffToQCO` regression proves that a serialized entry point with observable
results regains its marker without losing those results. A native QC-to-QIR
regression proves that `cf.assert` lowers through LLVM. The latest focused
validation results are:

    OpenQASM frontend and target: 96 tests passed.
    OpenQASM compiler corpus: 48 tests passed.
    Native QC-to-QIR assertion regression: passed.
    QC translation: 241 tests passed.
    QC-to-QCO: 124 tests passed.
    QCO-to-QC: 121 tests passed.
    `jeff` round trip: 113 tests passed.
    Compiler pipeline: 164 tests passed, including 48 corpus cases.
    QC-to-QIR Adaptive: 126 tests passed.
    QC-to-QIR Base: 107 tests passed.
    Legacy OpenQASM parser: 97 tests passed.
    Warning-as-error documentation: passed.
    Repository lint and diff checks: passed.
    Frontend and direct-emitter line coverage: 89.9 percent (4117/4579).

No public GitHub action is authorized by this plan.

## Interfaces and Dependencies

The source frontend continues to expose from
`mlir/include/mlir/Target/OpenQASM/Frontend.h`:

    ParseResult parseOpenQASM(llvm::SourceMgr&);
    ParseResult parseOpenQASM(llvm::StringRef);
    AnalysisResult analyzeOpenQASM(const ParsedProgram&,
                                   const FrontendOptions& = {});
    AnalysisResult analyzeOpenQASM(llvm::SourceMgr&,
                                   const FrontendOptions& = {});
    AnalysisResult analyzeOpenQASM(llvm::StringRef,
                                   const FrontendOptions& = {});

`ParseResult` and `AnalysisResult` carry diagnostics as data. `ParsedProgram`
owns persistent syntax. `TypedProgram` owns resolved source semantics. These
interfaces remain in `oq3::frontend` unless a separate, evidence-backed rename
is approved.

The public QC translation interface remains only the existing overloads in
`mlir/include/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h`:

    OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr&,
                                             MLIRContext*);
    OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::StringRef,
                                             MLIRContext*);

The private emitter header beside the translation source declares an internal
typed-program-to-QC function; it is not installed as a public header and no
compiler caller uses it directly. `MLIRQCTranslation` links
`MLIROpenQASMFrontend`, QC and its builder, and the standard MLIR dialects used
by the emitter. `MLIROpenQASMFrontend` must not link QC or depend on the
translation library.

The compiler acceptance interfaces are `QCProgram::fromQASMString`,
`QCProgram::intoQCO`, `QCOProgram::cleanup`, `QCOProgram::runPassPipeline`,
`QCOProgram::intoJeff`, `JeffProgram::cleanup`, `JeffProgram::toBytes`,
`JeffProgram::fromBytes`, `JeffProgram::intoQCO`, `QCOProgram::intoQC`,
`QCProgram::cleanup`, `QCProgram::intoQIR`, `QIRProgram::llvmIR`,
`QIRProgram::toBitcode`, and `runDefaultPipeline`. Tests must respect their
move-only ownership contracts by copying at explicit branch points.

Revision note (2026-07-16): this plan replaces the completed OQ3-intermediate
architecture with direct QC emission. Review feedback moved the implementation
into private emitter files, renamed the frontend target, assigned custom-gate
target preflight to emission, made full-chain tests precede downstream changes,
required parser-independent conversion regressions, removed fixture capability
flags, and defined exact `jeff` and QIR acceptance paths. The final whole-branch
cleanup removed unused resolved-program state and replaced the arbitrary
90-percent coverage gate with behavior-driven coverage evidence.

Revision note (2026-07-16): post-review work made QC-to-QCO-to-QC-to-QIR the
primary acceptance path and moved `jeff` compatibility to separate positive and
boundary-failure suites. It removed the constant-lattice preflight, implemented
runtime integer and inclusive-range semantics, relocated implementation headers,
fixed the remaining lint diagnostics, and added canonical `cf.assert` lowering
to QIR.
