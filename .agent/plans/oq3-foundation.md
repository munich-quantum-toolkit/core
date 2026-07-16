# Complete a staged, specification-driven OpenQASM frontend

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept current as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs an OpenQASM frontend that separates source-language validity from
target capability. Parsing and semantic analysis must work without an MLIR
context, valid programs must produce verified target-neutral IR, and a separate
conversion must either produce verified QC or report a precise target
limitation. This distinction is observable with a valid `pow`-modified gate:
analysis and OQ3 emission succeed, while conversion reports that QC cannot
represent power modifiers.

The frontend must support real source programs, not only isolated syntax. That
includes lexical scope, mutable scalar and bit state, `for`, `while`, `if`,
loop-carried values, includes, custom gates, dynamic indices, measurement, and
OpenQASM 2 compatibility behavior. Ordinary arithmetic and structured control
flow use standard MLIR dialects. The small OQ3 dialect retains only resolved
gate declarations, applications, and ordered modifiers for which a
target-neutral boundary adds observable value.

The repository-relative scope is `mlir/include/mlir/Target/OpenQASM`,
`mlir/lib/Target/OpenQASM`, `mlir/include/mlir/Dialect/OQ3`,
`mlir/lib/Dialect/OQ3`, `mlir/include/mlir/Conversion/OQ3ToQC`,
`mlir/lib/Conversion/OQ3ToQC`, the QC-to-QCO structured-control conversion,
focused tests, and directly related CMake and documentation files. The legacy
`QuantumComputation` parser remains available and its tests remain regression
coverage; it is not used as an adapter by the MLIR frontend.

## Progress

- [x] (2026-07-15) Replaced the direct syntax-to-QC experiment with explicit
  parse, analyze, emit OQ3, and convert-to-QC stages.
- [x] (2026-07-15) Evaluated the unresolved review findings on pull request 1910
      and the implementation and fixtures from pull request 1862. Identified the
      stream parser bridge, unreachable loop support, and missing carried state
      as release blockers.
- [x] (2026-07-15) Moved OQ3-to-QC into `mlir/Conversion/OQ3ToQC`, implemented
  it with MLIR dialect conversion, made OQ3 illegal after conversion, and
  removed speculative OQ3 classical and loop concepts.
- [x] (2026-07-16) Replaced the `SourceMgr`-to-`std::istringstream` bridge with
  an LLVM-native, zero-copy lexer and recursive-descent parser that produce
  persistent syntax and collected `SMLoc` diagnostics.
- [x] (2026-07-16) Centralized names, types, scope, initialization, gate
  availability, and source restrictions in semantic analysis. Added exact
  version handling, textual includes, strict UTF-8 identifiers, constants,
  assignments, comparisons, broadcasting, and dynamic index checks.
- [x] (2026-07-16) Implemented source `if`, `for`, and `while` with standard SCF
  operations. Mutation analysis carries only state written by nested control
  flow through region arguments, yields, and results.
- [x] (2026-07-16) Implemented canonical-reference dynamic-qubit dispatch,
  runtime bounds and alias checks, physical/virtual addressing diagnostics,
  and a 4,096-leaf expansion budget enforced at analysis and emitter trust
  boundaries.
- [x] (2026-07-16) Hardened OQ3 gate signatures, modifiers, operand provenance,
  custom-gate reachability, recursion, expansion cost, and structured custom
  gate target errors.
- [x] (2026-07-16) Updated QC-to-QCO to preserve existing classical SCF state,
  append quantum state, use region-local mappings, and convert structured
  terminators only after their regions have established final values.
- [x] (2026-07-16) Ported all 27 healthy pull request 1862 fixtures. Seventeen
  non-looping programs compare exact QC output and ten loop or dynamic-index
  programs assert behavior-sensitive structural contracts.
- [x] (2026-07-16) Removed iteration artifacts from the complete effective diff:
      obsolete cleanup scaffolding, duplicated declaration metadata, parallel
      structured-value maps, stale test names, repetitive revision notes, and
      incidental comments.
- [x] (2026-07-16) Rebuilt from clean build and documentation directories. All
      focused binaries, 97 legacy parser tests, warning-as-error documentation,
      repository lint, and diff checks pass. Refreshed substantive
      changed-surface line coverage is 91.4 percent.

## Surprises & Discoveries

- Observation: the previous MLIR entry point copied the `SourceMgr` main buffer
  into `std::istringstream` and invoked the legacy scanner and parser. The
  source manager therefore did not own included source or parser locations.
  Evidence: the replacement can destroy the caller's source manager and still
  diagnose and emit from included buffers.

- Observation: the legacy scanner recognized `for` and `while`, but its parser
  had no statement cases for them. The prior OQ3 loop operation was therefore
  unreachable from source. Evidence: the new source fixtures exercise both forms
  and inspect their emitted SCF regions.

- Observation: pull request 1862 contains useful zero-copy lexing mechanics,
  scope and loop grammar, and broad behavior fixtures, but its parser and sink
  share semantic decisions and emit MLIR immediately. Its mechanics and tests
  are reusable; its architecture is incompatible with a persistent typed stage.

- Observation: mutable outer values cannot be represented by retaining SSA
  values created inside a branch or loop. Those values do not dominate later
  uses. Explicit SCF region arguments, yields, and results are required, and
  carrying only transitively mutated values keeps signatures small.

- Observation: OpenQASM integer ranges are inclusive, may descend, and may span
  the full signed 64-bit domain. Normalizing them through i128 distance
  arithmetic avoids `stop + 1` overflow and makes dynamic zero steps and
  unrepresentable trip counts explicit assertions.

- Observation: parser-owned symbol resolution fails for textual includes because
  declarations become visible according to include expansion order, not buffer
  parse order. A grammar-only parser and one semantic pass correctly handle
  included declarations and declaration-before-use rules.

- Observation: the QC qubit type has reference semantics. Loading a dynamic
  element again can create an alias that QC-to-QCO cannot represent, and a
  qubit-valued `arith.select` is not lowerable. Structured `scf.if` dispatch
  applies operations to canonical register references in each branch.

- Observation: dialect-conversion worklist order does not guarantee that a
  structured terminator sees the final values produced in its region. A second
  conversion phase for `scf.yield` and `scf.condition`, sharing region-local
  maps with the first phase, makes value threading deterministic.

- Observation: QCO structured conditionals carry linear quantum state but not
  arbitrary classical results. QC-to-QCO therefore spills branch-local classical
  measurement results into distinct one-element scratch slots hoisted outside
  enclosing loops while keeping quantum values in SSA form.

- Observation: direct dynamic dispatch grows as the Cartesian product of dynamic
  operands. The 4,096-leaf budget prevents exponential IR construction before it
  starts and is also checked when a caller directly mutates a `TypedProgram`.

- Observation: a successful whole-module conversion can still visit OQ3 calls
  inside unused gate definitions. Erasing unreachable definitions before
  conversion prevents unused recursion or doubling chains from blocking or
  expanding otherwise valid programs.

- Observation: the repository-wide hook wrapper consults the shared Git index
  and can report missing-file metadata for intentional unstaged deletions.
  Running every hook over all live changed and untracked paths validates the
  same surface without staging work that has not been approved.

## Decision Log

- Decision: retain a minimal OQ3 dialect for resolved gates and ordered
  modifiers. Rationale: `pow` demonstrates valid source semantics that QC cannot
  represent, while standard MLIR already models classical values and structured
  control flow. Date/Author: 2026-07-15 / Codex.

- Decision: implement OQ3-to-QC as dialect conversion with OQ3 illegal in the
  final target. Rationale: success then proves mechanically that no OQ3
  operation remains. Date/Author: 2026-07-15 / Codex.

- Decision: adapt selected mechanics and all healthy behavior fixtures from pull
  request 1862 without adopting its direct parser-to-emitter design. Rationale:
  parsing must produce persistent target-independent data and semantic rules
  must have one owner. Date/Author: 2026-07-15 / Codex.

- Decision: keep the parser grammar-only and use one syntax expression graph for
  arithmetic, conditions, indices, and measurements. Rationale: names, types,
  scope, initialization, and gate order belong in semantic analysis, especially
  across textual includes. Date/Author: 2026-07-16 / Codex.

- Decision: represent ordinary control flow with SCF and thread only
  transitively mutated scalar and bit state. Rationale: SCF provides the
  required dominance and region contracts without duplicating them in OQ3.
  Date/Author: 2026-07-16 / Codex.

- Decision: normalize inclusive ranges to a zero-based positive SCF trip count
  using i128 calculations. Rationale: this handles ascending, descending,
  dynamic, empty, and boundary ranges without source-width overflow.
  Date/Author: 2026-07-16 / Codex.

- Decision: perform definite-initialization analysis and use `ub.poison` only
  for emitter slots proven unreachable before initialization. Rationale: the
  emitter must not invent source values. Date/Author: 2026-07-16 / Codex.

- Decision: follow the OpenQASM C99 conversion rules consistently in constant
  folding and MLIR emission. Rationale: mixed `int`, `uint`, float, comparison,
  and assignment behavior must agree at compile time and run time. Date/Author:
  2026-07-16 / Codex.

- Decision: treat standard-library includes as ordered events and custom
  includes as cached syntax expanded once per occurrence. Rationale: this
  preserves textual visibility while avoiding repeated lexing and still detects
  active recursive expansion. Date/Author: 2026-07-16 / Codex.

- Decision: allow gate bodies to read parameters, loop variables, built-in
  constants, and immutable global constants, but reject mutable global captures.
  Rationale: immutable values can be embedded safely; mutable values would
  create closure-like references in emitted gate symbols. Date/Author:
  2026-07-16 / Codex.

- Decision: use structured dispatch over canonical QC references for dynamic
  qubit indices and cap expansion at 4,096 leaves. Rationale: this preserves
  alias identity and downstream conversion while bounding generated IR.
  Date/Author: 2026-07-16 / Codex.

- Decision: preflight custom-gate reachability and expansion, convert reachable
  gate bodies before call sites, and reject modifiers on structured custom gates
  that QC cannot represent. Rationale: target conversion must never drop
  semantics or introduce fresh illegal operations. Date/Author: 2026-07-16 /
  Codex.

- Decision: convert QC structured parents and quantum operations before their
  terminators, using one region-local state model. Rationale: terminators must
  resolve values after all branch or loop-body updates, independently of the
  conversion driver's traversal order. Date/Author: 2026-07-16 / Codex.

- Decision: store frontend program IDs by canonical vector position and retain
  only metadata consumed after analysis. Rationale: declaration IDs duplicated
  their vector indices, gate parameter names were reduced immediately to counts,
  and statement locations already identify gate applications. Scalar names
  remain because source-order and include-identity tests inspect them. Removing
  the other metadata makes the typed model smaller and its invariants explicit.
  Date/Author: 2026-07-16 / Codex.

## Outcomes & Retrospective

The completed architecture separates source concerns from target concerns. The
LLVM-native frontend owns source buffers and diagnostics, parsing produces
persistent syntax, semantic analysis produces a compact typed program, and
emission uses standard MLIR wherever the semantics are already expressible. OQ3
retains the small gate/modifier boundary needed to distinguish valid source from
QC capability, and dialect conversion proves that successful QC output has no
residual OQ3 operations.

The implementation now supports source loops, assignments, comparisons, lexical
shadowing, definite initialization, local bits, negative and dynamic indices,
compound arithmetic, general conditions, targetless and targeted measurement,
whole-register bit assignment, OpenQASM 2 classical-register conditions, Unicode
identifiers, custom includes, immutable constants in gate bodies, and mixed
register/scalar broadcasting. SCF iteration arguments and results solve the
original dominance problem for nested mutable state.

Pull request 1862 is represented as behavior coverage rather than a second
frontend architecture. Seventeen imported programs compare exact QC and ten
exercise loop or dynamic-index structure. After cleanup, the focused evidence is
13 OQ3 tests, 87 staged frontend and target tests, 241 QC translation tests, 121
QC-to-QCO tests, 97 legacy parser tests, and 91.4 percent substantive
changed-surface line coverage (4,348 of 4,759 lines).

## Context and Orientation

`mlir/lib/Target/OpenQASM/Frontend.cpp` owns the source manager and returns an
opaque `ParsedProgram`. `OpenQASMLexer.cpp`, `OpenQASMParser.h`,
`OpenQASMSyntax.h`, and `OpenQASMSyntax.cpp` implement tokenization, grammar,
source recovery, and persistent target-independent syntax.
`OpenQASMSemantics.cpp` resolves that syntax into the `TypedProgram` declared in
`mlir/include/mlir/Target/OpenQASM/Frontend.h`. These stages do not depend on an
MLIR context.

`mlir/lib/Target/OpenQASM/OpenQASM.cpp` emits the typed program using builtin
MLIR plus `arith`, `cf`, `func`, `math`, `memref`, `scf`, `ub`, QC, and OQ3.
Mutable state is held by standard SSA values and explicitly crosses SCF region
boundaries. Dynamic qubit references become structured dispatch over canonical
QC references rather than new loads or qubit-valued selection.

`mlir/lib/Conversion/OQ3ToQC/OQ3ToQC.cpp` is the target boundary. It validates
and orders reachable custom gates, lowers supported OQ3 gate applications,
reports unsupported target semantics, erases gate symbols, and completes with
OQ3 marked illegal.

`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` converts QC reference semantics to QCO
value semantics. Structured operations preserve their classical operands and
results while appending quantum values. Region-local mappings and a separate
terminator phase preserve the latest branch and loop state.

Tests under `mlir/unittests/Target/OpenQASM` exercise parsing, analysis, OQ3
emission, QC conversion, source diagnostics, and the imported fixture corpus.
Tests under `mlir/unittests/Dialect/OQ3`,
`mlir/unittests/Dialect/QC/Translation`, and `mlir/unittests/Conversion/QCToQCO`
defend the IR and conversion boundaries.

## Plan of Work

### Milestone 1: establish the target-neutral boundary

Keep only source distinctions that QC cannot faithfully represent in OQ3. Move
target lowering to `Conversion/OQ3ToQC`, express it through MLIR dialect
conversion, and mark OQ3 illegal in the successful target. Demonstrate both a
supported gate program and the valid-but-unsupported `pow` boundary.

### Milestone 2: replace the legacy parser bridge

Use LLVM source ownership, `StringRef` token spans, `SMLoc` diagnostics, and
bump-allocated transient parser data. Parsing must return persistent syntax and
diagnostics without constructing MLIR. Includes must retain source identity and
lifetime after the caller source manager is destroyed.

### Milestone 3: implement source semantics and structured state

Resolve all names, types, scopes, initialization, gates, indices, assignments,
and conditions in one semantic pass. Emit `if`, `for`, and `while` as SCF with
minimal carried state. Cover ascending, descending, empty, dynamic, zero-step,
and integer-boundary ranges and nested updates to outer scalar and bit state.

### Milestone 4: close specification and target gaps

Add exact version and include behavior, strict identifiers and numeric forms,
OpenQASM 2 compatibility, custom gates, broadcasting, dynamic canonical-qubit
dispatch, physical addressing checks, recursion and expansion budgets, and
precise target errors for unsupported modifiers. Update QC-to-QCO wherever the
new valid structured QC exposes a downstream value-semantics gap.

### Milestone 5: prove and clean the complete result

Port the useful pull request 1862 fixtures, add behavior-driven tests at each
trust boundary, run changed-surface coverage, and obtain fresh read-only review.
Remove experimental scaffolding, duplicate concepts, stale names, repetitive
plan history, and comments that describe iteration rather than the final design.
Repeat all affected validation after cleanup.

## Concrete Steps

Run commands from the repository root. Configure with an installed MLIR 22.1
CMake package and build the affected targets:

    MLIR_DIR=/path/to/mlir/lib/cmake/mlir cmake --preset debug
    cmake --build build/debug --target mqt-core-mlir-unittest-oq3 mqt-core-mlir-unittest-openqasm-target mqt-core-mlir-unittest-qc-translation mqt-core-mlir-unittest-qc-to-qco -j4

Run the focused binaries:

    ./build/debug/mlir/unittests/Dialect/OQ3/mqt-core-mlir-unittest-oq3
    ./build/debug/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
    ./build/debug/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation
    ./build/debug/mlir/unittests/Conversion/QCToQCO/mqt-core-mlir-unittest-qc-to-qco

Build and run the legacy parser regression:

    cmake --build --preset debug --target mqt-core-ir-test -j4
    (cd build/debug/test/ir && ./mqt-core-ir-test --gtest_filter='Qasm3ParserTest.*')

Validate generated documentation and repository policy:

    MLIR_DIR=/path/to/mlir/lib/cmake/mlir uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check origin/main
    git status --short --branch

When deletions remain unstaged, pass every live changed and untracked path to
the repository hooks as a second lint run because the shared index can prevent
the wrapper from collecting that set itself. Do not stage files merely to make
the wrapper's path collection succeed.

For changed-surface coverage, build the coverage preset, remove stale `.gcda`
files, and run the OQ3, OpenQASM, QC translation, and QC-to-QCO binaries
sequentially. Concurrent runs corrupt shared counters. Generate the report with:

    gcovr --root . --object-directory build/coverage \
      --gcov-executable '/path/to/llvm-cov gcov' \
      --filter 'mlir/lib/Conversion/OQ3ToQC/OQ3ToQC.cpp' \
      --filter 'mlir/lib/Dialect/OQ3/IR/GateCatalog.cpp' \
      --filter 'mlir/lib/Dialect/OQ3/IR/OQ3Ops.cpp' \
      --filter 'mlir/lib/Target/OpenQASM/.*' \
      --json build/coverage/oq3-focused-coverage.json --print-summary

All generated output belongs under ignored build directories. Do not record a
machine-specific MLIR path in this plan.

## Validation and Acceptance

Parsing is accepted when tokens and diagnostics retain source identity, includes
remain owned, multiple recoverable errors are returned as data, and no parser or
analyzer operation requires MLIR. Semantic analysis is accepted when valid
programs produce a typed program and invalid programs fail with source-located
diagnostics before emission.

Emission is accepted when ordinary programs verify, source loops produce valid
SCF, nested mutable state crosses regions through explicit arguments and
results, and dynamic qubit operations use canonical references with bounded
structured dispatch. The valid `pow` program must still verify as OQ3.

OQ3-to-QC is accepted when supported programs produce verified QC with no OQ3
operations, while unsupported power or structured custom-gate modifiers fail
with target-specific diagnostics. Reachable recursion and excessive expansion
must fail; unreachable definitions must not affect valid programs.

QC-to-QCO is accepted when structured branches and loops preserve original
classical results and final quantum values independent of conversion traversal
order. Dynamic measurement dispatch must remain valid through this conversion.

Final acceptance requires all focused binaries and legacy parser tests to pass,
all 27 imported fixtures to assert behavior, documentation to build with
warnings as errors, repository hooks to pass over every live file,
`git diff --check origin/main` to succeed, and changed-surface substantive line
coverage to remain at least 90 percent.

## Idempotence and Recovery

Configuration, builds, tests, documentation, lint, and coverage commands are
repeatable and write only ignored output. Coverage binaries must be run
sequentially after removing stale counters.

Do not restore the stream bridge or introduce a parser-to-emitter fallback if a
syntax family fails. Preserve the failing source test and repair the staged
frontend at the owning layer. Preserve unrelated user changes, do not modify
another task worktree, and require a clean task worktree before any rebase.

This plan does not authorize pushing, changing pull request state, resolving
review threads, or publishing comments. Those actions require separate human
authorization.

## Artifacts and Notes

The original staged baseline was 7 OQ3 tests, 21 staged frontend and target
tests, and 224 QC translation tests. It also contained a stream adapter and no
source path from loop tokens to a loop statement.

The cleaned implementation produces:

    13 OQ3 tests passed.
    87 staged frontend and target tests passed.
    241 QC translation tests passed.
    121 QC-to-QCO tests passed.
    97 legacy OpenQASM parser tests passed.
    17 imported programs matched exact QC references.
    10 imported loop or dynamic-index programs passed structural contracts.
    4,348 of 4,759 substantive changed lines were covered (91.4 percent).

The target-boundary proof is:

    analyzeOpenQASM(pow-source) succeeds.
    emitOQ3(pow-source) returns a verified module.
    OQ3ToQC rejects pow because QC has no power operation.

No public GitHub action is authorized by this plan. Any later agent-authored
public text must begin with the disclosure required by `docs/ai_usage.md`.

## Interfaces and Dependencies

The source frontend exposes:

    ParseResult parseOpenQASM(llvm::SourceMgr&);
    AnalysisResult analyzeOpenQASM(const ParsedProgram&,
                                   const FrontendOptions& = {});

`ParseResult` and `AnalysisResult` carry diagnostics as data. `ParsedProgram`
owns persistent syntax. `TypedProgram` owns resolved expressions, conditions,
declarations, statements, source locations, and output information.

The target adapter exposes:

    OwningOpRef<ModuleOp> emitOQ3(const frontend::TypedProgram&, MLIRContext&);
    OwningOpRef<ModuleOp>
    translateOpenQASMToOQ3(llvm::SourceMgr&, MLIRContext&,
                           const OpenQASMTranslationOptions& = {});

The conversion exposes:

    std::unique_ptr<Pass> createOQ3ToQCPass();

The parser and analyzer use LLVM support but have no MLIR, ANTLR, Java, or Rust
dependency. Emission and conversion depend on OQ3, QC, builtin IR, `arith`,
`cf`, `func`, `math`, `memref`, `scf`, `ub`, and MLIR dialect conversion.
