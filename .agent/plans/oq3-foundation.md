# Establish a typed OpenQASM 3 frontend

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs an OpenQASM frontend that can represent valid OpenQASM 3 even
when the QC or JEFF target cannot lower every construct yet. After this work, a
caller can parse OpenQASM 3 or supported OpenQASM 2 compatibility syntax into a
verified, typed MLIR module using an experimental `oq3` dialect. A separate pass
lowers the supported part of that module to MQT's QC dialect. This split makes
source-language validity independent of target capability and prevents the
parser from becoming a second, target-specific type checker.

The current implementation is an architectural demonstrator. It proves the
parser, typed intermediate representation, diagnostics, and initial lowering
boundary, while deliberately leaving the established OpenQASM importer as the
production path. The demonstrator can be observed by running the focused test
binary described below: the tests parse representative OpenQASM 3 and 2
programs, verify the generated module, lower supported programs to QC, and
distinguish source errors from target-capability errors.

The repository-relative scope is the OpenQASM target under
`mlir/include/mlir/Target/OpenQASM` and `mlir/lib/Target/OpenQASM`, the OQ3
dialect under `mlir/include/mlir/Dialect/OQ3` and `mlir/lib/Dialect/OQ3`, its
focused tests under `mlir/unittests/Target/OpenQASM`, and the directly related
build and documentation files. Existing QC translation code remains a comparison
oracle and must not be replaced until parity is demonstrated. Unrelated changes
and other independently coordinated tasks are outside this plan and must be
preserved.

## Progress

- [x] (2026-07-15 10:10Z) Started the implementation from a fresh `origin/main`
      while keeping the legacy importer available.
- [x] (2026-07-15 10:35Z) Pinned the OpenQASM 3.1 grammar, generated the C++
      ANTLR parser, isolated the generated code in its own target, and recorded
      its provenance and regeneration command.
- [x] (2026-07-15 10:50Z) Added the experimental typed `oq3` dialect, its
  source-specific types and operations, verifiers, and OQ3-to-QC pass.
- [x] (2026-07-15 11:05Z) Added the staged frontend, OpenQASM 2 normalization,
      include handling, initial semantic checks, control flow, measurements,
      gates, modifiers, scalar state, ranges, tests, and conservative
      documentation.
- [x] (2026-07-15 11:10Z) Verified 24 focused tests, 224 existing QC translation
      tests, and 110 compiler pipeline tests. All passed. Repository lint passed
      except for the then-existing unrelated `ty` diagnostic in
      `python/mqt/core/plugins/qiskit/estimator.py:235`.
- [x] (2026-07-15 11:25Z) Fetched and analyzed merged PRs #1815 and #1907,
      fast-forwarded canonical `main`, re-read `AGENTS.md`, `.agent/PLANS.md`,
      and `docs/ai_usage.md`, and rebased the implementation onto commit
      `a228a3dc2`.
- [x] (2026-07-15 11:28Z) Resolved the #1815 `mqt-cc` overlap by preserving its
      JEFF and QCO program pipeline and adding OQ3/Math dialect registration.
      Removed the provisional changelog entry because it lacked the now-required
      PR and author references.
- [x] (2026-07-15 11:35Z) Rebuilt after the rebase and reran focused and
      regression tests. The 24 OQ3 tests, 224 QC translation tests, and expanded
      116-test compiler suite passed. An `mqt-cc --emit=qc-import` smoke test
      parsed and printed textual OQ3 IR alongside the new JEFF-aware driver.
- [x] (2026-07-15 11:40Z) Ran `uvx nox -s lint` on the rebased tree. Every
      formatting, spelling, license, metadata, Python, and type-checking hook
      passed.
- [x] (2026-07-15 11:48Z) Re-read the checkout-independent ExecPlan rules from
      PR #1908, regenerated this plan to use only repository-relative scope,
      paths, commands, and coordination boundaries, and verified the result with
      `uvx nox -s lint`.
- [x] (2026-07-15 12:00Z) Updated the version policy so explicit 3.0 and 3.1
      declarations select the same current OpenQASM 3 semantics, and made
      general prose version-neutral within OpenQASM 3. The expanded 25-test
      focused suite and `uvx nox -s lint` pass.
- [ ] Complete faithful signed and unsigned integer semantics, the remaining
      scalar cast and operator rules, and typed non-bit program inputs and
      outputs.
- [ ] Add checked grammar and semantic coverage for arrays, aliases,
  subroutines, externs, switch, break, continue, timing, calibration,
  annotations, and pragmas. Unsupported families must remain explicit
  diagnostics until implemented.
- [ ] Add broader upstream conformance, differential, dominance, overflow, and
  performance tests. Record parser and emitter scaling evidence.
- [ ] After PR #1603 is present on `main`, add an isolated lowering change for
      the exponent forms supported by QC and diagnose unsupported downstream
      forms.
- [ ] Switch the convenience translation path only after differential tests
  demonstrate parity for the supported surface and a human approves retiring
  the legacy behavior.

## Surprises & Discoveries

- Observation: MLIR arithmetic operations generally require signless integer
  types, so mapping both OpenQASM `int` and `uint` directly to builtin integer
  types loses the source signedness needed to choose comparisons, extensions,
  division, and casts. Evidence: the prototype currently documents complete
  signed/unsigned semantics as planned rather than claiming conformance.

- Observation: OpenQASM permits a range step to be a runtime value while also
  requiring it to be nonzero. The current QC and JEFF path has no faithful
  runtime assertion or trap for this condition. Evidence: typed `oq3.for`
  accepts a dynamic integer step, while QC lowering emits
  `dynamic range step cannot be proven nonzero for the selected target`.

- Observation: comparison-driven loop lowering avoids both division by zero and
  endpoint overflow. Evidence: the focused range tests cover positive, negative,
  empty, singleton, non-divisible, and maximum-width endpoints, with a widened
  internal induction value.

- Observation: the repository typo fixer interpreted MLIR's ordered-less-than
  floating-point predicate as a spelling error and changed the identifier,
  causing a compilation failure. Evidence: the predicate now carries a
  line-scoped spelling suppression and the rebuilt focused suite passes.

- Observation: PR #1815 completely reworked `mqt-cc`, added JEFF input/output,
  and replaced the former compiler-pipeline interface with the `Programs`
  abstraction and composable pass pipelines. Evidence: rebasing produced
  conflicts only in `mlir/tools/mqt-cc/CMakeLists.txt` and
  `mlir/tools/mqt-cc/mqt-cc.cpp`; resolution retained the new driver and added
  only the required OQ3 registrations.

- Observation: PR #1907 introduced the ExecPlan process after the first
  prototype commits had already been created. Evidence: this file records the
  completed work retrospectively and becomes the authoritative living plan for
  every subsequent iteration.

- Observation: PR #1908 clarified that a checked-in ExecPlan is shared project
  documentation rather than a record of one machine's orchestration state.
  Evidence: this revision removes the former machine-specific scope block, cache
  setup, and worktree-administration instructions while retaining
  repository-relative coordination boundaries.

## Decision Log

- Decision: Start from `origin/main`, not PR #1862. Rationale: `main` retains a
  separated parser, semantic passes, and established OpenQASM 2 tests, whereas
  #1862's direct parser-to-MLIR emission is the architectural element being
  replaced. Date/Author: 2026-07-15 / Codex.

- Decision: Use a small typed `oq3` dialect as a semantic high-level
  intermediate representation, not as a syntax tree and not with Any-like
  operands. Rationale: valid source constructs must remain representable before
  every target supports them, while builtin MLIR dialects should continue to
  represent ordinary classical computation. Date/Author: 2026-07-15 / Codex.

- Decision: Implement one current OpenQASM 3 semantic mode based on 3.1, accept
  explicit 3.0 and 3.1 declarations into that same mode, default versionless
  input to it, and normalize supported OpenQASM 2 syntax into the same typed
  representation. Rationale: callers should not be rejected solely for a 3.0
  declaration when the frontend can interpret the program with the maintained
  OpenQASM 3 semantics. Date/Author: 2026-07-15 / Codex.

- Decision: Keep generated grammar sources committed and isolated, with an exact
  upstream revision and a documented local spelling correction. Rationale:
  builds must not require parser generation or expose ANTLR implementation
  details through public headers. Date/Author: 2026-07-15 / Codex.

- Decision: Preserve a dynamic range step in typed OQ3 but reject it during QC
  lowering when nonzero cannot be proven. Rationale: it is valid typed source,
  but silently treating zero as an empty range or generating unsafe arithmetic
  would change program meaning. Date/Author: 2026-07-15 / Codex.

- Decision: Preserve `pow` modifiers in OQ3 and do not eagerly expand them.
  Rationale: QC power support belongs to PR #1603; the frontend must parse the
  full modifier without guessing downstream semantics. Date/Author: 2026-07-15 /
  Codex.

- Decision: Keep `translateQASM3ToQC` and `.qasm` handling on the legacy path
  until parity tests pass. Rationale: the demonstrator is not yet a complete
  replacement and the legacy importer is a valuable differential oracle.
  Date/Author: 2026-07-15 / Codex.

- Decision: Integrate with #1815 by registering OQ3 and Math in its new
  JEFF-aware `mqt-cc` dialect registry rather than restoring any deleted
  `CompilerPipeline` code. Rationale: the merged program abstraction is now the
  repository architecture and the OQ3 change only needs textual IR support at
  this stage. Date/Author: 2026-07-15 / Codex.

- Decision: Do not add a changelog entry until a human-reviewed PR number and
  complete author list are available. Rationale: current `AGENTS.md` requires
  both for changelog entries. Date/Author: 2026-07-15 / Codex.

- Decision: Keep this ExecPlan independent of any checkout, developer account,
  local filesystem layout, or ephemeral branch. Rationale: PR #1908 makes the
  plan a portable repository artifact that must be usable from any clone.
  Date/Author: 2026-07-15 / Codex.

## Outcomes & Retrospective

The first demonstrator milestone produced a verified typed boundary and proved
that the proposed architecture can coexist with the current importer. It
supports representative gates, ordered modifiers, broadcasting, measurements,
mutable scalar and bit state, structured control flow, inclusive ranges,
standard-library includes, and OpenQASM 2 compatibility. It also demonstrates
the intended diagnostic split: syntax and semantic errors occur in the frontend,
while dynamic range steps and power modifiers fail as explicit target capability
limitations.

The milestone does not yet meet the final goal of arbitrary OpenQASM 3 input.
The most important semantic gap is faithful integer signedness; the largest
feature gaps are aggregate classical types, callable constructs, advanced
control flow, timing, and calibration. The test suite establishes a useful
foundation but still needs imported conformance examples and measured scaling.
Keeping the legacy importer active was the right risk-control decision. The
rebase of PR #1815 confirms that the OQ3 layer can remain additive while the
compiler driver evolves independently.

## Context and Orientation

OpenQASM is a source language for quantum programs. The generated ANTLR parser
under `mlir/lib/Target/OpenQASM/Generated` recognizes its grammar. ANTLR is a
parser generator; its generated C++ code turns source text into a parse tree but
does not establish source types or target behavior. The reviewed grammar files
are under `mlir/lib/Target/OpenQASM/Grammar`, and their upstream revision is
recorded in `mlir/lib/Target/OpenQASM/README.md`.

The staged frontend implementation is `mlir/lib/Target/OpenQASM/OpenQASM.cpp`,
with its public interface in `mlir/include/mlir/Target/OpenQASM/OpenQASM.h`. Its
`translateOpenQASMToOQ3` entry point accepts either an LLVM source manager or an
in-memory string and returns a verified MLIR module. An LLVM source manager owns
the main file and included buffers so diagnostics can retain filenames, line
numbers, and include context.

The `oq3` dialect is defined under `mlir/include/mlir/Dialect/OQ3` and
implemented under `mlir/lib/Dialect/OQ3`. A dialect is an MLIR vocabulary of
types and operations with verifiers. This dialect adds only source distinctions
that builtin MLIR or existing MQT quantum dialects cannot faithfully retain,
including fixed-width bit values, angles, source gate symbols and applications,
ordered modifiers, and inclusive source ranges. Ordinary arithmetic, functions,
structured control flow, and mutable storage use the builtin `arith`, `math`,
`func`, `scf`, and `memref` dialects.

The lowering pass is `mlir/lib/Dialect/OQ3/Transforms/LowerOQ3ToQC.cpp`. It
rewrites supported OQ3 operations to the existing QC dialect and builtin
operations. A target capability diagnostic means the source is typed and valid
but the selected downstream representation cannot preserve it yet. This is
intentionally different from a source semantic diagnostic.

Focused tests live in `mlir/unittests/Target/OpenQASM/test_openqasm.cpp`. The
established importer and its regression fixtures remain under the QC translation
tests. Documentation of the deliberately conservative feature boundary lives in
`docs/mlir/OpenQASM.md`.

PR #1815, now part of `main`, added Python bindings for compiler programs,
replaced `CompilerPipeline` with `Programs`, and made `mqt-cc` understand QC,
QCO, JEFF, and QIR flows through explicit passes. The OQ3 prototype does not
replace that design. It only registers the OQ3 and Math dialects so textual OQ3
modules can be read; the production QASM input path remains unchanged until
parity is demonstrated.

## Plan of Work

First, keep the grammar dependency reproducible. Preserve the pinned grammar,
generated sources, provenance note, and isolated CMake target. When updating the
grammar, review upstream changes, regenerate all outputs in one operation, and
rerun parser-focused tests rather than hand-editing generated code.

Second, stabilize the typed representation before expanding features. Add tests
for every type and operation verifier, symbol lookup, modifier operand, scope,
and invalid programmatically constructed form. Resolve integer signedness in a
way that keeps arithmetic on MLIR-compatible signless integer values while
carrying enough semantic information to select signed or unsigned operations. Do
not introduce an untyped fallback operation.

Third, extend semantic construction family by family. For each grammar family,
either construct verified typed IR or emit one precise feature-named diagnostic.
Add tests for both success and failure. Preserve declared input and output
order, source locations, lexical scopes, and dominance. Mutable classical values
must remain in storage or explicit region-carried values; quantum SSA values
must cross regions as arguments and results rather than through a global side
table.

Fourth, extend lowering independently of parsing. A valid OQ3 operation may
remain unlowered until QC and JEFF have matching semantics. Lower positive and
negative inclusive ranges through widened comparison-driven control flow. Add
power lowering only after #1603 exists on `main`, and keep unsupported exponent
forms as target diagnostics.

Fifth, grow objective evidence. Import reviewed upstream examples, add
differential tests for programs supported by both importers, verify every
produced module, and benchmark a flat gate stream, nested expressions,
standard-library loading, and repeated custom gates. Require approximately
linear parser and emitter growth and avoid eager gate expansion in frontend
construction.

Finally, consider switching the convenience API only after the staged path has
parity for its advertised surface, all regression tests pass, and a human has
reviewed the generated changes. Keep the dialect explicitly experimental until
its operation shapes and lowering boundary have proven stable.

## Concrete Steps

Run every command in this section from the repository root. Configure the debug
preset first when `build/debug` does not exist:

    cmake --preset debug

To rebuild the driver and focused test after a change, run:

    cmake --build --preset debug \
      --target mqt-cc mqt-core-mlir-unittest-openqasm-target -j4

The build should finish with no failed compilation or link steps. Then run:

    ./build/debug/mlir/unittests/Target/OpenQASM/\
      mqt-core-mlir-unittest-openqasm-target

The expected current result is 24 passing tests. Build and run the established
translation regression binary with:

    cmake --build --preset debug \
      --target mqt-core-mlir-unittest-qc-translation -j4
    ./build/debug/mlir/unittests/Dialect/QC/Translation/\
      mqt-core-mlir-unittest-qc-translation

The expected baseline is 224 passing tests. The compiler regression binary is:

    cmake --build --preset debug \
      --target mqt-core-mlir-unittests-compiler -j4
    ./build/debug/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler

The expected baseline at the first milestone was 110 passing tests. PR #1815 may
change the count; success means every discovered test passes.

After each completed batch, run:

    uvx nox -s lint
    git diff --check origin/main...HEAD
    git status --short --branch

If lint reports an apparently unrelated diagnostic, verify it against an
unmodified `origin/main` before recording it here. PR #1815 changed the formerly
known `SparsePauliOp` typing line, so that prior limitation must not be assumed
for later revisions.

## Validation and Acceptance

A source containing explicit `OPENQASM 3.0;`, explicit `OPENQASM 3.1;`, or no
version declaration must produce a verified typed OQ3 module for the supported
surface using the same current OpenQASM 3 semantics. Explicit 2.0 must activate
the compatibility syntax, including `qelib1.inc` and legacy measurement arrows.

Standard gates must be unavailable unless the correct standard library is
included. Custom gates must reject unknown symbols, use before definition,
recursion, incorrect arity, and incompatible broadcasting. Modifier order and
dynamic modifier operands must be visible in typed OQ3 IR. Power must remain a
target diagnostic until supported by QC.

A constant zero range step must fail semantic analysis. A runtime step must
verify in OQ3 but fail initial QC lowering with the exact target diagnostic. A
constant positive or negative range must lower without division and without
forming `stop + 1`; boundary-width tests must demonstrate widened internal
arithmetic.

Supported measurement, assignment, `if`, `while`, and `for` programs must verify
without dominance errors when mutable state crosses regions. Declared bit inputs
and outputs must retain source order and width and lower to matching builtin
integer function arguments and results.

The complete focused, QC translation, and compiler regression binaries must
pass. `mqt-cc` must still build with #1815's JEFF support and parse textual OQ3
IR because the OQ3 and Math dialects are registered. Existing OpenQASM input to
`mqt-cc` must continue to use and pass the legacy path until the planned switch.

Final acceptance for replacing the legacy path additionally requires checked
coverage of every OpenQASM 3 grammar family, faithful scalar semantics,
representative QC-to-JEFF success, differential equivalence for overlapping
programs, and recorded linear-growth benchmarks.

## Idempotence and Recovery

Builds, focused tests, regression tests, and lint are repeatable and write only
inside the repository's ignored `build` directory. Grammar regeneration is
repeatable only when using the exact pinned upstream revision and ANTLR version
recorded in `mlir/lib/Target/OpenQASM/README.md`; review the complete generated
diff after regeneration.

Before rebasing, require a clean task checkout, fetch the `origin` remote, and
rebase onto `origin/main`, not onto a mutable local `main`. In a multi-worktree
environment, only the coordinating process may mutate shared worktree metadata.
Resolve overlapping files by preserving current mainline architecture and
reapplying the smallest OQ3-specific change. If a rebase conflict cannot be
resolved safely, abort the rebase rather than discarding either side. Never use
`git reset --hard`, never modify another task's worktree, and never force-push
without fresh user authorization.

The generated parser is additive. If the architecture is rejected, the three
layers can be removed independently: the frontend target and tests, the OQ3
dialect and lowering, then the grammar/runtime dependency. The unchanged legacy
importer remains a safe fallback throughout development.

## Artifacts and Notes

The initial pre-rebase validation evidence was:

    OpenQASMTargetTest: 24 tests passed
    QASM/QC translation: 224 tests passed
    Compiler pipeline: 110 tests passed

The post-rebase validation evidence is:

    OpenQASMTargetTest: 24 tests passed
    QASM/QC translation: 224 tests passed
    Compiler pipeline: 116 tests passed
    mqt-cc textual OQ3 smoke test: exit code 0

The target-capability diagnostic exercised by tests is:

    dynamic range step cannot be proven nonzero for the selected target

The first three reviewable implementation commits before the #1815/#1907 rebase
were organized as grammar provenance, typed dialect/lowering, and staged
frontend/tests. Their hashes change when rebased; use commit subjects rather
than old hashes when resuming this plan.

No push, pull request, comment, review, or other remote mutation is authorized
by this ExecPlan. The user must review the local changes and explicitly
authorize any later external action. Any agent-authored public body must begin
with `🤖 *AI text below* 🤖`.

## Interfaces and Dependencies

The public frontend interface must remain in namespace `mlir::oq3` and expose:

    struct OpenQASMTranslationOptions {
      llvm::SmallVector<std::string> includeDirectories;
    };

    OwningOpRef<ModuleOp>
    translateOpenQASMToOQ3(llvm::SourceMgr&, MLIRContext&,
                           const OpenQASMTranslationOptions& = {});

    OwningOpRef<ModuleOp>
    translateOpenQASMToOQ3(llvm::StringRef, MLIRContext&,
                           const OpenQASMTranslationOptions& = {});

The lowering interface must remain in namespace `mlir::oq3` and expose
`createLowerOQ3ToQCPass(OpenQASMLoweringOptions)`. The initial options type may
be empty, but it is the stable place for later target capability choices.

The frontend depends on LLVM `SourceMgr`, ANTLR 4.13.2, MLIR builtin IR,
`arith`, `math`, `func`, `scf`, `memref`, the experimental OQ3 dialect, and the
existing QC dialect. Generated parser implementation details must remain private
to `MQTOpenQASMParser`. The frontend library is `MLIROpenQASMTarget`; the
dialect and lowering libraries are `MLIROQ3Dialect` and `MLIROQ3Transforms`.

Source-specific OQ3 operations must remain typed and verified. The minimum
vocabulary contains symbol-bearing gate definitions, typed gate applications,
ordered modifier metadata plus dynamic modifier operands, bit packing and
unpacking at function boundaries, and an inclusive integer `oq3.for` with an
`oq3.yield` terminator. Ordinary classical arithmetic must not be duplicated in
OQ3, and no Any-like fallback type or operation may be introduced.

Revision note (2026-07-15): Created this ExecPlan after PR #1907 introduced the
repository process. It records the completed prototype, incorporates the #1815
rebase and its post-rebase validation, records the clean repository lint result,
and defines the remaining stabilization and conformance work. Regenerated after
PR #1908 to remove machine-specific orchestration details and express all scope,
commands, coordination boundaries, and recovery instructions in portable
repository-relative terms. Updated the version policy so explicit 3.0 and 3.1
declarations use the same maintained OpenQASM 3 semantics and general prose does
not overstate a 3.1 distinction where none matters.
