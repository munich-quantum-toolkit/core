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
preserve its ordered numeric exponent, and the direct emitter now creates
`qc.pow` after that operation became available on `main`. Downstream conversions
decide whether a particular power body can be canonicalized or represented.
There is no OQ3 MLIR dialect or OQ3-to-QC conversion. An intermediate dialect
that cannot proceed through the compiler adds maintenance and test surface
without user value.

The scope is the staged frontend under `mlir/include/mlir/Target/OpenQASM` and
`mlir/lib/Target/OpenQASM`, direct QC translation under
`mlir/lib/Dialect/QC/Translation`, the OpenQASM fixture corpus and compiler
tests, and only those existing conversion files for which a full-chain fixture
demonstrates a real defect. Preserve the legacy `QuantumComputation` parser and
unrelated behavior. The 2026-07-25 request authorizes updating the existing PR
branch after the rebase is validated, but does not authorize resolving review
threads or publishing new PR text.

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
      behavior and precise target diagnostics. The original `pow @` rejection
      was replaced after QC gained a native power modifier.
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
- [x] (2026-07-16) Ran documentation, lint, architecture, affected unit,
      legacy-parser, and clean sequential coverage validation for that
      historical revision. Its substantive coverage was 4117/4579 lines (89.9
      percent); later behavior-driven tests supersede this measurement.
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
- [x] (2026-07-25) Rebased the complete branch onto `origin/main` at `7c50a17a`,
      after #1932, #1933, #1934, #1935, #1936, #1938, #1939, and the QC/QCO
      power-modifier work had landed. Conflict resolution retained the extracted
      upstream implementations and removed those conversion changes from the
      effective OpenQASM diff.
- [x] (2026-07-25) Integrated the current QC power modifier into direct OpenQASM
      emission, preserved ordered and nested modifiers, accepted floating
      exponents, retained exact-f64 checks for constant integer exponents, and
      restored the `rccx` compatibility dispatch.
- [x] (2026-07-25) Extended the public compiler corpus with six representative
      power programs that pass QC to QCO to QC to QIR and `jeff`, two Base power
      cases, and one composite-body power case that fails explicitly at
      QCO-to-`jeff`.
- [x] (2026-07-25) Addressed the selected fresh-review findings without
      broadening the parser claim: made Boolean `&&` and `||` short-circuit,
      restricted measurement to its grammar statement contexts, and retained the
      compatibility-oriented implicit-output behavior.
- [x] (2026-07-25) Kept `stdgates.inc` and `qelib1.inc` as distinct semantic
      libraries while deliberately accepting either include in either source
      mode. Corrected the catalog membership of compatibility gates and retained
      the default compatibility policy used by legacy programs.
- [x] (2026-07-25) Routed translation failures through MLIR diagnostics with
      nested `CallSiteLoc` include stacks and corrected preloaded include-buffer
      parentage so diagnostics name the actual include chain.
- [x] (2026-07-25) Added bounded expression, block, modifier, custom-gate,
      register-storage, typed-statement, projected-emission, and actual-emission
      work. Replaced recursive expression copying, linear gate lookup, repeated
      dynamic-index `scf.if` chains, eager semantic-state snapshots, and
      per-control nested regions with bounded or more compact implementations.
- [x] (2026-07-25) Revalidated the public compiler corpus after making
      measurement grammar-correct. Reworked condition fixtures to use measured
      Boolean scalars so the intentional implicit-bit export contract remains
      unchanged, and removed the numeric loop-state fixture from downstream
      positive corpora because QCO-to-QC does not yet preserve such loop
      results. Direct QC retains dedicated coverage for the accepted source.
- [x] (2026-07-27) Rebuilt and reran the affected frontend, translation,
      compiler, Adaptive QIR, and Base QIR suites after formatting: all 788
      tests passed. The warning-as-error documentation build and the complete
      repository lint session also passed.
- [x] (2026-07-27) Rebuilt the coverage configuration, discarded only stale
      ignored `.gcda` counters, and ran the affected binaries sequentially. This
      intermediate revision reached 3537/3958 lines (89 percent) and 2855/4698
      branches (60 percent); the final coverage pass below supersedes it.
- [x] (2026-07-27) Closed the independent verification findings by making
      compile-time logical evaluation short-circuit without skipping dead-branch
      type validation and by attaching include provenance to each cached-source
      expansion rather than to the shared parsed buffer. Added focused faulting
      RHS and repeated-include regressions.
- [x] (2026-07-27) Rebased the complete 31-commit branch without conflicts onto
      `origin/main` at `07fafad95`, rebuilt the affected targets, and reran all
      788 focused tests, the complete repository lint session, and the
      warning-as-error documentation build.
- [x] (2026-07-27) Addressed selected review findings MF-01, MF-04, and MF-05:
      made standard-gate lowering recipes exhaustive, restored the distinct
      OpenQASM 2 and 3 U-family phase conventions, guarded runtime integer power
      exponents against inexact f64 conversion, and made the initial
      100000-operation construction budget authoritative with conservative
      preflight accounting.
- [x] (2026-07-27) Added an independent lightweight QC-IR matrix oracle for U,
      u2/u3/u, cu/cu3, and modifier phase behavior, plus variable,
      branch-joined, and loop-carried integer-power tests and early-rejection
      tests for wide, scalar-expression, structured-control, phase-correction,
      and power-check construction.
- [x] (2026-07-27) Implemented the selected FU-02 scalar builtins `ceiling()`
      and `floor()` with compile-time folding, runtime Math-dialect lowering,
      focused semantic and QC-emission coverage, and maintained documentation.
- [x] (2026-07-27) Implemented the selected FU-02 bit-vector builtins
      `popcount()`, `rotl()`, and `rotr()` with a typed bit-vector expression
      pool, atomic whole-register assignment, signed dynamic distances, linear
      emission preflight, and focused source and native QIR regressions.
- [x] (2026-07-27) Corrected the FU-02 bit-vector source contract by retaining
      the distinction between scalar `bit` and `bit[1]`, rejecting scalar bits
      from `popcount()`, `rotl()`, and `rotr()`, and adding result-level
      constant and runtime rotation and population-count oracles.
- [x] (2026-07-27) Audited OpenQASM issues #527, #594, #610, #612, #613, #614,
      and #617 and pull requests #624 and #666. None is a merged specification
      change that supersedes the released OpenQASM 3.1 constant-initializer
      rules, so the frontend implements the released same-type and
      promotable-constant matrix and treats the broader draft tables in #666 as
      non-normative follow-up work.
- [x] (2026-07-27) Completed the selected MF-06 and Q-01 cleanup. An exact
      Clang-Tidy 22.1.8 audit of the ten changed OpenQASM translation units and
      their headers fell from 1,376 diagnostics to zero without changing the
      repository policy. The pass also adopted project-style unqualified
      `size_t`, `int64_t`, and `uint64_t` names and renamed the internal
      logarithm expression kind from `Ln` to `Log`.
- [x] (2026-07-27) Added a deterministic custom-gate indexing structural stress
      regression with 2,048 indexed definitions and applications. Rebuilt the
      frontend, QC translation, and compiler-driver targets and ran 131 OpenQASM
      frontend/target tests plus 257 QC translation tests successfully.
- [x] (2026-07-27) Made the FU-02 QIR boundary explicit: constant-folded scalar
      rounding remains convertible, while retained `math.ceil`, `math.floor`,
      `math.ctpop`, `llvm.fshl`, and `llvm.fshr` operations produce
      feature-named diagnostics in both the Base and Adaptive profile
      conversions.
- [x] (2026-07-27) Regenerated coverage from clean counters and added
      behavior-driven tests for the runtime scalar-conversion matrix, all
      floating-point and signed/unsigned integer comparison predicates,
      inverse-trigonometric numeric conversions, unsigned constant arithmetic,
      bit-vector operations wider than 64 bits, and gate-catalog canonical-name
      round trips. All 137 frontend/target, 257 QC translation, 130 Adaptive
      QIR, 112 Base QIR, and 191 compiler tests passed. The five substantive
      frontend and emitter files reached 4149/4542 lines (91.35 percent) and
      3283/5195 branches (63.20 percent); changed production C++ reached
      4256/4651 lines (91.51 percent) against `origin/main`.
- [x] (2026-07-29 23:06Z) Refreshed the exact live revisions for this branch and
      pull request #1927. Confirmed that #1927 replaces per-measurement scalar
      results with allocated classical-register memrefs returned from the QC
      entry function, and identified the overlapping translation,
      compiler-corpus, and QIR conversion files.
- [x] (2026-07-30) Integrated the exact `bea1ce54e` classical-register contract
      from pull request #1927 while preserving the staged OpenQASM frontend and
      direct emitter. Resolved the old translator conflict in favor of the
      staged adapter and adapted the emitter and tests to return declared scalar
      and bit-register outputs in source order.
- [x] (2026-07-30) Added explicit resolved conversion expressions and
      context-sensitive builtin overload resolution, including the specified
      signed result for `pow(int, uint)` and the distinct angle type of gate
      parameters.
- [x] (2026-07-30) Made semantic expression analysis one bottom-up traversal
      rather than recursively revalidating and reevaluating the same syntax
      subtree.
- [x] (2026-07-30) Split the monolithic OpenQASM unit-test source by frontend,
      semantic, and target-emission responsibility; fixed the Windows iterator
      declaration, remaining Clang-Tidy diagnostics, frontend declaration style,
      and float-to-integer conversion documentation.
- [x] (2026-07-30) Extracted the OpenQASM-independent QIR builtin-profile
      diagnostics and native tests into a separate local task branch. The
      OpenQASM branch must then rely on the QIR behavior supplied by its base
      and by #1927 rather than carrying those conversion changes itself.
- [x] (2026-07-30) Rebuilt and ran 991 affected frontend, translation, compiler,
      and QIR tests plus all 286 legacy IR/importer tests. Full repository lint,
      warning-as-error documentation, diff checks, and an LLVM 22.1.8 Clang-Tidy
      audit pass. Clean coverage over the substantive frontend/emitter files
      reached 4342/4850 lines (89.5 percent), 314/334 functions (94.0 percent),
      and 3435/5519 branches (62.2 percent). A fresh read-only exact-head review
      found no blocker across the selected remediation items. Its one
      non-blocking output-coverage observation led to an additional
      QC-to-QCO-to-QC mixed-result regression; the compiler suite now contains
      202 passing tests.
- [x] (2026-07-30) Merged `origin/main` at `b91a0bf02` after #1927 and the
      documentation repair landed. Reviewed the exact #1923 head `7c2bbe5e2` as
      a separate integration input, including its scalar-qubit allocation and
      qubit-reuse pipeline, without merging that unreviewed PR into this branch.
- [x] (2026-07-30) Allocated `memref<nxi1>` storage only for actual bit outputs.
      Local bits now remain SSA values through assignments, measurement, dynamic
      selection, branches, and loops, while output-backed reads retain the
      established observable storage contract from #1927.
- [x] (2026-07-30) Moved rejection of mixed physical and declared qubits from
      source semantics to QC preflight, and replaced full-program mutation scans
      with deterministic sparse collection of only the state slots modified by a
      structured region.
- [x] (2026-07-30) Removed QCO-to-QC's quantum-only `scf.for` and `scf.while`
      assumption. Native regressions cover ordinary and type-changing classical
      loop state, and the restored `scalar-loop-state` fixture now completes
      both explicit and default QC-to-QCO-to-QC-to-QIR pipelines.
- [x] (2026-07-30) Completed the post-#1927 validation against `origin/main` at
      `b91a0bf02`. The 1,432 sequential affected and legacy tests pass, as do
      warning-as-error documentation, repository lint, and diff checks. A
      dynamic branch-write regression raised changed production C++ coverage to
      5,460/6,058 lines (90.1 percent); the five substantive frontend/emitter
      files cover 4,403/4,903 lines (89.8 percent) and 3,452/5,511 branches
      (62.6 percent).
- [x] (2026-07-30) Performed a read-only merge-tree assessment with the exact
      #1923 head. Its only textual conflict is the legacy
      `TranslateQASM3ToQC.cpp` implementation replaced by this branch. The
      scalar-allocation choice can be ported directly into the new emitter after
      #1923's independent review; no qubit-reuse implementation was copied here.
- [x] (2026-07-30) Ported the independent scalar-allocation choice into the new
      direct emitter: `qubit q;` now emits `qc.alloc`, while explicitly sized
      declarations, including `qubit[1]`, retain register allocation. The
      qubit-reuse pass and pipeline remain isolated in #1923. All 265 QC
      translation tests, 144 OpenQASM frontend/emitter tests, and 204 compiler
      pipeline tests pass with updated result-level references.
- [x] (2026-07-31) Raised the combined projected and actual QC-emission limit
      from 100000 to 10,000,000 operations so multi-million-operation Grover
      expansions remain accepted. Updated the cheap preflight regressions to
      cross the new limit through nested gate expansion and dynamic dispatch,
      adopted `std::ignore` for intentionally discarded nodiscard results, and
      renamed PR-owned C++ variables named `module` to `moduleOp`. All 142
      OpenQASM frontend/emitter tests, 265 QC translation tests, and 204
      compiler tests pass. The complete warning-as-error documentation build,
      repository lint session, and diff checks pass. The PR changelog entry
      credits both `@burgholzer` and `@denialhaag`.

## Surprises & Discoveries

- Observation: the old MLIR entry point copied the `SourceMgr` main buffer into
  `std::istringstream`, losing the source manager's include and location model.
  Evidence: the replacement parser consumes LLVM buffers directly and its
  persistent program retains included source identity.

- Observation: the legacy scanner recognized `for` and `while`, but its parser
  could not construct those statements. Evidence: the staged frontend now has
  source fixtures that produce and exercise standard SCF regions.

- Observation: valid source and downstream target support are distinct, but a
  source dialect is not required to preserve that distinction. Evidence: the
  typed semantic program retains ordered modifiers and source locations, the QC
  emitter creates `qc.pow`, and an unsupported composite power is rejected only
  by the later QIR or `jeff` conversion that cannot represent it.

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

- Observation: treating `measure` as an ordinary primary expression accepted
  forms outside the language's measurement statement contexts and made
  short-circuit behavior observable in places the grammar does not permit.
  Keeping measurement as a declaration, assignment, arrow, or targetless
  statement gives the parser and semantic analyzer a clearer boundary.

- Observation: MLIR's result-bearing `scf.if` builder may create empty regions
  rather than regions with an existing terminator. Short-circuit emission must
  therefore tolerate both region shapes before adding `scf.yield`.

- Observation: a chain of individually completed custom-gate definitions can
  still exceed the dependency-depth limit. Memoizing only visit state loses that
  depth; memoizing the computed dependency depth preserves linear validation.

- Observation: QC represents multiple classical values carried through `scf.for`
  and `scf.while`, but QCO-to-QC documented and implemented a quantum-only
  loop-result assumption. Once local OpenQASM bits stopped hiding behind memref
  loads, the standard pipeline exposed the mismatch as a region argument
  assertion. Preserving classical loop operands, results, yields, and conditions
  removes the assumption and restores the full pipeline.

- Observation: `qc.u` implements the conventional phaseful U matrix, while
  OpenQASM 3's language builtin U and OpenQASM 2/qelib U-family gates attach
  different global phases. Those phases become observable under control, so
  lowering aliases directly to `qc.u` was not semantics-preserving.

- Observation: a projected gate-expansion count does not bound operation
  construction from scalar expressions, comparisons, SCF scaffolding, phase
  correction, or runtime power checks. A builder listener supplies the
  authoritative construction count; conservative preflight retains early,
  source-located rejection for statically predictable work.

- Observation: eagerly unpacking every dynamic rotation would make nested
  `rotl()` and `rotr()` calls repeatedly rebuild the same bit vector. Retaining
  either source-order bits or one packed integer lets nested rotations remain
  packed until a consumer actually needs individual bits.

- Observation: a folded `popcount(bits)` used as a later dynamic index becomes
  stale when any element of `bits` changes. Register-wide bit generations are
  therefore part of the constant-fact dependency snapshot, including across
  structured control-flow joins.

- Observation: width alone cannot represent the OpenQASM type distinction
  between scalar `bit` and the one-element register `bit[1]`. Evidence: both
  previously produced a typed register declaration of width one, which let the
  register-only `popcount()`, `rotl()`, and `rotr()` signatures accept a scalar.

- Observation: MLIR canonicalization folds the constant population-count path
  but retains LLVM funnel shifts even when their operands are constants. A small
  test-only integer evaluator can inspect the returned bits without routing
  source correctness through QCO or QIR conversions.

- Observation: foldable scalar builtins pass the optimized Standard, `jeff`, and
  Base paths, but retained `math.ceil`, `math.floor`, population-count, and
  funnel-shift operations are explicit QC-to-QIR boundary failures. The
  bit-vector fixture is tracked independently in both the `jeff` and QIR
  boundary corpora; source acceptance must not imply retained-operation QIR
  support.

- Observation: #1927's first QC-to-QCO implementation treated every
  `memref.load` nested below structured control flow as quantum state. Evidence:
  an OpenQASM `while` carrying a bit register reached the conversion with an
  `i1` load and triggered the qubit-only state assertion. Filtering by
  `qc.qubit` element type fixes the conversion independently of the source
  parser.

- Observation: the order in which `DenseMap` iterates returned classical
  registers is not the order of function results. Evidence: Base QIR recorded
  the two outputs of the broadcast fixture in a nondeterministic order after
  adopting #1927. Retaining the returned allocation operations in signature
  order makes output recording deterministic without changing the lookup map.

- Observation: `jeff` represents a classical-register result with tensor storage
  even though QC and QCO expose the same value as a memref. Evidence: otherwise
  valid round trips failed an exact intermediate type-string check, while the
  reconstructed QCO and QC functions restored the original memref signature. The
  integration oracle normalizes only this known storage representation at the
  two `jeff` stages and still requires exact types on both sides.

- Observation: #1927's QIR result discovery treats each returned `memref<nxi1>`
  as a classical result register and requires its stores to come directly from
  measurements. Allocating the same storage for non-output local bits therefore
  made a valid measured output fail because an unrelated local initializer
  looked like output recording.

- Observation: #1923 changes scalar `qubit` declarations from one-element
  register allocation to scalar allocation so its reuse pass can see individual
  lifetimes. That representation is relevant to this emitter after #1923 is
  independently reviewed, but its old-translator patch cannot be merged verbatim
  after this PR replaces that implementation.

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

- Decision: lower numeric OpenQASM `pow @` modifiers to nested `qc.pow`
  operations in source order. Rationale: QC and QCO now represent integer,
  floating, and dynamic f64 exponents; retaining the modifier lets downstream
  canonicalization and target conversion own their actual capability limits.
  Constant integer exponents that cannot be represented exactly as f64 are
  rejected at QC emission rather than silently rounded. Date/Author: 2026-07-25
  / Codex.

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

- Decision: enforce one 10,000,000-operation projected-emission budget that
  composes custom-gate expansion and register dispatch with overflow-safe
  multiplication. Rationale: emitted work, not either mechanism independently,
  is the relevant safety bound. Ten million operations accommodates realistic
  multi-million-gate Grover circuits while retaining a guard against accidental
  unbounded expansion. Date/Author: 2026-07-16, revised 2026-07-31 / Codex.

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

- Decision: raise coverage through behavioral matrix and boundary tests rather
  than line-only execution. Rationale: runtime scalar conversions, every
  comparison predicate, greater-than-64-bit vectors, and gate-catalog
  round-tripping are observable contracts whose tests both close real gaps and
  establish a stable margin above 90 percent. Date/Author: 2026-07-27 / Codex.

- Decision: describe the parser as a supported-subset parser and use the live
  OpenQASM specification as the language reference. Rationale: this PR should
  accurately document implemented productions without vendoring a second grammar
  ground truth. Date/Author: 2026-07-25 / Codex.

- Decision: implement constant conversions according to the released OpenQASM
  3.1 matrix while upstream issues 527, 594, 610, 612, 613, 614, and 617 and
  pull requests 624 and 666 remain open. Rationale: the proposals explore
  broader or clarified conversion behavior but do not supersede the released
  specification; following them preemptively would make this frontend depend on
  unsettled semantics. Date/Author: 2026-07-27 / Codex.

- Decision: preserve compatibility leniency around standard-library spelling
  while retaining `stdgates.inc` and `qelib1.inc` identity internally.
  Rationale: strict mode can enforce the actual gate membership, while the
  default importer continues accepting common legacy and hybrid programs.
  Date/Author: 2026-07-25 / Codex.

- Decision: lower dynamic register selection with `scf.index_switch` and emit a
  single `qc.ctrl` region for each source control modifier, regardless of its
  arity. Rationale: these operations directly represent multi-way selection and
  variadic controls, reducing generated operation count and nested-region depth
  without changing source ordering. Date/Author: 2026-07-25 / Codex.

- Decision: enforce explicit frontend and emission resource budgets and use
  copy-on-write semantic bit state. Rationale: hostile or accidental deep and
  wide inputs must fail with source diagnostics before unbounded recursion or
  allocation, while ordinary control-flow analysis should not copy every bit
  register at each branch. Date/Author: 2026-07-25 / Codex.

- Decision: represent every catalog lowering with a closed `GateLowering` enum
  and use dedicated recipes for OpenQASM 3 U, OpenQASM 2 u2/u3/u, cu, and cu3.
  Rationale: string aliases cannot express phase corrections, and exhaustive
  switches make omissions a compile-time maintenance failure. Date/Author:
  2026-07-27 / Codex.

- Decision: reject a runtime integer power exponent unless removing its trailing
  binary zeroes leaves at most 53 significant bits before converting it to f64.
  Rationale: this is the exact IEEE-754 binary64 representability condition and
  prevents variable, branch-joined, and loop-carried values from silently
  rounding. Date/Author: 2026-07-27 / Codex.

- Decision: keep phase-convention tests in QC translation with a small local
  evaluator instead of routing them through QC-to-QCO, mapping, or DD
  construction. Rationale: the regression must independently test the emitted QC
  recipe without coupling translation correctness to downstream conversion and
  placement machinery. Date/Author: 2026-07-27 / Codex.

- Decision: represent runtime `ceiling()` and `floor()` with `math.ceil` and
  `math.floor`, while folding constant calls during semantic analysis.
  Rationale: the standard Math dialect expresses the source operations directly
  and keeps the frontend independent of MLIR; existing scalar-expression
  preflight and the authoritative builder listener continue to provide the one
  construction budget. Date/Author: 2026-07-27 / Codex.

- Decision: represent bit-vector builtins in a dedicated typed expression pool
  instead of pretending they are scalar expressions. Pack bit zero as the
  least-significant bit, use host permutations for constant rotations when bits
  are available, and use LLVM funnel shifts for packed or dynamic rotations.
  Rationale: the source width and atomic whole-register value remain explicit,
  while the lazy emitted representation avoids redundant pack/unpack chains.
  Date/Author: 2026-07-27 / Codex.

- Decision: retain runtime rounding, population count, and funnel shifts in
  their native MLIR dialects in QC, but reject them explicitly at both current
  QIR profile boundaries. Rationale: the released Base and Adaptive profiles do
  not admit these retained operations; constant-folded scalar rounding still
  converts normally, while both QIR and `jeff` keep explicit incompatibility
  fixtures for the unsupported runtime forms. Date/Author: 2026-07-27 / Codex.

- Decision: add one `isScalar` flag to typed register declarations and require a
  non-scalar bit register when constructing a bit-vector expression. Rationale:
  storage width and existing bit assignment remain unchanged, while the typed
  source model can enforce the OpenQASM `bit[_]` and `bit[n]` builtin
  signatures, including for width one. Date/Author: 2026-07-27 / Codex.

- Decision: verify rotation results through returned QC bits using an
  independent test-only evaluator for standard integer and funnel-shift
  operations. Rationale: the oracle directly checks positive, negative, zero,
  and over-width distances, source bit-index ordering, and
  `rotl(a, n) == rotr(a, -n)` without coupling the source contract to unrelated
  downstream conversions. Date/Author: 2026-07-27 / Codex.

- Decision: implement the released OpenQASM 3.1 constant-initializer matrix, not
  the broader draft conversion tables proposed in upstream pull request #666.
  Rationale: the related upstream issues and pull requests remain open, and no
  merged specification revision supersedes the released same-type and
  promotable-constant rules. Date/Author: 2026-07-27 / Codex.

- Decision: name the internal natural-logarithm expression kind `Log`, matching
  the accepted OpenQASM spelling `log()`; retain `ln()` only in the negative
  diagnostic regression. Rationale: this removes a misleading internal
  abbreviation without changing the public source contract. Date/Author:
  2026-07-27 / Codex.

- Decision: treat pull request #1927's memref-backed classical registers as the
  incoming QC contract while remediating this branch. Rationale: implementing
  implicit scalar outputs against the old per-bit result convention would
  immediately require another rewrite and would not compose with the conversion
  and QIR changes already reviewed in #1927. The staged parser and semantic
  model remain authoritative; only the direct emitter and integration tests
  adopt the new QC storage and result representation. Date/Author: 2026-07-29 /
  Codex.

- Decision: represent every source-level implicit or explicit output with one
  ordered typed output descriptor. Rationale: OpenQASM's default output rule
  applies to classical scalars as well as bit registers, whereas a
  `vector<RegisterId>` cannot retain scalar outputs, declared ordering, or the
  conversion required to materialize each result. Bit registers map to the
  memref result convention from #1927; scalar results retain their builtin MLIR
  scalar types. Date/Author: 2026-07-29 / Codex.

- Decision: record semantic conversions as typed expression nodes rather than
  asking the emitter to rediscover promotions. Rationale: overload resolution,
  assignment compatibility, gate-angle conversion, and constant folding are
  source-language decisions. Recording them once keeps the QC emitter mechanical
  and prevents parser acceptance from depending on target-specific type guesses.
  Date/Author: 2026-07-29 / Codex.

- Decision: remove the QIR builtin-profile validation changes from this OpenQASM
  branch and preserve them on an independent local branch with
  parser-independent tests. Rationale: those conversions validate retained MLIR
  operations regardless of how QC was produced, and #1927 substantially changes
  the same QIR conversion surface. Separating them reduces conflicts and lets
  each change be reviewed against its actual contract. Date/Author: 2026-07-29 /
  Codex.

- Decision: preserve scalar outputs as builtin SSA results and bit-register
  outputs as #1927's `memref<nxi1>` results, in one source-ordered result list.
  Rationale: scalars do not need mutable aggregate storage, while bit registers
  must retain the storage contract already established for QC, QCO, and QIR.
  Date/Author: 2026-07-30 / Codex.

- Decision: retain both a classical-register lookup map and a separate
  function-result-ordered list in QIR lowering. Rationale: lookup and ordered
  output emission have different requirements; deriving observable order from a
  hash map is incorrect. Date/Author: 2026-07-30 / Codex.

- Decision: compare `jeff` classical-register result storage by semantic shape,
  accepting the format's tensor spelling only while the program is in `jeff`.
  Rationale: this tests the exchange representation that exists today without
  weakening the exact QC and QCO round-trip contract. Date/Author: 2026-07-30 /
  Codex.

- Decision: validate mixed scalar and bit-register output order through
  QC-to-QCO-to-QC, but do not claim arbitrary scalar QIR output recording.
  Rationale: the current QIR entry-point ABI records returned bit registers and
  uses a scalar return as the entry status. Choosing an ABI for arbitrary scalar
  OpenQASM outputs is a separate target decision. Date/Author: 2026-07-30 /
  Codex.

- Decision: allocate classical memrefs only for bit registers selected as
  program outputs; retain local bit state in SSA and read output-backed bits
  through their observable storage. Rationale: #1927 identifies returned memrefs
  as result registers, while local variables are compiler state rather than ABI
  results. This preserves direct-measurement QIR recording without broadening
  the QIR output contract. Date/Author: 2026-07-30 / Codex.

- Decision: accept partially constrained programs in source semantics and reject
  mixed physical and declared qubits during QC preflight. Rationale: mixing is
  valid OpenQASM but incompatible with the current QC builder's
  static-versus-dynamic allocation mode; the diagnostic therefore belongs to
  target capability, before construction begins. Date/Author: 2026-07-30 /
  Codex.

- Decision: retain classical values in QCO-to-QC `scf.for` and `scf.while` while
  replacing quantum region arguments and results with QC references. Rationale:
  the conversion already preserves classical results for `if` and
  `index_switch`, and SCF natively supports mixed and type-changing loop state.
  Parser-independent regressions prove the general conversion contract.
  Date/Author: 2026-07-30 / Codex.

- Decision: port #1923's scalar-allocation representation into the direct
  emitter, independently of qubit reuse. Rationale: the typed frontend already
  distinguishes unsized `qubit` from explicitly sized `qubit[1]`, and `qc.alloc`
  is the faithful scalar QC representation. Register syntax must retain memref
  allocation. The reuse pass, its pipeline option, and its scheduling policy
  remain wholly owned by #1923. Date/Author: 2026-07-30 / Codex.

## Outcomes & Retrospective

The completed frontend groundwork is retained: the native parser and semantic
analyzer cover the source-language behavior needed by the compiler. The earlier
OQ3 target architecture has been removed in favor of direct QC emission.

The direct architecture and end-to-end behavior are implemented. Twenty-two
OpenQASM fixtures traverse direct QC, QCO cleanup and optimization,
reconstructed QC, and Adaptive QIR; the same fixtures pass `runDefaultPipeline`.
Six straight-line fixtures also reach Base QIR. A separate eleven-program corpus
round-trips through `jeff`, while five explicitly tracked cases reach optimized
QCO and then fail at `intoJeff()`. The standard corpus includes runtime and
induction-variable indexing, checked integer state, and dynamic ranges in
addition to custom gates and structured control flow.

The downstream production corrections are constrained to demonstrated conversion
invariants. QC-to-QCO preserves classical results alongside linear quantum state
through `if`, `for`, and `while` and converts their terminators after region
contents. QCO-to-QC preserves classical `if`, `index_switch`, `for`, and `while`
state while lowering quantum values to QC references. `JeffToQCO` restores
entry-point markers without losing observable results. These areas have
parser-independent native regressions.

Runtime-dynamic indices, multi-iteration induction indices, and non-folded
integer expressions now produce verified QC and reach QIR. Signed operations
assert overflow and invalid division, unsigned operations wrap at 64 bits, and
dynamic ranges use comparison-driven structured control flow. MLIR's canonical
`cf.assert` conversion preserves these checks in QIR. Cases that `jeff` cannot
represent fail at its conversion boundary instead of reducing source support.

The selected emitter remediation preserves exact OpenQASM U-family matrices,
including relative phases once gates are controlled or modified. Runtime integer
powers no longer round through f64, and all operation construction is bounded
independently of projected gate expansion. Focused tests exercise the matrix
contracts directly in QC IR and force each newly budgeted construction category
to fail during preflight.

The selected FU-02 builtins now cover constant and runtime scalar rounding plus
whole-register population count and rotation. Rotation assignment evaluates its
right-hand side from a snapshot before replacing the target, and dynamic
distances are normalized modulo the register width, including negative values.
Focused tests cover self-assignment, width one, nested packed rotations,
control-flow carrying, the shared construction budget, stale constant facts,
standard and Base/Adaptive QIR lowering, a positive `jeff` scalar-rounding
fixture, and an explicit `jeff` bit-vector boundary. Declaration initialization,
casts, sized-`uint` overloads, bit-string literals, and broader folding remain
follow-up work.

The source-side correction now treats scalar `bit` and `bit[1]` as distinct
typed declarations. Register-only population count and rotation reject scalar
bits, while explicit `bit[1]` retains the width-one behavior. Result-level tests
independently confirm constant and runtime rotations for zero, positive,
negative, and over-width distances, the specified low-to-high bit-index order,
the left/right inverse identity, and an observable population-count result.

The final coverage pass exercises behavior that was previously accepted but not
observed directly: every runtime scalar coercion and comparison predicate,
inverse-trigonometric integer conversion, unsigned constant operator family,
wide bit-vector packing and population-count narrowing, and the closed
gate-catalog canonical-name mapping. Clean sequential counters report 4149 of
4542 substantive frontend/emitter lines and 3283 of 5195 branches. A local
Cobertura plus `diff-cover` calculation against `origin/main` reports 4256 of
4651 changed production C++ lines.

The #1927 integration now emits declared scalar and bit-register outputs in
source order and retains OpenQASM's compatibility rule that, without explicit
output declarations, all global classical variables are observable. Semantic
analysis records every promotion and assignment conversion explicitly, gate
parameters carry the distinct angle type, mixed signed/unsigned integer powers
retain the specified signed result, and constant-expression queries are cached
per syntax expression. The emitter consumes those decisions instead of
re-resolving source typing. A public compiler regression additionally preserves
the ordered `i64`, `memref<2xi1>`, and `f64` signature through QC-to-QCO-to-QC.
Arbitrary scalar QIR output recording remains explicitly out of scope until its
entry-point ABI is defined.

Only output bit registers now allocate #1927's classical-result memrefs.
Non-output bits remain SSA state, including across structured control flow, so
local initialization cannot be mistaken for QIR output recording. A restored
numeric-and-bit loop fixture demonstrates the explicit and default
QC-to-QCO-to-QC-to-QIR pipelines, while native QCO regressions cover mixed and
type-changing SCF loop signatures.

The integration exposed and fixed three source-independent downstream defects:
QC-to-QCO no longer mistakes classical memref loads for qubit state, QCO-to-QC
retains classical loop state, and QIR output recording follows function-result
order rather than hash-map iteration. The public compiler corpus also
distinguishes the temporary tensor spelling of classical registers in `jeff`
from the exact memref signature required after restoration.

The monolithic OpenQASM target test has been split into parser, semantics, and
emitter files with a small namespaced helper header. The resulting suites retain
their existing target and coverage identity while making ownership and review
boundaries explicit.

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
before creating each affected application. A `pow @` modifier produces an
ordered `qc.pow` region; scalar exponentiation and the scalar `pow()` function
remain separate classical expressions.

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

Add positive direct-emission tests for representative primitive and custom gates
and for constant, floating, dynamic, nested, controlled, inverted, and broadcast
`pow @` modifiers. Retain a source-located failure for constant integer
exponents that cannot be represented exactly by QC's f64 exponent. Add
equivalent focused cases for every other frontend-accepted feature that the
emitter rejects. Update `mlir/unittests/Target/OpenQASM/CMakeLists.txt` to link
`MLIROpenQASMFrontend`, `MLIRQCTranslation`, and only directly used test
libraries.

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
straight-line subset. List `pow @` as supported by QC and record downstream
canonicalization restrictions separately. Update `CHANGELOG.md` to describe
direct OpenQASM import without an OQ3 dialect claim.

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
broadcasting, controls, inverse, negative controls, ordered power modifiers,
expressions, dynamic indices, measurements, reset, barrier, and structured
control flow must retain their tested behavior. Valid numeric `pow @` programs
must produce `qc.pow`; constant integer exponents must not be rounded silently
when converted to f64.

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

Make the architecture transition in coherent local commits when useful. Before
removing an old source, ensure its required direct-emission behavior has moved
into the private emitter and its tests pass. If a downstream fixture fails,
preserve the failing source, reduce it to native IR, and repair the owning
conversion instead of introducing a parser-side special case.

Never discard unrelated user changes or edit another task worktree. The current
authorization covers a force-with-lease update of the already-open PR branch
after the requested rebase. Changing pull request state, resolving review
threads, or publishing comments still requires separate human authorization and
the disclosure required by `docs/ai_usage.md`.

## Artifacts and Notes

The completed groundwork before this revision comprised an LLVM-native staged
frontend, 27 imported behavior fixtures, source control flow and carried state,
and clean focused validation. It also comprised an OQ3 dialect and OQ3-to-QC
pass that this plan now deliberately removes. Earlier OQ3-specific test counts
are historical evidence, not revised acceptance evidence.

The power target-boundary proof after implementation must read:

    analyzeOpenQASM(pow-source) succeeds.
    translateQASM3ToQC(pow-source) produces an ordered qc.pow region.
    inexact constant integer exponents fail before silent f64 rounding.
    unsupported composite powers fail at the owning downstream conversion.
    No OQ3 module is constructed.

The standard-chain proof must record a representative structured fixture
reaching:

    OpenQASM -> QC -> QCO -> optimized QCO -> QC -> Adaptive QIR
    -> LLVM IR and bitcode

The final corpus contains twenty-two standard programs, eleven `jeff` round-trip
programs, five `jeff`-incompatible programs, and six Base programs. One native
`JeffToQCO` regression proves that a serialized entry point with observable
results regains its marker without losing those results. A native QC-to-QIR
regression proves that `cf.assert` lowers through LLVM. The latest focused
validation results are:

    OpenQASM frontend and target: 137 tests passed.
    QC translation: 257 tests passed.
    Compiler pipeline: 191 tests passed.
    QC-to-QIR Adaptive: 130 tests passed.
    QC-to-QIR Base: 112 tests passed.
    Legacy OpenQASM parser: 101 tests passed.
    Changed-file repository hooks and diff checks: passed.
    Substantive frontend/emitter coverage: 4149/4542 lines and 3283/5195 branches.
    Changed production C++ diff coverage: 4256/4651 lines.

The remaining fresh-review findings are recorded outside this baseline repair
and require the review-selection gate before implementation.

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

Revision note (2026-07-27): selected PR review remediation added closed,
phase-aware U-family lowering recipes, exact runtime integer-to-f64 power
guards, and an authoritative operation-construction budget with complete
preflight accounting. Independent phase tests deliberately evaluate QC IR
locally rather than depending on downstream QCO conversion, mapping, or DD
functionality.

Revision note (2026-07-27): selected FU-02 work added constant and runtime
`ceiling()` and `floor()` through the existing typed scalar-expression path.

Revision note (2026-07-27): the second selected FU-02 slice added typed
whole-bit-register `popcount()`, `rotl()`, and `rotr()`. Emission uses a lazy
bits-or-packed representation, includes linear packing work in the existing
construction budget, invalidates constant facts through bit-register
generations, and preserves retained Math and LLVM operations until an owning
target conversion either supports or explicitly diagnoses them.

Revision note (2026-07-27): the FU-02 source correction retained scalar-versus-
register bit type identity, enforced the register-only builtin signatures, and
added returned-value rotation and population-count oracles without changing QIR
conversion code or widening support to casts, bit strings, or sized integers.

Revision note (2026-07-27): the final selected cleanup reconciled the released
constant-initializer contract with still-open upstream proposals, made the
changed OpenQASM surface clean under Clang-Tidy 22.1.8, adopted project integer
type spelling, renamed the internal logarithm kind to `Log`, and added a
2,048-definition custom-gate indexing stress test. Foldable builtins remain in
the positive optimized corpora; retained operations are tracked as explicit
`jeff` and QC-to-QIR boundary failures.

Revision note (2026-07-27): the QIR boundary correction keeps foldable scalar
rounding in the positive corpus and adds direct Base and Adaptive diagnostics
for retained rounding, population-count, and funnel-shift operations. It does
not broaden either profile or turn `jeff` compatibility into a prerequisite for
successful OpenQASM-to-QC translation.

Revision note (2026-07-31): maintainer cleanup raised the authoritative QC
emission ceiling to ten million operations, revised its preflight regressions to
remain bounded in test execution, used `std::ignore` for deliberately discarded
nodiscard results, and avoided `module` as a C++ variable name in the PR-owned
implementation and tests.
