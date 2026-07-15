# Build a typed OpenQASM 3 frontend from MQT Core's handwritten parser

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs one maintainable OpenQASM frontend that accepts OpenQASM 3 and
the supported OpenQASM 2 compatibility syntax, reports precise source
diagnostics, and can preserve valid source constructs even when QC or JEFF
cannot lower them yet. A caller should be able to parse a source file, perform
source-language semantic analysis, inspect a verified typed representation, and
then request target lowering. Source errors and target-capability errors must be
different failures.

The implementation will reuse MQT Core's existing handwritten `qasm3` lexer and
recursive-descent parser as its initial syntax source. It will not use ANTLR.
The MLIR frontend will own a separate semantic model and analyzer rather than
making the legacy `QuantumComputation` importer or its mutable environments the
architectural center. Sharing scanner and parser infrastructure is acceptable;
duplicating semantic utilities is preferable when that keeps the MLIR stages
clean. A separate semantic analyzer will resolve names, scopes, types,
constants, overloads, gate signatures, and include policy into an arena-owned
typed program. A small experimental OQ3 MLIR dialect will preserve source
concepts for which builtin MLIR has no faithful equivalent. A dedicated emitter
will map the already typed program to OQ3 plus builtin, `arith`, `math`, `func`,
`scf`, and `memref` operations. The emitter must not repeat source typing.

A human can observe the intended separation through three focused test layers.
Parser tests accept or reject syntax without constructing MLIR. Semantic tests
produce typed programs or source-spanned diagnostics without an MLIR context.
Emitter and lowering tests accept typed programs or programmatically built OQ3
IR, verify the module, and distinguish malformed IR from unsupported target
features.

The repository-relative scope is `include/mqt-core/qasm3`, `src/qasm3`, their
existing tests, the experimental dialect under `mlir/include/mlir/Dialect/OQ3`
and `mlir/lib/Dialect/OQ3`, the future adapter under
`mlir/include/mlir/Target/OpenQASM` and `mlir/lib/Target/OpenQASM`, focused
tests, and directly related build and documentation files. The existing
`QuantumComputation` importer stays available as a behavioral oracle, but it is
not required to adopt the MLIR semantic program. The production MLIR
`translateQASM3ToQC` entry point is replaced by the staged frontend in this
milestone rather than retaining two MLIR translation implementations. Unrelated
changes and other worktrees remain outside scope.

## Progress

- [x] (2026-07-15 10:10Z) Started from current main and kept the established
  importer available as a comparison oracle.
- [x] (2026-07-15 10:50Z) Added the experimental OQ3 types, gate operations,
      inclusive range operation, local verifiers, and initial OQ3-to-QC
      lowering.
- [x] (2026-07-15 11:40Z) Integrated the compiler architecture from pull request
      1815 and adopted the ExecPlan process introduced and refined by pull
      requests 1907 and 1908.
- [x] (2026-07-15 12:00Z) Established the version policy: explicit 3.0, explicit
      3.1, and versionless input select the same maintained OpenQASM 3
      semantics; explicit 2.0 selects compatibility mode.
- [x] (2026-07-15 13:20Z) Inspected Daniel's public and private feedback, MQT
      Core's existing parser and semantic passes, the legacy native-gate
      catalog, and Qiskit's `openqasm3_parser` source at commit
      `3eac9970f37baf6d030a3a185b9421cca3cf0a59`.
- [x] (2026-07-15 13:40Z) Removed the ANTLR runtime, generated parser, grammar
      target, direct parse-tree-to-MLIR builder, frontend tests tied to that
      builder, and generated-file lint exceptions. The historical implementation
      remains in Git history for comparison.
- [x] (2026-07-15 13:40Z) Added semantics-preserving QC lowering for `cu`,
  `cu3`, and `cu1`. Four-parameter `cu` retains its control-qubit phase;
  `cu3` lowers to controlled U and `cu1` to controlled P.
- [x] (2026-07-15 13:50Z) Added parser-independent OQ3 verifier and lowering
      tests, rebuilt OQ3 and `mqt-cc`, and passed 3 focused OQ3 tests, 224
      existing QC translation tests, and 116 compiler tests.
- [x] (2026-07-15 13:50Z) Passed repository lint, inspected and committed the
      architecture pivot, rebased onto current `origin/main` at `559fde4b2`, and
      repeated the focused OQ3, QC translation, and compiler tests.
- [x] (2026-07-15 13:55Z) Force-pushed the rebased branch with lease protection
      and updated draft pull request 1910 to describe the handwritten-parser
      architecture and current workbench status.
- [x] (2026-07-15 14:10Z) Expanded the delivery decision: replace the existing
      MLIR OpenQASM-to-QC translator immediately with parse, analyze, emit OQ3,
      and lower OQ3 stages; keep the legacy `QuantumComputation` importer
      independent and duplicate semantic code where sharing would couple them.
- [x] (2026-07-15 15:10Z) Introduced MLIR-owned opaque parsed-program and
      value-oriented typed-program APIs. Parsing and semantic-analysis tests run
      without constructing an MLIR context and retain source filenames and
      line/column diagnostics.
- [x] (2026-07-15 15:40Z) Added typed-program-to-OQ3 emission and made
      `translateQASM3ToQC` a thin composition of parse, analyze, emit, verify,
      and lower stages.
- [x] (2026-07-15 15:55Z) Deleted the 1,114-line direct AST-to-QC visitor. All
      117 existing OpenQASM translation fixtures now pass through the staged
      replacement, including the established OpenQASM 2 compatibility cases.
- [ ] Establish a checked upstream grammar and conformance snapshot as test
  ground truth, without making generated code or ANTLR a build dependency.
- [ ] Refactor the existing scanner and parser to return a source-spanned syntax
  program with recovery and multiple diagnostics (completed: one
  precedence-climbing table now parses the complete scalar binary-operator
  hierarchy, including right-associative `**`; remaining: byte spans, arena
  ownership, recovery, and unsupported syntax families).
- [ ] Replace the separate legacy constant/type passes with one semantic
  analyzer that produces an arena-owned typed program consumed by both the
  circuit importer and OQ3 emitter.
- [ ] Implement the thin parse/analyze/emit translation API and one defensive
  whole-program OQ3 verifier for cross-operation invariants.
- [ ] Complete OpenQASM 3 grammar and semantic coverage, OpenQASM 2
  normalization, differential tests, and measured linear scaling.
- [x] (2026-07-15 16:05Z) Added dedicated stage-boundary tests proving semantic
      analysis is MLIR-independent, OQ3 verifies before lowering, and the
      production convenience wrapper leaves no OQ3 operations after lowering.
- [x] (2026-07-15 16:25Z) Made standard-library availability explicit in the
      typed program. Strict mode requires `include "stdgates.inc";`, while the
      default compatibility mode preserves implicit native-gate convenience;
      user definitions may use unavailable standard-library names in strict
      mode.
- [x] (2026-07-15 16:35Z) Added typed mixed-numeric gate expressions with
      signed/unsigned MLIR casts, preserved valid `pow` modifiers through OQ3,
      and covered inverse native aliases. Passed 4 OQ3 tests, 10 staged-frontend
      tests, 224 translation tests, and 116 downstream compiler tests from a
      clean rebuild.
- [x] (2026-07-15 17:11Z) Simplified alternating `ctrl` and `negctrl` lowering
      to recursive single-control regions. Negative-control polarity flips are
      emitted once around the outermost modifier tree, and existing QC
      canonicalization still combines adjacent controls for compact target IR.
      Added a focused alternating-modifier test and retained all 224 QC
      translation parity results.
- [x] (2026-07-15 17:30Z) Added typed OQ3 emission for `sin`, `cos`, `tan`,
      `exp`, `ln`, and `sqrt`, and replaced the parser's incomplete expression
      ladder with the OpenQASM precedence hierarchy. Power now uses `**` and is
      right associative; `^` now means bitwise XOR; modulo, shifts, bitwise,
      logical, equality, and comparison operators all parse at their specified
      precedence. All 98 QASM parser tests and 12 staged-frontend tests pass.
- [x] (2026-07-15 18:10Z) Performed two independent MLIR-focused reviews of the
      draft and stabilized the accepted local invariants: OQ3 gate and range
      regions now require `oq3.yield`; gate applications reject surplus qubits
      without controls and non-integer control counts; variadic native gates
      retain their declared minimum arity; duplicate qubits and unsupported
      integer declarations receive source diagnostics; and explicit unsupported
      OpenQASM 3 minor versions are rejected. Added target and verifier tests,
      including a constant-zero range step that must fail before it can form an
      infinite target loop.
- [x] (2026-07-15 18:25Z) Fixed the OQ3 dialect documentation target after the
      draft pull request's Read the Docs build reported that multiple dialects
      occur in `OQ3Ops.td`. The generated OQ3 operation definition now shares
      that file with the dialect definition, so `add_mlir_doc` explicitly
      selects `oq3`; `OQ3DialectDocGen` builds successfully.
- [x] (2026-07-15 19:15Z) Made the exact Read the Docs command pass. Escaped a
      portable CMake regular-expression literal, added the OQ3 dialect page to
      the MLIR toctree, and generated its pass reference beneath `Passes/` so no
      orphaned generated Markdown page reaches Sphinx.
- [x] (2026-07-15 19:15Z) Added a scoped `mlir/.clang-tidy` policy based on the
      upstream MLIR check set. It retains correctness and performance checks
      suitable for MLIR while avoiding repository-wide checks that diagnose
      generated operation code and intentional MLIR conventions.
- [x] (2026-07-15 19:25Z) Addressed the remaining PR lint diagnostics in source
      rather than broadening suppressions: removed two redundant constant moves,
      made the parser operator table's members explicit, converted the implicit
      statement counter deliberately, and initialized the defensive unary-op
      fallback. The complete lint gate and focused parser/frontend tests pass.
- [x] (2026-07-15 19:45Z) Added behavior-driven staged-frontend tests for custom
      gates in conditionals, reset and barrier emission, OpenQASM 2 controlled
      gate compatibility prefixes, and observable diagnostics for unmeasured
      outputs, unmeasured conditions, zero controls, and mixed broadcast
      operands. All 20 focused frontend tests pass.
- [ ] Expand behavior-driven frontend and lowering tests until the C++ patch
      coverage of the OQ3 foundation reaches the configured 90% target (current
      Codecov report: 76.78%, 322 missed lines). Do not lower the threshold or
      exclude the frontend merely to make the status green.

## Surprises & Discoveries

- Observation: MQT Core's mainline OpenQASM implementation is not based on
  generated ANTLR code. `src/qasm3/Scanner.cpp` and `src/qasm3/Parser.cpp` are a
  handwritten scanner and recursive-descent parser; `ConstEvalPass.cpp` and
  `TypeCheckPass.cpp` already separate two semantic concerns from parsing. This
  makes extension and consolidation lower risk than introducing a second parser
  stack.

- Observation: Qiskit's reported eighty-fold parsing speedup is a README claim
  from a crude large-file comparison, not a reproducible benchmark in that
  repository. The architectural evidence is still useful: its custom lexer feeds
  an event-based recursive-descent and Pratt parser, a source-spanned syntax
  tree, and a distinct typed abstract semantic graph. Its repository has no
  Criterion benchmark or checked benchmark data supporting the exact factor.

- Observation: The removed ANTLR demonstrator combined parsing-tree traversal,
  source semantic checks, symbol state, builtin MLIR construction, and OQ3
  emission in one `SemanticBuilder`. The public function was staged in name but
  not internally separated. This confirmed Daniel's concern that another
  refactor would otherwise restart from the same coupled architecture.

- Observation: Operation-level MLIR verifiers cannot prove every OpenQASM
  invariant. Operand shape and type relationships are local, while recursive
  gates, declaration order, symbol visibility, and some modifier relationships
  need surrounding-program information. A single defensive whole-program
  verifier is required for programmatically constructed IR, but it must not
  become the source semantic analyzer.

- Observation: The legacy gate catalog intentionally accepts many names beyond
  the strict OpenQASM standard-library set. This is user-facing convenience, not
  accidental parser behavior. Strict conformance and compatibility require an
  explicit gate-policy choice rather than silently dropping or globally
  injecting those names.

- Observation: OpenQASM's four-parameter `cu(theta, phi, lambda, gamma)` is not
  exactly `ctrl @ U(theta, phi, lambda)`: it also applies `p(gamma)` to the
  control. Native lowering must retain that relative phase. The three-parameter
  `cu3` and one-parameter `cu1` aliases map directly to controlled U and
  controlled P.

- Observation: Direct use of `arith` and `memref` is not itself a layering
  problem. It becomes a problem only if those operations are built while source
  typing is still being decided. Emitting them from an already typed semantic
  program avoids duplicating an entire classical OQ3 operation set and keeps the
  semantic boundary clear.

- Observation: the shared parser injects a synthetic `cu` declaration, but its
  current `DebugInfo` is created after the include scanner has unwound and can
  therefore name the main source buffer. The new analyzer recognizes this
  parser-owned declaration structurally for now. The parser refactor must mark
  synthetic declarations explicitly instead of inferring provenance from a
  filename.

- Observation: the repository's broad `target/` ignore pattern also matches
  MLIR's conventional capitalized `Target/` source directories on
  case-insensitive filesystems. Narrow exceptions are required for the new
  source and test targets while retaining the build-artifact ignore.

- Observation: `qc.ctrl` already has a `MergeNestedCtrl` canonicalization that
  combines adjacent nested controls. The OQ3 lowerer can therefore construct the
  direct recursive semantics without duplicating grouping logic, while the
  standard QC cleanup pipeline still produces the compact multi-control form
  used by existing consumers and parity tests.

- Observation: the handwritten expression parser recognized tokens for all
  scalar operators but parsed only arithmetic and comparison subsets. It also
  interpreted `^` as power and gave comparisons higher precedence than addition.
  The maintained OpenQASM hierarchy uses right-associative `**` for power and
  `^` for bitwise XOR, followed by unary, multiplicative, additive, shift,
  comparison, equality, bitwise, and logical levels.

- Observation: a local operation verifier must defend IR created outside the
  source frontend. `oq3.apply_gate` previously accepted surplus qubits and
  control-count operands of arbitrary scalar type; `GateOp` and `ForOp` did not
  require the terminator that lowering omits while cloning. These are malformed
  typed IR, not source-language analysis concerns. Local checks and
  `SingleBlockImplicitTerminator` traits preserve that boundary.

- Observation: accepting an `int` or `uint` declaration and emitting it as a
  `bit` register silently changes signedness, width semantics, and stored
  values. The analyzer now rejects those declarations until typed storage and
  assignment lowering exist.

- Observation: the Read the Docs environment treats the `\.` escape in a quoted
  CMake string as a developer warning, whereas the local configuration did not
  expose it before OQ3 documentation generation reached the cleanup script.
  CMake requires `\\.` in the source string for a literal dot in the regular
  expression.

- Observation: Codecov's C++ patch status is red for substantive missing test
  coverage, not because the coverage job failed. Its report for this branch has
  1,065 covered and 322 missed patch lines (76.78%). The coverage target must be
  met by exercising error paths, custom gates, lowering variants, and parser
  diagnostics rather than by changing Codecov policy.

## Decision Log

- Decision: Remove ANTLR and the direct parse-tree-to-MLIR demonstrator now.
  Rationale: the implementation is preserved in Git history, while keeping it on
  the branch would impose a large dependency and generated-code review burden on
  an architecture that is being replaced. Date/Author: 2026-07-15 / Codex,
  following maintainer direction.

- Decision: Extend MQT Core's handwritten frontend and take architectural
  inspiration, but no copied implementation, from Qiskit's parser. Rationale:
  the existing C++ scanner, parser, AST, source debug information, type checker,
  constant evaluator, OpenQASM 2 support, and native gates provide a practical
  base. A Rust dependency or a transliteration of rust-analyzer's red/green tree
  would add cost without first proving a repository-specific benefit.
  Date/Author: 2026-07-15 / Codex.

- Decision: Define three explicit internal stages: `parseOpenQASM` returns a
  syntax program, `analyzeOpenQASM` returns a typed semantic program, and
  `emitOQ3` returns MLIR. `translateOpenQASMToOQ3` only composes them.
  Rationale: source typing remains testable without MLIR, and emission cannot
  silently add a second source type system. Date/Author: 2026-07-15 / Codex.

- Decision: Perform source semantic analysis during typed-program construction
  in one primary traversal. Rationale: this is the efficient place to maintain
  lexical scopes, resolve symbols, fold required constants, and attach types and
  diagnostics. Operation verifiers defend IR invariants; they do not replace
  language analysis. Date/Author: 2026-07-15 / Codex.

- Decision: Preserve original source spans in syntax and semantic nodes.
  Rationale: semantic diagnostics should use the source manager directly, and
  every emitted MLIR operation should receive a file/line/column range derived
  from the same span. Downstream diagnostics then remain useful without trying
  to reconstruct source positions from MLIR. Date/Author: 2026-07-15 / Codex.

- Decision: Use builtin MLIR, `arith`, `math`, `func`, `scf`, and `memref` after
  semantic analysis whenever their semantics match. Retain OQ3 operations only
  for source distinctions such as bit versus bool, angles, source gate symbols
  and ordered modifiers, inclusive ranges, timing, and calibration. In
  particular, emit ordinary `scf.if` and `scf.while` directly; retain `oq3.for`
  while its inclusive range and dynamic nonzero-step contract cannot be
  represented faithfully by a standard SCF operation. Date/Author: 2026-07-15 /
  Codex.

- Decision: Keep a strict specification gate policy and an MQT compatibility
  gate policy. The staged experimental API and existing convenience import API
  default to compatibility mode so the architectural replacement does not
  silently remove long-standing native gate names. Strict mode is explicit. Both
  policies draw from one canonical gate catalog containing availability, arity,
  native QC mapping, and aliases. Date/Author: 2026-07-15 / Codex.

- Decision: Implement `cu`, `cu3`, and `cu1` natively and never expand them to
  source gate definitions merely to lower them. Rationale: QC already represents
  their primitive operations and controls; direct lowering is linear, preserves
  dynamic parameters, and retains the legacy convenience. Date/Author:
  2026-07-15 / Codex.

- Decision: Maintain one current OpenQASM 3 semantic profile based on 3.1.
  Explicit 3.0 and 3.1 declarations and versionless input use that profile;
  explicit 2.0 uses compatibility normalization. Rationale: MQT tracks one
  maintained revision while avoiding unnecessary rejection of 3.0 headers.
  Date/Author: 2026-07-15 / Codex.

- Decision: Replace the production MLIR translation path in this branch instead
  of keeping the direct AST-to-QC implementation as a fallback. Rationale: two
  live MLIR importers would obscure which semantics are authoritative and would
  postpone the most valuable architectural test. The existing QC translation
  tests remain the oracle, and Git history provides rollback. Date/Author:
  2026-07-15 / Codex, following maintainer direction.

- Decision: Reuse the handwritten scanner and parser initially, but give the
  MLIR frontend its own value-oriented typed program and semantic analyzer. Do
  not require the legacy `QuantumComputation` importer to consume that model.
  Rationale: syntax sharing avoids gratuitous parser duplication while semantic
  independence permits clean source-to-MLIR design and incremental grammar
  improvements. Date/Author: 2026-07-15 / Codex, following maintainer direction.

- Decision: Expose explicit include names and the count of parser-injected
  statements as parser metadata without changing the legacy importer's default
  behavior. Rationale: semantic policy must distinguish a source include from
  the legacy parser's synthetic `cu` definition; checking a synthetic node's
  filename was unreliable after scanner-stack unwinding. Date/Author: 2026-07-15
  / Codex.

- Decision: Lower every expanded control as one recursive `qc.ctrl`, regardless
  of whether the source modifier is positive or negative, and emit X gates for
  every negative-control qubit before and after the entire modifier tree.
  Rationale: polarity changes must not execute conditionally inside an earlier
  positive control. This representation handles alternating controls and
  inverses uniformly, and QC's existing canonicalization combines adjacent
  controls when a compact form is desired. Date/Author: 2026-07-15 / Codex,
  following maintainer feedback.

- Decision: Replace the partial expression ladder now with a table-driven
  precedence-climbing parser while retaining the existing public parser entry
  points. Rationale: this fixes observable specification mismatches immediately
  and establishes the core of the planned Pratt-style expression parser without
  coupling syntax parsing to MLIR or forcing a simultaneous ownership rewrite.
  Date/Author: 2026-07-15 / Codex.

- Decision: Address only local, testable review findings in this stabilization
  batch and defer SourceMgr-backed include resolution. Rationale: operation
  verification, source arity, and version checks have narrow ownership and
  direct regression tests. Reworking the parser's shared include stack needs a
  dedicated MLIR adapter or duplicated resolver so it does not accidentally
  perturb the legacy `QuantumComputation` importer. Date/Author: 2026-07-15 /
  Codex.

- Decision: Use a scoped MLIR clang-tidy configuration rather than applying the
  repository-wide configuration to MLIR sources. Rationale: the CI reports 265
  diagnostics from generated operation fragments and checks that do not match
  MLIR's established conventions. The scoped configuration follows upstream
  MLIR's intentionally narrower safety, modernization, and performance set; it
  leaves the root configuration unchanged for the rest of MQT Core. Date/
  Author: 2026-07-15 / Codex.

## Outcomes & Retrospective

The first architecture demonstrator established that a small typed OQ3 dialect
can preserve gates, ordered modifiers, bit values, and inclusive ranges while
lowering supported operations to QC. Review then exposed that its frontend
boundary was only nominal: the ANTLR visitor still decided source semantics
while emitting MLIR. The useful dialect and lowering work has been retained, and
the parser stack has been removed before more features accumulated around the
wrong boundary.

The revised foundation reuses project knowledge instead of restarting. The
handwritten parser remains incomplete and its shared-pointer AST is not the
desired final ownership model, but it already contains tested OpenQASM 2
compatibility, native gate behavior, source debug information, and expression
parsing. The MLIR-owned typed program now isolates those implementation details,
OQ3 emission is a separate walk, and the production MLIR entry point uses the
staged implementation exclusively. The legacy importer retains its current
constant and type passes and serves as the 117-fixture behavioral oracle.

The latest iteration removed special-case control grouping from OQ3 lowering.
Alternating positive and negative controls now use one recursive rule, and the
existing QC cleanup still recovers multi-control operations. It also completed
the parser's scalar binary-operator precedence and added dynamic scalar math
function emission. Source spans, recovery, arena ownership, mutable classical
state, and the remaining statement families are still future milestones.

## Context and Orientation

`src/qasm3/Scanner.cpp` converts source characters to `qasm3::Token` values.
`src/qasm3/Parser.cpp` is a handwritten recursive-descent parser that constructs
the classes declared in `include/mqt-core/qasm3/Statement.hpp`. Expressions are
currently heap allocated through `std::shared_ptr`; source information is a
`DebugInfo` containing one line and column plus an include-parent chain.

`src/qasm3/passes/ConstEvalPass.cpp` and `src/qasm3/passes/TypeCheckPass.cpp`
traverse the syntax objects. The importer in `src/qasm3/Importer.cpp` currently
invokes both passes statement by statement and then immediately emits a
`qc::QuantumComputation`. This preserves some separation but does not expose a
complete typed program and can report only one failure at a time.

The new source layer will introduce `SourceId`, `SourceSpan`, and `Diagnostic`
under `include/mqt-core/qasm3`. A source ID identifies one main or included
buffer. A source span is a half-open byte range within that buffer. A diagnostic
contains severity, message, primary span, and optional related spans. Line and
column text is computed only when displaying a diagnostic, so scanning and
semantic analysis use compact offsets.

The syntax layer will own nodes in arenas and refer to them by small IDs instead
of recursive shared ownership. `SyntaxProgram` records statements and
expressions exactly as parsed, including unsupported families. The parser uses
recursive descent for statements and a Pratt parser for expressions. A Pratt
parser is a precedence-driven expression parser that handles prefix, infix, and
postfix operators in one table. Error recovery synchronizes at semicolons,
braces, and declaration or statement starters so one run can report multiple
independent errors.

The semantic layer will produce `TypedProgram`. Each expression has a resolved
OpenQASM type and each identifier use has a `SymbolId`; mutable declarations,
constants, gate definitions, inputs, outputs, ranges, and callable signatures
carry their source spans. This program is the only source-language semantic
truth consumed by both the circuit importer and the OQ3 emitter.

The OQ3 dialect lives under `mlir/include/mlir/Dialect/OQ3` and
`mlir/lib/Dialect/OQ3`. OQ3 is a semantic high-level IR, not a syntax tree. Its
operation verifiers check local shape and type invariants. One module-level
verification pass checks cross-operation symbol resolution, declaration order,
recursion, and modifier relationships for programmatically constructed IR. The
pass runs once after emission; it should diagnose compiler bugs or malformed
textual OQ3, not repeat source analysis.

## Plan of Work

The first milestone leaves a clean non-ANTLR base. Keep the OQ3 dialect and
lowering, remove every ANTLR dependency and generated source, and move tests
that exercise the dialect or lowering into `mlir/unittests/Dialect/OQ3` using
programmatically built IR. Test native `cu`, `cu3`, and `cu1` lowering,
including the `gamma` phase of four-parameter `cu`. Acceptance is that the build
contains no ANTLR target or generated parser and the OQ3 tests pass
independently of any source parser.

The second milestone makes upstream syntax measurable ground truth. Add a
pinned, unmodified OpenQASM 3.1 grammar and conformance-example snapshot under
`vendor/openqasm/` with upstream revision, license, hashes, and a regeneration
script. These files are test data, not compiled parser input, and are excluded
from typo fixing, license rewriting, and auto-formatting just like other
vendored sources. Generate a checked coverage manifest that maps every grammar
production to positive and negative parser tests. Add OpenQASM 2 fixtures
already supported by main. Acceptance is that CI can identify missing
productions and the normal build has no parser-generator dependency.

The third milestone replaces the production MLIR translation before the full
parser refactor. Add an opaque parsed-program API around the existing parser and
an MLIR-independent, value-oriented typed program. Analyze declarations, gate
signatures and bodies, constants, operands, modifiers, broadcasting,
measurements, resets, barriers, and conditionals into that program. Emit OQ3,
builtin classical operations, SCF, and existing QC state operations from the
typed program. Make `translateQASM3ToQC` compose parsing, analysis, OQ3
emission, verification, and OQ3-to-QC lowering, then remove the direct AST-to-QC
visitor. Acceptance is that no production MLIR path constructs QC while
resolving source types and all existing QC translation regressions run through
the staged path.

The fourth milestone refactors scanning and parsing. Add source-buffer
ownership, byte spans, structured diagnostics, arena-owned syntax nodes,
recovery, and a Pratt expression table while preserving the existing parser API
through a temporary adapter. Port statement families incrementally and run
legacy and new syntax tests together. Acceptance is complete syntax coverage,
multiple useful diagnostics from one invalid file, accurate include-stack spans,
and linear token and parse growth.

The fifth milestone completes the semantic analyzer. Consolidate symbol scopes,
type checking, required constant evaluation, gate lookup, broadcasting,
input/output ordering, version policy, and include policy into
`analyzeOpenQASM`. It emits `TypedProgram` only if no source semantic errors
remain. Unsupported target features are still representable. Acceptance is that
semantic tests run without MLIR and cover unknown, recursive,
use-before-defined, arity, type, width, index, scope, cast, range-step, and
compatibility failures with original source spans.

The sixth milestone completes MLIR emission beyond the initial production
replacement. Implement `translateOpenQASMToOQ3` as parse, analyze, emit.
`emitOQ3` maps typed values to builtin storage and arithmetic, uses SCF directly
where faithful, emits only source-specific OQ3 operations where needed, attaches
range locations, and does not perform source type inference. Add the
module-level OQ3 verifier and keep the textual dialect experimental. Acceptance
is that every successful module verifies and an injected malformed module fails
the defensive verifier at the source-derived operation location.

The seventh milestone preserves compatibility deliberately. Move the standard,
qelib1, and MQT-native names into one gate catalog. Strict mode makes only `U`
and `gphase` language builtins and loads standard libraries only when included.
Compatibility mode preserves the legacy implicit native catalog, including
additional MQT operations and aliases. Inventory every existing legacy name and
classify it as specification, qelib1, alias, or MQT extension. Acceptance is
that no gate silently changes availability and every native mapping has a QC
lowering test.

The final milestones complete source families and objective evidence. Add
arrays, aliases, subroutines and externs, switch and loop control, timing,
annotations, pragmas, calibration, and power lowering as downstream dialects
permit. Differentially compare both importers over their overlap. Benchmark a
flat gate stream, nested expressions, includes, and repeated custom gates at
increasing sizes. Require approximately linear growth, bounded diagnostic
recovery, no eager custom-gate expansion, and no expression-string cache keys or
shared-pointer ownership in the new syntax and semantic programs.

## Concrete Steps

Run every command from the repository root. Configure and build the OQ3
foundation with:

    cmake --preset debug
    cmake --build --preset debug --target mqt-core-mlir-unittest-oq3 mqt-cc -j4

Run the dialect and lowering tests with:

    ./build/debug/mlir/unittests/Dialect/OQ3/mqt-core-mlir-unittest-oq3

After parser work begins, build and run the existing QASM/QC regression suite as
the oracle:

    cmake --build --preset debug --target mqt-core-mlir-unittest-qc-translation -j4
    ./build/debug/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation

Add a dedicated frontend test target under the existing `test` or
`mlir/unittests` tree when `parseOpenQASM` and `analyzeOpenQASM` exist. Its
tests must be separable into syntax-only, semantic-only, emission, and lowering
filters.

For performance evidence, generate sources with fixed seeds and gate counts of
1,000, 10,000, and 100,000. Record bytes, tokens, syntax time, semantic time,
emission time, and peak resident memory separately. Run each size enough times
to report a median. Compare against main's existing parser and, if useful, the
historical ANTLR commit in an isolated checkout; do not restore ANTLR to the
task branch.

After each completed batch, run:

    uvx nox -s lint
    git diff --check origin/main...HEAD
    git status --short --branch

If MLIR is not discoverable, point `MLIR_DIR` at an installed MLIR 22.1 CMake
package without recording a machine-specific path in this plan. Keep generated
build output inside the ignored `build` directory.

## Validation and Acceptance

Syntax acceptance requires exact recognition of the pinned OpenQASM 3 grammar,
the supported OpenQASM 2 compatibility grammar, comments, version placement, and
includes. Explicit `OPENQASM 3.0;`, explicit `OPENQASM 3.1;`, and no version
declaration select the same current OpenQASM 3 semantics. Unsupported explicit
versions receive a source-spanned diagnostic.

Semantic acceptance requires one typed result per successful source program and
no MLIR dependency. Unknown symbols, use before definition, recursion, duplicate
bindings, incompatible types, illegal casts, arity, indexing, broadcasting,
input/output, and constant-zero range steps fail with primary and related source
spans where applicable. A runtime range step remains valid typed source.

Emitter acceptance requires that ordinary classical computation uses builtin
MLIR dialects after semantic analysis and that every operation has a location
derived from its source span. `if` and `while` use SCF directly. Inclusive
ranges remain OQ3 until their semantics can be lowered safely. No Any-like type
or fallback operation is permitted.

Lowering acceptance requires native `cu`, `cu3`, and `cu1`; ordered `inv`,
`ctrl`, and `negctrl`; safe positive and negative inclusive ranges; and clear
target diagnostics for dynamic zero risk and unsupported power forms. The
four-parameter `cu` test must observe both the control phase and controlled U.

Performance acceptance requires approximately linear time and memory for flat
programs and no unexplained regression against the existing handwritten parser.
The Qiskit README's factor of eighty is context, not an acceptance target. Every
benchmark must identify the stage measured so parser time is not conflated with
semantic analysis or MLIR construction.

Final replacement acceptance additionally requires all established OpenQASM 2
regressions, representative OpenQASM 3 conformance programs, differential QC
equivalence, and full module verification. The current replacement has parity
over all 117 established translation fixtures; broader conformance remains a
later milestone.

## Idempotence and Recovery

Builds, tests, benchmarks, and lint are repeatable and write only ignored build
or temporary output. The vendored grammar snapshot is updated only through its
documented script and exact upstream revision; review its hashes and complete
diff after regeneration.

Parser migration is additive until parity. Keep adapters from old entry points
to the new syntax and semantic stages, then remove old ownership and passes only
after both consumers pass. If a milestone fails, retain the existing importer
and remove only the incomplete adapter. Do not restore the removed ANTLR files;
they remain recoverable from Git history for isolated comparison.

Before rebasing, require a clean task checkout, fetch `origin`, and rebase onto
`origin/main`. Preserve current mainline behavior in conflicts. Never reset or
clean another worktree, and never force-push without authorization.

## Artifacts and Notes

The inspected Qiskit snapshot divides its frontend into lexer, parser, syntax,
source-file, and semantics crates. Its parser creates a flat event stream before
building a source-spanned syntax tree; its semantic context owns an abstract
semantic graph, scoped symbol table, constant map, and semantic-error list. This
plan adopts the separation and measurable stage boundaries, not the Rust crate
layout or implementation.

Daniel's review questions are answered as follows. Source semantic analysis
belongs in typed-program construction and uses original source spans. Local OQ3
verifiers check operation invariants. One whole-program verifier checks the few
cross-operation invariants that cannot be local. OQ3 emission is a separate
linear walk and may use `arith`, `memref`, and SCF because source typing has
already completed. There is no repeated source semantic walk.

No public GitHub comment or review is authorized by this plan alone. The draft
pull request may receive branch updates within the user's previously authorized
progress-tracking scope. Any agent-authored public text body must begin with
`🤖 *AI text below* 🤖`.

## Interfaces and Dependencies

The source frontend must provide value-oriented result types equivalent to:

    struct SourceSpan {
      SourceId source;
      std::uint32_t begin;
      std::uint32_t end;
    };

    struct ParseResult {
      std::optional<SyntaxProgram> program;
      std::vector<Diagnostic> diagnostics;
    };

    struct AnalysisResult {
      std::optional<TypedProgram> program;
      std::vector<Diagnostic> diagnostics;
    };

    ParseResult parseOpenQASM(SourceManager&, const ParseOptions& = {});
    AnalysisResult analyzeOpenQASM(const SyntaxProgram&,
                                   const AnalysisOptions& = {});

`SyntaxProgram` and `TypedProgram` own nodes in arenas and expose stable IDs;
they do not use shared pointers for tree ownership. Diagnostics are collected as
data and rendered by the caller. The parser and analyzer do not depend on MLIR.

The MLIR adapter must provide:

    OwningOpRef<ModuleOp> emitOQ3(const qasm3::TypedProgram&, MLIRContext&,
                                  const OpenQASMEmissionOptions& = {});

    OwningOpRef<ModuleOp>
    translateOpenQASMToOQ3(qasm3::SourceManager&, MLIRContext&,
                           const OpenQASMTranslationOptions& = {});

The OQ3 lowering interface remains
`createLowerOQ3ToQCPass(OpenQASMLoweringOptions)`. A module-level verifier pass
is added for cross-operation invariants. The adapter depends on the source
frontend, MLIR builtin IR, `arith`, `math`, `func`, `scf`, `memref`, OQ3, and
QC. The source frontend has no ANTLR, Java, Rust, or MLIR dependency.

One canonical gate catalog for the new MLIR path must describe name, parameter
count, qubit count, availability policy, primitive QC operation, implicit
controls, and special lowering. `cu` records three U parameters plus one control
phase; `cu3` records one control around U; `cu1` records one control around P.
Semantic lookup, OQ3 declarations, and lowering dispatch consume this catalog.
The legacy importer may retain its independent table; differential tests guard
their intentional overlap without coupling the implementations.

Revision note (2026-07-15): Replaced the ANTLR-based plan after maintainer
feedback and source-level comparison with MQT Core and Qiskit's parsers. This
revision removes the demonstrator, makes the existing handwritten frontend the
foundation, defines strict parse/analyze/emit boundaries, preserves original
source spans, limits MLIR verification to defensive IR checks, establishes an
explicit gate-compatibility policy, and requires native `cu`, `cu3`, and `cu1`
lowering plus reproducible stage-specific performance evidence.

Revision note (2026-07-15): Made the next iteration intentionally more
ambitious. The staged frontend now replaces the existing production MLIR
OpenQASM-to-QC visitor as soon as its current regression suite passes. Scanner
and parser code may be shared with the legacy importer, but the MLIR typed model
and semantic analysis are independently owned; duplication is allowed to avoid
coupling the new architecture to legacy `QuantumComputation` constraints.

Revision note (2026-07-15): Completed the immediate production replacement. The
new MLIR frontend has explicit parse, analyze, emit, and lower boundaries; the
direct AST-to-QC visitor is deleted; a canonical MLIR gate catalog drives
semantic lookup and lowering; and the existing 117-program oracle passes through
the staged implementation. Compatibility mode is the default so replacing the
architecture does not also remove established native-gate convenience.
