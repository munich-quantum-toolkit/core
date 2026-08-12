# Add OpenQASM-native fixed-width angle semantics

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's typed OpenQASM frontend recognizes the word `angle`, but it currently
treats angle expressions as ordinary `f64` values and rejects angle
declarations. This loses the fixed-width, modulo-2-pi semantics required by
OpenQASM 3.1 and prevents users from stating the precision of gate parameters.
After this work, a user can declare `angle[8] theta`, manipulate that value with
the language's unsigned-ring rules, compile it through QC and QCO, lower
nullary/specialized programs to QIR, or emit standards-compliant OpenQASM again.
A separate opt-in QCO pass will let a compiler client impose the same bit
precision on all gate parameters, including parameters supplied at runtime.

The implementation deliberately uses builtin MLIR integer, arithmetic, math,
memory, and structured-control operations. It does not add an MQT angle type,
use the MLIR Quant dialect, change QC or QCO gate operand types, change the QIR
ABI, or alter the default compiler pipeline. QC and QCO gates continue to
receive radians as `f64`; integer angle values are converted to radians only at
that boundary.

## Progress

- [x] (2026-08-09 22:54Z) Refreshed `origin/main`, created and verified the
  isolated task worktree, and read the workspace, repository, AI-use, skill,
  remediation, and ExecPlan policies.
- [x] (2026-08-09 22:54Z) Revalidated the approved design against the current
      typed frontend, direct QC emitter, QC-to-OpenQASM exporter, QCO pass API,
      and QIR lowering boundaries.
- [x] (2026-08-09 23:43Z) Added width-aware typed OpenQASM angle declarations,
      expressions, casts, constant semantics, and initial focused frontend
      tests.
- [x] (2026-08-09 23:43Z) Added shared builtin-MLIR conversion helpers and
      lowered angle storage, structured state, arithmetic, and gate boundaries
      to `iN` plus canonical `arith`/`math` bridges.
- [x] (2026-08-09 23:43Z) Extended QC-to-OpenQASM emission with scalar interface
      metadata and canonical angle conversion recognition; focused dynamic input
      and runtime-cast outputs reparse successfully.
- [x] (2026-08-09 23:43Z) Added the opt-in QCO gate-angle quantization pass and
      its textual, C++, and Python binding interfaces. Focused constant,
      dynamic, nested-power, idempotence, and invalid-width tests pass.
- [x] (2026-08-10 00:22Z) Reconciled the complete frontend/exporter suite and
  added semantic, QCO-boundary, compiler, round-trip, and Base/Adaptive QIR
  coverage. After the final semantic audit, the OpenQASM, compiler, and QCO
  optimization suites pass 179/179, 232/232, and 97/97 respectively.
- [x] (2026-08-10 00:22Z) Updated the OpenQASM support matrix, angle design
  rationale, C++/Python API documentation, and changelog; regenerated Python
  stubs and verified the focused Python binding test.
- [x] (2026-08-10 00:22Z) Built generated pass documentation, completed the
  warning-as-error Sphinx session, repeated the affected tests after the
  semantic audit, and obtained a clean repository-wide lint run.
- [x] (2026-08-10 01:31Z) Inspected the complete diff and completed a fresh
      independent verification. The reviewer found one non-angle-result exporter
      provenance defect; it was fixed, covered by semantic reparse assertions,
      and accepted in a focused follow-up review with no remaining findings.
- [x] (2026-08-10 13:31Z) Recorded the initial implementation as signed commit
      `f15098d3b`, refreshed `origin/main` to
      `d080fd8551dc907ac9282c096d95ceb814a7792c`, and integrated it in signed
      merge commit `28b210852`. The merge retained both the angle interface
      metadata and main's register-name provenance.
- [x] (2026-08-10 14:44Z) Remediated the selected review findings MF-01 through
      MF-04: stable runtime remainder normalization, official integer-literal
      angle promotion, scalar `uint`/`angle` population count and rotations, and
      recursive preflight rejection of foldable non-finite QCO parameters. Added
      semantic, MLIR, exporter-reparse, QCO, and Base/Adaptive QIR regressions.
- [x] (2026-08-10 14:46Z) Rebuilt the affected targets and passed the complete
      OpenQASM (182/182), QCO optimization (99/99), compiler (232/232), Base QIR
      (121/121), and Adaptive QIR (145/145) suites. Repository-wide lint and
      `git diff --check` pass after the hook-prescribed formatting fixed point.
- [x] (2026-08-10 14:54Z) Reconfirmed the warning-as-error documentation build
      with the documentation environment's CA bundle, refreshed live main with
      no further drift, and completed a fresh read-only independent verification
      of exact remediation commit `e3d753b98`. The verifier independently reran
      every affected suite and reported no actionable findings.
- [x] (2026-08-10 15:14Z) Published the reviewed branch, opened PR `#2040`,
      assigned `@burgholzer`, applied the existing feature/language/subsystem
      labels, and folded the new PR reference into the generic staged OpenQASM
      changelog entry in a signed follow-up commit.
- [x] (2026-08-10 19:08Z) Reproduced the initial PR CI failures on exact head
      `d3d0e5b6f`: ten LLVM 22 clang-tidy diagnostics and seven common
      OpenQASM/QC translation failures across the C++ platform matrix. Fixed the
      changed-source diagnostics, admitted the canonical angle bridge inside
      composite modifiers, and aligned legacy tests with retained scalar
      interface metadata and modulo-angle semantics.
- [x] (2026-08-10 19:08Z) Passed exact LLVM 22 clang-tidy for both changed
      translation sources, all 290 QC translation tests, a complete Release
      build, and repository-wide CTest (4627/4627, with two expected skips).
- [x] (2026-08-10 20:45Z) Refreshed live main to
      `0901b1bb6d5f17c929eb7bf8de7aaebe7ab321e3`, integrated it in signed merge
      commit `1137aaf58`, rebuilt the complete Release tree, and passed the
      expanded CTest suite (4629/4629, with two expected skips), exact LLVM 22
      clang-tidy, repository-wide lint, and `git diff --check` on the resulting
      publishable head.
- [x] (2026-08-12) Remediated the post-rebase conformance findings: exact
      binary64-to-angle conversion through 64 bits, lossless custom-gate
      parameters, exact repeated quantization, compound division assignment,
      mixed-width bitwise promotion, and context-correct bit-pattern export.
- [x] (2026-08-12) Restored QIR 2.1's nullary `i64` entry-point contract, added
      diagnostics for unresolved OpenQASM inputs and unsupported scalar output
      interfaces, corrected QIR capability flag widths at the LLVM translation
      boundary, and derived the Adaptive integer/float computation metadata.
- [x] (2026-08-12) Removed only proven redundancies, replaced fragile per-gate
      quantization sidecars with one validated terminal module marker, shared
      repeated dynamic conversions within a block, and completed a fresh
      independent review with no remaining angle-spec or round-trip finding.
- [x] (2026-08-12) Rebuilt the complete Release tree and passed 4731/4731 CTest
      cases with two expected QDMI skips, exact changed-line LLVM 22 clang-tidy,
      repository lint, generated MLIR documentation, warning-as-error Sphinx,
      and `git diff --check`. No commit, push, or stack mutation was performed.

## Surprises & Discoveries

- Observation: The current typed frontend already assigns `ScalarType::Angle` to
  gate parameters and the constants `pi` and `tau`, but the semantic layer
  permits angle-to-float conversion and the QC emitter emits every angle as
  `f64`. Evidence: `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` and
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` group `Angle` with
  `Float` in conversion and arithmetic branches.
- Observation: The current exporter intentionally reconstructs unsigned and bit
  distinctions from canonical MLIR operations rather than retaining a parallel
  source AST. Angle export should extend this existing provenance approach and
  keep translation buffered so unsupported shapes cannot produce partial text.
- Observation: `pi`, `tau`, and `euler` are `float[64]` constants in the
  normative OpenQASM type table; treating `pi` and `tau` as source angles would
  bypass the specified float-to-angle cast. Evidence: the OpenQASM 3.1 type and
  classical-expression documentation. The semantic model now records them as
  `Float`.
- Superseded observation: The first runtime float-to-angle conversion used an
  `i(N+1)` floating-to-integer intermediate so a rounded `2^N` could wrap to
  zero. Exact conversion through 64 bits now decomposes binary64 into integer
  fields, evaluates the quotient in `i128`, and truncates the modular result.
- Observation: Existing frontend tests counted all floating-point conversion and
  `floor` operations. Canonical angle normalization adds legitimate bridge
  operations, so those assertions must identify the user operation by dataflow
  rather than count every operation of the same class.
- Superseded observation: Forwarding converted function arguments through the
  replacement QIR entry block produced valid LLVM but violated the profile ABI.
  Both QIR 2.1 profiles require nullary entry points, so unresolved source
  inputs are now diagnosed before lowering.
- Observation: OpenQASM permits differing-width angle operands to follow
  unsigned-integer promotion, but only defines angle addition and subtraction
  between angles. The final semantic audit removed an over-permissive fallback
  that had accepted angle-plus-float/integer expressions.
- Observation: An explicit source cast cannot be validated by the implicit
  conversion matrix first. Separating those paths was necessary for the exact
  same-width `bit[N]` to `angle[N]` bridge required by OpenQASM.
- Observation: Reverse translation must retain unsigned intent for constants
  with their high bit set and must distinguish raw angle bit-pattern arithmetic
  from ordinary unsigned expressions. The final exporter wraps angle sources as
  same-width `bit`/`uint` values and reconstructs `angle[N]` only at angle uses.
- Observation: Dynamic indices narrower than the frontend's 64-bit index type
  need explicit unsigned extension before bounds checks. This avoids malformed
  mixed-width comparisons while retaining the source width everywhere else.
- Observation: A failing module pass must not leave earlier gates rewritten. The
  quantization pass therefore validates every constant parameter before it
  mutates the module, and its rewrite walk skips parameterless gates explicitly.
- Observation: Angle provenance is also required when an MLIR operation's result
  is not itself an angle. An independent review exposed comparisons and
  angle/angle division printing a tagged angle constant as an ordinary numeric
  literal. Source-operand emission now reconstructs every tagged angle operand,
  while the raw fallback retains a recursion escape for internal bit patterns.
- Superseded observation: Both quotient/subtraction and `arith.remf` avoid some
  large-input failures but still cannot choose exact buckets above 53 angle
  bits. The canonical bridge now uses exact integer quotient/remainder
  arithmetic relative to the binary64 representation of `2*pi`.
- Observation: The OpenQASM scalar builtins `popcount`, `rotl`, and `rotr` are
  not restricted to whole bit registers. Supporting their `uint[N]` and
  `angle[N]` forms maps directly to `math.ctpop` and LLVM funnel shifts, while
  preserving the existing register-specific lowering.
- Observation: Main now provides `utils::valueToConstantDouble`, which folds a
  pure scalar SSA expression DAG. Reusing it keeps static angle quantization and
  the QCO pass's failure-atomic non-finite preflight aligned.
- Observation: The exact runtime float-to-angle bridge contains guarded selects
  for binary64 decomposition and bounded `i128` shifts. Export treats the marked
  canonical dataflow as one semantic cast rather than exposing those internal
  operations as standalone OpenQASM expressions.
- Observation: Existing translation matrix helpers evaluated only constants and
  floating arithmetic. Evaluating the canonical `arith.uitofp` of an integer
  angle constant as unsigned is required to test high-bit-set `angle[64]` values
  without misinterpreting their signless MLIR spelling.
- Observation: The OpenQASM bucket selection cannot be implemented by a binary64
  divide/multiply/round pipeline above 53 significant bits. For example, exact
  evaluation of `angle[64](1.0)` produces `0x28be60db939105bd`, while the
  current self-oracled implementation produces `0x28be60db93910600`. Constant
  and runtime conversion need an integer decomposition of the source binary64
  value and the binary64 value of `2*pi`.
- Observation: Passing a custom-gate angle formal through radians changes its
  bit pattern before the expanded gate body uses it. Custom-gate formals must
  remain machine-width angle bits; only primitive QC/QCO gate operands cross the
  radians bridge.
- Observation: QIR 2.1 Base and Adaptive entry points take no parameters and
  return an `i64` exit code. The earlier argument-forwarding change made valid
  LLVM IR but invalid profile QIR. CUDA-Q specializes supplied invocation
  arguments before profile lowering, while the Q# compiler partially evaluates a
  concrete call into a nullary entry point. QIR defines output recording but no
  corresponding standard input ABI.
- Observation: OpenQASM scalar outputs are not ordinary QIR function results.
  Base Profile output is restricted to recorded quantum results; Adaptive
  Profile has optional recording functions for supported Boolean, signed
  integer, and floating values. In particular, no standard recording function
  preserves the full unsigned `uint[64]` or `angle[64]` domain as one atomic
  value.
- Observation: MLIR 22 translates every integer-valued `llvm.module_flags`
  attribute as `i32` and supports array-valued flags only for LLVM's CG profile.
  QIR 2.1 instead requires `i1` booleans, an `i2` backwards-branching flag, and
  metadata tuples for Adaptive classical precisions. A shared post-translation
  normalizer is therefore required for both `QIRProgram` and `mqt-cc` output.
- Observation: A custom-gate angle formal also has to retain its provenance
  during reverse translation. Composite helper extraction captures the angle bit
  pattern from outside the modifier region; emitting that capture as a generic
  integer expression exposes the internal exact-conversion selects. Angle-aware
  capture emission and direct helper-formal mapping preserve both the outer
  `angle[N](source)` cast and the inner gate's angle parameter.
- Observation: Frozen QIR 2.1 does not permit the `uitofp` and `bitcast`
  instructions required by runtime angle/radian bridges. It also leaves integer
  overflow and value-changing truncation undefined, so profile conversion must
  reject dynamic angle operations whose OpenQASM semantics depend on modular
  wrap or precision-losing resize. Constant bridges fold before QIR emission.
- Observation: Generic QC/QCO cleanup is not provenance-neutral after final
  quantization. Greedy register-shrink and canonicalization paths can fold exact
  angle bridges or rewrite gates without retaining their source width. Running
  full cleanup before a terminal quantization pass, then making marked cleanup a
  no-op, is smaller and more robust than teaching every rewrite about sidecars.

## Decision Log

- Decision: Represent a resolved `angle[N]` as builtin signless MLIR `iN` and
  use unsigned `arith` operations. Rationale: OpenQASM angles are unsigned bit
  rings, while MLIR Quant models scaled lossy conversion between expressed and
  stored types. Builtin integers give the required wraparound semantics and
  already lower through LLVM/QIR. Date/Author: 2026-08-09 / Codex.
- Decision: Support widths from 1 through 64 and resolve unsized angles and gate
  parameters to 64 bits. Rationale: This matches the existing frontend's
  machine-sized scalar and `f64` gate boundary while making unsupported wider
  values explicit. Date/Author: 2026-08-09 / Codex.
- Decision: Use round-to-nearest, ties-to-even for float-to-angle and narrowing
  conversions. Rationale: OpenQASM permits this narrowing policy and prescribes
  it for floating-point casts; a single documented mode avoids target-specific
  pragmas in this issue. Date/Author: 2026-08-09 / Codex.
- Decision: Keep QC and QCO gate parameters and QIR QIS declarations as `f64`.
  Rationale: Changing these public dialect and ABI surfaces would be disruptive;
  a canonical integer-to-radian bridge contains the new representation at the
  semantic boundary. Date/Author: 2026-08-09 / Codex.
- Decision: Expose hardware precision only through an explicit QCO pass and
  program API. Rationale: Targets do not currently provide a standard angle
  precision property, and later synthesis can create new angles. Callers must
  deliberately run the pass after their final synthesis step. Date/Author:
  2026-08-09 / Codex.
- Decision: Keep `ScalarType` unchanged and model same-width `bit[N]` casts as
  unsigned scalar bit patterns with explicit bit-cast provenance. Rationale: the
  typed frontend already separates bit registers from scalar values; this adds
  the angle-required bit-pattern bridge without introducing a second scalar enum
  or changing unrelated bit-register lowering. Date/Author: 2026-08-09 / Codex.
- Decision: Attach `mqt.openqasm.scalar` argument/result dictionary attributes
  and `mqt.openqasm.angle` unit operation provenance. Rationale: ordinary MLIR
  types alone cannot distinguish `angle[N]` from `uint[N]`, while namespaced
  metadata survives the existing QC/QCO conversion boundary without modifying
  either dialect's operation definitions. Date/Author: 2026-08-09 / Codex.
- Decision: Add the real pull-request reference only after publication is
  authorized, then fold it into the existing staged OpenQASM changelog entry
  instead of retaining a separate angle-only entry. Rationale: this preserves
  the curated subsystem-level release narrative and avoids guessing a PR number
  before it exists. Date/Author: 2026-08-10 / Codex.
- Decision: Evaluate every finite binary64-to-angle conversion exactly relative
  to the binary64 `2*pi` constant, using integer quotient/remainder arithmetic
  and ties-to-even rounding. Rationale: this is required for widths 54 through
  64 and supplies an implementation-independent bit-pattern contract for both
  constants and runtime values. Date/Author: 2026-08-11 / Codex.
- Decision: QIR profile conversion retains the existing `i64` status return but
  rejects unresolved entry arguments. Rationale: both QIR 2.1 profiles require
  nullary entry points, and CUDA-Q/Q# production precedents bind or partially
  evaluate source arguments before profile emission rather than defining a
  private QIR input ABI. Date/Author: 2026-08-11 / Codex.
- Decision: This PR diagnoses unsupported scalar outputs instead of returning
  them directly from the QIR entry point. Rationale: direct LLVM returns violate
  both profiles; Base cannot encode these values, and a complete Adaptive output
  policy for unsigned and angle domains is a distinct public-interface design.
  Measurement result recording remains unchanged. Date/Author: 2026-08-11 /
  Codex.
- Decision: Treat gate-angle quantization as a terminal transformation and mark
  its width once on the module. Rationale: canonical conversion dataflow already
  carries the exact bits; suppressing later QC/QCO cleanup avoids a fragile
  per-operation provenance system, while export validates every actual gate
  parameter and QIR deliberately folds constant bridges at its ABI boundary.
  Date/Author: 2026-08-12 / Codex.
- Decision: Prepare four coherent local review layers: exact angle utilities and
  typed frontend semantics; QC import/export and structured-control round-trip;
  opt-in terminal QCO quantization; and QIR 2.1 profile conformance. Rationale:
  they have distinct contracts and test oracles, and the QIR flag/profile repair
  is an orthogonal lower stack layer even though it is necessary for truthful
  QIR claims. Publishing or restacking remains a separately authorized action.
  Date/Author: 2026-08-12 / Codex.

## Outcomes & Retrospective

The post-rebase implementation and conformance remediation are complete in the
local worktree. Exact binary64 conversion, OpenQASM typing and round-trip,
structured carried state, terminal QCO quantization, and both frozen-QIR profile
boundaries now have direct behavioral coverage. The complete Release build and
all 4731 CTest cases pass, with two expected environment-dependent QDMI tests
skipped. The complete affected binaries pass 300 QC translation, 198 OpenQASM,
240 compiler, 114 QCO optimization, 50 dialect utility, 167 QC-to-QCO, 139
QCO-to-QC, two QC/QCO round-trip, 114 QIR IR, 123 Base QIR, and 146 Adaptive QIR
tests. Exact changed-line LLVM 22 clang-tidy, generated MLIR documentation,
warning-as-error Sphinx documentation, repository-wide lint, and
`git diff --check` are clean. A fresh independent exact-tree review found no
remaining actionable angle-spec or semantic round-trip defect and recommends
retaining the exact converter and terminal marker. The QIR 2.1 normalization is
the only remaining stack-splitting opportunity; no remote or history mutation
has been performed during this remediation.

## Context and Orientation

`mlir/include/mlir/Target/OpenQASM/Frontend.h` is the public typed frontend
model. `ScalarExpression` and `ScalarDeclaration` currently carry only the
`ScalarType` enum, and constants use a variant containing Boolean, signed and
unsigned 64-bit integers, or `double`. The implementation will add a resolved
bit width beside the enum and store angle constants as their unsigned bit
pattern.

`mlir/include/mlir/Target/OpenQASM/Detail/OpenQASMParser.h` parses the language.
`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` resolves names and types,
inserts casts, folds constants, and produces `TypedProgram`. It currently
rejects angle declarations and treats angle arithmetic as floating point.

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` consumes
`TypedProgram`. It threads mutable classical values through MLIR SCF region
arguments and results. Angle values must participate in that same mechanism as
`iN`; no separate mutable state representation is needed.

`mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp` performs the
reverse translation. It translates the QC module and supported builtin MLIR
operations to buffered OpenQASM text. It must recognize the canonical angle
conversion sequence and print an OpenQASM cast rather than the low-level
binary64 decomposition and integer quotient/remainder operations.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` declares QCO passes.
`mlir/include/mlir/Compiler/Programs.h` and `mlir/lib/Compiler/Programs.cpp`
provide value-semantic compiler APIs. `bindings/mlir/register_mlir.cpp` exposes
them to Python, and `python/mqt/core/mlir.pyi` is generated by the repository's
stub session rather than edited by hand.

OpenQASM represents `angle[N]` as an unsigned integer bit pattern `k` whose
mathematical angle is `2*pi*k/2^N`. Addition, subtraction, negation, bitwise
operations, and multiplication by `uint[N]` therefore use ordinary wrapping
integer operations. Float-to-angle conversion first reduces modulo `2*pi`, then
selects the nearest bit pattern with ties going to an even least significant
bit. Widening inserts zero least-significant bits; narrowing rounds discarded
least-significant bits with the same tie rule.

## Plan of Work

First, extend the typed model and parser. Add a resolved `bitWidth` field to
scalar expressions and declarations without renaming `ScalarType`. Parse
`angle`, `angle[N]`, and the width-qualified unsigned/bit casts required by the
angle rules. Resolve omitted sizes to 64 and reject zero or widths above 64 with
source diagnostics. Make semantic analysis model angle constants as `uint64_t`
bit patterns and implement the OpenQASM operator/cast matrix instead of
promoting angles to floats. Gate parameters are normalized to machine-width
angles.

Second, add one shared MLIR angle-conversion utility in the neutral MLIR support
library so the OpenQASM importer and QCO transform cannot drift. Decompose a
finite binary64 source into its sign, exponent, and significand, evaluate its
modular quotient relative to the binary64 `2*pi` significand in `i128`, and
round the exact integer quotient to nearest with ties to even before truncating
to `iN`. The reverse builder uses unsigned integer-to-float conversion and
scales by `2*pi/2^N`. Provide matchers for both canonical forms so repeated
passes and the exporter can recognize them. Constant and runtime paths share the
same integer algorithm.

Third, update the direct QC emitter. Angle constants, variables, SCF-carried
state, comparisons, and operations use the resolved `iN` type. Trigonometric
functions and actual gate parameters receive the shared radians bridge. Source
casts use the shared float-to-angle builder or integer width operations.
Existing integer, float, Boolean, bit-register, qubit, safety-budget, and
structured-control behavior remains unchanged.

Fourth, update reverse OpenQASM translation. Support integer widths from 1
through 64 as `uint[N]` where unsigned semantics are proven. Collapse matched
float-to-angle/radians sequences to standards-compliant `angle[N](...)` syntax.
When angle storage provenance is no longer available, emit its unsigned bit
computation and an exact same-width `bit[N]` to `angle[N]` cast at the gate or
external-interface boundary. Preserve angle input and output kinds using a
namespaced dictionary attribute on function arguments and results, and ensure
QC-to-QCO-to-QC conversion keeps that ordinary MLIR metadata.

Fifth, implement `quantize-gate-angles` as a module-level QCO pass with the
required `precision-bits` option. Validate 1 through 64. Walk nested operations,
rewrite parameters exposed by `qco::UnitaryOpInterface`, and explicitly exclude
the exponent operand of `qco.pow` while still visiting its body. Build the same
float-to-angle-to-radians sequence for constant and dynamic values. Recognize an
existing conversion at the requested width so the pass is idempotent.

Finally, expose `QCOProgram::quantizeGateAngles(uint32_t precisionBits)` and
`QCOProgram.quantize_gate_angles(precision_bits=...)`, regenerate stubs, update
the OpenQASM support documentation, MLIR pass documentation, changelog, and
upgrade notes if the typed frontend aggregate change warrants it. Run focused
tests after each stage, then the full required validation and independent
review.

## Concrete Steps

All commands run from the repository root and through `.agent/run.sh` when they
create caches or build artifacts.

Configure and build the release tree after the first compilable implementation:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release

Run the focused OpenQASM, QCO transform, compiler, and QIR tests. Update these
commands with the exact binaries discovered during implementation and record
their pass counts here:

    ./.agent/run.sh ctest --test-dir build/release -R 'OpenQASM|QCO|QIR|Compiler'

Regenerate Python stubs after binding changes and confirm that only the
generated interface change appears:

    ./.agent/run.sh uvx nox -s stubs

Build documentation and finish with repository validation:

    ./.agent/run.sh cmake --build --preset release --target mlir-doc
    ./.agent/run.sh uvx nox --non-interactive -s docs
    git diff --check
    ./.agent/run.sh uvx nox -s lint

Before handoff, refresh `origin/main`, compare it with the recorded base, and
repeat affected validation if it advanced. Do not push or create a pull request
without separate user authorization.

## Validation and Acceptance

Frontend tests must demonstrate `angle[1]`, `angle[8]`, `angle[53]`,
`angle[64]`, unsized angle resolution, modular addition and negation, angle/uint
multiplication and division, angle/angle division to unsigned, unsigned
comparisons, widening, narrowing, bit casts, and diagnostics for zero,
over-wide, unsupported, NaN, and infinity cases. The OpenQASM specification's
halfway example must convert to the unsigned bit pattern 64 at width 8.

QC translation tests must show `iN` values and unsigned `arith` operations in
the emitted module, including angle values carried through `scf.if`, `scf.for`,
and `scf.while`. Gate operands remain `f64`. The emitted IR must contain no
Quant dialect types or operations and no new custom angle operation.

Pass tests must cover constant and runtime parameters, negative and multi-turn
values, all boundary widths, `gphase`, parameterized one- and two-qubit gates,
nested control/inverse/power regions, exclusion of the power exponent, invalid
options, same-precision idempotence, and deliberate composition at different
precisions.

Round-trip tests must parse OpenQASM containing dynamic angle inputs, lower it
through QC and QCO, emit OpenQASM, parse that output again, and observe equal
angle bit patterns and gate parameters. QIR tests must instead use nullary
programs, verify the required `i64` status return and existing `double` QIS
signatures, and prove that both profiles reject unresolved inputs and arbitrary
scalar outputs before emitting non-profile LLVM signatures. Runtime angle
conversion remains valid in generic QC/QCO MLIR even when a particular QIR
profile cannot obtain its value from an entry-point parameter.

Acceptance requires all new focused tests, affected existing suites, generated
stubs, documentation, `git diff --check`, and the full lint session to pass, or
for any environment-only failure to be recorded with exact evidence without
weakening source or checks.

## Idempotence and Recovery

Parser, semantic, emitter, pass, build, test, stub, and documentation commands
are repeatable. The quantization pass itself must be idempotent for a given
precision. Build output remains under `build/` and caches remain under
`.cache/`. If a generated file changes unexpectedly, rerun its documented
generator and inspect the source inputs; never edit generated output by hand.
Preserve all unrelated worktrees and user changes. If `origin/main` advances,
integrate it normally and repeat affected tests rather than resetting or
discarding work.

## Artifacts and Notes

The published implementation history through PR head `068543ef4` records the
initial full-build, focused-suite, documentation, lint, and CI remediation
evidence. The current post-rebase remediation is local on top of that head. The
complete affected suite counts and validation evidence are recorded in
`Outcomes & Retrospective`; the final exact-tree independent review is complete,
and no local commit or separately authorized publication has been performed.
Dynamic OpenQASM inputs are no longer claimed as QIR-profile entry parameters:
both QIR 2.1 profiles require a nullary `i64` entry point, so inputs must be
specialized before QIR conversion and unsupported scalar outputs are diagnosed.

## Interfaces and Dependencies

`ScalarExpression` and `ScalarDeclaration` retain `ScalarType` and gain a
resolved `uint32_t bitWidth`. Angle constants use `uint64_t` in the existing
constant variant.

The QCO pass is registered under the textual name
`quantize-gate-angles{precision-bits=N}`. Its option uses `uint32_t`, treats
zero as missing/invalid, and accepts only 1 through 64.

The public C++ API is:

    [[nodiscard]] bool QCOProgram::quantizeGateAngles(uint32_t precisionBits);

The Python API is:

    QCOProgram.quantize_gate_angles(*, precision_bits: int) -> None

The implementation may depend on MLIR Arith, Math, Func, MemRef, SCF, CF, and
LLVM support libraries already used by the OpenQASM/QCO components. It must not
add the MLIR Quant dialect, change QIR runtime declarations, or introduce a
provider SDK or target property.

Revision note: This initial version records the approved design, current source
orientation, fixed public interfaces, milestones, and acceptance criteria so
implementation can resume from this file alone.
