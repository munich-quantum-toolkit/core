# Normalize compiler-wide global phases without changing quantum semantics

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core synthesis and canonicalization passes often emit one `qc.gphase` or
`qco.gphase` operation for each local rewrite so that the rewritten circuit has
the same complete unitary matrix, rather than merely the same matrix up to a
global scalar. Large synthesized programs therefore accumulate many zero-qubit
operations that all represent factors of the same scalar phase.

After this change, users can run the `normalize-global-phases` MLIR pass, the
typed C++ program method, or the corresponding Python method to reduce those
operations to at most one direct global phase per basic block. Standard QC and
QCO cleanup runs this normalization automatically. The implementation moves a
phase out of an inverse or an integral power modifier and converts a phase
inside a control modifier into the correct relative phase on the controls.
Function bodies, control-flow graph blocks, structured-control-flow regions,
and unknown regions remain independent scopes. Focused full-matrix tests,
including tests under an extra outer control, demonstrate that no rewrite hides
an incorrect phase behind an equivalence-up-to-phase comparison.

## Progress

- [x] (2026-08-01 10:42Z) Refresh `origin/main`, allocate the isolated task
  worktree, verify issue labels, and read repository policy.
- [x] (2026-08-01 10:42Z) Record the semantic contract and staged implementation
  in this ExecPlan.
- [x] (2026-08-01) Repair existing principal-power canonicalizations and add
  direct and outer-controlled full-matrix regressions for phase-producing power
  rewrites.
- [x] (2026-08-01) Add the shared QC/QCO global-phase normalization engine and
  its textual pass.
- [x] (2026-08-01) Factor normalized phases through inverse, integral power,
  and control modifiers with conservative SSA-slice hoisting.
- [x] (2026-08-01) Integrate normalization into cleanup, typed C++ APIs,
  generated Python bindings, conversion passes, and phase-producing synthesis
  passes.
- [x] (2026-08-01) Add exact-unitary modifier tests, scope-boundary and
  idempotence tests, conversion diagnostics, synthesis invariants, an
  OpenQASM-to-Jeff/QIR case, and 1k/10k/100k scaling coverage.
- [x] (2026-08-01) Run focused tests, the complete 4,297-case configured CTest
  suite, the release build, generated-stub and Python binding checks,
  repository lint, and final diff/status audits.
- [x] (2026-08-01) Refresh `origin/main`, inspect its two non-overlapping
  commits, rebase the three implementation stages, and repeat affected
  validation before handoff.

## Surprises & Discoveries

- Observation: current `PowOp` uses eigendecomposition and raises eigenvalues on
  the principal complex branch, but the existing `pow(r) { x }` general fold
  emits `gphase(-r*pi/2); rx(r*pi)`. For `r = 1/3`, the maximum entrywise error
  from principal `X^(1/3)` is approximately `0.866`, while changing the phase
  sign to positive reduces the error to floating-point noise. Existing
  measurement-oriented module-equivalence tests do not observe this global
  phase error.
- Observation: extracting a phase through a fractional matrix power is not
  generally valid. For a diagonal example, independently applying the principal
  power to `exp(i*phi) V` and factoring `exp(i*p*phi)` from `V^p` differs by
  magnitude `2.0` across a branch cut. Integral powers remain exact.
- Observation: QIR cannot represent a global phase nested in a control. The
  conversion now runs normalization itself: a hoistable phase becomes the
  exact relative `P` operation on the controls, while a phase whose angle
  depends on a non-speculatable call remains nested and produces the existing
  `Controlled GPhaseOps cannot be converted to QIR` diagnostic.
- Observation: the QCO DD fallback used by the exact-unitary oracle accepts
  full-width unitaries only in canonical qubit order. Reordered-control SSA
  threading is therefore verified structurally, while canonical control orders
  with zero through three controls are checked by complete matrix equality.
- Observation: release-mode normalization took 79,750 ns, 624,667 ns, and
  6,124,500 ns for 1,000, 10,000, and 100,000 phase contributions. The observed
  growth factors of approximately 7.8 and 9.8 for tenfold input growth are
  consistent with the intended linear traversal.
- Observation: generated stubs initially selected an unrelated MLIR and failed
  configuration. Selecting the repository's LLVM/MLIR 22.1.3 installation
  produced the wheel and completed the Nox `stubs` session successfully without
  source workarounds.
- Observation: the final upstream refresh advanced `origin/main` from
  `e772dba5c` to `a5757fe95` by two commits confined to DD serialization and ZX
  decomposition. They did not overlap the compiler changes and rebased
  cleanly.
- Observation: the complete configured CTest suite passed all 4,297 cases in
  39.62 seconds; two environment-dependent QDMI job-ID tests were reported by
  CTest as skipped. The all-files Nox lint session passed every hook, and the
  built Python 3.13 binding test passed.

## Decision Log

- Decision: preserve `qc.gphase` and `qco.gphase` as the canonical materialized
  representation and place at most one directly in each basic block.
  Rationale: this avoids a new phase-token dataflow system and retains all
  existing translations and lowerings. Date/Author: 2026-08-01, Codex.
- Decision: define each basic block as a normalization scope and never transport
  phase state through function signatures, CFG successors, or SCF
  arguments/results in this change. Rationale: one block-local traversal is
  linear, dominance-safe for runtime angles, and conservative at control-flow
  joins. Date/Author: 2026-08-01, Codex.
- Decision: factor through inverse unconditionally, through power only for a
  finite exactly integral compile-time exponent, and through control as a
  relative phase on the control register. Rationale: these are exact operator
  identities under MQT's current matrix semantics; fractional power
  factorization is not. Date/Author: 2026-08-01, Codex.
- Decision: compare rewritten unitary matrices directly and place
  phase-producing rewrites under an additional control in semantic tests.
  Rationale: probability-based tests and comparisons modulo global phase cannot
  detect the errors that become relative phases under control. Date/Author:
  2026-08-01, Codex.
- Decision: implement one shared traversal with small QC/QCO adapters instead of
  patterns that repeatedly scan enclosing modules. Rationale: this gives
  deterministic linear work and one place to enforce boundary and numerical
  rules. Date/Author: 2026-08-01, Codex.
- Decision: run normalization inside the QC-to-QIR Base/Adaptive and
  QCO-to-Jeff conversion passes, rather than relying only on typed-program
  wrappers. Rationale: textual pass pipelines and direct conversion users must
  receive the same sound boundary behavior without a duplicate traversal in
  the typed API. Date/Author: 2026-08-01, Codex.
- Decision: reduce each finite literal modulo `2*pi` while accumulating
  constants. Rationale: the phase identity permits this and it prevents two
  individually finite literals from overflowing their C++ accumulator before
  the final normalization. Dynamic additions remain unreassociated and in
  encounter order. Date/Author: 2026-08-01, Codex.
- Decision: return numeric phase contributions directly from Euler and Weyl
  synthesis, while dynamic optimization contributions remain canonical
  `qco.gphase` operations until the one post-pass normalization traversal.
  Rationale: all current synthesis corrections are compile-time doubles; an
  exposed double-or-SSA wrapper would add an unused abstraction. The shared
  normalizer already provides the common ordered accumulator for both numeric
  and SSA-valued contributions. Date/Author: 2026-08-01, Codex.

## Outcomes & Retrospective

The semantic repairs, shared transform, APIs, conversion integration, synthesis
refactor, generated stubs, and acceptance tests are implemented.
Complete QC and QCO matrices are compared entry-by-entry, including
phase-producing power folds and controlled-phase extraction under another
control. Euler/Weyl and native synthesis retain full-unitary equivalence while
emitting at most one direct phase per block. The release build, all 4,297
configured CTest cases, generated stubs, the built Python 3.13 binding test,
all-files repository lint, `git diff --check`, and focused linear-scaling
measurement passed. The three local commits are rebased onto `a5757fe95`.
No remote branch, issue, or pull-request state has been changed.

## Context and Orientation

The QC dialect represents quantum operations on reference-like qubit values.
Its operation definitions are in
`mlir/include/mlir/Dialect/QC/IR/QCOps.td`, with modifier implementations under
`mlir/lib/Dialect/QC/IR/Modifiers/`. QCO represents explicit linear quantum
dataflow: each operation consumes input qubit values and returns their
successors. Its corresponding definitions and modifier implementations are
under `mlir/include/mlir/Dialect/QCO/IR/` and
`mlir/lib/Dialect/QCO/IR/Modifiers/`.

Both dialects have a zero-target `GPhaseOp` with one `f64` SSA operand. A
`GPhaseOp(theta)` denotes multiplication of the complete quantum state by
`exp(i*theta)`. It has no qubit operand through which ordinary SSA analyses can
connect separate occurrences, so generic canonicalization does not combine
them.

A modifier is a region-owning quantum operation. `ctrl` applies the region body
only when every control is one, `inv` applies the adjoint of the composed body,
and `pow` applies a real matrix power. QCO's
`PowOp::getUnitaryMatrix()` explicitly uses principal-branch complex powers.
The exact identities needed here are:

    inv(exp(i*phi) V) = exp(-i*phi) inv(V)

    pow(n, exp(i*phi) V) = exp(i*n*phi) pow(n, V), for integer n

    ctrl_k(exp(i*phi) V)
      = relative_phase_on_all_k_controls(phi) ctrl_k(V)

For one control, the relative phase is `P(phi)`. For multiple controls, it is a
smaller controlled `P(phi)` whose target is the last original control and whose
controls are the remaining original controls. This factor commutes with the
controlled body because both are block diagonal in the control basis.

The cleanup pipelines are assembled in
`mlir/lib/Support/Passes.cpp`; typed program methods live in
`mlir/include/mlir/Compiler/Programs.h` and
`mlir/lib/Compiler/Programs.cpp`; Python bindings live in
`bindings/mlir/register_mlir.cpp`. The generated
`python/mqt/core/mlir.pyi` file must be regenerated, never edited manually.

Synthesis helpers under `mlir/lib/Dialect/QCO/Transforms/Decomposition/` and
optimization passes under `mlir/lib/Dialect/QCO/Transforms/Optimizations/`
currently materialize phase operations immediately. They will instead return
or accumulate a `double` or SSA-value phase contribution and materialize it
once per completed pass scope.

This work must remain inside its assigned worktree. It must not modify another
task's worktree. `AGENTS.md` and `docs/ai_usage.md` govern validation, generated
files, AI disclosure, and external actions. This ExecPlan authorizes no push,
pull request, issue comment, or other GitHub mutation.

## Plan of Work

First, repair the existing phase-producing power canonicalizations in both
dialects. Fixed-spectrum gates may retain real-exponent closed forms only when
the formula follows directly from their principal eigenphases. In particular,
use positive `r*pi/2` global phase for `X` and `Y`, positive `r*pi/4` for `SX`,
and negative `r*pi/4` for `SXdg`. Keep the existing exact `P`-family and iSWAP
spectral forms. Restrict folds that merely scale a runtime or arbitrary gate
parameter (`gphase`, rotations, `P`, `R`, and parameterized two-qubit gates) to
finite exactly integral exponents. Add QCO matrix tests that compare the
original `PowOp` matrix directly with the canonical replacement and repeat each
phase-producing case under an outer control. Mirror structural/canonicalization
coverage for QC.

Next, add a common transform library and a module pass named
`normalize-global-phases`. The traversal must recurse through child regions
first, specially factor modifier-body phases when legal, then normalize direct
phase operations in each block. It must collect contributions in textual order,
sum all finite constants in C++, build nonconstant additions as an ordered
`arith.addf` chain without fast-math, normalize the final finite constant modulo
`2*pi`, and emit one phase immediately before the block terminator. It must not
cancel symbolic opposites or fold non-finite constants into an apparent zero.
Mixed QC/QCO blocks are outside normal typed-program operation; handle each
dialect independently rather than choosing an arbitrary replacement type.

When factoring a modifier, first locate the one normalized direct phase at the
body exit. If its angle is defined outside the modifier, reuse it. If defined
inside, collect its backward SSA slice and move that slice only when every
operation is speculatable, has no memory effects, and has no block-argument
dependency. Otherwise leave the phase in place. Remove the phase and emit the
negated or integer-scaled phase after inverse or power. For QCO control, remove
the body phase, keep the original control for the remaining body, create `P` or
a smaller `CtrlOp` on the original control outputs, and replace downstream uses
of those control results while leaving target results unchanged. Implement the
same operator order in QC. A zero-control modifier leaves the phase global.

Register the pass and expose `normalizeGlobalPhases()` on both typed C++
programs plus `normalize_global_phases()` in Python. Add it after generic
canonicalization in QC and QCO cleanup, and immediately before target
conversions that cannot represent a controlled global phase. Regenerate Python
stubs with the repository Nox session.

Refactor phase-producing synthesis code only after the normalizer is proven.
Introduce a small `GlobalPhaseContribution` representation that contains
either a constant `double` or an existing `f64` SSA value. Euler and Weyl
synthesis return their correction with their synthesized outputs; native
two-qubit synthesis accumulates all constant corrections before materializing
one operation. Rotation merging and Hadamard lifting send their contributions
to the shared block accumulator. A standalone phase-producing pass invokes
normalization once after its rewrite traversal; a compound native pipeline does
so once after all internal stages.

Finally add end-to-end and scaling coverage. Exercise OpenQASM-to-QC,
QC-to-QCO, QCO-to-QC, Jeff, and QIR paths with modifier-contained phases and
structured control flow. Add a test or benchmark helper that creates 1,000,
10,000, and 100,000 phase contributions and records phase count and pass time.
The implementation is acceptable only if the phase count becomes the number of
nonzero block scopes and observed runtime growth remains linear.

## Concrete Steps

Run all commands from the repository root.

Inspect and iterate with the focused source and test searches:

    rg -n "GPhaseOp|PowOp|populateQ.CleanupPipeline" mlir

Configure a worktree-local release build:

    ./.agent/run.sh cmake --preset release

Build the focused dialect and compiler tests as their exact generated targets
become known:

    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-mlir-unittests-qco-ir \
      mqt-core-mlir-unittests-qc-ir \
      mqt-core-mlir-unittests-compiler

Run the corresponding binaries with GoogleTest filters while iterating, then
run the complete affected binaries without filters. Use
`./.agent/run.sh ctest --preset release` for the final configured C++ suite.

Regenerate binding stubs after changing `bindings/mlir/register_mlir.cpp`:

    ./.agent/run.sh uvx nox -s stubs

Run repository lint last:

    ./.agent/run.sh uvx nox -s lint

Record exact target names, test counts, timings, and any environment failures in
`Progress`, `Surprises & Discoveries`, and `Outcomes & Retrospective`.

## Validation and Acceptance

The `normalize-global-phases` pass succeeds when a block containing multiple
direct QC or QCO phases contains zero phases if the exact normalized sum is
zero, otherwise one phase immediately before its terminator. Running the pass a
second time must not change the printed module.

Modifier tests must compare complete matrices entry by entry within the
repository's numerical tolerance. Cover zero through three controls, composite
and multi-target bodies, inverse, integral powers `-3`, `-1`, `0`, `1`, `2`,
and `3`, fractional and dynamic power boundaries, nested modifier orders,
runtime angles, non-hoistable angle definitions, and reordered QCO operands.
Every phase-producing identity must also be tested under one additional
control. The previous negative-sign `pow(1/3) { X }` reference must fail before
the repair and pass afterward.

Boundary tests must show independent phases in separate functions, CFG blocks,
`qco.if` and `qco.index_switch` branches, and SCF loop/conditional regions.
They must show no new phase argument, result, or yield threading.

Synthesis tests must compare complete original and synthesized matrices in each
supported Euler basis and native gate menu. Their IR assertions must show that
phase operations scale with block scopes, not synthesized gates. Conversion
tests must demonstrate successful factorable controlled-phase lowering and a
precise failure for a deliberately non-hoistable controlled phase.

The final branch must pass focused modifier, synthesis, conversion, and compiler
tests, the affected complete test binaries, the release build, stub generation,
`./.agent/run.sh uvx nox -s lint`, `git diff --check`, and a clean review of
`git status --short`. Any unavailable broader check must be reported with its
exact failure and must not be weakened.

## Idempotence and Recovery

Source edits and tests are repeatable. The normalization pass itself must be
idempotent. CMake configuration, builds, tests, stub generation, and lint may be
rerun through `.agent/run.sh`; their caches remain local to this worktree.

If a semantic test exposes an invalid planned identity, retain the original
modifier boundary, record the counterexample here, and do not weaken the exact
matrix oracle. If generated stubs differ unexpectedly, regenerate them from the
binding source rather than editing them. Preserve unrelated files and never
reset, clean, or modify another worktree.

## Artifacts and Notes

The initial analytic counterexample is:

    max_entry_error(principal_pow(X, 1/3),
                    gphase(-pi/6) * rx(pi/3)) = 0.8660254037844386

    max_entry_error(principal_pow(X, 1/3),
                    gphase(+pi/6) * rx(pi/3)) = 1.1102230246251565e-16

This is an acceptance fixture, not merely design motivation.

## Interfaces and Dependencies

The final public interfaces are:

    bool mlir::QCProgram::normalizeGlobalPhases();
    bool mlir::QCOProgram::normalizeGlobalPhases();

    QCProgram.normalize_global_phases() -> None
    QCOProgram.normalize_global_phases() -> None

The textual pass argument is `normalize-global-phases` and operates on
`mlir::ModuleOp`. Its required dialects are QC, QCO, and Arith. The shared
implementation must depend only on existing MLIR IR, side-effect,
speculation, and rewrite utilities; it must not introduce a phase-token type,
change `GPhaseOp` syntax, or require Jeff/QIR to duplicate normalization logic.

Revision note: created on 2026-08-01 to turn the approved compiler-wide phase
normalization design into a self-contained staged implementation and validation
record.
