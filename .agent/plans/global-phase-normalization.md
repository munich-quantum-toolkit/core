# Normalize compiler-wide global phases without changing quantum semantics

Status: historical implementation record.

Later test decisions and remaining questions:
[global-phase audit](../audits/global-phase-normalization.md).

## Goal and scope

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
Function bodies, control-flow graph blocks, structured-control-flow regions, and
unknown regions remain independent scopes. Focused full-matrix tests, including
tests under an extra outer control, demonstrate that no rewrite hides an
incorrect phase behind an equivalence-up-to-phase comparison.

## Constraints

- current `PowOp` uses eigendecomposition and raises eigenvalues on the
  principal complex branch, but the existing `pow(r) { x }` general fold emits
  `gphase(-r*pi/2); rx(r*pi)`. For `r = 1/3`, the maximum entrywise error from
  principal `X^(1/3)` is approximately `0.866`, while changing the phase sign to
  positive reduces the error to floating-point noise. Existing
  measurement-oriented module-equivalence tests do not observe this global phase
  error.

- extracting a phase through a fractional matrix power is not generally valid.
  For a diagonal example, independently applying the principal power to
  `exp(i*phi) V` and factoring `exp(i*p*phi)` from `V^p` differs by magnitude
  `2.0` across a branch cut. Integral powers remain exact.

- QIR cannot represent a global phase nested in a control. The conversion now
  runs normalization itself: a hoistable phase becomes the exact relative `P`
  operation on the controls, while a phase whose angle depends on a
  non-speculatable call remains nested and produces the existing
  `Controlled GPhaseOps cannot be converted to QIR` diagnostic.

- the QCO DD fallback used by the exact-unitary oracle accepts full-width
  unitaries only in canonical qubit order. Reordered-control SSA threading is
  therefore verified structurally, while canonical control orders with zero
  through three controls are checked by complete matrix equality.

- release-mode normalization took 79,750 ns, 624,667 ns, and 6,124,500 ns for
  1,000, 10,000, and 100,000 phase contributions. The observed growth factors of
  approximately 7.8 and 9.8 for tenfold input growth are consistent with the
  intended linear traversal.

- before remediation, release-mode normalization of one dynamic phase under 100,
  200, 400, 800, 1,600, and 3,200 nested integral powers took 827 us, 2,855 us,
  10,259 us, 33,470 us, 130,212 us, and 532,003 us, respectively. Each modifier
  had materialized another SSA operation and the next modifier walked the entire
  growing chain.

- after deferring modifier arithmetic, 128, 256, 512, and 1,024 nested dynamic
  integral powers took 116,666 ns, 188,708 ns, 359,625 ns, and 754,500 ns in the
  focused release test. An eightfold increase in depth caused approximately
  6.5-fold runtime growth.

## Decisions

- preserve `qc.gphase` and `qco.gphase` as the canonical materialized
  representation and place at most one directly in each basic block. Rationale:
  this avoids a new phase-token dataflow system and retains all existing
  translations and lowerings.

- define each basic block as a normalization scope and never transport phase
  state through function signatures, CFG successors, or SCF arguments/results in
  this change. Rationale: one block-local traversal is linear, dominance-safe
  for runtime angles, and conservative at control-flow joins.

- factor through inverse unconditionally, through power only for a finite
  exactly integral compile-time exponent, and through control as a relative
  phase on the control register. Rationale: these are exact operator identities
  under MQT's current matrix semantics; fractional power factorization is not.

- compare rewritten unitary matrices directly and place phase-producing rewrites
  under an additional control in semantic tests. Rationale: probability-based
  tests and comparisons modulo global phase cannot detect the errors that become
  relative phases under control.

- implement one shared traversal with small QC/QCO adapters instead of patterns
  that repeatedly scan enclosing modules. Rationale: this gives deterministic
  linear work and one place to enforce boundary and numerical rules.

- run normalization inside the QC-to-QIR Base/Adaptive and QCO-to-Jeff
  conversion passes, rather than relying only on typed-program wrappers.
  Rationale: textual pass pipelines and direct conversion users must receive the
  same sound boundary behavior without a duplicate traversal in the typed API.

- reduce each finite literal modulo `2*pi` while accumulating constants.
  Rationale: the phase identity permits this and it prevents two individually
  finite literals from overflowing their C++ accumulator before the final
  normalization. Dynamic additions remain unreassociated and in encounter order.

- return numeric phase contributions directly from Euler and Weyl synthesis,
  while dynamic optimization contributions remain canonical `qco.gphase`
  operations until the one post-pass normalization traversal. Rationale: all
  current synthesis corrections are compile-time doubles; an exposed
  double-or-SSA wrapper would add an unused abstraction. The shared normalizer
  already provides the common ordered accumulator for both numeric and
  SSA-valued contributions.

- require constant `qc.gphase` and `qco.gphase` angles to be finite and no
  larger than 10,000 radians in magnitude, with the same documented runtime
  precondition for dynamic values. Rationale: this range is generous for
  compiler workloads while keeping binary64 modulo reduction within the
  exact-unitary test tolerance. A verifier makes malformed constants explicit
  and removes the need for overflow, NaN, and infinity branches throughout the
  normalizer.

- represent a phase being transported through modifiers as a postfix expression
  containing values, constants, ordered additions, negations, and scales.
  Rationale: inverse and power transformations remain exact and retain the
  original dynamic addition grouping, while the expression and its hoistable
  value leaves are traversed linearly and materialized only at the stopping
  scope.

- rely on the compiler invariant that a program uses either QC or QCO, never
  both, and assert this when contributions are combined. Rationale:
  mixed-dialect recovery code would complicate a state that cannot arise in
  supported programs.

## Outcome and validation

The shared transform, program APIs, conversion integration, and synthesis
changes preserve complete QC/QCO unitaries, including controlled phases.
Bounded-angle verification and deferred symbolic materialization avoid full-f64
argument reduction and repeated traversal.

The final recorded release build, configured CTests, hooks, stubs, Python
binding tests, exact-unitary tests, scaling checks, and focused clang-tidy
passed. Later decisions about representation-specific tests and remaining risks
are in the linked audit.

## Code and ownership

The QC dialect represents quantum operations on reference-like qubit values. Its
operation definitions are in `mlir/include/mlir/Dialect/QC/IR/QCOps.td`, with
modifier implementations under `mlir/lib/Dialect/QC/IR/Modifiers/`. QCO
represents explicit linear quantum dataflow: each operation consumes input qubit
values and returns their successors. Its corresponding definitions and modifier
implementations are under `mlir/include/mlir/Dialect/QCO/IR/` and
`mlir/lib/Dialect/QCO/IR/Modifiers/`.

Both dialects have a zero-target `GPhaseOp` with one `f64` SSA operand. A
`GPhaseOp(theta)` denotes multiplication of the complete quantum state by
`exp(i*theta)`. It has no qubit operand through which ordinary SSA analyses can
connect separate occurrences, so generic canonicalization does not combine them.

A modifier is a region-owning quantum operation. `ctrl` applies the region body
only when every control is one, `inv` applies the adjoint of the composed body,
and `pow` applies a real matrix power. QCO's `PowOp::getUnitaryMatrix()`
explicitly uses principal-branch complex powers. The exact identities needed
here are:

    inv(exp(i*phi) V) = exp(-i*phi) inv(V)

    pow(n, exp(i*phi) V) = exp(i*n*phi) pow(n, V), for integer n

    ctrl_k(exp(i*phi) V)
      = relative_phase_on_all_k_controls(phi) ctrl_k(V)

For one control, the relative phase is `P(phi)`. For multiple controls, it is a
smaller controlled `P(phi)` whose target is the last original control and whose
controls are the remaining original controls. This factor commutes with the
controlled body because both are block diagonal in the control basis.

The cleanup pipelines are assembled in `mlir/lib/Support/Passes.cpp`; typed
program methods live in `mlir/include/mlir/Compiler/Programs.h` and
`mlir/lib/Compiler/Programs.cpp`; Python bindings live in
`bindings/mlir/register_mlir.cpp`. The generated `python/mqt/core/mlir.pyi` file
must be regenerated, never edited manually.

Synthesis helpers under `mlir/lib/Dialect/QCO/Transforms/Decomposition/` and
optimization passes under `mlir/lib/Dialect/QCO/Transforms/Optimizations/`
currently materialize phase operations immediately. They will instead return or
accumulate a `double` or SSA-value phase contribution and materialize it once
per completed pass scope.

## Acceptance

The `normalize-global-phases` pass succeeds when a block containing multiple
direct QC or QCO phases contains zero phases if the exact normalized sum is
zero, otherwise one phase immediately before its terminator. Running the pass a
second time must not change the printed module.

Modifier tests must compare complete matrices entry by entry within the
repository's numerical tolerance. Cover zero through three controls, composite
and multi-target bodies, inverse, integral powers `-3`, `-1`, `0`, `1`, `2`, and
`3`, fractional and dynamic power boundaries, nested modifier orders, runtime
angles, non-hoistable angle definitions, and reordered QCO operands. Every
phase-producing identity must also be tested under one additional control. The
previous negative-sign `pow(1/3) { X }` reference must fail before the repair
and pass afterward.

Boundary tests must show independent phases in separate functions, CFG blocks,
`qco.if` and `qco.index_switch` branches, and SCF loop/conditional regions. They
must show no new phase argument, result, or yield threading.

Synthesis tests must compare complete original and synthesized matrices in each
supported Euler basis and native gate menu. Their IR assertions must show that
phase operations scale with block scopes, not synthesized gates. Conversion
tests must demonstrate successful factorable controlled-phase lowering and a
precise failure for a deliberately non-hoistable controlled phase.

The final branch must pass focused modifier, synthesis, conversion, and compiler
tests, the affected complete test binaries, the release build, stub generation,
`uvx nox -s lint`, `git diff --check`, and a clean review of
`git status --short`. Any unavailable broader check must be reported with its
exact failure and must not be weakened.

## Interfaces

The final public interfaces are:

    bool mlir::QCProgram::normalizeGlobalPhases();
    bool mlir::QCOProgram::normalizeGlobalPhases();

    QCProgram.normalize_global_phases() -> None
    QCOProgram.normalize_global_phases() -> None

The textual pass argument is `normalize-global-phases` and operates on
`mlir::ModuleOp`. Its required dialects are QC, QCO, and Arith. The shared
implementation must depend only on existing MLIR IR, side-effect, speculation,
and rewrite utilities; it must not introduce a phase-token type, change
`GPhaseOp` syntax, or require Jeff/QIR to duplicate normalization logic.
