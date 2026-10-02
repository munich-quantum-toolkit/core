# Prove affine OpenQASM quantum indices

Status: historical implementation record.

## Goal and scope

OpenQASM 3 permits an implementation to reject quantum-register indices that are
not compile-time constants. MQT Core currently accepts arbitrary runtime indices
and emits bounds, negative-index, overflow, and operand-distinctness guards.
Those guards prevent simple structured programs from lowering to `jeff`. After
this change, MQT Core accepts a larger-than-required but precisely defined
subset: constants and integer affine expressions whose safety follows from
enclosing positive-step `for` ranges. The frontend rejects every other quantum
index before MLIR emission. A user can compile the loop from issue #2188 to
`jeff`, and benchmark-shaped triangular loops such as QFT retain direct
`scf.for` structure without runtime guard scaffolding.

## Constraints

- The existing constant-range `scf.for` path still reconstructs its source
  induction value with `i128` multiply and add operations. Evidence:
  `OpenQASMToQCEmitter.cpp` creates `counterWide`, `offset`, and `inductionWide`
  in `emitFor`. The direct positive-range path must replace this code as well as
  the quantum-index guards.

- PR #2135 constructs QC modules directly, but QFT-shaped source requires
  relational constraints such as `j >= i + 1`. Independent intervals cannot
  prove that `q[j]` and `q[i]` differ.

- An access before an assignment in a repeating loop is not safe on later
  iterations. The semantic analyzer must find loop mutations before it analyzes
  the first iteration.

- Emitting proven loop arithmetic as `i64` still left index materialization that
  the `jeff` pipeline could not accept. Emitting the affine expression in MLIR's
  `index` type removes that conversion while an `i64` view remains available to
  ordinary scalar consumers.

- Existing compiler fixtures classified arbitrary runtime quantum indices as
  accepted input. The new frontend contract makes those fixtures
  semantic-rejection cases and moves the induction-index fixture into the
  `jeff`-compatible set.

- Mixed signed and unsigned range endpoints use unsigned comparison semantics. A
  direct signed-index loop is valid only when signed endpoints are proven
  nonnegative.

- Pairwise Presburger checks run before emitter resource checks. Wide affine
  barriers and broadcast gate calls therefore need their own semantic comparison
  limit.

- Rejecting all mutable scalar aliases forced simple constant indices into
  artificial one-iteration loops. Expanding each proven alias in the typed qubit
  reference lets dead scalar arithmetic disappear before `jeff` conversion while
  retaining the original OpenQASM program.

- The distinctness counter charged constant differences before the no-solver
  fast path. A broadcast of one proven operand pair could exhaust the budget
  even though each comparison was constant.

- OpenQASM defines constant negative indices relative to the end of a register.
  Compile-time normalization preserves that source behavior without runtime
  guards.

## Decisions

- Place all proof state in `OpenQASMSemantics.cpp` and expose only proof results
  through the typed OpenQASM frontend data. Rationale: OpenQASM owns the
  optional quantum-index extension; QC and QCO must remain independent of
  source-language policy.

- Use `MLIRPresburger` integer emptiness checks over affine loop domains.
  Rationale: bounds and same-register distinctness in triangular loops require
  relational reasoning, while this existing dependency provides exact integer
  checks.

- Admit only steps with a known positive integer value into the proven range
  path and ignore step congruence in the proof domain. Rationale: this is a safe
  over-approximation that covers the benchmark patterns and keeps lowering to a
  direct positive-step `scf.for`.

- Preserve generic loop and dynamic classical-index behavior. Rationale: only
  quantum-register indexing is optional in the base OpenQASM contract, and
  unrelated runtime behavior is outside issue #2188.

- Scan each repeating loop body for assignments before semantic analysis and
  exclude those scalar symbols from affine facts throughout that loop.
  Rationale: generation checks alone see only the first source-order iteration
  and could accept an index that is unsafe after the back edge.

- Track the current canonical affine expression for each scalar and invalidate
  that fact on unequal control-flow joins or possible loop mutation. Rationale:
  this admits known mutable values without treating mutability as runtime
  uncertainty. Canonical expressions also keep scalar aliases out of the
  emitter's proven-index path.

- Use direct positive-range lowering for mixed signed and unsigned endpoints
  only when every signed endpoint is proven nonnegative. Rationale: this
  condition makes unsigned promotion value-preserving; otherwise generic
  lowering retains the source semantics.

- Limit each barrier or gate call to 1,024 affine distinctness solver queries
  and resolve constant differences without charging the limit. Rationale: source
  limits permit wide operand lists, but repeated constant proofs do not consume
  Presburger resources.

## Outcome and validation

Semantic analysis proves affine indices against nested Presburger loop domains,
including relational bounds, overflow, known scalar values, and loop-back-edge
mutation. The emitter consumes those proofs and omits runtime quantum bounds,
wrapping, and distinctness scaffolding. Interval bounds alone cannot prove the
triangular and `j - step` cases.

The issue `#2188` program reached jeff without `i128`, `arith.select`, or
`cf.assert`. The full release/CTest validation passed with one expected QDMI
skip; the final scalar-fact revision passed 174 OpenQASM and 131 compiler tests.
Lint passed.

## Code and ownership

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` converts parsed syntax into the
typed structures declared in `mlir/include/mlir/Target/OpenQASM/Frontend.h`.
Before this change, quantum-register references carried an optional
`dynamicIndex` with no static-safety invariant.
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` then emits runtime
checks for such references and lowers only fully constant source ranges through
`scf.for`; other ranges become `scf.while`.

An affine expression is a constant plus integer coefficients multiplied by
active loop induction variables. A Presburger domain is a set of integer
solutions to linear equalities and inequalities. For an inclusive OpenQASM range
`[lower:step:upper]`, this work records `lower <= induction <= upper` when
`step` has a known positive integer value. Omitting the step's congruence class
adds possible values, so a property proved over this larger set is still safe.

The semantic analyzer will prove a quantum index by showing that adding either
`index < 0` or `index >= width` makes the active domain empty. It will prove
same-register operands distinct by showing that adding `left == right` makes the
domain empty. It will also prove every supported arithmetic node fits the signed
64-bit representation emitted by this frontend. The analyzer tracks canonical
affine expressions for scalar values and merges only equal facts across control
flow. A pre-scan excludes scalars assigned anywhere in a repeating loop so that
the proof remains valid after the loop back edge.

## Acceptance

The issue #2188 source must analyze, verify as QC, and pass the default compiler
pipeline to `jeff`. Its emitted QC must contain direct qubit loads inside
`scf.for` and no `i128`, quantum-index `cf.assert`, or negative-index
`arith.select`. A nested QFT-shaped loop with `j` ranging from `i + 1` through
the final register index must prove both bounds and operand distinctness. A
QFT-adder-shaped `j - step` index and reverse `N - 1 - i` index must also pass.
Constant negative indices must normalize relative to the register width.

Programs with possibly out-of-range, measurement-derived, nonlinear, mutated
loop-carried, unequally merged, or unsupported-step quantum indices must fail
semantic analysis with a precise diagnostic. Dynamic classical indexing and
generic runtime loops that do not supply quantum indices must retain their
current behavior. All existing test suites and lint checks must pass.
