# Preserve fixed-width bit-register expressions across formats

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

OpenQASM 3 fixed-width `bit[N]` values support runtime bitwise expressions.
After this change, a whole register is read once with `cbit.read`, while MLIR's
existing fixed-width integer operations represent `~`, `&`, `|`, `^`, `<<`, and
`>>`. The existing `cbit.cmp` remains the compact register-versus-constant form.
OpenQASM and Qiskit can therefore exchange the same expression semantics without
reconstructing loads or signed-comparison range trees.

## Progress

- [x] (2026-09-02 21:48Z) Confirmed the OpenQASM and Qiskit type contracts and
  selected semantic, rather than structural, Qiskit round trips.
- [x] (2026-09-02 22:00Z) Extended `cbit.cmp` and its shared bitwise lowering to
      signed predicates.
- [x] (2026-09-02 22:00Z) Canonicalized eligible exact-width OpenQASM casts to
  `cbit.cmp`.
- [x] (2026-09-02 22:00Z) Encoded signed `cbit.cmp` operations in Qiskit
  unsigned expressions.
- [x] (2026-09-02 22:00Z) Added focused dialect, OpenQASM, Qiskit, and lowering
  tests.
- [x] (2026-09-02 23:10Z) Confirmed that runtime fixed-width bitwise expressions
      are required language support, not an exporter workaround.
- [x] (2026-09-02 23:38Z) Added symmetric `cbit.read` and `cbit.write`
      operations and added lowering or interpretation in existing CBit
      consumers.
- [x] (2026-09-02 23:38Z) Represented and emitted the bounded OpenQASM `bit[N]`
      expression subset, including runtime unsigned shifts.
- [x] (2026-09-02 23:38Z) Used the shared representation in OpenQASM and Qiskit
      export/import.
- [x] (2026-09-03 00:43Z) Made OpenQASM export reject stale and cross-region
  register snapshots and emit canonical rotations.
- [x] (2026-09-03 01:33Z) Ran the complete affected suites and lint, inspected
  the final diff, and recorded the audit.
- [x] (2026-09-03) Added Qiskit `Store` import and export, moved signed
  comparison recognition to the Qiskit boundary, and shared whole-register
  decomposition between MemRef and Adaptive QIR lowering.
- [x] (2026-09-03) Completed an independent final audit, required signless
      whole-register integer values, and passed the final affected suites and
      lint.

## Surprises & Discoveries

- Observation: Qiskit has six comparison operators but its expression type is
  `Uint`; it has no signed integer expression type. Evidence: the local Qiskit
  adapter normalizes only `Bool`, `Uint`, and `Float` in
  `bindings/mlir/qiskit/QiskitTranslation.h`.
- Observation: The current adapter limits Qiskit integer expressions to 64 bits
  even though `cbit.cmp` stores arbitrary-width `APInt` constants. Evidence:
  `setExpressionType` and `expressionType` reject widths above 64.
- Observation: Qiskit serializes a sign-bit-XOR comparison as `(c ^ S) < C`.
  Rejecting that valid fixed-width OpenQASM expression is the shared compiler
  gap, so an exporter-only range split would preserve needless asymmetry.
- Observation: Qiskit exposes `Store` in Python, but not through its C API.
  Reusing the adapter's existing deferred Python-instruction path preserves
  whole-register assignment without changing the C API boundary.
- Observation: jeff 2.x has no conversion operations for arbitrary fixed-width
  integers. The jeff path reports `cbit.read` and `cbit.write` directly instead
  of accepting an expression it cannot preserve.
- Observation: Inlining a `cbit.read` at its expression use can read newer state
  after an intervening write. Export must validate the SSA snapshot even though
  the source expression has no explicit load syntax.
- Observation: MLIR integer types do not retain OpenQASM scalar signedness. An
  arbitrary signless shift distance therefore cannot be emitted as `uint`.
  Register bit vectors and `popcount` retain enough provenance; other dynamic
  scalar distances must fail closed.
- Observation: `AnyInteger` also admits MLIR signed and unsigned integer types,
  but every CBit decomposition produces signless `iN` values. The `cbit.read`
  and `cbit.write` type constraints must state the signless invariant.

## Decision Log

- Decision: Keep all ten MLIR integer predicates on the one `cbit.cmp`
  operation. Rationale: signedness belongs to the comparison, not to CBit
  storage, and every lowering already consumes MLIR predicates. Date/Author:
  2026-09-02, Codex with user direction.
- Decision: Encode signed Qiskit ordering by XOR-biasing the sign bit and using
  the corresponding unsigned predicate. Rationale: this is one fixed-width
  expression, and supporting it closes a real OpenQASM language gap shared by
  both format paths. Date/Author: 2026-09-02, Codex with user direction after
  independent specialist review.
- Decision: Recognize Qiskit's exact sign-bit-XOR comparison encoding only in
  the Qiskit importer and delete the generic CBit arithmetic-graph
  canonicalizer. Rationale: the frontend knows the encoding it introduced;
  shared IR should not guess the intent of arbitrary arithmetic graphs.
  Date/Author: 2026-09-03, Codex after independent simplification review.
- Decision: Add unsigned, fixed-width `cbit.read` and `cbit.write` operations
  and reuse `arith` for all bitwise computation. Keep `cbit.cmp` for direct
  register and constant comparisons. Rationale: register memory semantics and
  whole-write snapshot ordering belong in CBit; arithmetic already belongs in
  `arith`. Date/Author: 2026-09-02, Codex.
- Decision: Canonicalize only one whole bit register compared with a constant
  that fits the explicit cast domain. Rationale: this exact contract is easy to
  prove; all other cast expressions retain the existing general lowering.
  Date/Author: 2026-09-02, Codex.
- Decision: Require a nonconstant shift distance to have `uint` type and be less
  than the register width. Fold constant overshifts to zero. Rationale: MLIR
  shifts are undefined outside that range; one documented source precondition
  keeps the OpenQASM and Qiskit representation direct and avoids a custom
  guarded-shift operation. Date/Author: 2026-09-02, Codex.
- Decision: Treat tests as evidence, not as the language contract. Remove or
  relax tests that pin operation counts, bit-by-bit lowering trees, or helper
  evaluators. Retain small checks for parsing, memory effects, conversion, and
  end-to-end meaning. Date/Author: 2026-09-02, user direction.
- Decision: Export a `cbit.read` expression only in the read's block and only
  before the next write to the same register. Rationale: this keeps direct
  expression emission while rejecting cases where OpenQASM would re-read a
  different value. Preindex writes once so repeated expressions do not rescan
  the function. Date/Author: 2026-09-03, Codex after independent specialist
  review.
- Decision: Accept exported dynamic shifts only when the distance is a
  bit-register expression of at most 64 bits or a known unsigned bit-vector
  scalar. Rationale: treating an arbitrary signless integer as unsigned emits
  OpenQASM that may not parse or may change meaning. Date/Author: 2026-09-03,
  Codex after independent specialist review.
- Decision: Preserve Qiskit `Store` atomically as `cbit.store` or `cbit.write`,
  including dynamic register indices, while continuing to reject standalone
  mutable variables. Rationale: Clbits and ClassicalRegisters already have an
  exact CBit representation; standalone variables require a different storage
  abstraction. Date/Author: 2026-09-03, user-selected scope after independent
  simplification review.
- Decision: Keep Jeff's arbitrary-width `cbit.cmp` lowering and defer general
  register reads and writes until Jeff has integer casts and logical right
  shift. Rationale: promotion would require per-operation logical-width masking
  and would still be incomplete at width 64. Date/Author: 2026-09-03,
  user-selected scope after independent simplification review.

## Outcomes & Retrospective

The implementation now uses one CBit representation for runtime fixed-width
values: `cbit.read` and `cbit.write` define register snapshots and updates,
ordinary integer operations define computation, and `cbit.cmp` retains compact
register-versus-constant comparisons. OpenQASM and Qiskit share this IR instead
of reconstructing bit-load graphs. Each frontend emits compact comparisons where
their source meaning is known, and one explicit lowering decomposes
whole-register operations for MemRef and Adaptive QIR consumers.

The final audit found no remaining practical semantic defect. OpenQASM rejects
stale or cross-region snapshots and dynamic shift distances whose unsigned
provenance was erased. Qiskit imports and exports Clbit, indexed-register, and
atomic whole-register `Store` operations through its public Python classes. QIR
Base and Jeff reject general whole-register expressions that they cannot
represent; Adaptive QIR lowers internal values. These are explicit backend
boundaries rather than speculative emulation.

Validation passed for 1,646 tests across ten affected C++ binaries and 253
Qiskit translation tests. `uvx nox -s lint`, `uvx nox -s cpp-lint`, stub
regeneration, and `git diff --check` passed. No dependency was added.

## Context and Orientation

`mlir/include/mlir/Dialect/CBit/IR/CBitOps.td` defines `cbit.cmp`, which reads a
statically sized register and compares it with an `APInt` constant.
`mlir/lib/Dialect/CBit/IR/CBitOps.cpp` expands that operation to bit loads and
Boolean arithmetic for consumers without native CBit support. The OpenQASM
frontend records exact-width bit-register casts in
`mlir/include/mlir/Target/OpenQASM/Frontend.h`; semantic analysis and QC
emission live in `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` and
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`. QC export to OpenQASM
and Qiskit is implemented in
`mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp` and
`bindings/mlir/qiskit/QiskitExport.cpp`.

A signed comparison interprets register bit N minus one as a two's-complement
sign bit. Equality and inequality do not depend on signedness. CBit's bit-level
lowering and Qiskit both bias the sign bit with XOR before applying the matching
unsigned ordering predicate.

## Plan of Work

First, allow signed predicates in `cbit.cmp` and make the shared lowering bias
the sign bit before using its existing unsigned comparison algorithm. Extend the
dialect and conversion tests with one case that distinguishes signed and
unsigned order.

Second, add a narrow OpenQASM semantic canonicalizer. It unwraps only implicit
scalar casts, accepts an exact-width `int[N]` or `uint[N]` of one whole
register, and requires the constant to fit the selected N-bit domain. It records
whether ordering is signed on `RegisterComparison`; unmatched expressions
continue through the existing packed 64-bit lowering.

Third, add `cbit.read`, whose `iN` result is the register's little-endian bit
pattern at that program point, and `cbit.write`, which atomically updates the
whole register from an `iN` value. Give them register memory effects and shared
expansions to static bit operations. Teach CBit consumers to lower or evaluate
them directly.

Fourth, extend the existing frontend `BitVectorExpression` record rather than
adding a parallel IR. Support register leaves, fitting nonnegative constants,
bitwise not/and/or/xor, logical shifts, rotations, and popcount. Require equal
operand widths, exact-width casts, and initialized register leaves. Define an
overshift result as zero and reject negative shift distances.

Fifth, export the resulting `cbit.read` plus `arith` tree directly to OpenQASM
and Qiskit. Encode signed Qiskit ordering with sign-bit XOR. Do not reconstruct
that expression as one signed operation on import; semantic preservation is the
contract. Separately, unwrap Qiskit's lossless unsigned widening around a whole
register when its comparison constant still fits the register width.

## Concrete Steps

From the repository root, edit the files named above with `apply_patch`. Build
the focused targets with:

    cmake --build --preset release --target mqt-core-mlir-unittest-cbit-ir mqt-core-mlir-unittest-cbit-to-memref mqt-core-mlir-unittest-openqasm-target

Run those binaries with their signed comparison test filters. Rebuild the Python
extension if needed, then run:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py -k 'register and comparison'

Finish with:

    uvx nox -s lint
    uvx nox -s cpp-lint

## Validation and Acceptance

The CBit verifier must accept `slt`, `sle`, `sgt`, and `sge`. Its shared
lowering must distinguish signed from unsigned order at the sign bit. OpenQASM
`int[N](register)` comparisons with in-range constants must produce signed
`cbit.cmp`; `uint[N](register)` must produce unsigned `cbit.cmp`. Runtime
`bit[N]` bitwise expressions, assignments, casts, comparisons, shifts,
rotations, and popcount must lower through `cbit.read`; writes must occur only
after the RHS snapshot. OpenQASM export must parse back with the same meaning.
Qiskit export must produce an unsigned XOR-biased expression, and direct import
must recognize that exact encoding as a compact signed comparison. Qiskit
`Store` round trips must preserve Clbit, indexed-register, and atomic
whole-register assignment semantics.

## Idempotence and Recovery

All edits and tests are repeatable. Build output stays under `build/`. Remote
publication uses signed commits and an exact force-with-lease after rebasing.
Preserve unrelated working-tree changes; if a formatter changes a touched file,
inspect and retain only relevant output.

## Artifacts and Notes

The final plan revision records the affected test totals and the backend
capability boundary.

## Interfaces and Dependencies

No dependency is added. `cbit.cmp` continues to use `mlir::arith::CmpIPredicate`
and `llvm::APInt`; `cbit.read` returns builtin `iN` and `cbit.write` consumes
it. Qiskit export continues to use the normalized `Expression` tree and its
existing `BitXor` and comparison operations. OpenQASM extends its existing typed
bit-vector record rather than adding sized scalar-variable support.

Revision note (2026-09-02): Created the plan after confirming that signed
comparisons have a lossless Qiskit `Uint` encoding and that the user does not
require structural signed round trips.

Revision note (2026-09-02): Broadened the plan after the user required genuine
runtime fixed-width bitwise support and parity between OpenQASM and Qiskit.

Revision note (2026-09-03): Added Qiskit `Store` parity, removed generic
comparison reconstruction, shared explicit CBit decomposition, and retained Jeff
comparison-only support rather than adding an incomplete promotion pass.

Revision note (2026-09-03): Recorded the independent final audit, signless CBit
integer contract, and final local validation.
