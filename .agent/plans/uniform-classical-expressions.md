# Simplify fixed-width classical expressions

This ExecPlan follows `.agent/PLANS.md`, the repository development policy, and
`docs/ai_usage.md`. Keep Progress, Surprises & Discoveries, Decision Log, and
Outcomes & Retrospective current. The plan grants no remote GitHub authority.

## Purpose / Big Picture

OpenQASM, Qiskit, and jeff exchange classical computations through standard MLIR
integer operations. CBit owns mutable register storage. This cleanup removes
obsolete comparison reconstruction and private callback wrappers while
preserving supported format behavior. Focused regressions protect defects found
during the review, including stale measurement values and narrow integer
signedness.

## Progress

- [x] (2026-09-03) Review all PR files with OpenQASM, Qiskit, and adversarial
  specialists; distinguish production code from tests and design records.
- [x] (2026-09-03) Challenge the cleanup plan against width, memory ordering,
  expression depth, and backend capability contracts.
- [x] (2026-09-03) Remove obsolete Qiskit comparison reconstruction, unused
  OpenQASM state, and single-use CBit and QIR wrappers.
- [x] (2026-09-03) Consolidate the superseded CBit comparison design here.
- [x] (2026-09-03) Correct the confirmed frontend, snapshot, and jeff conversion
  defects and complete an independent cross-review of the cleanup.
- [x] (2026-09-03) Pass all 3,069 MLIR tests, 394 Qiskit/interchange tests, and
  stub regeneration, with no tracked stub changes.
- [x] (2026-09-03) Resolve the two C++ lint diagnostics and pass C++ and general
  lint without suppressions.
- [x] (2026-09-03) Inspect the final diff: 182 fewer production lines, with
  semantic regression tests and one consolidated design record.

## Surprises & Discoveries

Ordinary Qiskit expression emission already builds exact-width register reads,
casts, and comparisons. The former register-comparison recognizers only change
operation shape after removal of `cbit.cmp`; they are unnecessary for meaning.

Runtime probes found a measurement snapshot changing from 0 to 1 after Qiskit
export and import when its destination was overwritten. The existing snapshot
validator checked register loads and reads but omitted measurement values.

Sized integer declarations also reach consumers that previously assumed i64:
rotation distances, checked bit indices, integer gate powers, and switch
controls. These consumers need the existing signedness-aware cast before using
machine-width constants. jeff's signed comparison conversion similarly assumed
integer operands and crashed on valid index operands; promoted signed min/max
also selected the wrong result.

jeff has native integer widths 1, 8, 16, 32, and 64, with no general integer
cast or selection operation. Its array values are immutable, whereas CBit
registers are mutable; updating shared values requires snapshot preservation.
These are real backend constraints, not opportunities to remove validation.

## Decision Log

- Decision: Keep `cbit.read` and `cbit.write`, and represent computation with
  standard MLIR operations. Rationale: storage owns memory effects; integer
  operations already define width and comparison signedness. Date: 2026-09-03.
- Decision: Delete specialized Qiskit comparison recognition. Rationale: generic
  emission preserves the same values, including XOR-biased signed comparisons
  and unsigned widening. Date: 2026-09-03.
- Decision: Retain balanced bit reconstruction, compact population count,
  guarded shifts, and snapshot checks. Rationale: they prevent excessive source
  expression depth, oversized expansion, poison, and changed values. The
  adversarial review confirmed these constraints. Date: 2026-09-03.
- Decision: Reject unsupported scalar widths and array snapshot forms before
  producing a lossy result. Rationale: backend limits must be explicit; this
  cleanup does not add multiword arithmetic or a new alias model. Date:
  2026-09-03.

## Context and Orientation

`mlir/include/mlir/Dialect/CBit/IR/CBitOps.td` defines registers and their
memory operations. A whole-register read produces an immutable integer value
with bit zero least significant; a write updates the register from an
equal-width value. `mlir/lib/Dialect/CBit/IR/CBitOps.cpp` decomposes these
operations for MemRef and Adaptive QIR consumers. The private decomposition no
longer needs callbacks.

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` records scalar signedness and
width. `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits typed
MLIR; `TranslateQCToOpenQASM3.cpp` in the same directory exports expressions.
The Qiskit adapter, importer, and exporter live in `bindings/mlir/qiskit/`.

`mlir/lib/Support/IntegerExpressions.cpp` expands integer intrinsics only for
targets that lack them. `mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp` promotes
non-native widths at the jeff boundary and masks results to preserve their
logical width. `mlir/lib/Conversion/JeffToQCO/JeffToQCO.cpp` restores mutable
register operations while retaining observable array snapshots.

## Plan of Work

The first milestone removes dead and duplicated code without changing the
supported contract. Inline only private single-caller wrappers; keep the shared
CBit decomposition API. Use ordinary Qiskit expression emission for all
comparisons and existing constructors for signed comparison export.

The second milestone fixes confirmed semantic defects at their shared consumer
boundaries. Extend Qiskit's snapshot dependency walk to measurement destination
stores. Normalize narrow values before machine-width OpenQASM operations. Make
jeff integer conversion preserve index and signed ordering, and prevent switch
shortcuts from dropping effects. Each correction needs a small regression that
fails on the original PR head.

The final milestone validates the complete change and updates user documentation
for any explicit unsupported form. Keep public text about the resulting design;
do not retain obsolete CBit comparison instructions or duplicate design plans.

## Concrete Steps and Validation

Run commands from the repository root. Configure with `cmake --preset release`.
Use the configured LLVM/MLIR package matching the branch, then build affected
targets with `cmake --build --preset release`.

Run CTest filters for CBit, OpenQASM, jeff round trips, QIR, QC/QCO modifiers,
and DD functionality. These cover memory effects, decomposition, dynamic
indices, signed ordering, snapshot use, and conversion diagnostics. The focused
binaries include:

    build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
    build/release/mlir/unittests/Conversion/JeffRoundTrip/mqt-core-mlir-unittest-jeff-round-trip
    build/release/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation

Refresh the Python extension with
`uv sync --inexact --no-dev --no-build-isolation-package mqt-core`, then run
`uv run --no-sync pytest -q test/python/test_mlir_integer_interchange.py test/python/test_mlir_qiskit_translation.py`.
The interchange suite compares observable values through serialized jeff,
OpenQASM, and Qiskit, including cleanup, widths 1 through 64, and wide constant
comparisons. Keep those semantic checks; do not require one comparison graph.

Run `uvx nox -s stubs` after binding changes and `uvx nox -s cpp-lint` for every
changed C++ file. Inspect `git diff --check` and the final diff. Finish with
`uvx nox -s lint`. Report failed and unrun checks separately.

## Idempotence and Recovery

All edits and validation are local and repeatable. Preserve unrelated changes;
keep generated files and logs in ignored build directories. Do not edit fetched
dependencies or generated stubs. Any later commit must be signed and verified;
pushing or editing GitHub content requires explicit authorization.

## Interfaces and Dependencies

No dependency or public API is added. CBit uses signless exact-width integers.
OpenQASM and Qiskit retain their existing import/export APIs. Qiskit uses Uint
expressions and sign-bit XOR to encode signed order. jeff general integers
remain bounded at 64 bits; arbitrary-width register-versus-constant comparisons
retain their narrow bitwise lowering. QIR Base rejects whole-register
computation and Adaptive QIR lowers supported internal values. CBit register
calls remain unsupported in QIR.

## Outcomes & Retrospective

The final MLIR CTest suite passes all 3,069 tests. The Qiskit and integer
interchange suite passes all 394 tests. Stub regeneration succeeds without
tracked changes. Focused baseline probes demonstrated the measurement mismatch,
narrow-integer verifier failures, index assertion, and signed min/max mismatch
before their corrections.

The cleanup removes 182 net production lines and consolidates two design records
while adding semantic regressions. No dependency or public API was added. jeff
conservatively rejects live old arrays across mutating control flow, even when
the live array and mutated array differ. Preserving that external-jeff case
requires stronger alias reasoning and remains outside this cleanup.

C++ lint passes with zero findings, and general lint passes. No remote state was
changed.

Revision note (2026-09-03): Consolidated the superseded comparison plan,
recorded the specialist and adversarial review, and made cleanup validation
explicit.
