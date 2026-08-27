# Add Pauli twirling to QCO programs

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core cannot currently apply Pauli twirling to a QCO program. After this
change, a user can copy a `QCOProgram`, run the opt-in `pauli-twirl-2q-gates`
pass, and obtain a program in which every supported two-qubit gate is surrounded
by four explicit single-qubit Pauli gates while the exact program unitary
remains unchanged.

The first working milestone supports CX as the representative controlled gate.
The next milestone extends the same implementation to CZ, ECR, and iSWAP, which
form the complete initial supported gate set. Custom gate matrices, multiple
output programs, and target-aware resynthesis are outside this plan; QCO already
exposes target compilation as a separate operation.

## Progress

- [x] (2026-08-24 12:00Z) Traced the required transformation behavior, QCO gate
  representation, pass registration, Python pass-pipeline API, and focused
  MLIR test targets.
- [x] (2026-08-24 12:00Z) Created this ExecPlan and recorded the pass scope,
  semantic contract, and validation strategy.
- [x] (2026-08-24 12:10Z) Added and registered the opt-in, seeded
  `pauli-twirl-2q-gates` QCO pass.
- [x] (2026-08-24 12:13Z) Implemented exact CX twirling with four explicit Pauli
      operations and per-row global-phase correction.
- [x] (2026-08-24 12:15Z) Added exhaustive semantic and structural tests plus
      deterministic-seed, modifier-boundary, unsupported-gate, and
      unchanged-phase coverage.
- [x] (2026-08-24 12:17Z) Extended the table-driven implementation to CZ, ECR,
  and iSWAP after the shared representation passed the CX milestone.
- [x] (2026-08-24 12:19Z) Verified copied-program use through
  `QCOProgram.copy()` and the existing textual pass-pipeline API.
- [x] (2026-08-24 12:58Z) Rebuilt the focused targets, ran the Core checks and
      full lint session, inspected the final diff, and recorded the results
      below.

## Surprises & Discoveries

- Observation: The required public plumbing already exists.
  `QCOProgram.run_pass_pipeline()` accepts a registered textual MLIR pass, so
  the first release does not need a dedicated C++ or Python convenience API.
- Observation: Imported CX and CZ gates are not dedicated QCO operations. They
  are `qco.ctrl` operations with one control, one target, and a modifier body
  whose sole unitary is X or Z. ECR and iSWAP are direct QCO operations.
- Observation: Exact twirling sometimes requires a phase of pi. Four Pauli gates
  alone can produce the negative of the original matrix for valid rows. The pass
  must emit `qco.gphase` for those rows.
- Observation: Normalizing all global phases at the end of the pass altered
  otherwise ineligible programs. The generated phase corrections are already
  exact, so removing whole-module normalization both simplified the pass and
  preserved pre-existing phases byte-for-byte.
- Observation: All four supported gates use the same two-input/two-output
  rewiring path. Separate abstractions or a dedicated public Python wrapper were
  unnecessary.

## Decision Log

- Decision: Implement the feature in the QCO MLIR layer, not in the legacy
  `qc::CircuitOptimizer`. Rationale: QCO is the active pass and Python
  integration layer. Date/Author: 2026-08-24, Codex.
- Decision: Make twirling an opt-in module pass and keep it out of the default
  optimization pipeline. Rationale: The transform is randomized and expands
  circuit size by four single-qubit gates per selected two-qubit gate.
  Date/Author: 2026-08-24, Codex.
- Decision: Use the existing textual pass-pipeline API before adding a dedicated
  wrapper. Rationale: `copy()` plus `run_pass_pipeline()` already provides the
  required non-mutating behavior with no new public API. Date/Author:
  2026-08-24, Codex.
- Decision: Materialize identity operations and preserve exact phase. Rationale:
  This gives every twirl a uniform four-operation structure, including rows that
  choose I, while preserving the exact unitary. Date/Author: 2026-08-24, Codex.
- Decision: Derive and algebraically validate MQT-owned twirling tables from
  Core's gate matrices. Rationale: The implementation and its test oracle stay
  entirely within MQT Core and require no external table data. Date/Author:
  2026-08-24, Codex.
- Decision: Emit only the phase correction required by the selected table row
  and leave every pre-existing global phase alone. Rationale: Twirling should
  not perform unrelated cleanup, especially in programs with no eligible gate.
  Date/Author: 2026-08-24, Codex.

## Outcomes & Retrospective

The implementation is complete. MQT Core now exposes the seeded textual pass,
supports CX, CZ, ECR, and iSWAP, inserts four explicit Pauli operations per
eligible gate, corrects exact global phase when necessary, and leaves
unsupported or modifier-nested operations and pre-existing phases unchanged. The
existing generic `QCOProgram` API was sufficient, so no new binding, stub, or
dependency was added.

The Python integration test compiles a small program to QCO, twirls a fresh
copy, and verifies that the source remains unchanged. Custom gate matrices,
multiple independently twirled outputs, and target-aware synthesis remain
intentionally outside this first implementation.

## Context and Orientation

MQT Core represents optimized quantum programs in the QCO MLIR dialect. MLIR
uses static single assignment values, abbreviated SSA values, to connect the
output qubit of one operation to the input of the next operation. A transform
that inserts a gate must therefore reconnect both inputs to the two-qubit gate
and all later uses of its outputs.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` declares QCO passes and
their command-line options. A new definition there generates the base class and
factory for the pass. The implementation belongs in
`mlir/lib/Dialect/QCO/Transforms/Optimizations/PauliTwirling.cpp`; the transform
library already includes source files from that directory. The pass must be
registered in `mlir/lib/Support/Passes.cpp` so that
`QCOProgram.run_pass_pipeline()` can resolve its textual name.

`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` defines the existing I, X, Y, Z,
ECR, iSWAP, controlled-unitary, and global-phase operations. A CX is a
`qco.ctrl` with exactly one control and one target and an X operation as the
sole unitary in its body. The helper `mqt::getSoleBodyUnitary` in
`mlir/include/mlir/Dialect/MQT/Utils/Modifiers.h` recognizes that body. The pass
must inspect the outer `qco.ctrl` operands and results; it must not rewrite the
inner X operation.

A Pauli twirl chooses a pre-gate pair `(A, B)` and a post-gate pair `(C, D)`
from I, X, Y, and Z. For a supported two-qubit gate matrix `U`, every table row
must satisfy the exact equation

    exp(i * theta) * (C tensor D) * U * (A tensor B) = U

where `theta` is zero or pi. The circuit applies A and B before U and C and D
after U. The table contains 16 rows, one for every pre-gate pair, and sampling
must be uniform. The pass uses one local random-number engine per run, seeded
from a `uint64_t` pass option. A fixed default seed makes textual pipeline runs
reproducible; callers can supply a different seed.

## Plan of Work

First, add the pass declaration and registration. Give the pass the textual name
`pauli-twirl-2q-gates`, a short description that names its supported gates and
exact-phase behavior, and a `uint64_t` seed option with a reproducible default.
Build the generated pass declarations before adding rewrite logic to confirm the
name and base class.

Next, implement the CX milestone in `PauliTwirling.cpp`. Collect eligible outer
`qco.ctrl` operations before modifying the module. Reject operations with any
shape other than one control, one target, and a sole X body. Also reject an
eligible-looking operation when it is nested in a `qco.ctrl`, `qco.inv`, or
`qco.pow` modifier region. For every collected CX, sample one of 16 rows, insert
two explicit pre-Paulis, reconnect the outer control and target operands, insert
two explicit post-Paulis, and reconnect later users without replacing the
post-gates' own inputs. Insert `qco.gphase(pi)` in the same region when the row
requires it. Do not run the canonicalizer because it can erase identity gates.

Add `test_qco_pauli_twirling.cpp` to the optimization unit-test target. Build a
small QCO function or parse a short module with one CX. Tests must prove exact
functionality for every table row, four inserted Pauli operations, fixed-seed
stability, unchanged unsupported gates, and exclusion of modified-gate bodies.
Use the existing exact functionality helpers rather than adding a new matrix or
decision-diagram implementation.

After the CX tests pass, add CZ, ECR, and iSWAP through the same gate-kind and
table lookup. Do not introduce an interface or factory for four fixed cases.
Independently derive each 16-row table against Core's matrices and make the test
suite validate every row, including required phase corrections.

Finally, add a Python integration test that compiles a small program to QCO,
copies it, runs the textual pass on the copy, and proves the source remains
unchanged. Document the textual pass name, supported gates, exact-phase
behavior, and seed option in the compiler-collection guide.

## Concrete Steps

Run all Core commands from the MQT Core repository root. Build the focused
optimization target after the pass declaration and after each implementation
batch:

    cmake --build build/release --target mqt-core-mlir-unittest-optimizations

Run only the new tests while iterating:

    build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/\
      mqt-core-mlir-unittest-optimizations \
      --gtest_filter='*PauliTwirl*'

After Core tests pass, run the complete optimization and compiler test binaries:

    build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/\
      mqt-core-mlir-unittest-optimizations
    build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler

Finish with the repository lint and diff checks:

    uvx nox -s lint
    git diff --check

Record exact pass counts, failures, and unavailable checks in this plan.

## Validation Record

- `mqt-core-mlir-unittest-optimizations`: 188 tests passed, including 64 exact
  full-unitary checks covering every table row for every supported gate and two
  structural/regression tests.
- `mqt-core-mlir-unittests-compiler`: 131 tests passed.
- Focused Python textual-pipeline tests: 2 tests passed.
- `clang-format` passed for both new C++ files and the pass registration file.
  Rebuilding generated pass declarations verified the added `Passes.td`
  definition. Ruff passed for the touched Python files, the Markdown checker
  passed for the plan and compiler-collection page, and `git diff --check`
  passed.
- `uvx nox -s lint` passed in the isolated pull-request worktree.

## Validation and Acceptance

Acceptance requires the registered textual pass to transform a QCO CX without an
MLIR verifier error. Every transformed CX must have two explicit pre-Pauli
operations, the original CX, and two explicit post-Pauli operations. Exact
functionality must match before and after transformation for all 16 rows,
including rows whose correction is pi. Running the pass twice on equal copies
with the same seed must produce equal printed MLIR.

The pass must not twirl a multi-controlled X or an X nested inside a modifier
body. The pass must leave unsupported two-qubit operations unchanged during the
CX milestone. After the supported-gate extension, the same semantic and
structural contract applies to CZ, ECR, and iSWAP.

The Python integration test must run the textual pass on a copied QCO program,
leave the source text unchanged, and expose four inserted Pauli operations for a
fixed seed that selects the all-identity row.

## Idempotence and Recovery

All build and test commands are repeatable. Build products stay under `build/`
and are not committed. Source changes are additive and limited to the pass,
registration, focused tests, Python documentation, and this plan. Never use
reset, checkout, or a whole-file replacement to recover from an error. Apply a
narrow inverse patch only to Pauli-twirling lines, and inspect `git diff` before
every handoff.

## Artifacts and Notes

For a program with `N` eligible two-qubit gates, the pass retains those `N`
gates and inserts exactly `4 * N` Pauli operations.

The pass must preserve explicit identity operations. A later user-requested
cleanup pass can remove them, but twirling itself must not do so.

## Interfaces and Dependencies

Use the generated QCO pass base from `Passes.td`, `mlir::qco` gate builders,
`mqt::getSoleBodyUnitary`, `mlir::IRRewriter` or the nearest existing rewrite
utility, and a C++ standard-library random-number engine. Do not add a library
dependency. Use a fixed-size 16-row array for each supported gate and a compact
Pauli enum local to `PauliTwirling.cpp`.

The first public interface is the registered textual pass:

    pauli-twirl-2q-gates{seed=<uint64>}

Python callers use the existing methods:

    copied_program = program.copy()
    copied_program.run_pass_pipeline(...)

Do not add a dedicated Python binding unless the textual interface proves
insufficient. Follow `AGENTS.md` and `docs/ai_usage.md`; do not edit generated
type stubs, do not perform external GitHub actions, and disclose AI assistance
if this work later becomes a pull request.

Revision note (2026-08-24): Created the initial plan after tracing the current
QCO paths. Updated it after implementation to record complete four-gate support,
exact semantic coverage, Python integration, focused validation, and the
decision to preserve unrelated global phases unchanged.
