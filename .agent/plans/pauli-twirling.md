# Add Pauli twirling to QCO programs

Status: historical implementation record.

## Goal and scope

MQT Core cannot currently apply Pauli twirling to a QCO program. After this
change, a user can copy a `QCOProgram`, run the opt-in `pauli-twirl-2q-gates`
pass, and obtain a program in which every supported two-qubit gate is surrounded
by four explicit single-qubit Pauli gates while the exact program unitary
remains unchanged.

The supported gate set is CX, CZ, ECR, and iSWAP. Custom gate matrices, multiple
output programs, and target-aware resynthesis are outside this plan; QCO already
exposes target compilation as a separate operation.

## Constraints

- The required public plumbing already exists. `QCOProgram.run_pass_pipeline()`
  accepts a registered textual MLIR pass, so the first release does not need a
  dedicated C++ or Python convenience API.

- Imported CX and CZ gates are not dedicated QCO operations. They are `qco.ctrl`
  operations with one control, one target, and a modifier body whose sole
  unitary is X or Z. ECR and iSWAP are direct QCO operations.

- Exact twirling sometimes requires a phase of pi. Four Pauli gates alone can
  produce the negative of the original matrix for valid rows. The pass must emit
  `qco.gphase` for those rows.

- Normalizing all global phases at the end of the pass altered otherwise
  ineligible programs. The generated phase corrections are already exact, so
  removing whole-module normalization both simplified the pass and preserved
  pre-existing phases byte-for-byte.

- All four supported gates use the same two-input/two-output rewiring path.
  Separate abstractions or a dedicated public Python wrapper were unnecessary.

## Decisions

- Implement the feature in the QCO MLIR layer, not in the legacy
  `qc::CircuitOptimizer`. Rationale: QCO is the active pass and Python
  integration layer.

- Make twirling an opt-in module pass and keep it out of the default
  optimization pipeline. Rationale: The transform is randomized and expands
  circuit size by four single-qubit gates per selected two-qubit gate.

- Use the existing textual pass-pipeline API before adding a dedicated wrapper.
  Rationale: `copy()` plus `run_pass_pipeline()` already provides the required
  non-mutating behavior with no new public API.

- Materialize identity operations and preserve exact phase. Rationale: This
  gives every twirl a uniform four-operation structure, including rows that
  choose I, while preserving the exact unitary.

- Derive and algebraically validate MQT-owned twirling tables from Core's gate
  matrices. Rationale: The implementation and its test oracle stay entirely
  within MQT Core and require no external table data.

- Emit only the phase correction required by the selected table row and leave
  every pre-existing global phase alone. Rationale: Twirling should not perform
  unrelated cleanup, especially in programs with no eligible gate.

## Outcome and validation

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

Historical validation passed 188 optimization tests, including all 64
gate/table-row full-unitary cases, 131 compiler tests, two Python pipeline
tests, and repository lint. The focused target is
`mqt-core-mlir-unittest-optimizations`.

## Code and ownership

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

## Acceptance

Acceptance requires the registered textual pass to transform a QCO CX without an
MLIR verifier error. Every transformed CX must have two explicit pre-Pauli
operations, the original CX, and two explicit post-Pauli operations. Exact
functionality must match before and after transformation for all 16 rows,
including rows whose correction is pi. Running the pass twice on equal copies
with the same seed must produce equal printed MLIR.

The pass must not twirl a multi-controlled X or an X nested inside a modifier
body. The pass must leave unsupported two-qubit operations unchanged. The same
semantic and structural contract applies to all four supported gates.

The Python integration test must run the textual pass on a copied QCO program,
leave the source text unchanged, and expose four inserted Pauli operations for a
fixed seed that selects the all-identity row.

## Interfaces

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
insufficient.
