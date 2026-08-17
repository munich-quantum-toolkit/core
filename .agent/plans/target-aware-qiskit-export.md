# Add target-aware Qiskit export

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT target compilation produces programs whose static qubits refer to sites of a
`CompilerTarget`. A caller can now pass that target to `QCProgram.to_qiskit`.
The exporter then returns a canonical physical Qiskit circuit with one register
named `q`, one qubit for every target site, and dense Qiskit indices that match
the order of `CompilerTarget.sites`. Generic export without a target keeps its
existing behavior.

## Progress

- [x] (2026-08-17 09:09Z) Inspect target compilation, Qiskit export, Python
      bindings, tests, and documentation.
- [x] (2026-08-17 09:09Z) Define target-aware export behavior for dense and
      sparse target site IDs.
- [x] (2026-08-17 09:14Z) Add the optional target argument and target-aware
      resource validation.
- [x] (2026-08-17 09:14Z) Add end-to-end, sparse-site, and invalid-input tests.
- [x] (2026-08-17 09:14Z) Update the generated Python stub, documentation, and
      changelog.
- [x] (2026-08-17 09:25Z) Run focused builds and tests, documentation checks,
      include-cleaner, lint, and diff checks.

## Surprises & Discoveries

- Observation: A target site ID is not a Qiskit qubit index. A target may use
  sparse IDs such as 10 and 20 while containing only two sites. Evidence:
  `CompilerTarget::vertexForSite` maps those IDs to compiler vertices 0 and 1.
- Observation: Qiskit uses one owning quantum register named `q` for its
  canonical physical circuit form. Layout metadata is independent of that
  register representation.
- Observation: Target compilation does not store the routing layout in the
  returned `QCProgram`. Correct Qiskit layout metadata therefore needs a
  separate design and is outside this work.
- Observation: CUDA-Q represents device topology with a module-level
  `quake.wire_set` and records mapping permutations in the function attributes
  `mapping_reorder_idx` and `mapping_v2p`. This is a precedent for a separate
  routing-layout retention design; it does not change the target-aware export
  contract in this plan.

## Decision Log

- Decision: Make `CompilerTarget` an optional argument to export instead of
  adding target metadata to the MLIR module. Rationale: the target already owns
  the number and order of its sites, and the caller can supply the same
  immutable target used for compilation. Date/Author: 2026-08-17 / Codex.
- Decision: Map `qc.static` site IDs through `CompilerTarget::vertexForSite`.
  Rationale: Qiskit physical-qubit indices are consecutive and zero-based, while
  target site IDs can be sparse. Date/Author: 2026-08-17 / Codex.
- Decision: Reject dynamic quantum allocation during target-aware export.
  Rationale: a dynamic qubit has no target site and therefore no defined
  physical-qubit index. Date/Author: 2026-08-17 / Codex.
- Decision: Do not emit Qiskit layout metadata. Rationale: a canonical physical
  register describes the circuit's physical-qubit space but does not describe
  the compiler's initial or final logical-to-physical permutation. Date/Author:
  2026-08-17 / Codex.

## Outcomes & Retrospective

Target-aware export now uses the supplied `CompilerTarget` directly and stores
no target data on MLIR modules. The binding build succeeded. All 141 tests in
the two affected Python files passed, including the five focused target-aware
cases. All 133 compiler tests, generated-stub comparison, repository lint, and
`git diff --check` also passed. The warning-as-error documentation build and
focused include-cleaner checks passed.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines `mlir::CompilerTarget`. Its
`sites()` method returns sites in compiler-vertex order, `numQubits()` returns
their count, and `vertexForSite()` maps a target-defined site ID to that dense
order.

`bindings/mlir/qiskit/QiskitExport.cpp` collects quantum and classical resources
from a `QCProgram` and constructs a Qiskit `QuantumCircuit` through Qiskit's C
API. `bindings/mlir/qiskit/Qiskit.h` declares the native exporter.
`bindings/mlir/register_mlir.cpp` exposes it as `QCProgram.to_qiskit`.
`python/mqt/core/mlir.pyi` is the generated Python type stub.

Mapping replaces dynamic QCO allocations with `qco.static` operations that use
target site IDs. QCO-to-QC conversion preserves those IDs as `qc.static`.
Cleanup can remove unused static qubits, so the remaining program alone cannot
describe the full target. The target argument supplies the authoritative site
set and order during export.

## Milestones

The first milestone adds the optional target to the native and Python export
interfaces. At its end, the five focused Python tests pass: dense target export
uses the full target, sparse site 20 maps to Qiskit qubit 1, and invalid static
or dynamic resources fail with controlled errors. Generic export remains
unchanged.

The second milestone completes the public contract and validation. At its end,
the generated stub matches the binding, the target-compilation documentation
describes the canonical physical result and its preconditions, and all affected
Python tests pass. The compiler suite, documentation build, repository lint, and
diff checks also pass.

## Plan of Work

Extend the native export function with a borrowed optional `CompilerTarget`
pointer. Keep the no-target path unchanged. When a target is present, allocate
no loose quantum bits and create one register named `q` with
`target.numQubits()` qubits. Map every `qc.static` site through
`target.vertexForSite()`. Reject an unknown site and reject scalar or register
quantum allocations that are not static. Classical resources keep their current
behavior.

Expose the target as a keyword-only optional argument:

    qc_program.to_qiskit(*, target: CompilerTarget | None = None)

Update `test/python/test_mlir.py` with an end-to-end case that compiles a
two-qubit program for `CompilerTarget(5)`, converts it to QC, and exports it
with the same target. The result must contain five qubits in one register named
`q` and have no Qiskit layout metadata.

Update `test/python/test_mlir_qiskit_translation.py` with a sparse target whose
site IDs are 10 and 20. An operation on site 20 must act on Qiskit qubit 1. Also
verify that export rejects a site absent from the supplied target and rejects
dynamic scalar and register qubits. Existing tests continue to cover generic
export without a target.

Document the final API in `docs/mlir/target_compilation.md`. State that callers
must pass the same target used for mapping and that the option neither compiles
the program nor emits layout metadata. Keep one concise changelog entry.

## Concrete Steps

Run all commands from the repository root. Build the native binding and run the
two affected Python test files:

    uv sync --locked --only-group dev
    cmake --build build/python/MinSizeRel --target mqt-core-mlir-bindings --parallel 8
    uv run --no-sync pytest test/python/test_mlir.py \
      test/python/test_mlir_qiskit_translation.py

Regenerate the Python stubs from the compiled binding:

    uvx nox -s stubs

Build and run the compiler suite to cover the target-compilation side of the
end-to-end workflow:

    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Validate documentation and repository policy checks:

    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

Record the exact results in this plan. If the environment cannot run a check,
record the command and diagnostic instead of reporting a pass.

Completed results:

- The MinSizeRel Python binding build passed.
- All 141 tests in `test/python/test_mlir.py` and
  `test/python/test_mlir_qiskit_translation.py` passed in 2.70 seconds.
- All 133 tests in `mqt-core-mlir-unittests-compiler` passed.
- Temporary nanobind stub generation matched the checked-in `to_qiskit`
  signature and docstring.
- `uvx nox --non-interactive -s docs` passed with Sphinx warnings treated as
  errors.
- Clang-tidy 21.1.1 `misc-include-cleaner` checks passed with warnings treated
  as errors for the binding and mapping unity translation units.
- `uvx nox -s lint` passed every hook.
- `git diff --check` passed.

## Validation and Acceptance

The end-to-end test must return a five-qubit Qiskit circuit after compilation
uses only two sites of a five-site target. The circuit must have exactly one
quantum register named `q`, and `circuit.layout` must be `None`.

For a target with sites 10 and 20, an MLIR operation on `qc.static 20` must use
Qiskit qubit index 1 in a two-qubit circuit. A `qc.static` site outside the
target must fail with a controlled error. A target-aware export containing
`qc.alloc` or a quantum `memref.alloc` must also fail. Existing target-free
round-trip tests must continue to pass.

The generated stub must contain the keyword-only optional target argument. The
documentation build, focused test suites, repository lint, and
`git diff --check` must pass or have a recorded environment limitation.

## Idempotence and Recovery

Source edits and validation commands are repeatable. Export does not consume the
source `QCProgram`. Build output remains under `build/` and is not committed. If
a build or test fails, correct the focused source or test and rerun the same
command. No migration or destructive operation is required.

## Artifacts and Notes

The sparse mapping has this observable result:

    target sites:       [10, 20]
    Qiskit q indices:   [ 0,  1]
    qc.static 20:       q[1]

The canonical physical circuit has one owning register. It deliberately has no
`TranspileLayout` until compiler layout retention has its own contract.

## Interfaces and Dependencies

`mqt::bindings::qiskit::exportCircuit` accepts
`const mlir::CompilerTarget* target = nullptr`. The Python method accepts a
keyword-only `CompilerTarget | None`. The target is immutable and remains alive
for the synchronous export call; the exporter does not retain it.

The implementation uses existing `CompilerTarget::numQubits()` and
`CompilerTarget::vertexForSite()` methods. It uses the existing Qiskit 2.5 C API
translation layer and introduces no dependency.

Revision note (2026-08-17): Replaced module metadata with an optional
`CompilerTarget` export argument and rewrote the plan around the final public
behavior.
