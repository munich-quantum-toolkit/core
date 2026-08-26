# Extend QC program inspection methods

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Users can currently count all, single-qubit, and two-qubit unitary operations in
a `QCProgram`. This change adds a per-operation gate histogram and a static gate
depth. Both methods describe the entry-point MLIR representation rather than the
number of operations executed at runtime. A user can call
`program.gate_counts()` and `program.static_depth()` from Python without parsing
textual MLIR.

## Progress

- [x] (2026-08-26 15:30Z) Investigated the QC reference model, SCF constructs,
      modifier semantics, existing gate walk, Python bindings, and
      documentation.
- [x] (2026-08-26 15:40Z) Added the C++ gate histogram and static-depth
  interfaces and implementation.
- [x] (2026-08-26 15:40Z) Added C++ tests for straight-line code, modifiers,
  barriers, branches, loop bodies, and dynamic register indices.
- [x] (2026-08-26 15:40Z) Added Python bindings, regenerated stubs, and extended
  the Python MLIR guide.
- [x] (2026-08-26 15:40Z) Ran validation. All 136 compiler tests, stub
      generation, the Python smoke test, and the full lint suite pass. The
      complete documentation build cannot finish because the local Graphviz
      `dot` executable is absent; Sphinx reached notebook execution before this
      unrelated environment failure.

## Surprises & Discoveries

- Observation: QC uses mutable qubit references, and a qubit-register access can
  use a dynamic index. Distinct SSA values can therefore refer to the same
  runtime qubit. Evidence: `collectRegisterAccesses` in
  `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` records each `memref.load` as a
  register and index pair instead of relying only on the load result.
- Observation: Modifier operations implement `UnitaryOpInterface` and can
  contain several nested gates. The existing counts treat the outer modifier as
  one gate and skip its body.
- Observation: A dynamic loop index makes the selected register element unknown
  even though the loop body appears only once in the static metric. The focused
  test confirms that the analysis synchronizes this access with a preceding
  constant-index access. Evidence: `QCProgramStaticDepthWithDynamicIndex`
  returns 2 for `h q[0]` followed by `x q[i]` in a loop body.
- Observation: The complete documentation build requires the Graphviz `dot`
  executable for an existing notebook. Evidence: Sphinx stopped in
  `docs/dd_package.md` with
  `ExecutableNotFound: failed to execute PosixPath('dot')`.

## Decision Log

- Decision: `gateCounts()` returns `std::map<std::string, size_t>` and uses
  `UnitaryOpInterface::getBaseSymbol()` for each key. Rationale: The result is
  deterministic, maps directly to a Python dictionary, and partitions the
  existing `numGates()` result. Modifier keys are `ctrl`, `inv`, and `pow`.
  Date/Author: 2026-08-26 / Codex.
- Decision: `staticDepth()` measures unitary gate layers in the static IR. It
  skips barriers and modifier bodies, takes the maximum of mutually exclusive
  `scf.if` and `scf.index_switch` branches, and analyzes each loop region once.
  Rationale: This metric shows the compact structure of SCF IR without claiming
  to predict runtime iterations. Date/Author: 2026-08-26 / Codex.
- Decision: Dynamic accesses to one register must conservatively alias every
  access to that register. Constant register indices remain distinct. Rationale:
  This prevents an underestimated depth when the index is known only at runtime.
  Date/Author: 2026-08-26 / Codex.

## Outcomes & Retrospective

The C++ and Python APIs are implemented. All 136 compiler tests and the full
lint suite pass. A Python smoke test returns `{'ctrl': 1, 'h': 1, 'x': 1}` and
depth 2 for two parallel gates followed by a controlled gate. The generated stub
and Python guide expose both methods. The complete documentation build remains
unavailable only because the local Graphviz executable is missing.

## Context and Orientation

`mlir/include/mlir/Compiler/Programs.h` declares the typed compiler program API.
`mlir/lib/Compiler/Programs.cpp` implements it and contains `countGatesIf`,
which walks the entry-point operation and counts QC operations that implement
`UnitaryOpInterface`. The walk includes every nested region once, skips
barriers, and does not descend into `qc.ctrl`, `qc.inv`, or `qc.pow`.

QC has reference semantics: a `!qc.qubit` value denotes a mutable qubit.
`qc.alloc` creates a dynamic scalar qubit, `qc.static` identifies a hardware
qubit by a fixed integer, and `memref.load` can retrieve a qubit from register
storage. Static depth is the number of unitary gate layers on the longest
qubit-dependency chain in the IR. Gates on disjoint qubits can share a layer.

`bindings/mlir/register_mlir.cpp` exposes `QCProgram` through nanobind.
`python/mqt/core/mlir.pyi` is generated and must only change through
`uvx nox -s stubs`. `docs/mlir/python_compiler_collection.md` shows Python users
how to inspect compiler programs. The focused C++ tests live in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp` and run through
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.

## Plan of Work

First, refactor the existing gate traversal in `mlir/lib/Compiler/Programs.cpp`
so the histogram and scalar counts use the same gate selection rules. Add
`QCProgram::gateCounts()` to increment the entry for each outer gate's base
symbol.

Next, implement a small static-depth analysis in the same source file. Track a
depth for scalar qubits, static indices, and register elements. Resolve constant
register indices separately. Treat a dynamic index as an access to the complete
register and synchronize it with every known element. For a gate, take the
maximum depth of all its qubits, add one, and write that value back to every
qubit resource. Ignore zero-qubit gates because they have no qubit dependency.

Walk straight-line regions in operation order. Analyze each `scf.if` and
`scf.index_switch` branch from the same input state, then merge states by taking
the maximum depth for each resource. Analyze `scf.for` once. Analyze the before
and after regions of `scf.while` once in execution order. Skip modifier bodies
because the outer modifier is one logical gate.

Add tests that prove the histogram sums to `numGates()`, modifier names are
outer operation names, independent gates share a layer, dependent gates add a
layer, branches take a maximum instead of a sum, and loop bodies count once.
Include a register-index case to protect resource identity.

Finally, bind the methods as `gate_counts` and `static_depth`, regenerate Python
stubs, and update the inspection example and semantic explanation in the Python
compiler guide.

## Concrete Steps

Run all commands from the repository root.

After editing the C++ API and implementation, build and run the focused tests:

    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.QCProgram*'

After editing bindings, regenerate the checked-in type stub:

    uvx nox -s stubs

Validate the documentation and all repository checks:

    uvx nox --non-interactive -s docs
    uvx nox -s lint

## Validation and Acceptance

For a program containing two parallel one-qubit gates followed by a two-qubit
gate, `staticDepth()` must return 2. For an if statement with branch depths 1
and 2, it must return 2 before later gates are considered; the branches must not
add to 3. A loop with a body depth of 1 must contribute 1 regardless of its
runtime trip count.

For every test program, summing `gateCounts()` values must equal `numGates()`.
Barriers must appear in neither result. A controlled modifier around `x` must
increment `ctrl`, not `x`, because the public count treats the modifier as one
logical gate.

The focused C++ test binary, documentation build, and `uvx nox -s lint` must
finish successfully. Stub generation must leave only the intended API additions
in `python/mqt/core/mlir.pyi`.

## Idempotence and Recovery

Build, test, documentation, stub, and lint commands are repeatable. Stub
generation may rewrite the generated file; inspect its diff and rerun it after
each binding signature change. Preserve unrelated working-tree changes and do
not reset files to recover from a failed check.

## Artifacts and Notes

The existing gate-count contract is the reference for gate selection:

    operations in each nested region count once;
    barriers do not count;
    a modifier counts once and its body does not count again.

## Interfaces and Dependencies

At completion, `mlir::QCProgram` must provide:

    [[nodiscard]] std::map<std::string, size_t> gateCounts() const;
    [[nodiscard]] size_t staticDepth() const;

Python must provide:

    def gate_counts(self) -> dict[str, int]: ...
    def static_depth(self) -> int: ...

Use MLIR's QC operation interfaces and SCF operation classes already linked by
the compiler library. Do not introduce a new external dependency.

Plan revision note: Finalized after implementation and validation. Recorded the
successful compiler tests, stub generation, Python smoke test, and lint suite,
as well as the unrelated local documentation dependency failure.
