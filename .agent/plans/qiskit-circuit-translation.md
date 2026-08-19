# Add Qiskit circuit import and export

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` are
maintained as required by `.agent/PLANS.md`.

## Purpose / Big Picture

The compiler collection accepts a Qiskit `QuantumCircuit` as a direct frontend
input and can emit a Qiskit circuit from a compatible flat QC program. This
translation does not create a `QuantumComputation`. The existing
`mqt.core.load`, `qiskit_to_mqt`, and `mqt_to_qiskit` APIs remain independent
and retain their wider Qiskit compatibility.

The direct translation supports Qiskit `>=2.5.0,<2.6.0`. Import covers standard
gates and modifiers with supported numeric or symbolic parameters, global phase,
canonical registers, measurement, reset, barrier, recursive custom definitions,
and structured control flow with classical-bit and register conditions and
supported expressions. Standalone classical variables are rejected. Export
covers the flat constructible subset. Validation completes before the
destination program is created.

## Progress

- [x] (2026-08-13 01:34Z) Define an owned-module API and a dialect-owned
      standard-gate descriptor and emitter.
- [x] (2026-08-13 01:34Z) Integrate version-specific Qiskit translation into the
      existing nanobind extension and use scikit-build-core's stable-ABI
      request.
- [x] (2026-08-13 01:34Z) Implement complete import and export validation,
      recursive custom-definition expansion, structured import, canonical
      register allocation, and layout omission.
- [x] (2026-08-13 01:34Z) Add the PEP 723 adoption script and a self-contained
      Qiskit upstream nox session.
- [x] (2026-08-13 01:34Z) Add contract tests, executable documentation, and a
      four-commit history.
- [x] (2026-08-13 01:34Z) Audit the complete diff and remove unused builder
      inputs, standalone variables, duplicate gate data, broad suppressions, and
      implementation-oriented tests.

## Surprises & Discoveries

- Qiskit's header function tables and `qk_import()` state are local to one
  translation unit. All `Qk*` types, functions, and table access therefore stay
  in `Qiskit2_5.cpp`.
- A custom instruction's definition exposes the symbols and expressions bound at
  its call site. The symbolic-parameter translation validates those values
  against the current global and lexical identities; it needs no separate
  formal-parameter substitution scheme.
- Qiskit 2.5 provides native inspection for structured control flow and
  classical expressions but does not provide the corresponding constructors.
  Import can represent these structures in SCF and Arith. Export must reject
  them.
- Qiskit transpiler layout metadata is independent of the circuit instruction
  stream. Import can accept a circuit with `circ.layout` and translate the
  operations without preserving physical or virtual layout information.
- The QC builder already records source register names on allocation operations.
  Reusing `mqt.qubit_register_name` and `mqt.classical_register_name` avoids a
  Qiskit-specific metadata scheme and makes Qiskit-to-OpenQASM translation use
  the same representation.

## Decision Log

- Decision: Use the terms "Qiskit circuit import and export" and "Qiskit
  translation." Rationale: these terms match compiler frontend and emission
  concepts without naming an internal API mechanism. Date/Author: 2026-08-12 /
  Codex.
- Decision: Compile each supported minor's source into the existing nanobind
  extension. Rationale: nanobind owns Python object lifetimes and stable-ABI
  configuration, while source properties keep Qiskit's private headers and
  extension macro local. Date/Author: 2026-08-12 / Codex.
- Decision: Reject free compile-time parameters and arbitrary unitaries in the
  original change. Rationale: neither had a complete compiler representation and
  round-trip contract at that time. The free-parameter decision is superseded by
  `.agent/plans/qiskit-symbolic-parameters.md`; the arbitrary-unitary decision
  is unchanged here. Date/Author: 2026-08-12, partially superseded 2026-08-18 /
  Codex.
- Decision: Expand unknown instructions through their definitions. Rationale:
  recursive expansion supports composite gates without adding custom-operation
  semantics to QC. Expansion is bounded by a depth of 64 and 10 million
  operations. Date/Author: 2026-08-12 / Codex.
- Decision: Accept only leading loose bits followed by disjoint, contiguous,
  named registers. Rationale: this is the shared representation that QC
  allocation attributes and the Qiskit construction API can preserve without
  aliases. Date/Author: 2026-08-12 / Codex.
- Decision: Accept and ignore transpiler layouts. Rationale: layout metadata is
  not needed to translate the circuit instruction stream, and silently applying
  it would change instruction semantics. Date/Author: 2026-08-12 / Codex.
- Decision: Preflight import and export completely. Rationale: a rejected source
  must remain unchanged, and an output failure must not allocate a partial
  Python circuit. Date/Author: 2026-08-12 / Codex.

## Outcomes & Retrospective

The code is organized as a generic translation, a version registry, and one
clearly named version-specific source. The version-specific source holds the
Qiskit C API and uses nanobind handles for Python interaction. The generic
translation only sees normalized C++ reader and writer interfaces.

`Program::module()` provides a borrowed module while the program remains valid.
`QCProgram::fromModule()` verifies a context-matched module with QC operations
before accepting ownership. The Qiskit importer uses this API instead of a
friend class.

OpenQASM and Qiskit reuse a QC-owned standard-gate descriptor for gate identity,
parameter arity, control arity, target arity, operation identity, and primitive
emission. OpenQASM owns its canonical names, syntax aliases, and availability
rules. Qiskit owns its version-specific names.

The focused native ownership and gate-descriptor tests pass. The direct
translation, compiler binding, adoption tool, existing load, and existing Qiskit
conversion tests pass. Python 3.10 regular-ABI and Python 3.12/3.14 stable-ABI
builds import successfully. Stub generation, the executable documentation,
repository lint, and `git diff --check` pass.

## Context and Orientation

`bindings/mlir/register_mlir.cpp` exposes `QCProgram.from_qiskit`,
`QCProgram.to_qiskit`, and Qiskit input dispatch for `compile_program`.

`bindings/mlir/qiskit/QiskitImport.cpp` and `QiskitExport.cpp` contain the
generic directions. `QiskitVersion.cpp` selects a registered version.
`Qiskit2_5.cpp` contains all Qiskit-native declarations and calls.
`SupportedVersions.inc` is the version registry.

`mlir/include/mlir/Dialect/QC/Translation/StandardGate.h` describes the common
gate set. `OpenQASMToQCEmitter.cpp` and Qiskit import both use its emitter.

`scripts/qiskit_c_api_adopt.py` verifies and stages a released header snapshot,
compares the API surface, tests a candidate translation, updates the version
registry atomically, and tests the registered translation. Its PEP 723 metadata
makes it directly runnable with `uv` on Python 3.14 or newer.

## Milestones

The first milestone adds only the compiler-owned infrastructure. A compiler
program can lend its module to a read-only consumer, and `QCProgram` can accept
an owned module after it verifies the context and QC dialect. OpenQASM and other
frontends can use one QC gate descriptor and one primitive emitter. The compiler
and OpenQASM native tests prove this milestone without Qiskit installed.

The second milestone adds circuit import and export to the nanobind extension.
One generic importer and one generic exporter use normalized C++ interfaces.
`Qiskit2_5.cpp` contains the Qiskit-native calls and names. The importer checks
the complete reachable circuit before it creates an MLIR module. The exporter
checks the complete flat QC program before it creates a Qiskit circuit. The
focused Python tests demonstrate supported circuits and each rejection class.

The third milestone adds maintenance tooling. The PEP 723 script downloads one
exact released wheel, verifies its hash and embedded version, stages its header
snapshot, compares the native surface, builds the candidate, registers the
minor, and builds it again. The Qiskit nox session performs the same candidate
build against Qiskit main without changing the generic test helper.

The final milestone documents the public contract and validates the complete
change. The executable MyST page shows a direct round trip and lists the import
and export limits. The changelog names the new interface. Native tests, focused
Python tests, stable-ABI and regular-ABI builds, stubs, documentation, lint, and
the final diff must all pass.

## Plan of Work

The importer opens the source circuit through the selected version reader. It
then validates the full reachable instruction graph. Validation checks numeric
and supported symbolic parameters, gate and modifier arity, canonical registers,
classical expression types, control-flow mappings, custom-definition cycles,
definition arity, definition depth, and the operation budget. Arbitrary
unitaries, unsupported expressions, and other unsupported operations fail during
this pass. Only then does the importer allocate an MLIR context and module.

The importer creates leading anonymous allocations for loose resources and one
named allocation for each canonical register. Classical bit references map each
circuit bit to an allocation and local index. Control-flow blocks and custom
definitions compose their local-to-root bit maps with this mapping. Runtime bit
and register conditions load their values from classical storage. Supported
classical-expression trees contain constants but no variables. Loop induction
parameters remain lexically bound values.

The exporter borrows the program module, checks the single entry function,
accepts named `f64` inputs and supported Arith and Math expression graphs,
rejects other input types and structured or runtime classical execution, and
collects flat operations and allocation attributes. It validates the recovered
register layout before it selects a Qiskit writer or allocates a circuit.

The version registry parses `qiskit.__version__` before native API access. The
matching source calls `qk_import()` and verifies its reported major and minor.
An optional exact candidate source supports upstream CI without adding a Python
fallback.

## Concrete Steps

Run all commands from the repository root. Configure and build the native test
targets with an LLVM/MLIR 22.1 or newer installation:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build build/release --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-openqasm-target -j4
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target

Build the Python extension and run the direct and existing conversion tests:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev \
      --no-build-isolation-package mqt-core
    ./.agent/run.sh uv run --no-sync pytest \
      test/python/test_mlir_qiskit_translation.py \
      test/python/test_mlir.py \
      test/python/test_qiskit.py \
      test/python/test_qiskit_c_api_adopt.py

Check both packaging modes where the interpreters are available. Python 3.10
must produce a regular extension. Python 3.12 and newer must produce a stable
ABI extension. Then generate stubs, execute the documentation, and run the full
repository lint session:

    ./.agent/run.sh uvx nox -s tests-3.10
    ./.agent/run.sh uvx nox -s tests-3.12
    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check

## Validation and Acceptance

Acceptance requires:

- C++ tests for borrowed module access, checked ownership transfer, shared gate
  descriptors, and existing OpenQASM translation.
- Table-driven Qiskit gate import and export with finite numeric modifiers,
  global phase, canonical registers, measurement, reset, and barrier.
- Recursive parameterized custom definitions plus missing, cyclic, mismatched,
  and overly deep definitions.
- Nested structured control flow, loop-bound parameters, and representative
  Boolean, unsigned integer, and floating-point expressions.
- Supported free symbols and parameter expressions, plus early rejection of
  unsupported expressions, arbitrary unitaries, aliases, interleaved registers,
  and standalone classical variables without source mutation.
- Successful import of a circuit with `circ.layout`, with layout metadata absent
  from the compiler program.
- Flat export rejection for structured programs and unsupported runtime input
  types or expression graphs, unsupported version dispatch, lazy Qiskit import,
  and unchanged existing converter tests.
- Python 3.10 regular-ABI and Python 3.12 or newer stable-ABI builds where those
  interpreters are available, generated stubs, documentation, repository lint,
  and `git diff --check`.

All cache-producing commands run through `./.agent/run.sh`.

## Idempotence and Recovery

Import and export are read-only with respect to their source objects. Validation
does not allocate the destination. Repeating a successful translation produces
the same semantic program.

The adoption script stages downloads and generated files in temporary paths,
checks hashes and existing content, and uses atomic replacement. It accepts only
changes for the requested version when resuming.

Build products remain inside this worktree. The feature branch is replaced only
after the remote head still matches the recorded lease. Downstream pull requests
are outside this plan and require separate rebase authorization.

## Artifacts and Notes

The vendored Qiskit 2.5.0 snapshot includes its Apache-2.0 license,
`PROVENANCE.json`, `API_SURFACE.json`, and per-header SHA-256 hashes. It is a
private build input and is not installed as an MQT C++ interface.

The original #2031 validation produced these final summaries. The symbolic
parameter ExecPlan records the later symbolic validation:

    [  PASSED  ] 234 tests.  # compiler program and pipeline tests
    [  PASSED  ] 291 tests.  # QC and OpenQASM translation tests
    [  PASSED  ] 163 tests.  # OpenQASM frontend and target tests
    172 passed, 3 skipped    # Qiskit translation, compiler, and legacy tests
    11 passed                # adoption-tool tests
    12 executed cells        # compiler collection documentation
    All repository hooks passed.

## Interfaces and Dependencies

The public C++ additions are:

    ModuleOp Program::module() const;

    static std::optional<QCProgram> QCProgram::fromModule(
        std::shared_ptr<MLIRContext> context,
        OwningOpRef<ModuleOp> module);

The Python additions are:

    QCProgram.from_qiskit(circuit: qiskit.QuantumCircuit) -> QCProgram
    QCProgram.to_qiskit() -> qiskit.QuantumCircuit

Qiskit remains an optional Python dependency. Vendored headers are private to
the version-specific source, and no installed MQT header contains a `Qk*` type.

## Revision Note

The final revision records the reduced contract, the completed necessity audit,
the exact validation commands, and the observed results. It removes design paths
that are not part of the implemented import and export interface.
