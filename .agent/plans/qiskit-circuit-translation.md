# Add Qiskit circuit import and export

Status: historical implementation record.

Later capability additions: [symbolic parameters](qiskit-symbolic-parameters.md)
and [dense unitaries](dense-unitary-operations.md).

## Goal and scope

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

## Constraints

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

## Decisions

- Use the terms "Qiskit circuit import and export" and "Qiskit translation."
  Rationale: these terms match compiler frontend and emission concepts without
  naming an internal API mechanism.

- Compile each supported minor's source into the existing nanobind extension.
  Rationale: nanobind owns Python object lifetimes and stable-ABI configuration,
  while source properties keep Qiskit's private headers and extension macro
  local.

- Free scalar parameters and dense unitaries were excluded from the original
  scope. Their later contracts are recorded in
  [symbolic parameters](qiskit-symbolic-parameters.md) and
  [dense unitaries](dense-unitary-operations.md).

- Expand unknown instructions through their definitions. Rationale: recursive
  expansion supports composite gates without adding custom-operation semantics
  to QC. Expansion is bounded by a depth of 64 and 10 million operations.

- Accept only leading loose bits followed by disjoint, contiguous, named
  registers. Rationale: this is the shared representation that QC allocation
  attributes and the Qiskit construction API can preserve without aliases.

- Accept and ignore transpiler layouts. Rationale: layout metadata is not needed
  to translate the circuit instruction stream, and silently applying it would
  change instruction semantics.

- Preflight import and export completely. Rationale: a rejected source must
  remain unchanged, and an output failure must not allocate a partial Python
  circuit.

## Outcome and validation

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

## Code and ownership

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

## Acceptance

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

## Interfaces

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
