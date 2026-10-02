# Add Python bindings for the MQT MLIR compiler collection

Status: historical implementation record.

## Goal and scope

Before this change, the MQT MLIR compiler collection (`mqt-cc`) was available
only through C++ and the command line. Python users can now import
`mqt.core.mlir`, construct typed compiler programs from source, supported files,
MQT Core circuits, Qiskit circuits, and Qiskit subclasses, and compile to QC,
QCO, Jeff, or QIR. They can also run selected QCO transformations and serialize
Jeff or QIR results.

The observable end-to-end proof is `test/python/test_mlir.py`. It compiles a
Bell circuit from the supported Python input forms, exercises typed conversion,
Jeff serialization, QIR output, and Qiskit subclass support.

## Constraints

- A `module` prefix can name either inline MLIR or a file such as `module.mlir`.
  Evidence: the broad original prefix test sent a valid path to the parser. The
  final code requires a whitespace token boundary after `module`, tolerates
  leading whitespace, and validates actual file extensions.

- A user-defined Qiskit subclass has a user module such as `__main__`, not a
  `qiskit.*` module. Evidence: the initial regression raised
  `Program type CustomQuantumCircuit is not supported.` The final code checks an
  already loaded Qiskit circuit module and uses Python `isinstance`.

- MLIR can be syntactically valid but belong to the wrong typed dialect.
  Evidence: QC input could previously become a `QCOProgram`. The typed parser
  now rejects it and emits the requested-dialect diagnostic.

- A chained `uv sync` may not visibly print a following command. Evidence:
  inspecting the installed extension timestamp and rerunning focused tests was
  required to confirm a native binding rebuild.

## Decisions

- Use typed `QCProgram`, `QCOProgram`, `JeffProgram`, and `QIRProgram` wrappers
  instead of the older untyped pipeline API. Rationale: compiler-stage ownership
  and legal dialect transitions become explicit.

- Use nanobind rather than official MLIR Python bindings. Rationale: nanobind is
  already used by the project; official bindings are separately explored under
  issue #1693.

- Define Python API documentation in `bindings/mlir/register_mlir.cpp` with
  `R"pb(...)pb"` strings and regenerate the stub. Rationale: `.pyi` files are
  generated and the binding documentation serves runtime help and Python API
  documentation.

- Keep `--pass-pipeline` wording in mqt-cc. Rationale: it is MLIR's real option;
  changing it to the suggested `--passes` would be misleading.

- Keep `mqt-core-mlir-bindings` in `pyproject.toml` build targets. Rationale:
  package builds must explicitly create the extension; removing it prevented the
  editable extension rebuild.

## Outcome and validation

The merged feature delivers a documented Python API for the MQT MLIR compiler,
including format-selecting `compile_program`, custom QCO pipelines, Jeff
serialization, QIR emission, documentation, and packaging support. The review
pass made its boundary safer: it distinguishes paths from source, preserves
native paths, rejects wrong-dialect MLIR with diagnostics, registers standard
MLIR pass names, and accepts Qiskit subclasses.

No work remains for #1815. Exploring official MLIR Python bindings under #1693
is separate work and requires a new plan.

## Code and ownership

Scope: MLIR compiler bindings, their Python package and tests, compiler
documentation, and package build configuration.

The public module implementation is `bindings/mlir/register_mlir.cpp`. nanobind
is the C++ library that exposes C++ APIs to Python. The typed program
declarations and implementation are in `mlir/include/mlir/Compiler/Programs.h`
and `mlir/lib/Compiler/Programs.cpp`. `mlir/lib/Support/Passes.cpp` registers
textual pipelines. `mlir/tools/mqt-cc/mqt-cc.cpp` implements the CLI.

`bindings/patterns.txt` supplies nanobind stub-generation patterns, while
`python/mqt/core/mlir.pyi` is generated output. The focused Python and C++ test
files are `test/python/test_mlir.py` and
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`. User documentation is
`docs/mlir/python_compiler_collection.md`.

## Acceptance

`compile_program` accepts Bell OpenQASM, equivalent MLIR, a Jeff file, an MQT
Core circuit, a Qiskit circuit, and a Qiskit subclass. Requested output formats
return their typed program. `QCOProgram.run_pass_pipeline` accepts both the MQT
pipeline and standard `canonicalize,cse` passes. Unsupported files raise a
Python `RuntimeError`; invalid or wrong-dialect MLIR fails rather than crashing.
The focused C++ and Python suites cover these outcomes. Lint is mandatory after
every batch and passed at completion.

## Interfaces

The public module is `mqt.core.mlir`. Its primary types are `OutputFormat`,
`QIRProfile`, `Program`, `QCProgram`, `QCOProgram`, `JeffProgram`, and
`QIRProgram`. The `compile_program` function selects an output format and
returns the corresponding typed program. Its keyword arguments are `output`,
`inplace`, `qco_pipeline`, `enable_timing`, and `enable_statistics`.

The binding depends on nanobind, MQT Core `QuantumComputation`, and the MQT MLIR
compiler libraries. Qiskit is optional: unrelated inputs must not import it, but
loaded Qiskit circuits and subclasses are recognized by Python `isinstance`.
Stub generation uses `nanobind.stubgen` with `bindings/patterns.txt`; consumers
use the generated stub rather than editing it.
