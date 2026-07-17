# Add Python bindings for the MQT MLIR compiler collection

This ExecPlan is a completed retrospective record for pull request #1815, merged
as commit `44946b1b35eb35060bd884d6ced61a6e643edb3a`. It is maintained according
to `.agent/PLANS.md` and describes the completed work so that a new contributor
can understand and validate it without the original pull request.

## Purpose / Big Picture

Before this change, the MQT MLIR compiler collection (`mqt-cc`) was available
only through C++ and the command line. Python users can now import
`mqt.core.mlir`, construct typed compiler programs from source, supported files,
MQT Core circuits, Qiskit circuits, and Qiskit subclasses, and compile to QC,
QCO, Jeff, or QIR. They can also run selected QCO transformations and serialize
Jeff or QIR results.

The observable end-to-end proof is `test/python/test_mlir.py`. It compiles a
Bell circuit from the supported Python input forms, exercises typed conversion,
Jeff serialization, QIR output, and Qiskit subclass support.

## Progress

- [x] (2026-06-26 to 2026-07-15) Implemented typed compiler programs, the Python
      binding module, package integration, documentation, and tests; merged
      through pull request #1815.
- [x] (2026-07-15) Reviewed CodeRabbit feedback and incorporated valid input,
  type-validation, CLI, pass-registration, typing, and documentation fixes.
- [x] (2026-07-15) Added binding-owned Python docstrings and regenerated stubs.
- [x] (2026-07-15) Ran focused C++ and Python validation plus required lint.
- [x] (2026-07-15) Recorded this retrospective ExecPlan after the pull request
  merged.

## Surprises & Discoveries

- Observation: A `module` prefix can name either inline MLIR or a file such as
  `module.mlir`. Evidence: the broad original prefix test sent a valid path to
  the parser. The final code requires a whitespace token boundary after
  `module`, tolerates leading whitespace, and validates actual file extensions.
- Observation: A user-defined Qiskit subclass has a user module such as
  `__main__`, not a `qiskit.*` module. Evidence: the initial regression raised
  `Program type CustomQuantumCircuit is not supported.` The final code checks an
  already loaded Qiskit circuit module and uses Python `isinstance`.
- Observation: MLIR can be syntactically valid but belong to the wrong typed
  dialect. Evidence: QC input could previously become a `QCOProgram`. The typed
  parser now rejects it and emits the requested-dialect diagnostic.
- Observation: A chained `uv sync` may not visibly print a following command.
  Evidence: inspecting the installed extension timestamp and rerunning focused
  tests was required to confirm a native binding rebuild.

## Decision Log

- Decision: Use typed `QCProgram`, `QCOProgram`, `JeffProgram`, and `QIRProgram`
  wrappers instead of the older untyped pipeline API. Rationale: compiler-stage
  ownership and legal dialect transitions become explicit. Date/Author: PR #1815
  contributors, 2026-06-26 to 2026-07-15.
- Decision: Use nanobind rather than official MLIR Python bindings. Rationale:
  nanobind is already used by the project; official bindings are separately
  explored under issue #1693. Date/Author: PR #1815.
- Decision: Define Python API documentation in `bindings/mlir/register_mlir.cpp`
  with `R"pb(...)pb"` strings and regenerate the stub. Rationale: `.pyi` files
  are generated and the binding documentation serves runtime help and Python API
  documentation. Date/Author: 2026-07-15.
- Decision: Keep `--pass-pipeline` wording in mqt-cc. Rationale: it is MLIR's
  real option; changing it to the suggested `--passes` would be misleading.
  Date/Author: 2026-07-15.
- Decision: Keep `mqt-core-mlir-bindings` in `pyproject.toml` build targets.
  Rationale: package builds must explicitly create the extension; removing it
  prevented the editable extension rebuild. Date/Author: 2026-07-15.

## Outcomes & Retrospective

The merged feature delivers a documented Python API for the MQT MLIR compiler,
including format-selecting `compile_program`, custom QCO pipelines, Jeff
serialization, QIR emission, documentation, and packaging support. The review
pass made its boundary safer: it distinguishes paths from source, preserves
native paths, rejects wrong-dialect MLIR with diagnostics, registers standard
MLIR pass names, and accepts Qiskit subclasses.

No work remains for #1815. Exploring official MLIR Python bindings under #1693
is separate work and requires a new plan.

## Context and Orientation

This plan covers the repository-relative MLIR compiler-binding scope:
`bindings/`, `mlir/`, `python/mqt/core/`, `test/python/`, `docs/mlir/`, and the
relevant package and CI configuration. It must preserve unrelated user changes
and never modify another task's worktree. Repository rules come from `AGENTS.md`
and `docs/ai_usage.md`; generated `.pyi` files are never manually edited, and
this plan does not authorize external GitHub actions.

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

## Plan of Work

The completed implementation introduced explicit wrappers for each compiler
stage. QC is frontend MLIR, QCO is optimization MLIR, Jeff is the serializable
format, and QIR is the LLVM-compatible quantum representation. Every wrapper
owns an MLIR module and exposes only meaningful transitions.

The binding exposes these wrappers and `compile_program`. Inputs may be source,
`.mlir`, `.qasm`, or `.jeff` paths, MQT Core `QuantumComputation`, Qiskit
circuits, or typed programs. Constructors verify that parsed MLIR uses the
requested QC or QCO dialect. The review work added exact source recognition,
native path handling, subclass-safe Qiskit detection, diagnostic errors,
standard pass registration, `.qco` CLI auto-detection, and early custom-pipeline
failure. It also added `QC_IMPORT` overloads and complete binding docstrings.

## Concrete Steps

Run all commands from the repository root.

1. Configure MLIR-capable debug builds when necessary:

       cmake --preset debug

2. Build and run the focused C++ compiler test:

       cmake --build --preset debug --target mqt-core-mlir-unittests-compiler
       ./build/debug/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
         --gtest_filter='CompilerPipelineTest.TypedProgramImportsAndCopies:CompilerPipelineTest.TypedProgramsComposeWithoutImplicitCopies'

   Deliberately invalid input prints diagnostics but the tests pass.

3. Rebuild the editable extension and run Python acceptance tests:

       uv sync --inexact --no-dev --no-build-isolation-package mqt-core \
         --reinstall-package mqt-core
       uv run --no-sync pytest test/python/test_mlir.py

   The final observed result was `22 passed`.

4. Regenerate stubs after binding changes:

       uvx nox -s stubs

5. Run the final repository gate:

       uvx nox -s lint

   The final observed Nox lint session succeeded.

## Validation and Acceptance

`compile_program` accepts Bell OpenQASM, equivalent MLIR, a Jeff file, an MQT
Core circuit, a Qiskit circuit, and a Qiskit subclass. Requested output formats
return their typed program. `QCOProgram.run_pass_pipeline` accepts both the MQT
pipeline and standard `canonicalize,cse` passes. Unsupported files raise a
Python `RuntimeError`; invalid or wrong-dialect MLIR fails rather than crashing.
The C++ and Python commands above prove these outcomes. Lint is mandatory after
every batch and passed at completion.

## Idempotence and Recovery

Build, test, stub, and lint commands are repeatable. If Python appears to load
an old extension after changing a binding, rerun the `uv sync` command with
`--reinstall-package mqt-core`, then rerun the focused test. If CMake cannot
find MLIR, configure in an environment exposing LLVM/MLIR 22.1 or newer.

This is a completed retrospective plan. Do not replay its historical edits on
`main`; create a separately owned plan for any new follow-up work.

## Artifacts and Notes

The merged PR commit is:

    44946b1b3 ✨ Add Python bindings for `mqt-cc` (#1815)

The post-implementation review refinements are included in the merged result:
binding docstrings, input classification, wrong-dialect diagnostics, and Qiskit
subclass support. The tracked source and test files named in this plan are the
authoritative evidence; no checkout-specific branch history is required.

## Interfaces and Dependencies

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

Revision note (2026-07-15): created after #1815 merged to record the completed
implementation and review decisions under the ExecPlan policy added by #1907.
Revised after #1908 to remove checkout-specific metadata so the plan works from
any clone.
