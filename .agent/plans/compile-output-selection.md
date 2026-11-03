# Compile output selection

Status: complete; local validation passed, hosted CI remains separate.

## Contract

`compile_program` selects the returned representation independently of the
hardware target. Explicit `output` returns its typed program. The default
preserves the input family without a target or with a hardware-only
`CompilerTarget`; Qiskit targets select `QuantumCircuit`, and QDMI targets
select a submission-ready `CompiledProgram`. `output` is the sole compiler
format selector. Exact QDMI transport selection belongs to
`submit_program(..., program_format=...)`; automatic precompilation retains the
selected native payload for repeated submissions.

Python does not offer exact-wire-format precompilation into `CompiledProgram`.
Use automatic reusable precompilation, a typed export, or exact-format
compile-and-submit. The native environment-based API retains offline payload
compilation.

Source and path inputs return program objects. Raw QC import and QCO are
pre-compilation checkpoints and cannot be combined with a target. jeff does not
retain physical placement. Supported exporter subsets remain unchanged.

## Implementation and validation

The bindings retain frontend format identity, resolve target and output
constraints, and call the shared native pipeline. QC output retains layout by
deferring resource-changing export cleanup. Capabilities are restricted by both
the output representation and the target's known contract. Different formats
share structural control-flow constraints; matching language versions and
profiles also share format-specific capabilities, independently of text or
binary encoding. QC-to-QIR conversion calls the `Target/QIR` capability
validator against any attached contract for the emitted QIR version and profile,
including staged conversion and recompilation without an explicit target. The
validator and its flag mapping live together in the existing QIR translation
library; it accepts capability values without linking the compiler target model.
Qiskit export receives the complete environment when compiling, and retains a
target-only overload for direct conversions. All 11 supported capability IDs and
three structural limits are covered. Required QIR loop capabilities must be
unconstrained because final lowering does not retain a proof of their bounds.
The compiler owns format baselines in `PayloadSpecification::forProgramFormat`;
Python output selection and QIR preparation use it directly. QDMI maps wire
formats and encodings onto those baselines. QIR preparation follows the
requested output, independently of any attached target contract. QDMI 1.4 has no
detailed per-format capability descriptors; its device snapshots use format
baselines.

`bindings/patterns.txt` uses literal output overloads and a constrained type
variable for same-family defaults. Existing Python tests in
`test/python/test_mlir*.py` and `test/python/qdmi/test_compilation.py` cover
compilation, simulation, submission, ownership, layout, parameters, phase, and
capability limits. The native payload test checks the capability intersection
directly. These cover representative combinations without an input/target/output
Cartesian product. Tests use direct simulation, export, submission, and module
inspection; serialization, ownership, phase, and layout checks retain their
distinct oracles. Qiskit submission tests live with the Qiskit integration
tests; the QDMI tests need neither a package initializer nor a pytest
Python-path setting. Simulation APIs retain their direct QCO path and inline the
fixed conversion for other inputs.

Validation on main revision `a7953651a7`: 4,045 native tests with
`ctest --preset release-clang-ipo`; 1,471 MLIR, QDMI, and
magic-state-distillation Python tests with Qiskit 2.6.0; and 559 translation,
target, and parameter-binding tests with Qiskit 2.5.2. Generated stubs,
whole-file C++ lint, repository lint, and executable documentation with internal
links passed. The output-selection page was inspected in Chrome. Native
regressions cover staged QIR conversion, targetless recompilation, mismatched
payload contracts, and Base loop preparation despite Adaptive target metadata.
