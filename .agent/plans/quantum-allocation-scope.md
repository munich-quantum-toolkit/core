# Quantum allocation scope

Status: complete; validated locally.

## Goal and scope

Dynamic quantum allocations in QC/QCO programs belong in the entry block of the
function marked `mqt.entry_point`. Helper functions receive quantum resources as
arguments. This covers `qc.alloc`, `qco.alloc`, qubit `memref.alloc`, and
`qtensor.alloc`. Classical allocations and static qubit references are
unchanged.

## Decisions

The MQT entry-point attribute verifier owns the whole-program rule. It already
checks module-level entry-point uniqueness and can inspect all four allocation
forms without extending an upstream operation. QC/QCO program construction loads
the MQT verifier even for caller-supplied contexts and checks modules without an
entry marker. Raw unmarked MLIR fragments can be verified independently;
operation verification alone does not establish this program-wide invariant.

Builders reject invalid allocation placement before creating an operation.
OpenQASM semantic analysis already rejects non-global qubit declarations, and
loop emission restores the entry-block insertion point for later declarations.
The Adaptive conversion no longer scans allocation placement; Mapping discovers
allocations directly in the entry block.

Tests cover all four allocation forms, allowed and forbidden placement, missing
entry markers, caller-supplied contexts, builders, and frontend loop emission.
Pass tests use valid quantum-resource arguments or static references where the
behavior under test does not require allocation.

## Validation

With LLVM/MLIR 23.1.0, the full lint-preset build and all 2,358 configured MLIR
CTest entries passed, including the verifier, compiler, builder, and frontend
regressions. Commands from the repository root:

- `uvx nox -s lint`
- `uvx nox -s cpp-lint -- ec799daa09f855bd0edcbc5592a5fedd90836516`
- `ctest --test-dir build/cpp-lint -L mqt-mlir-unittests --output-on-failure -j8`

Full changed-file C++ lint passed with local clang-tidy 23.0.0git and the macOS
SDK headers configured. Hosted CI was not run for this local revision.
