# Remove the legacy QuantumComputation-to-MLIR bridge

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, the MQT Compiler Collection no longer imports the legacy
`qc::QuantumComputation` circuit representation into MLIR. The compiler's C++
and Python APIs accept native compiler inputs such as OpenQASM, Qiskit circuits,
textual MLIR, and typed MLIR program objects, and the MLIR libraries no longer
link the legacy Core IR solely for that adapter. The direct Qiskit C-API bridge
from the preceding stack layer remains independent of the removed adapter.

The decision-diagram integration remains available. In particular,
`MLIRQCODDFunctionality` still constructs and simulates decision diagrams
directly from static QCO functions, and its tests may retain
`QuantumComputation` instances as independent reference oracles. This retained
test-only use does not translate a `QuantumComputation` to or from MLIR.

## Progress

- [x] (2026-08-11 09:36Z) Audit production code, public headers, CMake targets,
  bindings, generated stubs, and tests for every `QuantumComputation` and
  `MLIRQCTranslation` dependency.
- [x] (2026-08-11 09:54Z) Remove the public translator, compiler factory, Python
      entry points, dedicated translation tests, and legacy program fixtures.
- [x] (2026-08-11 09:54Z) Relink surviving compiler and OpenQASM targets
      directly to the native OpenQASM-to-QC translation library and prove that
      non-DD MLIR code no longer depends on `QuantumComputation` or
      `MQT::CoreIR`.
- [ ] Regenerate Python stubs, update the generic compiler-collection changelog
      entry with the new pull request number, and validate C++, Python, DD, and
      repository lint. The stubs, focused C++/Python/DD tests, targeted
      clang-tidy, and full repository lint are complete; only the changelog
      number remains.
- [ ] Publish the signed branch as the new middle pull request in the issue
  #1590 stack without merging it.

## Surprises & Discoveries

- Observation: the bridge is one-way despite its broad dependency footprint.
  The only production conversion is
  `translateQuantumComputationToQC`; there is no MLIR-to-`QuantumComputation`
  exporter. Evidence: the complete repository search finds the public factory
  in `mlir/include/mlir/Compiler/Programs.h`, its implementation in
  `mlir/lib/Compiler/Programs.cpp`, and the translator under
  `mlir/lib/Dialect/QC/Translation`, but no inverse API.
- Observation: the preceding stack layer replaces the former indirect Qiskit
  route with a direct, version-gated C-API bridge. It neither constructs a
  `QuantumComputation` nor links `MQT::CoreIR`, so it can remain while the
  representation bridge is removed.
- Observation: the QCO decision-diagram implementation uses lightweight Core
  gate identifiers and qubit indices but does not construct a
  `QuantumComputation`. Some DD-oriented tests deliberately compare QCO DDs to
  the established `QuantumComputation` DD simulator. These are retained as the
  requested DD boundary and independent oracle.
- Observation: the Qiskit helper-gate matrix test remains a useful direct-bridge
  regression and is retained unchanged.

## Decision Log

- Decision: Remove `QuantumComputation` inputs while retaining the direct Qiskit
  C-API inputs introduced by the preceding pull request. Rationale: the direct
  bridge imports and exports QC MLIR without using the legacy representation,
  so removing it would add an unrelated regression rather than reduce the
  coupling targeted here. Date/Author: 2026-08-11, Codex.
- Decision: Preserve the MLIR QC dialect and its OpenQASM translation library.
  Rationale: `QC` is the compiler's frontend dialect and is distinct from the
  legacy C++ `qc::QuantumComputation` class. OpenQASM and textual QC MLIR remain
  native compiler inputs. Date/Author: 2026-08-11, Codex.
- Decision: Preserve `MLIRQCODDFunctionality` and DD-based reference tests even
  where the oracle is expressed as a `QuantumComputation`. Rationale: the user
  explicitly requested that DD code remain, and those tests exercise MLIR-to-DD
  behavior without translating between the two circuit representations.
  Date/Author: 2026-08-11, Codex.

## Outcomes & Retrospective

The implementation removes the legacy translator, factory, fixtures, and Core
IR links while retaining the direct Qiskit integration and DD functionality.
Validation and publication must be refreshed after the stack rebase.

## Context and Orientation

`qc::QuantumComputation` is the legacy circuit class implemented under
`include/mqt-core/ir` and `src/ir`. The MLIR QC dialect is a separate compiler
intermediate representation under `mlir/include/mlir/Dialect/QC` and
`mlir/lib/Dialect/QC`. The target named `MLIRQCTranslation` currently connects
those two representations through `TranslateQuantumComputationToQC.cpp`. By
contrast, `MLIRQCOpenQASMTranslation` imports OpenQASM directly into the QC
dialect and must remain.

`mlir/include/mlir/Compiler/Programs.h` exposes typed compiler program classes.
`QCProgram::fromQuantumComputation` wraps the translator, while the direct
Qiskit C-API adapter reaches QC MLIR independently. The dedicated translator
tests and the large
`mlir/unittests/programs/quantum_computation_programs.*` fixture set exist only
for this bridge.

The DD boundary lives in `mlir/include/mlir/Dialect/QCO/Utils/DDFunctionality.h`
and `mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp`. It maps static QCO
operations directly to the DD package. Tests under `mlir/unittests/Dialect/QCO`
may continue to use the older circuit simulator as an independent result oracle.

## Plan of Work

Delete the public translator header and implementation and remove the
`MLIRQCTranslation` CMake target. Relink the compiler pipeline, `mqt-cc`, and
the OpenQASM test target to `MLIRQCOpenQASMTranslation`, which provides the
native functions they actually use. Remove `QCProgram::fromQuantumComputation`
from the public compiler API and implementation.

In the nanobind module, remove the Core IR include and link dependency, the
`from_quantum_computation` factory, generic `QuantumComputation` input handling,
and the now-unneeded `mqt.core.ir` import. Preserve `from_qiskit`, `to_qiskit`,
and generic Qiskit input handling. Update the binding documentation and
stub-generation patterns accordingly. Replace Python tests that used a legacy
circuit merely to reach QCO with equivalent OpenQASM input.

Delete the dedicated translator test source and its CMake dependencies. Delete
the `quantum_computation_programs` fixture library. Simplify the compiler
pipeline parameterization to native QC builders only and remove regression tests
that existed solely to verify the deleted translator. Keep all DD files and
their tests unchanged except for build adjustments proven necessary.

Regenerate `python/mqt/core/mlir.pyi` through the repository stub session. Fold
the user-facing removal into the existing generic MQT Compiler Collection
changelog entry after GitHub assigns the new pull request number; do not create
a separate changelog bullet.

## Concrete Steps

All commands run from the repository root. Cache-producing commands use the
worktree-local wrapper.

First edit the source, CMake, bindings, and tests, then verify that only the DD
boundary still mentions the legacy class:

    rg -n 'QuantumComputation|MLIRQCTranslation|MLIRQuantumComputationPrograms' mlir bindings/mlir test/python/test_mlir.py

Configure and build the release preset with LLVM/MLIR 22, then run the focused
compiler, OpenQASM, and QCO DD test binaries:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target mqt-core-mlir-unittests-compiler mqt-core-mlir-unittest-openqasm-target mqt-core-mlir-unittest-qco-utils
    ./.agent/run.sh build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ./.agent/run.sh build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target

Build the bindings, regenerate stubs, and run the focused Python tests using the
repository's established Nox and pytest workflows. Finish with:

    ./.agent/run.sh uvx nox --non-interactive -s lint

## Validation and Acceptance

The structural contract is accepted when no installed MLIR header, compiler
library, MLIR binding, `mqt-cc`, or non-DD MLIR test references
`QuantumComputation`, `TranslateQuantumComputationToQC`, `MLIRQCTranslation`, or
`MLIRQuantumComputationPrograms`. CMake must no longer define those two legacy
targets, and the MLIR Python module must not link `MQT::CoreIR`.

OpenQASM strings and files, Qiskit circuits, textual QC MLIR, and typed compiler
programs must still traverse the default compiler pipeline. Supplying a legacy
`QuantumComputation` must raise the established unsupported-program error. The
generated stub must retain Qiskit types while removing only the legacy circuit
type.

The direct QCO DD construction and simulation tests must still pass. This proves
that keeping DD code did not accidentally retain the deleted representation
translator and that the useful MLIR-to-DD integration remains available.

## Idempotence and Recovery

Search, configure, build, test, stub generation, and lint commands are safe to
repeat. Deletions are tracked by Git and remain recoverable from the branch
parent until committed. If a surviving target still requests
`MLIRQCTranslation`, identify whether it uses OpenQASM translation and link
`MLIRQCOpenQASMTranslation` directly; do not recreate a compatibility alias. If
generated stubs differ beyond `mqt.core.mlir`, inspect the build environment
before committing and preserve unrelated generated modules.

## Artifacts and Notes

The initial dependency audit found that non-DD production coupling consists of
one public translator, one compiler factory, four Python input surfaces, and
their CMake links. The DD implementation remains isolated in
`MLIRQCODDFunctionality` and links `MQT::CoreDD`, as requested.

## Interfaces and Dependencies

At completion, `QCProgram` retains `fromMLIRString`, `fromMLIRFile`,
`fromQASMString`, and `fromQASMFile`; it no longer declares
`fromQuantumComputation`; it also retains the direct `from_qiskit` and
`to_qiskit` binding methods. `compile_program` accepts strings, paths, Qiskit
circuits, `QCProgram`, `QCOProgram`, `JeffProgram`, and `OpenQASMProgram` in
Python. The compiler pipeline and tools depend directly on
`MLIRQCOpenQASMTranslation`.

`MLIRQCODDFunctionality` continues to expose
`mlir::qco::buildFunctionality(func::FuncOp, dd::Package&)` and
`mlir::qco::simulate(func::FuncOp, const dd::VectorDD&, dd::Package&)` with no
public API change.

Revision note (2026-08-11): created this focused middle-stack plan after the
complete dependency audit and the user's decision to remove the bridge while
retaining DD code.

Revision note (2026-08-11): rebased after the direct Qiskit C-API bridge and
updated the scope so only the legacy `QuantumComputation` interaction is
removed.
