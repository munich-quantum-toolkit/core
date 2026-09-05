# Split compiler programs from pipeline orchestration

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The DDSIM QDMI device accepts OpenQASM by translating it to the QC dialect and
then to the QCO dialect before simulation. It currently links the complete
compiler pipeline library for those two conversions. After this change, DDSIM
links a focused compiler-program library and does not depend on target
compilation, QIR or jeff conversion, or default-pipeline orchestration.

The public compiler API and all runtime behavior remain unchanged. The result is
visible in the generated DDSIM link command: it contains `MQTCompilerPrograms`
and no `MQTCompilerPipeline`. Existing OpenQASM DDSIM, compiler, and
command-line tests continue to pass.

## Progress

- [x] (2026-09-03 05:00 CEST) Read issue #2329, the originating #2288 review
      thread, repository policy, and the current compiler and DDSIM dependency
      paths.
- [x] (2026-09-03 05:00 CEST) Checked open compiler work. PRs #2339 and #2340
      change `Programs.cpp`, but do not implement this boundary and have no
      relevant unresolved review comments.
- [x] (2026-09-03 07:08 CEST) Added the focused compiler-program target and
      split pass-backed and output functionality into the higher-level pipeline
      source without changing the public header.
- [x] (2026-09-03 07:08 CEST) Linked DDSIM only to the compiler-program target
      and verified that its generated link command excludes the compiler
      pipeline, compiler target, QCO transforms, and jeff and QIR converters.
- [x] (2026-09-03 07:08 CEST) Built every affected non-unity target and passed
      148 compiler tests, 62 DDSIM device tests, two `mqt-cc` tests, repository
      lint, C++ lint with zero findings, and `git diff --check`.

## Surprises & Discoveries

- Observation: `mlir/lib/Compiler/Programs.cpp` combines program storage and
  parsing with QC and QCO transformations, target compilation, jeff and QIR
  conversion, and the default pipeline. A new target around the unchanged file
  would retain all unwanted dependencies.
- Observation: Unity builds can combine every source of one target into a single
  object. Splitting the source without splitting the CMake target would not
  establish the requested link boundary.
- Observation: MQT compiler MLIR libraries are build-tree targets rather than
  entries in the installed `MQT_CORE_TARGETS` export. This issue therefore does
  not need a new installed alias or package component.
- Observation: PRs #2339 and #2340 add inliner preparation to both the compiler
  context factory that remains in `Programs.cpp` and higher-level methods that
  move to `Pipeline.cpp`. Restacking #2340 must preserve registration on both
  sides of the target boundary and link the inliner-extension libraries to each
  target that performs it.
- Observation: The lower target still links the jeff dialect because the shared
  compiler context registers that dialect. It does not link jeff conversion or
  translation libraries.

## Decision Log

- Decision: Add `MQTCompilerPrograms` below `MQTCompilerPipeline`. The lower
  target owns program storage, parsing, validation, copying, inspection, and the
  OpenQASM-to-QC-to-QCO path. Rationale: These are the exact facilities DDSIM
  uses, and they do not need target compilation or output conversion.
  Date/Author: 2026-09-03 / Codex.
- Decision: Keep `MQTCompilerPipeline` as a public umbrella that links
  `MQTCompilerPrograms`. Rationale: Existing bindings, tools, benchmarks, and
  tests keep their current target dependency and source API. Date/Author:
  2026-09-03 / Codex.
- Decision: Keep `mlir/Compiler/Programs.h` as the shared public header. Do not
  split the C++ type hierarchy or add a device-specific parsing API. Rationale:
  The requested boundary is a link dependency, not a new user-facing concept.
  Date/Author: 2026-09-03 / Codex.
- Decision: Put pass-backed cleanup, optimization, output conversion, target
  compilation, and default orchestration in one new `Pipeline.cpp`. Rationale:
  One higher-level source is sufficient; additional internal layers would not
  reduce dependencies or improve the contract. Date/Author: 2026-09-03 / Codex.
- Decision: Do not add changelog or upgrade-guide text. Rationale: This internal
  refactor preserves behavior and changes no released API. Date/Author:
  2026-09-03 / Codex.

## Outcomes & Retrospective

The split establishes the requested link boundary while retaining the existing
compiler API. The generated DDSIM link command includes
`libMQTCompilerPrograms.a` and excludes `libMQTCompilerPipeline.a`,
`libMQTCompilerTarget.a`, QCO transforms, jeff converters, and QC-to-QIR
converters.

All affected non-unity targets build. The compiler unit suite passes 148 tests,
the complete DDSIM device suite passes 62 tests, and the two `mqt-cc` CTests
pass. Repository lint and `git diff --check` also pass. No new behavior test was
needed: the existing compiler and DDSIM suites directly exercise the moved
symbols and the device path whose dependency was narrowed. C++ lint reports zero
clang-format and zero clang-tidy findings for both changed sources.

When #2339 and #2340 are restacked, their higher-level additions belong in
`Pipeline.cpp`. Inliner registration from #2340 must remain available both to
the factory in `Programs.cpp` and to higher-level paths that accept caller-owned
contexts; one file-local helper cannot serve both targets.

## Context and Orientation

`mlir/include/mlir/Compiler/Programs.h` declares owned program types for QC,
QCO, jeff, QIR, and OpenQASM. `mlir/lib/Compiler/Programs.cpp` currently
implements all of those types and `runDefaultPipeline`. The latter is the
high-level operation that coordinates optimization, target compilation, and
output conversion.

`mlir/lib/Compiler/TargetCompilation.cpp` builds the target-specific QCO pass
pipeline. `mlir/lib/Compiler/CMakeLists.txt` currently compiles both sources
into `MQTCompilerPipeline` and attaches `Programs.h` and `TargetCompilation.h`
to that target.

`src/qdmi/devices/dd/Device.cpp` translates QASM in `parseQASMToQCO` by calling
`QCProgram::fromQASMString` and then `QCProgram::intoQCO`. The device only
borrows the resulting QCO module for DD sampling or statevector simulation.
Nevertheless, `src/qdmi/devices/dd/CMakeLists.txt` links `MQTCompilerPipeline`,
which exposes the device to every higher-level compiler dependency.

A CMake target is a named group of sources and link requirements. A public link
from `MQTCompilerPipeline` to `MQTCompilerPrograms` means existing pipeline
consumers also receive the lower target. A private DDSIM link to
`MQTCompilerPrograms` gives the device only the narrower dependency closure.

## Plan of Work

First, update `mlir/lib/Compiler/CMakeLists.txt`. Add `MQTCompilerPrograms` with
`Programs.cpp`, the `Programs.h` file set, and only the dialect, parser,
pass-manager, OpenQASM import, and QC-to-QCO conversion libraries used by that
source. Change `MQTCompilerPipeline` to compile `Pipeline.cpp` and
`TargetCompilation.cpp`, link `MQTCompilerPrograms` publicly, and retain the
dependencies used by the higher-level methods.

Second, trim `Programs.cpp` to the foundational implementation. Retain compiler
context creation, source and module parsing, program storage, OpenQASM source
I/O, QC and QCO construction and copying, QC gate counts, QCO linearity checks,
and `QCProgram::intoQCO`. Implement that one conversion with its own small pass
manager so the lower source does not need an internal cross-library helper.

Move the remaining definitions without changing their bodies into a new
`mlir/lib/Compiler/Pipeline.cpp`: QC cleanup, phase normalization, OpenQASM and
QIR output; all pass-backed QCO methods; QC and jeff output conversions; jeff
and QIR program methods; LLVM IR and bitcode output; and `runDefaultPipeline`.
Keep shared pass helpers private to that source.

Third, replace `MQTCompilerPipeline` with `MQTCompilerPrograms` in
`src/qdmi/devices/dd/CMakeLists.txt`. Leave all other consumers on the umbrella
pipeline target. Inspect generated link commands to prove that the device does
not regain the pipeline transitively.

## Concrete Steps

Run all commands from the repository root. Configure and build the affected
non-unity targets with:

    cmake --preset release -DCMAKE_UNITY_BUILD=OFF
    cmake --build --preset release --target MQTCompilerPrograms
    cmake --build --preset release --target MQTCompilerPipeline
    cmake --build --preset release --target mqt-core-qdmi-ddsim-device
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    cmake --build --preset release --target mqt-cc

Run the focused and umbrella behavior checks with:

    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ./build/release/test/qdmi/devices/dd/mqt-core-qdmi-ddsim-device-test \
      --gtest_filter='HistogramTest.QASM2Program:HistogramTest.QASM3Program'
    ctest --test-dir build/release -R mqt-cc --output-on-failure

Inspect the generated DDSIM link command with Ninja's command tool. The output
must contain the compiler-program archive and must not contain the compiler
pipeline archive:

    ninja -C build/release -t commands mqt-core-qdmi-ddsim-device

Finish with:

    git diff --check
    uvx nox -s cpp-lint
    uvx nox -s lint

## Validation and Acceptance

The project must build with unity disabled so no accidental same-target source
aggregation can hide the boundary. `MQTCompilerPrograms`, `MQTCompilerPipeline`,
DDSIM, the compiler unit test, and `mqt-cc` must all link.

The DDSIM command closure must contain `MQTCompilerPrograms` and exclude
`MQTCompilerPipeline`. The compiler unit tests must still cover program
construction, conversions, pass methods, target compilation, and default
orchestration through the umbrella target. QASM 2 and QASM 3 jobs submitted to
DDSIM must retain their previous results. The `mqt-cc` tests must retain the
driver behavior.

Cpp-linter must report zero clang-format and clang-tidy findings for changed C++
files. The complete repository lint suite must pass. No generated, installed, or
public API file should change except the CMake ownership of the existing header.

## Idempotence and Recovery

The source move, configuration, builds, tests, and link-command inspection are
repeatable. The work is isolated from the branches for #2330, #2331, and #2332.
If #2339 or #2340 lands first, rebase and place their higher-level additions in
`Pipeline.cpp`; keep factory-side inliner registration in `Programs.cpp` and
link its required extension library to the lower target. Do not restore the
combined source or broaden DDSIM's dependency.

## Artifacts and Notes

The originating #2288 review thread is resolved. It asks for a Programs library
instead of a device dependency on the complete compiler pipeline. Issue #2329
has no comments. No open PR already implements this split.

## Interfaces and Dependencies

`MQTCompilerPrograms` must provide the existing symbols used by the DDSIM path:

    mlir::QCProgram::fromQASMString(std::string_view)
    mlir::QCProgram::intoQCO() &&
    mlir::Program::module() const

It must not link `MQTCompilerPipeline`, `MQTCompilerTarget`, QCO transform
libraries, QIR conversion libraries, or jeff conversion libraries.
`MQTCompilerPipeline` must link `MQTCompilerPrograms` publicly and continue to
provide every existing program and pipeline symbol to its current consumers.
