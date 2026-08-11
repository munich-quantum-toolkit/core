# Build the MQT MLIR compiler without C++ exceptions by default

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds. It is maintained in accordance with
`.agent/PLANS.md`.

## Purpose / Big Picture

MQT's MLIR libraries and `mqt-cc` should follow the supported LLVM/MLIR
toolchain's exception-disabled default without making the compiler collection
conditional. The few targets that compile exception-using FoMaC, DD, or Core QIR
interfaces remain explicit exception boundaries. Failures entering the compiler
through FoMaC are translated to `llvm::Error` before they reach the
exception-disabled driver.

The direct Qiskit bridge introduced by pull request #2031 is intentionally not
part of the legacy `QuantumComputation` to MLIR surface removed by #2054. Its
Python binding is an independent exception-using boundary and remains available.

## Progress

- [x] (2026-08-11) Build the fixed-width angle, explicit compiler-error, and
      OpenQASM error-propagation prerequisites as pull requests #2040, #2049,
      and #2050.
- [x] (2026-08-11) Insert #2031 above #2050 and preserve its direct Qiskit C API
      bridge while linearizing its history.
- [x] (2026-08-11) Rework #2054 above #2031 so it removes only the legacy
      `QuantumComputation` to MLIR conversions, translations, program inputs,
      Python entry points, and fixtures. Independent DD and direct Qiskit
      functionality remain.
- [x] (2026-08-11) Keep the complete compiler graph and use LLVM's normal
      `llvm_update_compile_flags` policy instead of a user option, directory
      generator expression, or custom exception target property.
- [x] (2026-08-11) Keep only three production exception boundaries: the FoMaC
      adapter, DD functionality, and the standalone QIR runner.
- [x] (2026-08-11) Prove that seven test executables also require exceptions
      because the Core IR and DD headers compiled into their translation units
      contain throwing inline code. Scope those exceptions to the test targets.
- [x] (2026-08-11) Build the affected graph from a fresh LLVM/MLIR 22.1.3 cache
      and pass all 990 MLIR unit tests in that build.
- [ ] Commit and publish the simplified #2051 revision above #2054.
- [ ] Rebuild and simplify #2048 above #2051, then validate the combined
      exception-disabled and RTTI-disabled graph with the portable toolchain.
- [ ] Inspect replacement CI at every exact published head and resolve only
      outdated or source-addressed review conversations.

## Surprises & Discoveries

- Observation: LLVM already provides the required target policy through
  `llvm_update_compile_flags`. The earlier directory-wide generator expression,
  custom `MQT_MLIR_REQUIRES_EXCEPTIONS` property, and manual `-fexceptions`
  flags duplicated that policy and made generated object targets and Visual
  Studio generators harder to handle.
- Observation: moving FoMaC device opening and exception translation into
  `MQTCompilerFoMaCAdapter` lets `mqt-cc` remain exception-disabled while
  preserving contextual LLVM errors.
- Observation: the seven affected test executables are not merely tests of the
  exception boundaries. Their own translation units instantiate throwing code
  from `IfElseOperation.hpp`, `StandardOperation.hpp`, and DD root-management
  headers, so removing their exception requirement produces compile errors.
- Observation: `add_llvm_tool` also applies LLVM's exception policy outside the
  MLIR subtree. The QIR runner compiles existing exception-based Core IR and DD
  interfaces and must declare `LLVM_REQUIRES_EH` around its target creation.
- Observation: the direct Qiskit bridge's nanobind translation unit uses
  exceptions independently of the removed legacy circuit bridge. It remains a
  narrow binding boundary rather than becoming part of the MLIR library or
  driver policy.

## Decision Log

- Decision: retain the one-way stack #2040, #2049, #2050, #2031, #2054, #2051,
  #2048. Rationale: each layer has a separately reviewable contract, while
  placing #2031 before #2054 allows the cleanup to preserve the sustainable
  direct Qiskit path. Date/Author: 2026-08-11, Codex.
- Decision: set `LLVM_ENABLE_EH` to `OFF` and run
  `llvm_update_compile_flags(target)` from the existing MLIR target-options
  helper. Rationale: this is LLVM's supported target-scoped mechanism and covers
  both MLIR libraries and generated object targets without a parallel MQT
  property system. Date/Author: 2026-08-11, Codex.
- Decision: set `LLVM_REQUIRES_EH` only while creating
  `MQTCompilerFoMaCAdapter`, `MLIRQCODDFunctionality`, and
  `mqt-core-qir-runner`. Rationale: these production targets compile APIs whose
  contracts still use exceptions; broadening this refactor into FoMaC, DD, or
  Core IR is outside issue #1590. Date/Author: 2026-08-11, Codex.
- Decision: let the unit-test configuration helper accept `REQUIRES_EH` and use
  it for exactly the seven proven targets. Rationale: function scope prevents
  the LLVM variable from leaking to siblings, and the named argument makes each
  test-only boundary visible at its declaration. Date/Author: 2026-08-11, Codex.
- Decision: do not request RTTI in #2051. Rationale: exception support and RTTI
  are distinct contracts; #2048 owns the final RTTI-disabled policy and must
  prove that these exception boundaries do not acquire RTTI. Date/Author:
  2026-08-11, Codex.
- Decision: do not add a feature option, preset, partial compiler graph, extra
  CI matrix entry, or generated installation-document edit. Rationale: the
  supported default build should enforce the contract directly. Date/Author:
  2026-08-11, Codex.

## Outcomes & Retrospective

The current local revision preserves every compiler target and replaces the
bespoke exception framework with LLVM's standard per-target policy. A fresh
RTTI-enabled LLVM/MLIR 22.1.3 build produced `mqt-cc`, the QIR runner, both
production boundary libraries, and the six initially identified test targets;
all 990 then-discovered MLIR unit tests passed. The later complete RTTI-off
build identified and scoped the optimization executable as the seventh test-only
boundary. Generated commands show `-fno-exceptions` on ordinary MLIR targets and
`mqt-cc`, while the documented boundaries use the compiler's normal
exception-enabled mode.

The remaining acceptance work is the aggregate RTTI-off build owned by #2048,
signed publication, exact-head CI, and review-thread hygiene. No pull request is
merged by this plan.

## Context and Orientation

`cmake/SetupMLIR.cmake` imports LLVM and MLIR's CMake helpers and sets
`LLVM_ENABLE_EH OFF`. `mlir/CMakeLists.txt` owns
`mqt_mlir_apply_target_options`, which links `MQT::ProjectOptions` and invokes
`llvm_update_compile_flags` for each MQT MLIR target.

`mlir/lib/Compiler/FoMaCAdapter.cpp` snapshots a FoMaC device into an immutable
`CompilerTarget`. It opens devices and catches failures inside its
exception-enabled library, exposing `llvm::Expected` to callers.
`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` is the corresponding DD
boundary. `src/qir/runner/Runner.cpp` is a standalone LLVM tool that compiles
the existing exception-based Core IR and DD interfaces.

`mlir/unittests/CMakeLists.txt` applies common target options. Its `REQUIRES_EH`
argument is used only where the test translation unit itself requires exception
syntax. It is not an installed interface or user option.

Pull request #2031 provides the direct `QCProgram.from_qiskit`, `to_qiskit`, and
`compile_program(QuantumCircuit)` path. Pull request #2054 follows it and
removes only the older `QuantumComputation` to MLIR interaction. The exception
policy belongs to pull request #2051, and pull request #2048 follows it and owns
the RTTI-disabled policy.

## Plan of Work

Keep `LLVM_ENABLE_EH OFF`. Apply `llvm_update_compile_flags` from the central
MQT MLIR target helper. Around creation of each production boundary, set
`LLVM_REQUIRES_EH ON` and unset it immediately afterwards. Do the equivalent in
function scope for the seven test executables. Do not introduce manual compiler
flags or target properties.

Keep all compiler libraries, tools, translations, QIR tests, DD functionality,
and the #2031 Qiskit bridge unconditional. Keep FoMaC device opening in the
adapter and return failures as LLVM errors to the exception-disabled driver.

After #2051 is committed, rebase #2048 onto it. The combined revision must keep
the exception boundaries above while disabling RTTI across every LLVM/MLIR
consumer, including those boundaries and the Qiskit binding.

## Concrete Steps

Run cache-producing commands through `.agent/run.sh`. Configure from a fresh
cache with supported LLVM/MLIR 22, build the affected targets, and run the MLIR
unit-test label:

    ./.agent/run.sh env MLIR_DIR=/path/to/llvm/lib/cmake/mlir \
      cmake --preset release -B build/eh-review
    ./.agent/run.sh cmake --build build/eh-review --target \
      mqt-cc mqt-core-qir-runner mqt-core-mlir-unittests -j4
    ./.agent/run.sh ctest --test-dir build/eh-review -L mqt-mlir-unittests \
      --output-on-failure

Audit `compile_commands.json`. `mqt-cc` and ordinary MQT MLIR sources must be
exception-disabled. Only the documented production and test boundaries may be
exception-enabled. A boundary may use the compiler default instead of carrying
an explicit `-fexceptions`; acceptance is based on effective mode, not the
presence of a particular spelling.

Finish with targeted lint, `git diff --check`, and a clean status. Repeat the
aggregate build from a fresh cache against the portable RTTI-off toolchain after
pull request #2048 is rebased.

## Validation and Acceptance

The complete compiler collection must configure and build without a new option.
All focused tests must pass. `mqt-cc` must be exception-disabled, and the Qiskit
bridge from #2031 must remain present and tested. There must be no
`MQT_MLIR_ENABLE_EXCEPTIONS`, custom exception target property, directory-wide
exception generator expression, no-exceptions preset, sentinel, or conditional
subdirectory.

The combined #2048 revision must additionally build against the compatible
portable LLVM/MLIR toolchain with RTTI disabled, without enabling RTTI on an
exception boundary. Replacement GitHub checks are evaluated at the exact signed
published heads; pending or external failures are reported rather than treated
as success.

## Idempotence and Recovery

Use fresh build directories for flag validation so cached LLVM settings cannot
hide a failure. Never reuse another worktree's mutable build directory. If a
target fails with exception syntax, first confirm whether its own translation
unit compiles an exception-based public API. Add the smallest target-scoped
`LLVM_REQUIRES_EH` declaration only when evidence requires it.

If a lower pull request changes, fetch its exact revision, verify ancestry,
rebase one layer at a time, rerun the affected validation, and push only with
the frozen previous remote commit as `--force-with-lease`. Resolve a review
conversation only after its source is removed, its request is implemented, or
GitHub marks it outdated.

## Interfaces and Dependencies

`mlir/Compiler/FoMaCAdapter.h` exposes both
`compilerTargetFromDevice(const fomac::Device&)` and
`compilerTargetFromDeviceId(std::string_view)` as
`llvm::Expected<CompilerTarget>`. The latter keeps device opening and exception
translation out of `mqt-cc`.

LLVM and MLIR remain version 22 or newer, C++ remains C++20, and no new runtime
dependency or installed CMake interface is introduced.

Revision note (2026-08-11): rewritten after inserting #2031 and simplifying
exception handling to LLVM's standard target-scoped mechanism.
