# Build the MQT MLIR compiler without C++ exceptions or RTTI by default

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds. It is maintained in accordance with
`.agent/PLANS.md`.

## Purpose / Big Picture

MQT's LLVM and MLIR consumers should match the supported portable toolchain's
exception-disabled and RTTI-disabled ABI without making the compiler collection
conditional. Targets that compile exception-using FoMaC, DD, Core QIR, or Python
APIs retain exception handling, but that does not grant them RTTI.

The direct Qiskit bridge introduced by pull request #2031 is intentionally not
part of the legacy `QuantumComputation` to MLIR surface removed by #2054. Its
Python extension remains exception-enabled and retains RTTI because nanobind
uses `typeid`; the MLIR libraries it links remain RTTI-free.

## Progress

- [x] (2026-08-11) Build the fixed-width angle, explicit compiler-error, and
      OpenQASM error-propagation prerequisites as pull requests #2040, #2049,
      and #2050.
- [x] (2026-08-11) Insert #2031 above #2050 and preserve its direct Qiskit C API
      bridge while linearizing its history.
- [x] (2026-08-11) Rework #2054 above #2031 so it removes only the legacy
      `QuantumComputation` to MLIR interaction. Independent DD and direct Qiskit
      functionality remain.
- [x] (2026-08-11) Simplify #2051 to LLVM's normal `llvm_update_compile_flags`
      policy, with exception handling limited to three production boundaries and
      seven proven test-only boundaries.
- [x] (2026-08-11) Build #2051 from a fresh LLVM/MLIR 22.1.3 cache and pass all
      990 MLIR unit tests.
- [x] (2026-08-11) Rebase #2048 above the final local #2051 revision and remove
      its directory-wide RTTI options and per-boundary RTTI requests.
- [x] (2026-08-11) Validate the complete #2048 graph from a fresh cache against
      the published portable LLVM/MLIR 22.1.8 RTTI-off artifact, including the
      native graph, CTests, direct Qiskit bridge, compiler driver, and effective
      compile flags.
- [ ] Commit and publish the signed revisions, inspect exact-head CI, and
      resolve only outdated or source-addressed review conversations.

## Surprises & Discoveries

- Observation: LLVM's `llvm_update_compile_flags` already gives ordinary MLIR
  targets the correct exception and RTTI policy. The earlier directory-wide
  generator expressions and custom exception target property duplicated that
  behavior and complicated generated object targets and Visual Studio builds.
- Observation: LLVM's helper couples `LLVM_REQUIRES_EH` to RTTI internally,
  although MQT's exception boundaries compile and link against the portable
  RTTI-off toolchain without language RTTI. A final target-scoped RTTI-disable
  option keeps those two policies independent.
- Observation: the FoMaC adapter, DD functionality, and QIR runner are the only
  production exception boundaries in the compiler graph. Seven test translation
  units also require exceptions because they instantiate throwing inline Core IR
  or DD header code.
- Observation: `mqt-cc` can remain exception-disabled after FoMaC device opening
  and exception translation move into `MQTCompilerFoMaCAdapter`.
- Observation: QIR JIT and the DD QDMI device consume LLVM without being created
  through AddLLVM/AddMLIR. They therefore need the same explicit RTTI ABI policy
  as the exception-enabled MLIR targets.
- Observation: the direct Qiskit bridge is part of a nanobind extension.
  nanobind's public headers use `typeid`, so that extension is the one proven
  LLVM/MLIR consumer that must retain language RTTI. Its linked MLIR libraries
  remain RTTI-free.
- Observation: the four unit-test program fixture libraries bypassed both
  AddMLIR and the unit-test target helper. Applying the same central MLIR target
  options to those ordinary, exception-free fixtures closes the final RTTI gap
  without granting another exception boundary.
- Observation: locally loading the new MLIR extension beside an unrelated
  installed FoMaC nanobind module produced a false exception-translation
  failure. Loading the matching IR, FoMaC, and MLIR extensions from one build
  restores the normal translator and passes every direct Qiskit bridge test.

## Decision Log

- Decision: retain the one-way stack #2040, #2049, #2050, #2031, #2054, #2051,
  #2048. Rationale: each layer has a separately reviewable contract, while
  placing #2031 before #2054 lets the cleanup preserve the sustainable direct
  Qiskit path. Date/Author: 2026-08-11, Codex.
- Decision: set both `LLVM_ENABLE_EH` and `LLVM_ENABLE_RTTI` to `OFF` without a
  user option. Rationale: this matches the portable toolchain and keeps the
  complete compiler graph as the default build. Date/Author: 2026-08-11, Codex.
- Decision: use `llvm_update_compile_flags` for every MQT MLIR target and a
  single `mqt_llvm_target_disable_rtti` helper for the final ABI override.
  Rationale: ordinary targets receive LLVM's policy directly, while exception
  boundaries remain RTTI-free without a second target-property framework.
  Date/Author: 2026-08-11, Codex.
- Decision: call the RTTI helper only from the central MLIR target-options
  function and the two direct non-AddMLIR consumers: QIR JIT and the DD QDMI
  device. The AddLLVM QIR runner also receives the final override because its
  exception request otherwise suppresses LLVM's RTTI-disable flag. Rationale:
  these are the exact RTTI-free ABI boundaries; applying the flag through
  `MQT::ProjectOptions` would unnecessarily change unrelated Core targets.
  Date/Author: 2026-08-11, Codex.
- Decision: retain RTTI only for the MLIR Python extension. Rationale: nanobind
  directly instantiates `typeid`, while no compiler library, driver, QIR tool,
  DD device, or MLIR test requires language RTTI. Date/Author: 2026-08-11,
  Codex.
- Decision: set `LLVM_REQUIRES_RTTI` only while AddLLVM/AddMLIR create the three
  production exception targets, then unset it and append the RTTI-disable flag.
  The seven test targets use the central exception-only path and never request
  RTTI. Rationale: this suppresses LLVM's incorrect developer warning without
  leaking the variable or changing any target's final ABI. Date/Author:
  2026-08-11, Codex.
- Decision: do not add a feature option, preset, partial compiler graph, extra
  CI matrix entry, or generated installation-document edit. Rationale: the
  supported default build should enforce the contract directly. Date/Author:
  2026-08-11, Codex.

## Outcomes & Retrospective

The current local series preserves every compiler target and replaces the old
exception/RTTI framework with LLVM's standard per-target flags plus one narrow
RTTI ABI helper. There is no custom exception property, manual exception-enable
flag, directory-wide generator expression, or RTTI request.

The simplified #2051 layer built `mqt-cc`, the QIR runner, both production MLIR
boundaries, and the six initially identified test targets; all 990
then-discovered MLIR unit tests passed. The complete #2048 build identified and
scoped the optimization executable as the seventh test-only boundary.

The final combined tree then configured and built all 598 native targets against
the published portable LLVM/MLIR 22.1.8 no-RTTI release. All 4,264 discovered
CTests passed, with only the two live job-ID tests skipped by design. The direct
Qiskit bridge passed all 118 tests against matching IR, FoMaC, and MLIR Python
extensions. `mqt-cc` listed the configured devices and compiled the Bell program
for `mqt.ddsim.default`. The compile-command audit found RTTI disabled for all
163 repository MLIR translation units; only the four nanobind extension sources
retain RTTI and exceptions. Publication remains pending. No pull request is
merged by this plan.

## Context and Orientation

`cmake/SetupMLIR.cmake` imports LLVM and MLIR's CMake helpers, sets
`LLVM_ENABLE_EH OFF` and `LLVM_ENABLE_RTTI OFF`, and defines the internal
`mqt_llvm_target_disable_rtti` helper.

`mlir/CMakeLists.txt` owns `mqt_mlir_apply_target_options`. It links
`MQT::ProjectOptions`, invokes `llvm_update_compile_flags`, and then enforces
the RTTI-off ABI. `mqt_mlir_target_use_project_options` applies this to both a
public MLIR library and its generated `obj.*` target when present.

`mlir/lib/Compiler/FoMaCAdapter.cpp` snapshots a FoMaC device and returns
`llvm::Expected` errors to the exception-disabled driver.
`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` is the DD boundary.
`src/qir/runner/Runner.cpp` compiles existing exception-based Core IR and DD
interfaces. Their target-creation scopes satisfy AddLLVM/AddMLIR's coupled
exception/RTTI assumption, immediately unset both variables, and finish with
RTTI disabled.

`src/qir/jit` and `src/qdmi/devices/dd` are direct LLVM consumers not governed
completely by the AddLLVM/AddMLIR target helpers. The MLIR binding includes the
direct Qiskit bridge from #2031 and remains an exception-enabled, RTTI-enabled
nanobind boundary.

Pull request #2054 removes only the older `QuantumComputation` to MLIR
interaction. Pull request #2051 owns exception handling, and #2048 owns the
final RTTI policy.

## Plan of Work

Keep the complete compiler graph and the direct Qiskit bridge unconditional. Let
LLVM configure ordinary targets. Retain `LLVM_REQUIRES_EH` only around the three
production exception targets and in function scope for the seven proven test
targets. Do not leave RTTI enabled for any exception boundary.

Apply the RTTI-off ABI centrally to all MLIR targets after LLVM's flags. Apply
the same helper only to direct LLVM/MLIR consumers that bypass or override the
ordinary LLVM target policy. Keep the nanobind extension RTTI-enabled, and do
not add either policy to project-wide options.

## Concrete Steps

Run cache-producing commands through `.agent/run.sh`. Configure a fresh Release
cache against the published portable LLVM/MLIR 22.1.8 RTTI-off toolchain and
build the complete graph:

    ./.agent/run.sh env \
      MLIR_DIR=/path/to/portable/lib/cmake/mlir \
      cmake --preset release -B build/rtti-off-review
    ./.agent/run.sh cmake --build build/rtti-off-review -j4
    ./.agent/run.sh ctest --test-dir build/rtti-off-review \
      --output-on-failure

Run the Qiskit bridge tests in the configured Python environment and exercise
`mqt-cc` device listing and compilation for `mqt.ddsim.default`.

Audit `compile_commands.json`. Every repository target that consumes LLVM or
MLIR must be RTTI-disabled. `mqt-cc` and ordinary MLIR sources must also be
exception-disabled. Only the documented production and test boundaries may be
exception-enabled. Acceptance is based on effective compiler mode, not the
presence of a particular enable-flag spelling.

Finish with targeted lint, `git diff --check`, and a clean worktree.

## Validation and Acceptance

The complete compiler collection must configure and build against the portable
LLVM/MLIR 22 toolchain without a new option. All discovered tests and the direct
Qiskit bridge tests must pass, apart from explicitly documented live tests.
`mqt-cc` must remain exception-disabled and compile a Bell input for the
configured DDSIM device.

Only the three AddLLVM/AddMLIR production target-creation scopes may temporarily
set `LLVM_REQUIRES_RTTI`, and the variable must be unset immediately. Every
LLVM/MLIR consumer except the nanobind extension must use the RTTI-off ABI; the
extension retains RTTI because nanobind requires it. There must be no custom
exception target property, directory-wide exception/RTTI generator expression,
strict preset, sentinel, or conditional compiler subdirectory.

Replacement GitHub checks are evaluated at the exact signed published heads;
pending or external failures are reported rather than treated as success.

## Idempotence and Recovery

Use a fresh build directory for flag validation so cached LLVM settings cannot
hide a failure. Never reuse another worktree's mutable build directory. If a
target fails with exception syntax, first establish that its translation unit
compiles an exception-based API before adding the smallest target-scoped
`LLVM_REQUIRES_EH` declaration. If a link fails with missing LLVM typeinfo,
inspect the final compile command and apply the RTTI helper only to the direct
LLVM consumer that escaped the existing policy.

If a lower pull request changes, fetch its exact revision, verify ancestry,
rebase one layer at a time, rerun affected validation, and push only with the
frozen previous remote commit as `--force-with-lease`. Resolve a review
conversation only after its source is removed, its request is implemented, or
GitHub marks it outdated.

## Interfaces and Dependencies

`mlir/Compiler/FoMaCAdapter.h` exposes
`compilerTargetFromDevice(const fomac::Device&)` and
`compilerTargetFromDeviceId(std::string_view)` as
`llvm::Expected<CompilerTarget>`. The latter keeps device opening and exception
translation out of `mqt-cc`.

`mqt_llvm_target_disable_rtti` is an internal CMake helper, not an installed
option or library interface. LLVM and MLIR remain version 22 or newer, C++
remains C++20, and no new runtime dependency is introduced.

Revision note (2026-08-11): rewritten after inserting #2031 and simplifying the
exception and RTTI policies to target-scoped LLVM mechanisms.
