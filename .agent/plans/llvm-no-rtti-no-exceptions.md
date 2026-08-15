# Support stock LLVM builds without exceptions or RTTI

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's MLIR compiler must build against a stock LLVM and MLIR installation
that disables C++ exceptions and run-time type information (RTTI). After this
change, `mqt-cc` and ordinary MLIR libraries inherit those settings from LLVM.
Small compatibility libraries that call exception-based Core or QDMI APIs keep
exceptions, but remain RTTI-free when LLVM is RTTI-free. The Python nanobind
extension remains the sole MLIR consumer that keeps both exceptions and RTTI
because nanobind uses both language features.

The result is visible in two ways. A fresh Release build against LLVM and MLIR
22.1.8 completes without changing LLVM's installed configuration, and
`compile_commands.json` shows `-fno-exceptions -fno-rtti` for `mqt-cc` and
ordinary MLIR sources. The driver can still list QDMI devices and compile a Bell
program for `mqt.ddsim.default`; QDMI failures become `llvm::Error` values and
produce normal diagnostics instead of escaping as C++ exceptions.

## Progress

- [x] (2026-08-15 19:03Z) Rebased the prerequisite compiler-input removal onto
  the current upstream main, regenerated stubs, validated it, and updated PR
  #2054 at exact head `b74138dd7`.
- [x] (2026-08-15 19:03Z) Created this combined ExecPlan from the validated
  prerequisite tree and recorded the no-exception and no-RTTI boundaries.
- [x] (2026-08-15 19:15Z) Removed the project and MLIR configuration overrides,
      then applied LLVM's target policy centrally to MLIR libraries, object
      libraries, tools, and test fixture libraries.
- [x] (2026-08-15 19:15Z) Added non-throwing QDMI device-list and device-ID
      compiler-target adapters, then removed exception-based QDMI calls and the
      top-level catch from `mqt-cc`.
- [x] (2026-08-15 19:15Z) Marked only the confirmed compatibility boundaries and
      header-dependent MLIR tests as exception-enabled while keeping them
      RTTI-free with an RTTI-free LLVM installation.
- [x] (2026-08-15 19:15Z) Added adapter and driver tests, and documented the
      non-throwing target path.
- [x] (2026-08-15 19:15Z) Updated every repository LLVM and MLIR pin to 22.1.8;
      the pull request reference remains pending.
- [x] (2026-08-15 19:20Z) Validated focused targets, the complete build and
      4,382 CTests, all supported Python suites, MLIR docs, lint, formatting,
      and compile flags against LLVM and MLIR 22.1.8. The full Sphinx build
      remains unavailable because this host has no Doxygen executable.
- [ ] Publish the combined draft pull request after PR #2054 merges and inspect
  all checks for the exact pushed head. A human performs the merge.

## Surprises & Discoveries

- Observation: The local LLVM and MLIR 22.1.8 installation reports both
  `-fno-exceptions` and `-fno-rtti` through `llvm-config --cxxflags`. Evidence:
  configuring an unchanged tree against that installation made ordinary MLIR
  code compile with those flags, but linking the DDSIM device exposed LLVM RTTI
  references from the exception-enabled QIR JIT boundary.
- Observation: The prerequisite PR's complete configured CTest suite contains
  4,162 tests. All passed; the two job-ID tests were skipped by design. The
  supported Python 3.10 through 3.14 sessions each passed all 48 affected MLIR
  binding tests.
- Observation: The full documentation session cannot finish on this host because
  the external `doxygen` executable is absent. The `mlir-doc` CMake target
  succeeds. This environmental limitation must be checked again before final
  handoff and reported if it remains.
- Observation: Loading the DDSIM QDMI device into an executable that exports
  statically linked LLVM symbols registers LLVM command-line options twice and
  aborts before QDMI can return an error. Evidence: the first Bell-program
  driver test reported
  `Option 'bitcode-mdindex-threshold' registered more than once`. Removing
  unused MLIR pass-plugin support and executable-symbol export from `mqt-cc`
  isolates the two LLVM copies; the QDMI driver test then passed.
- Observation: The fresh complete build confirmed the planned seven
  exception-enabled MLIR test executables without additions. Every ordinary MLIR
  compile command contains `-fno-exceptions -fno-rtti`; the two MLIR
  compatibility libraries and seven tests omit the exception-disable flag and
  contain `-fno-rtti`.

## Decision Log

- Decision: Inherit `LLVM_ENABLE_EH`, `LLVM_ENABLE_RTTI`, and ordinary compile
  flags from the imported LLVM package instead of adding an MQT option or
  overriding the imported values. Rationale: This matches stock MLIR consumer
  policy and keeps the full compiler enabled in one supported configuration.
  Date/Author: 2026-08-15 / Codex.
- Decision: Apply `llvm_update_compile_flags` in the central
  `mqt_mlir_target_use_project_options` path to both the named MLIR target and
  its `obj.<name>` target. Rationale: MLIR's CMake helpers often compile sources
  in object targets; configuring only the archive or executable does not change
  the actual compile commands. Date/Author: 2026-08-15 / Codex.
- Decision: Keep exceptions only in the QDMI adapter, QCO DD functionality, QIR
  JIT and runner, DDSIM QDMI device, proven MLIR test executables, and the
  nanobind extension. Rationale: These targets directly use headers or APIs that
  throw. Ordinary compiler code and `mqt-cc` can use `llvm::Expected` and
  `llvm::Error`. Date/Author: 2026-08-15 / Codex.
- Decision: When imported LLVM is RTTI-free, add LLVM's own platform-specific
  no-RTTI flag only to exception-enabled compatibility boundaries and direct
  LLVM consumers outside the central MLIR helper. Rationale: This keeps their
  ABI compatible without imposing LLVM policy on unrelated Core libraries.
  Date/Author: 2026-08-15 / Codex.
- Decision: Keep the nanobind extension exception-enabled and RTTI-enabled.
  Rationale: nanobind headers require both and form a language binding boundary,
  not part of the exception-free compiler driver. Date/Author: 2026-08-15 /
  Codex.
- Decision: Do not advertise dynamic pass-plugin support from `mqt-cc`.
  Rationale: The driver does not provide a pass-plugin interface, and exported
  LLVM symbols interpose the LLVM registries in a loaded QDMI device. Keeping
  executable symbols private makes QDMI device loading safe. Date/Author:
  2026-08-15 / Codex.

## Outcomes & Retrospective

The prerequisite rebase is complete and published. The combined implementation
configures and builds fully against the stock exception-free, RTTI-free LLVM and
MLIR 22.1.8 installation. All 4,382 configured CTests and all supported Python
suites pass, including new QDMI driver coverage. MLIR documentation and lint
pass. The complete compile audit matches the intended boundaries. The full
Sphinx build is blocked only by the missing Doxygen executable. Publication
remains pending until a human merges PR #2054.

## Context and Orientation

`cmake/CompilerOptions.cmake` defines options inherited by ordinary MQT targets.
It currently adds `-fexceptions` on non-MSVC builds. `cmake/SetupMLIR.cmake`
loads the installed LLVM and MLIR CMake packages, but currently forces their
exception and RTTI variables on. `llvm_update_compile_flags`, provided by LLVM's
`AddLLVM.cmake`, translates the imported LLVM policy into target compile flags.

`mlir/CMakeLists.txt` owns `mqt_mlir_target_use_project_options`. MLIR library
helpers often create both a public library and an `obj.<library>` object library
that compiles the source. The central helper therefore must configure both
targets. Test fixture libraries under `mlir/unittests/programs/CMakeLists.txt`
also compile MLIR code and must use the same helper.

An exception boundary is a small target compiled with C++ exceptions because it
calls an API that throws. `mlir/lib/Compiler/QDMIAdapter.cpp` is such a
boundary: it queries QDMI C++ client objects and converts their data into the
immutable `CompilerTarget` model. The public declarations are in
`mlir/include/mlir/Compiler/QDMIAdapter.h`. The driver in
`mlir/tools/mqt-cc/mqt-cc.cpp` must call only error-returning adapter functions,
so its own translation unit compiles with exceptions disabled.

Other exception boundaries are `mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp`,
the QIR JIT in `src/qir/jit/`, the QIR runner in `src/qir/runner/`, and the
DDSIM QDMI device in `src/qdmi/devices/dd/`. Some MLIR unit tests include
exception-based Core or GoogleTest headers and also need exceptions. The test
helper in `mlir/unittests/CMakeLists.txt` provides one central place to opt
those test executables in.

The repository pins LLVM and MLIR in CI, upstream and Slurm workflows, wheel
builds, Read the Docs, and the development container. All pins must use the
portable 22.1.8 release. No reduced compiler feature set, build option, or new
CI matrix is part of this work.

## Plan of Work

First, remove the unconditional `-fexceptions` option from
`cmake/CompilerOptions.cmake` and the forced LLVM exception and RTTI variables
from `cmake/SetupMLIR.cmake`. After LLVM's CMake modules are loaded, add a
helper that applies `${LLVM_CXXFLAGS_RTTI_DISABLE}` to a named C++ target only
when the imported `LLVM_ENABLE_RTTI` value is false. The helper must reject
missing targets and must not change unrelated Core targets.

Next, extend `mqt_mlir_apply_target_options` in `mlir/CMakeLists.txt` to call
`llvm_update_compile_flags` and then apply the no-RTTI helper. Keep
`mqt_mlir_target_use_project_options` responsible for both the named and object
targets. Remove the duplicate direct LLVM flag call from `mqt-cc`. Route every
fixture library in `mlir/unittests/programs/CMakeLists.txt` through this helper.

Wrap the QDMI adapter and QCO DD library creation with local `LLVM_REQUIRES_EH`
and `LLVM_REQUIRES_RTTI` values, then unset both immediately. LLVM's helper
treats these variables as the request to preserve exceptions at those targets;
the central helper then restores LLVM's no-RTTI flag when the toolchain is
RTTI-free. Do the same for the QIR runner, and call the no-RTTI helper for the
direct LLVM consumers in the QIR JIT and DDSIM device. Extend the MLIR unit-test
helper with a `REQUIRES_EH` keyword. Use that keyword for the seven tests
already proven to include exception-dependent headers, then use a fresh build to
add or remove entries based on compiler evidence.

Add `compilerTargetFromDeviceId(std::string_view)` and
`registeredQDMIDeviceIds()` to `QDMIAdapter.h`, both returning LLVM error-aware
types. In `QDMIAdapter.cpp`, keep the existing snapshot logic in an internal
function. Make every public factory catch `std::exception` and unknown
exceptions from QDMI, then return `llvm::createStringError`. The ID factory must
open the device and snapshot it entirely inside the adapter boundary. The list
function must contain `qdmi::Driver::get().registeredDeviceIds()` in the same
way. Change `mqt-cc` to use these functions, remove its QDMI client and driver
includes, remove its top-level `try` and `catch`, and report adapter errors with
a nonzero exit code.

Extend `mlir/unittests/Compiler/test_compiler_qdmi_adapter.cpp` with an unknown
device-ID failure and a registered-ID success case. Add driver-level CTest cases
under `test/qir/mqt-cc/` for device listing, invalid registry configuration, an
unknown device, and compiling a Bell OpenQASM program through
`mqt.ddsim.default`. The tests must assert normal nonzero exits and useful
diagnostics for failures.

Document the non-throwing C++ path in `docs/mlir/target_compilation.md`. Replace
every pin for the previous LLVM patch release with 22.1.8 in repository
configuration. Add the combined pull request reference to the existing generic
QC/QCO compiler infrastructure changelog entry after GitHub assigns the pull
request number.

## Concrete Steps

Run all commands from the repository root. During implementation, use the
installed LLVM and MLIR 22.1.8 package:

    cmake --preset release -B build/no-rtti-no-eh \
      -DMLIR_DIR=/path/to/llvm-22.1.8/lib/cmake/mlir \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

Build the narrow targets first so each boundary failure is local:

    cmake --build build/no-rtti-no-eh --target \
      mqt-cc MQTCompilerQDMIAdapter MLIRQCODDFunctionality \
      mqt-core-qir-jit mqt-core-qir-runner \
      mqt-core-qdmi-ddsim-device -j4
    cmake --build build/no-rtti-no-eh --target \
      mqt-core-mlir-unittests-compiler -j4

Then build and test the complete configured project:

    cmake --build build/no-rtti-no-eh -j4
    ctest --test-dir build/no-rtti-no-eh --output-on-failure -j4
    uvx nox --non-interactive -s tests -- test/python/test_mlir.py -q
    cmake --build build/no-rtti-no-eh --target mlir-doc -j4
    uvx nox --non-interactive -s docs
    uvx nox --non-interactive -s lint
    git diff --check

Search the compile database by source and target. Ordinary MLIR sources and
`mqt-cc.cpp` must contain `-fno-exceptions` and `-fno-rtti`. Each documented
compatibility boundary must contain exception-enable and no-RTTI flags, with the
enable flag occurring after any inherited disable flag. The nanobind module may
contain exception-enable and RTTI-enable flags. No other MLIR source may retain
both.

Before publication, run:

    rg '22\.1\.[0-7]' --hidden -g '!build/**' -g '!.git/**'
    git status --short
    git diff --stat <base>...HEAD

The first command must return no matches. Push only the combined branch, open a
draft pull request that targets the post-prerequisite main, and inspect all
checks for the exact pushed head. The pull request body must start with the AI
text disclosure required by `docs/ai_usage.md`, explain validation, state that
it fixes #1589 and #1590, and state that it supersedes #2048 and #2051.

## Validation and Acceptance

Acceptance requires a fresh Release configuration using LLVM and MLIR 22.1.8
without modifying that installation. The complete build and CTest suite must
pass. The compiler adapter tests must prove that a registered ID can be listed,
an unknown ID returns `llvm::Error`, and QDMI exceptions do not escape.

At driver level, `mqt-cc --qdmi-list-devices` must print `mqt.ddsim.default`. An
unknown device or invalid registry file must print an error and exit with a
nonzero status. Compiling the repository's Bell OpenQASM input with
`--qdmi-device=mqt.ddsim.default --emit=qco-optimized` must succeed and produce
QCO MLIR output.

The compile database is part of acceptance. `mqt-cc` and ordinary MLIR sources
must be exception-free and RTTI-free. The QDMI adapter, QCO DD library, QIR JIT
and runner, DDSIM device, and only the confirmed tests may enable exceptions;
they must remain RTTI-free. Only the nanobind extension may enable both. Every
toolchain reference must use 22.1.8.

Python binding tests, MLIR documentation, full documentation when Doxygen is
available, lint, and `git diff --check` must pass. Any unavailable check must be
recorded with its exact environmental cause. GitHub checks must belong to the
exact published head before the work is ready for human review.

## Idempotence and Recovery

All configure, build, CTest, Nox, search, and audit commands are repeatable. The
dedicated build directory prevents this work from altering another build. If
configuration or compilation fails, edit only the target named in the first
relevant diagnostic and rerun that target before resuming the complete build. Do
not delete or reset unrelated work. Generated Python stubs must be produced only
by `uvx nox -s stubs`; do not edit them by hand.

Rebasing and force-pushing the prerequisite used an exact remote lease. The
combined branch must use an ordinary push unless its own history is later
rewritten, in which case use a lease tied to the previously observed remote
head. Never merge either pull request; a human reviews and merges them.

## Artifacts and Notes

The prerequisite validation established a clean starting point:

    100% tests passed out of 4162
    48 passed on each of Python 3.10, 3.11, 3.12, 3.13, and 3.14
    lint: all hooks passed
    mlir-doc: built successfully

The full Sphinx session stopped before rendering native API documentation with:

    ExtensionError: Doxygen is required to build the native C++ API documentation

The combined C++ validation currently records:

    complete Release build against LLVM and MLIR 22.1.8: passed
    100% tests passed out of 4382
    QDMI adapter tests: 8 passed
    mqt-cc QIR and QDMI driver tests: 2 passed
    exception-enabled MLIR test executables: exactly 7, all with -fno-rtti

The supported Python sessions passed with these totals:

    Python 3.10: 646 passed, 7 skipped
    Python 3.11: 678 passed, 4 skipped
    Python 3.12: 678 passed, 4 skipped
    Python 3.13: 678 passed, 4 skipped
    Python 3.14: 689 passed, 3 skipped

The final compile database classification is:

    ordinary MLIR and mqt-cc entries: 55 with -fno-exceptions -fno-rtti
    MLIR compatibility and test entries: 9 with exceptions and -fno-rtti
    direct Core LLVM consumers: QIR JIT, QIR runner, and DDSIM with exceptions and -fno-rtti
    nanobind extension entries: 2 with compiler-default exceptions and RTTI

Update this section with focused compile errors that change the exception
boundary list and with the final compile-command audit summary.

## Interfaces and Dependencies

At completion, `mlir/include/mlir/Compiler/QDMIAdapter.h` exports:

    llvm::Expected<CompilerTarget>
    compilerTargetFromDevice(const qdmi::Device& device);

    llvm::Expected<CompilerTarget>
    compilerTargetFromDeviceId(std::string_view deviceId);

    llvm::Expected<std::vector<std::string>> registeredQDMIDeviceIds();

These APIs use `llvm::Expected` so exception-free consumers can inspect or
propagate failure. They depend on the existing QDMI C++ API only inside
`MQTCompilerQDMIAdapter`, which remains the compatibility boundary. No public
`mqt-cc` code directly includes `qdmi/Client.hpp` or `qdmi/driver/Driver.hpp`
after this change.

Revision note (2026-08-15): Created the combined plan after rebasing and
validating PR #2054. It supersedes the separate exception and RTTI plans for
this implementation and records the narrowed QDMI adapter design. Updated the
plan after the full 22.1.8 build to record the confirmed boundary list, the
symbol-export loader failure, completed C++ and Python validation, and the final
compile database audit.
