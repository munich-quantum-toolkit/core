# Support stock LLVM builds without exceptions or RTTI

Status: historical implementation record.

## Goal and scope

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

## Constraints

- The local LLVM and MLIR 22.1.8 installation reports both `-fno-exceptions` and
  `-fno-rtti` through `llvm-config --cxxflags`. Evidence: configuring an
  unchanged tree against that installation made ordinary MLIR code compile with
  those flags, but linking the DDSIM device exposed LLVM RTTI references from
  the exception-enabled QIR JIT boundary.

- Loading the DDSIM QDMI device into an executable that exports statically
  linked LLVM symbols registers LLVM command-line options twice and aborts
  before QDMI can return an error. Evidence: the first Bell-program driver test
  reported `Option 'bitcode-mdindex-threshold' registered more than once`.
  Hiding archive symbols in the DDSIM shared device with the ELF linker's
  `--exclude-libs,ALL` option isolates its LLVM copy while retaining plugin
  support in `mqt-cc`.

- The fresh complete build confirmed the planned seven exception-enabled MLIR
  test executables without additions. Every ordinary MLIR compile command
  contains `-fno-exceptions -fno-rtti`; the two MLIR compatibility libraries and
  seven tests omit the exception-disable flag and contain `-fno-rtti`.

- The macOS ARM debug driver test converted an invalid QDMI registry failure to
  `llvm::Error`, but the nested QDMI exception did not match `std::exception`
  across the platform boundary. The adapter therefore emitted its stable
  `Failed to discover registered QDMI devices` prefix followed by
  `unknown exception`. The driver test now checks the stable adapter diagnostic
  instead of a platform-specific nested exception string.

- The macOS 15 Qiskit job linked the nanobind extension with one undefined
  OpenQASM semantic-analyzer member-template instantiation. The normal C++ macOS
  job did not expose the defect because its build mode emitted the
  instantiation. Replacing the local member template with typed qubit and bit
  overloads backed by an ordinary helper removes the platform-dependent weak
  symbol without changing semantic behavior.

- C++ patch coverage first reported 82.1%. All eight missed lines were
  duplicated defensive exception-conversion branches in the QDMI adapter. A
  shared `std::exception_ptr` converter preserves standard diagnostics and the
  unknown-exception fallback while leaving only the fallback branch inherently
  unreachable through the current QDMI APIs.

## Decisions

- Inherit `LLVM_ENABLE_EH`, `LLVM_ENABLE_RTTI`, and ordinary compile flags from
  the imported LLVM package instead of adding an MQT option or overriding the
  imported values. Rationale: This matches stock MLIR consumer policy and keeps
  the full compiler enabled in one supported configuration.

- Apply `llvm_update_compile_flags` in the central
  `mqt_mlir_target_use_project_options` path to both the named MLIR target and
  its `obj.<name>` target. Rationale: MLIR's CMake helpers often compile sources
  in object targets; configuring only the archive or executable does not change
  the actual compile commands.

- Keep exceptions only in the QDMI adapter, QCO DD functionality, QIR JIT and
  runner, DDSIM QDMI device, proven MLIR test executables, and the nanobind
  extension. Rationale: These targets directly use headers or APIs that throw.
  Ordinary compiler code and `mqt-cc` can use `llvm::Expected` and
  `llvm::Error`.

- When imported LLVM is RTTI-free, add LLVM's own platform-specific no-RTTI flag
  only to exception-enabled compatibility boundaries and direct LLVM consumers
  outside the central MLIR helper. Rationale: This keeps their ABI compatible
  without imposing LLVM policy on unrelated Core libraries.

- Keep the nanobind extension exception-enabled and RTTI-enabled. Rationale:
  nanobind headers require both and form a language binding boundary, not part
  of the exception-free compiler driver.

- Retain dynamic pass-plugin support in `mqt-cc` and hide symbols from static
  dependencies in the DDSIM shared device on ELF platforms. Rationale: the QDMI
  entry points remain exported, but the device's LLVM copy cannot interpose the
  host's LLVM registries.

- The LLVM upgrade required a shared workflow version capable of installing the
  selected portable toolchain. Keep the workflow and toolchain requirements
  aligned; the workflow files, not this historical record, own the current pins.

## Outcome and validation

The prerequisite merged as PR #2054. The combined implementation configures and
builds fully against the stock exception-free, RTTI-free LLVM and MLIR 22.1.8
installation. All 4,382 configured CTests and all supported Python suites pass,
including new QDMI driver coverage. MLIR documentation and lint pass. The
complete compile audit matches the intended boundaries. The full Sphinx build is
blocked only by the missing Doxygen executable. Draft PR #2125 contains the
combined change.

## Code and ownership

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

## Acceptance

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

## Interfaces

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
