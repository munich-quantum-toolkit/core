# Remove the standalone QIR runner and internalize QIR execution

Status: historical implementation record.

## Goal and scope

MQT Core must stop presenting its LLVM-based Quantum Intermediate Representation
(QIR) executor as a standalone product. After this change, an installation
contains no `mqt-core-qir-runner` command and no public QIR runtime or
just-in-time compiler headers. The build tree keeps the `MQT::CoreQIRJIT` and
`MQT::CoreQIRRuntime` aliases for consistent internal linking. The DDSIM QDMI
device still accepts and executes supported QIR assembly and bitcode jobs. Its
focused tests prove that every job retains independent simulator, runtime,
output, and random number generator state.

## Constraints

- The JIT and runtime libraries are not members of `MQT_CORE_TARGETS`, so the
  installed CMake package does not export them. The build tree can retain
  namespaced aliases without restoring an installed API.

- The runner test is only a subprocess smoke test. JIT session tests and DDSIM
  device tests already cover execution, repeated runs, ownership, concurrent
  jobs, and result isolation without requiring a production executable.
  Evidence: `test/qir/runner/test_qir_runner.cpp`,
  `mlir/unittests/Dialect/QIR/Execution/JIT/test_jit_session.cpp`, and
  `test/qdmi/devices/dd/concurrency_test.cpp`.

- The compiler and executor repeated QIR attribute, profile, schema, and
  irreversible-operation literals. Both now use
  `mlir/Dialect/QIR/QIRDefinitions.h` without a cross-tree include directory.

- Two of issue #1734's three intended outcomes are now covered. Base Profile IR
  and pipeline tests exist, and QIR-Runner provides external interoperability
  checks. The Base Profile conversion still contains its own
  `ConvertMemRefLoadOp`; centralizing dynamic-to-static qubit conversion remains
  separate compiler work.

## Decisions

- Keep `MQT::CoreQIRJIT` and `MQT::CoreQIRRuntime` as build-tree aliases for the
  unexported static targets. Rationale: Namespaced aliases match other internal
  links without adding the libraries to the installed package.

- Place the executor below `mlir/Dialect/QIR/Execution` and expose its headers
  only through build-interface usage requirements. Rationale: The MLIR QIR
  compiler and DDSIM executor now have one owner and no cross-tree include
  dependency.

- Remove the subprocess runner test instead of converting it. Rationale: The
  issue explicitly removes subprocess tests, while in-process JIT and DDSIM
  tests own the retained behavioral contracts.

- Store shared QIR attribute, profile, and schema literals in
  `mlir/Dialect/QIR/QIRDefinitions.h` as C++20 string views. Rationale: Compiler
  generation and internal execution must agree on these spellings without
  array-to-pointer decay. The header is not part of an installed file set.

- Keep deterministic seeding as DDSIM QDMI custom job parameter 1 instead of a
  JIT session option. Rationale: The job owns sampling policy and can apply one
  seed contract to OpenQASM and QIR.

- Require exactly one standard QIR entry point and remove named selection.
  Rationale: MQT's compiler emits one entry point, while QIR-Runner already
  serves external modules that need selection.

- Keep the QIR output driver test below `mlir/unittests/Compiler` and let the
  JIT and DDSIM tests load their own fixtures. Rationale: The test checks
  `mqt-cc`, while a separate library for one small file loader created needless
  coupling between the MLIR and QDMI test trees.

## Outcome and validation

The native QIR runner, its subprocess test, its user guide, and all public QIR
runtime/JIT headers are removed. The retained implementation lives below the
MLIR QIR subtree and is shared by DDSIM, the compiler, and focused tests. Its
unexported targets retain consistent build-tree aliases. DDSIM exposes
deterministic OpenQASM and QIR sampling through custom job parameter 1. The
internal JIT accepts only in-memory modules with one standard QIR entry point.
The obsolete `test/qir` subtree is gone. The focused tests, all 4,035 configured
C++ tests, documentation build, install-surface scan, focused clang-tidy check,
and lint pass.

## Code and ownership

QIR is an LLVM-based interchange format for quantum programs. MQT Core compiles
programs to QIR and executes QIR inside the decision diagram simulator device in
`src/qdmi/devices/dd/Device.cpp`. The executor lives below
`mlir/lib/Dialect/QIR/Execution`. Its runtime implements quantum instructions
and QIR resource and output functions. Its JIT loads LLVM modules, validates
declarations, binds runtime symbols, and invokes an entry point. Focused tests
live under `mlir/unittests/Dialect/QIR/Execution`; shared QIR test data and
DDSIM integration tests remain under `test`.

Installed targets are controlled through `MQT_CORE_TARGETS` in
`src/CMakeLists.txt`. The QIR libraries are absent from that list. Plain static
targets with manually defined aliases can provide consistent build-tree links
without exporting the libraries or their headers.

The QIR executor and its direct consumers remain internal. This change covers
runtime/JIT/runner build files, headers, tests, and documentation.

## Acceptance

Configuration must define the `MQT::CoreQIRJIT` and `MQT::CoreQIRRuntime`
build-tree aliases and no runner target. Building the default or focused targets
must not produce `mqt-core-qir-runner`. The install manifest and generated
`mqt-core-targets.cmake` must contain no QIR executable, executor header, or QIR
JIT/runtime target.

The JIT test binary must pass module parsing, ABI validation, repeated session,
ownership, output, and state-extraction cases. The runtime test binary must pass
resource, gate, measurement, output-schema, and state ownership cases. The DDSIM
device test binary must pass QIR assembly and bitcode submission, sampling,
statevector, error, repeated-job, and concurrency cases. A zero exit status and
GoogleTest's complete pass summary are required.

DDSIM custom job parameter 1 must accept positive `int` seeds, reject invalid
values, and reproduce both OpenQASM and QIR sampling results across independent
jobs. Modules with multiple `entry_point` functions must be rejected.

The QIR guide must no longer tell users to build or invoke the removed native
runner. `UPGRADING.md` must not add migration text for the unreleased runner.
`uvx nox -s lint` must pass before handoff unless a documented environment
limitation prevents the run.

## Interfaces

At completion, `mqt-core-qir-runtime` and `mqt-core-qir-jit` are internal static
C++20 libraries below `mlir/lib/Dialect/QIR/Execution`. Both use
position-independent code because the shared DDSIM QDMI device links them. Their
`MQT::CoreQIRRuntime` and `MQT::CoreQIRJIT` aliases exist only in the build
tree; neither raw target has an install or export rule.

`src/qdmi/devices/dd/Device.cpp` gains the executor headers through private
links to the aliases. Existing in-process tests use the same aliases. The
existing QDMI custom parameter interface gains a DDSIM-specific contract: custom
parameter 1 is a positive `int` sampling seed. No binding or generated stub
changes are required. `JitSession` accepts only in-memory QIR plus an execution
mode and requires exactly one entry point.
