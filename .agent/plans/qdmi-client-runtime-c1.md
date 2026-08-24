# Load the standard QDMI Client runtime

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently links its C++ QDMI wrappers to its own default driver. A
deployment therefore cannot replace the QDMI Client implementation without
rebuilding Core. This change makes `MQT::CoreQDMI` a shared, driver-independent
Client library. It loads one complete standard QDMI 1.4 Client implementation,
validates the Client ABI before use, and routes every opaque handle through that
implementation. Users can select a third-party Client driver by an explicit path
or by `MQT_CORE_QDMI_DRIVER`; otherwise Core uses its packaged driver.

The observable proof is a focused test that loads a fake third-party Client,
rejects incomplete and incompatible libraries without freezing the process, then
allocates independent authenticated sessions and opens a device by its stable
`QDMI_DEVICE_PROPERTY_ID`. A second test runs a consumer from a different
working directory and proves that the packaged shared runtime remains usable.

## Progress

- [x] (2026-08-24 15:13Z) Created an isolated worktree at the exact reduced
      payload head and reviewed the proven combined implementation.
- [x] (2026-08-24 16:16Z) Added the complete transactional QDMI 1.4 Client
      loader and shared Session ownership without private extension symbols.
- [x] (2026-08-24 16:16Z) Decoupled the shared Core QDMI wrapper from the shared
      packaged driver and colocated the in-tree runtime on all supported
      platforms.
- [x] (2026-08-24 16:16Z) Migrated C++, Python, MLIR, Qiskit, PennyLane, and
      Slurm generic consumers to standard Client sessions and stable IDs.
- [x] (2026-08-24 16:16Z) Added focused fake-third-party, packaged-runtime, ABI,
      retry, authentication, malformed-result, and descendant-lifetime tests.
- [x] (2026-08-24 16:16Z) Updated documentation, the upgrade guide, changelog,
      and generated stubs.
- [x] (2026-08-24 16:16Z) Passed the release build, 4,085 configured C++ tests,
      726 Python tests, stubs, documentation, and lint on the initial
      checkpoint.
- [x] (2026-08-24 16:28Z) Replayed the one C1 commit onto repaired stack head
      `d2caff06`, verified the range-diff, and passed the incremental release
      build, focused C++ and Python tests, and lint.

## Surprises & Discoveries

- Observation: `MQT::CoreQDMI` directly links `MQT::CoreQDMIDriver`, and
  `Session::openDevice` bypasses the standard Client catalog through the
  process-global registry. Evidence: `src/qdmi/CMakeLists.txt` lists the driver
  as a public dependency, while `src/qdmi/Client.cpp` calls
  `Driver::get().openFresh`.
- Observation: a static `MQT::CoreQDMI` would give each linked image a separate
  loader-selection singleton. Evidence: the selection state belongs in
  `src/qdmi/Client.cpp`, so the wrapper must be shared to enforce one selection
  per process.
- Observation: publishing the function table before a potentially throwing path
  copy could unload the selected shared library while callers still held its
  function pointers. Evidence: `selectClient` now transfers the library, copies
  the `std::shared_ptr`, and swaps a prepared path; all three operations are
  non-throwing, and the selection state lives for the process.
- Observation: a byte count that is not a multiple of the requested element type
  could make Core allocate too few elements and still pass the full byte count
  to a Driver. Evidence: the fake Client reports odd sizes for Session devices,
  Device sites, and Operation sites, and all three queries now reject the result
  before allocation.
- Observation: GoogleTest discovery can run a Windows test executable before its
  post-build Driver copies finish. Evidence: the main QDMI test now uses
  `PRE_TEST` discovery on Windows, after which discovery runs at CTest time.
- Observation: the packaged Driver's standard `QDMI_session_alloc` wrapper could
  let a C++ allocation exception cross the C ABI. Evidence: the wrapper now
  clears the output first and maps allocation, argument, and unknown exceptions
  to QDMI status values.
- Observation: the documentation build needs generated MLIR reference files.
  Evidence: the first Sphinx run reported 25 missing-file warnings; building the
  `mlir-doc` target first made the same documentation command pass.

## Decision Log

- Decision: Implement only the standard QDMI Client boundary in this change. Do
  not add manifest discovery, registration, targeted sessions, or private
  extension symbols. Rationale: those concerns form a separate default-driver
  layer and must not prevent a third-party Client from working. Date/Author:
  2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Select a driver in the order explicit `SessionConfig::driverPath`,
  `MQT_CORE_QDMI_DRIVER`, then packaged driver. Keep strict failures and freeze
  only after a successful session allocation. Rationale: explicit deployment
  choices must not silently fall back, while failed probes must remain
  retryable. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Keep the selected dynamic library loaded until process teardown and
  make every descendant wrapper share the owning Client session. Rationale: QDMI
  handles are opaque and remain valid only with their implementation code and
  session owner. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Build both Core QDMI libraries as shared libraries but keep
  `MQT::CoreQDMI` link-independent from `MQT::CoreQDMIDriver`. Rationale: one
  process must have one Client selection, and the packaged driver is only the
  default runtime choice. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Reject empty and embedded-NUL stable device IDs at the registry and
  public selection boundaries. Rationale: a C string cannot distinguish `foo`
  from a configured `foo\0bar`, so accepting either would make selection
  ambiguous. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.
- Decision: Keep installed-package and imported-consumer transitive runtime
  deployment out of this change. Rationale: direct in-tree runtime colocation is
  enough for C1 to build and test independently; public deployment hardening is
  a separate follow-up. Date/Author: 2026-08-24 / GPT-5.6 Sol via Codex.

## Outcomes & Retrospective

The final change implements the standard QDMI 1.4 Client boundary without
private Driver extensions. `MQT::CoreQDMI` loads and validates one complete
Client table, sessions own every descendant handle, and generic consumers use
stable enumerated IDs and standard authentication. The fake third-party Client
proves retry after load, ABI, symbol, and allocation failures. The packaged
runtime test proves that direct build-tree colocation is sufficient. C1 remains
independently green and has no required C2 manifest or C3 deployment dependency.

The complete checkpoint validation passed 4,085 CTest tests with one configured
simulator-dependent skip and 726 Python tests with three expected Qiskit skips.
Stubs, documentation, and lint also passed. After the final replay onto
`d2caff06`, the incremental release build passed. The 152 compiler tests, 234
main Client tests, focused third-party and packaged-runtime tests, 396 affected
Python tests, and lint also passed. Windows path loading and delayed discovery
remain hosted-platform checks.

## Context and Orientation

QDMI is a C interface with opaque session, device, site, operation, and job
handles. A QDMI Client driver is a shared library that exports the standard
functions declared by `qdmi/client.h`. QDMI 1.4 adds
`QDMI_driver_get_client_abi_version` so a loader can validate compatibility
without starting a session, and it defines `QDMI_DEVICE_PROPERTY_ID` as the
stable Client-visible device identifier.

`include/mqt-core/qdmi/Client.hpp` declares Core's C++ wrappers.
`src/qdmi/Client.cpp` implements them. `bindings/qdmi/qdmi.cpp` exposes the same
wrappers to Python. `src/qdmi/driver` is Core's packaged default Client driver.
Generic consumers also exist in `src/qdmi/Slurm.cpp`,
`mlir/lib/Compiler/QDMIAdapter.cpp`, and the Qiskit and PennyLane modules under
`python/mqt/core/plugins`.

The complete function table must contain the ABI query and all twenty standard
Client operations that Core can call. Core must resolve the complete table
before allocating a session. It must publish the selected library, table, and
normalized path only through operations that cannot throw. A failed load, symbol
check, ABI check, or session allocation must close temporary resources and leave
selection unset.

This plan coordinates only the generic Client layer. A later change owns
default-driver manifests and targeted device sessions. Another later change owns
installed-package and transitive deployment hardening. This plan may copy the
in-tree shared libraries and their Windows runtime DLLs beside test and tool
consumers because that is required for this change to pass its own build-tree
tests.

## Plan of Work

First, add a platform loader to `src/qdmi/Client.cpp`. Use `LoadLibraryExW` with
safe search flags on Windows and `dlopen` with `RTLD_NOW | RTLD_LOCAL` on POSIX.
Use the existing UTF-8 path helpers from `qdmi/common/Common.hpp`. Resolve and
validate the complete standard QDMI 1.4 function table. Store the selected
library for process lifetime. Protect selection with one mutex. Prepare all
potentially allocating values before a raw session exists, guard each returned
session until a `ClientSession` owns it, and publish selection with no-throw
handle transfer and path swap.

Second, add the function table and owning `ClientSession` to
`include/mqt-core/qdmi/Client.hpp`. Add `driverPath` to `SessionConfig`. Route
all Client calls through the selected table. Make `Device`, `Site`, `Operation`,
and `Job` retain the owning session. Add `Device::getId`. Implement
`Session::openDevice` by creating a normal session, enumerating its devices, and
matching the exact stable ID. Preserve the existing generic authentication
fields and payload APIs.

Third, force `MQT::CoreQDMI` and `MQT::CoreQDMIDriver` to be shared. Remove the
link edge from the wrapper to the driver and add only the operating-system
loader dependency. Export the standard Client symbols from the packaged driver.
Extend the existing build-tree runtime-copy helper only enough to place the two
shared Core libraries, the packaged driver dependencies, device libraries, and
Windows runtime DLLs beside in-tree consumers.

Fourth, expose `ClientSession`, `driver_path`, stable `Device.id`, and generic
`open_device` in `bindings/qdmi/qdmi.cpp`. Remove generic binding and consumer
use of the internal registry. Update Slurm, the MLIR QDMI adapter, Qiskit, and
PennyLane to enumerate or open through standard Client sessions. Regenerate
stubs; do not add a `default_driver` module or package metadata discovery.

Finally, add one fake third-party Client library and focused tests. Prove strict
path precedence, missing-symbol and incompatible-ABI rejection, retry after
failure, one-driver freeze, independent authentication, stable-ID selection,
session-owned descendants, UTF-8 paths, and packaged runtime discovery from a
different working directory. Update user-facing prose and run all validation.

## Milestones

### Milestone 1: Standard loader and ownership

At the end of this milestone, `MQT::CoreQDMI` has no direct driver dependency.
The focused runtime test loads a complete third-party Client, rejects bad
candidates, retries successfully, and continues to use a descendant after its
creating `Session` wrapper has gone out of scope.

### Milestone 2: Stable generic consumers

At the end of this milestone, C++ and Python expose stable device IDs and open
devices only by enumerating a standard Client session. MLIR, Slurm, Qiskit, and
PennyLane contain no direct generic registry access. Existing payload behavior
from the base remains unchanged.

### Milestone 3: Independent build-tree runtime

At the end of this milestone, a packaged-runtime test starts from a different
working directory and finds the colocated Core QDMI shared library, packaged
driver, and device runtime. The fake third-party Client test does not export or
depend on any MQT-private symbol.

### Milestone 4: Documentation and release checks

At the end of this milestone, stubs and migration text describe the standard
Client boundary. The release build, focused Python tests, complete configured
C++ suite, documentation build, and lint pass or have a precisely recorded
external dependency limitation.

## Concrete Steps

Run all commands from the repository root. Configure and build with:

    cmake --preset release
    cmake --build --preset release --target \
      mqt-core-qdmi-client-runtime-test mqt-core-qdmi-packaged-runtime-test

Run the focused binaries and Python tests with:

    ./build/release/test/qdmi/mqt-core-qdmi-client-runtime-test
    ./build/release/test/qdmi/mqt-core-qdmi-packaged-runtime-test
    uv run --no-sync pytest test/python/qdmi/test_qdmi.py \
      test/python/plugins/qiskit test/python/plugins/qdmi_pennylane

Regenerate stubs and run complete suitable checks with:

    uvx nox -s stubs
    cmake --build --preset release
    ctest --preset release
    uvx nox --non-interactive -s docs
    uvx nox -s lint

Successful commands report no failed tests. Generated changes must match the
binding edits. The final diff must contain no private driver extension,
manifest-discovery, installed-deployment, credential, or unrelated compiler
change.

## Validation and Acceptance

An explicit driver path overrides the environment. The environment overrides the
packaged driver. Invalid explicit choices fail strictly. Missing symbols, an
incompatible major or minor Client ABI, and failed session allocation leave
selection retryable. The first successful session freezes the normalized path.
The selected library remains loaded while any wrapper exists.

`Session::openDevice(id)` creates a standard authenticated Client session,
enumerates `QDMI_SESSION_PROPERTY_DEVICES`, and returns the exact matching
`QDMI_DEVICE_PROPERTY_ID`. An unknown ID reports the available stable IDs. The
generic path does not call the internal registry or use per-device
configuration.

All device, site, operation, and job calls use the selected function table.
Every descendant keeps the owning session alive, and a job is freed before the
session. The Python API mirrors this ownership through `ClientSession`,
`Device.id`, and `open_device`.

## Idempotence and Recovery

Configuration, build, tests, documentation, and stub generation are repeatable.
Tests restore modified environment variables and use generated build-tree paths.
A failed driver candidate is owned by a temporary loader object and is closed
automatically. No step changes remote state. If configuration retains a stale
QDMI checkout, remove only that build directory through the normal CMake
reconfiguration workflow; never alter source worktrees to recover.

## Artifacts and Notes

The complete checkpoint and final replay produced these concise results:

    mqt-core-qdmi-client-runtime-test: 1 passed
    mqt-core-qdmi-test: 234 passed
    mqt-core-qdmi-driver-test: 98 passed
    mqt-core-qdmi-registry-test: 15 passed
    mqt-core-mlir-unittests-compiler: 152 passed
    ctest --preset release: 4085 passed, 1 configured skip
    uv run --no-sync pytest -q: 726 passed, 3 skipped
    uvx nox --non-interactive -s docs: passed after mlir-doc
    uvx nox --non-interactive -s lint: passed
    final incremental release build: passed
    final affected Python tests: 396 passed

Windows behavior remains a hosted-check requirement because this validation ran
on Linux.

## Interfaces and Dependencies

`qdmi::SessionConfig` gains `std::optional<std::filesystem::path> driverPath`.
`qdmi::Session` remains move-only and gains static
`openDevice(std::string_view, const SessionConfig&)`. `qdmi::Device` gains
`std::string getId() const`. The internal `qdmi::detail::ClientApi` stores exact
function-pointer types derived from the QDMI declarations.
`qdmi::detail::ClientSession` stores a shared function table and raw
`QDMI_Session` and frees that session through the same table.

The implementation uses only C++20, QDMI 1.4, and native dynamic-library APIs.
It adds no dependency and no private exported symbol.

Revision note: The 2026-08-24 implementation update records the completed C1
scope, failure-path discoveries, final upstream replay, and validation.
