# Add configurable QDMI devices

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core applications should discover QDMI device libraries from installed,
system, user, project, and in-process configuration without embedding
machine-specific paths. Applications should also be able to register a device
under a stable ID and create a fresh device session with per-call credentials.

After this change, placing a `qdmi.json` file in a supported configuration
location or adding `[tool.qdmi]` to `pyproject.toml` makes its enabled devices
available to the QDMI client catalog. Python and C++ applications can register
the same definition programmatically, then open it by ID. Each stable-ID FoMaC
open creates a fresh device session, and all wrappers derived from that device
keep the session alive.

## Progress

- [x] (2026-07-15 15:00Z) Added schema-versioned JSON and TOML parsing, source
      precedence, path resolution, disabled-ID reservations, and focused tests.
- [x] (2026-07-15 17:00Z) Added stable-ID registration, cached Driver opens,
      fresh FoMaC opens with session overrides, and explicit catalog snapshots.
- [x] (2026-07-15 19:00Z) Added relocatable CMake manifests for built-in,
      external, and imported device targets.
- [x] (2026-07-16 15:00Z) Added C++ and Python APIs, generated type stubs,
      documentation, upgrade guidance, and lifetime tests.
- [x] (2026-07-24 21:30Z) Aligned names with the implemented QDMI model:
      `qdmi.json`, `[tool.qdmi]`, device terminology, filesystem authentication
      paths, and shared device-handle ownership.
- [x] (2026-07-24 21:45Z) Ran the focused C++ and Python tests, imported-device
      CMake tests, stub generation, full lint suite, and diff audit. The
      documentation build reached Sphinx but could not download the QDMI tag
      file because the local Python installation rejected its TLS certificate.
- [x] (2026-07-25 00:15Z) Added direct coverage that a child `Device` retains
      its fresh root session, completed the required ExecPlan sections, and
      reran focused validation.

## Surprises & Discoveries

- Observation: Device registration must not load native code. This permits
  validation and precedence tests to use missing library paths safely. Evidence:
  `DeviceRegistrationTest.RegistrationDoesNotLoadLibraries` registers a missing
  path and observes failure only when opening it.
- Observation: A QDMI job is owned by its device implementation, so the device
  session must outlive the job deleter. Evidence:
  `DeviceRegistrationTest.FreshJobRetainsItsDeviceSession` destroys its FoMaC
  `Device` before querying and freeing the `Job`.
- Observation: A child QDMI device handle is stored inside its root device
  wrapper. Evidence:
  `DeviceRegistrationTest.FreshChildDeviceRetainsItsRootSession` destroys the
  root FoMaC wrapper before querying the child, so the child must retain the
  shared root session rather than only its raw handle.
- Observation: Each CTest test case may run in a separate process. Tests for
  fresh sessions therefore register their own test device rather than relying on
  another test case.
- Observation: Authentication-file paths are resolved while parsing
  configuration and should remain `std::filesystem::path` values until the QDMI
  C function is called. Converting earlier loses the path abstraction and
  complicates Python `PathLike` support.

## Decision Log

- Decision: Use schema version 1 with a `qdmi.devices` array in JSON and the
  same device array under `[tool.qdmi]` in TOML. Rationale: both formats map to
  one strict parser and can reject unknown keys consistently. Date/Author:
  2026-07-15 / GPT-5 via Codex.
- Decision: Merge sources by stable device ID, field by field, from packaged
  fragments through system, user, project, and in-process configuration.
  Rationale: higher-precedence sources can override credentials or disable an
  inherited device without repeating its complete definition. Date/Author:
  2026-07-15 / GPT-5 via Codex.
- Decision: Keep registration separate from opening and reserve disabled IDs.
  Rationale: discovery remains free of native-code execution, and fallback
  registration cannot undo an administrator's explicit disable. Date/Author:
  2026-07-15 / GPT-5 via Codex.
- Decision: Keep `Driver::open` cached while `fomac::Session::openDevice`
  creates a fresh session with merged overrides. Rationale: the QDMI client
  catalog needs stable process-owned handles, while separate backend instances
  need different credentials and device-session state. Date/Author: 2026-07-16 /
  GPT-5 via Codex.
- Decision: Represent FoMaC device lifetime with a shared device handle and use
  aliasing shared pointers for child devices. Rationale: `Device`, `Site`,
  `Operation`, and `Job` then retain the exact session they use without parallel
  opaque ownership fields. Date/Author: 2026-07-24 / GPT-5.6 via Codex.
- Decision: Store authentication files as `std::filesystem::path` in C++ and
  accept Python path-like objects, converting to a stable narrow string only for
  the QDMI C call. Rationale: paths remain natural at C++ and Python boundaries
  while the C ABI remains unchanged. Date/Author: 2026-07-24 / GPT-5.6 via
  Codex.

## Outcomes & Retrospective

MQT Core now has one configuration and registration model for built-in and
external QDMI devices. Generated manifests are relocatable, configuration
discovery is deterministic, and disabled IDs remain reserved. Stable-ID FoMaC
opens support per-call overrides and use normal shared-handle semantics to keep
device sessions alive through child wrappers and jobs.

The implementation remains within the current QDMI interfaces. It does not
introduce another public device-manager abstraction or expose stored
credentials. The registry, Driver, and FoMaC C++ suites passed with 11, 109, and
174 tests respectively; the focused Python suite passed 175 tests; isolated
fresh-session tests passed four tests and imported-device CMake tests passed
two. Stub generation and the full lint suite passed. Documentation generation
was blocked before source rendering by a local TLS certificate failure while
downloading the external QDMI tag file.

## Context and Orientation

QDMI provides a C client interface and a C device interface. A device library
exports the device-interface symbols with a configured prefix. The MQT Core
Driver in `include/mqt-core/qdmi/driver/Driver.hpp` and
`src/qdmi/driver/Driver.cpp` loads those symbols and exposes client-facing
device handles.

`src/qdmi/driver/DeviceRegistry.cpp` discovers configuration and materializes
`qdmi::DeviceDefinition` values. A definition contains a stable ID, library
path, symbol prefix, and default device-session parameters. Dedicated files are
named `qdmi.json`; project TOML configuration uses `[tool.qdmi]`. Generated
fragments retain the `*.qdmi.json` suffix so multiple installed device libraries
can contribute definitions.

`cmake/AddMQTQDMIDevice.cmake` creates relocatable fragments and records device
target metadata. `mqt_copy_qdmi_runtime` copies selected libraries and manifests
beside a static consumer. Built-in device targets use the same helper as
external targets.

FoMaC is the C++ wrapper in `include/mqt-core/fomac/FoMaC.hpp` and
`src/fomac/FoMaC.cpp`. `fomac::Session::openDevice` creates a fresh registered
device session. Its shared device handle is copied into derived `Site`,
`Operation`, and `Job` values; child `Device` objects use an aliasing shared
pointer to retain the root session while addressing the child handle.
`bindings/fomac/fomac.cpp` exposes registration and opening to Python.

## Milestones

### Milestone 1: Discover and merge device definitions

Add strict JSON and TOML parsing in `src/qdmi/driver/DeviceRegistry.cpp`, then
merge packaged, system, user, project, and in-process definitions by stable ID.
At the end of this milestone, the registry test binary accepts `qdmi.json` and
`[tool.qdmi]`, resolves relative paths against their source file, and preserves
an ID disabled by the highest-precedence source.

### Milestone 2: Register and open stable device IDs

Extend `qdmi::Driver` to register definitions without loading their libraries,
cache normal opens, and create a fresh session for FoMaC with per-call
overrides. At the end of this milestone, focused Driver tests show that broken
libraries are isolated, duplicate and disabled IDs are handled deliberately, and
separate fresh opens have separate device-session state.

### Milestone 3: Preserve FoMaC object lifetimes

Store the QDMI device session in one shared handle throughout FoMaC and use an
aliasing shared pointer for a child handle owned by its root device. At the end
of this milestone, focused tests can destroy the original `Device` and still
query a derived child `Device`, `Site`, `Operation`, or `Job`; destroying the
last derived wrapper releases the session.

### Milestone 4: Package, bind, document, and validate

Generate relocatable device manifests through `cmake/AddMQTQDMIDevice.cmake`,
expose stable-ID operations to Python, regenerate `python/mqt/core/fomac.pyi`,
and document configuration and precedence. Complete the milestone when
imported-device CMake tests, focused C++ and Python tests, stub generation,
warning-as-error documentation, and the full lint suite pass.

## Plan of Work

Implement strict JSON and TOML readers in `src/qdmi/driver/DeviceRegistry.cpp`.
Resolve relative library and authentication-file paths against their declaring
file. Discover packaged fragments, system and user `qdmi.json`, the nearest
project `qdmi.json` or `[tool.qdmi]`, and the JSON environment override in
documented precedence order. Merge definitions by ID and retain final disabled
IDs separately.

Extend the Driver with definition registration, idempotent fallback
registration, cached opening, and a private fresh-session open used by FoMaC.
Copy the initially configured handle set into each QDMI client session so later
runtime registrations do not mutate an allocated session's catalog. Cache
initialized dynamic libraries by normalized path and prefix.

Use `cmake/AddMQTQDMIDevice.cmake` for built-in device targets and export
`MQT_QDMI_DEVICE_ID`, `MQT_QDMI_DEVICE_PREFIX`, and manifest-name metadata. Test
both built and imported targets.

Expose `DeviceDefinition`, `register_device`, `register_device_if_absent`, and
`open_device` in the FoMaC Python module. Accept `str` and path-like
authentication-file values. Regenerate `python/mqt/core/fomac.pyi`; do not edit
generated stubs manually.

Document configuration, precedence, runtime registration, fresh sessions, and
static-consumer packaging in `docs/qdmi/configuration.md`. Keep `CHANGELOG.md`
and `UPGRADING.md` consistent with those names and behaviors.

## Concrete Steps

Run all commands from the repository root. Configure and build the focused
components with:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-qdmi-registry-test mqt-core-qdmi-driver-test mqt-core-fomac-test

Run the focused C++ suites, including isolated fresh-session cases:

    ./.agent/run.sh ./build/release/test/qdmi/registry/mqt-core-qdmi-registry-test
    ./.agent/run.sh ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test
    ./.agent/run.sh ./build/release/test/fomac/mqt-core-fomac-test
    ./.agent/run.sh ctest --test-dir build/release \
      -R 'DeviceRegistrationTest\.(FreshOverridesMergeValuesOwnTheirSessionAndStayOutOfCatalog|FreshOpenCreatesDistinctSessions|FreshJobRetainsItsDeviceSession|FreshChildDeviceRetainsItsRootSession)' \
      --output-on-failure

Regenerate stubs and run focused Python tests:

    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox -s tests-3.14 -- \
      test/python/fomac/test_fomac.py

Build documentation and run the repository lint suite:

    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint

After all cache-producing processes have exited, clean only this worktree's
agent caches:

    ./.agent/clean-caches.sh

## Validation and Acceptance

Registry tests must show that `qdmi.json` and `[tool.qdmi]` are discovered,
relative paths resolve against their source, precedence merges by ID, and
disabled IDs remain reserved. Driver tests must show that registration does not
load a library, cached opening is stable, fresh opening merges overrides, and
runtime registrations stay outside the configured client catalog.

FoMaC and Python tests must show that a `Device`, child `Device`, `Site`,
`Operation`, or `Job` remains valid after the wrapper that produced it is
destroyed. Python tests must accept both a string and `pathlib.Path` for
`auth_file`, and the generated stub must advertise path-like input.

The CMake tests must configure and build an imported device target using only
exported metadata. The documentation build and full lint suite must complete
without errors. The local TLS failure recorded in `Outcomes & Retrospective`
does not relax that acceptance criterion; a validation environment able to
download the QDMI tag file must produce a successful warning-as-error build. A
final diff search, excluding `vendor/tomlplusplus`, must show no superseded
configuration names, historical-version wording, or ambiguous terminology for
device implementations.

## Idempotence and Recovery

Configuration discovery and registration tests use temporary directories and
restore environment variables and current paths. Re-running CMake generation,
builds, tests, stub generation, documentation, and lint is safe. If a generated
stub differs, inspect the binding signature and regenerate it rather than
editing the stub.

If configuration fails because a dependency is unavailable, install the
repository's build and test groups with:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test

Then repeat the failed command. Do not remove another worktree's build or cache
directories and do not use real QDMI credentials during validation.

## Artifacts and Notes

The focused validation evidence at the completed implementation point is:

    registry tests: 11 passed
    Driver tests: 109 passed
    FoMaC tests: 174 passed
    focused Python tests: 175 passed
    isolated fresh-session tests: 4 passed
    imported-device CMake tests: 2 passed
    stub generation and full lint: passed

The documentation command reached Sphinx but stopped before rendering sources
with `SSL: CERTIFICATE_VERIFY_FAILED` while downloading the external QDMI tag
file. This records the local validation gap; it is not an expected application
error or an accepted documentation result.

## Interfaces and Dependencies

`qdmi::DeviceDefinition` and `qdmi::DeviceSessionConfig` remain public input
types in `include/mqt-core/qdmi/driver/Driver.hpp`.
`qdmi::Driver::registerDevice`, `registerDeviceIfAbsent`, and cached `open`
manage stable IDs; the private `openFresh` creates a separately owned session
for `fomac::Session::openDevice`. Authentication files use
`std::filesystem::path` in C++ and path-like values in Python, then convert to a
string only at the QDMI C interface.

`fomac::Device`, `Site`, `Operation`, and `Job` retain a shared
`QDMI_Device_impl_d` handle. A child `Device` uses the aliasing
`std::shared_ptr` constructor so its stored pointer addresses the child while
its control block owns the root. The implementation depends on the QDMI C
headers, `nlohmann_json`, the vendored toml++ header, spdlog, nanobind, CMake,
and platform dynamic-loading APIs. Configuration parsing must not execute native
device code.

Revision note (2026-07-25): Updated this completed plan to match the implemented
device terminology and ownership model, restored the required milestone,
artifact, and interface sections, added direct child-device lifetime coverage,
and distinguished normative documentation acceptance from the recorded local TLS
validation block.
