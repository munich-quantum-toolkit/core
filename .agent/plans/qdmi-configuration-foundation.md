# Add configurable QDMI devices

Status: historical implementation record.

## Goal and scope

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

## Constraints

- Device registration must not load native code. This permits validation and
  precedence tests to use missing library paths safely. Evidence:
  `DeviceRegistrationTest.RegistrationDoesNotLoadLibraries` registers a missing
  path and observes failure only when opening it.

- A QDMI job is owned by its device implementation, so the device session must
  outlive the job deleter. Evidence:
  `DeviceRegistrationTest.FreshJobRetainsItsDeviceSession` destroys its FoMaC
  `Device` before querying and freeing the `Job`.

- A child QDMI device handle is stored inside its root device wrapper. Evidence:
  `DeviceRegistrationTest.FreshChildDeviceRetainsItsRootSession` destroys the
  root FoMaC wrapper before querying the child, so the child must retain the
  shared root session rather than only its raw handle.

- Each CTest test case may run in a separate process. Tests for fresh sessions
  therefore register their own test device rather than relying on another test
  case.

- Authentication-file paths are resolved while parsing configuration and should
  remain `std::filesystem::path` values until the QDMI C function is called.
  Converting earlier loses the path abstraction and complicates Python
  `PathLike` support.

## Decisions

- Use schema version 1 with a `qdmi.devices` array in JSON and the same device
  array under `[tool.qdmi]` in TOML. Rationale: both formats map to one strict
  parser and can reject unknown keys consistently.

- Merge sources by stable device ID, field by field, from packaged fragments
  through system, user, project, and in-process configuration. Rationale:
  higher-precedence sources can override credentials or disable an inherited
  device without repeating its complete definition.

- Keep registration separate from opening and reserve disabled IDs. Rationale:
  discovery remains free of native-code execution, and fallback registration
  cannot undo an administrator's explicit disable.

- Keep `Driver::open` cached while `fomac::Session::openDevice` creates a fresh
  session with merged overrides. Rationale: the QDMI client catalog needs stable
  process-owned handles, while separate backend instances need different
  credentials and device-session state.

- Represent FoMaC device lifetime with a shared device handle and use aliasing
  shared pointers for child devices. Rationale: `Device`, `Site`, `Operation`,
  and `Job` then retain the exact session they use without parallel opaque
  ownership fields.

- Store authentication files as `std::filesystem::path` in C++ and accept Python
  path-like objects, converting to a stable narrow string only for the QDMI C
  call. Rationale: paths remain natural at C++ and Python boundaries while the C
  ABI remains unchanged.

## Outcome and validation

MQT Core now has one configuration and registration model for built-in and
external QDMI devices. Generated manifests are relocatable, configuration
discovery is deterministic, and disabled IDs remain reserved. Stable-ID FoMaC
opens support per-call overrides and use normal shared-handle semantics to keep
device sessions alive through child wrappers and jobs.

The implementation remains within the current QDMI interfaces. It does not
introduce another public device-manager abstraction or expose stored
credentials. Focused registry, Driver, FoMaC, Python, relocation, and imported
target checks passed, along with stub generation and the full lint suite.

## Code and ownership

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
beside a static consumer. Built-in devices use `mqt_configure_qdmi_device`;
external device targets export neutral metadata consumed by
`mqt_copy_qdmi_runtime`.

FoMaC is the C++ wrapper in `include/mqt-core/fomac/FoMaC.hpp` and
`src/fomac/FoMaC.cpp`. `fomac::Session::openDevice` creates a fresh registered
device session. Its shared device handle is copied into derived `Site`,
`Operation`, and `Job` values; child `Device` objects use an aliasing shared
pointer to retain the root session while addressing the child handle.
`bindings/fomac/fomac.cpp` exposes registration and opening to Python.

## Acceptance

Registry tests must show that `qdmi.json` and `[tool.qdmi]` are discovered,
relative paths resolve against their source, precedence merges by ID, and
disabled IDs remain reserved. Driver tests must show that registration does not
load a library, cached opening is stable, fresh opening merges overrides, and
runtime registrations stay outside the configured client catalog.

FoMaC and Python tests must show that a `Device`, child `Device`, `Site`,
`Operation`, or `Job` remains valid after the wrapper that produced it is
destroyed. Python tests must accept both a string and `pathlib.Path` for
`auth_file`, and the generated stub must advertise path-like input.

CMake tests must configure and build an imported device target using exported
metadata. Strict documentation and lint remain required; the original
documentation check was blocked and must not be treated as a pass.

## Interfaces

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
