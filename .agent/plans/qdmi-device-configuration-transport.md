# Add typed QDMI device configuration transport

Status: historical implementation record.

## Goal and scope

MQT Core users can currently configure generic QDMI custom parameters, but
inline device JSON and a device-description file cannot replace one another
atomically when registry layers merge. After this change, a device definition or
an individual `open_device` call can select exactly one typed configuration
source. The Driver maps that source onto QDMI v1 CUSTOM1 or CUSTOM2 only at the
native-session boundary. A small integration test demonstrates that a
higher-precedence file source replaces, rather than coexists with, inherited
inline JSON.

This is the additive first change in a three-part series. It does not alter the
neutral-atom or superconducting providers. Those providers consume this
transport in later changes with their own ExecPlans.

## Constraints

- `DeviceRegistry.cpp` already resolves session file paths relative to the
  declaring JSON or TOML file, so typed device-description paths can use that
  same path-resolution boundary. Evidence: `parseSessionPatch` resolves
  `auth-file` against its `base` argument.

- the combined prototype contained provider integration tests and related test
  build definitions in the same files as transport tests. Evidence:
  reconstruction retained only the session test device and generic metadata
  runtime file so this branch remains additive.

- exercising `mqt_configure_qdmi_device(RUNTIME_FILES ...)` directly on a
  main-tree test target would install a test-only data file as part of MQT Core.
  Evidence: a nested CMake fixture now builds and installs an isolated dummy
  provider, covering the helper without altering the product install surface.

- root-level Markdown files included in the Sphinx documentation must address
  documents relative to the Sphinx source tree. Evidence: Read the Docs could
  not resolve `docs/qdmi/configuration.md`; the explicit MyST `{doc}` target is
  `qdmi/configuration`.

## Decisions

- Represent configuration as
  `optional<variant<InlineDeviceConfiguration, FileDeviceConfiguration>>`.
  Rationale: one optional variant has replacement semantics across registry
  layers and cannot preserve an inherited CUSTOM1 while adding CUSTOM2.

- Reject typed configuration combined with raw CUSTOM1 or CUSTOM2 after defaults
  and overrides are merged. Rationale: those QDMI v1 slots are the adapter
  transport and two simultaneous meanings would be ambiguous.

- Keep provider schemas, loaders, default JSON, calibration, and the
  multi-technology validation CLI out of this change. Rationale: this branch
  must compile and test independently while preserving existing provider
  behavior.

## Outcome and validation

The additive transport layer carries configuration through registry, Driver,
bindings, and generic runtime-file packaging. It does not implement provider
schemas, calibration, or a new validation CLI. Release/documentation builds,
registry/driver/Python tests, CMake runtime fixtures, stubs, and lint passed. PR
`#1967` is the prerequisite for the provider-specific migrations.

## Code and ownership

`include/mqt-core/qdmi/driver/Driver.hpp` defines `DeviceSessionConfig`, the
public C++ value stored in each `DeviceDefinition`.
`src/qdmi/driver/DeviceRegistry.cpp` parses JSON and TOML configuration layers
into patches and merges each optional session field.
`src/qdmi/driver/Driver.cpp` applies per-open overrides and sends the merged
values through the QDMI v1 session-parameter ABI. `bindings/fomac/fomac.cpp`
exposes definitions and opening to Python. `cmake/AddMQTQDMIDevice.cmake` stages
relocatable device libraries and their registration manifests. A runtime file is
a data file that must remain beside a provider library after build, install, or
copying into a static consumer.

## Acceptance

Parsing rejects unknown `device-config` keys, missing or simultaneous `inline`
and `file`, and incorrect types. A relative file becomes an absolute normalized
path relative to its declaring configuration file. A higher-precedence source
replaces the full lower-precedence source. Driver tests observe inline bytes in
CUSTOM1 or a file path in CUSTOM2 and observe no value in the other slot.
Combining the typed field with raw CUSTOM1 or CUSTOM2 throws before session
initialization. Python exposes mutually exclusive ergonomic arguments.

## Interfaces

`qdmi::InlineDeviceConfiguration` owns a `std::string json`.
`qdmi::FileDeviceConfiguration` owns a `std::filesystem::path path`.
`qdmi::DeviceConfigurationSource` is their `std::variant`.
`DeviceSessionConfig::deviceConfiguration` is an optional source. The
implementation uses existing nlohmann JSON, TOML, nanobind, and CMake facilities
and adds no dependency. QDMI v1 CUSTOM1 and CUSTOM2 remain confined to
`Driver.cpp` and provider ABI adapters.
