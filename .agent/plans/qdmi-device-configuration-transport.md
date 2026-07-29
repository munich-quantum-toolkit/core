# Add typed QDMI device configuration transport

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-07-30 00:00Z) Created a clean worktree from current `origin/main`
  and reviewed repository policy and the completed combined prototype.
- [x] (2026-07-30 00:00Z) Reconstructed only the typed C++ model, atomic
  registry merge, Driver adapter, Python arguments, and generic runtime-file
  propagation.
- [x] (2026-07-30 00:00Z) Reconstructed focused registry, Driver, Python, and
  imported-target tests without any provider behavior changes.
- [x] (2026-07-30 01:15 CEST) Regenerated Python stubs and confirmed that
  `python/mqt/core/fomac.pyi` is the only generated source change.
- [x] (2026-07-30 01:15 CEST) Configured and built the release tree, built the
  documentation, and passed focused C++, Python, CMake, and full lint
  validation.
- [x] (2026-07-30 01:15 CEST) Audited the complete diff against the PR A
  boundary; it contains no neutral-atom or superconducting provider changes.

## Surprises & Discoveries

- Observation: `DeviceRegistry.cpp` already resolves session file paths relative
  to the declaring JSON or TOML file, so typed device-description paths can use
  that same path-resolution boundary. Evidence: `parseSessionPatch` resolves
  `auth-file` against its `base` argument.
- Observation: the combined prototype contained provider integration tests and
  related test build definitions in the same files as transport tests. Evidence:
  reconstruction retained only the session test device and generic metadata
  runtime file so this branch remains additive.
- Observation: exercising `mqt_configure_qdmi_device(RUNTIME_FILES ...)`
  directly on a main-tree test target would install a test-only data file as
  part of MQT Core. Evidence: a nested CMake fixture now builds and installs an
  isolated dummy provider, covering the helper without altering the product
  install surface.

## Decision Log

- Decision: Represent configuration as
  `optional<variant<InlineDeviceConfiguration, FileDeviceConfiguration>>`.
  Rationale: one optional variant has replacement semantics across registry
  layers and cannot preserve an inherited CUSTOM1 while adding CUSTOM2.
  Date/Author: 2026-07-30 / Codex.
- Decision: Reject typed configuration combined with raw CUSTOM1 or CUSTOM2
  after defaults and overrides are merged. Rationale: those QDMI v1 slots are
  the adapter transport and two simultaneous meanings would be ambiguous.
  Date/Author: 2026-07-30 / Codex.
- Decision: Keep provider schemas, loaders, default JSON, calibration, and the
  multi-technology validation CLI out of this change. Rationale: this branch
  must compile and test independently while preserving existing provider
  behavior. Date/Author: 2026-07-30 / Codex.

## Outcomes & Retrospective

PR A is reconstructed as an independently buildable transport layer. The
release build and documentation build succeeded. Focused validation passed:
15/15 DeviceRegistry tests, 111/111 Driver tests, the selected Python test, and
6/6 CMake imported-target/runtime-file fixture tests. Stub generation changed
only the expected FoMaC `.pyi`, the full repository lint session passed, and
`git diff --check` reported no whitespace errors.

The final scope audit found changes only in the public configuration type,
registry and Driver transport, Python bindings/stub, generic CMake runtime-file
helper, documentation, and their focused tests. No NA or SC provider
implementation, schema, default configuration, calibration logic, or unified
validation CLI is present. The changelog remains intentionally deferred until a
PR number exists, per the repository PR workflow.

## Context and Orientation

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

## Plan of Work

Add public source wrapper structs and the variant alias in `Driver.hpp`, then
add the optional field to `DeviceSessionConfig`. Extend the registry patch type
and JSON parser to accept `session.device-config` with exactly one of `inline`
and `file`. Serialize the inline JSON subtree to a compact string and resolve a
file value against the declaring configuration directory. Merge the optional
variant atomically.

In `Driver.cpp`, merge this field as one value, reject conflicts with raw
CUSTOM1 and CUSTOM2, and pass inline content through CUSTOM1 or a file path
through CUSTOM2. Keep CUSTOM3 through CUSTOM5 unchanged. Extend the nanobind
helpers and both `DeviceDefinition` and `open_device` APIs with mutually
exclusive `device_config` and `device_config_file` arguments.

Extend `mqt_configure_qdmi_device` with `RUNTIME_FILES`. Copy each input beside
the provider after build and install it beside the provider. Export only
basenames through `QDMI_RUNTIME_FILES`. Extend `mqt_copy_qdmi_runtime` to copy
those files from built and imported targets.

Add registry parsing and override tests, Driver adapter and conflict tests,
Python argument tests, and imported-target/runtime-copy coverage. Update
`docs/qdmi/configuration.md`.

## Concrete Steps

Run commands from the repository root through `.agent/run.sh` when they create
tool caches or build output:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./build/release/test/qdmi/registry/mqt-core-qdmi-registry-test
    ./build/release/test/qdmi/driver/mqt-core-qdmi-driver-test
    ./.agent/run.sh uvx nox -s tests-3.14 -- test/python/fomac/test_fomac.py -k device_configuration_arguments
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox -s lint

Successful GoogleTest runs report zero failed tests. Stub generation must leave
only expected generated `.pyi` changes. The imported-device CTest must configure
and build a consumer that receives the runtime data file beside the copied
library.

## Validation and Acceptance

Parsing rejects unknown `device-config` keys, missing or simultaneous `inline`
and `file`, and incorrect types. A relative file becomes an absolute normalized
path relative to its declaring configuration file. A higher-precedence source
replaces the full lower-precedence source. Driver tests observe inline bytes in
CUSTOM1 or a file path in CUSTOM2 and observe no value in the other slot.
Combining the typed field with raw CUSTOM1 or CUSTOM2 throws before session
initialization. Python exposes mutually exclusive ergonomic arguments.

## Idempotence and Recovery

Configuration, build, test, stub, and lint commands are repeatable. Generated
stubs are regenerated from bindings rather than edited. If a registry parse or
session open fails, no global registry definition or initialized native session
is committed. No command in this plan publishes remote state.

## Artifacts and Notes

The CTest fixture names are `mqt-core-qdmi-imported-device-configure`,
`mqt-core-qdmi-imported-device-build`,
`mqt-core-qdmi-runtime-file-configure`,
`mqt-core-qdmi-runtime-file-build`,
`mqt-core-qdmi-runtime-file-install`, and
`mqt-core-qdmi-runtime-file-install-verify`. The exported runtime-file list is
`metadata-runtime.json`; both the imported consumer and isolated install
verified byte-identical copies beside their runtime artifacts.

## Interfaces and Dependencies

`qdmi::InlineDeviceConfiguration` owns a `std::string json`.
`qdmi::FileDeviceConfiguration` owns a `std::filesystem::path path`.
`qdmi::DeviceConfigurationSource` is their `std::variant`.
`DeviceSessionConfig::deviceConfiguration` is an optional source. The
implementation uses existing nlohmann JSON, TOML, nanobind, and CMake facilities
and adds no dependency. QDMI v1 CUSTOM1 and CUSTOM2 remain confined to
`Driver.cpp` and provider ABI adapters.

Revision note: reconstructed as an independently reviewable additive change on
2026-07-30; provider migrations remain in the later NA and SC changes.
