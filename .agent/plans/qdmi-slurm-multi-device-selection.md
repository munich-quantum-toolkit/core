# Select a QDMI device per contended resource from a Slurm license

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date. Maintain this document in accordance with `.agent/PLANS.md`
from the repository root.

## Purpose / Big Picture

`mqt.core.qdmi.slurm.open_device_from_license()` reads the environment variable
`SLURM_JOB_LICENSES` that Slurm sets for a job, and opens the registered QDMI
device whose stable ID equals the license name. A "stable ID" is the `id` field
of a device definition in a QDMI registry file, for example `mqt.ddsim.default`.
A "provider" is a shared library that implements the QDMI device interface for
one vendor.

A provider can serve several quantum computers through one shared library, so it
has more contended machines than libraries. It still needs one Slurm license per
machine, because a license is how a cluster regulates contention. Before this
change, three things hid that from a reader.

The first is documentation. `docs/qdmi/slurm.md` never states that a provider
may register several device IDs over one library, so a reader concludes that the
library is the unit of registration and that a per-machine license can never
match a device ID.

The second is the failure message. When the license name is not a registered ID,
`open_device_from_license()` says so and stops without naming the IDs that are
registered, so the reader cannot see how far off the name is.

The third is the CMake helper `mqt_configure_qdmi_device` in
`cmake/AddMQTQDMIDevice.cmake`. A provider uses it to generate the registry file
that ships beside its library. The helper could add extra device IDs only when
they differed by a device configuration file. A provider that separates its
machines with a session parameter such as `custom2` could not use the helper at
all and had to write the registry file by hand.

After this change, a provider generates a per-machine catalogue with the helper,
a site names each license after one of those IDs, and a wrong name prints the
IDs that would have worked.

## Progress

- [x] (2026-08-20 14:20Z) Extend `mqt_configure_qdmi_device` with a `DEVICES`
      keyword and one shared entry loop.
- [x] (2026-08-20 14:25Z) Name the registered IDs in the unknown-license error.
- [x] (2026-08-20 14:40Z) Document the pattern in `docs/qdmi/slurm.md` and
      `docs/qdmi/configuration.md`.
- [x] (2026-08-20 14:45Z) Add nine configure-rejection tests, one generated
      manifest check, and one adapter test; update `CHANGELOG.md`.
- [x] (2026-08-20 16:10Z) Run the QDMI C++ tests, the Python QDMI tests, the
      documentation build, and the lint hooks.

## Surprises & Discoveries

- Observation: MQT Core already ships the pattern this plan documents.
  `src/qdmi/devices/sc/CMakeLists.txt` registers `mqt.sc.iqm.garnet` and
  `mqt.sc.iqm.emerald` over the same superconducting library as
  `mqt.sc.default`. Evidence: `test/qdmi/registry/test_device_registry.cpp`,
  test `DiscoversGeneratedBuildTreeManifests`, asserts that a clean registry
  holds exactly those four definitions.
- Observation: `docs/qdmi/slurm.md` claims that MQT Core installs persistent
  definitions for two devices, but it installs four. Evidence:
  `registered_device_ids()` reports `mqt.ddsim.default`, `mqt.sc.default`,
  `mqt.sc.iqm.emerald`, and `mqt.sc.iqm.garnet`.
- Observation: rewriting the `CONFIGURATIONS` shorthand into the new form
  reproduces the previous output byte for byte. Evidence: the generated
  `mqt-core-qdmi-sc-device.qdmi.json` is unchanged, and the existing
  `DiscoversGeneratedBuildTreeManifests` and install-verify tests pass.

## Decision Log

- Decision: Answer the design question with the existing registry model rather
  than with a new mapping from license name to device ID plus session
  parameters. Rationale: a device definition already carries session parameters,
  and several definitions may share one library and prefix. A second mapping
  would give one machine two names. Date/Author: 2026-08-20.
- Decision: Let a generated device entry carry only `base-url`, `auth-url`,
  `custom1` through `custom5`, and `device-config-file`. Rationale: the
  generated file is installed with the package and is world-readable. A
  credential or a host-specific path belongs in a trusted registry file such as
  `/etc/mqt-core/qdmi.json`. Date/Author: 2026-08-20.
- Decision: Keep the `CONFIGURATIONS` keyword and rewrite each of its entries
  into the new form before one shared loop emits the JSON. Rationale: existing
  callers inside and outside this repository keep working, and the generated
  bytes for those callers do not change. Date/Author: 2026-08-20.
- Decision: Leave the C++ and Python doc comments of `openDeviceFromLicense`
  unchanged. Rationale: they state the contract of one call, which this change
  does not alter. The deployment model belongs in the prose documentation, and
  editing the binding text would require a stub regeneration for no gain.
  Date/Author: 2026-08-20.

## Outcomes & Retrospective

The purpose is met. A provider with more machines than libraries can now
generate its catalogue with `mqt_configure_qdmi_device`, the documentation
states the rule and shows the registry file and the license line, and a wrong
license name reports what would have worked. The registry itself needed no
change, which confirms that the existing model was already sufficient and only
its surface was missing.

Remaining: nothing in this repository. A provider that ships its own SPANK
plugin still has to name its licenses after the stable IDs it registers.

## Context and Orientation

Four areas matter, all reachable from the repository root.

`cmake/AddMQTQDMIDevice.cmake` holds `mqt_configure_qdmi_device`. The function
generates a JSON file named `<target>.qdmi.json` beside a device library, in the
build tree and in the install tree. That file is a "registry file": it lists
device definitions, each with an `id`, a `library` filename, a QDMI symbol
`prefix`, an `enabled` flag, and an optional `session` object. The function is
public: `src/CMakeLists.txt` installs it and `cmake/mqt-core-config.cmake.in`
includes it, so a provider outside this repository calls it too.

`src/qdmi/driver/DeviceRegistry.cpp` reads registry files. Its function
`parseSessionPatch` accepts the session keys `base-url`, `token`, `auth-file`,
`auth-url`, `username`, `password`, `custom1` through `custom5`, and
`device-config`. Several definitions may name the same `library` and `prefix`;
the Driver initializes the library once and opens one QDMI session per
definition, each with its own session parameters.

`src/qdmi/Slurm.cpp` holds the license adapter. Its file-local function
`parseLicense` validates the value of `SLURM_JOB_LICENSES` and returns the
device ID. Its last check compares that ID against
`qdmi::Driver::get().registeredDeviceIds()`.

`docs/qdmi/slurm.md` is the cluster tutorial and `docs/qdmi/configuration.md` is
the registry reference. Both restate the adapter contract.

## Plan of Work

In `cmake/AddMQTQDMIDevice.cmake`, add the multi-value keyword `DEVICES` to
`mqt_configure_qdmi_device`. Each entry uses the form
`<device-id>|<key>=<value>` with one or more parameters. Add a file-local
function `_mqt_qdmi_session_object` that validates the parameters and returns
the JSON `session` object. Rewrite each `CONFIGURATIONS` entry
`<device-id>|<runtime-file-name>` into `<device-id>|device-config-file=<name>`
so one loop emits every generated entry.

In `src/qdmi/Slurm.cpp`, extend the unknown-ID error so it lists the registered
IDs, or states that no device is registered.

In `docs/qdmi/slurm.md`, correct the list of installed definitions and add a
section that explains why each contended resource needs its own device ID. In
`docs/qdmi/configuration.md`, connect the existing sentence about several
definitions over one library to license naming, and document `DEVICES`.

## Concrete Steps

From the repository root, edit the files named above, then build and test:

    cmake --preset release
    cmake --build --preset release
    ctest --preset release -R qdmi

The CMake rejection tests configure a small sub-project and expect the configure
step to fail, so a passing run reports them as passed through the `WILL_FAIL`
property.

## Validation and Acceptance

From the repository root:

    ./build/release/test/qdmi/mqt-core-qdmi-test --gtest_filter='SlurmAdapter*'
    ./build/release/test/qdmi/registry/mqt-core-qdmi-registry-test
    uv run --no-sync pytest test/python/qdmi
    uvx nox --non-interactive -s docs
    uvx nox -s lint

Inspect one generated registry file and confirm that a device entry carries its
session parameters:

    cat build/release/lib/mqt-core-qdmi-sc-device.qdmi.json

Acceptance: a job whose `SLURM_JOB_LICENSES` names an unregistered device prints
the registered IDs; a provider that passes `DEVICES` to
`mqt_configure_qdmi_device` gets one registry entry per named device, each with
its own session parameters; the documentation states the rule and shows the
registry file and the matching `Licenses=` line.

## Idempotence and Recovery

Every step is repeatable. The generated registry file is rewritten on each
build. A failed CMake configure leaves no artifact that blocks a retry; remove
the affected directory under `build/` and configure again.

## Artifacts and Notes

The generated registry file of the runtime-file test device, after the change:

    {
      "schema-version": 1,
      "qdmi": {
        "devices": [
          {
            "id": "test.runtime-file",
            "library": "libmqt-core-qdmi-runtime-file-device.so",
            "prefix": "TEST_RUNTIME",
            "enabled": true
          },
          {
            "id": "test.runtime-file.variant",
            "library": "libmqt-core-qdmi-runtime-file-device.so",
            "prefix": "TEST_RUNTIME",
            "enabled": true,
            "session": {
              "device-config": {
                "file": "metadata-runtime.json"
              }
            }
          },
          {
            "id": "test.runtime-file.session",
            "library": "libmqt-core-qdmi-runtime-file-device.so",
            "prefix": "TEST_RUNTIME",
            "enabled": true,
            "session": {
              "base-url": "https://device.example",
              "custom3": "session"
            }
          }
        ]
      }
    }

The reported failure of the issue, after the change:

    $ SLURM_JOB_LICENSES=iqm_qc_emerald:1 python -c \
        "from mqt.core.qdmi import slurm; slurm.open_device_from_license()"
    RuntimeError: Slurm license 'iqm_qc_emerald' is not a registered QDMI
    device ID; registered IDs are: mqt.ddsim.default, mqt.sc.default,
    mqt.sc.iqm.emerald, mqt.sc.iqm.garnet

## Interfaces and Dependencies

The CMake helper keeps its name and gains one optional keyword:

    mqt_configure_qdmi_device(<target>
      ID <id> PREFIX <prefix>
      [RUNTIME_FILES <file>...]
      [CONFIGURATIONS "<device-id>|<runtime-file-name>"...]
      [DEVICES "<device-id>|<key>=<value>..."...])

No new library dependency. The generated JSON stays a QDMI registry file with
`schema-version` 1.
