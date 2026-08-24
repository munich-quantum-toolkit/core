# QDMI device configuration

MQT Core discovers QDMI device definitions from versioned JSON configuration.
Discovery only parses definitions. When the QDMI driver initializes a client
session, it opens the configured native libraries. The stable-ID API opens only
the requested device.

:::{warning}
QDMI configuration is a native-code loading trust boundary. Use configuration
files and device libraries only from trusted sources. Project discovery starts
at the current working directory and searches its parent directories. Before you
process an untrusted checkout, set `MQT_CORE_QDMI_CONFIG_FILE` to an
administrator-controlled file or use a working directory outside that checkout.
The explicit file replaces system, user, and project discovery but retains
packaged device definitions. Treat `MQT_CORE_QDMI_CONFIG_JSON` as trusted input
too.
:::

## Device definitions

The following `qdmi.json` registers one device:

```json
{
  "schema-version": 1,
  "qdmi": {
    "devices": [
      {
        "id": "example.device",
        "library": "libexample-device.so",
        "prefix": "EXAMPLE",
        "enabled": true,
        "session": {
          "base-url": "https://device.example",
          "auth-file": "credentials.json",
          "device-config": {
            "file": "device.json"
          }
        }
      }
    ]
  }
}
```

Every enabled definition requires a stable, unique `id`, a `library`, and a QDMI
symbol `prefix`. The `session` object supports `base-url`, `token`, `auth-file`,
`auth-url`, `username`, `password`, `device-config`, and `custom1` through
`custom5`.

`device-config` selects exactly one provider configuration source:

```json
{"device-config": {"inline": {"schema-version": 1}}}
```

or:

```json
{"device-config": {"file": "device.json"}}
```

The inline value must be a JSON object. A relative file path is resolved against
the registry file that declares it. The complete source is one merge field:
changing from `inline` to `file` at a higher-precedence layer replaces the
inherited inline JSON. The Driver adapts inline JSON to QDMI v1 CUSTOM1 and a
file path to CUSTOM2 when opening the native session. Consequently,
`device-config` cannot be combined with raw `custom1` or `custom2`; CUSTOM3
through CUSTOM5 remain available to providers.

Relative library and authentication-file paths are resolved against the file
that declared them. For `MQT_CORE_QDMI_CONFIG_JSON`, they resolve against the
current working directory.

Unknown keys, invalid types, duplicate IDs within one source, unsupported schema
versions, and incomplete enabled definitions are hard errors. Diagnostics name
the source and configuration path. Credentials and session values are not
included in Driver warnings.

## Discovery and precedence

Definitions are merged field by field by ID, from lowest to highest precedence:

1. generated `*.qdmi.json` fragments packaged beside the MQT Core Driver and
   trusted manifests staged by installed packages;
2. the system `qdmi.json`;
3. the user or XDG `qdmi.json`;
4. the nearest project `qdmi.json`;
5. `MQT_CORE_QDMI_CONFIG_JSON`.

On Unix, file configuration uses `/etc/mqt-core/qdmi.json` and then
`${XDG_CONFIG_HOME}/mqt-core/qdmi.json`, falling back to
`${HOME}/.config/mqt-core/qdmi.json`. On Windows, it uses the corresponding
`mqt-core/qdmi.json` files below `PROGRAMDATA` and `APPDATA`.

An entry containing only its ID and `"enabled": false` masks an inherited
definition. Since definitions are merged field by field, a later definition with
the same ID must explicitly set `"enabled": true` to enable it again. The final
disabled ID remains reserved, so fallback registration cannot silently re-enable
a device that an administrator disabled.

`MQT_CORE_QDMI_CONFIG_FILE` replaces the system, user, and project levels while
retaining packaged built-ins.

## Installed Python package manifests

A Python distribution can advertise one trusted device manifest without
importing its provider package. Add an entry point to the distribution's
`pyproject.toml`:

```toml
[project.entry-points."mqt.core.qdmi.manifests"]
"example.qdmi.json" = "vendor.device"
```

The entry-point name must be the exact, path-free basename of one `*.qdmi.json`
file. The value must be a dotted Python module name that anchors the owning
package. The distribution's wheel `RECORD` must contain exactly one file with
that basename below the corresponding module path. In this example, the path
starts with `vendor/device/`. MQT Core resolves the file through the
distribution metadata. It does not load the entry point or import the provider
module.

Importing {py:mod}`mqt.core.qdmi` stages every valid advertised manifest in the
packaged Driver's lowest-precedence layer. Invalid or ambiguous automatic
entries cause one `RuntimeWarning` each and are skipped. A metadata enumeration
failure causes one warning and skips automatic discovery. Applications can stage
a known manifest explicitly when an error must stop startup:

```python
from pathlib import Path

from mqt.core.qdmi import default_driver

default_driver.add_manifest(Path("vendor/device/example.qdmi.json"))
```

Explicit staging reports malformed manifests, missing libraries, and conflicting
device IDs as errors. Staging the same canonical path more than once is
idempotent, including after the packaged Driver freezes its registry. A new path
cannot be staged after the packaged Driver successfully constructs and freezes
its registry during a session-allocation request. A failed Driver construction
rolls the freeze back so startup can be retried. Staging loads the packaged
Driver library but does not select it as the process's generic Client driver.
Package staging and default targeted opens ignore `MQT_CORE_QDMI_DRIVER` and use
the packaged Driver. An explicit targeted `driver_path` overrides that default.
Standard Client sessions continue to honor the environment override.

## Using configured devices

When the packaged QDMI Driver initializes a Client session, it opens the
configured definitions. A failure to load one definition does not hide the
remaining devices.

```python
from mqt.core.qdmi import ClientSession, open_device

for discovered in ClientSession().devices:
    print(discovered.id, open_device(discovered.id).name())
```

Set `MQT_CORE_QDMI_CONFIG_FILE` or `MQT_CORE_QDMI_CONFIG_JSON` before the first
Driver call. Every {py:func}`~mqt.core.qdmi.open_device` call creates a fresh
Client session and finds the stable ID in the standard Client device list. The
returned {py:class}`~mqt.core.qdmi.Device` and any
{py:class}`~mqt.core.qdmi.Device.Site`,
{py:class}`~mqt.core.qdmi.Device.Operation`, or {py:class}`~mqt.core.qdmi.Job`
wrapper derived from it keeps that Client session alive. The session is released
after the last such wrapper is destroyed.

The equivalent C++ API is {cpp-api:class}`qdmi::Session`. `getDevices()`
enumerates one authenticated session. {cpp-api:func}`qdmi::Session::openDevice`
creates a fresh session and opens one enumerated ID.

Multiple definitions may refer to the same library and prefix. MQT Core reuses
the initialized library while creating a fresh QDMI device session, with its own
session parameters, for every definition.

## Selecting a device from a Slurm license environment

MQT Core provides a mechanism-specific adapter for jobs that use local Slurm
licenses for cluster-wide admission. The license name must equal one stable ID
reported by the selected QDMI Driver. Each job must request one license. For
example:

```bash
sbatch --licenses=mqt.ddsim.default:1 simulation.sh
```

The job can then open the named device:

```python
from mqt.core.qdmi import slurm

device = slurm.open_device_from_license()
```

The equivalent C++ function is `qdmi::slurm::openDeviceFromLicense()` from
`qdmi/Slurm.hpp`. Both functions read `SLURM_JOB_LICENSES`. They accept only
`<device-id>` or `<device-id>:1`. They reject remote, compound, and non-unit
license values.

The adapter opens a fresh device session from the persistent definition. It does
not replace configuration or inject credentials. Each provider defines its own
credential sources. The adapter accepts QDMI device status `IDLE` and `BUSY`. It
rejects all other device states.

`SLURM_JOB_LICENSES` is process-mutable. The adapter uses this value only for
device selection. It does not verify that Slurm allocated the license. It does
not authenticate the caller or authorize access to the device. Provider
credentials must authorize remote devices. The operating system must isolate a
local device when access requires enforcement. A caller can also bypass this
adapter and call {py:func}`~mqt.core.qdmi.open_device` with a stable device ID.
A different Slurm lookup would therefore not make MQT Core an access control
boundary.

A cluster can configure more than one license for a device. For example,
`mqt.ddsim.default:2` permits two independent jobs to request one license each.
The count is a Slurm admission limit. It is not an access permission, a provider
availability check, or a provider queue length.

## Relocatable packages and static consumers

Built-in targets generate manifests beside their runtime libraries in both build
and install trees. Library paths in those fragments contain only the target
filename, so moving an installed tree or Python wheel preserves discovery.
Automatic discovery searches relative to the MQT Core Driver, not every library
loaded by the process. A separately installed Python distribution can use the
`mqt.core.qdmi.manifests` entry point described above. Other applications copy
the manifest beside the Driver or register the definition by stable ID.

A fully static executable has no portable shared-module location. Place the
fragments beside the executable, point `MQT_CORE_QDMI_CONFIG_FILE` at a complete
configuration, or use {cpp-api:func}`qdmi::Driver::registerDevice` and
{cpp-api:func}`qdmi::Driver::open`. No install prefix is compiled into the
manifests.

An installed MQT Core CMake package provides a helper that colocates selected
device libraries and manifests with an executable:

```cmake
find_package(mqt-core CONFIG REQUIRED)
add_executable(my-application main.cpp)
target_link_libraries(my-application PRIVATE MQT::CoreQDMI)
mqt_copy_qdmi_runtime(my-application MQT::CoreQDMIScDevice MQT::CoreQDMI_DDSIM_Device)
```

Inside an MQT Core build, omitting the device list copies every device
registered through `mqt_configure_qdmi_device`. Installed consumers select the
exported device targets they need, as shown above.

An external device implementation does not need MQT Core as a build dependency.
It can export its stable ID and prefix as target metadata:

```cmake
set_target_properties(
  example-device
  PROPERTIES QDMI_DEVICE_ID "example.device"
             QDMI_DEVICE_PREFIX "EXAMPLE")
set_property(
  TARGET example-device
  APPEND
  PROPERTY EXPORT_PROPERTIES QDMI_DEVICE_ID QDMI_DEVICE_PREFIX)
```

When `mqt_copy_qdmi_runtime` receives that built or imported target, it
generates the relocatable manifest while copying the device. Device targets may
also declare `RUNTIME_FILES` through `mqt_configure_qdmi_device`; their exported
`QDMI_RUNTIME_FILES` basenames are copied beside the provider as part of the
same operation.
