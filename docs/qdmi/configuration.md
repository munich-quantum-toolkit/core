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

The driver returns the configured `id` through `QDMI_DEVICE_PROPERTY_ID`,
overriding any device-reported default. Child-device IDs remain optional; the
driver forwards a reported ID or `QDMI_ERROR_NOTSUPPORTED` without generating
child IDs.

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

## Installed device manifests

Python distributions advertise the module containing their device manifests:

```toml
[project.entry-points."mqt.core.qdmi.manifests"]
vendor = "vendor.qdmi"
```

The entry-point name identifies the provider. Its value is a module path, not a
function to import. MQT Core reads the distribution's installed file list and
registers the `*.qdmi.json` files below that module. Discovery imports no
provider modules and loads no device libraries. Invalid entries emit a warning
and are skipped. Installed manifests form the lowest-precedence configuration
layer.

Applications can register a manifest explicitly when a missing or invalid file
must stop startup:

```python
from mqt.core.qdmi import builtin_driver

builtin_driver.add_manifest("vendor/device/example.qdmi.json")
```

Register manifests before opening devices with the MQT Core QDMI driver.
Registering the same file again is harmless; conflicting IDs in distinct
manifests are errors. Configuration becomes fixed after the first successful
session allocation. Failed initialization can be retried with corrected input.

`builtin_driver` always uses the MQT Core QDMI driver, independently of
`MQT_CORE_QDMI_DRIVER`. Standard `Session` and `open_device` calls honor that
environment variable.

## Using configured devices

When the MQT Core QDMI driver initializes a driver session, it opens the
configured definitions. A failure to load one definition does not hide the
remaining devices.

```python
from mqt.core.qdmi import Session, open_device

for discovered in Session().devices:
    print(discovered.id, open_device(discovered.id).name())
```

Set `MQT_CORE_QDMI_CONFIG_FILE` or `MQT_CORE_QDMI_CONFIG_JSON` before the first
driver call. Every {py:func}`~mqt.core.qdmi.open_device` call creates a fresh
driver session and finds the stable ID in the session’s device list. The
returned {py:class}`~mqt.core.qdmi.Device` and any
{py:class}`~mqt.core.qdmi.Device.Site`,
{py:class}`~mqt.core.qdmi.Device.Operation`, or {py:class}`~mqt.core.qdmi.Job`
wrapper derived from it keeps that driver session alive. The session is released
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
reported by the selected QDMI driver. Each job must request one license. For
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

## Installed C++ applications

The MQT Core Python distribution also supplies a CMake package. Use
`find_package(mqt-core)` to link its C++ QDMI library and copy the driver and
selected devices beside your application:

```cmake
find_package(mqt-core CONFIG REQUIRED)
add_executable(my-application main.cpp)
target_link_libraries(my-application PRIVATE MQT::CoreQDMI)
mqt_copy_qdmi_runtime(my-application MQT::CoreQDMIScDevice MQT::CoreQDMI_DDSIM_Device)
```

The helper copies shared libraries, device manifests, and configuration files.
It also copies DLL dependencies on Windows and dependencies shipped beside
installed libraries on Linux and macOS. Static libraries are linked into the
application and need no copy. The application uses its build RPATH during the
build. This also works with a source installation of MQT Core.

Manifests contain library filenames relative to their own directory. Keep each
manifest beside its device library when moving an installation. The MQT Core
QDMI driver discovers manifests beside itself; an explicit
`MQT_CORE_QDMI_CONFIG_FILE` can instead select devices installed elsewhere.

Inside a Core build, omitting the device list copies all devices registered
through `mqt_configure_qdmi_device`. An installed consumer selects the exported
targets it needs, as above.

An external device implementation needs no Core build dependency. It can export
its stable ID and prefix as target metadata:

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

For such a target, `mqt_copy_qdmi_runtime` generates the manifest. Targets with
an existing manifest can export `QDMI_MANIFEST_NAME` instead. Additional files
listed in `QDMI_RUNTIME_FILES` are copied from the device library's directory.
