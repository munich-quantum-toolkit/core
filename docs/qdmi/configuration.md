# QDMI device configuration

MQT Core discovers QDMI device definitions from versioned JSON or TOML
configuration. Discovery only parses definitions. The QDMI client opens the
configured native libraries when its first session is allocated, while stable-ID
APIs open only the requested device. Configuration is therefore a trusted input.

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
          "custom1": "device-specific"
        }
      }
    ]
  }
}
```

Every enabled definition requires a stable, unique `id`, a `library`, and a QDMI
symbol `prefix`. The `session` object supports `base-url`, `token`, `auth-file`,
`auth-url`, `username`, `password`, and `custom1` through `custom5`.

Relative library and authentication-file paths are resolved against the file
that declared them. For `MQT_CORE_QDMI_CONFIG_JSON`, they resolve against the
current working directory.

Unknown keys, invalid types, duplicate IDs within one source, unsupported schema
versions, and incomplete enabled definitions are hard errors. Diagnostics name
the source and configuration path. Credentials and session values are not
included in Driver warnings.

## Discovery and precedence

Definitions are merged field by field by ID, from lowest to highest precedence:

1. generated `*.qdmi.json` fragments packaged beside the MQT Core Driver;
2. the system `qdmi.json`;
3. the user or XDG `qdmi.json`;
4. the nearest project `qdmi.json`, or `[tool.qdmi]` in `pyproject.toml` when no
   dedicated file exists in that directory;
5. `MQT_CORE_QDMI_CONFIG_JSON`.

On Unix, file configuration uses `/etc/mqt-core/qdmi.json` and then
`${XDG_CONFIG_HOME}/mqt-core/qdmi.json`, falling back to
`${HOME}/.config/mqt-core/qdmi.json`. On Windows, it uses the corresponding
`mqt-core/qdmi.json` files below `PROGRAMDATA` and `APPDATA`.

An entry containing only its ID and `"enabled": false` masks an inherited
definition. A later complete definition with the same ID enables it again.
Within one directory, `qdmi.json` takes precedence over `pyproject.toml`. The
final disabled ID remains reserved, so fallback registration cannot silently
re-enable a device that an administrator disabled.

`MQT_CORE_QDMI_CONFIG_FILE` replaces the system, user, and project levels while
retaining packaged built-ins.

## Using configured devices

The Driver opens the discovered definitions when the first client session is
allocated, and a failure to load one definition does not hide the remaining
devices. Registration alone does not initialize device libraries.

```python
from mqt.core.fomac import Session

for device in Session().get_devices():
    print(device.name())
```

Set `MQT_CORE_QDMI_CONFIG_FILE` or `MQT_CORE_QDMI_CONFIG_JSON` before creating
the first session. Applications can also register a definition without loading
its library and open it later by stable ID:

```python
from mqt.core.fomac import DeviceDefinition, open_device, register_device

register_device(
    DeviceDefinition(
        "example.device",
        "/path/to/libexample-device.so",
        "EXAMPLE",
        base_url="https://device.example",
    )
)
device = open_device("example.device")
```

Every `open_device` call creates a fresh device session while preserving the
registered defaults and stable ID. This lets separate backend instances use
different credentials without registering process-lifetime UUIDs. The returned
`Device` and any child `Device`, `Site`, `Operation`, or `Job` wrappers derived
from it keep that fresh device session alive. The session is released after the
last such wrapper is destroyed.

Code paths that may be imported more than once can use
`register_device_if_absent(definition)`. It returns whether the definition was
inserted and ignores an existing or explicitly disabled stable ID; malformed
definitions still raise an error.

The equivalent C++ registration operation is `qdmi::Driver::registerDevice`.
Duplicate IDs are rejected unless `replace` is true, and an opened definition
cannot be replaced. `qdmi::Driver::open(id)` returns the cached device.
`fomac::Session::openDevice(id, overrides)` returns a fresh device session and
does not add it to the QDMI client catalog. Runtime registrations and explicit
opens are not added to that catalog.

Multiple definitions may refer to the same library and prefix. MQT Core reuses
the initialized library while creating a fresh QDMI device session, with its own
session parameters, for every definition.

## Relocatable packages and static consumers

Built-in targets generate manifests beside their runtime libraries in both build
and install trees. Library paths in those fragments contain only the target
filename, so moving an installed tree or Python wheel preserves discovery.
Automatic discovery searches relative to the MQT Core Driver, not every library
loaded by the process. An application using a separately installed device
implementation therefore copies its manifest beside the Driver or registers its
definition by stable ID.

A fully static executable has no portable shared-module location. Place the
fragments beside the executable, point `MQT_CORE_QDMI_CONFIG_FILE` at a complete
configuration, or use `qdmi::Driver::registerDevice` and `qdmi::Driver::open`.
No install prefix is compiled into the manifests.

An installed MQT Core CMake package provides a helper that colocates selected
device libraries and manifests with an executable:

```cmake
find_package(mqt-core CONFIG REQUIRED)
add_executable(my-application main.cpp)
target_link_libraries(my-application PRIVATE MQT::CoreFoMaC)
mqt_copy_qdmi_runtime(
  my-application
  MQT::CoreQDMINaDevice
  MQT::CoreQDMIScDevice
  MQT::CoreQDMI_DDSIM_Device)
```

Inside an MQT Core build, omitting the device list copies every device
registered through `mqt_configure_qdmi_device`. Installed consumers select the
exported device targets they need, as shown above.

An external device implementation does not need MQT Core as a build dependency.
It can export its stable ID and prefix as target metadata:

```cmake
set_target_properties(
  example-device
  PROPERTIES MQT_QDMI_DEVICE_ID "example.device"
             MQT_QDMI_DEVICE_PREFIX "EXAMPLE")
set_property(
  TARGET example-device
  APPEND
  PROPERTY EXPORT_PROPERTIES MQT_QDMI_DEVICE_ID MQT_QDMI_DEVICE_PREFIX)
```

When `mqt_copy_qdmi_runtime` receives that built or imported target, it
generates the relocatable manifest while copying the device.
