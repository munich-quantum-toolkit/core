# QDMI device configuration

MQT Core discovers QDMI device definitions from versioned JSON or TOML
configuration. Discovery only parses definitions; a native device library is
loaded when its stable ID is opened. Configuration is therefore trusted input.

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
symbol `prefix`. The optional `session` object supports `base-url`, `token`,
`auth-file`, `auth-url`, `username`, `password`, and `custom1` through
`custom5`.

Relative library and authentication-file paths are resolved against the file
that declared them. Paths in `MQT_CORE_QDMI_CONFIG_JSON` resolve against the
current working directory. Unknown keys, invalid types, duplicate IDs within one
source, unsupported schema versions, and incomplete enabled definitions are
errors whose diagnostics name the source and configuration path.

## Discovery and precedence

Definitions are merged field by field by ID, from lowest to highest precedence:

1. generated `*.qdmi.json` fragments packaged beside MQT Core;
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
definition. A later configuration layer must explicitly set `"enabled": true` to
enable the ID again. Within one directory, `qdmi.json` takes precedence over
`pyproject.toml`. A finally disabled ID remains reserved, so fallback
registration cannot silently override an administrator's choice.

`MQT_CORE_QDMI_CONFIG_FILE` replaces the system, user, and project levels while
retaining packaged definitions.

## Registering and opening devices

Constructing a {cpp-api:class}`qdmi::DeviceRegistry` or the Python
{py:class}`mqt.core.qdmi.DeviceRegistry` without arguments performs standard
discovery. A registry can be extended before it is moved into a device manager:

```cpp
#include "qdmi/DeviceManager.hpp"
#include "qdmi/DeviceRegistry.hpp"

qdmi::DeviceRegistry registry;
registry.registerDeviceIfAbsent({
    .id = "example.device",
    .library = "/path/to/libexample-device.so",
    .prefix = "EXAMPLE",
});

qdmi::DeviceManager manager(std::move(registry));
auto device = manager.open("example.device");
```

`registerDeviceIfAbsent` is intended for device packages that provide a
programmatic fallback. It returns `false` when the stable ID already exists or
is disabled. `registerDevice` rejects duplicates unless `replace` is `true`;
explicit replacement can also re-enable a disabled ID.

The equivalent Python API is:

```python
from mqt.core.qdmi import DeviceDefinition, DeviceManager, DeviceRegistry

registry = DeviceRegistry()
registry.register_device_if_absent(
    DeviceDefinition(
        "example.device",
        "/path/to/libexample-device.so",
        "EXAMPLE",
    )
)
device = DeviceManager(registry).open("example.device")
```

Construct a registry from a list of definitions when configuration discovery is
not wanted:

```python
registry = DeviceRegistry([DeviceDefinition("example.device", "/path/to/device", "EXAMPLE")])
manager = DeviceManager(registry)
```

A manager owns an immutable snapshot of its registry. Each `open` call creates a
fresh QDMI device session and applies the supplied `SessionParameters` over the
definition defaults. Compatible sessions share the initialized native library.
Returned devices, child devices, sites, operations, and jobs retain the state
they need and may outlive the manager.

## Relocatable packages and static consumers

Built-in targets generate manifests beside their runtime libraries in build and
install trees. Library paths in those fragments contain only the target
filename, so moving an installed tree or Python wheel preserves discovery.
Automatic discovery searches relative to MQT Core, not every library loaded by
the process.

A fully static executable has no portable shared-module location. Place the
fragments beside the executable, point `MQT_CORE_QDMI_CONFIG_FILE` at a complete
configuration, or construct an explicit registry.

An installed MQT Core CMake package provides a helper that colocates selected
device libraries and manifests with an executable:

```cmake
find_package(mqt-core CONFIG REQUIRED)
add_executable(my-application main.cpp)
target_link_libraries(my-application PRIVATE MQT::CoreQDMI)
mqt_copy_qdmi_runtime(
  my-application
  MQT::CoreQDMINaDevice
  MQT::CoreQDMIScDevice
  MQT::CoreQDMI_DDSIM_Device)
```

Inside an MQT Core build, omitting the device list copies every device
registered through `mqt_configure_qdmi_device`. Installed consumers select the
exported device targets they need.

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
generates the relocatable manifest while copying the device.
