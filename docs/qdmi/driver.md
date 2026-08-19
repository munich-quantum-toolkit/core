---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core QDMI device management

## Objective

MQT Core loads QDMI devices, opens client sessions, and provides owning C++ and
Python wrappers for devices, sites, operations, and jobs. A device definition
contains inert metadata: a stable ID, the native library path, the QDMI symbol
prefix, and default session configuration. Discovering or registering a
definition does not load native code.

{cpp-api:class}`qdmi::DeviceRegistry` stores definitions.
{cpp-api:class}`qdmi::DeviceManager` takes an immutable registry snapshot and
opens fresh device sessions. A manager is not a singleton. A returned device or
derived object owns the session and native-library state that it needs, so it
can outlive its manager and parent wrappers.

MQT Core also provides process-default functions for applications that need one
shared catalog: {cpp-api:func}`qdmi::registerDevice`,
{cpp-api:func}`qdmi::registeredDeviceIds`, and {cpp-api:func}`qdmi::openDevice`.
These functions are the compatibility and adapter interface. Explicit registries
and managers remain isolated from this process-default catalog.

## Building the bundled devices

Standalone MQT Core builds include the DDSIM and superconducting QDMI device
libraries by default. When MQT Core is embedded in another CMake project using
{code}`FetchContent` or {code}`add_subdirectory`, these device libraries are
disabled by default so the consumer does not build implementations it may not
use. Select them independently before making MQT Core available:

- {code}`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`
- {code}`BUILD_MQT_CORE_QDMI_SC_DEVICE`

The QDMI object model, registry, manager, and runtime configuration work without
the bundled devices. Device-specific integration tests run only for the
implementations enabled in a build.

## Python bindings

The Python module exposes owning QDMI entities through {py:mod}`mqt.core.qdmi`.
Its {py:mod}`mqt.core.qdmi.driver` submodule provides
{py:class}`~mqt.core.qdmi.driver.DeviceRegistry`,
{py:class}`~mqt.core.qdmi.driver.DeviceManager`, registration, discovery, and
opening.

## Process-default catalog

The shortest form opens each device registered in the process-default catalog:

```{code-cell} ipython3
from mqt.core.qdmi.driver import open_device, registered_device_ids

for device_id in registered_device_ids():
    device = open_device(device_id)
    print(device.name())
```

Each {py:func}`~mqt.core.qdmi.driver.open_device` call creates a fresh session.
Registration replacement changes later opens. It does not change a live device.

## Isolated registry and manager

Use an explicit registry when a component must not depend on process-global
registration:

```python
from mqt.core.qdmi.driver import DeviceDefinition, DeviceManager, DeviceRegistry

registry = DeviceRegistry([
    DeviceDefinition(
        "example.device",
        "/path/to/libexample-device.so",
        "EXAMPLE",
    )
])
manager = DeviceManager(registry)
device = manager.open("example.device")
```

A manager copies the supplied registry. Later changes to the registry do not
change the manager snapshot.
{py:meth}`~mqt.core.qdmi.driver.DeviceManager.open_all` continues after an
individual open fails and returns successful devices and error messages by
stable ID.

See the [QDMI device configuration guide](configuration.md) for discovery,
precedence, typed session configuration, and relocatable manifests.
