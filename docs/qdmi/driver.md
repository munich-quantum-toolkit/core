# QDMI device management

MQT Core discovers QDMI device definitions without loading their native
libraries. {cpp-api:class}`qdmi::DeviceManager` opens a selected device lazily,
applies its independent session parameters, and returns
{cpp-api:class}`qdmi::Device`. Jobs, sites, operations, and child devices share
the underlying library/session lifetime.

```cpp
#include "qdmi/DeviceManager.hpp"

qdmi::DeviceManager manager;
for (const auto& definition : manager.definitions()) {
  auto device = manager.open(definition.id);
}
```

The equivalent Python API lives in {py:mod}`mqt.core.qdmi`:

```python
from mqt.core.qdmi import DeviceManager

manager = DeviceManager()
for definition in manager.definitions:
    device = manager.open(definition.id)
    print(device.name())
```

Opening one device does not open any other definition. `open_all()` returns a
pair of successful devices and per-ID error messages so one unavailable provider
does not hide the remaining devices.

See [QDMI configuration](configuration.md) for discovery, precedence, and
registration examples.
