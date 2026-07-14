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
  try {
    auto device = manager.open(definition.id);
    // Use this device independently of the manager.
  } catch (const std::exception& error) {
    // An unavailable provider does not affect other definitions.
  }
}
```

The equivalent Python API lives in {py:mod}`mqt.core.qdmi`:

```python
from mqt.core.qdmi import DeviceManager

manager = DeviceManager()
for definition in manager.definitions:
    try:
        device = manager.open(definition.id)
    except RuntimeError as error:
        print(f"{definition.id}: {error}")
        continue
    print(device.name())
```

Discovery is side-effect free. Each `open` call loads and initializes only the
selected definition, and separate definitions can share one loaded provider
library while retaining independent sessions.

See [QDMI configuration](configuration.md) for discovery, precedence, and
registration examples.
