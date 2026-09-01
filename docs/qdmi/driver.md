---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core's QDMI Driver Implementation

## Objective

A QDMI Driver manages the communication between QDMI devices, such as
[MQT Core's SC QDMI Device](sc_device.md) or
[MQT Core's DDSIM QDMI Device](ddsim_device.md), and QDMI clients, see the
[QDMI specification](https://munich-quantum-software-stack.github.io/QDMI/).
It is responsible for loading the device, forwarding requests from the client to
the device, and sending back the results. MQT Core's QDMI Driver,
{cpp-api:class}`qdmi::Driver`, comes with several preloaded devices when the
bundled devices are enabled. Other devices can be loaded dynamically at runtime
via {cpp-api:func}`qdmi::Driver::registerDevice` and
{cpp-api:func}`qdmi::Driver::open`. Built-in and external devices can also be
registered through
[versioned QDMI device configuration](configuration.md).

## Building the Bundled Devices

Standalone MQT Core builds include the DDSIM and superconducting QDMI device
libraries by default. When MQT Core is embedded in another CMake project using
{code}`FetchContent` or {code}`add_subdirectory`, these device libraries are
disabled by default so the consumer does not build implementations it may not
use. They can be selected independently before making MQT Core available:

- {code}`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`
- {code}`BUILD_MQT_CORE_QDMI_SC_DEVICE`

The {code}`BUILD_MQT_CORE_MLIR` option controls the MLIR compiler infrastructure
and QIR support in the DDSIM device. A build with this option set to {code}`OFF`
retains the DDSIM device's OpenQASM 2 and OpenQASM 3 support.

For example, an embedded simulator consumer can enable only the DDSIM device,
while CUDA-Q can enable the DDSIM and superconducting devices used by its
integration tests.

The QDMI driver and QDMI libraries are available independently. Device-free
builds can register external device libraries through
[QDMI device configuration](configuration.md). Building MQT Core's C++ tests
requires both bundled devices so that the complete device integration is tested.

## Python Bindings

The QDMI interface is the low-level contract implemented by a QDMI device. The
MQT Core QDMI driver loads device libraries and implements the QDMI client
interface. The C++ QDMI library adds owning wrappers for QDMI devices, sites,
operations, and jobs. The Python module exposes these QDMI entities through
{py:mod}`mqt.core.qdmi`. Its {py:mod}`mqt.core.qdmi.driver` submodule provides
device discovery, registration, and opening.

## Usage

The following example opens each registered device by its stable ID.

```{code-cell} ipython3
from mqt.core.qdmi.driver import open_device, registered_device_ids

for device_id in registered_device_ids():
    device = open_device(device_id)
    print(device.name())
```
