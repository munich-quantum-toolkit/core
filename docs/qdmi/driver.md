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
[MQT Core's NA QDMI Device](na_device.md) or
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

Standalone MQT Core builds include the DDSIM, superconducting, and neutral-atom
QDMI device libraries by default. When MQT Core is embedded in another CMake
project using {code}`FetchContent` or {code}`add_subdirectory`, these device
libraries are disabled by default so the consumer does not build device
implementations it may not use. Set {code}`BUILD_MQT_CORE_QDMI_DEVICES=ON`
before making MQT Core available to include them.

The QDMI driver and FoMaC libraries are available in either configuration.
Device-free builds can register external device libraries through
[QDMI device configuration](configuration.md). When MQT Core's C++ tests are
enabled, tests that require the bundled devices are omitted if
{code}`BUILD_MQT_CORE_QDMI_DEVICES` is disabled.

## Python Bindings

The QDMI Driver is implemented in C++ and exposed to Python via
[{code}`nanobind`](https://nanobind.readthedocs.io/). Direct binding of the QDMI
Client interface functions is not feasible due to technical limitations.
Instead, a FoMaC (Figure of Merits and Constraints) library defines wrapper
classes ({cpp-api:class}`~fomac::Session`, {cpp-api:class}`~fomac::Device`,
{cpp-api:class}`~fomac::Site`, {cpp-api:class}`~fomac::Operation`,
{cpp-api:class}`~fomac::Job`) for the QDMI entities. These classes together with
their methods are then exposed to Python, see
{py:class}`~mqt.core.fomac.Session`, {py:class}`~mqt.core.fomac.Device`,
{py:class}`~mqt.core.fomac.Device.Site`,
{py:class}`~mqt.core.fomac.Device.Operation`, {py:class}`~mqt.core.fomac.Job`.

## Usage

The following example shows how to create a session and get devices from the
QDMI driver.

```{code-cell} ipython3
from mqt.core.fomac import Session

# Create a session to interact with QDMI devices
session = Session()

# Get a list of all available devices
available_devices = session.get_devices()

# Print the name of every device
for device in available_devices:
    print(device.name())

```
