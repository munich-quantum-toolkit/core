---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core's QDMI Driver Implementation

## Objective

A QDMI Driver manages the communication between QDMI devices, such as [MQT Core's NA QDMI Device](na_device.md), and QDMI clients, see the [QDMI specification](https://munich-quantum-software-stack.github.io/QDMI/).
It is responsible for loading the device, forwarding requests from the client to the device, and sending back the results.
The MQT Core's QDMI Driver, {cpp:class}`qdmi::Driver`, comes with the [MQT Core's NA QDMI Device](na_device.md) that is already statically linked into the driver and can directly be used.
Other devices can be loaded dynamically at runtime via {cpp:func}`qdmi::Driver::addDynamicDeviceLibrary`.

## Python Bindings

The QDMI Driver is implemented in C++ and exposed to Python via [pybind11](https://pybind11.readthedocs.io).
Direct binding of the QDMI Client interface functions is not feasible due to technical limitations.
Instead, a FoMaC (Figure of Merits and Constraints) library defines wrapper classes ({cpp:class}`fomac::Device`, {cpp:class}`fomac::Site`, {cpp:class}`fomac::Operation`) for the QDMI entities.
These classes together with their methods are then exposed to Python, see {py:class}`~mqt.core.qdmi.fomac.Device`, {py:class}`~mqt.core.qdmi.fomac.Site`, {py:class}`~mqt.core.qdmi.fomac.Operation`.

## Usage

The following example shows how to get a device from the QDMI driver and access its name.

```{code-cell} ipython3
from mqt.core.qdmi.fomac import devices

# get a list of all available devices
available_devices = devices()

# print the name of every device (by default there is only the NA device)
for device in available_devices:
    print(device.name())
```
