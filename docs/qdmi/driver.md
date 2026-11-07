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
{cpp:class}`qdmi::Driver`, comes with several preloaded devices when the bundled
devices are enabled. Other devices can be loaded dynamically at runtime via
{cpp:func}`qdmi::Driver::registerDevice` and {cpp:func}`qdmi::Driver::open`.
Built-in and external devices can also be registered through
[versioned QDMI device configuration](configuration.md).

## Building the Bundled Devices

Standalone MQT Core builds include the DDSIM and superconducting QDMI device
libraries by default. When MQT Core is embedded in another CMake project using
{code}`FetchContent` or {code}`add_subdirectory`, these device libraries are
disabled by default so the consumer does not build implementations it may not
use. They can be selected independently before making MQT Core available:

- {code}`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`
- {code}`BUILD_MQT_CORE_QDMI_SC_DEVICE`

For example, an embedded simulator consumer can enable only the DDSIM device,
while CUDA-Q can enable the DDSIM and superconducting devices used by its
integration tests.

The QDMI driver and QDMI libraries are available independently. Device-free
builds can register external device libraries through
[QDMI device configuration](configuration.md). When MQT Core's C++ tests are
enabled, device-specific tests are omitted with their corresponding device.

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
    print(device.id, device.name())
```

The read-only {py:attr}`~mqt.core.qdmi.Device.id` property returns the stable ID
of an opened device. Save this value to select the same configured device in a
later call to {py:func}`~mqt.core.qdmi.driver.open_device`. The C++ equivalent
is {cpp:func}`qdmi::Device::getId`.

The driver implements QDMI 1.4.0 program submission through
`QDMI_job_set_programs`. C clients use a zero-based program index when querying
programs, program status, or results.

### Selecting a driver

Use {py:class}`~mqt.core.qdmi.Session` to access devices through a compatible
QDMI Client Interface. Pass `driver_path` to load another driver library. When
this argument is omitted, `MQT_CORE_QDMI_DRIVER` selects the library, or the
session uses MQT Core's built-in driver. The client checks the ABI major and
minor versions and all required symbols before it allocates a session. Patch
versions can differ. Devices and jobs retain their session and driver library.

```{code-cell} ipython3
from mqt.core.qdmi import Session

session = Session()
device = session.get_device("mqt.ddsim.default")
print(device.id)
```

For a replacement driver, use `Session(driver_path="/path/to/driver")` or
`mqt.core.qdmi.open_device(device_id, driver_path="/path/to/driver")`. Driver
sessions accept QDMI authentication parameters such as `token` and `auth_file`.
In C++, set `qdmi::SessionConfig::driverPath` and construct a `qdmi::Session`.
Use `getDeviceIds()` and `getDevice(id)` to access its device catalog.

The existing `mqt.core.qdmi.driver` functions register and open devices with MQT
Core's built-in driver. Those functions keep their per-device configuration and
ignore `MQT_CORE_QDMI_DRIVER`.

### Program lists

`Device.submit_job` accepts one program or an ordered list with a common format
and job parameters. Text programs use strings. For exact-byte submission, pass
bytes with the terminating null byte included for text formats. The shot count
applies to every program in the list.

```{code-cell} ipython3
from mqt.core.qdmi import ProgramFormat

device = open_device("mqt.ddsim.default")
program = "OPENQASM 3; qubit q; bit c; c = measure q;"
job = device.submit_job([program, program], ProgramFormat.QASM3, num_shots=32)
assert job.wait()
for index in range(job.num_programs):
    print(job.get_program_status(index), job.get_counts(index))
```

The C++ `Device::submitJob` overloads accept spans of strings or spans of byte
spans. Use `Job::getNumPrograms()` to inspect the count. Program and result
queries accept an optional index that defaults to `0`, preserving existing
single-program calls. In Python, use `job.get_program(index)` for text or
`job.get_program(bytes, index)` for exact bytes. The `program` and
`program_bytes` properties continue to return the first program.

Use `job.get_program_status(index)` to inspect individual outcomes when the
device supports them. A successful program's results remain accessible when
another program fails or is cancelled. The list shares one job handle, wait
operation, and cancellation operation.

Use `Device.try_submit_job` (C++ `Device::trySubmitJob`) to try list submission.
It returns `None` (`std::nullopt`) if the device rejects the programs before
submission. The caller can then submit separate jobs. Validation,
authentication, and submission errors still raise exceptions and must not
trigger fallback. Text strings can include one final null byte. Embedded null
bytes are invalid.
