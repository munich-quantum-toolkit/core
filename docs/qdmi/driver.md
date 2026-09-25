---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Using QDMI drivers

The C++ QDMI library (`MQT::CoreQDMI`) provides owning wrappers for the standard
QDMI Client Interface. Applications can replace the driver without rebuilding.
The driver handles device libraries, configuration, and authorization; the C++
library uses the same interface regardless of the device implementation.

MQT Core supplies the QDMI driver `MQT::CoreQDMIDriver`. It loads devices such
as [the SC device](sc_device.md) and [the DDSIM device](ddsim_device.md).

## Driver selection

Each session selects a driver in this order:

1. `qdmi::SessionConfig::driverPath` or Python `driver_path`;
2. the `MQT_CORE_QDMI_DRIVER` environment variable;
3. the MQT Core QDMI driver.

MQT Core validates the required functions and ABI major/minor versions before
allocating a session. Patch differences are compatible. Sessions may use
different drivers in the same process. Devices and jobs keep their originating
session alive, so opening another driver does not invalidate them. Validated
driver libraries remain loaded for the process lifetime, including calls from
global destructors.

The MQT Core QDMI driver shares device libraries across path aliases with the
same symbol prefix. Independent sessions keep their own parameters. A slow
provider initializer does not block initialization of unrelated providers.

## Opening configured devices

Use `mqt.core.qdmi.builtin_driver.open_device` or
`qdmi::builtin_driver::openDevice` to open one configured device with the MQT
Core QDMI driver. These calls create independent device sessions and accept
per-call overrides of the manifest's session parameters. They do not initialize
unrelated devices.

```python
from mqt.core.qdmi import builtin_driver

device = builtin_driver.open_device("mqt.ddsim.default")
```

Use `builtin_driver.registered_device_ids()` to list enabled configured IDs
without loading device libraries or contacting providers. Register manifests
before the first enumeration or device opening.

The Qiskit `QDMIBackend.from_device_id` factory and PennyLane's
`qml.device("mqt.ddsim.default", wires=4)` use this opening API. Python
`QDMISessionParameters` describes the supported overrides. To use another driver
with these SDKs, pass an already-open `Device` to the backend constructor.

The MQT Core QDMI driver provides two optional private functions for manifest
registration and targeted session allocation. Standard-interface drivers need
neither function. The generic `Session` and `open_device` APIs use only the
standard Client Interface.

## Multi-program jobs

`Device.submit_programs` submits an ordered program list with one format and
common job parameters. `num_shots` applies to each program. Text programs carry
one terminating null byte; binary programs retain their exact bytes.
`try_submit_programs` returns `None` only when the device rejects the list before
submission. Submission errors propagate, since retrying an uncertain submission
could duplicate execution.

Use `job.num_programs` and the optional `program_index` argument on result
methods to retrieve results in input order. `job.program_statuses` reports
individual outcomes when supported, or `None` otherwise. A successful program's
results remain available if another program fails or is cancelled. Cancelling
uses the shared native job handle.

## Building the Bundled Devices

Standalone MQT Core builds include the DDSIM and superconducting QDMI device
libraries by default. When MQT Core is embedded in another CMake project using
{code}`FetchContent` or {code}`add_subdirectory`, these device libraries are
disabled by default so the consumer does not build implementations it may not
use. They can be selected independently before making MQT Core available:

- {code}`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`
- {code}`BUILD_MQT_CORE_QDMI_SC_DEVICE`

The DDSIM device uses the MLIR compiler infrastructure for both OpenQASM and QIR
programs. Its target is skipped when {code}`BUILD_MQT_CORE_MLIR` is {code}`OFF`,
while the QDMI driver and superconducting device remain available.

For example, an embedded simulator consumer can enable only the DDSIM device,
while CUDA-Q can enable the DDSIM and superconducting devices used by its
integration tests.

The driver is a shared library. The C++ QDMI library follows the project’s
static/shared build setting and is shared in Python wheels. Device-free builds
can use another QDMI driver through `driver_path` or `MQT_CORE_QDMI_DRIVER`. The
MQT Core QDMI driver can load external device libraries through
[QDMI device configuration](configuration.md). C++ test builds require the
bundled devices available in the selected build configuration.

## Python Bindings

The C++ QDMI library adds owning wrappers for driver sessions, devices, sites,
operations, and jobs. Each wrapper retains the driver session that owns its raw
handle. The Python module exposes the same entities through
{py:mod}`mqt.core.qdmi`.

Native device opening, property queries, job calls, and compiler-target
snapshots release Python's GIL. Other Python threads can run while a provider
waits for a remote response. Python argument and result conversion still holds
the GIL. Concurrent calls into a shared device or job must satisfy the
provider's thread safety contract; releasing the GIL does not serialize provider
access.

## Usage

Use `device_ids()` to list the stable IDs visible with default session
parameters. `open_device` starts a fresh session and finds the requested ID in
its device list. Use `Session` when authentication or other session parameters
are needed.

```{code-cell} ipython3
from mqt.core.qdmi import device_ids, open_device

for device_id in device_ids():
    device = open_device(device_id)
    print(device.name())
```

All session keywords map to standard QDMI parameters. They are `token`,
`auth_file`, `auth_url`, `username`, `password`, `project_id`, and `custom1`
through `custom5`. The selected driver defines validation, precedence, and the
meaning of these values.
