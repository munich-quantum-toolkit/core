---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Using QDMI drivers

The source-installed C++ QDMI library (`MQT::CoreQDMI`) provides owning wrappers
for the standard QDMI Client Interface. Applications can replace the driver
without rebuilding. The library calls every driver through that interface. Each
driver handles its own device implementations, configuration, and authorization.

The builtin MQT Core QDMI driver (`MQT::CoreQDMIDriver`, exposed in Python as
`builtin_driver`) loads devices such as [the SC device](sc_device.md) and
[the DDSIM device](ddsim_device.md).

## Driver selection

Each session selects a driver in this order:

1. `qdmi::SessionConfig::driverPath` or Python `driver_path`;
2. the `MQT_CORE_QDMI_DRIVER` environment variable;
3. the builtin MQT Core QDMI driver.

MQT Core validates the required functions and ABI major/minor versions before
allocating a session. Patch differences are compatible. Applications may use
different drivers in the same process. Devices and jobs keep their originating
session alive, so opening another driver does not invalidate them. Validated
driver libraries remain loaded for the process lifetime, including calls from
global destructors.

The builtin MQT Core QDMI driver shares device libraries across path aliases
with the same symbol prefix. Independent sessions keep their own parameters.
Initializing one device does not block initialization of unrelated devices.

Any compatible driver can be selected through the standard opening API:

```python
from mqt.core.qdmi import open_device

device = open_device("example.device", driver_path="/path/to/libdriver.so", token="...")
```

The driver defines how it uses the token and which devices it exposes. Only the
builtin driver accepts per-device session overrides from manifests.

## Opening configured devices

The following discovery and configuration conveniences are specific to the MQT
Core driver. Use {py:func}`mqt.core.qdmi.builtin_driver.open_device` or
{cpp-api:func}`qdmi::builtin_driver::openDevice` to open one configured device
with the builtin MQT Core QDMI driver. These calls create independent device
sessions and accept per-call overrides of the session settings in its manifest.
They do not initialize unrelated devices.

```python
from mqt.core.qdmi import builtin_driver

device = builtin_driver.open_device("mqt.ddsim.default")
```

Use {py:func}`mqt.core.qdmi.builtin_driver.registered_device_ids` to list
enabled configured IDs without loading device libraries or contacting devices.
Register manifests before the first enumeration or device opening.

The Qiskit `QDMIBackend.from_device_id` factory and PennyLane's
`qml.device("mqt.ddsim.default", wires=4)` use this opening API. Python
{py:class}`mqt.core.typing.QDMISessionParameters` describes the supported
overrides for device session parameters. To use another driver with these SDKs,
pass an already-open `Device` to the backend constructor.

The builtin MQT Core QDMI driver provides optional private functions for
manifest registration, metadata-only ID enumeration, and targeted session
allocation. Standard-interface drivers need none of them. The generic `Session`
and `open_device` APIs use only the standard Client Interface. The driver's C++
implementation is internal; only the Client Interface and these three C
extensions are exported from its shared library.

## Multi-program jobs

`Device.submit_job` accepts one program or an ordered program list with one
format and common job parameters. `num_shots` applies to each program. Text
programs carry one terminating null byte; binary programs retain their exact
bytes. `try_submit_job` returns `None` only when the device reports that it
cannot accept the format or program count before submission. Device
unavailability and submission errors propagate, since retrying an uncertain
submission could duplicate execution.

Use `job.num_programs` and the optional `program_index` argument on result
methods to retrieve results in input order. Use `job.get_program(index)` for
text or `job.get_program(bytes, index)` for exact bytes; a retrieved historical
job may not expose it. `job.get_program_status(index)` reports one outcome when
supported, or `None` otherwise. A successful program's results remain available
if another program fails or is cancelled. Cancelling uses the shared native job
handle.

## Building the Bundled Devices

Standalone MQT Core builds include the DDSIM and superconducting QDMI device
libraries by default. When MQT Core is embedded in another CMake project using
{code}`FetchContent` or {code}`add_subdirectory`, these device libraries are
disabled by default so the consumer does not build implementations it may not
use. They can be selected independently before making MQT Core available:

- {code}`BUILD_MQT_CORE_QDMI_DDSIM_DEVICE`
- {code}`BUILD_MQT_CORE_QDMI_SC_DEVICE`

All source builds require LLVM/MLIR. The DDSIM device uses the compiler for both
OpenQASM and QIR programs.

For example, an embedded simulator consumer can enable only the DDSIM device,
while CUDA-Q can enable the DDSIM and superconducting devices used by its
integration tests.

The driver is a shared library. The C++ QDMI library follows the project’s
static/shared build setting and is shared in Python wheels. Device-free builds
can use another QDMI driver through `driver_path` or `MQT_CORE_QDMI_DRIVER`. The
builtin MQT Core QDMI driver can load external device libraries through
[QDMI device configuration](configuration.md). C++ test builds require both
bundled devices.

## Python Bindings

The C++ QDMI library adds owning wrappers for driver sessions, devices, sites,
operations, and jobs. Each wrapper retains the driver session that owns its raw
handle. The Python module exposes the same entities through
{py:mod}`mqt.core.qdmi`.

Native device opening, property queries, job calls, and compiler-target
snapshots release Python's GIL. Other Python threads can run while a device
waits for a remote response. Python argument and result conversion still holds
the GIL. Concurrent calls into a shared device or job must satisfy the device
implementation's thread safety contract; releasing the GIL does not serialize
device access.

### Custom job parameter types

The `custom1` through `custom5` arguments of
{py:meth}`mqt.core.qdmi.Device.submit_job` and
{py:func}`mqt.core.mlir.submit_program` use the device's documented types.
Strings include a null terminator; `bool`, `int`, and `float` use C++ `bool`,
`int`, and `double`. For a device-defined binary payload, pass nonempty `bytes`:

```python
job = device.submit_job(program, program_format, custom1=b"\x01\x00\xff")
```

QDMI copies raw bytes without a terminator; empty payloads raise `ValueError`.
The device defines their meaning and size. In C++, pass a
`std::span<const std::byte>` whose buffer stays valid until submission returns.
The bytes describe a local QDMI ABI value, not a network encoding.

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
