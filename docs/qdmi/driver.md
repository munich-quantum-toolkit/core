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

The DDSIM device uses the MLIR compiler infrastructure for both OpenQASM and QIR
programs. Its target is skipped when {code}`BUILD_MQT_CORE_MLIR` is {code}`OFF`,
while the QDMI driver and superconducting device remain available.

For example, an embedded simulator consumer can enable only the DDSIM device,
while CUDA-Q can enable the DDSIM and superconducting devices used by its
integration tests.

The QDMI driver and QDMI libraries are available independently. Device-free
builds can register external device libraries through
[QDMI device configuration](configuration.md). C++ test builds require every
bundled device available in the selected build configuration.

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

## Native multi-program jobs

`Device.submit_programs` submits an ordered list of programs with one format and
an optional shot count. A supporting provider returns one job ID and one
lifecycle for the complete list. Result index `i` refers to input program `i`,
regardless of execution order. Results are available only after every program
succeeds. Cancellation and failure apply to the aggregate job.

The list may contain one program. DDSIM supports this case; it does not yet
support larger lists. The superconducting model device does not execute jobs.

```{code-cell} ipython3
from mqt.core.qdmi import ProgramFormat

device = open_device("mqt.ddsim.default")
program = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; x q; c = measure q;'
job = device.submit_programs([program], ProgramFormat.QASM3, 32)
assert job.wait()
assert job.programs_num == 1
assert job.get_counts(program_index=0) == {"1": 32}
```

Pass strings for text formats and bytes for binary formats. Binary payloads
retain every byte, including embedded NULs. Omit `num_shots` to leave the device
default unchanged. Existing single-program calls and result access without an
index continue to work; the default index is zero.

Native multi-program submission is not concurrent submission of independent
jobs. A provider may reject lists with more than one program. Applications and
SDK integrations must then retain their separate single-program workflow; they
must not represent unrelated remote jobs as one native aggregate job.

At the C interface, `QDMI_job_set_programs` replaces the program setter and
copies the whole list atomically. A rejected update leaves the previous list
unchanged. Result retrieval takes a program index. The C++ counterparts are
`Device::submitPrograms`, `Job::getProgramsNum`, and the indexed result methods.
