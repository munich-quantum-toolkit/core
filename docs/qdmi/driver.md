---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# QDMI Client and Driver Runtime

## Objective

MQT Core consumes the standard QDMI 1.4 Client interface. `MQT::CoreQDMI` owns
the C++ wrappers and loads one Client driver at runtime. It does not link to a
specific Driver implementation. `MQT::CoreQDMIDriver` is the packaged shared
Driver. It loads devices such as [the SC QDMI Device](sc_device.md) and
[the DDSIM QDMI Device](ddsim_device.md).

This boundary lets another QDMI 1.4 Driver implement the Client interface
without linking to MQT Core's Driver. MQT Core checks the complete Client
function table and the QDMI Client ABI major and minor versions before it
allocates a session.

## Driver Selection

MQT Core selects the Client driver for the process after the Driver passes ABI
and function-table validation and allocates the first raw session. The selection
order is:

1. `qdmi::SessionConfig::driverPath` or Python `driver_path`;
2. the UTF-8 `MQT_CORE_QDMI_DRIVER` environment value;
3. the packaged `MQT::CoreQDMIDriver` library.

The selection remains active until process exit. A later explicit request for a
different Driver fails. A failed load, ABI check, symbol check, or raw-session
allocation does not select a Driver, so a later call can retry. MQT Core keeps
the selected shared library loaded while its function pointers can be used.

## Optional Packaged-Driver Extension

MQT Core's packaged Driver adds two private symbols to the same shared library
that exports the standard Client interface:

- `MQT_CORE_QDMI_driver_add_manifest_v1` stages a trusted package manifest.
- `MQT_CORE_QDMI_driver_session_alloc_for_device_v1` allocates a session for one
  configured stable ID.

These symbols are an MQT Core extension. They are not part of a public QDMI
header, and another Client driver can omit them. MQT Core resolves them as
optional symbols and calls them only through the extension API. Missing
extension symbols do not prevent standard Client sessions. Generic
{cpp-api:func}`qdmi::Session::openDevice` and Python
{py:func}`mqt.core.qdmi.open_device` enumerate the standard Client device list
and never use the private targeted-session symbol.

Use {cpp-api:func}`qdmi::default_driver::addManifest` or Python
{py:func}`mqt.core.qdmi.default_driver.add_manifest` before the packaged Driver
freezes its registry. Staging the packaged library does not select it as the
generic Client driver. The first successful raw standard or targeted session
allocation selects a Client driver. A later targeted-session initialization or
device query failure does not undo that selection.

By default, the `default_driver` extension resolves MQT Core's packaged Driver
and ignores `MQT_CORE_QDMI_DRIVER`. An explicit `driver_path` overrides that
default for a compatible extension. The process selection rule still prevents
switching Drivers after a successful raw allocation. Standard Client sessions
use the selection order above, including the environment override.

Use {cpp-api:func}`qdmi::default_driver::openDevice` or Python
{py:func}`mqt.core.qdmi.default_driver.open_device` when an application
deliberately depends on the packaged Driver. The targeted call merges manifest
defaults with the supplied JSON or Python overrides. It rejects unsupported
parameters and malformed configuration, propagates device-library status codes,
and requires the session to expose exactly one device. Each call creates an
independent session. The returned device and its derived wrappers retain that
session until the last wrapper is destroyed.

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

The Client and Driver libraries are separate shared libraries. Device-free
builds can use another QDMI 1.4 Driver through `driver_path` or
`MQT_CORE_QDMI_DRIVER`. The packaged Driver can load external device libraries
through [QDMI device configuration](configuration.md). C++ test builds require
the bundled devices available in the selected build configuration.

## Python Bindings

The C++ QDMI library adds owning wrappers for Client sessions, devices, sites,
operations, and jobs. Each wrapper retains the Client session that owns its raw
handle. The Python module exposes the same entities through
{py:mod}`mqt.core.qdmi`.

## Usage

The following example enumerates the devices visible to one authenticated Client
session. Each Driver supplies a stable `id` property. `open_device` starts a
fresh session and finds that ID in the standard Client device list.

```{code-cell} ipython3
from mqt.core.qdmi import ClientSession, open_device

for discovered in ClientSession().devices:
    device = open_device(discovered.id)
    print(device.name())
```

All session keywords map to standard QDMI parameters. They are `token`,
`auth_file`, `auth_url`, `username`, `password`, `project_id`, and `custom1`
through `custom5`. The selected Driver defines validation, precedence, and the
meaning of these values.
