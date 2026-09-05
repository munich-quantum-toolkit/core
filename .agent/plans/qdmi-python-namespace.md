# Establish the QDMI Python namespace

Status: historical implementation record.

The transitional FoMaC aliases were subsequently removed; see
[compatibility API removal](remove-python-compatibility-apis.md).

## Goal and scope

Python users must be able to distinguish QDMI entities from the layer that
registers and opens QDMI devices. After this change, users import entities from
`mqt.core.qdmi` and device-registry operations from `mqt.core.qdmi.driver`. MQT
Core v3 programs that import `mqt.core.fomac` continue to work, but the
compatibility module warns that it will be removed in MQT Core 4.0. The C++
FoMaC API does not change.

The result is visible by importing `Device` from `mqt.core.qdmi`, opening
`mqt.ddsim.default` through `mqt.core.qdmi.driver.open_device`, and running a
job. The compatibility module re-exports the same objects.

## Constraints

- The Python `Session` does not own a device registry. Explicit device opening
  delegates to `qdmi::Driver::get().openFresh`.

- Native ABI compatibility does not guarantee identical configured providers.
  Validate packaged device discovery in a clean installation so stale build
  configuration cannot mask missing providers.

## Decisions

- Expose registry operations as functions in `mqt.core.qdmi.driver`. Rationale:
  The present API does not provide isolated registry instances.

- Keep C++ FoMaC names and the `MQT::CoreFoMaC` target. Rationale: FoMaC remains
  the C++ wrapper library above the QDMI client interface. The change only
  corrects the public Python namespace.

- Make `mqt.core.fomac` a Python module that aliases native objects. Rationale:
  Aliasing preserves exact type identity and v3 source compatibility without a
  second extension or wrapper hierarchy.

- Publish the neutral-atom specialization as `mqt.core.na.qdmi` and keep
  `mqt.core.na.fomac` as a v3 alias. Rationale: This completes the Python
  namespace transition without duplicating native wrapper types.

## Outcome and validation

The native import, compatibility identity, device operations, and affected
Python integrations pass focused tests. The recursive stubs, editable build,
wheel build and isolated wheel install also pass. Documentation and all lint
hooks pass. Final hosted CI was not recorded.

## Code and ownership

QDMI defines a low-level device interface and a client interface. QDMI device
libraries implement the device interface. `src/qdmi/driver` loads these
libraries and implements the client interface for MQT Core. `src/fomac` and
`include/mqt-core/fomac` contain the C++ FoMaC wrappers that give QDMI handles
normal C++ ownership and methods.

Before this change, `bindings/fomac/fomac.cpp` exposed the wrappers as the
native module `mqt.core.fomac`. The same binding also exposes stable device
registration and opening functions backed by the singleton C++ driver.
`noxfile.py` runs `nanobind.stubgen` for installed native modules, and generated
stubs live in `python/mqt/core`.

## Acceptance

The namespace test must prove that QDMI entities live in `mqt.core.qdmi`, that
`Site` and `Operation` remain nested under `Device`, and that the compatibility
module exports the exact QDMI and driver objects. Constructing `driver.Session`
and importing `mqt.core.fomac` must each emit the documented warning.

The QDMI Python tests must still cover runtime registration, custom operation
queries, queue properties, job retrieval, device and job ownership, binary
programs, and execution. Qiskit, PennyLane, MLIR, and neutral-atom tests must
import the new module. A wheel and an editable installation must import on all
supported platforms. Generated-file, type, spelling, documentation, and lint
checks must pass.

## Interfaces

The final Python interfaces include:

    from mqt.core.qdmi.driver import (
        open_device,
        register_device,
        register_device_if_absent,
        registered_device_ids,
    )
    register_device(definition)
    register_device_if_absent(definition)
    registered_device_ids()
    open_device(device_id, **device_session_overrides)

`mqt.core.qdmi.Device`, `mqt.core.qdmi.Job`, `mqt.core.qdmi.ProgramFormat`,
`mqt.core.qdmi.CustomProperty`, and the nested status enumerations remain native
objects. `driver.DeviceDefinition` and `driver.Session` remain native objects;
`driver.Session` is a deprecated v3 compatibility API. The C++ FoMaC headers,
namespace, library, and CMake target remain unchanged.
