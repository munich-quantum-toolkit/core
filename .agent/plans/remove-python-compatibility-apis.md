# Remove the deprecated QDMI Python compatibility APIs

Status: historical implementation record.

## Goal and scope

MQT Core 4 removes the FoMaC compatibility name from its C++ and Python QDMI
APIs. Python users import QDMI entities from `mqt.core.qdmi`, use registry and
device-opening functions from `mqt.core.qdmi.driver`, and import the
neutral-atom specialization from `mqt.core.na.qdmi`. Imports of `mqt.core.fomac`
and `mqt.core.na.fomac` fail, and `mqt.core.qdmi.driver.Session` is absent. C++
users include `qdmi/Client.hpp` and use the `qdmi` namespace and `MQT::CoreQDMI`
target.

The result is visible in a clean wheel installation. The supported modules
import and open registered devices. The removed modules do not exist in the
wheel and cannot be imported.

## Constraints

- The neutral-atom compatibility path is a native nanobind submodule, while
  `mqt.core.fomac` is a Python file. Evidence: `bindings/na/register_na.cpp`
  creates `fomac`, and `python/mqt/core/fomac.py` re-exports QDMI objects.

- The deprecated Python `Session` owns no independent registry. The supported
  module functions use the process-wide QDMI driver. Removing the binding
  therefore removes no capability that is unique to an isolated Python object.

- `nanobind.stubgen` regenerates exported modules but does not delete a stale
  stub for a removed module. The change must therefore delete
  `python/mqt/core/na/fomac.pyi` explicitly. A clean checkout does not recreate
  that file because the native module no longer exports `fomac`.

## Decisions

- Deliver the removals in one pull request. Rationale: Runtime exports,
  generated stubs, tests, and migration documentation describe one atomic Python
  compatibility break.

- Do not add tombstone modules or attribute fallbacks. Rationale: MQT Core 3
  contains the deprecation path. MQT Core 4 should fail removed imports directly
  and expose only the supported namespace.

- Remove the FoMaC name from both language APIs in MQT Core 4. Rationale:
  Keeping the C++ namespace, installed headers, and CMake targets would leave
  two names for the same QDMI client abstraction after the Python transition.
  The major release is the correct boundary for one complete rename.

## Outcome and validation

MQT Core now exposes the QDMI API through the `qdmi` C++ namespace,
`MQT::CoreQDMI`, `mqt.core.qdmi`, `mqt.core.qdmi.driver`, and
`mqt.core.na.qdmi`. The clean wheel contains the supported Python modules, has
no compatibility-module file, and exposes no Python `Session` class. Installed
C++ packages contain no FoMaC header, namespace, library, or target.

The final review removed an unrelated ownership statement from the driver guide.
The migration guide is the single place that explains the removed APIs and their
replacements. No compatibility tombstone or duplicate runtime path remains.

## Code and ownership

QDMI defines a low-level device interface and a client interface. QDMI device
libraries implement the device interface. `src/qdmi/driver` loads these
libraries and implements the client interface for MQT Core.
`src/qdmi/Client.cpp` and `include/mqt-core/qdmi/Client.hpp` provide the owning
C++ wrappers.

The native extension in `bindings/qdmi/qdmi.cpp` exposes QDMI entities as
`mqt.core.qdmi` and registry operations as `mqt.core.qdmi.driver`. It also
exposes the deprecated Python `Session` class. `python/mqt/core/fomac.py`
re-exports objects from both supported modules. The neutral-atom extension in
`bindings/na/register_na.cpp` creates both `qdmi` and deprecated `fomac`
submodules. Nox invokes `nanobind.stubgen`; the generated `.pyi` files live in
`python/mqt/core` and must not be edited by hand.

MQT Core 3 must receive the already merged namespace and Slurm changes before
this pull request merges. Those backports preserve the announced deprecation
window. This pull request targets only `main` and must not be backported.

## Acceptance

The QDMI test suite must still register devices, list stable IDs, open fresh
device sessions, apply per-open configuration, retain device and job lifetimes,
and execute jobs. The C++ build and installed CMake package must expose only the
QDMI names.

Stub generation must produce no manual edits. Documentation must build with
warnings treated as errors. Lint must pass without broad suppressions. A clean
wheel must contain the three supported modules and no file or native export for
the removed paths. Platform CI must verify Windows, free-threaded Python, and
all supported wheel targets.

## Interfaces

The final supported Python interfaces are:

    from mqt.core import qdmi
    from mqt.core.na import qdmi as na_qdmi
    from mqt.core.qdmi.driver import (
        DeviceDefinition,
        open_device,
        register_device,
        register_device_if_absent,
        registered_device_ids,
    )

No new dependency is required. `mqt.core.qdmi.driver` continues to delegate to
the process-wide `qdmi::Driver`. Returned C++ and Python devices continue to own
the provider library and native device session through the QDMI client wrappers.
