# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list
of changes including minor and patch releases, please refer to the
[changelog](CHANGELOG.md).

## [Unreleased]

### QDMI device management has been redesigned

The legacy device-management namespace, Python module, global session object,
and compatibility CMake target have been removed. The public API is now
consistently named `qdmi`:

| Concern | C++ | Python |
| --- | --- | --- |
| Discover registrations | `qdmi::DeviceRegistry` | `qdmi.DeviceManager.definitions` |
| Open one device | `qdmi::DeviceManager::open` | `DeviceManager.open` |
| Open every device independently | `qdmi::DeviceManager::openAll` | `DeviceManager.open_all` |
| Device capabilities | `qdmi::Device` | `qdmi.Device` |
| Sites, operations, and jobs | `qdmi::Site`, `Operation`, `Job` | `Device.Site`, `Device.Operation`, `Job` |
| Neutral-atom view | `na::qdmi::Device` | `mqt.core.na.qdmi.Device` |
| CMake target | `MQT::CoreQDMI` | not applicable |

There is intentionally no compatibility module in v4. Replace imports from the
removed device-management module with `mqt.core.qdmi`, includes with
`qdmi/Device.hpp` or `qdmi/DeviceManager.hpp`, and the removed CMake target with
`MQT::CoreQDMI`.

#### Opening devices

Code that previously created a global session and enumerated its devices should
create a manager and open stable device IDs. Discovery parses configuration but
does not load native libraries; loading and session initialization happen only
when `open` or `open_all` is called.

Python:

```python
from mqt.core import qdmi

manager = qdmi.DeviceManager()
print([definition.id for definition in manager.definitions])

device = manager.open("mqt.ddsim.default")
print(device.name())

devices, errors = manager.open_all()
for device_id, error in errors.items():
    print(f"{device_id} could not be opened: {error}")
```

C++:

```cpp
#include "qdmi/DeviceManager.hpp"

qdmi::DeviceManager manager;
const auto device = manager.open("mqt.ddsim.default");

const auto opened = manager.openAll();
for (const auto& [id, error] : opened.errors) {
  // Handle one failed provider without losing successfully opened devices.
}
```

`openAll` isolates failures by device ID. This differs from the removed global
session, where one library or initialization failure could affect discovery as a
whole.

#### Per-device session parameters

Authentication and provider settings no longer belong to a process-wide session.
Put defaults on each device definition and override them for one `open` call:

```python
from mqt.core import qdmi

parameters = qdmi.SessionParameters()
parameters.token = obtain_token()
parameters.project_id = "research-project"

device = qdmi.DeviceManager().open(
    "vendor.qpu.production",
    session_overrides=parameters,
)
```

```cpp
qdmi::SessionParameters parameters;
parameters.token = obtainToken();
parameters.projectId = "research-project";

auto device = manager.open("vendor.qpu.production", parameters);
```

Multiple definitions may refer to the same shared library while using
independent session parameters. Open devices, child devices, sites, operations,
and jobs share the required internal state, so these objects remain valid after
the manager that opened them is destroyed. Unregistering a definition likewise
does not invalidate already opened devices.

#### Registering devices with configuration

Device registration is versioned and keyed by a stable, unique `id`:

```json
{
  "schema-version": 1,
  "qdmi": {
    "devices": [
      {
        "id": "vendor.qpu.production",
        "library": "./libvendor-qdmi-device.so",
        "abi": "qdmi-v1",
        "prefix": "VENDOR",
        "enabled": true,
        "session": {
          "base-url": "https://qpu.example",
          "auth-file": "./credentials.json"
        }
      }
    ]
  }
}
```

Relative paths are resolved against the file containing them. Configuration
layers are merged field by field by `id`, and `enabled = false` masks a
lower-precedence definition. Duplicate IDs within one source, unknown keys,
invalid types, and incomplete enabled definitions are errors with source and
configuration-path diagnostics.

The precedence order, from lowest to highest, is:

1. packaged manifest fragments;
2. system `mqt-core.json`;
3. user or XDG `mqt-core.json`;
4. the nearest project `mqt-core.json` or `[tool.mqt-core.qdmi]` table in
   `pyproject.toml`;
5. `MQT_CORE_QDMI_CONFIG_JSON`;
6. C++ or Python runtime overrides.

A dedicated `mqt-core.json` wins over `pyproject.toml` in the same directory.
`MQT_CORE_QDMI_CONFIG_FILE` or `ConfigOptions.explicitFile` replaces the system,
user, and project layers while retaining packaged devices. Set `isolated` to
exclude packaged manifests as well. See the
[configuration reference](docs/qdmi/configuration.md) for complete schemas and
administrator, project, environment, and runtime examples.

Static C++ executables must set `ConfigOptions.configRoot`: unlike a shared
library or Python extension, a fully static executable has no portable module
location from which built-in manifests can be discovered.

#### QDMI child devices

`qdmi::Device::getChildDevices()` and `Device.child_devices()` return direct
child devices as ordinary device objects. Each child owns the provider session
and library state required by its QDMI v1 handle; retaining a child is therefore
safe even after discarding its parent or manager. Providers without child-device
support return an empty list.

#### Qiskit integration

`QDMIProvider` now uses `DeviceManager` internally and creates a backend for
every successfully opened, convertible device. Existing code that only creates
`QDMIProvider()` does not need to manage device objects itself. Authentication
keyword arguments are converted to per-open `SessionParameters`.

Code that directly constructs a `QDMIBackend` should pass a
`mqt.core.qdmi.Device` returned by `DeviceManager.open`. Tests and downstream
integrations that mocked global device enumeration should instead inject runtime
definitions or mock `DeviceManager.open_all`.

### MLIR enabled by default for C++ and Python package builds

The MLIR-based functionality within MQT Core has long been experimental and
opt-in. Starting with this release, MLIR is enabled by default for C++ library
builds. This means that LLVM 22.1+ (including MLIR) is now a required dependency
for building MQT Core from source.

We offer pre-built distributions for all supported platforms as part of the
`setup-mlir` project at
[munich-quantum-software/setup-mlir](https://github.com/munich-quantum-software/setup-mlir).
Please follow the instructions there to install the distribution for your
platform. You can then point CMake to the installation directory using the
`-DMLIR_DIR=/path/to/mlir/installation/lib/cmake/mlir` option.

As of this release, MLIR is also enabled for Python package builds, since the
package now exposes an MLIR-based compiler entry point in `mqt.core.mlir`.

For local development, you can configure `MLIR_DIR` once in a repository-local
`.env` file (for example, `MLIR_DIR=/path/to/installation/lib/cmake/mlir`). MQT
Core's CMake setup will pick this up automatically when `MLIR_DIR` is not
otherwise provided.

The MLIR components can still be manually disabled by passing
`-DBUILD_MQT_CORE_MLIR=OFF` to CMake.

Known limitations:

- Our pre-built distributions are incompatible with GCC on macOS. Use
  (Apple)Clang instead or compile LLVM from source using your preferred
  compiler.
- AppleClang 17+ is required to build MQT Core with MLIR enabled due to some
  C++20 features being used that are not yet properly supported by older
  versions.

### Removal of the density matrix support from the DD package

The density matrix support within the DD package has been removed. This change
was made to reduce the maintenance burden of the package. Any libraries that
depend on the density matrix functionality, such as [MQT DDSIM], need to
implement it on their own or use an alternative solution. In a related fashion,
this PR also removes the noise operations from the MQT Core IR as they no longer
serve a purpose.

### Removal of the `datastructures` (sub)library

The `datastructures` (sub)library has been removed from the MQT Core repository.
Its functionality has only ever been used in [MQT QMAP] since its inception. As
a consequence, the code shall be moved to [MQT QMAP] once QMAP adopts an MQT
Core version that includes this change.

### Dev container

A [dev container](https://containers.dev/) configuration is available to provide
a consistent local development environment. Common IDEs like
[CLion](https://www.jetbrains.com/help/clion/dev-containers-starting-page.html)
and [VS Code](https://code.visualstudio.com/docs/devcontainers/containers) can
open the repository directly inside the container. If you are on Windows, we
recommend using Docker Desktop with the WSL 2 backend.

## [3.7.0]

The shared library ABI version (`SOVERSION`) is increased from `3.6` to `3.7`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### `nanobind` updated to version 2.13.0

This release updates the `nanobind` dependency to version 2.13.0, which includes
an ABI bump. Any existing code that uses the `mqt-core` Python bindings will
need to be recompiled with the new `nanobind` version.

### QDMI updated to version 1.3.2

While not a breaking change, this release updates the QDMI dependency to version
1.3.2

### CMake presets

[CMake presets] have been added to provide a standardized and reproducible way
to configure builds across different platforms. These presets are also used in
our CI. They assume that `MLIR_DIR` is defined in your environment and pointing
to an MLIR installation.

On Unix systems, the `debug`, `release`, and `coverage` presets can be used to
configure, build, and test MQT Core.

```console
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Additionally, the `lint` preset can be used to configure and build MQT Core in
preparation for a `clang-tidy` run.

If you are on Windows, use the `debug-windows` and `release-windows` presets.

## [3.6.0]

The shared library ABI version (`SOVERSION`) is increased from `3.5` to `3.6`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### Changes to builtin QDMI devices

The builtin QDMI devices (with prefixes `MQT_SC`, `MQT_NA`, and `MQT_DDSIM`) are
now all built as shared libraries by default. In turn, the shared library
wrappers (with prefixes `MQT_SC_DYN` and `MQT_NA_DYN`) have been removed
entirely. MQT Core's QDMI driver will automatically load the shared libraries of
the builtin devices if they are available in the library search path. If you
were previously using the statically builtin devices, no changes should be
necessary as the shared libraries are now the default. If you were previously
using the shared library wrappers, you should switch to using the builtin
devices instead, which are now shared libraries by default.

### Broader operation support in QDMI Qiskit converter

The QDMI Qiskit converter now supports a broader range of operations, including
multi-controlled gates such as `mcx`, `mcz`, `mcrx`, and more. As a consequence,
these operations can now be directly used without requiring decomposition, for
example, with the builtin `DDSIM` QDMI device.

### Minimum supported Qiskit version

From this release onwards, MQT Core requires Qiskit version 1.1.0 or higher.
This is due to the fact that we are relying on some fixes to Qiskit primitives
that were introduced in that version. If you are using MQT Core with Qiskit,
please ensure that you have updated to Qiskit 1.1.0 or higher to avoid any
compatibility issues.

## [3.5.1]

No breaking changes.

### Component-based CMake installs

Fixed exported `nlohmann_json` CMake metadata so `find_package(mqt-core CONFIG)`
no longer propagates an invalid `.../COMPONENT` include directory in
component-based installations. Anyone relying on an installed version of
`mqt-core` should update from 3.5.0 to 3.5.1.

## [3.5.0]

The shared library ABI version (`SOVERSION`) is increased from `3.4` to `3.5`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### `nanobind` updated to version 2.12.0

This release updates the `nanobind` dependency to version 2.12.0, which includes
an ABI bump. Any existing code that uses the `mqt-core` Python bindings will
need to be recompiled with the new `nanobind` version.

## [3.4.0]

The shared library ABI version (`SOVERSION`) is increased from `3.3` to `3.4`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### Python wheels

This release contains two changes to the distributed wheels.

First, we have removed all wheels for Python 3.13t. Free-threading Python was
introduced as an experimental feature in Python 3.13. It became stable in Python
3.14.

Second, for Python 3.12+, we are now providing Stable ABI wheels instead of
separate version-specific wheels. This was enabled by migrating our Python
bindings from `pybind11` to `nanobind`.

Both of these changes were made in the interest of conserving PyPI space and
reducing CI/CD build times. The full list of wheels now reads:

- 3.10
- 3.11
- 3.12+ Stable ABI
- 3.14t

### QDMI-Qiskit integration

This release introduces a Qiskit `BackendV2`-compatible interface to QDMI
devices. The `mqt.core.plugins.qiskit` module has been extended with
`QDMIProvider`, `QDMIBackend`, and `QDMIJob` classes that allow running Qiskit
circuits on QDMI-compliant devices.

Users can now execute Qiskit circuits directly on QDMI devices:

```python
from mqt.core.plugins.qiskit import QDMIProvider

provider = QDMIProvider()
backend = provider.get_backend("MQT Core DDSIM QDMI Device")
job = backend.run(circuit, shots=1024)
result = job.result()
```

The backend automatically converts circuits to QASM, introspects device
capabilities, validates circuits, and formats results. The existing QDMI
interface (`mqt.core.qdmi`) remains fully supported for direct, low-level access
to QDMI devices.

Install with Qiskit support: `uv pip install "mqt-core[qiskit]"`

See the
[Qiskit Backend documentation](https://mqt.readthedocs.io/projects/core/en/latest/qdmi/qiskit_backend.html)
for details.

### Argument name changes in `QuantumComputation` and `CompoundOperation` dunder methods

Since we enabled `ty` for type checking, it revealed that some of the dunder
methods of `QuantumComputation` and `CompoundOperation` had incorrect argument
names, which would prevent these classes from properly implementing the
`MutableSequence` protocol. This release fixes these issues by renaming the
arguments of the following methods:

- `QuantumComputation.__getitem__`
- `QuantumComputation.__setitem__`
- `QuantumComputation.__delitem__`
- `QuantumComputation.insert`
- `QuantumComputation.append`
- `CompoundOperation.__getitem__`
- `CompoundOperation.__setitem__`
- `CompoundOperation.__delitem__`
- `CompoundOperation.insert`
- `CompoundOperation.append`

All index arguments are now named `index` instead of `idx` (or `i` or `slice`)
and all values are now named `value` instead of `val` (or `op` or `ops`).

### DD Package evaluation

This release moves the DD Package evaluation functionality from within the
`mqt.core` package to a dedicated script in the `eval` directory. In the
process, the `mqt-core-dd-compare` entry point as well as the `evaluation` extra
have been removed. The `eval/dd_evaluation.py` script acts as a drop-in
replacement for the previous CLI entry point. Since the `eval` directory is not
part of the Python package, this functionality is only available via source
installations or by cloning the repository.

## [3.3.0]

The shared library ABI version (`SOVERSION`) is increased from `3.2` to `3.3`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

### IfElseOperation

This release introduces an `IfElseOperation` to the C++ library and the Python
package to support Qiskit's `IfElseOp`. The new operation replaces the
`ClassicControlledOperation`.

An `IfElseOperation` can be added to a `QuantumComputation` using `if_else()`.

```python
qc.if_else(
    then_operation=StandardOperation(target=0, op_type=OpType.x),
    else_operation=StandardOperation(target=0, op_type=OpType.y),
    control_bit=0,
)
```

If no else operation is needed, the `if_()` method can be used.

```python
qc.if_(op_type=OpType.x, target=0, control_bit=0)
```

### End of support for Python 3.9

Starting with this release, MQT Core no longer supports Python 3.9. This is in
line with the scheduled end of life of the version. As a result, MQT Core is no
longer tested under Python 3.9 and no longer ships Python 3.9 wheels.

## [3.2.0]

The shared library ABI version (`SOVERSION`) is increased from `3.1` to `3.2`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

With this release, the minimum required C++ version has been raised from C++17
to C++20. The default compilers of our test systems support all relevant
features of the standard. Some frameworks we plan to integrate with even require
C++20 by now.

The `dd.BasisStates`, `ir.operations.ComparisonKind`,
`ir.operations.Control.Type`, and `ir.operations.OpType` enums are now exposed
via `pybind11`'s new `py::native_enum`, which makes them compatible with
Python's `enum.Enum` class (PEP 435). As a result, the enums can no longer be
initialized using a string. Instead of `OpType("x")`, use `OpType.x`.

## [3.1.0]

The shared library ABI version (`SOVERSION`) is increased from `3.0` to `3.1`.
Thus, consuming libraries need to update their wheel repair configuration for
`cibuildwheel` to ensure the `mqt-core` libraries are properly skipped in the
wheel repair step.

Even though this is not a breaking change, it is worth mentioning to developers
of MQT Core that all Python code (except tests) has been moved to the top-level
`python` directory. Furthermore, the C++ code for the Python bindings has been
moved to the top-level `bindings` directory.

### DD Package

The `makeZeroState`, `makeBasisState`, `makeGHZState`, `makeWState`, and
`makeStateFromVector` methods have been refactored to functions taking the DD
package as an argument. These functions reside in the `StateGeneration` header.
Any existing code that uses these methods must replace the respective calls with
their function counterpart.

## [3.0.0]

This major release introduces several breaking changes, including the removal of
deprecated features and the introduction of new APIs. In preparation for this
release, most direct dependents of MQT Core have been updated to use the new
APIs. The following sections describe the most important changes and how to
adapt your code accordingly. We intend to provide a more comprehensive migration
guide for future releases.

### Intermediate Representation (IR)

The OpenQASM parser has been encapsulated in its own library, which is now a
dedicated target in the CMake build system. Any use of
`qc::QuantumComputation::import...` needs to be replaced with the respective
`qasm3::Importer::load...` function.

Several parsers have been removed, including the `.real`, `.qc`, `.tfc`, and
`GRCS` parsers. The `.real` parser lives on as part of the [MQT SyReC] project.
All others have been removed without replacement.

The `Teleportation` gate has been removed from the IR. This was a placeholder
gate and was only used in a single method (in [MQT QMAP]), which is bound to be
removed as part of [MQT QMAP] `v3.0.0`.

[MQT QCEC], [MQT QMAP], and [MQT DDSIM] have been updated to use the new API,
which will be released in [MQT QCEC] `v3.0.0`, [MQT QMAP] `v3.0.0` and
[MQT DDSIM] `v2.0.0`.

### DD Package

The DD package has undergone some initial refactoring to streamline the
implementation and prepare it for future extensions. The `Config` template has
been removed in favor of a constructor that takes the configuration as a
parameter. Any existing code using `dd::Package<...>` needs to be updated to use
`dd::Package` or `dd::Package(numQubits, ...)` instead. The `MemoryManager` and
adjacent classes have been refactored to remove the template parameters. This
should not have user-visible effects, but it is a breaking change nonetheless.
Depending libraries may now also use the `mqt-core` Python package to interact
with the DD package.

[MQT QCEC] and [MQT DDSIM] have been updated to use the new API, which will be
released in [MQT QCEC] `v3.0.0` and [MQT DDSIM] `v2.0.0`.

### Neutral Atom Quantum Computing

The `NAComputation` class hierarchy has been refactored to use an MLIR-inspired
design. This will act as a foundation for future extensions and improvements.

[MQT QMAP] has been updated to use the new API, which will be released in
[MQT QMAP] `v3.0.0`.

### General

MQT Core has moved to the
[munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub
organization under <https://github.com/munich-quantum-toolkit/core>. While most
links should be automatically redirected, please update any links in your code
to point to the new location. All links in the documentation have been updated
accordingly.

MQT Core now ships all its C++ libraries as shared libraries with the `mqt-core`
Python package. Depending packages can now solely rely on the Python package for
obtaining the C++ libraries. This is demonstrated in [MQT QCEC] `v3.0.0`,
[MQT QMAP] `v3.0.0` and [MQT DDSIM] `v2.0.0`, which will be released in the near
future.

MQT Core now requires CMake 3.24 or higher. Most modern operating systems should
have this version available in their package manager. Alternatively, CMake can
be conveniently installed from PyPI using the
[`cmake`](https://pypi.org/project/cmake/) package.

It also requires the `uv` library version 0.5.20 or higher.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v3.7.0...HEAD
[3.7.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.6.0...v3.7.0
[3.6.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.5.1...v3.6.0
[3.5.1]: https://github.com/munich-quantum-toolkit/core/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/munich-quantum-toolkit/core/compare/v2.7.0...v3.0.0

<!-- Other links -->

[MQT DDSIM]: https://github.com/cda-tum/mqt-ddsim
[MQT QMAP]: https://github.com/cda-tum/mqt-qmap
[MQT QCEC]: https://github.com/cda-tum/mqt-qcec
[MQT SyReC]: https://github.com/cda-tum/mqt-syrec
[CMake presets]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
