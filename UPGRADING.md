# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

With this release, the minimum required C++ version has been raised from C++17 to C++20.
The default compilers of our test systems support all relevant features of the standard.
Some frameworks we plan to integrate with even require C++20 by now.

## [3.1.0]

The shared library ABI version (`SOVERSION`) is increased from `3.0` to `3.1`.
Thus, consuming libraries need to update their wheel repair configuration for cibuildwheel to ensure the mqt-core libraries are properly skipped in the wheel repair step.

Even tough this is not a breaking change, it is worth mentioning to developers of MQT Core that all Python code (except tests) has been moved to the top-level `python` directory.
Furthermore, the C++ code for the Python bindings has been moved to the top-level `bindings` directory.

### DD Package

The `makeZeroState`, `makeBasisState`, `makeGHZState`, `makeWState`, and `makeStateFromVector` methods have been refactored to functions taking the DD package as an argument. These functions reside in the `StateGeneration` header. Any existing code that uses these methods must replace the respective calls with their function counterpart.

## [3.0.0]

This major release introduces several breaking changes, including the removal of deprecated features and the introduction of new APIs.
In preparation for this release, most direct dependents of MQT Core have been updated to use the new APIs.
The following sections describe the most important changes and how to adapt your code accordingly.
We intend to provide a more comprehensive migration guide for future releases.

### Intermediate Representation (IR)

The OpenQASM parser has been encapsulated in its own library, which is now a dedicated target in the CMake build system.
Any use of `qc::QuantumComputation::import...` needs to be replaced with the respective `qasm3::Importer::load...` function.

Several parsers have been removed, including the `.real`, `.qc`, `.tfc`, and `GRCS` parsers.
The `.real` parser lives on as part of the [MQT SyReC] project. All others have been removed without replacement.

The `Teleportation` gate has been removed from the IR. This was a placeholder gate and was only used in a single method (in [MQT QMAP]), which is bound to be removed as part of [MQT QMAP] `v3.0.0`.

[MQT QCEC], [MQT QMAP], and [MQT DDSIM] have been updated to use the new API, which will be released in [MQT QCEC] `v3.0.0`, [MQT QMAP] `v3.0.0` and [MQT DDSIM] `v2.0.0`.

### DD Package

The DD package has undergone some initial refactoring to streamline the implementation and prepare it for future extensions.
The `Config` template has been removed in favor of a constructor that takes the configuration as a parameter.
Any existing code using `dd::Package<...>` needs to be updated to use `dd::Package` or `dd::Package(numQubits, ...)` instead.
The `MemoryManager` and adjacent classes have been refactored to remove the template parameters.
This should not have user-visible effects, but it is a breaking change nonetheless.
Depending libraries may now also use the `mqt-core` Python package to interact with the DD package.

[MQT QCEC] and [MQT DDSIM] have been updated to use the new API, which will be released in [MQT QCEC] `v3.0.0` and [MQT DDSIM] `v2.0.0`.

### Neutral Atom Quantum Computing

The `NAComputation` class hierarchy has been refactored to use an MLIR-inspired design. This will act as a foundation for future extensions and improvements.

[MQT QMAP] has been updated to use the new API, which will be released in [MQT QMAP] `v3.0.0`.

### General

MQT Core has moved to the [munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub organization under https://github.com/munich-quantum-toolkit/core.
While most links should be automatically redirected, please update any links in your code to point to the new location.
All links in the documentation have been updated accordingly.

MQT Core now ships all its C++ libraries as shared libraries with the `mqt-core` Python package.
Depending packages can now solely rely on the Python package for obtaining the C++ libraries.
This is demonstrated in [MQT QCEC] `v3.0.0`, [MQT QMAP] `v3.0.0` and [MQT DDSIM] `v2.0.0`, which will be released in the near future.

MQT Core now requires CMake 3.24 or higher.
Most modern operating systems should have this version available in their package manager.
Alternatively, CMake can be conveniently installed from PyPI using the [`cmake`](https://pypi.org/project/cmake/) package.

It also requires the `uv` library version 0.5.20 or higher.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v3.1.0...HEAD
[3.1.0]: https://github.com/munich-quantum-toolkit/core/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/munich-quantum-toolkit/core/compare/v2.7.0...v3.0.0

<!-- Other links -->

[MQT DDSIM]: https://github.com/cda-tum/mqt-ddsim
[MQT QMAP]: https://github.com/cda-tum/mqt-qmap
[MQT QCEC]: https://github.com/cda-tum/mqt-qcec
[MQT SyReC]: https://github.com/cda-tum/mqt-syrec
