<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Changed

- 🚸 Improve native gate support for the Qiskit-to-OpenQASM3 conversion in the QDMI-Qiskit interface ([#1719]) ([**@burgholzer**])

## [3.6.0] - 2026-05-13

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#360)._

### Added

- 🚸 Add a measurement instruction to the default SC QDMI device ([#1694]) ([**@burgholzer**])
- ✨ Add support for multi-controlled gates to the QDMI Qiskit backend converter ([#1694]) ([**@burgholzer**])

### Changed

- ♻️ Build all built-in QDMI devices as shared libraries ([#1694]) ([**@burgholzer**])
- ⬆️ Update minimum supported Qiskit version to 1.1.0 ([#1694]) ([**@burgholzer**])

### Fixed

- 🐛 Fix missing `nlohmann_json.natvis` in Windows component-based CMake installs ([#1702]) ([**@burgholzer**])
- 🐛 Fix segfault in DD `sample` method when idle classical bits are present ([#1694]) ([**@burgholzer**])

### Removed

- 🔥 Remove shared library wrappers for QDMI devices ([#1694]) ([**@burgholzer**])

## [3.5.1] - 2026-04-23

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#351)._

### Fixed

- 🐛 Fix malformed include directories in exported `nlohmann_json` CMake targets for component-based installs ([#1662]) ([**@burgholzer**])

## [3.5.0] - 2026-04-21

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#350)._

### Added

- ✨ Add support for multi-controlled gates to ZX package ([#1380]) ([**@keefehuang**], [**@denialhaag**])
- ✨ Add Sampler and Estimator primitives to the QDMI-Qiskit interface ([#1507]) ([**@marcelwa**])

### Changed

- ⬆️ Update `nanobind` to version 2.12.0 ([#1528])
- ⬆️ Update QDMI to `v1.3.0` ([#1652]) ([**@burgholzer**])
- 📦 Switch to component-based installation for the MQT Core Python package ([#1596]) ([**@burgholzer**])
- ⬆️ Update QDMI to latest version from stable `v1.2.x` branch ([#1593]) ([**@burgholzer**])
- ⬆️ Update `clang-tidy` to version 22 ([#1564]) ([**@denialhaag**], [**@burgholzer**])
- 👷 Build on `macos-26`/`macos-26-intel` by default and `macos-15`/`macos-15-intel` for extensive tests ([#1571]) ([**@denialhaag**])

## [3.4.1] - 2026-02-01

### Changed

- ⬆️ Update `nanobind` to version 2.11.0 ([#1481]) ([**@denialhaag**])
- ⬆️ Update Boost to version 1.89.0 ([#1453]) ([**@burgholzer**])
- ⬆️ Update QDMI to latest version from stable `v1.2.x` branch ([#1453]) ([**@burgholzer**])
- ⬆️ Update `spdlog` to version 1.17.0 ([#1453]) ([**@burgholzer**])
- ♻️ Use `llc` instead of random `clang` for compiling QIR test circuits to improve robustness and handle opaque pointers correctly across LLVM versions ([#1447]) ([**@burgholzer**])
- ♻️ Extract singleton pattern into reusable template base class for QDMI devices and driver ([#1444]) ([**@ystade**], [**@burgholzer**])
- 🚚 Reorganize QDMI code structure by moving devices into dedicated subdirectories and separating driver and common utilities ([#1444]) ([**@ystade**])

### Removed

- 🔥 No longer actively type check Python code with `mypy` and solely rely on `ty` ([#1437]) ([**@burgholzer**])

## [3.4.0] - 2026-01-08

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#340)._

### Added

- ✨ Return device handle from `add_dynamic_device_library` for direct backend creation ([#1381]) ([**@marcelwa**])
- ✨ Add IQM JSON support for job submission in Qiskit-QDMI Backend ([#1375], [#1382]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Add authentication support for QDMI sessions with token, username/password, auth file, auth URL, and project ID parameters ([#1355]) ([**@marcelwa**])
- ✨ Add a new QDMI device that represents a superconducting architecture featuring a coupling map ([#1328]) ([**@ystade**])
- ✨ Add bi-directional iterator that traverses the def-use chain of a qubit value ([#1310]) ([**@MatthiasReumann**])
- ✨ Add `OptionalDependencyTester` to lazily handle optional Python dependencies like Qiskit ([#1243]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Expose the QDMI job interface through FoMaC ([#1243]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Add Qiskit backend wrapper with job submission support for QDMI devices through a provider interface ([#1243], [#1385]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Support `QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS` in the NA QDMI Device and the DDSIM QDMI Device ([#1243]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Support `QDMI_DEVICE_JOB_PROPERTY_PROGRAM` in the NA QDMI Device ([#1243]) ([**@marcelwa**], [**@burgholzer**])

### Changed

- 📦🏁 Build Windows x86 wheels on `windows-2025` runner for newer compiler ([#1415]) ([**@burgholzer**])
- 👷 Build on `macos-15`/`windows-2025` by default and `macos-14`/`windows-2022` for extensive tests ([#1414]) ([**@burgholzer**])
- 📦🍎 Build macOS arm64 wheels on macos-15 runner for newer compiler ([#1413]) ([**@burgholzer**])
- ⚡ Improve uv build caching by removing unconditional `reinstall-package` and configuring dedicated `cache-keys` ([#1412]) ([**@burgholzer**])
- 👨‍💻📦 Build `spdlog` and QDMI generators as shared libraries in Python package builds ([#1411], [#1403]) ([**@burgholzer**])
- ♻️🏁 Remove Windows-specific restrictions for dynamic QDMI device library handling ([#1406]) ([**@burgholzer**])
- ♻️ Migrate Python bindings from `pybind11` to `nanobind` ([#1383]) ([**@denialhaag**], [**@burgholzer**])
- 📦️ Provide Stable ABI wheels for Python 3.12+ ([#1383]) ([**@burgholzer**], [**@denialhaag**])
- 🚚 Create dedicated `mqt.core.na` submodule to closely follow the structure of other submodules ([#1383]) ([**@burgholzer**])
- ✨ Add common definitions and utilities for QDMI ([#1355]) ([**@burgholzer**])
- 🚚 Move `NA` QDMI device in its right place next to other QDMI devices ([#1355]) ([**@burgholzer**])
- ♻️ Allow repeated loading of QDMI device library with potentially different session configurations ([#1355]) ([**@burgholzer**])
- ♻️ Enable thread-safe reference counting for QDMI devices singletons ([#1355]) ([**@burgholzer**])
- ♻️ Refactor `FoMaC` singleton to instantiable `Session` class with configurable authentication parameters ([#1355]) ([**@marcelwa**])
- 👷 Stop testing on `ubuntu-22.04` and `ubuntu-22.04-arm` runners ([#1359]) ([**@denialhaag**], [**@burgholzer**])
- 👷 Stop testing with `clang-19` and start testing with `clang-21` ([#1359]) ([**@denialhaag**], [**@burgholzer**])
- 👷 Fix macOS tests with Homebrew Clang via new `munich-quantum-toolkit/workflows` version ([#1359]) ([**@denialhaag**], [**@burgholzer**])
- 👷 Re-enable macOS tests with GCC by disabling module scanning ([#1359]) ([**@denialhaag**], [**@burgholzer**])
- ♻️ Group circuit operations into scheduling units for MLIR routing ([#1301]) ([**@MatthiasReumann**])
- 👷 Use `munich-quantum-software/setup-mlir` to set up MLIR ([#1294]) ([**@denialhaag**])
- ♻️ Preserve tuple structure and improve site type clarity of the MQT NA Default QDMI Device ([#1299]) ([**@marcelwa**])
- ♻️ Move DD package evaluation module to standalone script ([#1327]) ([**@burgholzer**])
- ⬆️ Bump QDMI version to 1.2.0 ([#1243]) ([**@marcelwa**], [**@burgholzer**])

### Fixed

- 🔧 Install all available QDMI device targets in Python package builds ([#1403]) ([**@burgholzer**])
- 🐛 Fix operation validation in Qiskit backend to handle device-specific gate naming conventions ([#1384]) ([**@marcelwa**])
- 🐛 Fix conditional branch handling when importing MLIR from `QuantumComputation` ([#1378]) ([**@lirem101**])
- 🐛 Fix custom QDMI property and parameter handling in SC and NA devices ([#1355]) ([**@burgholzer**])
- 🚨 Fix argument naming of `QuantumComputation` and `CompoundOperation` dunder methods for properly implementing the `MutableSequence` protocol ([#1338]) ([**@burgholzer**])
- 🐛 Fix memory management in dynamic QDMI device by making it explicit ([#1336]) ([**@ystade**])

### Removed

- 🔥 Remove wheel builds for Python 3.13t ([#1371]) ([**@burgholzer**])
- 🔥 Remove the `evaluation` extra from the MQT Core Python package ([#1327]) ([**@burgholzer**])
- 🔥 Remove the `mqt-core-dd-compare` entry point from the MQT Core Python package ([#1327]) ([**@burgholzer**])

## [3.3.3] - 2025-11-10

### Added

- ✨ Add support for bridge gates for the neutral atom hybrid mapper ([#1293]) ([**@lsschmid**])

### Fixed

- 🐛 Revert change to `opTypeFromString()` signature made in [#1283] ([#1300]) ([**@denialhaag**])

## [3.3.2] - 2025-11-04

### Added

- ✨ Add DD-based simulator QDMI device ([#1287]) ([**@burgholzer**])
- ✨ A `--reuse-qubits` pass implementing an advanced form of qubit reuse to reduce the qubit count of quantum circuits ([#1108]) ([**@DRovara**])
- ✨ A `--lift-measurements` pass that attempts to move measurements up as much as possible, used for instance to enable better qubit reuse ([#1108]) ([**@DRovara**])
- ✨ Add native support for `R(theta, phi)` gate ([#1283]) ([**@burgholzer**])
- ✨ Add A\*-search-based routing algorithm to MLIR transpilation routines ([#1237], [#1271], [#1279]) ([**@MatthiasReumann**])

### Fixed

- 🐛 Fix edge-case in validation of `NAComputation` ([#1276]) ([**@ystade**])
- 🐛 Allow integer QASM version declarations ([#1269]) ([**@denialhaag**])

## [3.3.1] - 2025-10-14

### Fixed

- 🐛 Ensure `spdlog` dependency can be found from `mqt-core` install ([#1263]) ([**@burgholzer**])

## [3.3.0] - 2025-10-13

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#330)._

### Added

- 👷 Enable testing on Python 3.14 ([#1246]) ([**@denialhaag**])
- ✨ Add dedicated `PlacementPass` to MLIR transpilation routines ([#1232]) ([**@MatthiasReumann**])
- ✨ Add an NA-specific FoMaC implementation ([#1223], [#1236]) ([**@ystade**], [**@burgholzer**])
- ✨ Enable import of BarrierOp into MQTRef ([#1224]) ([**@denialhaag**])
- ✨ Add naive quantum program routing MLIR pass ([#1148]) ([**@MatthiasReumann**])
- ✨ Add QIR runtime using DD-based simulation ([#1210]) ([**@ystade**], [**@burgholzer**])
- ✨ Add SWAP reconstruction patterns to the newly-named `SwapReconstructionAndElision` MLIR pass ([#1207]) ([**@taminob**], [**@burgholzer**])
- ✨ Add two-way conversions between MQTRef and QIR ([#1091]) ([**@li-mingbao**])
- 🚸 Define custom assembly formats for MLIR operations ([#1209]) ([**@denialhaag**])
- ✨ Add support for translating `IfElseOperation`s to the `MQTRef` MLIR dialect ([#1164]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Add MQT's implementation of a generic FoMaC with Python bindings ([#1150], [#1186], [#1223]) ([**@ystade**])
- ✨ Add new MLIR pass `ElidePermutations` for SWAP gate elimination ([#1151]) ([**@taminob**])
- ✨ Add new pattern to MLIR pass `GateElimination` for identity gate removal ([#1140]) ([**@taminob**])
- ✨ Add Clifford block collection pass to `CircuitOptimizer` module ([#885]) ([**jannikpflieger**], [**@burgholzer**])
- ✨ Add `isControlled()` method to the `UnitaryInterface` MLIR class ([#1157]) ([**@taminob**], [**@burgholzer**])
- 📝 Integrate generated MLIR documentation ([#1147]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Add `IfElseOperation` to C++ library and Python package to support Qiskit's `IfElseOp` ([#1117]) ([**@denialhaag**], [**@burgholzer**], [**@lavanya-m-k**])
- ✨ Add `allocQubit` and `deallocQubit` operations for dynamically working with single qubits to the MLIR dialects ([#1139]) ([**@DRovara**], [**@burgholzer**])
- ✨ Add `qubit` operation for static qubit addressing to the MLIR dialects ([#1098], [#1116]) ([**@MatthiasReumann**])
- ✨ Add MQT's implementation of a QDMI Driver ([#1010]) ([**@ystade**])
- ✨ Add MQT's implementation of a QDMI Device for neutral atom-based quantum computing ([#996], [#1010], [#1100]) ([**@ystade**], [**@burgholzer**])
- ✨ Add translation from `QuantumComputation` to the `MQTRef` MLIR dialect ([#1099]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Add `reset` operations to the MLIR dialects ([#1106]) ([**@DRovara**])

### Changed

- ♻️ Replace custom `AllocOp`, `DeallocOp`, `ExtractOp`, and `InsertOp` with MLIR-native `memref` operations ([#1211]) ([**@denialhaag**])
- 🚚 Rename MLIR pass `ElidePermutations` to `SwapReconstructionAndElision` ([#1207]) ([**@taminob**])
- ⬆️ Require LLVM 21 for building the MLIR library ([#1180]) ([**@denialhaag**])
- ⬆️ Update to version 21 of `clang-tidy` ([#1180]) ([**@denialhaag**])
- 🚚 Rename MLIR pass `CancelConsecutiveInverses` to `GateElimination` ([#1140]) ([**@taminob**])
- 🚚 Rename `xxminusyy` to `xx_minus_yy` and `xxplusyy` to `xx_plus_yy` in MLIR dialects ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])
- 🚸 Add custom assembly format for operations in the MLIR dialects ([#1139]) ([**@burgholzer**])
- 🚸 Enable `InferTypeOpInterface` in the MLIR dialects to reduce explicit type information ([#1139]) ([**@burgholzer**])
- 🚚 Rename `check-quantum-opt` test target to `mqt-core-mlir-lit-test` ([#1139]) ([**@burgholzer**])
- ♻️ Update the `measure` operations in the MLIR dialects to no longer support more than one qubit being measured at once ([#1106]) ([**@DRovara**])
- 🚚 Rename `XXminusYY` to `XXminusYYOp` and `XXplusYY` to `XXplusYYOp` in MLIR dialects ([#1099]) ([**@denialhaag**])
- 🚚 Rename `MQTDyn` MLIR dialect to `MQTRef` ([#1098]) ([**@MatthiasReumann**])

### Removed

- 🔥 Drop support for Python 3.9 ([#1181]) ([**@denialhaag**])
- 🔥 Remove `ClassicControlledOperation` from C++ library and Python package ([#1117]) ([**@denialhaag**])

### Fixed

- 🐛 Fix CMake installation to make `find_package(mqt-core CONFIG)` succeed ([#1247]) ([**@burgholzer**], [**@denialhaag**])
- 🏁 Fix stack overflows in OpenQASM layout parsing on Windows for large circuits ([#1235]) ([**@burgholzer**])
- ✨ Add missing `StandardOperation` conversions in MLIR roundtrip pass ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])

## [3.2.1] - 2025-08-01

### Fixed

- 🐛 Fix usage of `std::accumulate` by changing accumulator parameter from reference to value ([#1089]) ([**@denialhaag**])
- 🐛 Fix erroneous `contains` check in DD package ([#1088]) ([**@denialhaag**])

## [3.2.0] - 2025-07-31

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#320)._

### Added

- 🐍 Build Python 3.14 wheels ([#1076]) ([**@denialhaag**])
- ✨ Add MQT-internal MLIR dialect conversions ([#1001]) ([**@li-mingbao**])

### Changed

- ✨ Expose enums to Python via `pybind11`'s new (`enum.Enum`-compatible) `py::native_enum` ([#1075]) ([**@denialhaag**])
- ⬆️ Require C++20 ([#897]) ([**@burgholzer**], [**@denialhaag**])

## [3.1.0] - 2025-07-11

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#310)._

### Added

- ✨ Add MLIR pass for merging rotation gates ([#1019]) ([**@denialhaag**])
- ✨ Add functions to generate random vector DDs ([#975]) ([**@MatthiasReumann**])
- ✨ Add function to approximate decision diagrams ([#908]) ([**@MatthiasReumann**])
- 📦 Add Windows ARM64 wheels ([#926]) ([**@burgholzer**])
- 📝 Add documentation page for MLIR ([#931]) ([**@ystade**])
- ✨ Initial implementation of the mqtdyn Dialect ([#900]) ([**@DRovara**], [**@ystade**])

### Fixed

- 🐛 Fix bug in MLIR roundtrip passes caused by accessing an invalidated iterator after erasure in a loop ([#932]) ([**@flowerthrower**])
- 🐛 Add missing support for `sxdg` gates in Qiskit circuit import ([#930]) ([**@burgholzer**])
- 🐛 Fix bug related to initialization of operations with duplicate operands ([#964]) ([**@ystade**])
- 🐛 Open issue for Qiskit upstream test only when the test is actually failing not when it was cancelled ([#973]) ([**@ystade**])
- 🐛 Fix parsing of `GPhase` in the `MQTOpt` MLIR dialect ([#1042]) ([**@ystade**], [**@DRovara**])

### Changed

- ⬆️ Bump shared library ABI version from `3.0` to `3.1` ([#1047]) ([**@denialhaag**])
- ♻️ Switch from reference counting to mark-and-sweep garbage collection in decision diagram package ([#1020]) ([**@MatthiasReumann**], [**burgholzer**], [**q-inho**])
- ♻️ Move the C++ code for the Python bindings to the top-level `bindings` directory ([#982]) ([**@denialhaag**])
- ♻️ Move all Python code (no tests) to the top-level `python` directory ([#982]) ([**@denialhaag**])
- ⚡ Improve performance of getNqubits for StandardOperations ([#959]) ([**@ystade**])
- ♻️ Move Make-State Functionality To StateGeneration ([#984]) ([**@MatthiasReumann**])
- ♻️ Outsource definition of standard operations from MLIR dialects to reduce redundancy ([#933]) ([**@ystade**])
- ♻️ Unify operands and results in MLIR dialects ([#931]) ([**@ystade**])
- ⏪️ Restore support for (MLIR and) LLVM v19 ([#934]) ([**@flowerthrower**], [**@ystade**])
- ⬆️ Update nlohmann_json to `v3.12.0` ([#921]) ([**@burgholzer**])

## [3.0.2] - 2025-04-07

### Added

- 📝 Add JOSS journal reference and citation information ([#913]) ([**@burgholzer**])
- 📝 Add new links to Python package metadata ([#911]) ([**@burgholzer**])

### Fixed

- 📝 Fix old links in Python package metadata ([#911]) ([**@burgholzer**])

## [3.0.1] - 2025-04-07

### Fixed

- 🐛 Fix doxygen build on RtD to include C++ API docs ([#912]) ([**@burgholzer**])

## [3.0.0] - 2025-04-06

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#300)._

### Added

- ✨ Ship shared C++ libraries with `mqt-core` Python package ([#662]) ([**@burgholzer**])
- ✨ Add Python bindings for the DD package ([#838]) ([**@burgholzer**])
- ✨ Add direct MQT `QuantumComputation` to Qiskit `QuantumCircuit` export ([#859]) ([**@burgholzer**])
- ✨ Support for Qiskit 2.0+ ([#860]) ([**@burgholzer**])
- ✨ Add initial infrastructure for MLIR within the MQT ([#878], [#879], [#892], [#893], [#895]) ([**@burgholzer**], [**@ystade**], [**@DRovara**], [**@flowerthrower**], [**@BertiFlorea**])
- ✨ Add State Preparation Algorithm ([#543]) ([**@M-J-Hochreiter**])
- 🚸 Add support for indexed identifiers to OpenQASM 3 parser ([#832]) ([**@burgholzer**])
- 🚸 Allow indexed registers as operation arguments ([#839]) ([**@burgholzer**])
- 📝 Add documentation for the DD package ([#831]) ([**@burgholzer**])
- 📝 Add documentation for the ZX package ([#817]) ([**@pehamTom**])
- 📝 Add C++ API docs setup ([#817]) ([**@pehamTom**], [**@burgholzer**])

### Changed

- **Breaking**: 🚚 MQT Core has moved to the [munich-quantum-toolkit] GitHub organization
- **Breaking**: ✨ Adopt [PEP 735] dependency groups ([#762]) ([**@burgholzer**])
- **Breaking**: ♻️ Encapsulate the OpenQASM parser in its own library ([#822]) ([**@burgholzer**])
- **Breaking**: ♻️ Replace `Config` template from DD package with constructor argument ([#886]) ([**@burgholzer**])
- **Breaking**: ♻️ Remove template parameters from `MemoryManager` and adjacent classes ([#866]) ([**@rotmanjanez**])
- **Breaking**: ♻️ Refactor algorithms to use factory functions instead of inheritance ([**@a9b7e70**]) ([**@burgholzer**])
- **Breaking**: ♻️ Change pointer parameters to references in DD package ([#798]) ([**@burgholzer**])
- **Breaking**: ♻️ Change registers from typedef to actual type ([#807]) ([**@burgholzer**])
- **Breaking**: ♻️ Refactor `NAComputation` class hierarchy ([#846], [#877]) ([**@ystade**])
- **Breaking**: ⬆️ Bump minimum required CMake version to `3.24.0` ([#879]) ([**@burgholzer**])
- **Breaking**: ⬆️ Bump minimum required `uv` version to `0.5.20` ([#802]) ([**@burgholzer**])
- 📝 Rework existing project documentation ([#789], [#842]) ([**@burgholzer**])
- 📄 Use [PEP 639] license expressions ([#847]) ([**@burgholzer**])

### Removed

- **Breaking**: 🔥 Remove the `Teleportation` gate from the IR ([#882]) ([**@burgholzer**])
- **Breaking**: 🔥 Remove parsers for `.real`, `.qc`, `.tfc`, and `GRCS` files ([#822]) ([**@burgholzer**])
- **Breaking**: 🔥 Remove tensor dump functionality ([#798]) ([**@burgholzer**])
- **Breaking**: 🔥 Remove `extract_probability_vector` functionality ([#883]) ([**@burgholzer**])

### Fixed

- 🐛 Fix Qiskit layout import and handling ([#849], [#858]) ([**@burgholzer**])
- 🐛 Properly handle timing literals in QASM parser ([#724]) ([**@burgholzer**])
- 🐛 Fix stripping of idle qubits ([#763]) ([**@burgholzer**])
- 🐛 Fix permutation handling in OpenQASM dump ([#810]) ([**@burgholzer**])
- 🐛 Fix out-of-bounds error in ZX `EdgeIterator` ([#758]) ([**@burgholzer**])
- 🐛 Fix endianness in DCX and XX_minus_YY gate matrix definition ([#741]) ([**@burgholzer**])
- 🐛 Fix needless dummy register in empty circuit construction ([#758]) ([**@burgholzer**])

## [2.7.0] - 2024-10-08

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/core/releases) for previous changelogs._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v3.6.0...HEAD
[3.6.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.6.0
[3.5.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.5.1
[3.5.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.5.0
[3.4.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.4.1
[3.4.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.4.0
[3.3.3]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.3
[3.3.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.2
[3.3.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.1
[3.3.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.0
[3.2.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.1
[3.2.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.0
[3.1.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.1.0
[3.0.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.2
[3.0.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.1
[3.0.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.0
[2.7.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v2.7.0

<!-- PR links -->

[#1719]: https://github.com/munich-quantum-toolkit/core/pull/1719
[#1702]: https://github.com/munich-quantum-toolkit/core/pull/1702
[#1694]: https://github.com/munich-quantum-toolkit/core/pull/1694
[#1662]: https://github.com/munich-quantum-toolkit/core/pull/1662
[#1654]: https://github.com/munich-quantum-toolkit/core/pull/1654
[#1652]: https://github.com/munich-quantum-toolkit/core/pull/1652
[#1596]: https://github.com/munich-quantum-toolkit/core/pull/1596
[#1593]: https://github.com/munich-quantum-toolkit/core/pull/1593
[#1571]: https://github.com/munich-quantum-toolkit/core/pull/1571
[#1564]: https://github.com/munich-quantum-toolkit/core/pull/1564
[#1528]: https://github.com/munich-quantum-toolkit/core/pull/1528
[#1507]: https://github.com/munich-quantum-toolkit/core/pull/1507
[#1481]: https://github.com/munich-quantum-toolkit/core/pull/1481
[#1453]: https://github.com/munich-quantum-toolkit/core/pull/1453
[#1447]: https://github.com/munich-quantum-toolkit/core/pull/1447
[#1444]: https://github.com/munich-quantum-toolkit/core/pull/1444
[#1437]: https://github.com/munich-quantum-toolkit/core/pull/1437
[#1415]: https://github.com/munich-quantum-toolkit/core/pull/1415
[#1414]: https://github.com/munich-quantum-toolkit/core/pull/1414
[#1413]: https://github.com/munich-quantum-toolkit/core/pull/1413
[#1412]: https://github.com/munich-quantum-toolkit/core/pull/1412
[#1411]: https://github.com/munich-quantum-toolkit/core/pull/1411
[#1406]: https://github.com/munich-quantum-toolkit/core/pull/1406
[#1403]: https://github.com/munich-quantum-toolkit/core/pull/1403
[#1402]: https://github.com/munich-quantum-toolkit/core/pull/1402
[#1385]: https://github.com/munich-quantum-toolkit/core/pull/1385
[#1384]: https://github.com/munich-quantum-toolkit/core/pull/1384
[#1383]: https://github.com/munich-quantum-toolkit/core/pull/1383
[#1382]: https://github.com/munich-quantum-toolkit/core/pull/1382
[#1381]: https://github.com/munich-quantum-toolkit/core/pull/1381
[#1380]: https://github.com/munich-quantum-toolkit/core/pull/1380
[#1378]: https://github.com/munich-quantum-toolkit/core/pull/1378
[#1375]: https://github.com/munich-quantum-toolkit/core/pull/1375
[#1371]: https://github.com/munich-quantum-toolkit/core/pull/1371
[#1359]: https://github.com/munich-quantum-toolkit/core/pull/1359
[#1355]: https://github.com/munich-quantum-toolkit/core/pull/1355
[#1338]: https://github.com/munich-quantum-toolkit/core/pull/1338
[#1336]: https://github.com/munich-quantum-toolkit/core/pull/1336
[#1328]: https://github.com/munich-quantum-toolkit/core/pull/1328
[#1327]: https://github.com/munich-quantum-toolkit/core/pull/1327
[#1310]: https://github.com/munich-quantum-toolkit/core/pull/1310
[#1301]: https://github.com/munich-quantum-toolkit/core/pull/1301
[#1300]: https://github.com/munich-quantum-toolkit/core/pull/1300
[#1299]: https://github.com/munich-quantum-toolkit/core/pull/1299
[#1294]: https://github.com/munich-quantum-toolkit/core/pull/1294
[#1293]: https://github.com/munich-quantum-toolkit/core/pull/1293
[#1287]: https://github.com/munich-quantum-toolkit/core/pull/1287
[#1283]: https://github.com/munich-quantum-toolkit/core/pull/1283
[#1279]: https://github.com/munich-quantum-toolkit/core/pull/1279
[#1276]: https://github.com/munich-quantum-toolkit/core/pull/1276
[#1271]: https://github.com/munich-quantum-toolkit/core/pull/1271
[#1269]: https://github.com/munich-quantum-toolkit/core/pull/1269
[#1263]: https://github.com/munich-quantum-toolkit/core/pull/1263
[#1247]: https://github.com/munich-quantum-toolkit/core/pull/1247
[#1246]: https://github.com/munich-quantum-toolkit/core/pull/1246
[#1243]: https://github.com/munich-quantum-toolkit/core/pull/1243
[#1237]: https://github.com/munich-quantum-toolkit/core/pull/1237
[#1236]: https://github.com/munich-quantum-toolkit/core/pull/1236
[#1235]: https://github.com/munich-quantum-toolkit/core/pull/1235
[#1232]: https://github.com/munich-quantum-toolkit/core/pull/1232
[#1224]: https://github.com/munich-quantum-toolkit/core/pull/1224
[#1223]: https://github.com/munich-quantum-toolkit/core/pull/1223
[#1211]: https://github.com/munich-quantum-toolkit/core/pull/1211
[#1210]: https://github.com/munich-quantum-toolkit/core/pull/1210
[#1209]: https://github.com/munich-quantum-toolkit/core/pull/1209
[#1207]: https://github.com/munich-quantum-toolkit/core/pull/1207
[#1186]: https://github.com/munich-quantum-toolkit/core/pull/1186
[#1181]: https://github.com/munich-quantum-toolkit/core/pull/1181
[#1180]: https://github.com/munich-quantum-toolkit/core/pull/1180
[#1165]: https://github.com/munich-quantum-toolkit/core/pull/1165
[#1164]: https://github.com/munich-quantum-toolkit/core/pull/1164
[#1157]: https://github.com/munich-quantum-toolkit/core/pull/1157
[#1151]: https://github.com/munich-quantum-toolkit/core/pull/1151
[#1148]: https://github.com/munich-quantum-toolkit/core/pull/1148
[#1147]: https://github.com/munich-quantum-toolkit/core/pull/1147
[#1140]: https://github.com/munich-quantum-toolkit/core/pull/1140
[#1139]: https://github.com/munich-quantum-toolkit/core/pull/1139
[#1117]: https://github.com/munich-quantum-toolkit/core/pull/1117
[#1116]: https://github.com/munich-quantum-toolkit/core/pull/1116
[#1108]: https://github.com/munich-quantum-toolkit/core/pull/1108
[#1106]: https://github.com/munich-quantum-toolkit/core/pull/1106
[#1100]: https://github.com/munich-quantum-toolkit/core/pull/1100
[#1099]: https://github.com/munich-quantum-toolkit/core/pull/1099
[#1098]: https://github.com/munich-quantum-toolkit/core/pull/1098
[#1091]: https://github.com/munich-quantum-toolkit/core/pull/1091
[#1089]: https://github.com/munich-quantum-toolkit/core/pull/1089
[#1088]: https://github.com/munich-quantum-toolkit/core/pull/1088
[#1076]: https://github.com/munich-quantum-toolkit/core/pull/1076
[#1075]: https://github.com/munich-quantum-toolkit/core/pull/1075
[#1071]: https://github.com/munich-quantum-toolkit/core/pull/1071
[#1047]: https://github.com/munich-quantum-toolkit/core/pull/1047
[#1042]: https://github.com/munich-quantum-toolkit/core/pull/1042
[#1020]: https://github.com/munich-quantum-toolkit/core/pull/1020
[#1019]: https://github.com/munich-quantum-toolkit/core/pull/1019
[#1010]: https://github.com/munich-quantum-toolkit/core/pull/1010
[#1001]: https://github.com/munich-quantum-toolkit/core/pull/1001
[#996]: https://github.com/munich-quantum-toolkit/core/pull/996
[#984]: https://github.com/munich-quantum-toolkit/core/pull/984
[#982]: https://github.com/munich-quantum-toolkit/core/pull/982
[#975]: https://github.com/munich-quantum-toolkit/core/pull/975
[#973]: https://github.com/munich-quantum-toolkit/core/pull/973
[#964]: https://github.com/munich-quantum-toolkit/core/pull/964
[#959]: https://github.com/munich-quantum-toolkit/core/pull/959
[#934]: https://github.com/munich-quantum-toolkit/core/pull/934
[#933]: https://github.com/munich-quantum-toolkit/core/pull/933
[#932]: https://github.com/munich-quantum-toolkit/core/pull/932
[#931]: https://github.com/munich-quantum-toolkit/core/pull/931
[#930]: https://github.com/munich-quantum-toolkit/core/pull/930
[#926]: https://github.com/munich-quantum-toolkit/core/pull/926
[#921]: https://github.com/munich-quantum-toolkit/core/pull/921
[#913]: https://github.com/munich-quantum-toolkit/core/pull/913
[#912]: https://github.com/munich-quantum-toolkit/core/pull/912
[#911]: https://github.com/munich-quantum-toolkit/core/pull/911
[#908]: https://github.com/munich-quantum-toolkit/core/pull/908
[#900]: https://github.com/munich-quantum-toolkit/core/pull/900
[#897]: https://github.com/munich-quantum-toolkit/core/pull/897
[#895]: https://github.com/munich-quantum-toolkit/core/pull/895
[#893]: https://github.com/munich-quantum-toolkit/core/pull/893
[#892]: https://github.com/munich-quantum-toolkit/core/pull/892
[#886]: https://github.com/munich-quantum-toolkit/core/pull/886
[#885]: https://github.com/munich-quantum-toolkit/core/pull/885
[#883]: https://github.com/munich-quantum-toolkit/core/pull/883
[#882]: https://github.com/munich-quantum-toolkit/core/pull/882
[#879]: https://github.com/munich-quantum-toolkit/core/pull/879
[#878]: https://github.com/munich-quantum-toolkit/core/pull/878
[#877]: https://github.com/munich-quantum-toolkit/core/pull/877
[#866]: https://github.com/munich-quantum-toolkit/core/pull/866
[#860]: https://github.com/munich-quantum-toolkit/core/pull/860
[#859]: https://github.com/munich-quantum-toolkit/core/pull/859
[#858]: https://github.com/munich-quantum-toolkit/core/pull/858
[#849]: https://github.com/munich-quantum-toolkit/core/pull/849
[#847]: https://github.com/munich-quantum-toolkit/core/pull/847
[#846]: https://github.com/munich-quantum-toolkit/core/pull/846
[#842]: https://github.com/munich-quantum-toolkit/core/pull/842
[#839]: https://github.com/munich-quantum-toolkit/core/pull/839
[#838]: https://github.com/munich-quantum-toolkit/core/pull/838
[#832]: https://github.com/munich-quantum-toolkit/core/pull/832
[#831]: https://github.com/munich-quantum-toolkit/core/pull/831
[#822]: https://github.com/munich-quantum-toolkit/core/pull/822
[#817]: https://github.com/munich-quantum-toolkit/core/pull/817
[#810]: https://github.com/munich-quantum-toolkit/core/pull/810
[#807]: https://github.com/munich-quantum-toolkit/core/pull/807
[#802]: https://github.com/munich-quantum-toolkit/core/pull/802
[#798]: https://github.com/munich-quantum-toolkit/core/pull/798
[#789]: https://github.com/munich-quantum-toolkit/core/pull/789
[#763]: https://github.com/munich-quantum-toolkit/core/pull/763
[#762]: https://github.com/munich-quantum-toolkit/core/pull/762
[#758]: https://github.com/munich-quantum-toolkit/core/pull/758
[#741]: https://github.com/munich-quantum-toolkit/core/pull/741
[#724]: https://github.com/munich-quantum-toolkit/core/pull/724
[#662]: https://github.com/munich-quantum-toolkit/core/pull/662
[#543]: https://github.com/munich-quantum-toolkit/core/pull/543
[**@a9b7e70**]: https://github.com/munich-quantum-toolkit/core/pull/798/commits/a9b7e70aaeb532fe8e1e31a7decca86d81eb523f

<!-- Contributor -->

[**@burgholzer**]: https://github.com/burgholzer
[**@ystade**]: https://github.com/ystade
[**@DRovara**]: https://github.com/DRovara
[**@flowerthrower**]: https://github.com/flowerthrower
[**@BertiFlorea**]: https://github.com/BertiFlorea
[**@M-J-Hochreiter**]: https://github.com/M-J-Hochreiter
[**@rotmanjanez**]: https://github.com/rotmanjanez
[**@pehamTom**]: https://github.com/pehamTom
[**@MatthiasReumann**]: https://github.com/MatthiasReumann
[**@denialhaag**]: https://github.com/denialhaag
[**q-inho**]: https://github.com/q-inho
[**@li-mingbao**]: https://github.com/li-mingbao
[**@lavanya-m-k**]: https://github.com/lavanya-m-k
[**@taminob**]: https://github.com/taminob
[**@jannikpflieger**]: https://github.com/jannikpflieger
[**@lsschmid**]: https://github.com/lsschmid
[**@marcelwa**]: https://github.com/marcelwa
[**@lirem101**]: https://github.com/lirem101
[**@keefehuang**]: https://github.com/keefehuang

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/core/releases
[munich-quantum-toolkit]: https://github.com/munich-quantum-toolkit
[PEP 639]: https://peps.python.org/pep-0639/
[PEP 735]: https://peps.python.org/pep-0735/
