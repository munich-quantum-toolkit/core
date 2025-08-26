<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Added

- ✨ Add new pattern to MLIR pass `GateElimination` for identity gate removal ([#1140]) ([**@taminob**])
- ✨ Rename MLIR pass `CancelConsecutiveInverses` to `GateElimination`([#1140]) ([**@taminob**])
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

- 🚚 Rename `xxminusyy` to `xx_minus_yy` and `xxplusyy` to `xx_plus_yy` in MLIR dialects ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])
- 🚸 Add custom assembly format for operations in the MLIR dialects ([#1139]) ([**@burgholzer**])
- 🚸 Enable `InferTypeOpInterface` in the MLIR dialects to reduce explicit type information ([#1139]) ([**@burgholzer**])
- 🚚 Rename `check-quantum-opt` test target to `mqt-core-mlir-lit-test` ([#1139]) ([**@burgholzer**])
- ♻️ Update the `measure` operations in the MLIR dialects to no longer support more than one qubit being measured at once ([#1106]) ([**@DRovara**])
- 🚚 Rename `XXminusYY` to `XXminusYYOp` and `XXplusYY` to `XXplusYYOp` in MLIR dialects ([#1099]) ([**@denialhaag**])
- 🚚 Rename `MQTDyn` MLIR dialect to `MQTRef` ([#1098]) ([**@MatthiasReumann**])

### Removed

- Remove `ClassicControlledOperation` from C++ library and Python package ([#1117]) ([**@denialhaag**])

### Fixed

- ✨ Add missing `StandardOperation` conversions in MLIR roundtrip pass ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])

## [3.2.1] - 2025-08-01

### Fixed

- 🐛 Fix usage of `std::accumulate` by changing accumulator parameter from reference to value ([#1089]) ([**@denialhaag**])
- 🐛 Fix erroneous `contains` check in DD package ([#1088]) ([**@denialhaag**])

## [3.2.0] - 2025-07-31

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

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v3.2.1...HEAD
[3.2.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.1
[3.2.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.0
[3.1.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.1.0
[3.0.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.2
[3.0.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.1
[3.0.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.0
[2.7.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v2.7.0

<!-- PR links -->

[#1157]: https://github.com/munich-quantum-toolkit/core/pull/1157
[#1147]: https://github.com/munich-quantum-toolkit/core/pull/1147
[#1140]: https://github.com/munich-quantum-toolkit/core/pull/1140
[#1139]: https://github.com/munich-quantum-toolkit/core/pull/1139
[#1117]: https://github.com/munich-quantum-toolkit/core/pull/1117
[#1116]: https://github.com/munich-quantum-toolkit/core/pull/1116
[#1106]: https://github.com/munich-quantum-toolkit/core/pull/1106
[#1100]: https://github.com/munich-quantum-toolkit/core/pull/1100
[#1099]: https://github.com/munich-quantum-toolkit/core/pull/1099
[#1098]: https://github.com/munich-quantum-toolkit/core/pull/1098
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

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/core/releases
[munich-quantum-toolkit]: https://github.com/munich-quantum-toolkit
[PEP 639]: https://peps.python.org/pep-0639/
[PEP 735]: https://peps.python.org/pep-0735/
