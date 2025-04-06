# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Common Changelog](https://common-changelog.org).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the exception that minor releases may include breaking changes.

## [Unreleased]

## [3.0.0] - 2025-04-06

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#300)._

### Added

- ✨ Ship shared C++ libraries with `mqt-core` Python package ([#662](https://github.com/munich-quantum-toolkit/core/issues/662)) ([**@burgholzer**](https://github.com/burgholzer))
- ✨ Add Python bindings for the DD package ([#838](https://github.com/munich-quantum-toolkit/core/issues/838)) ([**@burgholzer**](https://github.com/burgholzer))
- ✨ Add direct MQT `QuantumComputation` to Qiskit `QuantumCircuit` export ([#859](https://github.com/munich-quantum-toolkit/core/issues/859)) ([**@burgholzer**](https://github.com/burgholzer))
- ✨ Support for Qiskit 2.0+ ([#860](https://github.com/munich-quantum-toolkit/core/issues/860)) ([**@burgholzer**](https://github.com/burgholzer))
- ✨ Add initial infrastructure for MLIR within the MQT ([#878](https://github.com/munich-quantum-toolkit/core/issues/878), [#879](https://github.com/munich-quantum-toolkit/core/issues/879), [#892](https://github.com/munich-quantum-toolkit/core/issues/892), [#893](https://github.com/munich-quantum-toolkit/core/issues/893), [#895](https://github.com/munich-quantum-toolkit/core/issues/895)) ([**@burgholzer**](https://github.com/burgholzer), [**@ystade**](https://github.com/ystade), [**@DRovara**](https://github.com/DRovara), [**@flowerthrower**](https://github.com/flowerthrower), [**@BertiFlorea**](https://github.com/BertiFlorea))
- ✨ Add State Preparation Algorithm ([#543](https://github.com/munich-quantum-toolkit/core/issues/543)) ([**@M-J-Hochreiter**](https://github.com/M-J-Hochreiter))
- 🚸 Add support for indexed identifiers to OpenQASM 3 parser ([#832](https://github.com/munich-quantum-toolkit/core/issues/832)) ([**@burgholzer**](https://github.com/burgholzer))
- 🚸 Allow indexed registers as operation arguments ([#839](https://github.com/munich-quantum-toolkit/core/issues/839)) ([**@burgholzer**](https://github.com/burgholzer))
- 📝 Add documentation for the DD package ([#831](https://github.com/munich-quantum-toolkit/core/issues/831)) ([**@burgholzer**](https://github.com/burgholzer))
- 📝 Add documentation for the ZX package ([#817](https://github.com/munich-quantum-toolkit/core/issues/817)) ([**@pehamTom**](https://github.com/pehamTom))
- 📝 Add C++ API docs setup ([#817](https://github.com/munich-quantum-toolkit/core/issues/817)) ([**@pehamTom**](https://github.com/pehamTom), [**@burgholzer**](https://github.com/burgholzer))

### Changed

- **Breaking**: 🚚 MQT Core has moved to the [munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub organization
- **Breaking**: ✨ Adopt [PEP 735] dependency groups ([#762](https://github.com/munich-quantum-toolkit/core/issues/762)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Encapsulate the OpenQASM parser in its own library ([#822](https://github.com/munich-quantum-toolkit/core/issues/822)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Replace `Config` template from DD package with constructor argument ([#886](https://github.com/munich-quantum-toolkit/core/issues/886)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Remove template parameters from `MemoryManager` and adjacent classes ([#866](https://github.com/munich-quantum-toolkit/core/issues/866)) ([**@rotmanjanez**](https://github.com/rotmanjanez))
- **Breaking**: ♻️ Refactor algorithms to use factory functions instead of inheritance ([**@a9b7e70**](https://github.com/munich-quantum-toolkit/core/pull/798/commits/a9b7e70aaeb532fe8e1e31a7decca86d81eb523f)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Change pointer parameters to references in DD package ([#798](https://github.com/munich-quantum-toolkit/core/pull/798)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Change registers from typedef to actual type ([#807](https://github.com/munich-quantum-toolkit/core/issues/807)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ♻️ Refactor `NAComputation` class hierarchy ([#846](https://github.com/munich-quantum-toolkit/core/issues/846), [#877](https://github.com/munich-quantum-toolkit/core/issues/877)) ([**@ystade**](https://github.com/ystade))
- **Breaking**: ⬆️ Bump minimum required CMake version to `3.24.0` ([#879](https://github.com/munich-quantum-toolkit/core/issues/879)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: ⬆️ Bump minimum required `uv` version to `0.5.20` ([#802](https://github.com/munich-quantum-toolkit/core/issues/802)) ([**@burgholzer**](https://github.com/burgholzer))
- 📝 Rework existing project documentation ([#789](https://github.com/munich-quantum-toolkit/core/issues/789), [#842](https://github.com/munich-quantum-toolkit/core/issues/842)) ([**@burgholzer**](https://github.com/burgholzer))
- 📄 Use [PEP 639] license expressions ([#847](https://github.com/munich-quantum-toolkit/core/issues/847)) ([**@burgholzer**](https://github.com/burgholzer))

### Removed

- **Breaking**: 🔥 Remove the `Teleportation` gate from the IR ([#882](https://github.com/munich-quantum-toolkit/core/issues/882)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: 🔥 Remove parsers for `.real`, `.qc`, `.tfc`, and `GRCS` files ([#822](https://github.com/munich-quantum-toolkit/core/issues/822)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: 🔥 Remove tensor dump functionality ([#798](https://github.com/munich-quantum-toolkit/core/issues/798)) ([**@burgholzer**](https://github.com/burgholzer))
- **Breaking**: 🔥 Remove `extract_probability_vector` functionality ([#883](https://github.com/munich-quantum-toolkit/core/issues/883)) ([**@burgholzer**](https://github.com/burgholzer))

### Fixed

- 🐛 Fix Qiskit layout import and handling ([#849](https://github.com/munich-quantum-toolkit/core/issues/849), [#858](https://github.com/munich-quantum-toolkit/core/issues/858)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Properly handle timing literals in QASM parser ([#724](https://github.com/munich-quantum-toolkit/core/issues/724)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Fix stripping of idle qubits ([#763](https://github.com/munich-quantum-toolkit/core/issues/763)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Fix permutation handling in OpenQASM dump ([#810](https://github.com/munich-quantum-toolkit/core/issues/810)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Fix out-of-bounds error in ZX `EdgeIterator` ([#758](https://github.com/munich-quantum-toolkit/core/issues/758)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Fix endianness in DCX and XX_minus_YY gate matrix definition ([#741](https://github.com/munich-quantum-toolkit/core/issues/741)) ([**@burgholzer**](https://github.com/burgholzer))
- 🐛 Fix needless dummy register in empty circuit construction ([#758](https://github.com/munich-quantum-toolkit/core/issues/758)) ([**@burgholzer**](https://github.com/burgholzer))

## [2.7.0] - 2024-10-08

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/core/releases) for previous changelogs._

[PEP 639]: https://peps.python.org/pep-0639/
[PEP 735]: https://peps.python.org/pep-0735/
[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/munich-quantum-toolkit/core/compare/v2.7.0...v3.0.0
[2.7.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v2.7.0
