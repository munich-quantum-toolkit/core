/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"

#include "nanobind/nanobind.h"

#include <cstddef>
#include <optional>
#include <string>

namespace mqt::bindings::qiskit {

namespace nb = nanobind;

/// Import a Qiskit QuantumCircuit into an independently owned QC program.
///
/// Ordinary import translates gate definitions and allocates program qubits.
/// Native import fixes circuit indices to device qubits and uses standard QC
/// operations or opaque native calls, without reading gate definitions.
/// The caller validates device support and placements. A complete transpiler
/// layout is retained as metadata; it does not remap circuit indices.
///
/// @param circuit Borrowed Python circuit; it is neither modified nor consumed.
/// @param native Whether the circuit is already compiled for a device.
/// @param ignoreLayout Whether to skip importing the transpiler layout.
/// @returns A program owning its MLIR context and translated module.
/// @throws std::runtime_error If the circuit is outside the supported import
/// subset.
/// @throws nanobind::python_error If a Python API call fails.
[[nodiscard]] mlir::QCProgram importCircuit(nb::handle circuit,
                                            bool native = false,
                                            bool ignoreLayout = false);

/// Snapshot a Qiskit Target or BackendV2 into an owned compiler target.
///
/// Unrepresentable gates warn by default and fail when selected explicitly.
/// Instructions without gate applicability are skipped by default.
[[nodiscard]] mlir::CompilerTarget
importTarget(nb::handle target, nb::handle operationNames,
             const std::optional<std::string>& name);

/// Import gate capabilities independently of device width and placement.
[[nodiscard]] mlir::CompilerTarget::NativeOperations
importNativeOperations(nb::handle target, nb::handle operationNames);

/// Import a Qiskit target and its output capabilities together.
///
/// A native width selects gates that fit and ignores device placement.
/// Strict import rejects unsupported applicable gates.
[[nodiscard]] mlir::TargetEnvironment
importTargetEnvironment(nb::handle target, std::optional<size_t> nativeWidth,
                        bool strict);

/// Describe Qiskit output capabilities, restricted to a target when supplied.
[[nodiscard]] mlir::PayloadSpecification
payloadSpecification(nb::handle target = nb::none());

/// Consume a QC program to build a Qiskit circuit for an optional target.
[[nodiscard]] nb::object
exportCircuit(mlir::QCProgram program,
              const mlir::CompilerTarget* target = nullptr);

/// Export a compiled circuit, including layout and payload control-flow limits.
[[nodiscard]] nb::object
exportCircuit(mlir::QCProgram program,
              const mlir::TargetEnvironment& environment);

} // namespace mqt::bindings::qiskit
