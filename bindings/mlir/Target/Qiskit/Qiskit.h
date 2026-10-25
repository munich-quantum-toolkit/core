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

/// Import a Qiskit QuantumCircuit into a newly owned QC program.
[[nodiscard]] mlir::QCProgram importCircuit(nb::handle circuit);

/// Snapshot a Qiskit Target or BackendV2 into an owned compiler target.
///
/// Unrepresentable gates warn by default and fail when selected explicitly.
/// Instructions without gate applicability are skipped by default.
[[nodiscard]] mlir::CompilerTarget
importTarget(nb::handle target, nb::handle operationNames,
             const std::optional<std::string>& name);

/// Import gate capabilities independently of physical width and placement.
[[nodiscard]] mlir::CompilerTarget::NativeOperations
importNativeOperations(nb::handle target, nb::handle operationNames);

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
