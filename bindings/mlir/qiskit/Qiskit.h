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

#include "nanobind/nanobind.h"

#include <optional>
#include <string>

namespace mqt::bindings::qiskit {

namespace nb = nanobind;

/// Import a Qiskit QuantumCircuit into a newly owned QC program.
[[nodiscard]] mlir::QCProgram importCircuit(nb::handle circuit);

/// Return the Qiskit name of a standard gate supported as a native capability.
[[nodiscard]] std::optional<std::string> nativeGateName(nb::handle operation);

/// Return a new Qiskit QuantumCircuit, optionally for a compiler target.
[[nodiscard]] nb::object
exportCircuit(const mlir::QCProgram& program,
              const mlir::CompilerTarget* target = nullptr);

} // namespace mqt::bindings::qiskit
