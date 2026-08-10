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

#include "mlir/Compiler/Programs.h"

namespace mqt::bindings::qiskit {

using PythonHandle = void;

[[nodiscard]] mlir::QCProgram importCircuit(PythonHandle* circuit);

/** Return a new owned Python reference to a Qiskit QuantumCircuit. */
[[nodiscard]] PythonHandle* exportCircuit(const mlir::QCProgram& program);

/** Inspect Qiskit lazily and report whether a released adapter is available. */
[[nodiscard]] bool compilerBridgeAvailable();

} // namespace mqt::bindings::qiskit
