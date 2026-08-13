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

#include <nanobind/nanobind.h>

namespace mqt::bindings::qiskit {

namespace nb = nanobind;

/** Import a Qiskit QuantumCircuit into a newly owned QC program. */
[[nodiscard]] mlir::QCProgram importCircuit(nb::handle circuit);

/** Return a new Qiskit QuantumCircuit. */
[[nodiscard]] nb::object exportCircuit(const mlir::QCProgram& program);

} // namespace mqt::bindings::qiskit
