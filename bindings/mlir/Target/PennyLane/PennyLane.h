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

#include "nanobind/nanobind.h"

namespace mqt::bindings::pennylane {

/// Import preprocessed gates and expose all wires as terminal classical
/// outputs.
[[nodiscard]] mlir::QCProgram importCircuit(const nanobind::object& tape,
                                            const nanobind::object& wires);

/// Register the native context and sample decoder in a private submodule.
void registerBindings(nanobind::module_& m);

} // namespace mqt::bindings::pennylane
