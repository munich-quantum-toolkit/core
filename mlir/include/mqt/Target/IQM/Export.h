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

#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace mlir {
class CompilerTarget;

namespace qir {

/// Export placed QIR Base Profile to an IQM circuit JSON object.
///
/// Supports constant PRX and CZ gates followed by terminal measurements.
/// Each measurement must be recorded exactly once, in the desired output
/// order. Output array labels and positions define IQM Qiskit measurement keys.
/// Target sites supply the physical names; the module is not modified.
[[nodiscard]] llvm::Expected<std::string>
exportIQMJSON(const llvm::Module& module, const CompilerTarget& target);

} // namespace qir
} // namespace mlir
