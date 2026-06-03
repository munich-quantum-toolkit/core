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

#include <cstddef>

namespace mlir::qir {

/**
 * @brief State object for tracking QIR metadata during conversion
 *
 * @details
 * This struct maintains metadata about the QIR program being built:
 * - Qubit and result counts for QIR metadata
 * - Whether dynamic memory management is needed
 */
struct QIRMetadata {
  /// Number of qubits used in the module
  size_t numQubits{0};
  /// Number of measurement results stored in the module
  size_t numResults{0};
  /// Whether the module uses dynamic qubit management
  bool useDynamicQubit{false};
  /// Whether the module uses dynamic result management
  bool useDynamicResult{false};
};

} // namespace mlir::qir
