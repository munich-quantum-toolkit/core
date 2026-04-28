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

#include "GateKind.h"

#include <llvm/ADT/SmallVector.h>

#include <cstddef>

namespace mlir::qco::decomposition {

using QubitId = std::size_t;

/**
 * Lightweight decomposition-time gate record.
 *
 * This struct is intentionally independent from MLIR operations so helper code
 * can build and manipulate abstract one- and two-qubit circuits before they
 * are materialized back into the IR.
 */
struct Gate {
  /// Operation kind represented by this gate.
  GateKind type{GateKind::I};

  /// Gate parameters in operation-specific order.
  llvm::SmallVector<double, 3> parameter;

  /// Logical qubit ids used by the gate, in operand order.
  llvm::SmallVector<QubitId, 2> qubitId = {0};
};

} // namespace mlir::qco::decomposition
