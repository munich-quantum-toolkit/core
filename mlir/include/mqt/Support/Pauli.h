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

#include <cstdint>

namespace mlir::mqt {

/// Single-qubit identity or Pauli operator, encoded as I=0, X=1, Y=2, Z=3.
enum class PauliAxis : uint8_t { I = 0, X = 1, Y = 2, Z = 3 };

} // namespace mlir::mqt
