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

namespace mlir::qco::decomposition {

/**
 * Lightweight gate identifiers used by decomposition utilities.
 *
 * These kinds intentionally stay independent from the core IR `qc::OpType`
 * enum so the MLIR/QCO decomposition layer does not depend on the `ir`
 * package.
 */
enum class GateKind : std::uint8_t {
  I = 0,
  H,
  P,
  U,
  U2,
  X,
  Y,
  Z,
  SX,
  RX,
  RY,
  RZ,
  R,
  RXX,
  RYY,
  RZZ,
  GPhase,
};

} // namespace mlir::qco::decomposition
