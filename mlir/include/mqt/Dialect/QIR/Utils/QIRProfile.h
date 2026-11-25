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

namespace mlir {

/// The QIR profile represented by a QIR program.
enum class QIRProfile : uint8_t {
  /// The QIR Base Profile.
  Base,
  /// The QIR Adaptive Profile.
  Adaptive,
};

} // namespace mlir
