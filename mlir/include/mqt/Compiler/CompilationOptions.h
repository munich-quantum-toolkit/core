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
#include <cstdint>
#include <optional>

namespace mlir {

/// Controls for native placement and routing.
struct MappingOptions {
  /// Positive trial count; omission uses the available logical CPU count.
  std::optional<size_t> trials;
  /// Positive number of forward/backward rounds to refine the initial layout.
  size_t iterations = 1;
  /// Additional two-qubit gates considered during routing; zero disables
  /// lookahead.
  size_t lookahead = 20;
};

/// Options shared by compilation entry points.
struct CompilationOptions {
  /// Override compiler randomness, including explicitly seeded custom passes.
  /// Omission preserves each pass's default or explicitly configured seed.
  std::optional<uint64_t> seed;
  bool enableTiming = false;
  bool enableStatistics = false;
  MappingOptions mapping;
};

} // namespace mlir
