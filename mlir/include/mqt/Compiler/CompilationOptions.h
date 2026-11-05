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
  /// Positive trial count; omission uses the available logical CPU count, with
  /// a minimum of 8.
  std::optional<size_t> trials;
  /// Forward/backward refinement rounds; zero scores each start directly.
  size_t iterations = 2;
  /// Additional two-qubit gates considered during routing; zero disables
  /// lookahead.
  size_t lookahead = 20;
  /// Estimated node and layout bytes per routing search, per concurrent trial.
  ///
  /// Zero disables node expansion. Container overhead, caches, and IR are
  /// extra.
  size_t searchMemoryLimit = 256UL * 1024 * 1024;
};

/// Options shared by compilation entry points.
struct CompilationOptions {
  /// Override compiler randomness, including explicitly seeded custom passes.
  ///
  /// Omission preserves each pass's default or explicitly configured seed.
  std::optional<uint64_t> seed;
  bool enableTiming = false;
  bool enableStatistics = false;
  MappingOptions mapping;
  /// Borrow existing live helpers for multi-controlled gates when beneficial.
  ///
  /// False disables external borrowing; original controls remain available as
  /// internal workspace. Helpers are restored; no new qubits are allocated.
  /// Custom textual pipelines use the pass's borrow-ancillas option.
  bool borrowAncillas = true;
};

} // namespace mlir
