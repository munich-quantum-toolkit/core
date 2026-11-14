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
#include <filesystem>
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
  /// Write a Chrome trace of the MLIR actions to this file.
  ///
  /// MLIR actions include pass runs and pattern applications. No value
  /// disables tracing, and `-` selects stdout. runDefaultPipeline and
  /// CompiledProgram::compile write one trace for the complete compilation.
  /// Typed transformations write one trace for their pass-manager run.
  std::optional<std::filesystem::path> traceFile;
};

} // namespace mlir
