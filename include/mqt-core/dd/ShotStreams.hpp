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

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <random>

namespace dd {
/// A logical sampling stream, independent of its executing thread.
struct ShotStream {
  size_t shots;
  uint64_t seed;
};

/// Partition shots and seeds across a fixed number of logical streams.
///
/// The caller supplies a positive stream count no greater than the shot count.
[[nodiscard]] inline llvm::SmallVector<ShotStream, 8>
makeShotStreams(size_t shots, size_t streams, uint64_t seed) {
  std::mt19937_64 rng(seed);
  llvm::SmallVector<ShotStream, 8> result;
  result.reserve(streams);
  for (size_t i = 0; i < streams; ++i) {
    result.push_back(
        {shots / streams + (i < shots % streams), i == 0 ? seed : rng()});
  }
  return result;
}
} // namespace dd
