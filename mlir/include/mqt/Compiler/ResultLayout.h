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
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class StringRef;
template <typename T> class Expected;
} // namespace llvm

namespace mlir {

/// Reconstruct logical output bits from a compiled program's binary samples.
///
/// Both indices and entries use least-significant-bit-first order. An index
/// selects a raw output bit; an empty entry supplies zero. The number of
/// entries is the logical output width.
struct ResultLayout {
  size_t rawWidth = 0;
  std::vector<std::optional<size_t>> bits;

  /// Whether raw samples already have the logical width and order.
  [[nodiscard]] bool isIdentity() const noexcept;

  /// Validate and reconstruct one MSB-first binary sample.
  [[nodiscard]] llvm::Expected<std::string>
  decodeBits(llvm::StringRef raw) const;
};

/// Validate and decode a histogram, merging equal logical outcomes.
/// ASCII spaces separate registers and are ignored. Width is the logical width;
/// when absent, infer it from the layout or first sample. Require exactly
/// shots.
[[nodiscard]] llvm::Expected<std::map<std::string, size_t>>
decodeCounts(std::map<std::string, size_t> counts, std::optional<size_t> width,
             size_t shots,
             const std::optional<ResultLayout>& layout = std::nullopt);

/// Validate and decode samples without changing their order or correlations.
/// Uses the same space, width, and shot-total policy as decodeCounts.
[[nodiscard]] llvm::Expected<std::vector<std::string>>
decodeShots(std::vector<std::string> samples, std::optional<size_t> width,
            size_t shots,
            const std::optional<ResultLayout>& layout = std::nullopt);

} // namespace mlir
