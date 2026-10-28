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
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class StringRef;
template <typename T> class Expected;
} // namespace llvm

namespace mlir {

/// Receiver syntax and terminal-readout requirements.
enum class SerializationProfile : uint8_t {
  Default,
  BraketSimulator,
  BraketNative,
  IBM,
};

/// Reconstruct logical output bits from a compiled program's binary samples.
///
/// Both indices and entries use least-significant-bit-first order. An integer
/// selects a raw output bit; a Boolean supplies a constant. The number of
/// entries is the logical output width.
struct ResultLayout {
  size_t rawWidth = 0;
  std::vector<std::variant<size_t, bool>> bits;

  /// Whether raw samples already have the logical width and order.
  [[nodiscard]] bool isIdentity() const noexcept;

  /// Validate and reconstruct one MSB-first binary sample.
  [[nodiscard]] llvm::Expected<std::string>
  decodeBits(llvm::StringRef raw) const;
};

} // namespace mlir
