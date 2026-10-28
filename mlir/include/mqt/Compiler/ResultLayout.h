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
#include <string>
#include <vector>

namespace llvm {
class StringRef;
template <typename T> class Expected;
} // namespace llvm

namespace mlir {

/// Canonical terminal readout over measured sites or all addressed qubits.
enum class TerminalReadout : uint8_t { Measured, All };

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

} // namespace mlir
