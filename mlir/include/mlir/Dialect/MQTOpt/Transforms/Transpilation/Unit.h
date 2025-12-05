/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <utility>

namespace mqt::ir::opt {

/// @brief A Unit divides a quantum-classical program into routable sections.
class Unit {
public:
  Unit(Layout layout, mlir::Region* region, bool restore = false)
      : layout_(std::move(layout)), region_(region), restore_(restore) {}

  /// @returns the managed layout.
  [[nodiscard]] Layout& layout() { return layout_; }

  /// @returns true iff. the unit has to be restored.
  [[nodiscard]] bool restore() const { return restore_; }

protected:
  /// @brief The layout this unit manages.
  Layout layout_;
  /// @brief The region this unit belongs to.
  mlir::Region* region_;
  /// @brief Pointer to the next dividing operation.
  mlir::Operation* divider_{};
  /// @brief Whether to uncompute the inserted SWAP sequence.
  bool restore_;
};

} // namespace mqt::ir::opt
