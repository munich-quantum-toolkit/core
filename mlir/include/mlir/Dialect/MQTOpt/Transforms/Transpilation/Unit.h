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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <utility>

namespace mqt::ir::opt {

/// @brief A Unit divides a quantum-classical program into routable sections.
template <class Derived> class Unit {
public:
  /// @brief Compute and return subsequent units.
  [[nodiscard]] mlir::SmallVector<Derived, 3> next() {
    return static_cast<Derived*>(this)->nextImpl();
  }

  /// @returns an iterator pointing at the first element of the unit.
  [[nodiscard]] auto begin() const {
    return static_cast<const Derived*>(this)->beginImpl();
  }

  /// @returns an iterator pointing at the past-the-end position.
  [[nodiscard]] auto end() const {
    return static_cast<const Derived*>(this)->endImpl();
  }

  /// @returns the managed layout.
  [[nodiscard]] Layout& layout() { return layout_; }

protected:
  Unit(Layout layout, mlir::Region* region)
      : layout_(std::move(layout)), region_(region) {}

  /// @brief The layout this unit manages.
  Layout layout_;
  /// @brief The region this unit belongs to.
  mlir::Region* region_;
  /// @brief Pointer to the next dividing operation.
  mlir::Operation* divider_{};
};

} // namespace mqt::ir::opt
