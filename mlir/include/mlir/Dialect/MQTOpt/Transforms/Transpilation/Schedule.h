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

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

struct Schedule {
  using GateLayer = SmallVector<QubitIndexPair>;
  using GateLayers = SmallVector<GateLayer>;

  struct OpLayer {
    /// @brief All ops contained inside this layer.
    SmallVector<Operation*, 64> ops;
    /// @brief The first op in ops in textual IR order.
    Operation* anchor{};

    /// @brief Add op to ops and reset anchor if necessary.
    void addOp(Operation* op) {
      ops.emplace_back(op);
      if (anchor == nullptr || op->isBeforeInBlock(anchor)) {
        anchor = op;
      }
    }

    [[nodiscard]] bool empty() const { return ops.empty(); }
  };

  using OpLayers = SmallVector<OpLayer, 0>;

  /// @brief Vector of layers containing the program indices of the two-qubit
  /// gates inside it.
  GateLayers gateLayers;

  /// @brief Vector of layers containing the ops inside it.
  OpLayers opLayers;

  /// @returns a window of gate layers from (start, start + nlookahead).
  MutableArrayRef<GateLayer> getWindow(std::size_t start,
                                       std::size_t nlookahead);

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const;
#endif
};

/**
 * @brief Given a layout, divide the circuit into layers and schedule the ops in
 * their respective layer.
 */
Schedule getSchedule(const Layout& layout, Region& region);
} // namespace mqt::ir::opt
