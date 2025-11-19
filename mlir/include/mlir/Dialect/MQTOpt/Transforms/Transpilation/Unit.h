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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Router.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/WireIterator.h"

#include <cstddef>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>
#include <utility>

#define DEBUG_TYPE "route-astar-sc"

namespace mqt::ir::opt {

using namespace mlir;

struct Wire {
  Wire(const WireIterator& it, QubitIndex index) : it(it), index(index) {}

  WireIterator it;
  QubitIndex index;
};

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

struct Schedule {
  /// @returns a window of gate layers from (start, start + nlookahead).
  [[nodiscard]] MutableArrayRef<GateLayer> window(std::size_t start,
                                                  std::size_t nlookahead) const;

#ifndef NDEBUG
  /// @brief Debug dump of the scheduled unit.
  LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const;
#endif

  /// @brief Layers of program indices of the two-qubit gates.
  GateLayers gateLayers;

  /// @brief Layers of the ops inside it.
  OpLayers opLayers;

  /// @brief Pointer to the next dividing operation.
  Operation* next{};
};

class Unit {
public:
  Unit(Layout layout, Region* region, bool restore = false)
      : layout(std::move(layout)), region(region), restore(restore) {}

  /// @brief Schedule the unit.
  void schedule();
  /// @brief Route the unit.
  void route(const AStarHeuristicRouter& router, std::size_t nlookahead,
             const Architecture& arch, PatternRewriter& rewriter);
  /// @brief If possible, advance to the next unit.
  [[nodiscard]] SmallVector<Unit, 3> advance();

private:
  Schedule s;
  Layout layout;
  Region* region;
  bool restore;
};

} // namespace mqt::ir::opt
