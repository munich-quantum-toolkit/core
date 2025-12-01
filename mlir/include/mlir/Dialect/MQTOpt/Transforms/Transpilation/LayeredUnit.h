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
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

using GateLayer = mlir::SmallVector<QubitIndexPair, 16>;

struct OpLayer {
  /// @brief All ops contained inside this layer.
  mlir::SmallVector<mlir::Operation*, 64> ops;
  /// @brief The first op in ops in textual IR order.
  mlir::Operation* anchor{};

  /// @brief Add op to ops and reset anchor if necessary.
  void addOp(mlir::Operation* op) {
    ops.emplace_back(op);
    if (anchor == nullptr || op->isBeforeInBlock(anchor)) {
      anchor = op;
    }
  }

  [[nodiscard]] bool empty() const { return ops.empty(); }
};

struct WindowView {
  mlir::ArrayRef<GateLayer> gateLayers;
  const OpLayer* opLayer{};
  mlir::Operation* nextAnchor{};
};

class SlidingWindow {
public:
  struct Iterator {
    using value_type = WindowView;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    Iterator() = default;
    Iterator(const SlidingWindow* window, std::size_t pos)
        : window(window), pos(pos) {}

    WindowView operator*() const {
      WindowView w;
      const auto sz = window->opLayers->size();
      const auto len = std::min(1 + window->nlookahead, sz - pos);
      w.gateLayers = mlir::ArrayRef<GateLayer>(*window->gateLayers).slice(pos, len);
      w.opLayer = &(*window->opLayers)[pos];
      if (pos + 1 < window->gateLayers->size()) {
        w.nextAnchor = (*window->opLayers)[pos + 1].anchor;
      }
      return w;
    }

    Iterator& operator++() {
      ++pos;
      return *this;
    }

    Iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const Iterator& other) const {
      return pos == other.pos && window == other.window;
    }

  private:
    const SlidingWindow* window;
    std::size_t pos;
  };

  explicit SlidingWindow(const mlir::SmallVector<GateLayer>& gateLayers,
                         const mlir::SmallVector<OpLayer, 0>& opLayers,
                         const std::size_t nlookahead)
      : gateLayers(&gateLayers), opLayers(&opLayers), nlookahead(nlookahead) {}

  [[nodiscard]] Iterator begin() const { return {this, 0}; }
  [[nodiscard]] Iterator end() const { return {this, opLayers->size()}; }

  static_assert(std::forward_iterator<Iterator>);

private:
  const mlir::SmallVector<GateLayer>* gateLayers;
  const mlir::SmallVector<OpLayer, 0>* opLayers;
  std::size_t nlookahead;
};

/// @brief A LayeredUnit traverses a program layer-by-layer.
class LayeredUnit : public Unit {
public:
  static LayeredUnit fromEntryPointFunction(mlir::func::FuncOp func,
                                            const std::size_t nqubits) {
    Layout layout(nqubits);
    for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
      layout.add(op.getIndex(), op.getIndex(), op.getQubit());
    });
    return {std::move(layout), &func.getBody()};
  }

  LayeredUnit(Layout layout, mlir::Region* region, bool restore = false);

  [[nodiscard]] mlir::SmallVector<LayeredUnit, 3> next();
  [[nodiscard]] SlidingWindow slidingWindow(std::size_t nlookahead) const;

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const;
#endif

private:
  mlir::SmallVector<GateLayer> gateLayers;
  mlir::SmallVector<OpLayer, 0> opLayers;
};
} // namespace mqt::ir::opt
