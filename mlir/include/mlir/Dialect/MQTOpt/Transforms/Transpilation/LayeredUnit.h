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
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

struct Layer {
  /// @brief All (zero, one, two-qubit) ops contained inside this layer.
  mlir::SmallVector<mlir::Operation*, 64> ops;
  /// @brief The program index pairs of all two-qubit ops.
  mlir::SmallVector<QubitIndexPair, 16> twoQubitProgs;
  /// @brief The first op in ops in textual IR order.
  mlir::Operation* anchor{};

  /// @brief Add op to ops and reset anchor if necessary.
  void addOp(mlir::Operation* op) {
    ops.emplace_back(op);
    if (anchor == nullptr || op->isBeforeInBlock(anchor)) {
      anchor = op;
    }
  }
  /// @returns true iff. there are no ops in this layer.
  [[nodiscard]] bool hasZeroOps() const { return ops.empty(); }
  /// @returns true iff. there are no two-qubit ops in this layer.
  [[nodiscard]] bool hasZero2QOps() const { return twoQubitProgs.empty(); }
};

/// @brief A LayeredUnit traverses a program layer-by-layer.
class LayeredUnit : public Unit {
public:
  using Layers = mlir::SmallVector<Layer, 0>;

  [[nodiscard]] static LayeredUnit
  fromEntryPointFunction(mlir::func::FuncOp func, std::size_t nqubits);

  LayeredUnit(Layout layout, mlir::Region* region, bool restore = false);

  [[nodiscard]] mlir::SmallVector<LayeredUnit, 3> next();
  [[nodiscard]] Layers::const_iterator begin() const { return layers_.begin(); }
  [[nodiscard]] Layers::const_iterator end() const { return layers_.end(); }
  [[nodiscard]] const Layer& operator[](std::size_t i) const {
    return layers_[i];
  }
  [[nodiscard]] std::size_t size() const { return layers_.size(); }

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(llvm::raw_ostream& os = llvm::dbgs()) const;
#endif

private:
  Layers layers_;
};
} // namespace mqt::ir::opt
