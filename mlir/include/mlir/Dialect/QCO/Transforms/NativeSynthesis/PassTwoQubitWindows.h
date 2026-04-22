/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// \file
/// Helpers for `NativeGateSynthesisPass` two-qubit window consolidation. Not
/// a stable public API; kept in-tree for reuse by the pass (and its tests).

#pragma once

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <Eigen/Core>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <vector>

namespace mlir::qco::native_synth {

/// State for one maximal two-qubit window (plus absorbed one-qubit ops)
/// during consolidation.
struct TwoQubitBlock {
  Value wireA;
  Value wireB;
  llvm::SmallVector<Operation*, 8> ops;
  Eigen::Matrix4cd accum = Eigen::Matrix4cd::Identity();
  unsigned numTwoQ = 0;
  unsigned numOneQ = 0;
  bool anyNonNative = false;
  bool open = true;
};

/// Pre-order walk: every op implementing `UnitaryOpInterface` under `root`.
void collectUnitaryOpsInPreOrder(Operation* root, std::vector<Operation*>& ops);

/// Tracks overlapping two-qubit windows on a module slice; implemented in
/// ``NativeSynthesis/PassTwoQubitWindows.cpp``.
struct TwoQubitWindowConsolidator {
  llvm::SmallVector<TwoQubitBlock, 0> blocks;
  llvm::DenseMap<Value, size_t> wireToBlock;

  void closeBlock(size_t idx);
  void closeBlockOnWire(Value v);
  void process(Operation* op, const NativeProfileSpec& spec);
  void materialize(IRRewriter& rewriter, const NativeProfileSpec& spec,
                   const ScoreWeights& weights);
};

} // namespace mlir::qco::native_synth
