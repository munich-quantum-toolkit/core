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
/// Helpers for `NativeGateSynthesisPass` two-qubit window consolidation.

#pragma once

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"

#include <Eigen/Core>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

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
void collectUnitaryOpsInPreOrder(Operation* root,
                                 llvm::SmallVectorImpl<Operation*>& ops);

/// Tracks overlapping two-qubit windows on a module slice.
struct TwoQubitWindowConsolidator {
  /// Append-only list of windows discovered so far; closed windows are kept
  /// so `materialize()` can still rewrite them.
  llvm::SmallVector<TwoQubitBlock, 0> blocks;
  /// Maps each currently-open SSA qubit value to the index of the block
  /// that owns its trailing wire.
  llvm::DenseMap<Value, size_t> wireToBlock;

  /// Mark block `idx` as closed and remove its tracked wires from
  /// `wireToBlock`.
  void closeBlock(size_t idx);

  /// If `v` is currently tracked, close the block that owns it; otherwise
  /// do nothing. Used at synchronization points (barriers, fan-out, etc.).
  void closeBlockOnWire(Value v);

  /// State-machine step for one IR op, called in pre-order walk order.
  /// Extends an existing window, starts a fresh one, or closes conflicting
  /// windows depending on the op's kind and operand use pattern.
  void process(Operation* op, const NativeProfileSpec& spec);

  /// Rewrite each collected window whose accumulated unitary can be
  /// realized more cheaply by the native-gate synthesizer.
  /// Picks the best candidate per block via `selectBestCandidate`,
  /// gates the replacement on `shouldApplyBlockReplacement`, and emits the
  /// new sequence through `rewriter`.
  void materialize(IRRewriter& rewriter, const NativeProfileSpec& spec,
                   const ScoreWeights& weights);
};

} // namespace mlir::qco::native_synth
