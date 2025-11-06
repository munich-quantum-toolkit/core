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

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <utility>

#define DEBUG_TYPE "route-sc"

namespace mqt::ir::opt {

using namespace mlir;

/**
 * @brief Create and return SWAPOp for two qubits.
 *
 * Expects the rewriter to be set to the correct position.
 *
 * @param location The Location to attach to the created op.
 * @param in0 First input qubit SSA value.
 * @param in1 Second input qubit SSA value.
 * @param rewriter A PatternRewriter.
 * @return The created SWAPOp.
 */
[[nodiscard]] inline SWAPOp createSwap(Location location, Value in0, Value in1,
                                       PatternRewriter& rewriter) {
  const SmallVector<Type> resultTypes{in0.getType(), in1.getType()};
  const SmallVector<Value> inQubits{in0, in1};

  return rewriter.create<SWAPOp>(
      /* location = */ location,
      /* out_qubits = */ resultTypes,
      /* pos_ctrl_out_qubits = */ TypeRange{},
      /* neg_ctrl_out_qubits = */ TypeRange{},
      /* static_params = */ nullptr,
      /* params_mask = */ nullptr,
      /* params = */ ValueRange{},
      /* in_qubits = */ inQubits,
      /* pos_ctrl_in_qubits = */ ValueRange{},
      /* neg_ctrl_in_qubits = */ ValueRange{});
}

/**
 * @brief Replace all uses of a value within a region and its nested regions,
 * except for a specific operation.
 *
 * @param oldValue The value to replace
 * @param newValue The new value to use
 * @param region The region in which to perform replacements
 * @param exceptOp Operation to exclude from replacements
 * @param rewriter The pattern rewriter
 */
inline void replaceAllUsesInRegionAndChildrenExcept(Value oldValue,
                                                    Value newValue,
                                                    Region* region,
                                                    Operation* exceptOp,
                                                    PatternRewriter& rewriter) {
  if (oldValue == newValue) {
    return;
  }

  rewriter.replaceUsesWithIf(oldValue, newValue, [&](OpOperand& use) {
    Operation* user = use.getOwner();
    if (user == exceptOp) {
      return false;
    }

    // For other blocks, check if in region tree
    Region* userRegion = user->getParentRegion();
    while (userRegion) {
      if (userRegion == region) {
        return true;
      }
      userRegion = userRegion->getParentRegion();
    }
    return false;
  });
}

class MapperBase {
public:
  explicit MapperBase(std::unique_ptr<Architecture> arch)
      : arch(std::move(arch)) {}

  virtual ~MapperBase() = default;

  [[nodiscard]] virtual LogicalResult
  rewrite(func::FuncOp func, PatternRewriter& rewriter) const = 0;

  [[nodiscard]] LogicalResult rewrite(ModuleOp module) const {
    PatternRewriter rewriter(module->getContext());
    for (auto func : module.getOps<func::FuncOp>()) {
      if (rewrite(func, rewriter).failed()) {
        return failure();
      }
    }
    return success();
  }

protected:
  /**
   * @brief Insert SWAPs at the rewriter's insertion point and update the
   * layout.
   */
  static void insertSWAPs(ArrayRef<QubitIndexPair> swaps, Layout& layout,
                          Location anchor, PatternRewriter& rewriter) {
    for (const auto [hw0, hw1] : swaps) {
      const Value in0 = layout.lookupHardwareValue(hw0);
      const Value in1 = layout.lookupHardwareValue(hw1);
      [[maybe_unused]] const auto [prog0, prog1] =
          layout.getProgramIndices(hw0, hw1);

      LLVM_DEBUG({
        llvm::dbgs() << llvm::format(
            "route: swap= p%d:h%d, p%d:h%d <- p%d:h%d, p%d:h%d\n", prog1, hw0,
            prog0, hw1, prog0, hw0, prog1, hw1);
      });

      auto swap = createSwap(anchor, in0, in1, rewriter);
      const auto [out0, out1] = getOuts(swap);

      rewriter.setInsertionPointAfter(swap);
      replaceAllUsesInRegionAndChildrenExcept(
          in0, out1, swap->getParentRegion(), swap, rewriter);
      replaceAllUsesInRegionAndChildrenExcept(
          in1, out0, swap->getParentRegion(), swap, rewriter);

      layout.swap(in0, in1);
      layout.remapQubitValue(in0, out0);
      layout.remapQubitValue(in1, out1);
    }
  }

  /**
   * @returns true iff @p op is executable on the targeted architecture.
   */
  [[nodiscard]] bool isExecutable(UnitaryInterface op, Layout& layout) const {
    const auto [in0, in1] = getIns(op);
    return arch->areAdjacent(layout.lookupHardwareIndex(in0),
                             layout.lookupHardwareIndex(in1));
  }

  std::unique_ptr<Architecture> arch;
};
} // namespace mqt::ir::opt
