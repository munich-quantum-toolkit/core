/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::qco {

#define GEN_PASS_DEF_ELIDEPERMUTATIONS
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

/// Follow a validated straight-line wire to its terminal disposal operand.
static Value terminalQubit(Value root) {
  WireIterator wire(root);
  while (!isa<SinkOp, qtensor::InsertOp>(*wire)) {
    ++wire;
  }
  return wire.qubit();
}

namespace {

struct ElidePermutations final
    : impl::ElidePermutationsBase<ElidePermutations> {
  using ElidePermutationsBase::ElidePermutationsBase;

protected:
  void runOnOperation() override {
    auto function = mqt::getEntryPoint(getOperation());
    if (!function || !llvm::hasSingleElement(function.getBody()) ||
        llvm::any_of(function.getArgumentTypes(), isLinearQubitType) ||
        llvm::none_of(function.getResultTypes(), [](Type type) {
          return isa<cbit::RegisterType>(type);
        })) {
      return;
    }
    auto& block = function.getBody().front();
    auto swaps = llvm::to_vector(block.getOps<SWAPOp>());
    if (swaps.empty()) {
      return;
    }

    SmallVector<Value> roots;
    SmallVector<qtensor::InsertOp> insertions;
    Operation* firstDisposal = nullptr;
    for (auto& op : block) {
      if (isa<SinkOp, qtensor::DeallocOp>(op)) {
        if (firstDisposal == nullptr) {
          firstDisposal = &op;
        }
        continue;
      }
      if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
        if (!getConstantIntValue(insert.getIndex()) ||
            (firstDisposal != nullptr &&
             !insert.getIndex().getDefiningOp()->isBeforeInBlock(
                 firstDisposal))) {
          return;
        }
        insertions.push_back(insert);
        continue;
      }
      if (auto alloc = dyn_cast<AllocOp>(op)) {
        roots.push_back(alloc.getResult());
      } else if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
        if (!tensor.getResult().getType().hasStaticShape()) {
          return;
        }
      } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
        if (!getConstantIntValue(extract.getIndex()) ||
            !isa<qtensor::AllocOp, qtensor::ExtractOp>(
                extract.getTensor().getDefiningOp())) {
          return;
        }
        roots.push_back(extract.getResult());
      } else if (!isa<UnitaryOpInterface, MeasureOp, ResetOp>(op)) {
        if (op.getNumRegions() != 0 ||
            llvm::any_of(op.getOperandTypes(), isLinearQubitType) ||
            llvm::any_of(op.getResultTypes(), isLinearQubitType)) {
          return;
        }
        continue;
      }
      if (firstDisposal != nullptr) {
        return;
      }
    }

    // Every supported wire starts at an owned root and ends in a sink or
    // a final insertion. Save its resource slot before changing the wiring.
    SmallVector<OpOperand*> destinations;
    for (auto root : roots) {
      auto output = terminalQubit(root);
      // Unmeasured programs expose quantum outputs through layout metadata.
      // Eliding their SWAPs would require recording the output permutation.
      if (!output.getDefiningOp<MeasureOp>()) {
        return;
      }
      destinations.push_back(&*output.use_begin());
    }
    IRRewriter rewriter(&getContext());
    for (auto swap : swaps) {
      rewriter.replaceOp(swap, {swap.getQubit1In(), swap.getQubit0In()});
    }
    auto outputs = llvm::map_to_vector(roots, terminalQubit);

    // Move the pure insertion chains behind all quantum producers so each
    // permuted value dominates its original slot. Keep deallocations in place.
    for (auto insert : insertions) {
      rewriter.moveOpBefore(insert, firstDisposal);
    }
    for (auto [destination, value] : llvm::zip_equal(destinations, outputs)) {
      rewriter.modifyOpInPlace(destination->getOwner(),
                               [&] { destination->set(value); });
    }
  }
};

} // namespace
} // namespace mlir::qco
