/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <iterator>
#include <memory>

namespace mlir::qco {

/// Find the disposal of a straight-line qubit wire.
static Value terminalQubit(Value root) {
  WireIterator wire(root);
  while (wire != std::default_sentinel) {
    if (isa<SinkOp, qtensor::InsertOp>(*wire)) {
      return wire.qubit();
    }
    ++wire;
  }
  return {};
}

namespace {

struct ElideTerminalSwapsPass final
    : PassWrapper<ElideTerminalSwapsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElideTerminalSwapsPass)

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto function = mqt::getEntryPoint(moduleOp);
    if (!moduleOp->hasAttr(mqt::kSourceQubitCountAttr) ||
        moduleOp->hasAttr("mqt.layout") || !function ||
        !llvm::hasSingleElement(function.getBody()) ||
        llvm::any_of(function.getArgumentTypes(), isLinearQubitType)) {
      return;
    }
    auto& block = function.getBody().front();
    if (block.getOps<SWAPOp>().empty()) {
      return;
    }

    // shortcut: only flat owned wires and terminal tensor packing; extend the
    // ownership analysis separately when supporting tensor reuse or SCF.
    SmallVector<Value> roots;
    SmallVector<int64_t> positions;
    DenseMap<Value, int64_t> tensorOffsets;
    DenseMap<Operation*, int64_t> insertionPositions;
    SmallVector<qtensor::InsertOp> insertions;
    SmallVector<SWAPOp> swaps;
    Operation* firstDisposal = nullptr;
    int64_t width = 0;
    bool measured = false;
    for (Operation& op : block) {
      if (isa<SinkOp, qtensor::DeallocOp>(op)) {
        if (firstDisposal == nullptr) {
          firstDisposal = &op;
        }
        continue;
      }
      if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
        const auto index = getConstantIntValue(insert.getIndex());
        auto offset = tensorOffsets.find(insert.getDest());
        if (!index || offset == tensorOffsets.end() ||
            (firstDisposal != nullptr &&
             !insert.getIndex().getDefiningOp()->isBeforeInBlock(
                 firstDisposal))) {
          return;
        }
        insertionPositions[insert] = offset->second + *index;
        tensorOffsets[insert.getResult()] = offset->second;
        insertions.push_back(insert);
        continue;
      }
      if (auto alloc = dyn_cast<AllocOp>(op)) {
        roots.push_back(alloc.getResult());
        positions.push_back(width++);
      } else if (auto tensor = dyn_cast<qtensor::AllocOp>(op)) {
        auto type = tensor.getResult().getType();
        if (!type.hasStaticShape()) {
          return;
        }
        tensorOffsets[tensor.getResult()] = width;
        width += type.getNumElements();
      } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
        const auto index = getConstantIntValue(extract.getIndex());
        auto offset = tensorOffsets.find(extract.getTensor());
        if (!index || offset == tensorOffsets.end() ||
            !isa<qtensor::AllocOp, qtensor::ExtractOp>(
                extract.getTensor().getDefiningOp())) {
          return;
        }
        const auto position = offset->second + *index;
        tensorOffsets[extract.getOutTensor()] = offset->second;
        roots.push_back(extract.getResult());
        positions.push_back(position);
      } else if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
        if (measured) {
          return;
        }
        if (auto swap = dyn_cast<SWAPOp>(op)) {
          swaps.push_back(swap);
        } else if (unitary.getNumQubits() > 1) {
          swaps.clear();
        }
      } else if (isa<MeasureOp>(op)) {
        measured = true;
      } else if (isa<ResetOp>(op)) {
        swaps.clear();
      } else {
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
    if (swaps.empty()) {
      return;
    }

    SmallVector<OpOperand*> destinations;
    for (auto [root, position] : llvm::zip_equal(roots, positions)) {
      auto output = terminalQubit(root);
      if (!output) {
        return;
      }
      auto* destination = &*output.use_begin();
      if (isa<qtensor::InsertOp>(destination->getOwner()) &&
          insertionPositions.lookup(destination->getOwner()) != position) {
        return;
      }
      destinations.push_back(destination);
    }
    IRRewriter rewriter(&getContext());
    for (auto swap : swaps) {
      rewriter.replaceOp(swap, {swap.getQubit1In(), swap.getQubit0In()});
    }
    auto outputs = llvm::map_to_vector(roots, terminalQubit);
    DenseMap<Value, int64_t> outputPositions;
    for (auto [output, position] : llvm::zip_equal(outputs, positions)) {
      outputPositions[output] = position;
    }
    mqt::QubitLayout layout;
    layout.inputCount = width;
    for (int64_t position = 0; position < width; ++position) {
      layout.initial.push_back(position);
    }
    layout.routing = layout.initial;
    for (auto [destination, position] :
         llvm::zip_equal(destinations, positions)) {
      (*layout.routing)[position] = outputPositions.lookup(destination->get());
    }

    // Keep packing behind quantum producers and retain each resource slot.
    for (auto insert : insertions) {
      rewriter.moveOpBefore(insert, firstDisposal);
    }
    for (auto [destination, value] : llvm::zip_equal(destinations, outputs)) {
      destination->set(value);
    }
    if (*layout.routing != layout.initial) {
      moduleOp->setAttr("mqt.layout", layout.toAttr(&getContext()));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createElideTerminalSwapsPass() {
  return std::make_unique<ElideTerminalSwapsPass>();
}

} // namespace mlir::qco
