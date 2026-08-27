/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ModifierUtils.h"

#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/RegionUtils.h>

#include <cstddef>

namespace mlir::qco::detail {

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
  const auto hasNonUnitaryOperation =
      body.walk([](Operation* operation) {
            return isa<cbit::AllocOp, cbit::LoadOp, cbit::StoreOp, AllocOp,
                       SinkOp, StaticOp, MeasureOp, ResetOp, qtensor::ExtractOp,
                       qtensor::InsertOp>(operation)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted();
  if (hasNonUnitaryOperation) {
    return modifierOp->emitOpError(
        "body must not contain non-unitary operations or access registers");
  }

  SetVector<Value> captures;
  getUsedValuesDefinedAbove(modifierOp->getRegions(), captures);
  if (llvm::any_of(captures, [](Value value) {
        return isa<QubitType>(value.getType());
      })) {
    return modifierOp->emitOpError(
        "body must not capture qubits from above; use only its aliased block "
        "arguments");
  }

  return success();
}

SmallVector<size_t> getUsedQubitIndices(Block& body) {
  SmallVector<size_t> used;
  for (auto [index, arg, yielded] : llvm::enumerate(
           body.getArguments(), body.getTerminator()->getOperands())) {
    // A qubit that the body only yields back is not acted upon.
    if (!arg.hasOneUse() || yielded != arg) {
      used.push_back(index);
    }
  }
  return used;
}

SmallVector<Value> restoreUnusedQubits(ValueRange inputs, ArrayRef<size_t> used,
                                       ValueRange narrowedResults) {
  SmallVector<Value> results(inputs);
  for (auto [index, result] : llvm::zip_equal(used, narrowedResults)) {
    results[index] = result;
  }
  return results;
}

LogicalResult
dropUnusedQubits(Operation* modifierOp, Block& body, ValueRange qubits,
                 function_ref<Operation*(ValueRange, ArrayRef<size_t>)> rebuild,
                 RewriterBase& rewriter) {
  const auto used = getUsedQubitIndices(body);
  if (used.size() == qubits.size()) {
    return failure();
  }

  const auto narrowedQubits = llvm::map_to_vector(
      used, [&](const size_t index) { return qubits[index]; });
  auto* narrowedModifier = rebuild(narrowedQubits, used);
  rewriter.replaceOp(
      modifierOp,
      restoreUnusedQubits(qubits, used, narrowedModifier->getResults()));
  return success();
}

SmallVector<Value> inlineNarrowedBody(Block& body, ValueRange qubits,
                                      ArrayRef<size_t> used, ValueRange args,
                                      RewriterBase& rewriter) {
  SmallVector<Value> replacements(qubits);
  for (auto [index, arg] : llvm::zip_equal(used, args)) {
    replacements[index] = arg;
  }

  const auto yielded =
      mqt::inlineBodyReturningYields(body, replacements, rewriter);
  return llvm::map_to_vector(
      used, [&](const size_t index) { return yielded[index]; });
}

} // namespace mlir::qco::detail
