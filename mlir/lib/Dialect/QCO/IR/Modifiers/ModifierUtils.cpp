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

#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <cstddef>

namespace mlir::qco::detail {

// Follow unitary ties only after nested operations have been verified.
static bool hasPositionalBodyYields(Block& body) {
  // A valid modifier cannot permute fewer than two wires.
  if (body.getNumArguments() < 2) {
    return true;
  }

  for (auto [argument, yielded] : llvm::zip_equal(
           body.getArguments(), body.getTerminator()->getOperands())) {
    Value origin = yielded;
    while (origin != argument) {
      auto unitary = origin.getDefiningOp<UnitaryOpInterface>();
      if (!unitary) {
        return false;
      }
      origin = unitary.getInputForOutput(origin);
    }
  }
  return true;
}

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
  auto unitary = cast<UnitaryOpInterface>(modifierOp);
  if (!llvm::equal(body.getArgumentTypes(),
                   unitary.getInputTargets().getTypes())) {
    return modifierOp->emitOpError("body argument types must match targets");
  }
  if (!llvm::equal(body.getTerminator()->getOperandTypes(),
                   body.getArgumentTypes())) {
    return modifierOp->emitOpError("yield types must match body arguments");
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

  const auto hasNonUnitaryOperation =
      llvm::any_of(body.without_terminator(), [](Operation& operation) {
        if (isa<UnitaryOpInterface>(operation)) {
          return false;
        }
        const auto isQubit = [](Type type) { return isa<QubitType>(type); };
        return operation.getNumRegions() != 0 ||
               !isMemoryEffectFree(&operation) ||
               llvm::any_of(operation.getOperandTypes(), isQubit) ||
               llvm::any_of(operation.getResultTypes(), isQubit);
      });
  if (hasNonUnitaryOperation) {
    return modifierOp->emitOpError("body must contain only unitary operations "
                                   "and memory-effect-free classical "
                                   "operations without regions");
  }

  if (!hasPositionalBodyYields(body)) {
    return modifierOp->emitOpError(
        "yielded qubits must continue body arguments positionally");
  }

  SmallPtrSet<Value, 4> uniqueQubits;
  for (auto qubit : unitary.getInputQubits()) {
    if (!uniqueQubits.insert(qubit).second) {
      return modifierOp->emitOpError("duplicate qubit found");
    }
  }
  return success();
}

SmallVector<size_t> getUsedQubitIndices(Block& body) {
  SmallVector<size_t> used;
  for (auto [index, arg, yielded] : llvm::enumerate(
           body.getArguments(), body.getTerminator()->getOperands())) {
    // A qubit that the body only yields back is not acted upon.
    if (yielded != arg) {
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
