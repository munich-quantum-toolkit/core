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

#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

#include <cstddef>
#include <utility>

namespace mlir::qco::detail {

bool isModifierMatrixNestingSupported(Operation* modifierOp) {
  constexpr size_t maxModifierNesting = 64;
  SmallVector<std::pair<Operation*, size_t>> worklist{{modifierOp, 0}};
  while (!worklist.empty()) {
    auto [operation, parentDepth] = worklist.pop_back_val();
    const size_t depth =
        parentDepth + static_cast<size_t>(isa<CtrlOp, InvOp, PowOp>(operation));
    if (depth > maxModifierNesting) {
      return false;
    }
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.emplace_back(&nested, depth);
        }
      }
    }
  }
  return true;
}

[[nodiscard]] static bool containsQubit(Type type) {
  if (isa<QubitType>(type)) {
    return true;
  }
  const auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && isa<QubitType>(shapedType.getElementType());
}

[[nodiscard]] static bool
isForbiddenModifierBodyOperation(Operation* operation) {
  const auto carriesQubit =
      llvm::any_of(operation->getOperandTypes(), containsQubit) ||
      llvm::any_of(operation->getResultTypes(), containsQubit);
  if (isa<UnitaryOpInterface>(operation) ||
      operation->hasTrait<OpTrait::IsTerminator>() ||
      operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    return false;
  }
  if (isa<CallOpInterface>(operation)) {
    return carriesQubit;
  }
  if (!isMemoryEffectFree(operation)) {
    return true;
  }
  return carriesQubit;
}

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(modifierOp->getRegions(), captures);
  if (llvm::any_of(captures, [](Value value) {
        return containsQubit(value.getType());
      })) {
    return modifierOp->emitOpError(
        "body must not capture qubits from above; use only its aliased block "
        "arguments");
  }

  SmallVector<Operation*> worklist;
  for (Operation& operation : body) {
    worklist.push_back(&operation);
  }
  while (!worklist.empty()) {
    Operation* operation = worklist.pop_back_val();
    if (isForbiddenModifierBodyOperation(operation)) {
      return modifierOp->emitOpError(
          "body must not contain non-unitary operations or access registers");
    }
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.push_back(&nested);
        }
      }
    }
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
