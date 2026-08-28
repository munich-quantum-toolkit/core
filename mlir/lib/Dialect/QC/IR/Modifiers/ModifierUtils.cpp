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
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
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

namespace mlir::qc::detail {

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
  auto targets = cast<UnitaryOpInterface>(modifierOp).getTargets();
  if (body.getNumArguments() != targets.size()) {
    return modifierOp->emitOpError(
        "number of block arguments must match the number of targets");
  }
  SmallPtrSet<Value, 4> uniqueTargets;
  for (auto [index, argument, target] :
       llvm::enumerate(body.getArguments(), targets)) {
    if (argument.getType() != target.getType()) {
      return modifierOp->emitOpError("block argument type at index ")
             << index << " does not match target type";
    }
    if (!uniqueTargets.insert(target).second) {
      return modifierOp->emitOpError("duplicate target qubit found");
    }
  }

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
  for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
    if (!arg.use_empty()) {
      used.push_back(index);
    }
  }
  return used;
}

LogicalResult
dropUnusedQubits(Operation* modifierOp, Block& body, ValueRange qubits,
                 function_ref<void(ValueRange, ArrayRef<size_t>)> rebuild,
                 RewriterBase& rewriter) {
  const auto used = getUsedQubitIndices(body);
  if (used.size() == qubits.size()) {
    return failure();
  }

  const auto narrowedQubits = llvm::map_to_vector(
      used, [&](const size_t index) { return qubits[index]; });
  rebuild(narrowedQubits, used);
  rewriter.eraseOp(modifierOp);
  return success();
}

void inlineNarrowedBody(Block& body, ValueRange qubits, ArrayRef<size_t> used,
                        ValueRange args, RewriterBase& rewriter) {
  SmallVector<Value> replacements(qubits);
  for (auto [index, arg] : llvm::zip_equal(used, args)) {
    replacements[index] = arg;
  }
  mqt::inlineBodyReturningYields(body, replacements, rewriter);
}

} // namespace mlir::qc::detail
