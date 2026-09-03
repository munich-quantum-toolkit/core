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
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/RegionUtils.h>

#include <cstddef>

namespace mlir::qc::detail {

static bool containsQubit(Type type) {
  return type.walk([](QubitType) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

static bool carriesQubit(Operation* operation) {
  return llvm::any_of(operation->getOperandTypes(), containsQubit) ||
         llvm::any_of(operation->getResultTypes(), containsQubit);
}

static bool isClassical(Operation* operation) {
  if (!isPure(operation)) {
    return false;
  }
  return !operation
              ->walk([](Operation* nested) {
                return isa<UnitaryOpInterface>(nested) || carriesQubit(nested)
                           ? WalkResult::interrupt()
                           : WalkResult::advance();
              })
              .wasInterrupted();
}

LogicalResult verifyModifierBody(Operation* modifierOp, Block& body) {
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
        return !isa<UnitaryOpInterface>(operation) && !isClassical(&operation);
      });
  if (hasNonUnitaryOperation) {
    return modifierOp->emitOpError(
        "body may contain only unitary operations and pure, speculatable "
        "classical support operations");
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
