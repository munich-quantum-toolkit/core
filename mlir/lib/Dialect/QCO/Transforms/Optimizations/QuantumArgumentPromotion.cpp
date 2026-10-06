/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTUMARGUMENTPROMOTION
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/// A tensor slot that crosses the call boundary as a scalar qubit.
///
/// The qubit is taken out of the tensor at `extractIndex` and put back at
/// `insertIndex`; the two indices need not agree.
struct PromotedSlot {
  /// The extraction taking the qubit out of the tensor.
  qtensor::ExtractOp extract;
  /// The insertion putting the qubit back.
  qtensor::InsertOp insert;
  /// The index the qubit is taken from.
  int64_t extractIndex;
  /// The index the qubit is put back at.
  int64_t insertIndex;
};

/// How a tensor argument is replaced by scalar qubits.
struct Promotion {
  /// The slots that cross the call boundary.
  SmallVector<PromotedSlot> slots;
  /// The function result that hands the tensor back to the caller.
  unsigned resultIndex;
};

} // namespace

/// Find where an extracted qubit is put back into a tensor.
///
/// Follows the qubit produced by @p extract forward through gate-like
/// operations until it is inserted back into a tensor at a compile-time
/// constant index.
///
/// @param extract The extraction whose qubit is followed.
/// @return The matching insertion, or a null op if the qubit never comes back.
static qtensor::InsertOp findInsertForExtract(qtensor::ExtractOp extract) {
  Value currentValue = extract.getResult();
  while (currentValue) {
    if (!currentValue.hasOneUse()) {
      // Qubits are linear, so this should not happen.
      return nullptr;
    }
    auto* user = *currentValue.getUsers().begin();

    if (auto insertOp = dyn_cast<qtensor::InsertOp>(user)) {
      if (insertOp.getScalar() != currentValue ||
          !getConstantIntValue(insertOp.getIndex())) {
        return nullptr;
      }
      return insertOp;
    }
    if (auto unitaryOp = dyn_cast<UnitaryOpInterface>(user)) {
      currentValue = unitaryOp.getOutputForInput(currentValue);
      continue;
    }
    if (auto measureOp = dyn_cast<MeasureOp>(user)) {
      currentValue = measureOp.getQubitOut();
      continue;
    }
    if (auto resetOp = dyn_cast<ResetOp>(user)) {
      currentValue = resetOp.getQubitOut();
      continue;
    }
    // Anything else is not known to thread the qubit. Guessing that a single
    // result carries it on would let a slot be promoted that no longer holds
    // the extracted qubit, so give up instead.
    return nullptr;
  }
  return nullptr;
}

/// Determine whether a tensor argument can be replaced by scalar qubits.
///
/// This requires that every operation on the argument's tensor chain is a
/// `qtensor.extract` or `qtensor.insert` at a constant index, that every
/// extracted qubit is inserted back into the same chain, and that the chain
/// ends in a function result.
///
/// @param arg The tensor argument to analyze.
/// @return The promotion, or `std::nullopt` if the argument cannot be promoted.
static std::optional<Promotion> canPromoteArgument(BlockArgument arg) {
  const auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
  if (!tensorType || !isa<QubitType>(tensorType.getElementType())) {
    return {};
  }

  auto funcOp = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (!funcOp) {
    return {};
  }

  // Promotion rewrites the signature and every call site, so every reference to
  // the function has to be a direct call. A symbol captured anywhere else would
  // be left pointing at the old signature, so bail before anything is changed.
  const auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
  if (!uses) {
    return {};
  }
  for (const auto use : *uses) {
    auto callOp = dyn_cast<func::CallOp>(use.getUser());
    if (!callOp || callOp.getCallee() != funcOp.getName()) {
      return {};
    }
  }

  // Walk the chain of the threaded tensor and collect the accesses on it.
  SmallVector<qtensor::ExtractOp> extracts;
  DenseSet<Operation*> insertsOnChain;
  // Where each slot is read and written along the chain. The rewrite reorders
  // the accesses, so their relative order has to be checked below.
  DenseMap<int64_t, unsigned> extractPositions;
  DenseMap<int64_t, unsigned> insertPositions;
  unsigned position = 0;
  Value currentTensor = arg;
  std::optional<unsigned> resultIndex;

  while (currentTensor) {
    if (!currentTensor.hasOneUse()) {
      // Qubit tensors are linear, so this should not happen.
      return {};
    }
    auto* user = *currentTensor.getUsers().begin();

    if (auto extractOp = dyn_cast<qtensor::ExtractOp>(user)) {
      const auto index = getConstantIntValue(extractOp.getIndex());
      if (!index) {
        return {};
      }
      // Reading one slot twice would take a qubit out of a slot the first read
      // already emptied, because the call site extracts from a single tensor.
      if (!extractPositions.try_emplace(*index, position++).second) {
        return {};
      }
      extracts.emplace_back(extractOp);
      currentTensor = extractOp.getOutTensor();
      continue;
    }
    if (auto insertOp = dyn_cast<qtensor::InsertOp>(user)) {
      const auto index = getConstantIntValue(insertOp.getIndex());
      if (insertOp.getDest() != currentTensor || !index) {
        return {};
      }
      // Two writes to one slot only differ in their order, which the rewrite
      // does not preserve.
      if (!insertPositions.try_emplace(*index, position++).second) {
        return {};
      }
      insertsOnChain.insert(insertOp);
      currentTensor = insertOp.getResult();
      continue;
    }
    if (isa<func::ReturnOp>(user)) {
      resultIndex = currentTensor.getUses().begin()->getOperandNumber();
      break;
    }
    // Anything else (a call, a dealloc, ...) keeps the tensor alive.
    return {};
  }

  if (!resultIndex || extracts.empty()) {
    return {};
  }

  // The rewrite takes every promoted qubit out of the caller's tensor before
  // the call and puts them all back afterwards. A slot that the callee writes
  // before it reads it would then be read from the caller's original tensor
  // instead of from the value written to it, so reject that ordering.
  for (const auto& [index, insertPosition] : insertPositions) {
    const auto extractPosition = extractPositions.find(index);
    if (extractPosition != extractPositions.end() &&
        extractPosition->second > insertPosition) {
      return {};
    }
  }

  // Every extracted qubit has to find its way back into the same chain.
  SmallVector<PromotedSlot> slots;
  for (auto extractOp : extracts) {
    auto insertOp = findInsertForExtract(extractOp);
    if (!insertOp || !insertsOnChain.contains(insertOp)) {
      return {};
    }
    slots.emplace_back(PromotedSlot{
        .extract = extractOp,
        .insert = insertOp,
        .extractIndex = *getConstantIntValue(extractOp.getIndex()),
        .insertIndex = *getConstantIntValue(insertOp.getIndex()),
    });
  }

  // Every insertion has to belong to one of the promoted slots. One that does
  // not is left behind by the rewrite and keeps using the tensor argument that
  // is erased right after, which trips MLIR's `use_empty()` assertion.
  DenseSet<Operation*> matchedInserts;
  for (const auto& slot : slots) {
    matchedInserts.insert(slot.insert);
  }
  if (matchedInserts.size() != insertsOnChain.size()) {
    return {};
  }

  return Promotion{.slots = std::move(slots), .resultIndex = *resultIndex};
}

/// Replace a tensor argument by one scalar qubit argument per slot.
///
/// Rewrites the function signature and body, and updates every call site so
/// that the promoted elements are taken out of the tensor before the call and
/// put back afterwards.
///
/// @param arg The tensor argument to promote.
/// @param promotion The promotion, as returned by `canPromoteArgument`.
static void promoteArgument(BlockArgument arg, Promotion& promotion) {
  Block* entryBlock = arg.getOwner();
  auto funcOp = cast<func::FuncOp>(entryBlock->getParentOp());

  OpBuilder builder(funcOp);
  MLIRContext* ctx = funcOp.getContext();
  const unsigned argIndex = arg.getArgNumber();
  const auto loc = arg.getLoc();
  const auto tensorType = cast<RankedTensorType>(arg.getType());
  const auto qubitType = tensorType.getElementType();
  MutableArrayRef<PromotedSlot> slots = promotion.slots;
  const auto numSlots = slots.size();
  const auto resultIndex = promotion.resultIndex;

  // ====================================================
  // 1. Update the function signature
  // ====================================================

  SmallVector<Type> newArgTypes = llvm::to_vector(funcOp.getArgumentTypes());
  newArgTypes.erase(std::next(newArgTypes.begin(), argIndex));
  for (size_t i = 0; i < numSlots; ++i) {
    newArgTypes.insert(
        std::next(newArgTypes.begin(), static_cast<ptrdiff_t>(argIndex + i)),
        qubitType);
  }

  // The promoted qubits take the place of the result that hands the tensor
  // back, just as their arguments take the place of the tensor argument. A
  // callee that returns its quantum arguments in argument order still does.
  SmallVector<Type> newResultTypes = llvm::to_vector(funcOp.getResultTypes());
  newResultTypes.erase(std::next(newResultTypes.begin(), resultIndex));
  newResultTypes.insert(std::next(newResultTypes.begin(), resultIndex),
                        numSlots, qubitType);

  funcOp.setFunctionType(FunctionType::get(ctx, newArgTypes, newResultTypes));

  // ====================================================
  // 2. Add the scalar block arguments
  // ====================================================

  SmallVector<Value> newArgs;
  newArgs.reserve(numSlots);
  for (size_t i = 0; i < numSlots; ++i) {
    // Insert behind the original argument to keep its index stable for now.
    newArgs.emplace_back(
        entryBlock->insertArgument(argIndex + i + 1, qubitType, loc));
  }

  // ====================================================
  // 3. Drop the tensor accesses from the body
  // ====================================================

  // The qubit reaching the insert is what the function returns for that slot.
  SmallVector<Value> returnedQubits;
  returnedQubits.reserve(numSlots);
  for (auto&& [i, slot] : llvm::enumerate(slots)) {
    // A pass-through slot hands the new argument straight back.
    returnedQubits.emplace_back(slot.insert.getScalar() ==
                                        slot.extract.getResult()
                                    ? newArgs[i]
                                    : slot.insert.getScalar());
  }

  for (const auto& [i, constSlot] : llvm::enumerate(slots)) {
    auto slot = constSlot;
    // Feed the new argument in where the qubit used to be extracted, and let
    // the tensor bypass both accesses. All of them collapse onto `arg`.
    slot.extract.getResult().replaceAllUsesWith(newArgs[i]);
    slot.extract.getOutTensor().replaceAllUsesWith(slot.extract.getTensor());
    slot.insert.getResult().replaceAllUsesWith(slot.insert.getDest());
  }
  for (auto slot : slots) {
    slot.insert.erase();
    slot.extract.erase();
  }

  // ====================================================
  // 4. Update the terminator
  // ====================================================

  auto returnOp = cast<func::ReturnOp>(entryBlock->getTerminator());
  SmallVector<Value> newReturns = llvm::to_vector(returnOp.getOperands());
  newReturns.erase(std::next(newReturns.begin(), resultIndex));
  newReturns.insert(std::next(newReturns.begin(), resultIndex),
                    returnedQubits.begin(), returnedQubits.end());
  returnOp->setOperands(newReturns);

  entryBlock->eraseArgument(argIndex);

  // ====================================================
  // 5. Update the call sites
  // ====================================================

  auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
  if (!uses) {
    return;
  }
  for (auto use : *uses) {
    auto callOp = dyn_cast<func::CallOp>(use.getUser());
    if (!callOp) {
      continue;
    }
    builder.setInsertionPoint(callOp);
    const auto callLoc = callOp.getLoc();

    SmallVector<Value> newOperands = llvm::to_vector(callOp.getOperands());
    Value currentTensor = newOperands[argIndex];
    newOperands.erase(std::next(newOperands.begin(), argIndex));

    // Take the promoted qubits out of the tensor before the call, ...
    for (size_t i = 0; i < numSlots; ++i) {
      Value index = arith::ConstantIndexOp::create(builder, callLoc,
                                                   slots[i].extractIndex);
      auto extractOp =
          qtensor::ExtractOp::create(builder, callLoc, currentTensor, index);
      currentTensor = extractOp.getOutTensor();
      newOperands.insert(
          std::next(newOperands.begin(), static_cast<ptrdiff_t>(argIndex + i)),
          extractOp.getResult());
    }

    auto newCall = func::CallOp::create(builder, callLoc, funcOp, newOperands);

    // ... and put them back afterwards.
    for (size_t i = 0; i < numSlots; ++i) {
      Value index = arith::ConstantIndexOp::create(builder, callLoc,
                                                   slots[i].insertIndex);
      currentTensor =
          qtensor::InsertOp::create(
              builder, callLoc,
              newCall.getResult(resultIndex + static_cast<unsigned>(i)),
              currentTensor, index)
              .getResult();
    }

    for (auto [r, result] : llvm::enumerate(callOp.getResults())) {
      if (r == resultIndex) {
        result.replaceAllUsesWith(currentTensor);
      } else if (r < resultIndex) {
        result.replaceAllUsesWith(newCall.getResult(r));
      } else {
        result.replaceAllUsesWith(newCall.getResult(r + numSlots - 1));
      }
    }
    callOp.erase();
  }
}

/// Promote tensor arguments to scalar qubits across the whole module.
///
/// Externally visible functions and declarations are skipped because their
/// signature cannot be changed. At most one argument per function is promoted
/// per run, because promoting shifts the indices of the remaining arguments.
///
/// @param moduleOp The module to transform.
namespace {
/// Replaces qubit-tensor arguments by the scalar qubits a callee uses.
struct QuantumArgumentPromotion final
    : impl::QuantumArgumentPromotionBase<QuantumArgumentPromotion> {
  using impl::QuantumArgumentPromotionBase<
      QuantumArgumentPromotion>::QuantumArgumentPromotionBase;

protected:
  void runOnOperation() override {
    SmallVector<BlockArgument> argsToPromote;

    getOperation().walk([&](func::FuncOp func) {
      if (func.isPublic() || func.isDeclaration()) {
        return;
      }
      for (auto arg : func.getArguments()) {
        if (canPromoteArgument(arg)) {
          argsToPromote.emplace_back(arg);
          // Promoting shifts the indices of the remaining arguments, so handle
          // at most one argument per function. Any further tensor argument is
          // picked up the next time the pass runs.
          break;
        }
      }
    });

    for (auto arg : argsToPromote) {
      // Promoting one function rewrites the call sites in others, which can
      // invalidate a chain analyzed during the walk. Re-derive it right before
      // use rather than recording it.
      auto promotion = canPromoteArgument(arg);
      if (!promotion) {
        continue;
      }
      promoteArgument(arg, *promotion);
    }
  }
};
} // namespace

} // namespace mlir::qco
