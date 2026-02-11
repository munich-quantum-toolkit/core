/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @brief This pass performs quantum inter-procedural optimizations (IPO).
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <algorithm>
#include <array>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

using namespace mlir;

static bool areInsertExtractRelated(tensor::ExtractOp extract,
                                    tensor::InsertOp insert) {
  auto current = insert.getDest();
  while (current) {
    auto users = current.getUsers();
    for (const auto user : users) {
      if (user == extract) {
        return true;
      }
    }
    auto* def = current.getDefiningOp();
    if (!def) {
      return false;
    }
    if (auto insertOp = dyn_cast<tensor::InsertOp>(def)) {
      current = insertOp.getDest();
    } else {
      return false;
    }
  }
  return false;
}

static tensor::InsertOp findInsertForExtract(tensor::ExtractOp extract) {
  auto currentVal = extract.getResult();
  while (currentVal) {
    if (!currentVal.hasOneUse()) {
      // Multiple users, should not happen.
      return nullptr;
    }
    auto* user = *currentVal.getUsers().begin();
    if (auto insertOp = dyn_cast<tensor::InsertOp>(user)) {
      const auto indexValue = insertOp.getIndices()[0];
      if (auto constIndex =
              dyn_cast<arith::ConstantIndexOp>(indexValue.getDefiningOp())) {
        return insertOp;
      }
      return nullptr;
    }
    if (auto unitaryOp = dyn_cast<qco::UnitaryOpInterface>(user)) {
      currentVal = unitaryOp.getOutputForInput(currentVal);
      continue;
    }
    if (auto measureOp = dyn_cast<mlir::qco::MeasureOp>(user)) {
      currentVal = measureOp.getQubitOut();
      continue;
    }
    if (user->getNumResults() != 1) {
      // Multiple results, should not happen.
      return nullptr;
    }
    currentVal = user->getResult(0);
  }
  return nullptr;
}

static SmallVector<std::pair<int64_t, int64_t>>
canPromoteArgument(BlockArgument arg) {
  const auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
  if (!tensorType) {
    return {};
  }

  SmallVector<std::pair<int64_t, int64_t>> usedIndices;
  for (auto* user : arg.getUsers()) {
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(user)) {
      if (extractOp.getIndices().size() != 1) {
        return {};
      }
      auto insertOp = findInsertForExtract(extractOp);
      if (!insertOp) {
        return {};
      }
      if (!areInsertExtractRelated(extractOp, insertOp)) {
        return {};
      }
      auto extractIndexValue = extractOp.getIndices()[0];
      auto insertIndex = dyn_cast<arith::ConstantIndexOp>(
                             insertOp.getIndices()[0].getDefiningOp())
                             .value();
      if (auto constExtractIndex = dyn_cast<arith::ConstantIndexOp>(
              extractIndexValue.getDefiningOp())) {
        usedIndices.emplace_back(constExtractIndex.value(), insertIndex);
      } else {
        return {};
      }
    } else {
      if (!isa<tensor::InsertOp>(user)) {
        return {};
      }
    }
  }

  if (usedIndices.empty()) {
    return {};
  }
  return usedIndices;
}

void promoteArgument(BlockArgument arg,
                     SmallVector<std::pair<int64_t, int64_t>> indexMap,
                     SymbolTable& symTable) {
  // We know some pre-conditions for this method to be executed:
  //   - `arg` is a tensor type
  //   - All users of `arg` are either `tensor::ExtractOp` or `tensor::InsertOp`
  //   - All values extracted from `arg` are later on once again inserted into
  //   it
  //     - The insertion indices do not necessarily map to the extraction
  //     indices.
  //     - `indexMap` provides the mapping from extraction index to insertion
  //     index.
  //   - At least one entry of the vector is not used (and does not appear in
  //   the `indexMap`.
  // 1. Get Context and Function
  // Fix: BlockArgs belong to a Block, not an Op. Get the block's parent Op.
  Block* entryBlock = arg.getOwner();
  auto funcOp = dyn_cast<func::FuncOp>(entryBlock->getParentOp());
  if (!funcOp)
    return; // Should not happen if arg is a func argument

  OpBuilder builder(funcOp);
  MLIRContext* ctx = funcOp.getContext();
  const unsigned argIndex = arg.getArgNumber();
  const auto loc = arg.getLoc();

  // Determine the Scalar Type (QubitType)
  // Assuming the input tensor contains the QubitType
  auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
  const auto qubitType = tensorType.getElementType();

  // ====================================================
  // 2. Update Function Signature
  // ====================================================

  // A. Input Types
  SmallVector<Type> newArgTypes = llvm::to_vector(funcOp.getArgumentTypes());

  // Remove the old tensor argument
  newArgTypes.erase(newArgTypes.begin() + argIndex);

  // Insert N new scalar arguments
  for (size_t i = 0; i < indexMap.size(); ++i) {
    newArgTypes.insert(newArgTypes.begin() + argIndex + i, qubitType);
  }

  // B. Result Types
  // (We assume the function returns the modified tensor, which must also be
  // split)
  SmallVector<Type> newResultTypes = llvm::to_vector(funcOp.getResultTypes());

  // Find which result corresponds to our tensor.
  // For simplicity, we assume the function has 1 result which is this tensor.
  // In a complex case, you'd trace the return op operands.
  const int resultIndexToReplace = 0; // TODO
  if (!newResultTypes.empty() && newResultTypes[0] == tensorType) {
    newResultTypes.erase(newResultTypes.begin() + resultIndexToReplace);
    for (size_t i = 0; i < indexMap.size(); ++i) {
      newResultTypes.insert(newResultTypes.begin() + resultIndexToReplace + i,
                            qubitType);
    }
  }

  // Set the new function type
  auto newFuncType = FunctionType::get(ctx, newArgTypes, newResultTypes);
  funcOp.setFunctionType(newFuncType);

  // ====================================================
  // 3. Update Entry Block Arguments
  // ====================================================

  // Create the new scalar arguments in the block
  SmallVector<Value> newArgs;
  for (size_t i = 0; i < indexMap.size(); ++i) {
    // Insert after the original arg to keep indices stable for a moment
    BlockArgument newArg =
        entryBlock->insertArgument(argIndex + i + 1, qubitType, loc);
    newArgs.push_back(newArg);
  }

  // Map: Extraction Index -> New Block Argument
  DenseMap<int64_t, Value> extractToArgMap;
  for (size_t i = 0; i < indexMap.size(); ++i) {
    extractToArgMap[indexMap[i].first] = newArgs[i];
  }

  // ====================================================
  // 4. Rewrite Body (Extracts & Inserts)
  // ====================================================

  // We need to collect "Insert" values to update the ReturnOp later.
  // Map: Insertion Index -> Value to Return
  DenseMap<int64_t, Value> insertToRetMap;

  // We cannot iterate safely while modifying, so collect users first.
  // Note: We only look at direct users of the argument for Extracts,
  // but we might need to follow the chain for Inserts.
  llvm::SmallVector<Operation*> users(arg.getUsers());

  for (Operation* user : users) {
    // --- Handle Extracts ---
    if (auto extractOp = dyn_cast<tensor::ExtractOp>(user)) {
      // Check if indices are constant
      int64_t idx = -1;
      if (auto cst = getConstantIntValue(extractOp.getIndices().front())) {
        idx = *cst;
      }

      // If this index is in our map, replace the extract op
      if (extractToArgMap.count(idx)) {
        extractOp.replaceAllUsesWith(extractToArgMap[idx]);
        extractOp.erase();
      }
      continue;
    }

    // --- Handle Inserts (Chain collection) ---
    // If the argument is the 'dest' of an insert, it starts a chain.
    if (auto insertOp = dyn_cast<tensor::InsertOp>(user)) {
      if (insertOp.getDest() == arg) {
        // This is the start of the insertion chain.
        // We walk down the chain of inserts to find all scalar values.
        Operation* current = insertOp;
        while (auto currentInsert =
                   dyn_cast_or_null<tensor::InsertOp>(current)) {

          // Get the scalar value being inserted
          Value scalarVal = currentInsert.getScalar();

          // Get the index
          if (auto cst =
                  getConstantIntValue(currentInsert.getIndices().front())) {
            insertToRetMap[*cst] = scalarVal;
          }

          // Move to the next user of this insert's result
          Value result = currentInsert.getResult();
          if (result.hasOneUse()) {
            current = *result.getUsers().begin();
          } else {
            // If used by ReturnOp, we stop.
            if (!result.getUsers().empty() &&
                isa<func::ReturnOp>(*result.getUsers().begin()))
              break;
            current = nullptr;
          }

          // Mark the insert to be erased (it's no longer needed)
          // We can't erase immediately while walking, so we mark/defer or rely
          // on DCE. For now, we leave it; it will become dead code when we
          // update the ReturnOp.
        }
      }
    }
  }

  // ====================================================
  // 5. Update Return Op
  // ====================================================

  Operation* terminator = entryBlock->getTerminator();
  if (auto returnOp = dyn_cast<func::ReturnOp>(terminator)) {
    SmallVector<Value> newReturns = llvm::to_vector(returnOp.getOperands());

    // Remove the old tensor return
    newReturns.erase(newReturns.begin() + resultIndexToReplace);

    // Insert the new scalar returns based on the Insertion Map
    for (size_t i = 0; i < indexMap.size(); ++i) {
      int64_t requiredInsertIdx = indexMap[i].second;

      // We must have found a value for this index, or the logic is broken
      // (or we default to passing through the input arg if unchanged?)
      Value retVal = insertToRetMap[requiredInsertIdx];

      // Fallback: If no insert touched this index, it means the value is
      // unchanged. We pass the input argument directly to the output.
      if (!retVal) {
        retVal = extractToArgMap[indexMap[i].first];
      }

      newReturns.insert(newReturns.begin() + resultIndexToReplace + i, retVal);
    }
    returnOp->setOperands(newReturns);
  }

  // Clean up the original argument
  const auto eraseRecursively = [](auto&& self, Operation* op) -> void {
    for (auto user : op->getUsers()) {
      self(self, user);
    }
    op->erase();
  };
  for (auto user : entryBlock->getArgument(argIndex).getUsers()) {
    eraseRecursively(eraseRecursively, user);
  }
  entryBlock->eraseArgument(argIndex);

  // ====================================================
  // 6. Update Call Sites
  // ====================================================

  // We use the SymbolTable to find all calls to this function
  if (auto uses = symTable.getSymbolUses(funcOp, funcOp->getParentOp())) {
    for (auto use : *uses) {
      if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
        builder.setInsertionPoint(callOp);

        // A. Prepare New Operands (Caller side extraction)
        SmallVector<Value> newCallOperands =
            llvm::to_vector(callOp.getOperands());
        Value originalTensorInput = newCallOperands[argIndex];

        // Remove old tensor operand
        newCallOperands.erase(newCallOperands.begin() + argIndex);

        // Add new extracted operands
        SmallVector<Value> extractedScalars;
        for (size_t i = 0; i < indexMap.size(); ++i) {
          int64_t idx = indexMap[i].first;
          Value idxVal = builder.create<arith::ConstantIndexOp>(loc, idx);
          Value scalar = builder.create<tensor::ExtractOp>(
              loc, originalTensorInput, idxVal);

          extractedScalars.push_back(scalar);
          newCallOperands.insert(newCallOperands.begin() + argIndex + i,
                                 scalar);
        }

        // B. Create New Call
        auto newCall =
            builder.create<func::CallOp>(loc, funcOp, newCallOperands);

        // C. Reconstruct Tensor (Caller side insertion)
        // We take the existing tensor (originalTensorInput) or a fresh init?
        // Usually, in SROA, we are updating the tensor.
        Value reconstructedTensor = originalTensorInput;

        // The new call returns N scalars. We insert them back.
        for (size_t i = 0; i < indexMap.size(); ++i) {
          int64_t idx = indexMap[i].second;
          Value scalarResult = newCall.getResult(resultIndexToReplace + i);
          Value idxVal = builder.create<arith::ConstantIndexOp>(loc, idx);

          reconstructedTensor = builder.create<tensor::InsertOp>(
              loc, scalarResult, reconstructedTensor, idxVal);
        }

        // D. Replace users of the old call result
        callOp.getResult(resultIndexToReplace)
            .replaceAllUsesWith(reconstructedTensor);
        callOp.erase();
      }
    }
  }
}
} // namespace

namespace mlir::qco {

void runQuantumArgumentPromotion(ModuleOp module, SymbolTable& symbolTable) {
  std::vector<
      std::pair<BlockArgument, SmallVector<std::pair<int64_t, int64_t>>>>
      argsToPromote;

  module.walk([&](func::FuncOp func) {
    if (func.isPublic() || func.isDeclaration()) {
      return;
    }
    for (auto arg : func.getArguments()) {
      auto usedIndices = canPromoteArgument(arg);
      if (!usedIndices.empty()) {
        argsToPromote.emplace_back(arg, usedIndices);
      }
    }
  });

  for (auto& arg : argsToPromote) {
    promoteArgument(arg.first, arg.second, symbolTable);
  }
}

} // namespace mlir::qco
