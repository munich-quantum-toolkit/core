/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/IRVerification.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <unordered_map>

namespace {
using namespace mlir;

/// Compute a structural hash for an operation (excluding SSA value identities).
/// This hash is based on operation name, types, and attributes only.
struct OperationStructuralHash {
  size_t operator()(Operation* op) const {
    size_t hash = llvm::hash_value(op->getName().getStringRef());

    // Hash result types
    for (auto type : op->getResultTypes()) {
      hash = llvm::hash_combine(hash, type.getAsOpaquePointer());
    }

    // Hash operand types (not values)
    for (auto operand : op->getOperands()) {
      hash = llvm::hash_combine(hash, operand.getType().getAsOpaquePointer());
    }

    // Hash attributes
    // for (const auto& attr : op->getAttrDictionary()) {
    //   hash = llvm::hash_combine(hash, attr.getName().str());
    //   hash = llvm::hash_combine(hash, attr.getValue().getAsOpaquePointer());
    // }

    return hash;
  }
};

/// Check if two operations are structurally equivalent (excluding SSA value
/// identities).
struct OperationStructuralEquality {
  bool operator()(Operation* lhs, Operation* rhs) const {
    // Check operation name
    if (lhs->getName() != rhs->getName()) {
      return false;
    }

    // Check result types
    if (lhs->getResultTypes() != rhs->getResultTypes()) {
      return false;
    }

    // Check operand types (not values)
    auto lhsOperandTypes = lhs->getOperandTypes();
    auto rhsOperandTypes = rhs->getOperandTypes();
    return llvm::equal(lhsOperandTypes, rhsOperandTypes);

    // Note: Attributes are intentionally not checked here to allow relaxed
    // comparison. Attributes like function names, parameter names, etc. may
    // differ while operations are still structurally equivalent.
  }
};

/// Map to track value equivalence between two modules.
using ValueEquivalenceMap = llvm::DenseMap<mlir::Value, mlir::Value>;

/// Compare two operations for structural equivalence.
/// Updates valueMap to track corresponding SSA values.
bool areOperationsEquivalent(Operation* lhs, Operation* rhs,
                             ValueEquivalenceMap& valueMap) {
  // Check operation name
  if (lhs->getName() != rhs->getName()) {
    return false;
  }

  // Check arith::ConstantOp
  if (auto lhsConst = llvm::dyn_cast<arith::ConstantOp>(lhs)) {
    auto rhsConst = llvm::dyn_cast<arith::ConstantOp>(rhs);
    if (!rhsConst) {
      return false;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-slicing)
    if (lhsConst.getValue() != rhsConst.getValue()) {
      return false;
    }
  }

  // Check LLVM::ConstantOp
  if (auto lhsConst = llvm::dyn_cast<LLVM::ConstantOp>(lhs)) {
    auto rhsConst = llvm::dyn_cast<LLVM::ConstantOp>(rhs);
    if (!rhsConst) {
      return false;
    }
    if (lhsConst.getValue() != rhsConst.getValue()) {
      return false;
    }
  }

  // Check LLVM::CallOp
  if (auto lhsCall = llvm::dyn_cast<LLVM::CallOp>(lhs)) {
    auto rhsCall = llvm::dyn_cast<LLVM::CallOp>(rhs);
    if (!rhsCall) {
      return false;
    }
    if (lhsCall.getCallee() != rhsCall.getCallee()) {
      return false;
    }
  }

  // Check number of operands and results
  if (lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      lhs->getNumRegions() != rhs->getNumRegions()) {
    return false;
  }

  // Note: Attributes are intentionally not checked to allow relaxed comparison

  // Check result types
  if (lhs->getResultTypes() != rhs->getResultTypes()) {
    return false;
  }

  // Check operands according to value mapping
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (auto it = valueMap.find(lhsOperand); it != valueMap.end()) {
      // Value already mapped, must match
      if (it->second != rhsOperand) {
        return false;
      }
    } else {
      // Establish new mapping
      valueMap[lhsOperand] = rhsOperand;
    }
  }

  // Update value mapping for results
  for (auto [lhsResult, rhsResult] :
       llvm::zip(lhs->getResults(), rhs->getResults())) {
    valueMap[lhsResult] = rhsResult;
  }

  return true;
}

/// Forward declaration for mutual recursion.
bool areBlocksEquivalent(Block& lhs, Block& rhs, ValueEquivalenceMap& valueMap);

/// Compare two regions for structural equivalence.
bool areRegionsEquivalent(Region& lhs, Region& rhs,
                          ValueEquivalenceMap& valueMap) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  for (auto [lhsBlock, rhsBlock] : llvm::zip(lhs, rhs)) {
    if (!areBlocksEquivalent(lhsBlock, rhsBlock, valueMap)) {
      return false;
    }
  }

  return true;
}

/// Check if an operation has memory effects or control flow side effects
/// that would prevent reordering.
bool hasOrderingConstraints(Operation* op) {
  // Terminators must maintain their position
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return true;
  }

  // Symbol-defining operations (like function declarations) can be reordered
  if (op->hasTrait<OpTrait::SymbolTable>() ||
      llvm::isa<LLVM::LLVMFuncOp, func::FuncOp>(op)) {
    return false;
  }

  // Check for memory effects that enforce ordering
  if (auto memInterface = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
    llvm::SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);

    bool hasNonAllocFreeEffects = false;
    for (const auto& effect : effects) {
      // Allow operations with no effects or pure allocation/free effects
      if (!llvm::isa<MemoryEffects::Allocate, MemoryEffects::Free>(
              effect.getEffect())) {
        hasNonAllocFreeEffects = true;
        break;
      }
    }

    if (hasNonAllocFreeEffects) {
      return true;
    }
  }

  return false;
}

/// Build a dependence graph for operations.
/// Returns a map from each operation to the set of operations it depends on.
llvm::DenseMap<Operation*, llvm::DenseSet<Operation*>>
buildDependenceGraph(ArrayRef<Operation*> ops) {
  llvm::DenseMap<Operation*, llvm::DenseSet<Operation*>> dependsOn;
  llvm::DenseMap<Value, Operation*> valueProducers;

  // Build value-to-producer map and dependence relationships
  for (Operation* op : ops) {
    dependsOn[op] = llvm::DenseSet<Operation*>();

    // This operation depends on the producers of its operands
    for (const auto operand : op->getOperands()) {
      if (auto it = valueProducers.find(operand); it != valueProducers.end()) {
        dependsOn[op].insert(it->second);
      }
    }

    // Register this operation as the producer of its results
    for (auto result : op->getResults()) {
      valueProducers[result] = op;
    }
  }

  return dependsOn;
}

/// Partition operations into groups that can be compared as multisets.
/// Operations in the same group are independent and can be reordered.
std::vector<llvm::SmallVector<Operation*>>
partitionIndependentGroups(ArrayRef<Operation*> ops) {
  std::vector<llvm::SmallVector<Operation*>> groups;
  if (ops.empty()) {
    return groups;
  }

  auto dependsOn = buildDependenceGraph(ops);
  const llvm::DenseSet<Operation*> processed;
  llvm::SmallVector<Operation*> currentGroup;

  for (auto* op : ops) {
    bool dependsOnCurrent = false;

    // Check if this operation depends on any operation in the current group
    for (const auto* groupOp : currentGroup) {
      if (dependsOn[op].contains(groupOp)) {
        dependsOnCurrent = true;
        break;
      }
    }

    // Check if this operation has ordering constraints
    const auto hasConstraints = hasOrderingConstraints(op);

    // If it depends on current group or has ordering constraints,
    // finalize the current group and start a new one
    if (dependsOnCurrent || (hasConstraints && !currentGroup.empty())) {
      if (!currentGroup.empty()) {
        groups.push_back(std::move(currentGroup));
        currentGroup = {};
      }
    }

    currentGroup.push_back(op);

    // If this operation has ordering constraints, finalize the group
    if (hasConstraints) {
      groups.push_back(std::move(currentGroup));
      currentGroup = {};
    }
  }

  // Add any remaining operations
  if (!currentGroup.empty()) {
    groups.push_back(std::move(currentGroup));
  }

  return groups;
}

/// Compare two groups of independent operations using multiset equivalence.
bool areIndependentGroupsEquivalent(ArrayRef<Operation*> lhsOps,
                                    ArrayRef<Operation*> rhsOps) {
  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  // Build frequency maps for both groups
  std::unordered_map<Operation*, size_t, OperationStructuralHash,
                     OperationStructuralEquality>
      lhsFrequencyMap;
  std::unordered_map<Operation*, size_t, OperationStructuralHash,
                     OperationStructuralEquality>
      rhsFrequencyMap;

  for (auto* op : lhsOps) {
    lhsFrequencyMap[op]++;
  }

  for (auto* op : rhsOps) {
    rhsFrequencyMap[op]++;
  }

  // Check structural equivalence
  if (lhsFrequencyMap.size() != rhsFrequencyMap.size()) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
  for (const auto& [lhsOp, lhsCount] : lhsFrequencyMap) {
    auto it = rhsFrequencyMap.find(lhsOp);
    if (it == rhsFrequencyMap.end() || it->second != lhsCount) {
      return false;
    }
  }

  return true;
}

/// Compare two blocks for structural equivalence, allowing permutations
/// of independent operations.
bool areBlocksEquivalent(Block& lhs, Block& rhs,
                         ValueEquivalenceMap& valueMap) {
  // Check block arguments
  if (lhs.getNumArguments() != rhs.getNumArguments()) {
    return false;
  }

  for (auto [lhsArg, rhsArg] :
       llvm::zip(lhs.getArguments(), rhs.getArguments())) {
    if (lhsArg.getType() != rhsArg.getType()) {
      return false;
    }
    valueMap[lhsArg] = rhsArg;
  }

  // Collect all operations
  llvm::SmallVector<Operation*> lhsOps;
  llvm::SmallVector<Operation*> rhsOps;

  for (Operation& op : lhs) {
    lhsOps.push_back(&op);
  }

  for (Operation& op : rhs) {
    rhsOps.push_back(&op);
  }

  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  // Partition operations into independent groups
  auto lhsGroups = partitionIndependentGroups(lhsOps);
  auto rhsGroups = partitionIndependentGroups(rhsOps);

  if (lhsGroups.size() != rhsGroups.size()) {
    return false;
  }

  // Compare each group
  for (size_t groupIdx = 0; groupIdx < lhsGroups.size(); ++groupIdx) {
    auto& lhsGroup = lhsGroups[groupIdx];
    auto& rhsGroup = rhsGroups[groupIdx];

    if (!areIndependentGroupsEquivalent(lhsGroup, rhsGroup)) {
      return false;
    }

    // Update value mappings for operations in this group
    // We need to match operations and update the value map
    // Since they are structurally equivalent, we can match them
    // by trying all permutations (for small groups) or use a greedy approach

    // Use a simple greedy matching
    llvm::DenseSet<Operation*> matchedRhs;
    for (Operation* lhsOp : lhsGroup) {
      bool matched = false;
      for (Operation* rhsOp : rhsGroup) {
        if (matchedRhs.contains(rhsOp)) {
          continue;
        }

        ValueEquivalenceMap tempMap = valueMap;
        if (areOperationsEquivalent(lhsOp, rhsOp, tempMap)) {
          valueMap = std::move(tempMap);
          matchedRhs.insert(rhsOp);
          matched = true;

          // Recursively compare regions
          for (auto [lhsRegion, rhsRegion] :
               llvm::zip(lhsOp->getRegions(), rhsOp->getRegions())) {
            if (!areRegionsEquivalent(lhsRegion, rhsRegion, valueMap)) {
              return false;
            }
          }
          break;
        }
      }

      if (!matched) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  ValueEquivalenceMap valueMap;
  return areRegionsEquivalent(lhs.getBodyRegion(), rhs.getBodyRegion(),
                              valueMap);
}
