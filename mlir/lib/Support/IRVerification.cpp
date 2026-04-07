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

#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <cmath>
#include <cstddef>
#include <utility>

using namespace mlir;

namespace {

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

/// Wrapper for Operation* with structural comparison semantics
struct StructuralOperationKey {
  Operation* op;

  explicit StructuralOperationKey(Operation* operation = nullptr)
      : op(operation) {}

  bool operator==(const StructuralOperationKey& other) const {
    if (op == other.op) {
      return true;
    }
    if (op == nullptr || other.op == nullptr) {
      return false;
    }
    return OperationStructuralEquality{}(op, other.op);
  }

  bool operator!=(const StructuralOperationKey& other) const {
    return !(*this == other);
  }
};

/// Map to track value equivalence between two modules.
using ValueEquivalenceMap = llvm::DenseMap<mlir::Value, mlir::Value>;

using OperationSet = llvm::DenseSet<Operation*>;

struct InsertWrite {
  Value scalar;
  Value index;
};

struct InsertChainSummary {
  Value baseTensor;
  Value finalTensor;
  llvm::SmallVector<InsertWrite> writes;
};

} /**
 * @brief Ensure that `lhs` is mapped to `rhs` in the provided equivalence map.
 *
 * If `lhs` already has an entry in `valueMap`, this verifies the existing
 * mapping equals `rhs`. Otherwise, it inserts the mapping `lhs -> rhs`.
 *
 * @param lhs The value from the left-hand side to check or record.
 * @param rhs The value from the right-hand side to associate with `lhs`.
 * @param valueMap Mapping of already-established SSA value equivalences.
 * @return `true` if `lhs` is (or was successfully) mapped to `rhs`, `false` otherwise.
 */

static bool areValuesEquivalent(Value lhs, Value rhs,
                                ValueEquivalenceMap& valueMap) {
  if (auto it = valueMap.find(lhs); it != valueMap.end()) {
    return it->second == rhs;
  }
  valueMap[lhs] = rhs;
  return true;
}

/**
 * @brief Determines whether two index values should be considered equivalent.
 *
 * First uses the specialized index equivalence check; if that does not
 * consider them equivalent, falls back to SSA-based equivalence which may
 * establish or validate a mapping in |valueMap|.
 *
 * @param lhs The left-hand index value.
 * @param rhs The right-hand index value.
 * @param valueMap Mapping of already-established equivalent SSA values;
 *                 may be updated to record a new equivalence lhs -> rhs.
 * @return true if the indices are equivalent (either by the specialized
 *         check or by a consistent/established SSA mapping), false otherwise.
 */
static bool areIndexValuesEquivalent(Value lhs, Value rhs,
                                     ValueEquivalenceMap& valueMap) {
  if (qtensor::areEquivalentIndices(lhs, rhs)) {
    return true;
  }
  return areValuesEquivalent(lhs, rhs, valueMap);
}

/**
 * @brief Determines whether an operation is a qtensor::InsertOp.
 *
 * @param op Operation pointer to test.
 * @return true if `op` is a `qtensor::InsertOp`, false otherwise.
 */
static bool isQTensorInsertOp(Operation* op) {
  return llvm::isa<qtensor::InsertOp>(op);
}

/**
 * @brief Determines whether a dependency between two operations that are
 * qtensor::InsertOp can be treated as commutable (i.e., the dependent can be
 * reordered relative to the dependency without semantic conflict).
 *
 * Both parameters must be `qtensor::InsertOp`; the dependent must take the
 * dependency's result as its destination, both insert indices must be constant
 * integers, and the two indices must refer to different element positions.
 *
 * @param dependent The operation that depends on `dependency` (consumer).
 * @param dependency The operation producing a tensor used by `dependent` (producer).
 * @return true if the dependency edge can be considered commutable, false otherwise.
 */
static bool isCommutableQTensorInsertDependency(Operation* dependent,
                                                Operation* dependency) {
  auto dependentInsert = llvm::dyn_cast<qtensor::InsertOp>(dependent);
  auto dependencyInsert = llvm::dyn_cast<qtensor::InsertOp>(dependency);
  if (!dependentInsert || !dependencyInsert) {
    return false;
  }
  if (dependentInsert.getDest() != dependencyInsert.getResult()) {
    return false;
  }
  auto dependentIndex = dependentInsert.getIndex();
  auto dependencyIndex = dependencyInsert.getIndex();
  if (!getConstantIntValue(dependentIndex) ||
      !getConstantIntValue(dependencyIndex)) {
    return false;
  }
  return !qtensor::areEquivalentIndices(dependentIndex, dependencyIndex);
}

/**
 * @brief Find the earliest tensor in an insert chain that lies within a given group.
 *
 * Walks backwards from `tensor` through defining `qtensor::InsertOp` operations while each
 * encountered insert operation is a member of `group`, and returns the first tensor reached
 * that is not produced by an insert in `group`.
 *
 * @param tensor Starting tensor value at the end of an insert chain.
 * @param group Set of operations considered part of the insert group.
 * @return Value The base tensor of the chain (the earliest tensor not defined by a group insert).
 */
static Value getInsertChainBaseTensor(Value tensor, const OperationSet& group) {
  auto current = tensor;
  while (auto insertOp = current.getDefiningOp<qtensor::InsertOp>()) {
    if (!group.contains(insertOp.getOperation())) {
      break;
    }
    current = insertOp.getDest();
  }
  return current;
}

/**
 * @brief Summarizes a set of `qtensor.insert` operations into insert chains.
 *
 * Builds one InsertChainSummary per distinct base tensor found among the given
 * operations. Each chain records the base tensor, the ordered sequence of
 * `{scalar, index}` writes encountered for that base, and the final tensor
 * produced by the chain.
 *
 * @param ops Array of operations expected to be `qtensor::InsertOp`.
 * @param[out] chains Populated with the resulting InsertChainSummary entries;
 *                    existing contents will be appended to.
 * @return true if every chain has a single unambiguous final tensor and no
 *         chain contains multiple writes that target equivalent indices;
 *         returns false on ambiguity or on duplicate-equivalent-index writes.
 */
static bool
summarizeInsertGroup(llvm::ArrayRef<Operation*> ops,
                     llvm::SmallVectorImpl<InsertChainSummary>& chains) {
  OperationSet groupOps;
  for (Operation* op : ops) {
    groupOps.insert(op);
  }

  llvm::DenseSet<Value> consumedInsertResults;
  for (Operation* op : ops) {
    auto insertOp = llvm::cast<qtensor::InsertOp>(op);
    if (auto definingInsert =
            insertOp.getDest().getDefiningOp<qtensor::InsertOp>()) {
      if (groupOps.contains(definingInsert.getOperation())) {
        consumedInsertResults.insert(insertOp.getDest());
      }
    }
  }

  llvm::DenseMap<Value, size_t> chainByBaseTensor;
  for (Operation* op : ops) {
    auto insertOp = llvm::cast<qtensor::InsertOp>(op);
    const Value baseTensor =
        getInsertChainBaseTensor(insertOp.getDest(), groupOps);

    size_t chainIdx = 0;
    if (auto it = chainByBaseTensor.find(baseTensor);
        it != chainByBaseTensor.end()) {
      chainIdx = it->second;
    } else {
      chainIdx = chains.size();
      chainByBaseTensor[baseTensor] = chainIdx;
      InsertChainSummary summary;
      summary.baseTensor = baseTensor;
      chains.emplace_back(std::move(summary));
    }

    auto& chain = chains[chainIdx];
    chain.writes.push_back(InsertWrite{.scalar = insertOp.getScalar(),
                                       .index = insertOp.getIndex()});

    if (!consumedInsertResults.contains(insertOp.getResult())) {
      if (chain.finalTensor) {
        return false;
      }
      chain.finalTensor = insertOp.getResult();
    }
  }

  for (const auto& chain : chains) {
    if (!chain.finalTensor) {
      return false;
    }

    // Reordering writes to the same index is not semantics-preserving.
    llvm::SmallVector<Value> seenIndices;
    for (const auto& write : chain.writes) {
      if (llvm::any_of(seenIndices, [&](Value seenIndex) {
            return qtensor::areEquivalentIndices(seenIndex, write.index);
          })) {
        return false;
      }
      seenIndices.push_back(write.index);
    }
  }

  return true;
}

/**
 * @brief Determines whether two sequences of insert writes can be matched one-to-one.
 *
 * Attempts to match each write in `lhsWrites` to a distinct, unused write in
 * `rhsWrites` (order-independent) using backtracking. A match requires the
 * corresponding scalar values to be equivalent and the indices to be
 * equivalent under the equivalence checks; temporary SSA mappings are merged
 * into `valueMap` only when a complete matching for all remaining writes is
 * found.
 *
 * @param lhsIdx Index of the current write in `lhsWrites` to match; callers
 *               should pass 0 to start a full matching.
 * @param lhsWrites Left-hand sequence of writes to match.
 * @param rhsWrites Right-hand sequence of writes to match against.
 * @param rhsUsed Mutable boolean-like vector (chars) parallel to `rhsWrites`
 *               that marks which `rhsWrites` entries are already used during
 *               the current matching search; its contents are preserved on
 *               failure paths and reflect usage on success (matching is
 *               committed into `valueMap`).
 * @param valueMap Mapping of SSA values from left to right that will be
 *                 tentatively extended during matching and permanently updated
 *                 only if a complete match is found.
 * @return true if there exists a one-to-one matching of all writes in
 *         `lhsWrites` to distinct writes in `rhsWrites` satisfying scalar and
 *         index equivalence; `false` otherwise.
 */
static bool areInsertWritesEquivalentRec(const size_t lhsIdx,
                                         llvm::ArrayRef<InsertWrite> lhsWrites,
                                         llvm::ArrayRef<InsertWrite> rhsWrites,
                                         llvm::SmallVectorImpl<char>& rhsUsed,
                                         ValueEquivalenceMap& valueMap) {
  if (lhsIdx == lhsWrites.size()) {
    return true;
  }

  for (size_t rhsIdx = 0; rhsIdx < rhsWrites.size(); ++rhsIdx) {
    if (rhsUsed[rhsIdx] != 0) {
      continue;
    }

    ValueEquivalenceMap tempMap = valueMap;
    if (!areValuesEquivalent(lhsWrites[lhsIdx].scalar, rhsWrites[rhsIdx].scalar,
                             tempMap) ||
        !areIndexValuesEquivalent(lhsWrites[lhsIdx].index,
                                  rhsWrites[rhsIdx].index, tempMap)) {
      continue;
    }

    rhsUsed[rhsIdx] = 1;
    if (areInsertWritesEquivalentRec(lhsIdx + 1, lhsWrites, rhsWrites, rhsUsed,
                                     tempMap)) {
      valueMap = std::move(tempMap);
      return true;
    }
    rhsUsed[rhsIdx] = 0;
  }

  return false;
}

/**
 * @brief Determines whether two sequences of insert writes are equivalent as a
 *        multiset, allowing writes to be reordered.
 *
 * Compares the `lhsWrites` and `rhsWrites` sequences by finding a one-to-one
 * pairing between writes such that corresponding scalar values and indices are
 * equivalent. The provided `valueMap` is tentatively extended during matching
 * and is committed only if a complete matching succeeds; on failure `valueMap`
 * is left unchanged.
 *
 * @param lhsWrites Sequence of writes (each contains a scalar and an index) from
 *                  the left-hand side.
 * @param rhsWrites Sequence of writes from the right-hand side.
 * @param valueMap  Mapping of SSA values from lhs to rhs that will be updated
 *                  only when the sequences are found equivalent.
 * @return true if the two write sequences have the same size and can be
 *         matched one-to-one with equivalent scalars and indices, `false`
 *         otherwise.
 */
static bool areInsertWritesEquivalent(llvm::ArrayRef<InsertWrite> lhsWrites,
                                      llvm::ArrayRef<InsertWrite> rhsWrites,
                                      ValueEquivalenceMap& valueMap) {
  if (lhsWrites.size() != rhsWrites.size()) {
    return false;
  }
  llvm::SmallVector<char> rhsUsed(rhsWrites.size(), 0);
  return areInsertWritesEquivalentRec(0, lhsWrites, rhsWrites, rhsUsed,
                                      valueMap);
}

/**
 * @brief Determine whether two insert-chain summaries are equivalent and, if so, commit the resulting SSA value mappings.
 *
 * Compares the two chains' base tensors, ordered writes, and final tensors for structural and SSA equivalence
 * while tentatively extending the provided value mapping. The input mapping is only updated when the chains
 * are fully matched.
 *
 * @param lhsChain The insert-chain summary from the left-hand side.
 * @param rhsChain The insert-chain summary from the right-hand side.
 * @param valueMap Mapping of SSA values from lhs to rhs; extended on success to include new equivalences.
 * @return true if the chains are equivalent and `valueMap` has been updated with the matching; false otherwise.
 */
static bool areInsertChainsEquivalent(const InsertChainSummary& lhsChain,
                                      const InsertChainSummary& rhsChain,
                                      ValueEquivalenceMap& valueMap) {
  ValueEquivalenceMap tempMap = valueMap;
  if (!areValuesEquivalent(lhsChain.baseTensor, rhsChain.baseTensor, tempMap)) {
    return false;
  }

  if (!areInsertWritesEquivalent(lhsChain.writes, rhsChain.writes, tempMap)) {
    return false;
  }

  if (!areValuesEquivalent(lhsChain.finalTensor, rhsChain.finalTensor,
                           tempMap)) {
    return false;
  }

  valueMap = std::move(tempMap);
  return true;
}

/**
 * @brief Recursively matches each insert chain in `lhsChains` to a distinct chain in `rhsChains`, establishing SSA value equivalences.
 *
 * Attempts to find a bijection between `lhsChains` and `rhsChains` by backtracking: for each unmatched left chain (starting at `lhsChainIdx`),
 * it tries unused right chains and tests chain-level equivalence. When a full matching is found, `valueMap` is updated to include all established
 * equivalences; on failure `valueMap` is left unchanged.
 *
 * @param lhsChainIdx Index of the current left-chain to match.
 * @param lhsChains Array of left-side insert chain summaries to match.
 * @param rhsChains Array of right-side insert chain summaries to match.
 * @param rhsChainUsed Mutable bitmap indicating which `rhsChains` entries are already matched (nonzero = used).
 * @param valueMap Mutable SSA value equivalence map that is extended on successful complete matching.
 * @return true if a complete injective matching from remaining `lhsChains` to unused `rhsChains` exists and `valueMap` has been updated accordingly, `false` otherwise.
 */
static bool areInsertGroupsEquivalentRec(
    const size_t lhsChainIdx, llvm::ArrayRef<InsertChainSummary> lhsChains,
    llvm::ArrayRef<InsertChainSummary> rhsChains,
    llvm::SmallVectorImpl<char>& rhsChainUsed, ValueEquivalenceMap& valueMap) {
  if (lhsChainIdx == lhsChains.size()) {
    return true;
  }

  for (size_t rhsChainIdx = 0; rhsChainIdx < rhsChains.size(); ++rhsChainIdx) {
    if (rhsChainUsed[rhsChainIdx] != 0) {
      continue;
    }

    ValueEquivalenceMap tempMap = valueMap;
    if (!areInsertChainsEquivalent(lhsChains[lhsChainIdx],
                                   rhsChains[rhsChainIdx], tempMap)) {
      continue;
    }

    rhsChainUsed[rhsChainIdx] = 1;
    if (areInsertGroupsEquivalentRec(lhsChainIdx + 1, lhsChains, rhsChains,
                                     rhsChainUsed, tempMap)) {
      valueMap = std::move(tempMap);
      return true;
    }
    rhsChainUsed[rhsChainIdx] = 0;
  }

  return false;
}

/**
 * @brief Determines whether two collections of `qtensor.insert` operations form
 * equivalent insert groups and, if so, extends the SSA value equivalence map.
 *
 * Both input arrays are treated as unordered groups of insert chains; the
 * function validates that the groups contain the same number of operations,
 * that each side can be summarized into compatible insert chains, and that the
 * chains and their writes can be pairwise matched under a consistent value
 * mapping. On success, `valueMap` is updated with the discovered equivalences.
 *
 * @param lhsOps Array of operations from the left-hand group (expected to be `qtensor.insert`).
 * @param rhsOps Array of operations from the right-hand group (expected to be `qtensor.insert`).
 * @param valueMap Mapping of SSA values from lhs to rhs which will be extended when equivalence is found.
 * @return `true` if the two groups are equivalent and `valueMap` was extended accordingly, `false` otherwise.
 */
static bool areInsertGroupsEquivalent(llvm::ArrayRef<Operation*> lhsOps,
                                      llvm::ArrayRef<Operation*> rhsOps,
                                      ValueEquivalenceMap& valueMap) {
  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  llvm::SmallVector<InsertChainSummary> lhsChains;
  llvm::SmallVector<InsertChainSummary> rhsChains;
  if (!summarizeInsertGroup(lhsOps, lhsChains) ||
      !summarizeInsertGroup(rhsOps, rhsChains)) {
    return false;
  }
  if (lhsChains.size() != rhsChains.size()) {
    return false;
  }

  llvm::SmallVector<char> rhsChainUsed(rhsChains.size(), 0);
  return areInsertGroupsEquivalentRec(0, lhsChains, rhsChains, rhsChainUsed,
                                      valueMap);
}

/// DenseMapInfo specialization for StructuralOperationKey
template <> struct llvm::DenseMapInfo<StructuralOperationKey> {
  /**
   * @brief Constructs an empty StructuralOperationKey sentinel.
   *
   * @return StructuralOperationKey A key representing the DenseMap empty sentinel for an `Operation*`.
   */
  static StructuralOperationKey getEmptyKey() {
    return StructuralOperationKey(DenseMapInfo<Operation*>::getEmptyKey());
  }

  static StructuralOperationKey getTombstoneKey() {
    return StructuralOperationKey(DenseMapInfo<Operation*>::getTombstoneKey());
  }

  static unsigned getHashValue(const StructuralOperationKey& key) {
    if (key.op == getEmptyKey().op || key.op == getTombstoneKey().op) {
      return DenseMapInfo<Operation*>::getHashValue(key.op);
    }
    return OperationStructuralHash{}(key.op);
  }

  static bool isEqual(const StructuralOperationKey& lhs,
                      const StructuralOperationKey& rhs) {
    // Handle special keys
    if (lhs.op == getEmptyKey().op) {
      return rhs.op == getEmptyKey().op;
    }
    if (lhs.op == getTombstoneKey().op) {
      return rhs.op == getTombstoneKey().op;
    }
    if (rhs.op == getEmptyKey().op || rhs.op == getTombstoneKey().op) {
      return false;
    }
    return lhs == rhs;
  }
};

static bool areFloatValuesNear(const llvm::APFloat& lhs,
                               const llvm::APFloat& rhs, const unsigned width) {
  if (lhs.isNaN() || rhs.isNaN()) {
    return lhs.isNaN() && rhs.isNaN();
  }
  if (lhs.isInfinity() || rhs.isInfinity()) {
    return lhs.isInfinity() && rhs.isInfinity() &&
           lhs.isNegative() == rhs.isNegative();
  }

  const double lhsVal = lhs.convertToDouble();
  const double rhsVal = rhs.convertToDouble();
  const double absDiff = std::fabs(lhsVal - rhsVal);
  const double absLhs = std::fabs(lhsVal);
  const double absRhs = std::fabs(rhsVal);
  const double scale = absLhs > absRhs ? absLhs : absRhs;

  double relTol = 1e-12;
  double absTol = 1e-15;
  if (width <= 16) {
    relTol = 1e-3;
    absTol = 1e-6;
  } else if (width <= 32) {
    relTol = 1e-9;
    absTol = 1e-12;
  }
  return absDiff <= absTol + (relTol * scale);
}

static bool areConstantAttributesEquivalent(const Attribute& lhs,
                                            const Attribute& rhs) {
  if (lhs == rhs) {
    return true;
  }

  if (auto lhsFloat = llvm::dyn_cast<FloatAttr>(lhs)) {
    auto rhsFloat = llvm::dyn_cast<FloatAttr>(rhs);
    if (!rhsFloat) {
      return false;
    }
    return areFloatValuesNear(lhsFloat.getValue(), rhsFloat.getValue(),
                              lhsFloat.getType().getIntOrFloatBitWidth());
  }

  return false;
}

/// Compare two operations for structural equivalence.
/// Updates valueMap to track corresponding SSA values.
static bool areOperationsEquivalent(Operation* lhs, Operation* rhs,
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

    if (!areConstantAttributesEquivalent(lhsConst.getValue(),
                                         rhsConst.getValue())) {
      return false;
    }
  }

  // Check LLVM::ConstantOp
  if (auto lhsConst = llvm::dyn_cast<LLVM::ConstantOp>(lhs)) {
    auto rhsConst = llvm::dyn_cast<LLVM::ConstantOp>(rhs);
    if (!rhsConst) {
      return false;
    }
    if (!areConstantAttributesEquivalent(lhsConst.getValue(),
                                         rhsConst.getValue())) {
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
static bool areBlocksEquivalent(Block& lhs, Block& rhs,
                                ValueEquivalenceMap& valueMap);

/// Compare two regions for structural equivalence.
static bool areRegionsEquivalent(Region& lhs, Region& rhs,
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
static bool hasOrderingConstraints(Operation* op) {
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
llvm::DenseMap<
    Operation*,
    llvm::DenseSet<
        Operation*>> static buildDependenceGraph(llvm::ArrayRef<Operation*>
                                                     ops) {
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
/**
 * @brief Partition a sequence of operations into sequential groups whose
 * operations are independent and may be reordered within each group.
 *
 * The function preserves the original program order between groups. An
 * operation is placed into the current group unless it depends on any
 * operation already in that group (per the dependence graph) or the
 * operation imposes ordering constraints; in those cases a new group is
 * started. A dependence from an insert operation to another insert is
 * treated as commutable (and thus ignored) when `isCommutableQTensorInsertDependency`
 * permits it. Operations for which `hasOrderingConstraints` returns true
 * force a group boundary.
 *
 * @param ops Sequence of operations in program order to partition.
 * @return llvm::SmallVector<llvm::SmallVector<Operation*>> A vector of groups
 *         (in original order). Each inner vector contains operations that are
 *         independent of one another and may be reordered within that group.
 */
llvm::SmallVector<llvm::SmallVector<
    Operation*>> static partitionIndependentGroups(llvm::ArrayRef<Operation*>
                                                       ops) {
  llvm::SmallVector<llvm::SmallVector<Operation*>> groups;
  if (ops.empty()) {
    return groups;
  }

  auto dependsOn = buildDependenceGraph(ops);
  llvm::SmallVector<Operation*> currentGroup;

  for (auto* op : ops) {
    bool dependsOnCurrent = false;

    // Check if this operation depends on any operation in the current group
    for (auto* groupOp : currentGroup) {
      if (!dependsOn[op].contains(groupOp)) {
        continue;
      }
      if (isCommutableQTensorInsertDependency(op, groupOp)) {
        continue;
      }
      dependsOnCurrent = true;
      break;
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
static bool areIndependentGroupsEquivalent(llvm::ArrayRef<Operation*> lhsOps,
                                           llvm::ArrayRef<Operation*> rhsOps) {
  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  // Build frequency maps for both groups
  llvm::DenseMap<StructuralOperationKey, size_t> lhsFrequencyMap;
  llvm::DenseMap<StructuralOperationKey, size_t> rhsFrequencyMap;

  for (auto* op : lhsOps) {
    lhsFrequencyMap[StructuralOperationKey(op)]++;
  }

  for (auto* op : rhsOps) {
    rhsFrequencyMap[StructuralOperationKey(op)]++;
  }

  // Check structural equivalence
  if (lhsFrequencyMap.size() != rhsFrequencyMap.size()) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-nondeterministic-pointer-iteration-order)
  for (const auto& [lhsKey, lhsCount] : lhsFrequencyMap) {
    auto it = rhsFrequencyMap.find(lhsKey);
    if (it == rhsFrequencyMap.end() || it->second != lhsCount) {
      return false;
    }
  }

  return true;
}

/// Compare two blocks for structural equivalence, allowing permutations
/**
 * @brief Determines whether two blocks are structurally equivalent up to
 *        reordering of independent operations.
 *
 * Compares block arguments (types and establishes initial SSA mappings), then
 * partitions each block's operations into independent groups and verifies
 * equivalence group-by-group. Groups composed entirely of `qtensor.insert`
 * operations are compared with insert-chain semantics; other groups are matched
 * as multisets of structurally equivalent operations while updating the SSA
 * mapping and recursively checking nested regions.
 *
 * @param lhs Left-hand block to compare.
 * @param rhs Right-hand block to compare.
 * @param valueMap Mapping of SSA values from `lhs` to `rhs`; updated on success
 *                 to reflect established equivalences.
 * @return true if the blocks are equivalent (modulo permitted independent
 *         operation reorderings) and `valueMap` reflects the mapping between
 *         their SSA values, `false` otherwise.
 */
static bool areBlocksEquivalent(Block& lhs, Block& rhs,
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

    const bool lhsInsertGroup = llvm::all_of(lhsGroup, isQTensorInsertOp);
    const bool rhsInsertGroup = llvm::all_of(rhsGroup, isQTensorInsertOp);
    if (lhsInsertGroup || rhsInsertGroup) {
      if (!lhsInsertGroup || !rhsInsertGroup) {
        return false;
      }
      if (!areInsertGroupsEquivalent(lhsGroup, rhsGroup, valueMap)) {
        return false;
      }
      continue;
    }

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

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  ValueEquivalenceMap valueMap;
  return areRegionsEquivalent(lhs.getBodyRegion(), rhs.getBodyRegion(),
                              valueMap);
}
