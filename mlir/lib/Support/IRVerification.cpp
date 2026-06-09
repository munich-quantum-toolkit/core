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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

using namespace mlir;

namespace {

using Slice = SetVector<Operation*>;

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
using ValueEquivalenceMap = DenseMap<mlir::Value, mlir::Value>;

using OperationSet = DenseSet<Operation*>;

struct InsertWrite {
  Value scalar;
  Value index;
};

struct InsertChainSummary {
  Value baseTensor;
  Value finalTensor;
  SmallVector<InsertWrite> writes;
};

} // namespace

static bool areValuesEquivalent(Value lhs, Value rhs,
                                ValueEquivalenceMap& valueMap) {
  if (auto it = valueMap.find(lhs); it != valueMap.end()) {
    return it->second == rhs;
  }
  valueMap[lhs] = rhs;
  return true;
}

static bool areIndexValuesEquivalent(Value lhs, Value rhs,
                                     ValueEquivalenceMap& valueMap) {
  if (qtensor::areEquivalentIndices(lhs, rhs)) {
    return true;
  }
  return areValuesEquivalent(lhs, rhs, valueMap);
}

static bool isQTensorInsertOp(Operation* op) {
  return isa<qtensor::InsertOp>(op);
}

static bool isCommutableQTensorInsertDependency(Operation* dependent,
                                                Operation* dependency) {
  auto dependentInsert = dyn_cast<qtensor::InsertOp>(dependent);
  auto dependencyInsert = dyn_cast<qtensor::InsertOp>(dependency);
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

static bool summarizeInsertGroup(ArrayRef<Operation*> ops,
                                 SmallVectorImpl<InsertChainSummary>& chains) {
  OperationSet groupOps;
  for (Operation* op : ops) {
    groupOps.insert(op);
  }

  DenseSet<Value> consumedInsertResults;
  for (Operation* op : ops) {
    auto insertOp = cast<qtensor::InsertOp>(op);
    if (auto definingInsert =
            insertOp.getDest().getDefiningOp<qtensor::InsertOp>()) {
      if (groupOps.contains(definingInsert.getOperation())) {
        consumedInsertResults.insert(insertOp.getDest());
      }
    }
  }

  DenseMap<Value, size_t> chainByBaseTensor;
  for (Operation* op : ops) {
    auto insertOp = cast<qtensor::InsertOp>(op);
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
    SmallVector<Value> seenIndices;
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

static bool areInsertWritesEquivalentRec(const size_t lhsIdx,
                                         ArrayRef<InsertWrite> lhsWrites,
                                         ArrayRef<InsertWrite> rhsWrites,
                                         SmallVectorImpl<char>& rhsUsed,
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

static bool areInsertWritesEquivalent(ArrayRef<InsertWrite> lhsWrites,
                                      ArrayRef<InsertWrite> rhsWrites,
                                      ValueEquivalenceMap& valueMap) {
  if (lhsWrites.size() != rhsWrites.size()) {
    return false;
  }
  SmallVector<char> rhsUsed(rhsWrites.size(), 0);
  return areInsertWritesEquivalentRec(0, lhsWrites, rhsWrites, rhsUsed,
                                      valueMap);
}

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

static bool areInsertGroupsEquivalentRec(const size_t lhsChainIdx,
                                         ArrayRef<InsertChainSummary> lhsChains,
                                         ArrayRef<InsertChainSummary> rhsChains,
                                         SmallVectorImpl<char>& rhsChainUsed,
                                         ValueEquivalenceMap& valueMap) {
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

static bool areInsertGroupsEquivalent(ArrayRef<Operation*> lhsOps,
                                      ArrayRef<Operation*> rhsOps,
                                      ValueEquivalenceMap& valueMap) {
  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  SmallVector<InsertChainSummary> lhsChains;
  SmallVector<InsertChainSummary> rhsChains;
  if (!summarizeInsertGroup(lhsOps, lhsChains) ||
      !summarizeInsertGroup(rhsOps, rhsChains)) {
    return false;
  }
  if (lhsChains.size() != rhsChains.size()) {
    return false;
  }

  SmallVector<char> rhsChainUsed(rhsChains.size(), 0);
  return areInsertGroupsEquivalentRec(0, lhsChains, rhsChains, rhsChainUsed,
                                      valueMap);
}

/// DenseMapInfo specialization for StructuralOperationKey
template <> struct llvm::DenseMapInfo<StructuralOperationKey> {
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

static bool approxCompareFloats(const APFloat& lhs, const APFloat& rhs,
                                const unsigned width) {
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

static bool compareAttributes(const Attribute& attrA, const Attribute& attrB) {
  if (dyn_cast<UnitAttr>(attrA)) {
    if (!dyn_cast<UnitAttr>(attrB)) {
      return false;
    }
  } else if (auto intAttrA = dyn_cast<IntegerAttr>(attrA)) {
    auto intAttrB = dyn_cast<IntegerAttr>(attrB);
    if (!intAttrB || intAttrA.getValue() != intAttrB.getValue()) {
      return false;
    }
  } else if (auto floatAttrA = dyn_cast<FloatAttr>(attrA)) {
    auto floatAttrB = dyn_cast<FloatAttr>(attrB);

    if (!floatAttrB ||
        !approxCompareFloats(floatAttrA.getValue(), floatAttrB.getValue(),
                             floatAttrA.getType().getIntOrFloatBitWidth())) {
      return false;
    }
  } else if (auto strAttrA = dyn_cast<StringAttr>(attrA)) {
    auto strAttrB = dyn_cast<StringAttr>(attrB);
    if (!strAttrB || strAttrA.getValue() != strAttrB.getValue()) {
      return false;
    }
  } else if (auto arrayAttrA = dyn_cast<ArrayAttr>(attrA)) {
    auto arrayAttrB = dyn_cast<ArrayAttr>(attrB);
    if (!arrayAttrB) {
      return false;
    }

    if (arrayAttrA.size() != arrayAttrB.size()) {
      return false;
    }

    // Note: This assumes that the array attributes are equivalently ordered.
    for (const auto [elementAttrA, elementAttrB] :
         llvm::zip_equal(arrayAttrA, arrayAttrB)) {
      if (!compareAttributes(elementAttrA, elementAttrB)) {
        return false;
      }
    }
  } else if (auto denseArrayAttrA =
                 llvm::dyn_cast<mlir::DenseArrayAttr>(attrA)) {
    auto denseArrayAttrB = llvm::dyn_cast<mlir::DenseArrayAttr>(attrB);
    if (!denseArrayAttrB || denseArrayAttrA.size() != denseArrayAttrB.size() ||
        denseArrayAttrA.getElementType() != denseArrayAttrB.getElementType()) {
      return false;
    }

    for (const auto [valA, valB] : llvm::zip_equal(
             denseArrayAttrA.getRawData(), denseArrayAttrB.getRawData())) {
      if (valA != valB) {
        return false;
      }
    }

  } else {
    attrA.dump();
    llvm::reportFatalInternalError("unhandled attribute type!");
    llvm::llvm_unreachable_internal();
  }

  return true;
}

static bool isQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

static bool compareOperations(Operation* opA, Operation* opB, IRMapping& m) {

  // Compare top-level signature-like characteristics.

  if (opA->getName() != opB->getName() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      opA->getOperandTypes() != opB->getOperandTypes() ||
      opA->getNumResults() != opB->getNumResults() ||
      opA->getResultTypes() != opB->getResultTypes() ||
      opA->getAttrs().size() != opB->getAttrs().size() ||
      opA->getNumRegions() != opB->getNumRegions() ||
      opA->getDialect()->getNamespace() != opB->getDialect()->getNamespace()) {
    return false;
  }

  // Compare attributes.

  const DenseSet<StringRef> ignore{"function_type"};

  for (const auto& namedAttrA : opA->getAttrs()) {
    const StringRef keyA = namedAttrA.getName().strref();

    if (ignore.contains(keyA)) {
      return true;
    }

    if (!opB->hasAttr(keyA)) {
      return false;
    }

    if (!compareAttributes(namedAttrA.getValue(), opB->getAttr(keyA))) {
      return false;
    }
  }

  // Compare operands.
  // TODO: Equal type check.

  for (const auto& [operandA, operandB] :
       llvm::zip_equal(opA->getOperands(), opB->getOperands())) {

    if (isQubitTensor(operandA)) {
      if (!isQubitTensor(operandB)) { // TODO: Assertion?
        return false;
      }

      auto tensorA = cast<TypedValue<RankedTensorType>>(operandA);
      qtensor::TensorIterator itA(tensorA);
      while (std::prev(itA) != itA) {
        --itA;
      }

      auto tensorB = cast<TypedValue<RankedTensorType>>(operandB);
      qtensor::TensorIterator itB(tensorB);
      while (std::prev(itB) != itB) {
        --itB;
      }

      if (itA.operation() == nullptr) { // Block-Argument.
        if (itB.operation() != nullptr) {
          return false;
        }
      } else {
        auto allocA = cast<qtensor::AllocOp>(itA.operation());
        auto allocB = cast<qtensor::AllocOp>(itB.operation());
        if (m.lookup(allocA.getResult()) != allocB.getResult()) {
          return false;
        }
      }
    }
  }

  for (const auto& [resA, resB] :
       llvm::zip_equal(opA->getResults(), opB->getResults())) {
    if (!isa<qtensor::AllocOp>(opA) && isQubitTensor(resA)) {
      if (!isQubitTensor(resB)) {
        return false;
      }
      continue;
    }

    m.map(resA, resB);
  }

  return true;
}

/// Returns a vector of maximum reachable independent DAGs, as defined via their
/// def-use chain, where the operations of the setvector are topologically
/// sorted. The vector is sorted by the size of the DAG (i.e., the number of
/// operations).
static SmallVector<Slice> getDisjointSlices(Block& b) {
  const auto filter = [&b](Operation* op) { return op->getBlock() == &b; };

  const BackwardSliceOptions backwardSliceOptions(filter);
  const SliceOptions forwardSliceOptions(filter);

  DenseSet<Operation*> visited;
  visited.reserve(range_size(b.getOperations()));

  SmallVector<SetVector<Operation*>> dags;
  for (Operation& op : b.getOperations()) {
    if (visited.contains(&op)) {
      continue;
    }
    const auto slice = getSlice(&op, backwardSliceOptions, forwardSliceOptions);
    const auto& dag = dags.emplace_back(slice);
    visited.insert_range(dag);
  }

  sort(dags, [](const auto& lhs, const auto& rhs) {
    return lhs.size() < rhs.size();
  });

  return dags;
}

/// Extract and return "ready" operations from the slice.
static Slice getReadyOps(const Slice& slice, DenseSet<Operation*>& visited) {
  const auto isReady = [&](OpOperand& operand) {
    if (isa<BlockArgument>(operand.get())) {
      return true;
    }
    return visited.contains(operand.get().getDefiningOp());
  };

  Slice ready;
  for (Operation* op : slice) {
    if (visited.contains(op)) {
      continue;
    }

    if (llvm::all_of(op->getOpOperands(), isReady)) {
      ready.insert(op);
      continue;
    }

    // If the destination of a tensor insert, has been produced by an insert
    // operation as well, these two should be interchangable. Thus, also add it
    // to the ready set vector. Any valid IR will ensure that the indices of the
    // two insertions are not equivalent, hence, we don't check them here.

    if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {
      Operation* prev = insert.getDest().getDefiningOp();
      if (isa<qtensor::InsertOp>(prev)) {
        if (ready.contains(prev)) {
          ready.insert(insert.getOperation());
          continue;
        }
      }
    }

    // Analogously for the extract operation.

    if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      Operation* prev = extract.getTensor().getDefiningOp();
      if (isa<qtensor::ExtractOp>(prev)) {
        if (ready.contains(prev)) {
          ready.insert(extract.getOperation());
          continue;
        }
      }
    }
  }

  return ready;
}

static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& visited, IRMapping& m);

static bool compareSlices(const Slice& lhs, const Slice& rhs,
                          DenseSet<Operation*>& visited, IRMapping& m) {
  while (true) {
    const auto readyLhs = getReadyOps(lhs, visited);
    const auto readyRhs = getReadyOps(rhs, visited);

    if (readyLhs.empty() || readyRhs.empty()) {
      break;
    }

    if (readyLhs.size() != readyRhs.size()) {
      return false;
    }

    // Greedily find structural equivalent operation for each op on the lefthand
    // side.
    SmallVector<std::pair<Operation*, Operation*>> nested;
    for (Operation* opA : readyLhs) {
      bool partnerFound{false};
      for (Operation* opB : readyRhs) {
        if (compareOperations(opA, opB, m)) {
          llvm::dbgs() << opA->getName() << " == " << opB->getName() << '\n';
          m.map(opA, opB); // Create op mapping.

          if (opA->getNumRegions() != 0) {
            nested.emplace_back(opA, opB);
          }

          partnerFound = true;
          break;
        }
      }

      if (!partnerFound) {
        llvm::dbgs() << "no matching op found: " << opA->getName() << '\n';
        return false;
      }
    }

    visited.insert_range(readyLhs);
    visited.insert_range(readyRhs);

    for (auto& [opA, opB] : nested) {
      for (const auto [regionA, regionB] :
           llvm::zip_equal(opA->getRegions(), opB->getRegions())) {
        if (!compareRegions(regionA, regionB, visited, m)) {
          return false;
        }
      }
    }
  }

  return lhs.empty() && rhs.empty();
}

/// Compare two blocks for structural equivalence, allowing permutations
/// of independent operations.
static bool compareBlocks(Block& blockA, Block& blockB,
                          DenseSet<Operation*>& visited, IRMapping& m) {
  if (blockA.getNumArguments() != blockB.getNumArguments()) {
    return false;
  }

  for (auto [lArg, rArg] :
       llvm::zip_equal(blockA.getArguments(), blockB.getArguments())) {
    if (lArg.getType() != rArg.getType()) {
      return false;
    }

    m.map(lArg, rArg);
  }

  auto lDAGs = getDisjointSlices(blockA);
  auto rDAGs = getDisjointSlices(blockB);

  if (lDAGs.size() != rDAGs.size()) {
    return false;
  }

  for (const auto& [lDAG, rDAG] : llvm::zip_equal(lDAGs, rDAGs)) {
    if (lDAG.size() != rDAG.size() || !compareSlices(lDAG, rDAG, visited, m)) {
      return false;
    }
  }

  return true;
}

/// Compare two regions for structural equivalence.
static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& visited, IRMapping& m) {
  if (regionA.getBlocks().size() != regionB.getBlocks().size()) {
    return false;
  }

  for (const auto [blockA, blockB] : llvm::zip_equal(regionA, regionB)) {
    if (!compareBlocks(blockA, blockB, visited, m)) {
      return false;
    }
  }

  return true;
}

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  IRMapping m;
  DenseSet<Operation*> visited;

  return compareRegions(lhs.getBodyRegion(), rhs.getBodyRegion(), visited, m);
}
