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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
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

static bool areFloatValuesNear(const APFloat& lhs, const APFloat& rhs,
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

static bool areConstantAttributesEquivalent(const Attribute& lhs,
                                            const Attribute& rhs) {
  if (lhs == rhs) {
    return true;
  }

  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    auto rhsFloat = dyn_cast<FloatAttr>(rhs);
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
  if (auto lhsConst = dyn_cast<arith::ConstantOp>(lhs)) {
    auto rhsConst = dyn_cast<arith::ConstantOp>(rhs);
    if (!rhsConst) {
      return false;
    }

    if (!areConstantAttributesEquivalent(lhsConst.getValue(),
                                         rhsConst.getValue())) {
      return false;
    }
  }

  // Check LLVM::ConstantOp
  if (auto lhsConst = dyn_cast<LLVM::ConstantOp>(lhs)) {
    auto rhsConst = dyn_cast<LLVM::ConstantOp>(rhs);
    if (!rhsConst) {
      return false;
    }
    if (!areConstantAttributesEquivalent(lhsConst.getValue(),
                                         rhsConst.getValue())) {
      return false;
    }
  }

  // Check LLVM::CallOp
  if (auto lhsCall = dyn_cast<LLVM::CallOp>(lhs)) {
    auto rhsCall = dyn_cast<LLVM::CallOp>(rhs);
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
       llvm::zip_equal(lhs->getOperands(), rhs->getOperands())) {
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

/// Check if an operation has memory effects or control flow side effects
/// that would prevent reordering.
static bool hasOrderingConstraints(Operation* op) {
  // Terminators must maintain their position
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return true;
  }

  // Symbol-defining operations (like function declarations) can be reordered
  if (op->hasTrait<OpTrait::SymbolTable>() ||
      isa<LLVM::LLVMFuncOp, func::FuncOp>(op)) {
    return false;
  }

  // Check for memory effects that enforce ordering
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);

    bool hasNonAllocFreeEffects = false;
    for (const auto& effect : effects) {
      // Allow operations with no effects or pure allocation/free effects
      if (!isa<MemoryEffects::Allocate, MemoryEffects::Free>(
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

static bool areTopLevelEquivalent(Operation* lhs, Operation* rhs) {
  return lhs->getName() == rhs->getName() &&
         lhs->getNumOperands() == rhs->getNumOperands() &&
         lhs->getOperandTypes() == rhs->getOperandTypes() &&
         lhs->getNumResults() == rhs->getNumResults() &&
         lhs->getResultTypes() == rhs->getResultTypes() &&
         lhs->getAttrs().size() == rhs->getAttrs().size() &&
         lhs->getNumRegions() == rhs->getNumRegions();

  
}

/// Extract and return "ready" operations from the slice.
static Slice getReadyOps(Slice& slice, DenseSet<Operation*> finished) {
  const auto isReady = [&](OpOperand& operand) {
    if (isa<BlockArgument>(operand.get())) {
      return true; // The defining op is finished?
    }
    return finished.contains(operand.get().getDefiningOp());
  };

  Slice ready;
  for (Operation* op : slice) {
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
        }
      }
    }

    // Analogously for the extract operation.

    if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      Operation* prev = extract.getTensor().getDefiningOp();
      if (isa<qtensor::ExtractOp>(prev)) {
        if (ready.contains(prev)) {
          ready.insert(extract.getOperation());
        }
      }
    }
  }

  return ready;
}

static bool areEquivalent(Region& lhs, Region& rhs, IRMapping& m);

static bool areEquivalent(Slice& lhs, Slice& rhs, IRMapping& m) {
  DenseSet<Operation*> finished;
  finished.reserve(lhs.size() + rhs.size());

  while (true) {
    const auto readyLhs = getReadyOps(lhs, finished);
    const auto readyRhs = getReadyOps(rhs, finished);

    if (readyLhs.empty() || readyRhs.empty()) {
      break;
    }

    if (readyLhs.size() != readyRhs.size()) {
      return false;
    }

    // Greedily search for a structural equivalent operation.
    for (Operation* opLhs : readyLhs) {
      for (Operation* opRhs : readyRhs) {
        if (areTopLevelEquivalent(opLhs, opRhs)) { // TODO: Full equivalence check with attrs etc. 
          m.map(opLhs, opRhs); // Map operations.
          // Map operands.
          // Map results.

          llvm::dbgs() << opLhs->getName() << " == " << opRhs->getName()
                       << '\n';
        }
      }
    }

    // for (Operation* op : readyLhs) {
    //   op->dumpPretty();
    //   for (Region& region : op->getRegions()) {
    //     IRMapping m;
    //     if (!areEquivalent(region, region, m)) {
    //       return false;
    //     }
    //   }
    // }

    finished.insert_range(readyLhs);
    lhs.set_subtract(readyLhs);

    finished.insert_range(readyRhs);
    rhs.set_subtract(readyRhs);
  }

  return lhs.empty() && rhs.empty();
}

/// Compare two blocks for structural equivalence, allowing permutations
/// of independent operations.
static bool areEquivalent(Block& lhs, Block& rhs, IRMapping& m) {
  if (lhs.getNumArguments() != rhs.getNumArguments()) {
    return false;
  }

  for (auto [lArg, rArg] :
       llvm::zip_equal(lhs.getArguments(), rhs.getArguments())) {
    if (lArg.getType() != rArg.getType()) {
      return false;
    }

    m.map(lArg, rArg);
  }

  auto lDAGs = getDisjointSlices(lhs);
  auto rDAGs = getDisjointSlices(rhs);

  if (lDAGs.size() != rDAGs.size()) {
    return false;
  }

  for (const auto& [lDAG, rDAG] : llvm::zip_equal(lDAGs, rDAGs)) {
    if (lDAG.size() != rDAG.size() || !areEquivalent(lDAG, rDAG, m)) {
      return false;
    }
  }

  return true;
}

/// Compare two regions for structural equivalence.
static bool areEquivalent(Region& lhs, Region& rhs, IRMapping& m) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  for (auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhs, rhs)) {
    if (!areEquivalent(lhsBlock, rhsBlock, m)) {
      return false;
    }
  }

  return true;
}

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  IRMapping m;
  return areEquivalent(lhs.getBodyRegion(), rhs.getBodyRegion(), m);
}
