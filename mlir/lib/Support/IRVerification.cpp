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

#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numeric>

using namespace mlir;

namespace {
struct TensorMapping {
  /// Maps all tensor values of the lhs to its equiv group.
  DenseMap<Value, size_t> lhsEquivGroups;
  /// Maps all tensor values of the rhs to its equiv group.
  DenseMap<Value, size_t> rhsEquivGroups;
  /// Maps the i-th group of lhs to the j-th group of rhs.
  DenseMap<size_t, size_t> equivGroupMapping;

  /// Map equivalence group identifiers of two tensors.
  void map(Value lhs, Value rhs) {
    equivGroupMapping[lhsEquivGroups[lhs]] = rhsEquivGroups[rhs];
  }

  /// Return true if the given tensor values have the same equiv group.
  [[nodiscard]] bool equals(Value lhs, Value rhs) const {
    const auto i = lhsEquivGroups.at(lhs);
    return equivGroupMapping.at(i) == rhsEquivGroups.at(rhs);
  }
};
} // namespace

static bool compareRegions(Region& lhs, Region& rhs,
                           SetVector<Operation*>& lhsClosed,
                           SetVector<Operation*>& rhsClosed, IRMapping& m,
                           TensorMapping& tm);

/// Return true, if the given value has the type `tensor<qco.qubit>`.
static bool hasTypeQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

/// Recursively initialize the equivalence group for a tensor value.
static void initEquivGroup(TypedValue<RankedTensorType> v, size_t id,
                           DenseMap<Value, size_t>& group) {
  for (qtensor::TensorIterator it(v); it != std::default_sentinel; ++it) {
    if (it.tensor() == nullptr) {
      continue;
    }

    group[it.tensor()] = id;

    if (isa<BlockArgument>(it.tensor())) {
      continue;
    }

    if (auto op = dyn_cast<qco::IfOp>(it.operation())) {
      const auto prev = std::prev(it);
      const auto qubits = op.getQubits();
      const auto qIt = llvm::find(qubits, prev.tensor());
      assert(qIt != op.getQubits().end());
      const auto idx = std::distance(qubits.begin(), qIt);

      auto& thenRegion = op.getThenRegion();
      auto& elseRegion = op.getElseRegion();

      const auto& thenArg = thenRegion.getArgument(idx);
      const auto& elseArg = elseRegion.getArgument(idx);

      initEquivGroup(cast<TypedValue<RankedTensorType>>(thenArg), id, group);
      initEquivGroup(cast<TypedValue<RankedTensorType>>(elseArg), id, group);
    } else if (auto op = dyn_cast<qco::IndexSwitchOp>(it.operation())) {
      const auto prev = std::prev(it);
      const auto targets = op.getTargets();
      const auto targetIt = llvm::find(targets, prev.tensor());
      assert(targetIt != targets.end());
      const auto idx = std::distance(targets.begin(), targetIt);

      for (Region* region : op.getRegions()) {
        initEquivGroup(
            cast<TypedValue<RankedTensorType>>(region->getArgument(idx)), id,
            group);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(it.operation())) {
      const auto& arg =
          forOp.getTiedLoopRegionIterArg(cast<OpResult>(it.tensor()));
      initEquivGroup(cast<TypedValue<RankedTensorType>>(arg), id, group);
    }
  }
}

/// Generate equivalence group for all allocated and created tensors.
static DenseMap<Value, size_t> getEquivGroup(ModuleOp mod) {
  size_t id = 0;
  DenseMap<Value, size_t> group;

  mod->walk([&](Operation* op) {
    if (auto alloc = dyn_cast<qtensor::AllocOp>(op)) {
      initEquivGroup(alloc.getResult(), id, group);
      ++id;
    } else if (auto from = dyn_cast<qtensor::FromElementsOp>(op)) {
      initEquivGroup(cast<TypedValue<RankedTensorType>>(from.getResult()), id,
                     group);
      ++id;
    }
  });

  return group;
}

/// Map all results from one op to another using the given permutation.
/// Assumes that `lhs->getNumResults() == rhs->getNumResults()`.
/// Assumes that the two operations are equivalent to each other.
static void mapResults(Operation* lhs, Operation* rhs,
                       ArrayRef<size_t> permutation, IRMapping& m) {
  for (const auto& [i, lhsResult] : llvm::enumerate(lhs->getResults())) {
    m.map(lhsResult, rhs->getResult(permutation[i]));
  }
}

/// Map a classical result prefix positionally and a linear result suffix using
/// the given permutation.
static void mapSegmentedResults(ValueRange lhsClassical,
                                ValueRange rhsClassical, ValueRange lhsLinear,
                                ValueRange rhsLinear,
                                ArrayRef<size_t> linearPermutation,
                                IRMapping& mapping) {
  for (const auto [lhsResult, rhsResult] :
       llvm::zip_equal(lhsClassical, rhsClassical)) {
    mapping.map(lhsResult, rhsResult);
  }
  for (const auto [index, lhsResult] : llvm::enumerate(lhsLinear)) {
    mapping.map(lhsResult, rhsLinear[linearPermutation[index]]);
  }
}

/// Map arguments from one block to another using the given permutation.
/// Assumes that `lhs.getNumArguments() == rhs.getNumArguments()`.
/// Assumes that `permutation.size() == lhs.getNumArguments()`.
static void mapArguments(Block& lhs, Block& rhs, ArrayRef<size_t> permutation,
                         IRMapping& m) {
  for (const auto& [i, lhsArg] : enumerate(lhs.getArguments())) {
    m.map(lhsArg, rhs.getArgument(permutation[i]));
  }
}

/// Return a permutation vector, where permutation[i] maps the i-th value of the
/// lhs range to the j-th value of the rhs range.
template <typename LhsRange, typename RhsRange>
static FailureOr<SmallVector<size_t>>
getPermutation(const LhsRange& lhs, const RhsRange& rhs, const IRMapping& m,
               const TensorMapping& tm) {
  SmallVector<size_t> permutation(lhs.size());
  for (const auto& [i, lhsValue] : llvm::enumerate(lhs)) {
    const auto it = hasTypeQubitTensor(lhsValue)
                        ? llvm::find_if(rhs,
                                        [&](const auto rhsValue) {
                                          if (!hasTypeQubitTensor(rhsValue)) {
                                            return false;
                                          }
                                          return tm.equals(lhsValue, rhsValue);
                                        })
                        : llvm::find(rhs, m.lookup(lhsValue));
    if (it == rhs.end()) {
      return failure();
    }
    const auto j = std::distance(rhs.begin(), it);
    permutation[i] = j;
  }
  return permutation;
}

/// Compare two value lists, allowing permutations.
template <typename LhsRange, typename RhsRange>
static bool compareValueLists(const LhsRange& lhs, const RhsRange& rhs,
                              const IRMapping& m, const TensorMapping& tm) {
  DenseSet<Value> workset;
  workset.insert_range(rhs);

  for (const auto lhsValue : lhs) {
    Value mapped;
    if (hasTypeQubitTensor(lhsValue)) {
      const auto it = llvm::find_if(rhs, [&](const auto rhsValue) {
        return hasTypeQubitTensor(rhsValue) && tm.equals(lhsValue, rhsValue);
      });
      if (it == rhs.end()) {
        return false;
      }
      mapped = *it;
    } else {
      mapped = m.lookup(lhsValue);
    }
    if (!workset.contains(mapped)) {
      return false;
    }
    workset.erase(mapped);
  }

  return workset.empty();
}

/// Compare two floating point numbers for approximate equivalence.
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

/// Compare two attributes for equivality.
/// Explicitly checks `UnitAttr`, `IntegerAttr`, `FloatAttr`, `StringAttr`,
/// and `FlatSymbolRefAttr`. For any other type, the function simply returns
/// true.
static bool compareAttributes(Attribute lhs, Attribute rhs) {
  if (dyn_cast<UnitAttr>(lhs)) {
    if (!dyn_cast<UnitAttr>(rhs)) {
      return false;
    }
  } else if (auto intAttrA = dyn_cast<IntegerAttr>(lhs)) {
    if (auto intAttrB = dyn_cast<IntegerAttr>(rhs);
        !intAttrB || intAttrA.getValue() != intAttrB.getValue() ||
        (intAttrA.getType().isInteger() && !intAttrB.getType().isInteger())) {
      return false;
    }
  } else if (auto floatAttrA = dyn_cast<FloatAttr>(lhs)) {
    if (auto floatAttrB = dyn_cast<FloatAttr>(rhs);
        !floatAttrB ||
        !approxCompareFloats(floatAttrA.getValue(), floatAttrB.getValue(),
                             floatAttrA.getType().getIntOrFloatBitWidth())) {
      return false;
    }
  } else if (auto strAttrA = dyn_cast<StringAttr>(lhs)) {
    if (auto strAttrB = dyn_cast<StringAttr>(rhs);
        !strAttrB || strAttrA.getValue() != strAttrB.getValue()) {
      return false;
    }
  } else if (auto arrayAttrA = llvm::dyn_cast<ArrayAttr>(lhs)) {
    auto arrayAttrB = llvm::dyn_cast<ArrayAttr>(rhs);
    if (!arrayAttrB) {
      return false;
    }
    if (arrayAttrA.size() != arrayAttrB.size()) {
      return false;
    }

    for (const auto& [subAttrA, subAttrB] :
         llvm::zip_equal(arrayAttrA, arrayAttrB)) {
      if (!compareAttributes(subAttrA, subAttrB)) {
        return false;
      }
    }

  } else if (auto symbolRefAttrA = dyn_cast<FlatSymbolRefAttr>(lhs)) {
    auto symbolRefAttrB = dyn_cast<FlatSymbolRefAttr>(rhs);
    if (!symbolRefAttrB) {
      return false;
    }

    if (symbolRefAttrA.getValue() != symbolRefAttrB.getValue()) {
      return false;
    }
  } else if (auto tailCallAttrA = dyn_cast<LLVM::TailCallKindAttr>(lhs)) {
    auto tailCallAttrB = dyn_cast<LLVM::TailCallKindAttr>(rhs);
    if (!tailCallAttrB) {
      return false;
    }

    if (tailCallAttrA.getTailCallKind() != tailCallAttrB.getTailCallKind()) {
      return false;
    }
  } else if (auto fastMathAttrA = dyn_cast<LLVM::FastmathFlagsAttr>(lhs)) {
    auto fastMathAttrB = dyn_cast<LLVM::FastmathFlagsAttr>(rhs);
    if (!fastMathAttrB) {
      return false;
    }

    if (fastMathAttrA.getValue() != fastMathAttrB.getValue()) {
      return false;
    }
  } else if (auto cconvAttrA = dyn_cast<LLVM::CConvAttr>(lhs)) {
    auto cconvAttrB = dyn_cast<LLVM::CConvAttr>(rhs);
    if (!cconvAttrB) {
      return false;
    }

    if (cconvAttrA.getCallingConv() != cconvAttrB.getCallingConv()) {
      return false;
    }
  } else if (auto modFlagAttrA = dyn_cast<LLVM::ModuleFlagAttr>(lhs)) {
    auto modFlagAttrB = dyn_cast<LLVM::ModuleFlagAttr>(rhs);
    if (!modFlagAttrB) {
      return false;
    }

    if (modFlagAttrA.getBehavior() != modFlagAttrB.getBehavior() ||
        modFlagAttrA.getKey() != modFlagAttrB.getKey() ||
        modFlagAttrA.getValue() != modFlagAttrB.getValue()) {
      return false;
    }
  }

  return true;
}

/// Compare two operations for structural equivalence, applying special
/// rules for `CtrlOp` s and `qtensor` s.
static bool compareOperations(Operation* lhs, Operation* rhs,
                              const IRMapping& m, const TensorMapping& tm) {

  // Compare top-level signature-like characteristics.

  if (lhs->getName() != rhs->getName() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getOperandTypes() != rhs->getOperandTypes() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      lhs->getResultTypes() != rhs->getResultTypes() ||
      lhs->getNumRegions() != rhs->getNumRegions()) {
    return false;
  }

  // Compare attributes with specific types.
  // Silently ignore missing ones.

  for (const auto& namedAttrLhs : lhs->getAttrs()) {
    const StringRef keyLhs = namedAttrLhs.getName().strref();
    if (!rhs->hasAttr(keyLhs)) {
      continue;
    }

    if (!compareAttributes(namedAttrLhs.getValue(), rhs->getAttr(keyLhs))) {
      return false;
    }
  }

  // Compare operands.
  // Because the order of target (control) qubits of CtrlOps doesn't matter,
  // explicitly handle them here.

  if (isa<qc::CtrlOp>(lhs)) {
    assert(isa<qc::CtrlOp>(rhs));
    auto lhsCtrl = cast<qc::CtrlOp>(lhs);
    auto rhsCtrl = cast<qc::CtrlOp>(rhs);
    if (!compareValueLists(lhsCtrl.getControls(), rhsCtrl.getControls(), m,
                           tm) ||
        !compareValueLists(lhsCtrl.getTargets(), rhsCtrl.getTargets(), m, tm)) {
      return false;
    }
  } else if (isa<qco::CtrlOp>(lhs)) {
    assert(isa<qco::CtrlOp>(rhs));
    auto lhsCtrl = cast<qco::CtrlOp>(lhs);
    auto rhsCtrl = cast<qco::CtrlOp>(rhs);
    if (!compareValueLists(lhsCtrl.getInputControls(),
                           rhsCtrl.getInputControls(), m, tm) ||
        !compareValueLists(lhsCtrl.getInputTargets(), rhsCtrl.getInputTargets(),
                           m, tm)) {
      return false;
    }
  } else if (isa<qco::IfOp>(lhs)) {
    assert(isa<qco::IfOp>(rhs));
    if (!compareValueLists(cast<qco::IfOp>(lhs).getQubits(),
                           cast<qco::IfOp>(rhs).getQubits(), m, tm)) {
      return false;
    }
  } else if (isa<qco::IndexSwitchOp>(lhs)) {
    assert(isa<qco::IndexSwitchOp>(rhs));
    auto lhsSwitch = cast<qco::IndexSwitchOp>(lhs);
    auto rhsSwitch = cast<qco::IndexSwitchOp>(rhs);
    if (lhsSwitch.getCases() != rhsSwitch.getCases() ||
        !compareValueLists(lhsSwitch.getTargets(), rhsSwitch.getTargets(), m,
                           tm)) {
      return false;
    }
  } else if (isa<qco::YieldOp>(lhs)) {
    assert(isa<qco::YieldOp>(rhs));
    auto lhsYield = cast<qco::YieldOp>(lhs);
    auto rhsYield = cast<qco::YieldOp>(rhs);

    size_t numClassicalResults = 0;
    if (auto ifOp = dyn_cast<qco::IfOp>(lhs->getParentOp())) {
      numClassicalResults = ifOp.getClassicalResults().size();
    } else if (auto switchOp =
                   dyn_cast<qco::IndexSwitchOp>(lhs->getParentOp())) {
      numClassicalResults = switchOp.getClassicalResults().size();
    }

    for (const auto [lhsValue, rhsValue] : llvm::zip_equal(
             lhsYield.getTargets().take_front(numClassicalResults),
             rhsYield.getTargets().take_front(numClassicalResults))) {
      if (m.lookup(lhsValue) != rhsValue) {
        return false;
      }
    }
    if (!compareValueLists(
            lhsYield.getTargets().drop_front(numClassicalResults),
            rhsYield.getTargets().drop_front(numClassicalResults), m, tm)) {
      return false;
    }
  } else {
    for (const auto& [lhsOperand, rhsOperand] :
         llvm::zip_equal(lhs->getOperands(), rhs->getOperands())) {
      if (hasTypeQubitTensor(lhsOperand)) {
        assert(hasTypeQubitTensor(rhsOperand));

        if (!tm.equals(lhsOperand, rhsOperand)) {
          return false;
        }
      } else {
        const auto& v = m.lookup(lhsOperand);
        if (v != rhsOperand) {
          return false;
        }
      }
    }
  }

  return true;
}

/// Extract and return "ready" operations.
/// These are operations that are independent from each other.
static SetVector<Operation*> getReadyOps(const SetVector<Operation*>& open,
                                         const SetVector<Operation*>& closed) {
  const auto isReady = [&closed](Value v) {
    if (isa<BlockArgument>(v)) {
      return true;
    }
    return closed.contains(v.getDefiningOp());
  };

  SetVector<Operation*> ready;
  for (Operation* op : open) {
    if (ready.contains(op)) {
      continue;
    }

    if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {

      // If any of the inserts on the chain are ready, we consider the entire
      // chain ready because the ready operations could be moved to the front
      // of the chain. The analogous logic is applied to extracts.

      SmallVector<Operation*> chain;
      for (qtensor::TensorIterator it(insert.getResult());
           it != std::default_sentinel; ++it) {
        auto chainInsert = dyn_cast<qtensor::InsertOp>(it.operation());
        if (!chainInsert) {
          break;
        }
        if (isReady(chainInsert.getScalar()) &&
            isReady(chainInsert.getIndex()) && !closed.contains(chainInsert)) {
          chain.emplace_back(chainInsert);
        }
      }

      if (!chain.empty()) {
        ready.insert_range(chain);
      }

    } else if (auto extract = dyn_cast<qtensor::ExtractOp>(op)) {
      SmallVector<Operation*> chain;
      for (qtensor::TensorIterator it(extract.getOutTensor());
           it != std::default_sentinel; ++it) {
        auto chainExtract = dyn_cast<qtensor::ExtractOp>(it.operation());
        if (!chainExtract) {
          break;
        }

        if (isReady(chainExtract.getIndex()) &&
            !closed.contains(chainExtract)) {
          chain.emplace_back(chainExtract);
        }
      }

      if (!chain.empty()) {
        ready.insert_range(chain);
      }
    } else if (auto dealloc = dyn_cast<qtensor::DeallocOp>(op)) {

      // Deallocations are ready whenever we've visited each op on the tensor
      // chain. Because we initialize the iterator with its input tensor, the
      // iterator already points at the previous operation. Thus use a
      // do-while loop instead of a regular while.

      bool fullChain{true};
      qtensor::TensorIterator it(dealloc.getTensor());

      do {
        if (!closed.contains(it.operation())) {
          fullChain = false;
          break;
        }

        --it;
      } while (std::prev(it) != it);

      if (fullChain) {
        ready.insert(dealloc);
      }

    } else {

      // Otherwise, simply check if all operands are ready.

      if (llvm::all_of(op->getOperands(), isReady)) {
        ready.insert(op);
      }
    }
  }

  return ready;
}

static bool compareBlocks(Block& lhs, Block& rhs,
                          SetVector<Operation*>& lhsClosed,
                          SetVector<Operation*>& rhsClosed, IRMapping& m,
                          TensorMapping& tm) {
  if (lhs.getNumArguments() != rhs.getNumArguments()) {
    return false;
  }

  // Map block arguments while allowing commutation of operands for `CtrlOp`s.

  if (isa<qc::CtrlOp>(lhs.getParentOp())) {
    assert(isa<qc::CtrlOp>(rhs.getParentOp()));
    auto lhsCtrl = cast<qc::CtrlOp>(lhs.getParentOp());
    auto rhsCtrl = cast<qc::CtrlOp>(rhs.getParentOp());
    const auto permutation =
        getPermutation(lhsCtrl.getTargets(), rhsCtrl.getTargets(), m, tm);
    if (failed(permutation)) {
      return false;
    }
    mapArguments(lhs, rhs, *permutation, m);
  } else if (isa<qco::CtrlOp>(lhs.getParentOp())) {
    assert(isa<qco::CtrlOp>(rhs.getParentOp()));
    auto lhsCtrl = cast<qco::CtrlOp>(lhs.getParentOp());
    auto rhsCtrl = cast<qco::CtrlOp>(rhs.getParentOp());
    const auto permutation = getPermutation(lhsCtrl.getInputTargets(),
                                            rhsCtrl.getInputTargets(), m, tm);
    if (failed(permutation)) {
      return false;
    }
    mapArguments(lhs, rhs, *permutation, m);
  } else if (isa<qco::IfOp>(lhs.getParentOp())) {
    assert(isa<qco::IfOp>(rhs.getParentOp()));
    auto lhsIf = cast<qco::IfOp>(lhs.getParentOp());
    auto rhsIf = cast<qco::IfOp>(rhs.getParentOp());
    const auto permutation =
        getPermutation(lhsIf.getQubits(), rhsIf.getQubits(), m, tm);
    if (failed(permutation)) {
      return false;
    }
    mapArguments(lhs, rhs, *permutation, m);
  } else if (isa<qco::IndexSwitchOp>(lhs.getParentOp())) {
    assert(isa<qco::IndexSwitchOp>(rhs.getParentOp()));
    auto lhsSwitch = cast<qco::IndexSwitchOp>(lhs.getParentOp());
    auto rhsSwitch = cast<qco::IndexSwitchOp>(rhs.getParentOp());
    const auto permutation =
        getPermutation(lhsSwitch.getTargets(), rhsSwitch.getTargets(), m, tm);
    if (failed(permutation)) {
      return false;
    }
    mapArguments(lhs, rhs, *permutation, m);
  } else {
    SmallVector<size_t> permutation(lhs.getNumArguments());
    std::iota(permutation.begin(), permutation.end(), 0);
    mapArguments(lhs, rhs, permutation, m);
  }

  SetVector<Operation*> lhsOpen;
  SetVector<Operation*> rhsOpen;

  for_each(lhs.getOperations(), [&](auto& op) { lhsOpen.insert(&op); });
  for_each(rhs.getOperations(), [&](auto& op) { rhsOpen.insert(&op); });

  // Compare block operations topologically.

  while (true) {
    const auto lhsReady = getReadyOps(lhsOpen, lhsClosed);
    const auto rhsReady = getReadyOps(rhsOpen, rhsClosed);

    if (lhsReady.empty() && rhsReady.empty()) {
      break;
    }

    if (lhsReady.size() != rhsReady.size()) {
      return false;
    }

    // Because there may be multiple structural equivalent operations (think
    // arith.constant, for example), we apply the assumption that the first
    // occurrence on the lhs corresponds to the first one on the rhs, etc.

    DenseSet<Operation*> matched;
    matched.reserve(rhsReady.size());

    for (Operation* lhsOp : lhsReady) {
      SetVector<Operation*>::iterator it = rhsReady.begin();
      for (; it != rhsReady.end(); it = std::next(it)) {
        Operation* rhsOp = *it;

        if (matched.contains(rhsOp)) {
          continue;
        }

        if (compareOperations(lhsOp, rhsOp, m, tm)) {
          matched.insert(rhsOp);

          if (isa<qco::CtrlOp>(lhsOp)) {
            assert(isa<qco::CtrlOp>(rhsOp));
            auto lhsCtrl = cast<qco::CtrlOp>(lhsOp);
            auto rhsCtrl = cast<qco::CtrlOp>(rhsOp);

            const auto controlPermutation = getPermutation(
                lhsCtrl.getInputControls(), rhsCtrl.getInputControls(), m, tm);
            const auto targetPermutation = getPermutation(
                lhsCtrl.getInputTargets(), rhsCtrl.getInputTargets(), m, tm);
            if (failed(controlPermutation) || failed(targetPermutation)) {
              return false;
            }

            SmallVector<size_t> permutation(*controlPermutation);
            permutation.reserve(lhsCtrl.getNumQubits());
            for (const auto i : *targetPermutation) {
              permutation.emplace_back(lhsCtrl.getNumControls() + i);
            }
            mapResults(lhsCtrl, rhsCtrl, permutation, m);
          } else if (isa<qco::IfOp>(lhsOp)) {
            assert(isa<qco::IfOp>(rhsOp));
            auto lhsIf = cast<qco::IfOp>(lhsOp);
            auto rhsIf = cast<qco::IfOp>(rhsOp);
            const auto permutation =
                getPermutation(lhsIf.getQubits(), rhsIf.getQubits(), m, tm);
            if (failed(permutation)) {
              return false;
            }
            mapSegmentedResults(lhsIf.getClassicalResults(),
                                rhsIf.getClassicalResults(),
                                lhsIf.getLinearResults(),
                                rhsIf.getLinearResults(), *permutation, m);
          } else if (isa<qco::IndexSwitchOp>(lhsOp)) {
            assert(isa<qco::IndexSwitchOp>(rhsOp));
            auto lhsSwitch = cast<qco::IndexSwitchOp>(lhsOp);
            auto rhsSwitch = cast<qco::IndexSwitchOp>(rhsOp);
            const auto permutation = getPermutation(
                lhsSwitch.getTargets(), rhsSwitch.getTargets(), m, tm);
            if (failed(permutation)) {
              return false;
            }
            mapSegmentedResults(lhsSwitch.getClassicalResults(),
                                rhsSwitch.getClassicalResults(),
                                lhsSwitch.getLinearResults(),
                                rhsSwitch.getLinearResults(), *permutation, m);
          } else if (isa<qtensor::AllocOp>(lhsOp)) {
            assert(isa<qtensor::AllocOp>(rhsOp));
            auto lhsAlloc = cast<qtensor::AllocOp>(lhsOp);
            auto rhsAlloc = cast<qtensor::AllocOp>(rhsOp);
            tm.map(lhsAlloc.getResult(), rhsAlloc.getResult());
          } else if (isa<qtensor::FromElementsOp>(lhsOp)) {
            assert(isa<qtensor::FromElementsOp>(rhsOp));
            auto lhsFrom = cast<qtensor::FromElementsOp>(lhsOp);
            auto rhsFrom = cast<qtensor::FromElementsOp>(rhsOp);
            tm.map(lhsFrom.getResult(), rhsFrom.getResult());
          } else if (isa<qtensor::ExtractOp>(lhsOp)) {
            assert(isa<qtensor::ExtractOp>(rhsOp));
            auto lhsExtract = cast<qtensor::ExtractOp>(lhsOp);
            auto rhsExtract = cast<qtensor::ExtractOp>(rhsOp);
            m.map(lhsExtract.getResult(), rhsExtract.getResult());
          } else {
            SmallVector<size_t> permutation(lhsOp->getNumResults());
            std::iota(permutation.begin(), permutation.end(), 0);
            mapResults(lhsOp, rhsOp, permutation, m);
          }

          m.map(lhsOp, rhsOp);
          break;
        }
      }

      if (it == rhsReady.end()) {
        return false;
      }
    }

    // At this point, we've successfully matched each operation on the lhs
    // with one on the rhs. Subsequently, update the open and closed sets and
    // recursively compare the nested regions of each operation pair.

    lhsOpen.set_subtract(lhsReady);
    lhsClosed.set_union(lhsReady);

    rhsOpen.set_subtract(rhsReady);
    rhsClosed.set_union(rhsReady);

    SetVector<Operation*>::iterator it = lhsReady.begin();
    for (; it != lhsReady.end(); it = std::next(it)) {
      Operation* opLhs = *it;

      if (opLhs->getNumRegions() > 0) {
        Operation* opRhs = m.lookup(opLhs);
        assert(opLhs->getNumRegions() == opRhs->getNumRegions());
        const auto nequiv = range_size(make_filter_range(
            llvm::zip_equal(opLhs->getRegions(), opRhs->getRegions()),
            [&](const auto& zip) {
              const auto& [lhsRegion, rhsRegion] = zip;
              return compareRegions(lhsRegion, rhsRegion, lhsClosed, rhsClosed,
                                    m, tm);
            }));
        if (nequiv != opLhs->getNumRegions()) {
          break;
        }
      }
    }

    if (it != lhsReady.end()) {
      return false;
    }
  }

  return true;
}

/// Compare two regions for structural equivalence.
static bool compareRegions(Region& lhs, Region& rhs,
                           SetVector<Operation*>& lhsClosed,
                           SetVector<Operation*>& rhsClosed, IRMapping& m,
                           TensorMapping& tm) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  for (const auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhs, rhs)) {
    if (!compareBlocks(lhsBlock, rhsBlock, lhsClosed, rhsClosed, m, tm)) {
      return false;
    }

    m.map(&lhsBlock, &rhsBlock);
  }

  return true;
}

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  IRMapping m;
  SetVector<Operation*> lhsClosed;
  SetVector<Operation*> rhsClosed;
  TensorMapping tm{.lhsEquivGroups = getEquivGroup(lhs),
                   .rhsEquivGroups = getEquivGroup(rhs),
                   .equivGroupMapping = DenseMap<size_t, size_t>{}};

  return compareRegions(lhs.getBodyRegion(), rhs.getBodyRegion(), lhsClosed,
                        rhsClosed, m, tm);
}
