/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"

#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"
#include "mqt/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>

using namespace mlir;

bool areModulesStructurallyEquivalent(ModuleOp lhs, ModuleOp rhs) {
  IRMapping mapping;
  bool consistent = true;
  const auto matchValue = [&](Value lhsValue, Value rhsValue) {
    if (auto mapped = mapping.lookupOrNull(lhsValue)) {
      return success(mapped == rhsValue);
    }
    mapping.map(lhsValue, rhsValue);
    return success();
  };
  /// A use can precede its definition in region order. Check that each later
  /// definition agrees with the correspondence established by its uses.
  const auto markValue = [&](Value lhsValue, Value rhsValue) {
    consistent &= succeeded(matchValue(lhsValue, rhsValue));
  };
  return OperationEquivalence::isEquivalentTo(
             lhs, rhs, matchValue, markValue,
             OperationEquivalence::IgnoreLocations |
                 OperationEquivalence::IgnoreCommutativity) &&
         consistent;
}

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

  /// Return true if the given lhs value takes part in the equivalence
  /// tracking. Only tensors reachable from a `qtensor` allocation are tracked;
  /// builtin tensors of qubits are compared through the regular SSA mapping.
  [[nodiscard]] bool tracksLhs(Value lhs) const {
    return lhsEquivGroups.contains(lhs);
  }

  /// Return true if the given rhs value takes part in the equivalence
  /// tracking.
  [[nodiscard]] bool tracksRhs(Value rhs) const {
    return rhsEquivGroups.contains(rhs);
  }
};
} // namespace

static bool compareRegions(Region& lhs, Region& rhs,
                           SetVector<Operation*>& lhsClosed,
                           SetVector<Operation*>& rhsClosed, IRMapping& m,
                           TensorMapping& tm);

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
      auto qubits = op.getQubits();
      const auto qIt = llvm::find(qubits, prev.tensor());
      assert(qIt != op.getQubits().end());
      const auto idx = std::distance(qubits.begin(), qIt);

      auto& thenRegion = op.getThenRegion();
      auto& elseRegion = op.getElseRegion();

      auto thenArg = thenRegion.getArgument(idx);
      auto elseArg = elseRegion.getArgument(idx);

      initEquivGroup(cast<TypedValue<RankedTensorType>>(thenArg), id, group);
      initEquivGroup(cast<TypedValue<RankedTensorType>>(elseArg), id, group);
    } else if (auto op = dyn_cast<qco::IndexSwitchOp>(it.operation())) {
      const auto prev = std::prev(it);
      auto targets = op.getTargets();
      const auto targetIt = llvm::find(targets, prev.tensor());
      assert(targetIt != targets.end());
      const auto idx = std::distance(targets.begin(), targetIt);

      for (Region* region : op.getRegions()) {
        initEquivGroup(
            cast<TypedValue<RankedTensorType>>(region->getArgument(idx)), id,
            group);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(it.operation())) {
      auto arg = forOp.getTiedLoopRegionIterArg(cast<OpResult>(it.tensor()));
      initEquivGroup(cast<TypedValue<RankedTensorType>>(arg), id, group);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(it.operation())) {
      const auto previous = std::prev(it);
      const auto init = llvm::find(whileOp.getInits(), previous.tensor());
      assert(init != whileOp.getInits().end());
      const auto initNumber = static_cast<unsigned>(
          std::distance(whileOp.getInits().begin(), init));
      initEquivGroup(cast<TypedValue<RankedTensorType>>(
                         whileOp.getBeforeBody()->getArgument(initNumber)),
                     id, group);

      auto result = cast<OpResult>(it.tensor());
      initEquivGroup(
          cast<TypedValue<RankedTensorType>>(
              whileOp.getAfterBody()->getArgument(result.getResultNumber())),
          id, group);
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
  for (auto [lhsResult, rhsResult] :
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
    const auto it = tm.tracksLhs(lhsValue)
                        ? llvm::find_if(rhs,
                                        [&](const auto rhsValue) {
                                          if (!tm.tracksRhs(rhsValue)) {
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
    if (tm.tracksLhs(lhsValue)) {
      const auto it = llvm::find_if(rhs, [&](const auto rhsValue) {
        return tm.tracksRhs(rhsValue) && tm.equals(lhsValue, rhsValue);
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

/// Compare values using the established SSA or tensor correspondence.
static bool compareValues(Value lhs, Value rhs, const IRMapping& mapping,
                          const TensorMapping& tensors) {
  if (tensors.tracksLhs(lhs)) {
    return tensors.tracksRhs(rhs) && tensors.equals(lhs, rhs);
  }
  return mapping.lookupOrNull(lhs) == rhs;
}

static bool compareOperations(Operation* lhs, Operation* rhs,
                              const IRMapping& m, const TensorMapping& tm) {

  // Compare top-level signature-like characteristics.

  if (lhs->getName() != rhs->getName() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getOperandTypes() != rhs->getOperandTypes() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      lhs->getResultTypes() != rhs->getResultTypes() ||
      lhs->getNumRegions() != rhs->getNumRegions() ||
      lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
      lhs->getAttrDictionary() != rhs->getAttrDictionary()) {
    return false;
  }

  for (auto [lhsSuccessor, rhsSuccessor] :
       llvm::zip_equal(lhs->getSuccessors(), rhs->getSuccessors())) {
    if (m.lookupOrNull(lhsSuccessor) != rhsSuccessor) {
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
    if (m.lookupOrNull(cast<qco::IfOp>(lhs).getCondition()) !=
            cast<qco::IfOp>(rhs).getCondition() ||
        !compareValueLists(cast<qco::IfOp>(lhs).getQubits(),
                           cast<qco::IfOp>(rhs).getQubits(), m, tm)) {
      return false;
    }
  } else if (isa<qco::IndexSwitchOp>(lhs)) {
    assert(isa<qco::IndexSwitchOp>(rhs));
    auto lhsSwitch = cast<qco::IndexSwitchOp>(lhs);
    auto rhsSwitch = cast<qco::IndexSwitchOp>(rhs);
    if (m.lookupOrNull(lhsSwitch.getArg()) != rhsSwitch.getArg() ||
        !compareValueLists(lhsSwitch.getTargets(), rhsSwitch.getTargets(), m,
                           tm)) {
      return false;
    }
  } else if (isa<qco::YieldOp>(lhs)) {
    /// Controls are the only parent results not supplied by qco.yield.
    auto parentResults =
        lhs->getParentOp()->getResults().take_back(lhs->getNumOperands());
    const auto rhsOffset =
        rhs->getParentOp()->getNumResults() - rhs->getNumOperands();
    for (auto [value, result] :
         llvm::zip_equal(lhs->getOperands(), parentResults)) {
      const auto position = cast<OpResult>(m.lookup(result)).getResultNumber();
      if (!compareValues(value, rhs->getOperand(position - rhsOffset), m, tm)) {
        return false;
      }
    }
  } else {
    for (auto [lhsOperand, rhsOperand] :
         llvm::zip_equal(lhs->getOperands(), rhs->getOperands())) {
      if (!compareValues(lhsOperand, rhsOperand, m, tm)) {
        return false;
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
  Operation* firstEffect = nullptr;
  bool blockedEffects = false;
  for (Operation* op : open) {
    if (ready.contains(op)) {
      continue;
    }

    /// SSA dependencies do not order writes to QC references or other memory.
    /// Fresh QC/CBit/QCO allocations and independently owned linear quantum
    /// disposal can commute; SSA dependencies preserve their lifetimes.
    /// Module symbols are definitions, not execution-order dependencies.
    if (!isMemoryEffectFree(op) &&
        !(isa<SymbolOpInterface>(op) && isa<ModuleOp>(op->getParentOp())) &&
        !isa<cbit::AllocOp, qc::AllocOp, qco::AllocOp, qco::SinkOp,
             qtensor::AllocOp, qtensor::DeallocOp>(op)) {
      if (firstEffect != nullptr) {
        if (blockedEffects ||
            !(isa<qc::DeallocOp>(firstEffect) && isa<qc::DeallocOp>(op))) {
          blockedEffects = true;
          continue;
        }
      } else {
        firstEffect = op;
      }
    }
    SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    if (!llvm::all_of(captures, isReady)) {
      continue;
    }

    if (isa<qtensor::InsertOp, qtensor::ExtractOp>(op)) {
      /// Accesses can commute only after their input tensor is available.
      /// Both operations thread the tensor through result zero and put the
      /// index last; insert also consumes a scalar before its tensor operand.
      const bool isInsert = isa<qtensor::InsertOp>(op);
      if (!isReady(op->getOperand(isInsert ? 1 : 0))) {
        continue;
      }
      llvm::SmallDenseSet<int64_t> indices;
      for (Operation* access = op;
           open.contains(access) && access->getName() == op->getName();
           access = *access->getResult(0).user_begin()) {
        auto indexValue = access->getOperands().back();
        auto index = getConstantIntValue(indexValue);
        /// ponytail: only distinct constant slots commute; use an alias proof
        /// if a future test needs to reorder dynamic accesses.
        if ((access != op && !index) ||
            (index && !indices.insert(*index).second)) {
          break;
        }
        if (isReady(indexValue) &&
            (!isInsert || isReady(access->getOperand(0)))) {
          ready.insert(access);
        }
        if (!index) {
          break;
        }
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
            // The threaded tensor is only covered by the equivalence groups
            // when it descends from an allocation, so map it here as well.
            m.map(lhsExtract.getOutTensor(), rhsExtract.getOutTensor());
          } else {
            m.map(lhsOp->getResults(), rhsOp->getResults());
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

    for (Operation* lhsOp : lhsReady) {
      Operation* rhsOp = m.lookup(lhsOp);
      for (auto [lhsRegion, rhsRegion] :
           llvm::zip_equal(lhsOp->getRegions(), rhsOp->getRegions())) {
        if (!compareRegions(lhsRegion, rhsRegion, lhsClosed, rhsClosed, m,
                            tm)) {
          return false;
        }
      }
    }
  }

  return lhsOpen.empty() && rhsOpen.empty();
}

/// Compare two regions for structural equivalence.
static bool compareRegions(Region& lhs, Region& rhs,
                           SetVector<Operation*>& lhsClosed,
                           SetVector<Operation*>& rhsClosed, IRMapping& m,
                           TensorMapping& tm) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  /// Map CFG destinations and block arguments before comparing operations.
  for (auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhs, rhs)) {
    if (lhsBlock.getArgumentTypes() != rhsBlock.getArgumentTypes()) {
      return false;
    }
    m.map(&lhsBlock, &rhsBlock);
    m.map(lhsBlock.getArguments(), rhsBlock.getArguments());
  }
  for (auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhs, rhs)) {
    if (!compareBlocks(lhsBlock, rhsBlock, lhsClosed, rhsClosed, m, tm)) {
      return false;
    }
  }

  return true;
}

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  if (areModulesStructurallyEquivalent(lhs, rhs)) {
    return true;
  }
  IRMapping m;
  SetVector<Operation*> lhsClosed;
  SetVector<Operation*> rhsClosed;
  TensorMapping tm{
      .lhsEquivGroups = getEquivGroup(lhs),
      .rhsEquivGroups = getEquivGroup(rhs),
      .equivGroupMapping = DenseMap<size_t, size_t>{},
  };

  return compareOperations(lhs, rhs, m, tm) &&
         compareRegions(lhs.getBodyRegion(), rhs.getBodyRegion(), lhsClosed,
                        rhsClosed, m, tm);
}
