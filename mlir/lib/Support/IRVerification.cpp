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
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Analysis/SliceAnalysis.h>
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
      const auto qIt = llvm::find(op.getQubits(), prev.tensor());
      assert(qIt != op.getQubits().end());
      const auto idx = std::distance(op.getQubits().begin(), qIt);

      auto& thenRegion = op.getThenRegion();
      auto& elseRegion = op.getElseRegion();

      const auto& thenArg = thenRegion.getArgument(idx);
      const auto& elseArg = elseRegion.getArgument(idx);

      initEquivGroup(cast<TypedValue<RankedTensorType>>(thenArg), id, group);
      initEquivGroup(cast<TypedValue<RankedTensorType>>(elseArg), id, group);
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

/// Map arguments from one block to another using the given permutation.
/// Assumes that `lhs.getNumArguments() == rhs.getNumArguments()`.
/// Assumes that `permutation.size() == lhs.getNumArguments()`.
static void mapArguments(Block& lhs, Block& rhs, ArrayRef<size_t> permutation,
                         IRMapping& m) {

  for (const auto& [i, lhsArg] : enumerate(lhs.getArguments())) {
    m.map(lhsArg, rhs.getArgument(permutation[i]));
  }
}

/// Return a permutation vector, where permutation[i] maps the i-th target of
/// the lhs to the j-th target of the rhs.
static SmallVector<size_t> getTargetPermutation(qc::CtrlOp lhs, qc::CtrlOp rhs,
                                                const IRMapping& m) {
  SmallVector<size_t> permutation(lhs.getNumTargets());
  for (const auto& [i, trgt] : llvm::enumerate(lhs.getTargets())) {
    const auto it = llvm::find(rhs.getTargets(), m.lookup(trgt));
    const auto j = std::distance(rhs.getTargets().begin(), it);
    permutation[i] = j;
  }
  return permutation;
}

/// Return a permutation vector, where permutation[i] maps the i-th input
/// target of the lhs to the j-th input target of the rhs.
static SmallVector<size_t>
getTargetPermutation(qco::CtrlOp lhs, qco::CtrlOp rhs, const IRMapping& m) {
  SmallVector<size_t> permutation(lhs.getNumTargets());
  for (const auto& [i, trgt] : llvm::enumerate(lhs.getInputTargets())) {
    const auto it = llvm::find(rhs.getInputTargets(), m.lookup(trgt));
    const auto j = std::distance(rhs.getInputTargets().begin(), it);
    permutation[i] = j;
  }
  return permutation;
}

/// Return a permutation vector, where permutation[i] maps the i-th input
/// target of the lhs to the j-th input target of the rhs.
static SmallVector<size_t>
getControlPermutation(qco::CtrlOp lhs, qco::CtrlOp rhs, const IRMapping& m) {
  SmallVector<size_t> permutation(lhs.getNumControls());
  for (const auto& [i, trgt] : llvm::enumerate(lhs.getInputControls())) {
    const auto it = llvm::find(rhs.getInputControls(), m.lookup(trgt));
    const auto j = std::distance(rhs.getInputControls().begin(), it);
    permutation[i] = j;
  }
  return permutation;
}

/// Compare two ctrl operations, allowing permutations of control and target
/// qubits.
static bool compareCtrlOps(qc::CtrlOp lhs, qc::CtrlOp rhs, const IRMapping& m) {
  DenseSet<Value> workset;
  workset.insert_range(rhs.getControls());
  for (const auto& ctrl : lhs.getControls()) {
    const auto& v = m.lookup(ctrl);
    if (!workset.contains(v)) {
      return false;
    }
    workset.erase(v);
  }

  if (!workset.empty()) {
    return false;
  }

  workset.insert_range(rhs.getTargets());
  for (const auto& trgt : lhs.getTargets()) {
    const auto& v = m.lookup(trgt);
    if (!workset.contains(v)) {
      return false;
    }
    workset.erase(v);
  }

  return workset.empty();
}

/// Compare two ctrl operations, allowing permutations of input control and
/// input target qubits.
static bool compareCtrlOps(qco::CtrlOp lhs, qco::CtrlOp rhs,
                           const IRMapping& m) {
  DenseSet<Value> workset;
  workset.insert_range(rhs.getInputControls());
  for (const auto& ctrl : lhs.getInputControls()) {
    const auto& v = m.lookup(ctrl);
    if (!workset.contains(v)) {
      return false;
    }
    workset.erase(v);
  }

  if (!workset.empty()) {
    return false;
  }

  workset.insert_range(rhs.getInputTargets());
  for (const auto& trgt : lhs.getInputTargets()) {
    const auto& v = m.lookup(trgt);
    if (!workset.contains(v)) {
      return false;
    }
    workset.erase(v);
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
  } else if (auto symbolRefAttrA =
                 llvm::dyn_cast<mlir::FlatSymbolRefAttr>(lhs)) {
    auto symbolRefAttrB = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(rhs);
    if (!symbolRefAttrB) {
      return false;
    }

    if (symbolRefAttrA.getValue() != symbolRefAttrB.getValue()) {
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
    if (!compareCtrlOps(cast<qc::CtrlOp>(lhs), cast<qc::CtrlOp>(rhs), m)) {
      return false;
    }
  } else if (isa<qco::CtrlOp>(lhs)) {
    assert(isa<qco::CtrlOp>(rhs));
    if (!compareCtrlOps(cast<qco::CtrlOp>(lhs), cast<qco::CtrlOp>(rhs), m)) {
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
      // of the chain.
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

      // We apply the analogous logic to extracts.
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
    mapArguments(lhs, rhs, getTargetPermutation(lhsCtrl, rhsCtrl, m), m);
  } else if (isa<qco::CtrlOp>(lhs.getParentOp())) {
    assert(isa<qco::CtrlOp>(rhs.getParentOp()));
    auto lhsCtrl = cast<qco::CtrlOp>(lhs.getParentOp());
    auto rhsCtrl = cast<qco::CtrlOp>(rhs.getParentOp());
    mapArguments(lhs, rhs, getTargetPermutation(lhsCtrl, rhsCtrl, m), m);
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

            SmallVector<size_t> permutation;
            permutation.reserve(lhsCtrl.getNumQubits());
            permutation.append(getControlPermutation(lhsCtrl, rhsCtrl, m));
            for (const auto i : getTargetPermutation(lhsCtrl, rhsCtrl, m)) {
              permutation.emplace_back(lhsCtrl.getNumControls() + i);
            }
            mapResults(lhsCtrl, rhsCtrl, permutation, m);
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

    lhsBlock.dump();
    rhsBlock.dump();
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
