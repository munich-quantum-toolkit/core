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
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
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
#include <iterator>

using namespace mlir;

static bool compareRegions(Region& lhs, Region& rhs,
                           SetVector<Operation*>& lhsClosed,
                           SetVector<Operation*>& rhsClosed, IRMapping& m);

/// Return true, if the given value has the type `tensor<qco.qubit>`.
static bool hasTypeQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

/// Map all results from one op to another.
/// Assumes that `lhs->getNumResults() == rhs->getNumResults()`.
/// Assumes that the two operations are equivalent to each other.
static void mapResults(Operation* lhs, Operation* rhs, IRMapping& m) {
  for (const auto& [fromResult, toResult] :
       llvm::zip_equal(lhs->getResults(), rhs->getResults())) {
    if (!isa<qtensor::AllocOp>(lhs) && !isa<qtensor::FromElementsOp>(lhs) &&
        hasTypeQubitTensor(fromResult)) {
      assert(hasTypeQubitTensor(toResult));
      continue;
    }
    m.map(fromResult, toResult);
  }
}

/// Compares two floating point numbers for approximate equivalence.
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
/// Explicitly checks `UnitAttr`, `IntegerAttr`, `FloatAttr`, `StringAttr`, and
/// `FlatSymbolRefAttr`.
/// For any other type, the function simply returns true.
static bool compareAttributes(Attribute lhs, Attribute rhs) {
  if (dyn_cast<UnitAttr>(lhs)) {
    if (!dyn_cast<UnitAttr>(rhs)) {
      return false;
    }
  } else if (auto intAttrA = dyn_cast<IntegerAttr>(lhs)) {
    auto intAttrB = dyn_cast<IntegerAttr>(rhs);
    if (!intAttrB || intAttrA.getValue() != intAttrB.getValue()) {
      return false;
    }
  } else if (auto floatAttrA = dyn_cast<FloatAttr>(lhs)) {
    auto floatAttrB = dyn_cast<FloatAttr>(rhs);

    if (!floatAttrB ||
        !approxCompareFloats(floatAttrA.getValue(), floatAttrB.getValue(),
                             floatAttrA.getType().getIntOrFloatBitWidth())) {
      return false;
    }
  } else if (auto strAttrA = dyn_cast<StringAttr>(lhs)) {
    auto strAttrB = dyn_cast<StringAttr>(rhs);
    if (!strAttrB || strAttrA.getValue() != strAttrB.getValue()) {
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

/// Compare two operations for structural equivalence, applying some special
/// rules for `CtrlOp` s and `qtensor` s.
static bool compareOperations(Operation* lhs, Operation* rhs,
                              const IRMapping& m) {

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

    auto ctrlLhs = cast<qc::CtrlOp>(lhs);
    auto ctrlRhs = cast<qc::CtrlOp>(rhs);

    DenseSet<Value> workset;
    workset.insert_range(ctrlRhs.getControls());
    for (const auto& ctrl : ctrlLhs.getControls()) {
      const auto& mapped = m.lookup(ctrl);
      if (!workset.contains(mapped)) {
        return false;
      }
      workset.erase(mapped);
    }

    assert(workset.empty());

    // Analogously for the targets.

    workset.clear();
    workset.insert_range(ctrlRhs.getTargets());
    for (const auto& trgt : ctrlLhs.getTargets()) {
      const auto& operand = m.lookup(trgt);
      if (!workset.contains(operand)) {
        return false;
      }
      workset.erase(operand);
    }

    assert(workset.empty());

  } else {
    for (const auto& [operandLhs, operandRhs] :
         llvm::zip_equal(lhs->getOperands(), rhs->getOperands())) {

      if (hasTypeQubitTensor(operandLhs)) {
        assert(hasTypeQubitTensor(operandRhs));

        auto tensorLhs = cast<TypedValue<RankedTensorType>>(operandLhs);
        qtensor::TensorIterator itLhs(tensorLhs);
        while (std::prev(itLhs) != itLhs) {
          --itLhs;
        }

        auto tensorRhs = cast<TypedValue<RankedTensorType>>(operandRhs);
        qtensor::TensorIterator itRhs(tensorRhs);
        while (std::prev(itRhs) != itRhs) {
          --itRhs;
        }

        if (isa<BlockArgument>(itLhs.tensor())) {
          if (!isa<BlockArgument>(itRhs.tensor())) {
            return false;
          }
        } else if (isa<qtensor::AllocOp>(itLhs.operation())) {
          if (!isa<qtensor::AllocOp>(itRhs.operation())) {
            return false;
          }

          auto allocLhs = cast<qtensor::AllocOp>(itLhs.operation());
          auto allocRhs = cast<qtensor::AllocOp>(itRhs.operation());

          if (m.lookup(allocLhs.getResult()) != allocRhs.getResult()) {
            return false;
          }
        } else if (isa<qtensor::FromElementsOp>(itLhs.operation())) {
          if (!isa<qtensor::FromElementsOp>(itRhs.operation())) {
            return false;
          }

          auto fromLhs = cast<qtensor::FromElementsOp>(itLhs.operation());
          auto fromRhs = cast<qtensor::FromElementsOp>(itRhs.operation());

          if (m.lookup(fromLhs.getResult()) != fromRhs.getResult()) {
            return false;
          }
        } else {
          llvm::reportFatalInternalError("unhandled qtensor source");
        }
      } else {
        auto operand = m.lookup(operandLhs);
        if (operand != operandRhs) {
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
      qtensor::TensorIterator it(insert.getResult());

      for (; it != std::default_sentinel; ++it) {
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
      qtensor::TensorIterator it(extract.getOutTensor());

      for (; it != std::default_sentinel; ++it) {
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
      // iterator already points at the previous operation. Thus use a do-while
      // loop instead of a regular while.

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
                          SetVector<Operation*>& rhsClosed, IRMapping& m) {
  if (lhs.getNumArguments() != rhs.getNumArguments()) {
    return false;
  }

  for (const auto [lhsArg, rhsArg] :
       llvm::zip_equal(lhs.getArguments(), rhs.getArguments())) {
    if (lhsArg.getType() != rhsArg.getType()) {
      return false;
    }

    m.map(lhsArg, rhsArg);
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

    for (Operation* opLhs : lhsReady) {
      SetVector<Operation*>::iterator it = rhsReady.begin();
      for (; it != rhsReady.end(); it = std::next(it)) {
        Operation* opRhs = *it;

        if (matched.contains(opRhs)) {
          continue;
        }

        if (compareOperations(opLhs, opRhs, m)) {
          matched.insert(opRhs);
          mapResults(opLhs, opRhs, m);
          m.map(opLhs, opRhs);
          break;
        }
      }

      if (it == rhsReady.end()) {
        return false;
      }
    }

    // At this point, we've successfully matched each operation on the lhs with
    // one on the rhs.

    lhsOpen.set_subtract(lhsReady);
    lhsClosed.set_union(lhsReady);

    rhsOpen.set_subtract(rhsReady);
    rhsClosed.set_union(rhsReady);

    // Once all ready operations have been matched, recursively compare
    // the nested regions of each operation pair.

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
                                    m);
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
                           SetVector<Operation*>& rhsClosed, IRMapping& m) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  for (const auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhs, rhs)) {
    if (!compareBlocks(lhsBlock, rhsBlock, lhsClosed, rhsClosed, m)) {
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
  return compareRegions(lhs.getBodyRegion(), rhs.getBodyRegion(), lhsClosed,
                        rhsClosed, m);
}
