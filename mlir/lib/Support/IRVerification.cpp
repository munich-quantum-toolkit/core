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
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
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
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cmath>
#include <iterator>
#include <utility>

using namespace mlir;

static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& closed, IRMapping& m);

/// Return true, if the given value has the type `tensor<qco.qubit>`.
static bool hasTypeQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

/// Map all results from one op to another.
/// Assumes `op->getNumResults() == other->getNumResults()`.
/// Assumes that the two operations are equivalent to each other.
static void mapResults(Operation* opL, Operation* opR, IRMapping& m) {
  if (opL->getNumResults() == 0) {
    return;
  }

  for (const auto& [fromResult, toResult] :
       llvm::zip_equal(opL->getResults(), opR->getResults())) {
    if (!isa<qtensor::AllocOp>(opL) && hasTypeQubitTensor(fromResult)) {
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

  } else if (auto symbolRefAttrA =
                 llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attrA)) {
    auto symbolRefAttrB = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attrB);
    if (!symbolRefAttrB) {
      return false;
    }

    if (symbolRefAttrA.getValue() != symbolRefAttrB.getValue()) {
      return false;
    }
  } else {
    // attrA.dump();
    // llvm::dbgs() << "unhandled attribute type!\n";
    // attrA.dump();
    // llvm::reportFatalInternalError("unhandled attribute type!");
    // llvm::llvm_unreachable_internal();
  }

  return true;
}

static bool compareOperations(Operation* opA, Operation* opB,
                              const IRMapping& m) {

  // Compare top-level signature-like characteristics.

  if (opA->getName() != opB->getName() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      opA->getOperandTypes() != opB->getOperandTypes() ||
      opA->getNumResults() != opB->getNumResults() ||
      opA->getResultTypes() != opB->getResultTypes() ||
      // opA->getAttrs().size() != opB->getAttrs().size() ||
      opA->getNumRegions() != opB->getNumRegions()) {
    return false;
  }

  // Compare attributes.

  const DenseSet<StringRef> ignore{"passthrough"};

  for (const auto& namedAttrA : opA->getAttrs()) {
    const StringRef keyA = namedAttrA.getName().strref();

    if (ignore.contains(keyA)) {
      llvm::dbgs() << "ignoring: " << keyA << '\n';
      continue;
    }

    if (!opB->hasAttr(keyA)) {
      llvm::dbgs() << "missing attribute: " << keyA << '\n';
      continue;
    }

    if (!compareAttributes(namedAttrA.getValue(), opB->getAttr(keyA))) {
      return false;
    }
  }

  // Compare operands.
  // TODO: Equal type check.

  // else if (isa<qco::CtrlOp>(opA)) {
  //   assert(isa<qco::CtrlOp>(opB));
  //   llvm::reportFatalInternalError("not implemented");
  // } else if (opA->hasTrait<mlir::OpTrait::IsCommutative>()) {
  //   assert(opB->hasTrait<mlir::OpTrait::IsCommutative>());
  //   llvm::reportFatalInternalError("not implemented");
  // }

  if (isa<qc::CtrlOp>(opA)) {
    assert(isa<qc::CtrlOp>(opB));

    auto ctrlL = cast<qc::CtrlOp>(opA);
    auto ctrlR = cast<qc::CtrlOp>(opB);

    DenseSet<Value> workset;
    workset.insert_range(ctrlR.getControls());
    for (const auto& ctrl : ctrlL.getControls()) {
      const auto& mapped = m.lookup(ctrl);
      if (!workset.contains(mapped)) {
        return false;
      }
      workset.erase(mapped);
    }

    assert(workset.empty());

    // Analogously for the targets.

    workset.clear();
    workset.insert_range(ctrlR.getTargets());
    for (const auto& trgt : ctrlL.getTargets()) {
      const auto& operand = m.lookup(trgt);
      if (!workset.contains(operand)) {
        return false;
      }
      workset.erase(operand);
    }

    assert(workset.empty());

  } else {
    for (const auto& [operandA, operandB] :
         llvm::zip_equal(opA->getOperands(), opB->getOperands())) {

      if (hasTypeQubitTensor(operandA)) {
        if (!hasTypeQubitTensor(operandB)) { // TODO: Assertion?
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
      } else {
        auto operand = m.lookup(operandA);
        if (operand != operandB) {
          return false;
        }
      }
    }
  }

  return true;
}

/// Extract and return "ready" operations.
/// These are operations that are independent from each other.
static SetVector<Operation*> getReadyOps(ArrayRef<Operation*> open,
                                         DenseSet<Operation*>& closed) {
  const auto isReady = [&closed](Value v) {
    if (isa<BlockArgument>(v)) {
      return true;
    }
    return closed.contains(v.getDefiningOp());
  };

  SetVector<Operation*> ready;
  for (Operation* op : open) {
    if (closed.contains(op) || ready.contains(op)) {
      continue;
    }

    if (auto insert = dyn_cast<qtensor::InsertOp>(op)) {

      // If any of the inserts on the chain are ready, we consider the whole
      // chain ready because the one ready operation could be moved the front
      // (the top from IR perspective) of the chain.
      // An insert is considered ready when both the inserted qubit and the
      // index are ready.

      SmallVector<Operation*> chain;
      qtensor::TensorIterator it(insert.getResult());

      for (; it != std::default_sentinel &&
             isa<qtensor::InsertOp>(it.operation());
           ++it) {
        auto chainInsert = cast<qtensor::InsertOp>(it.operation());
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

      for (; it != std::default_sentinel &&
             isa<qtensor::ExtractOp>(it.operation());
           ++it) {
        auto chainExtract = cast<qtensor::ExtractOp>(it.operation());
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
      // chain.

      bool fullChain{true};
      qtensor::TensorIterator it(dealloc.getTensor());

      for (; std::prev(it) != it; --it) {
        if (!closed.contains(it.operation())) {
          fullChain = false;
          break;
        }
      }

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

static bool compareTopologically(ArrayRef<Operation*> openA,
                                 ArrayRef<Operation*> openB,
                                 DenseSet<Operation*>& closed, IRMapping& m) {
  const auto readyA = getReadyOps(openA, closed);
  const auto readyB = getReadyOps(openB, closed);

  if (readyA.empty() && readyB.empty()) {
    return true;
  }

  if (readyA.size() != readyB.size()) {
    return false;
  }

  // Because there may be multiple structural equivalent operations (think
  // arith.constant, for example), collect them in a vector for further
  // recursive processing. If there are no partners, no matching operation has
  // been found and the blocks are not equivalent.

  DenseSet<Operation*> matched;
  matched.reserve(readyB.size());

  for (Operation* opA : readyA) {
    SetVector<Operation*>::iterator it = readyB.begin();
    for (; it != readyB.end(); it = std::next(it)) {
      Operation* opB = *it;
      if (matched.contains(opB)) {
        continue;
      }

      if (compareOperations(opA, opB, m)) {
        matched.insert(opB);
        mapResults(opA, opB, m);
        m.map(opA, opB);
        break;
      }
    }

    if (it == readyB.end()) {
      llvm::dbgs() << "unmatched op a: " << *opA << '\n';
      return false;
    }
  }

  // At this point, we've successfully matched each operation on the lhs with
  // one on the rhs.

  closed.insert_range(readyA);
  closed.insert_range(readyB);

  SetVector<Operation*>::iterator it = readyA.begin();
  for (; it != readyA.end(); it = std::next(it)) {
    Operation* opA = *it;

    // Otherwise, if opA has one or more regions, try each mapping to find the
    // equivalent operation. Each mapping uniquely identify opA with one
    // potential partner thus use the mapping to obtain this partner and
    // compare their respective regions.

    if (opA->getNumRegions() > 0) {
      Operation* opB = m.lookup(opA);
      assert(opA->getNumRegions() == opB->getNumRegions());

      const auto nequiv = range_size(make_filter_range(
          llvm::zip_equal(opA->getRegions(), opB->getRegions()),
          [&](const auto& zip) {
            const auto& [regionA, regionB] = zip;
            return compareRegions(regionA, regionB, closed, m);
          }));
      if (nequiv != opA->getNumRegions()) {
        break;
      }
    }
  }

  if (it != readyA.end()) {
    return false;
  }

  return compareTopologically(openA, openB, closed, m);
}

static bool compareBlocks(Block& blockA, Block& blockB,
                          DenseSet<Operation*>& closed, IRMapping& m) {
  if (blockA.getNumArguments() != blockB.getNumArguments()) {
    return false;
  }

  for (const auto [lArg, rArg] :
       llvm::zip_equal(blockA.getArguments(), blockB.getArguments())) {
    if (lArg.getType() != rArg.getType()) {
      llvm::dbgs() << "unequivalent block arguments\n";
      return false;
    }

    m.map(lArg, rArg);
  }

  SmallVector<Operation*> openA;
  SmallVector<Operation*> openB;

  openA.reserve(range_size(blockA.getOperations()));
  openB.reserve(range_size(blockB.getOperations()));

  for_each(blockA.getOperations(), [&](auto& op) { openA.emplace_back(&op); });
  for_each(blockB.getOperations(), [&](auto& op) { openB.emplace_back(&op); });

  return compareTopologically(openA, openB, closed, m);
}

/// Compare two regions for structural equivalence.
static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& closed, IRMapping& m) {
  if (regionA.getBlocks().size() != regionB.getBlocks().size()) {
    return false;
  }

  for (const auto [blockA, blockB] : llvm::zip_equal(regionA, regionB)) {
    if (!compareBlocks(blockA, blockB, closed, m)) {
      return false;
    }
  }

  return true;
}

bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  IRMapping m;
  DenseSet<Operation*> closed;
  return compareRegions(lhs.getBodyRegion(), rhs.getBodyRegion(), closed, m);
}
