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
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/iterator_range.h>
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
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstddef>

using namespace mlir;

static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& closed, IRMapping& m);

static bool hasTypeQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

static void mapResults(Operation* fromOp, Operation* toOp, IRMapping& m) {
  for (const auto& [fromResult, toResult] :
       llvm::zip_equal(fromOp->getResults(), toOp->getResults())) {
    if (!isa<qtensor::AllocOp>(fromOp) && hasTypeQubitTensor(fromResult)) {
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
      auto mappedOperand = m.lookupOrNull(operandA);
      if (!mappedOperand || mappedOperand != operandB) {
        return false;
      }
    }
  }

  return true;
}

/// Extract and return "ready" operations.
/// These are operations that are independent from each other.
static SetVector<Operation*> getReadyOps(ArrayRef<Operation*> open,
                                         DenseSet<Operation*>& visited) {
  const auto isReady = [&](OpOperand& operand) {
    if (isa<BlockArgument>(operand.get())) {
      return true;
    }
    return visited.contains(operand.get().getDefiningOp());
  };

  SetVector<Operation*> ready;
  for (Operation* op : open) {
    if (visited.contains(op)) {
      continue;
    }

    if (llvm::all_of(op->getOpOperands(), isReady)) {
      ready.insert(op);
      continue;
    }

    // If the destination of a tensor insert, has been produced by an insert
    // operation as well, these two should be interchangeable. Thus, also add it
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

static void cartesianMappings(
    const SetVector<Operation*>::const_iterator readyIt,
    const SmallVector<SmallVector<Operation*, 2>>::const_iterator partnerIt,
    const SetVector<Operation*>::const_iterator readyEnd,
    const SmallVector<SmallVector<Operation*, 2>>::const_iterator partnerEnd,
    IRMapping& m, SmallVectorImpl<IRMapping>& out) {
  if (readyIt == readyEnd) {
    assert(partnerIt == partnerEnd);
    out.emplace_back(m);
    return;
  }

  Operation* opA = *readyIt;
  for (Operation* opB : *partnerIt) {
    IRMapping m2(m);
    m2.map(opA, opB);
    mapResults(opA, opB, m2);
    cartesianMappings(std::next(readyIt), std::next(partnerIt), readyEnd,
                      partnerEnd, m2, out);
  }
}

static bool compareReady(ArrayRef<Operation*> openA, ArrayRef<Operation*> openB,
                         DenseSet<Operation*>& closed, IRMapping& m) {
  const auto readyA = getReadyOps(openA, closed);
  const auto readyB = getReadyOps(openB, closed);

  if (readyA.empty() && readyB.empty()) {
    return true;
  }

  if ((readyA.empty() && !readyB.empty()) ||
      (!readyA.empty() && readyB.empty()) || readyA.size() != readyB.size()) {
    return false;
  }

  // Because there may be multiple structural equivalent operations (think
  // arith.constant, for example), collect them in a vector for further
  // recursive processing. If there are no partners, no matching operation has
  // been found and the blocks are not equivalent.

  SmallVector<SmallVector<Operation*, 2>> partners;
  for (Operation* opA : readyA) {
    const auto isEmpty =
        partners
            .emplace_back(make_filter_range(
                readyB,
                [&](Operation* opB) { return compareOperations(opA, opB, m); }))
            .empty();
    if (isEmpty) {
      return false;
    }
  }

  assert(partners.size() == readyA.size());

  closed.insert_range(readyA);
  closed.insert_range(readyB);

  SmallVector<IRMapping> mappings;
  cartesianMappings(readyA.begin(), partners.begin(), readyA.end(),
                    partners.end(), m, mappings);

  for (Operation* opA : readyA) {

    // If opA has one or more regions, try each mapping to find the equivalent
    // operation. Each mapping uniquely identify opA with one potential
    // partner thus use the mapping to obtain this partner and compare their
    // respective regions.

    if (opA->getNumRegions() > 0) {
      SmallVector<IRMapping>::iterator it = mappings.begin();
      for (; it != mappings.end(); it = std::next(it)) {
        Operation* opB = it->lookup(opA);
        assert(opA->getNumRegions() == opB->getNumRegions());

        const auto nequiv = range_size(make_filter_range(
            llvm::zip_equal(opA->getRegions(), opB->getRegions()),
            [&](const auto& zip) {
              const auto& [regionA, regionB] = zip;
              return compareRegions(regionA, regionB, closed, *it);
            }));
        if (nequiv == opA->getNumRegions()) {
          break;
        }
      }

      if (it == mappings.end()) {
        return false;
      }
    }
  }

  SmallVector<IRMapping>::iterator it = mappings.begin();
  for (; it != mappings.end(); it = std::next(it)) {
    DenseSet<Operation*> closed2(closed);
    if (compareReady(openA, openB, closed2, *it)) {
      closed = std::move(closed2);
      m = std::move(*it);
      break;
    }
  }

  return it != mappings.end();
}

static bool compareBlocks(Block& blockA, Block& blockB,
                          DenseSet<Operation*>& closed, IRMapping& m) {
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

  SmallVector<Operation*> openA;
  SmallVector<Operation*> openB;

  openA.reserve(range_size(blockA.getOperations()));
  openB.reserve(range_size(blockB.getOperations()));

  for_each(blockA.getOperations(), [&](auto& op) { openA.emplace_back(&op); });
  for_each(blockB.getOperations(), [&](auto& op) { openB.emplace_back(&op); });

  return compareReady(openA, openB, closed, m);
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
