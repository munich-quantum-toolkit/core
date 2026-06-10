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
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstddef>
#include <utility>

using namespace mlir;

static bool compareRegions(Region& regionA, Region& regionB,
                           DenseSet<Operation*>& visited, IRMapping& m);

static bool hasTypeQubitTensor(Value v) {
  auto tensor = dyn_cast<RankedTensorType>(v.getType());
  if (!tensor) {
    return false;
  }

  return isa<qco::QubitType>(tensor.getElementType());
}

static void remapResults(Operation* fromOp, Operation* toOp, IRMapping& m) {
  for (const auto& [fromResult, toResult] :
       llvm::zip_equal(fromOp->getResults(), toOp->getResults())) {
    if (!isa<qtensor::AllocOp>(fromOp) && hasTypeQubitTensor(fromResult)) {
      assert(hasTypeQubitTensor(toResult));
      continue;
    }
    m.map(fromResult, toResult);
  }
}

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
                              DenseSet<Operation*>& visited, IRMapping& m) {

  // Compare top-level signature-like characteristics.

  if (opA->getName() != opB->getName() ||
      opA->getNumOperands() != opB->getNumOperands() ||
      opA->getOperandTypes() != opB->getOperandTypes() ||
      opA->getNumResults() != opB->getNumResults() ||
      opA->getResultTypes() != opB->getResultTypes() ||
      // opA->getAttrs().size() != opB->getAttrs().size() ||
      opA->getNumRegions() != opB->getNumRegions()) {
    llvm::dbgs() << "\t\tUNEQUAL 2!\n";
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
      llvm::dbgs() << "\t\tUNEQUAL 3!\n";
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
        llvm::dbgs() << "\t\tUNEQUAL 6!\n";
        llvm::dbgs() << "\t\t" << mappedOperand << '\n';
        llvm::dbgs() << "\t\t" << operandB << '\n';
        return false;
      }
    }
  }

  for (const auto [regionA, regionB] :
       llvm::zip_equal(opA->getRegions(), opB->getRegions())) {
    if (!compareRegions(regionA, regionB, visited, m)) {
      llvm::dbgs() << "\t\tUNEQUAL 7!\n";
      return false;
    }
  }

  // If the function reached this point, the two operations are equal.

  llvm::dbgs() << "\t\tEQUAL!\n";

  return true;
}

/// Extract and return "ready" operations.
static SetVector<Operation*> getReadyOps(ArrayRef<Operation*> ops,
                                         DenseSet<Operation*>& visited) {
  const auto isReady = [&](OpOperand& operand) {
    if (isa<BlockArgument>(operand.get())) {
      return true;
    }
    return visited.contains(operand.get().getDefiningOp());
  };

  SetVector<Operation*> ready;
  for (Operation* op : ops) {
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

static bool compareReady(SetVector<Operation*>::const_iterator it,
                         const SetVector<Operation*>& readyA,
                         const SetVector<Operation*>& readyB,
                         DenseSet<Operation*>& visited, IRMapping& m) {
  if (it == readyA.end()) {
    return true;
  }

  Operation* opA = *it;

  // Because there may be multiple structural equivalent operations,
  // collect them in a vector for further recursive processing. If there are
  // no partners, no matching operation has been found and the blocks are
  // not equivalent.

  llvm::dbgs() << "--- compare ---\n";
  opA->dumpPretty();
  SmallVector<Operation*> partners(
      make_filter_range(readyB, [&](Operation* op) {
        llvm::dbgs() << "\t";
        op->dumpPretty();
        return compareOperations(opA, op, visited, m);
      }));

  if (partners.empty()) {
    llvm::dbgs() << "no matching op found: " << *opA << '\n';
    return false;
  }

  // If there were only one partner, we could simply update the mapping m
  // here. Unfortunately, multiple operations can be structurally equivalent
  // (think arith.constant, for example). Thus, we must test each possible
  // alternative mapping to identify (if possible) the equivalent partner.

  bool found{false};
  for (Operation* partner : partners) {
    llvm::dbgs() << "trying partner: " << *partner << '\n';
    IRMapping partnerM(m);
    remapResults(opA, partner, partnerM);

    DenseSet<Operation*> altVisited(visited);
    if (compareReady(std::next(it), readyA, readyB, altVisited, partnerM)) {
      visited = std::move(altVisited);
      m = std::move(partnerM);
      found = true;
      break;
    }
  }

  return found;
}

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

  SmallVector<Operation*> opsA;
  SmallVector<Operation*> opsB;

  opsA.reserve(llvm::range_size(blockA.getOperations()));
  opsB.reserve(llvm::range_size(blockB.getOperations()));

  for_each(blockA.getOperations(), [&](auto& op) { opsA.emplace_back(&op); });
  for_each(blockB.getOperations(), [&](auto& op) { opsB.emplace_back(&op); });

  while (true) {
    const auto readyA = getReadyOps(opsA, visited);
    const auto readyB = getReadyOps(opsB, visited);

    visited.insert_range(readyA);
    visited.insert_range(readyB);

    if (readyA.empty() && readyB.empty()) {
      break;
    }

    if ((readyA.empty() && !readyB.empty()) ||
        (!readyA.empty() && readyB.empty()) || readyA.size() != readyB.size()) {
      return false;
    }

    if (!compareReady(readyA.begin(), readyA, readyB, visited, m)) {
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
