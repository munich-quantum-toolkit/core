/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QC/IR/QCDialect.h" // IWYU pragma: associated

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep (template instantiations)
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep (template instantiations)

#include <cstdint>

using namespace mlir;
using namespace mlir::qc;

namespace {

struct QCInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* call, Operation* callable,
                       bool /*wouldBeCloned*/) const final {
    auto callee = dyn_cast<func::FuncOp>(callable);
    return isa<CallOp>(call) && callee && !callee.getNoInline();
  }

  bool isLegalToInline(Region* destination, Region* source,
                       bool /*wouldBeCloned*/,
                       IRMapping& /*valueMapping*/) const final {
    if (!destination->hasOneBlock() || !source->hasOneBlock()) {
      return false;
    }
    // Modifier bodies cannot contain structured control flow.
    return !isa<UnitaryOpInterface>(destination->getParentOp()) ||
           llvm::all_of(source->front(), [](Operation& operation) {
             return operation.getNumRegions() == 0 ||
                    isa<UnitaryOpInterface>(operation);
           });
  }

  bool isLegalToInline(Operation* /*operation*/, Region* /*destination*/,
                       bool /*wouldBeCloned*/,
                       IRMapping& /*valueMapping*/) const final {
    return true;
  }
};

} // namespace

static ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  return mqt::parseTargetAliasing<QubitType>(parser, region, operands);
}

static void printTargetAliasing(OpAsmPrinter& printer, Operation* /*op*/,
                                Region& region, OperandRange targetsIn) {
  mqt::printTargetAliasing(printer, region, targetsIn);
}

void MeasureOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "b");
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mqt/Dialect/QC/IR/QCOpsDialect.cpp.inc"

void QCDialect::initialize() {
  addInterfaces<QCInlinerInterface>();

  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mqt/Dialect/QC/IR/QCOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/QC/IR/QCOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mqt/Dialect/QC/IR/QCOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

LogicalResult mlir::qc::verifyDistinctQubitOperands(Operation* op,
                                                    ValueRange operands) {
  if (operands.size() < 2) {
    return success();
  }
  llvm::SmallDenseSet<Value, 4> qubits;
  llvm::SmallDenseSet<uint64_t, 4> staticIndices;
  for (Value qubit : operands) {
    if (!isa<QubitType>(qubit.getType())) {
      continue;
    }
    if (!qubits.insert(qubit).second) {
      return op->emitOpError("requires distinct qubit operands");
    }
    auto ref = qubit.getDefiningOp<StaticOp>();
    if (!ref) {
      continue;
    }
    // The producer may not have been verified yet.
    auto index = ref->getAttrOfType<IntegerAttr>(ref.getIndexAttrName());
    if (!index || !index.getType().isSignlessInteger(64) ||
        index.getValue().isNegative()) {
      continue;
    }
    if (!staticIndices.insert(index.getValue().getZExtValue()).second) {
      return op->emitOpError(
          "requires distinct qubit operands; references have the same static "
          "index");
    }
  }
  return success();
}

LogicalResult mlir::qc::verifyUnitaryOpInterface(Operation* op) {
  auto unitary = cast<UnitaryOpInterface>(op);
  if (failed(verifyDistinctQubitOperands(op, unitary.getQubits()))) {
    return failure();
  }
  return mqt::verifyFiniteConstantParameters(op, unitary.getParameters());
}

#include "mqt/Dialect/QC/IR/QCInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mqt/Dialect/QC/IR/QCOps.cpp.inc"
