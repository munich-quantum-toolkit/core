/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/QC/IR/QCDialect.h" // IWYU pragma: associated
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <complex>
#include <cstdint>

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

using namespace mlir;
using namespace mlir::qc;

static ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  return utils::parseTargetAliasing<QubitType>(parser, region, operands);
}

static void printTargetAliasing(OpAsmPrinter& printer, Operation* /*op*/,
                                Region& region, OperandRange targetsIn) {
  utils::printTargetAliasing(printer, region, targetsIn);
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QC/IR/QCOpsDialect.cpp.inc"

void QCDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/QC/IR/QCOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QC/IR/QCOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QC/IR/QCOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QC/IR/QCInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QC/IR/QCOps.cpp.inc"

LogicalResult UnitaryOp::verify() {
  const auto matrix = dyn_cast<DenseElementsAttr>(getMatrix());
  if (!matrix) {
    return emitOpError("matrix must use dense element storage");
  }
  const auto type = dyn_cast<RankedTensorType>(matrix.getType());
  if (!type || type.getRank() != 2 ||
      type.getShape()[0] != type.getShape()[1]) {
    return emitOpError("matrix must be a square rank-two tensor");
  }
  const auto complexType = dyn_cast<ComplexType>(type.getElementType());
  if (!complexType || !complexType.getElementType().isF64()) {
    return emitOpError("matrix elements must have type complex<f64>");
  }
  if (getQubits().empty()) {
    return emitOpError("requires at least one target qubit");
  }
  if (getQubits().size() >= 63U) {
    return emitOpError(
        "has too many target qubits to represent its matrix dimension");
  }
  const auto expectedDimension =
      static_cast<int64_t>(uint64_t{1} << getQubits().size());
  if (type.getShape()[0] != expectedDimension) {
    return emitOpError() << "matrix dimension must be 2^n = "
                         << expectedDimension << " for " << getQubits().size()
                         << " target qubits";
  }
  if (llvm::any_of(matrix.getValues<std::complex<double>>(),
                   [](const std::complex<double> value) {
                     return !std::isfinite(value.real()) ||
                            !std::isfinite(value.imag());
                   })) {
    return emitOpError("matrix entries must be finite");
  }
  return success();
}
