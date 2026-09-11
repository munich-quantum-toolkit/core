// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QTENSOR_IR_QTENSORDIALECT_TD
#define MLIR_DIALECT_QTENSOR_IR_QTENSORDIALECT_TD

include "mlir/IR/DialectBase.td"

def QTensorDialect : Dialect {
  let name = "qtensor";

  let summary = "The QTensor dialect for one-dimensional tensors of qubits "
                "with linear semantics.";

  let description = [{
    The QTensor dialect represents one-dimensional tensors of qubits with
    linear ownership. Each quantum SSA value must have exactly one use.
    Extraction consumes a tensor and returns both the extracted qubits and
    the remaining tensor. Allocation and deallocation create and consume
    whole qubit tensors.
  }];

  let dependentDialects = ["::mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];

  let hasCanonicalizer = 1;

  let cppNamespace = "::mlir::qtensor";
}

#endif // MLIR_DIALECT_QTENSOR_IR_QTENSORDIALECT_TD
