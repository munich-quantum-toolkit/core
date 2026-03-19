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

  let summary = "The QTensor dialect for one-dimensional tensors of qubits with linear typing.";

  let description = [{
    The QTensor dialect is an adjusted variant of the standard tensor dialect of MLIR that supports linear typing for one-dimensional tensors of qubits.
    In order to support linear typing, the extract operations of this dialect are modified so that they also return the updated tensor.
    In addition, alloc/dealloc operations are added to the dialect to support the bulk allocation and deallocation of qubit tensors with linear types.
  }];

  let dependentDialects = ["::mlir::qco::QCODialect"];

  let cppNamespace = "::mlir::qtensor";
}

#endif // MLIR_DIALECT_QTENSOR_IR_QTENSORDIALECT_TD
