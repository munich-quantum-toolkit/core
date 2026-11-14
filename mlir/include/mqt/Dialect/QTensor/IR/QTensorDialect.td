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
    QTensor groups QCO qubits into one-dimensional tensors, for example to
    represent a register. Like individual QCO qubit values, each tensor value
    has exactly one use.

    Extracting a qubit consumes the tensor and returns both the qubit and the
    remaining tensor. The extracted slot is unavailable until the qubit is
    inserted back into it. Gates may change the qubit's state, but reinsertion
    must restore the same qubit identity to the same slot. Allocation and
    deallocation create and consume complete tensors.

    A gate or call must receive distinct quantum resources. The shared
    [quantum resource rules](index.md#quantum-resource-rules) explain how
    these rules apply to helpers, loops, and dynamically computed indices.
  }];

  let dependentDialects = ["::mlir::qco::QCODialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];

  let hasCanonicalizer = 1;

  let cppNamespace = "::mlir::qtensor";
}

#endif // MLIR_DIALECT_QTENSOR_IR_QTENSORDIALECT_TD
