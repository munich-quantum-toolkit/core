// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQCO : Pass<"qc-to-qco", "mlir::ModuleOp"> {
  let summary = "Convert QC dialect to QCO dialect.";

  let description = [{
    This pass converts all operations from the QC dialect to their equivalent operations in the QCO dialect.
    It handles the transformation of qubit references in QC to qubit values in QCO, ensuring that the semantics of quantum operations are preserved during the conversion process.
    Control flow must use structured SCF operations. Operations with block
    successors, including `cf.br`, `cf.cond_br`, and `cf.switch`, are diagnosed
    before conversion.
    Private helpers may borrow fixed-size rank-one qubit memrefs with identity
    layout and default memory space. They become qubit tensors. Updated scalar
    qubits and complete tensors are appended to ordinary results in argument
    order. Duplicate register operands, register/element aliases, and consumption
    or explicit returns of borrowed registers are diagnosed before conversion.
    Static qubit references are hoisted and coalesced when needed. General
    classical folding and common subexpression elimination are not guaranteed
    by this conversion; use the corresponding simplification passes when needed.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::qco::QCODialect",
                           "mlir::qtensor::QTensorDialect"];
}
