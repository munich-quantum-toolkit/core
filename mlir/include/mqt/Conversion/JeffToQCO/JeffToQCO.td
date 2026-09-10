// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def JeffToQCO : Pass<"jeff-to-qco", "mlir::ModuleOp"> {
  let summary = "Convert `jeff` operations to QCO operations";

  let description = [{
    This pass converts all operations from the `jeff` dialect to their equivalent operations in the QCO dialect.
    It ensures that the returned module is a valid QCO module.

    Note that this pass is still in development as QCO and `jeff` do not have full feature parity yet.

    Known limitations:

    - Only specific `CustomOp`s are currently supported
    - Only specific `PPROp`s are currently supported
    - Mutable classical-register arguments in helper functions are not supported

    Function definitions and calls are preserved. Only the designated entry
    point remains public; helper functions become private. Calls remain generic
    `func.call` operations because jeff does not specify function unitarity.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect", "mlir::cbit::CBitDialect",
                           "mlir::math::MathDialect", "mlir::mqt::MQTDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::qco::QCODialect",
                           "mlir::qtensor::QTensorDialect",
  ];
}
