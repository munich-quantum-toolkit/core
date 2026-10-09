// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToJeff : Pass<"qco-to-jeff", "mlir::ModuleOp"> {
  let summary = "Convert QCO operations to `jeff` operations";

  let description = [{
    This pass converts all operations from the QCO dialect to their equivalent operations in the `jeff` dialect.
    It ensures that the returned module is a valid `jeff` module that can be serialized.

    Defined single-block functions and their calls are preserved. Mutable
    classical-register arguments in helpers are not supported. Unitary calls
    inside quantum modifiers must be expanded with `unroll-modifiers` first.
    Each remaining modifier must contain one unitary using all body arguments
    in order. Canonicalize or unroll modifiers outside this normal form before
    conversion.

    `qco.static` exports as `jeff.qubit_alloc`; static site indices are not
    preserved. Exchange logical programs before device mapping.
  }];

  let dependentDialects = ["mlir::func::FuncDialect", "mlir::jeff::JeffDialect",
                           "mlir::arith::ArithDialect"];
}
