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
    Input may contain at most 64 region-owning operations along any IR path.

    Note that this pass is still in development as QCO and `jeff` do not have full feature parity yet.

    The module must contain exactly one defined, single-block function marked
    with `mqt.entry_point`, and static and dynamic allocation modes must not be
    mixed. Static qubit indices must be unique.

    `qco.static` is currently converted to `jeff.qubit_alloc` because `jeff` does not yet represent static (index-based) qubits.
    As the index is not preserved in `jeff`, it is not possible to round-tripping static qubits.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect", "mlir::jeff::JeffDialect",
                           "mlir::qc::QCDialect", "mlir::qco::QCODialect"];
}
