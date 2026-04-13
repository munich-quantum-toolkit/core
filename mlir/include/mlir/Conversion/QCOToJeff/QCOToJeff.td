// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToJeff : Pass<"qco-to-jeff"> {
  let summary = "Convert QCO operations to Jeff operations";

  let description = [{
    This pass converts all operations from the QCO dialect to their equivalent operations in the Jeff dialect.
    It ensures that the returned module is a valid Jeff module that can be serialized.

    Note that this pass is still in development as QCO and Jeff do not have full feature parity yet.

    `qco.static` is currently converted to `jeff.qubit_alloc` because Jeff does not yet represent static (index-based) qubits; the index is not preserved in Jeff IR. Round-tripping through Jeff therefore maps static qubits to dynamic allocation on the QCO side.
  }];

  let dependentDialects = ["mlir::jeff::JeffDialect"];
}
