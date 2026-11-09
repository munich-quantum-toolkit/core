// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QC_IR_QCDIALECT_TD
#define MLIR_DIALECT_QC_IR_QCDIALECT_TD

include "mlir/IR/DialectBase.td"

def QCDialect : Dialect {
  let name = "qc";

  let summary = "The QC (reference semantics) dialect for quantum computing.";

  let description = [{
    QC (Quantum Circuit) represents qubits as stable references. A gate acts
    on the referenced qubit without producing a new reference, so operation
    order determines how its state changes. This model, called **reference semantics**, is
    convenient for translating circuits from and to external formats.

    ```mlir
    qc.h %q
    qc.swap %q0, %q1
    ```

    Allocation operations for qubits and registers must appear in the entry block of the
    `mqt.entry_point` function. Helper functions receive their qubits as
    arguments; the entry-point verifier checks this allocation rule.

    Each register slot continues to refer to the same qubit. A gate or call
    must receive distinct qubits, even when its operands include registers or
    slices. See the shared [quantum resource rules](index.md#quantum-resource-rules)
    for the rules on register access, helper functions, and conversion to QCO.
  }];

  let cppNamespace = "::mlir::qc";
  let dependentDialects = ["::mlir::arith::ArithDialect"];

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_QC_IR_QCDIALECT_TD
