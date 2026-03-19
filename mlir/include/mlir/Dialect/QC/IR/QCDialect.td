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
        The QC dialect uses **reference semantics** where quantum operations
        modify qubits in place, similar to how hardware physically transforms
        quantum states. This model provides:

        - Natural mapping to hardware execution models
        - Intuitive representation for circuit descriptions
        - Direct compatibility with imperative quantum programming languages
        - Straightforward backend code generation

        The name "QC" stands for "Quantum Circuit."

        Example:
        ```mlir
        qc.h %q          // Applies Hadamard to qubit %q in place
        qc.swap %q0, %q1 // Applies SWAP using %q0, %q1 as targets
        ```
    }];

  let cppNamespace = "::mlir::qc";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_QC_IR_QCDIALECT_TD
