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
        modify qubits in place. Operation order determines the sequence of
        transformations on each qubit.

        The name "QC" stands for "Quantum Circuit."

        In a program, dynamic qubit and qubit-register allocations must be in
        the entry block of the `mqt.entry_point` function. Helper functions
        receive quantum resources as arguments. The MQT entry-point verifier
        checks this rule across the program.

        Example:
        ```mlir
        qc.h %q          // Applies Hadamard to qubit %q in place
        qc.swap %q0, %q1 // Applies SWAP using %q0, %q1 as targets
        ```
    }];

  let cppNamespace = "::mlir::qc";
  let dependentDialects = ["::mlir::arith::ArithDialect"];

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_QC_IR_QCDIALECT_TD
