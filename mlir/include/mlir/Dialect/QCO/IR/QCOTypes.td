// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_IR_QCOTYPES_TD
#define MLIR_DIALECT_QCO_IR_QCOTYPES_TD

include "mlir/Dialect/QCO/IR/QCODialect.td"

include "mlir/IR/AttrTypeBase.td"

class QCOType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QCODialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def QubitType : QCOType<"Qubit", "qubit"> {
  let summary = "QCO qubit value type";
  let description = [{
        The `!qco.qubit` type represents an SSA value holding a quantum bit
        in the QCO dialect. Operations using this type consume input qubits
        and produce new output qubits following value semantics and the SSA
        paradigm, enabling powerful dataflow analysis and optimization.

        Example:
        ```mlir
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
        %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
        ```
    }];
}

#endif // MLIR_DIALECT_QCO_IR_QCOTYPES_TD
