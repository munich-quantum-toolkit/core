// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QC_IR_QCTYPES_TD
#define MLIR_DIALECT_QC_IR_QCTYPES_TD

include "mlir/Dialect/QC/IR/QCDialect.td"

include "mlir/IR/AttrTypeBase.td"

class QCType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QCDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def QubitType : QCType<"Qubit", "qubit"> {
  let summary = "QC qubit reference type";
  let description = [{
        The `!qc.qubit` type represents a reference to a quantum bit in the
        QC dialect. Operations using this type modify qubits in place using
        reference semantics, similar to how classical imperative languages handle
        mutable references.
    }];
}

#endif // MLIR_DIALECT_QC_IR_QCTYPES_TD
