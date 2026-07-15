// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_IR_QCOATTRDEFS_TD
#define MLIR_DIALECT_QCO_IR_QCOATTRDEFS_TD

include "mlir/Dialect/QCO/IR/QCODialect.td"

include "mlir/IR/EnumAttr.td"

def Pauli
    : I32EnumAttr<"Pauli",
                  "A single Pauli operator, which can be either I, X, Y, or Z.",
                  [I32EnumAttrCase<"I", 0b00>, I32EnumAttrCase<"X", 0b10>,
                   I32EnumAttrCase<"Y", 0b11>, I32EnumAttrCase<"Z", 0b01>,
]> {
  let cppNamespace = "::mlir::qco";
}

def PauliAttr : EnumAttr<QCODialect, Pauli, "pauli">;

// A pauli string, e.g. [X, Y, Y].
def PauliWord : TypedArrayAttrBase<PauliAttr, "A product of Pauli operators.">;

#endif // MLIR_DIALECT_QCO_IR_QCOATTRDEFS_TD
