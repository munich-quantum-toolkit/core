// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_OQ3_IR_OQ3TYPES_TD
#define MLIR_DIALECT_OQ3_IR_OQ3TYPES_TD

include "mlir/Dialect/OQ3/IR/OQ3Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class OQ3Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<OQ3Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def BitType : OQ3Type<"Bit", "bit", [MemRefElementTypeInterface]> {
  let summary = "OpenQASM bit or fixed-width bit-register value";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

def AngleType : OQ3Type<"Angle", "angle", [MemRefElementTypeInterface]> {
  let summary = "OpenQASM fixed-width modular angle";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

def DurationType
    : OQ3Type<"Duration", "duration", [MemRefElementTypeInterface]> {
  let summary = "OpenQASM duration value";
}

def StretchType : OQ3Type<"Stretch", "stretch", [MemRefElementTypeInterface]> {
  let summary = "OpenQASM stretch value";
}

#endif // MLIR_DIALECT_OQ3_IR_OQ3TYPES_TD
