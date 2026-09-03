// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_CBIT_IR_CBITOPS_TD
#define MLIR_DIALECT_CBIT_IR_CBITOPS_TD

include "mlir/Dialect/CBit/IR/CBitDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Attributes and types
//===----------------------------------------------------------------------===//

def CBit_Initialization
    : I32EnumAttr<"Initialization", "Classical-bit register initialization",
                  [I32EnumAttrCase<"Zero", 0, "zero">,
                   I32EnumAttrCase<"Undefined", 1, "undefined">]> {
  let cppNamespace = "::mlir::cbit";
  let genSpecializedAttr = 0;
}

def CBit_InitializationAttr
    : EnumAttr<CBitDialect, CBit_Initialization, "init"> {
  let assemblyFormat = "`<` $value `>`";
}

def CBit_ComparisonPredicate
    : I64EnumAttr<"ComparisonPredicate", "unsigned comparison predicate",
                  [I64EnumAttrCase<"Equal", 0, "eq">,
                   I64EnumAttrCase<"NotEqual", 1, "ne">,
                   I64EnumAttrCase<"Less", 2, "ult">,
                   I64EnumAttrCase<"LessEqual", 3, "ule">,
                   I64EnumAttrCase<"Greater", 4, "ugt">,
                   I64EnumAttrCase<"GreaterEqual", 5, "uge">]> {
  let cppNamespace = "::mlir::cbit";
}

def CBit_RegisterType : TypeDef<CBitDialect, "Register"> {
  let mnemonic = "reg";
  let summary = "A static classical-bit register";
  let description = [{
    `!cbit.reg<N>` is a non-aliasing register of `N` classical bits. `N` must
    be positive and is part of the type.
  }];
  let parameters = (ins "int64_t":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

class CBitOp<string mnemonic, list<Trait> traits = []>
    : Op<CBitDialect, mnemonic, traits>;

def AllocOp : CBitOp<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a classical-bit register";
  let description = [{
    Allocates a static classical-bit register. `zero` initialization defines
    every element as false. Reading an element of an `undefined` register
    before a store is undefined behavior. The optional `mqt.register_name`
    discardable attribute records the source-level name.

    Example:
    ```mlir
    %c = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"}
        : !cbit.reg<2>
    ```
  }];

  let arguments = (ins CBit_InitializationAttr:$initialization);
  let results = (outs CBit_RegisterType:$result);
  let assemblyFormat = [{
    `(` qualified($initialization) `)` attr-dict `:` qualified(type($result))
  }];
}

def LoadOp : CBitOp<"load"> {
  let summary = "Load a classical bit";
  let description = [{
    Reads one `i1` value from a classical-bit register. A constant index must
    be within the static register width.

    Example:
    ```mlir
    %bit = cbit.load %c[%index] : !cbit.reg<2>
    ```
  }];

  let arguments = (ins Arg<CBit_RegisterType, "register", [MemRead]>:$reg,
      Index:$index);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $reg `[` $index `]` attr-dict `:` qualified(type($reg))
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def CompareOp : CBitOp<"cmp"> {
  let summary = "Compare a classical-bit register with an integer";
  let description = [{
    Compares the complete register with an unsigned integer of the same width.
    The predicate must be `eq`, `ne`, `ult`, `ule`, `ugt`, or `uge`.

    Example:
    ```mlir
    %matches = cbit.cmp eq, %c, 1 : i2 : !cbit.reg<2>
    ```
  }];

  let arguments = (ins CBit_ComparisonPredicate:$predicate,
      Arg<CBit_RegisterType, "register", [MemRead]>:$reg, APIntAttr:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $predicate `,` $reg `,` $rhs attr-dict `:` qualified(type($reg))
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def StoreOp : CBitOp<"store"> {
  let summary = "Store a classical bit";
  let description = [{
    Writes one `i1` value to a classical-bit register. A constant index must be
    within the static register width.

    Example:
    ```mlir
    cbit.store %bit, %c[%index] : !cbit.reg<2>
    ```
  }];

  let arguments = (ins I1:$value,
      Arg<CBit_RegisterType, "register", [MemWrite]>:$reg, Index:$index);
  let assemblyFormat = [{
    $value `,` $reg `[` $index `]` attr-dict `:` qualified(type($reg))
  }];
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_CBIT_IR_CBITOPS_TD
