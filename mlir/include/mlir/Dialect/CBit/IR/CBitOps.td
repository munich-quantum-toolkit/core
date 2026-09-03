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

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Attributes and types
//===----------------------------------------------------------------------===//

def CBit_InitializationAttr
    : EnumAttr<CBitDialect, CBit_Initialization, "init"> {
  let assemblyFormat = "`<` $value `>`";
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

def ReadOp : CBitOp<"read"> {
  let summary = "Read a classical-bit register as an integer";
  let description = [{
    Reads the complete register as an unsigned fixed-width integer. Register
    element zero is the least-significant result bit, and the result width must
    equal the static register width.

    Example:
    ```mlir
    %value = cbit.read %c : !cbit.reg<3> -> i3
    ```
  }];

  let arguments = (ins Arg<CBit_RegisterType, "register", [MemRead]>:$reg);
  let results = (outs AnyInteger:$result);
  let assemblyFormat = [{
    $reg attr-dict `:` qualified(type($reg)) `->` qualified(type($result))
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def WriteOp : CBitOp<"write"> {
  let summary = "Write an integer to a classical-bit register";
  let description = [{
    Writes a fixed-width integer to the complete register. Integer bit zero is
    stored in register element zero, and the value width must equal the static
    register width.

    Example:
    ```mlir
    cbit.write %value, %c : i3, !cbit.reg<3>
    ```
  }];

  let arguments = (ins AnyInteger:$value,
      Arg<CBit_RegisterType, "register", [MemWrite]>:$reg);
  let assemblyFormat = [{
    $value `,` $reg attr-dict `:` qualified(type($value)) `,`
        qualified(type($reg))
  }];
  let hasVerifier = 1;
}

def CompareOp : CBitOp<"cmp"> {
  let summary = "Compare a classical-bit register with an integer";
  let description = [{
    Compares the complete register with an integer of the same width. Signed
    predicates interpret the most significant register bit as a two's-complement
    sign bit.

    Example:
    ```mlir
    %matches = cbit.cmp eq, %c, 1 : i2 : !cbit.reg<2>
    ```
  }];

  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
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
