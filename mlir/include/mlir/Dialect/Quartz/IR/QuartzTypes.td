// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QUARTZ_TYPES
#define QUARTZ_TYPES

include "mlir/Dialect/Quartz/IR/QuartzDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Quartz Type Definitions
//===----------------------------------------------------------------------===//

class Quartz_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Quartz_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Quartz_QubitType : Quartz_Type<"Qubit", "qubit", [MemRefElementTypeInterface]> {
    let summary = "Quartz qubit reference type";
    let description = [{
        The `!quartz.qubit` type represents a reference to a quantum bit in the
        Quartz dialect. Operations using this type modify qubits in place using
        reference semantics, similar to how classical imperative languages handle
        mutable references.

        This type can be used as an element type in memref to represent quantum
        registers:
        ```mlir
        %qreg = memref.alloc() : memref<2x!quartz.qubit>
        %q0 = memref.load %qreg[%c0] : memref<2x!quartz.qubit>
        ```
    }];
}

#endif // QUARTZ_TYPES
