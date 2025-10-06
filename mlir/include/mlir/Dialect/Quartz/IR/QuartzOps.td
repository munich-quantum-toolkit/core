// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QUARTZ_OPS
#define QUARTZ_OPS

include "mlir/Dialect/Quartz/IR/QuartzDialect.td"
include "mlir/Dialect/Quartz/IR/QuartzTypes.td"
include "mlir/Dialect/Quartz/IR/QuartzInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class Quartz_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quartz_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def Quartz_AllocOp : Quartz_Op<"alloc", [Pure]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns a reference to it.
        The qubit is initialized to the |0⟩ state.

        Example:
        ```mlir
        %q = quartz.alloc : !quartz.qubit
        ```
    }];

    let results = (outs Quartz_QubitType:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def Quartz_DeallocOp : Quartz_Op<"dealloc"> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        quartz.dealloc %q : !quartz.qubit
        ```
    }];

    let arguments = (ins Quartz_QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def Quartz_MeasureOp : Quartz_Op<"measure"> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning a classical bit result.

        Example:
        ```mlir
        %c = quartz.measure %q : !quartz.qubit -> i1
        ```
    }];

    let arguments = (ins Quartz_QubitType:$qubit);
    let results = (outs I1:$result);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit) `->` type($result)";
}

def Quartz_ResetOp : Quartz_Op<"reset"> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state.

        Example:
        ```mlir
        quartz.reset %q : !quartz.qubit
        ```
    }];

    let arguments = (ins Quartz_QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";

    let hasCanonicalizer = 1;
}

#endif // QUARTZ_OPS
