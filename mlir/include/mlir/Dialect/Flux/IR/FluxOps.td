// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef FLUX_OPS
#define FLUX_OPS

include "mlir/Dialect/Flux/IR/FluxDialect.td"
include "mlir/Dialect/Flux/IR/FluxTypes.td"
include "mlir/Dialect/Flux/IR/FluxInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class Flux_Op<string mnemonic, list<Trait> traits = []> :
    Op<Flux_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def Flux_AllocOp : Flux_Op<"alloc", [Pure]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns an SSA value representing it.
        The qubit is initialized to the |0⟩ state.

        Example:
        ```mlir
        %q = flux.alloc : !flux.qubit
        ```
    }];

    let results = (outs Flux_QubitType:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def Flux_DeallocOp : Flux_Op<"dealloc"> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        flux.dealloc %q : !flux.qubit
        ```
    }];

    let arguments = (ins Flux_QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def Flux_MeasureOp : Flux_Op<"measure"> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning both the output qubit and a classical bit result.

        Example:
        ```mlir
        %q_out, %c = flux.measure %q_in : !flux.qubit
        ```
    }];

    let arguments = (ins Flux_QubitType:$qubit_in);
    let results = (outs Flux_QubitType:$qubit_out, I1:$result);
}

def Flux_ResetOp : Flux_Op<"reset"> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state,
        and returns the reset qubit.

        Example:
        ```mlir
        %q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
        ```
    }];

    let arguments = (ins Flux_QubitType:$qubit_in);
    let results = (outs Flux_QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let hasCanonicalizer = 1;
}

#endif // FLUX_OPS
