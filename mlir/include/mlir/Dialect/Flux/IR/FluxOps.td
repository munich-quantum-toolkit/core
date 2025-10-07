// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef FluxOPS
#define FluxOPS

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def FluxDialect : Dialect {
    let name = "flux";

    let summary = "The Flux (value semantics) dialect for quantum computing.";

    let description = [{
        The Flux dialect uses **value semantics** where quantum operations
        consume input qubits and produce new output values, following the
        functional programming and SSA paradigm. This model enables:

        - Powerful compiler optimizations through clear dataflow
        - Safe reordering and parallelization analysis
        - Advanced transformation passes
        - Explicit dependency tracking

        The name "Flux" captures the flowing, transformative nature of
        value-based representations—quantum states flow through operations,
        each transformation producing new values like a river flowing through
        a landscape.

        Example:
        ```mlir
        %q_out = flux.h %q_in                        // Consumes %q_in, produces %q_out
        %q0_out, %q1_out = flux.swap %q0_in, %q1_in  // Consumes inputs, produces outputs
        ```
    }];

    let cppNamespace = "::mlir::flux";

    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Flux Type Definitions
//===----------------------------------------------------------------------===//

class FluxType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<FluxDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : FluxType<"Qubit", "qubit", [MemRefElementTypeInterface]> {
    let summary = "Flux qubit value type";
    let description = [{
        The `!flux.qubit` type represents an SSA value holding a quantum bit
        in the Flux dialect. Operations using this type consume input qubits
        and produce new output qubits following value semantics and the SSA
        paradigm, enabling powerful dataflow analysis and optimization.

        Example:
        ```mlir
        %q0 = flux.alloc : !flux.qubit
        %q1 = flux.h %q0 : !flux.qubit -> !flux.qubit
        %q2 = flux.x %q1 : !flux.qubit -> !flux.qubit
        ```
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class FluxOp<string mnemonic, list<Trait> traits = []> :
    Op<FluxDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def AllocOp : FluxOp<"alloc", [Pure]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns an SSA value representing it.
        The qubit is initialized to the |0⟩ state.

        Example:
        ```mlir
        %q = flux.alloc : !flux.qubit
        ```
    }];

    let results = (outs QubitType:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def DeallocOp : FluxOp<"dealloc", [Pure]> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        flux.dealloc %q : !flux.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

def StaticOp : FluxOp<"static", [Pure]> {
    let summary = "Retrieve a static qubit by index";
    let description = [{
        The `flux.static` operation produces an SSA value representing a qubit
        identified by a static index. This is useful for referring to fixed
        qubits in a quantum program or to hardware-mapped qubits.

        Example:
        ```mlir
        %q = flux.static 0 : !flux.qubit
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def MeasureOp : FluxOp<"measure", [Pure]> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning both the output qubit and a classical bit result.

        Example:
        ```mlir
        %q_out, %c = flux.measure %q_in : !flux.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit_in);
    let results = (outs QubitType:$qubit_out, I1:$result);
}

def ResetOp : FluxOp<"reset", [Pure, Idempotent, SameOperandsAndResultType]> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state,
        and returns the reset qubit.

        Example:
        ```mlir
        %q_out = flux.reset %q_in : !flux.qubit -> !flux.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let hasCanonicalizer = 1;
}

#endif // FluxOPS
