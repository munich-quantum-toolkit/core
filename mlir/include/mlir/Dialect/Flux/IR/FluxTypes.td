// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef FLUX_TYPES
#define FLUX_TYPES

include "mlir/Dialect/Flux/IR/FluxDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Flux Type Definitions
//===----------------------------------------------------------------------===//

class Flux_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<Flux_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Flux_QubitType : Flux_Type<"Qubit", "qubit"> {
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

#endif // FLUX_TYPES
