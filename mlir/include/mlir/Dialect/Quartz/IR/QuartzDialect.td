// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QUARTZ_DIALECT
#define QUARTZ_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Quartz Dialect Definition
//===----------------------------------------------------------------------===//

def Quartz_Dialect : Dialect {
    let name = "quartz";

    let summary = "The Quartz (reference semantics) dialect for quantum computing.";

    let description = [{
        The Quartz dialect uses **reference semantics** where quantum operations
        modify qubits in place, similar to how hardware physically transforms
        quantum states. This model provides:

        - Natural mapping to hardware execution models
        - Intuitive representation for circuit descriptions
        - Direct compatibility with imperative quantum programming languages
        - Straightforward backend code generation

        The name "Quartz" reflects the crystalline, structured nature of
        hardware-oriented representations—operations have fixed positions and
        transform states in place, like atoms in a crystal lattice.

        Example:
        ```mlir
        quartz.h %q              // Applies Hadamard to qubit %q in place
        quartz.swap %q0, %q1     // Applies SWAP using %q0, %q1 as targets
        ```
    }];

    let cppNamespace = "::mlir::quartz";

    let useDefaultTypePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif // QUARTZ_DIALECT
