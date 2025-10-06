// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef FLUX_DIALECT
#define FLUX_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Flux Dialect Definition
//===----------------------------------------------------------------------===//

def Flux_Dialect : Dialect {
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

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif // FLUX_DIALECT
