// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QCODIALECT
#define QCODIALECT

include "mlir/IR/DialectBase.td"

def QCODialect : Dialect {
    let name = "qco";

    let summary = "The QCO (value semantics) dialect for quantum computing.";

    let description = [{
        The QCO dialect uses **value semantics** where quantum operations
        consume input qubits and produce new output values, following the
        functional programming and SSA paradigm. This model enables:

        - Powerful compiler optimizations through clear dataflow
        - Safe reordering and parallelization analysis
        - Advanced transformation passes
        - Explicit dependency tracking

        The name "QCO" stands for "Quantum Circuit Optimization."

        Example:
        ```mlir
        %q_out = qco.h %q_in                       // Consumes %q_in, produces %q_out
        %q0_out, %q1_out = qco.swap %q0_in, %q1_in // Consumes inputs, produces outputs
        ```
    }];

    let cppNamespace = "::mlir::qco";

    let useDefaultTypePrinterParser = 1;
}

#endif // QCODIALECT
