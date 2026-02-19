// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QC_INTERFACES
#define QC_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UnitaryOpInterface
//===----------------------------------------------------------------------===//

def UnitaryOpInterface : OpInterface<"UnitaryOpInterface"> {
    let description = [{
        This interface provides a unified API for all operations that apply or
        produce a unitary transformation. This includes base gates, user-defined
        gates, modifier operations (control, inverse, power), and sequences.

        The interface enables uniform introspection and composition capabilities
        across all unitary operations in the QC dialect.
    }];

    let cppNamespace = "::mlir::qc";

    let methods = [
        // Qubit accessors
        InterfaceMethod<
            "Returns the number of qubits acted on by the unitary operation.",
            "size_t", "getNumQubits", (ins)
        >,
        InterfaceMethod<
            "Returns the number of target qubits (excluding control qubits).",
            "size_t", "getNumTargets", (ins)
        >,
        InterfaceMethod<
            "Returns the number of control qubits (both positive and negative).",
            "size_t", "getNumControls", (ins)
        >,
        InterfaceMethod<
            "Returns the i-th qubit (targets + controls combined).",
            "Value", "getQubit", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th target qubit.",
            "Value", "getTarget", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th control qubit.",
            "Value", "getControl", (ins "size_t":$i)
        >,

        // Parameter handling
        InterfaceMethod<
            "Returns the number of parameters.",
            "size_t", "getNumParams", (ins)
        >,
        InterfaceMethod<
            "Returns the i-th parameter.",
            "Value", "getParameter", (ins "size_t":$i)
        >,

        // Convenience methods
        InterfaceMethod<
            "Returns true if the operation has any control qubits, otherwise false.",
            "bool", "isControlled", (ins),
            [{ return $_op.getNumControls() > 0; }]
        >,
        InterfaceMethod<
            "Returns true if the operation only acts on a single qubit.",
            "bool", "isSingleQubit", (ins),
            [{ return $_op.getNumQubits() == 1; }]
        >,
        InterfaceMethod<
            "Returns true if the operation acts on two qubits.",
            "bool", "isTwoQubit", (ins),
            [{ return $_op.getNumQubits() == 2; }]
        >,

        // Identification
        InterfaceMethod<
            "Returns the base symbol/mnemonic of the operation.",
            "StringRef", "getBaseSymbol", (ins)
        >,
    ];
}

#endif // QC_INTERFACES
