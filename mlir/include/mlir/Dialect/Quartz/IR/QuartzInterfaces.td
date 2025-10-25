// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QUARTZ_INTERFACES
#define QUARTZ_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UnitaryOpInterface
//===----------------------------------------------------------------------===//

def Quartz_UnitaryOpInterface : OpInterface<"UnitaryOpInterface"> {
    let description = [{
        This interface provides a unified API for all operations that apply or
        produce a unitary transformation. This includes base gates, user-defined
        gates, modifier operations (control, inverse, power), and sequences.

        The interface enables uniform introspection and composition capabilities
        across all unitary operations in the Quartz dialect.
    }];

    let cppNamespace = "::mlir::quartz";

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
            "Returns the number of positive control qubits.",
            "size_t", "getNumPosControls", (ins)
        >,
        InterfaceMethod<
            "Returns the number of negative control qubits.",
            "size_t", "getNumNegControls", (ins)
        >,
        InterfaceMethod<
            "Returns the i-th qubit (targets + controls combined).",
            "::mlir::Value", "getQubit", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th target qubit.",
            "::mlir::Value", "getTarget", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th positive control qubit.",
            "::mlir::Value", "getPosControl", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th negative control qubit.",
            "::mlir::Value", "getNegControl", (ins "size_t":$i)
        >,

        // Parameter handling
        InterfaceMethod<
            "Returns the number of parameters.",
            "size_t", "getNumParams", (ins)
        >,
        InterfaceMethod<
            "Returns the i-th parameter.",
            "ParameterDescriptor", "getParameter", (ins "size_t":$i)
        >,
        /// TODO: Add convenience methods as necessary

        // Convenience methods
        InterfaceMethod<
            "Returns true if the operation has any control qubits, otherwise false.",
            "bool", "isControlled", (ins),
            [{ return getNumControls(impl, tablegen_opaque_val) > 0; }]
        >,
        InterfaceMethod<
            "Returns true if the operation only acts on a single qubit.",
            "bool", "isSingleQubit", (ins),
            [{ return getNumQubits(impl, tablegen_opaque_val) == 1; }]
        >,
        /// TODO: I am fairly sure that there are quite some further convenience methods that would be helpful here, e.g., whether it is a two-qubit gate

        // Matrix extraction
        InterfaceMethod<
            "Returns true if the operation has a static unitary matrix.",
            "bool", "hasStaticUnitary", (ins)
        >,
        InterfaceMethod<
            "Attempts to extract the static unitary matrix. "
            "Returns std::nullopt if the operation is symbolic or dynamic.",
            "std::optional<::mlir::DenseElementsAttr>", "tryGetStaticMatrix", (ins)
        >,

        // Modifier state
        /// TODO: Implement interface methods to query modifier state

        // Identification
        InterfaceMethod<
            "Returns the base symbol/mnemonic of the operation.",
            "::llvm::StringRef", "getBaseSymbol", (ins)
        >,
        InterfaceMethod<
            "Returns the canonical descriptor of the operation. This can be used for equivalence testing.",
            "CanonicalDescriptor", "getCanonicalDescriptor", (ins)
        >,
    ];
}

#endif // QUARTZ_INTERFACES
