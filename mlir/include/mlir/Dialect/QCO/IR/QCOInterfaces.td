// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QCO_INTERFACES
#define QCO_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UnitaryOpInterface
//===----------------------------------------------------------------------===//

class UnitaryOpInterface<code matrixDefinition = [{}]> : OpInterface<"UnitaryOpInterface"> {
    let description = [{
        This interface provides a unified API for all operations that apply or
        produce a unitary transformation in the QCO dialect. This includes base
        gates, user-defined gates, modifier operations (control, inverse, power),
        and sequences.

        The interface enables uniform introspection and composition capabilities
        across all unitary operations with value semantics.
    }];

    let cppNamespace = "::mlir::qco";

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
            "Returns the i-th input qubit (targets + controls combined).",
            "Value", "getInputQubit", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th output qubit (targets + controls combined).",
            "Value", "getOutputQubit", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th target input qubit.",
            "Value", "getInputTarget", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th target output qubit.",
            "Value", "getOutputTarget", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th control input qubit.",
            "Value", "getInputControl", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th control output qubit.",
            "Value", "getOutputControl", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the input qubit corresponding to the given output qubit.",
            "Value", "getInputForOutput", (ins "Value":$output)
        >,
        InterfaceMethod<
            "Returns the output qubit corresponding to the given input qubit.",
            "Value", "getOutputForInput", (ins "Value":$input)
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
            [{ return getNumControls(impl, tablegen_opaque_val) > 0; }]
        >,
        InterfaceMethod<
            "Returns true if the operation only acts on a single qubit.",
            "bool", "isSingleQubit", (ins),
            [{ return getNumQubits(impl, tablegen_opaque_val) == 1; }]
        >,
        InterfaceMethod<
            "Returns true if the operation acts on two qubits.",
            "bool", "isTwoQubit", (ins),
            [{ return getNumQubits(impl, tablegen_opaque_val) == 2; }]
        >,

        // Identification
        InterfaceMethod<
            "Returns the base symbol/mnemonic of the operation.",
            "StringRef", "getBaseSymbol", (ins)
        >,

        // Unitary matrix methods
        InterfaceMethod<
            "Returns the unitary matrix definition of the operation.",
            "Eigen::MatrixXcd", "getUnitaryMatrix", (ins),
            matrixDefinition, [{ throw std::logic_error{ "Operation has no matrix definition!" } }]
        >,
    ];

    let extraClassDeclaration = [{
      template<typename MatrixType>
      MatrixType getUnitaryMatrix() {
        static_assert(std::is_same_v<MatrixType, UnitaryMatrixType>,
         "Provided matrix type does not equal expected type for the contained operation.");
      }
    }];
}

#endif // QCO_INTERFACES
