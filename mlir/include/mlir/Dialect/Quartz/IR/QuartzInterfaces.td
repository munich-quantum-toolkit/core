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
        gates, modified operations (control, inverse, power), and sequences.

        The interface enables uniform introspection and composition capabilities
        across all unitary operations in the Quartz dialect.
    }];

    let cppNamespace = "::mlir::quartz";

    let methods = [
        // Qubit accessors
        InterfaceMethod<
            "Returns the number of target qubits (excluding control qubits).",
            "size_t", "getNumTargets", (ins)
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

        // Value semantics threading (identity in reference semantics)
        InterfaceMethod<
            "Returns the i-th input qubit (targets + controls combined).",
            "::mlir::Value", "getInput", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns the i-th output qubit (targets + controls combined). "
            "In reference semantics, returns the same as getInput.",
            "::mlir::Value", "getOutput", (ins "size_t":$i),
            /*methodBody=*/"", /*defaultImplementation=*/[{
                return $_op.getInput(i);
            }]
        >,
        InterfaceMethod<
            "Returns the output qubit corresponding to the given input qubit. "
            "In reference semantics, returns the input itself.",
            "::mlir::Value", "getOutputForInput", (ins "::mlir::Value":$input),
            /*methodBody=*/"", /*defaultImplementation=*/[{
                return input;
            }]
        >,
        InterfaceMethod<
            "Returns the input qubit corresponding to the given output qubit. "
            "In reference semantics, returns the output itself.",
            "::mlir::Value", "getInputForOutput", (ins "::mlir::Value":$output),
            /*methodBody=*/"", /*defaultImplementation=*/[{
                return output;
            }]
        >,

        // Parameter handling
        InterfaceMethod<
            "Returns the number of parameters.",
            "size_t", "getNumParams", (ins)
        >,

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
        InterfaceMethod<
            "Returns true if the operation is inverted.",
            "bool", "isInverted", (ins),
            /*methodBody=*/"", /*defaultImplementation=*/[{
                return false;
            }]
        >,
        InterfaceMethod<
            "Returns the power exponent if applicable, otherwise std::nullopt.",
            "std::optional<double>", "getPower", (ins),
            /*methodBody=*/"", /*defaultImplementation=*/[{
                return std::nullopt;
            }]
        >,

        // Identification
        InterfaceMethod<
            "Returns the base symbol/mnemonic of the operation.",
            "::llvm::StringRef", "getBaseSymbol", (ins)
        >,
    ];
}

#endif // QUARTZ_INTERFACES
