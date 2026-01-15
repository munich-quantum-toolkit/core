// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

def UnitaryOpInterface : OpInterface<"UnitaryOpInterface"> {
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

        // Internal helper methods, should not be used by user
        InterfaceMethod<
            "Returns the unitary matrix definition of the operation.",
            [{
              std::variant<
                std::nullopt_t,
                Eigen::Matrix<std::complex<double>, 1, 1>,
                Eigen::Matrix2cd,
                Eigen::Matrix4cd,
                Eigen::MatrixXcd
              >
            }],
            "internalGetUnitaryVariant", (ins),
            [{
              if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
                // is matrix in std::optional
                if (auto&& matrix = $_op.getUnitaryMatrix()) {
                  return *matrix;
                }
                return std::nullopt;
              } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
                // is static matrix
                return $_op.getUnitaryMatrix();
              } else {
                llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
              }
            }]
        >,
    ];

    let extraClassDeclaration = [{
      template<typename MatrixType>
      std::optional<MatrixType> getUnitaryMatrix() {
        auto&& matrix = this->internalGetUnitaryVariant();

        if constexpr (std::is_same_v<std::remove_cvref_t<MatrixType>, Eigen::MatrixXcd>) {
          // cast to Eigen::MatrixXcd via visitor pattern
          auto&& visitor = [](auto&& m) -> std::optional<MatrixType> {
            // implicit cast to Eigen::MatrixXcd if variant holds a matrix type;
            // if variant holds std::nullopt_t, std::optional will be initialized with
            // the std::nullopt_t instance, so std::nullopt
            return m;
          };
          return std::visit(visitor, matrix);
        } else {
          // if an explicit size has been requested, use checked access to variant
          // to try to return the correct type
          if (auto* ptr = std::get_if<MatrixType>(&matrix)) {
            return *ptr;
          }
          // if variant contains a dynamically-sized matrix, check if it is possible
          // to convert it to the given fixed-size matrix
          if (auto* ptr = std::get_if<Eigen::MatrixXcd>(&matrix); ptr && ptr->size() == Eigen::Matrix4cd::SizeAtCompileTime) {
            return *ptr;
          }
          return std::nullopt;
        }
      }
    }];
}

#endif // QCO_INTERFACES
