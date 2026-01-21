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

    // Generic implementation body for getUnitaryMatrix methods
    defvar unitaryMatrixMethodBody = [{
      auto process = [&]<typename MatrixType>(MatrixType&& m) -> bool {
          using TargetT = std::remove_cvref_t<decltype(out)>;
          using SourceT = std::remove_cvref_t<MatrixType>;

          constexpr bool isTargetDynamic =
              (TargetT::SizeAtCompileTime == Eigen::Dynamic);
          constexpr bool isSourceDynamic =
              (SourceT::SizeAtCompileTime == Eigen::Dynamic);

          // Case 1: Target is Dynamic. Always accepts source.
          if constexpr (isTargetDynamic) {
            out = std::forward<MatrixType>(m);
            return true;
          }
          // Case 2: Target is Fixed.
          else {
            // Case 2a: Source is Dynamic. Runtime dimension check required.
            if constexpr (isSourceDynamic) {
              if (m.rows() == static_cast<Eigen::Index>(TargetT::RowsAtCompileTime) &&
                  m.cols() == static_cast<Eigen::Index>(TargetT::ColsAtCompileTime))
                  [[likely]] {
                out = std::forward<MatrixType>(m);
                return true;
              }
            }
            // Case 2b: Source is Fixed. Compile-time check.
            else if constexpr (static_cast<Eigen::Index>(
                                   SourceT::RowsAtCompileTime) ==
                                   static_cast<Eigen::Index>(
                                       TargetT::RowsAtCompileTime) &&
                               static_cast<Eigen::Index>(
                                   SourceT::ColsAtCompileTime) ==
                                   static_cast<Eigen::Index>(
                                       TargetT::ColsAtCompileTime)) {
              out = std::forward<MatrixType>(m);
              return true;
            }
          }
          return false;
        };


      if constexpr (requires { $_op.getUnitaryMatrix().has_value(); }) {
        if (auto&& matrix = $_op.getUnitaryMatrix()) {
          return process(std::move(*matrix));
        }
        return false;
      } else if constexpr (requires { $_op.getUnitaryMatrix(); }) {
        return process($_op.getUnitaryMatrix());
      } else {
        llvm::reportFatalUsageError("Operation '" + $_op.getBaseSymbol() + "' has no unitary matrix definition!");
      }
    }];

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
            "Returns a range of all input qubits (targets + controls combined).",
            "ValueRange", "getInputQubits", (ins)
        >,
        InterfaceMethod<
            "Returns the i-th output qubit (targets + controls combined).",
            "Value", "getOutputQubit", (ins "size_t":$i)
        >,
        InterfaceMethod<
            "Returns a range of all output qubits (targets + controls combined).",
            "ValueRange", "getOutputQubits", (ins)
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

        // Unitary matrix helpers
        InterfaceMethod<
            "Populates the given 1x1 unitary matrix if possible.",
            "bool", "getUnitaryMatrix1x1",
            (ins "Eigen::Matrix<std::complex<double>, 1, 1>&":$out),
            unitaryMatrixMethodBody
        >,
        InterfaceMethod<
            "Populates the given 2x2 unitary matrix if possible.",
            "bool", "getUnitaryMatrix2x2",
            (ins "Eigen::Matrix2cd&":$out),
            unitaryMatrixMethodBody
        >,
        InterfaceMethod<
            "Populates the given 4x4 unitary matrix if possible.",
            "bool", "getUnitaryMatrix4x4",
            (ins "Eigen::Matrix4cd&":$out),
            unitaryMatrixMethodBody
        >,
        InterfaceMethod<
            "Populates the given dynamic unitary matrix.",
            "bool", "getUnitaryMatrixDynamic",
            (ins "Eigen::MatrixXcd&":$out),
            unitaryMatrixMethodBody
        >
    ];

    let extraClassDeclaration = [{
        template<typename MatrixType>
        std::optional<MatrixType> getUnitaryMatrix() {
          MatrixType out;
          bool result = false;

          // Dispatch to the appropriate fixed-size or dynamic method based on the
          // matrix type.
          if constexpr (MatrixType::RowsAtCompileTime == 1 &&
                        MatrixType::ColsAtCompileTime == 1) {
            result = this->getUnitaryMatrix1x1(out);
          } else if constexpr (MatrixType::RowsAtCompileTime == 2 &&
                               MatrixType::ColsAtCompileTime == 2) {
            result = this->getUnitaryMatrix2x2(out);
          } else if constexpr (MatrixType::RowsAtCompileTime == 4 &&
                               MatrixType::ColsAtCompileTime == 4) {
            result = this->getUnitaryMatrix4x4(out);
          } else if constexpr (MatrixType::SizeAtCompileTime == Eigen::Dynamic) {
            result = this->getUnitaryMatrixDynamic(out);
          } else {
            // Fallback: Try obtaining dynamic matrix and see if size matches
            Eigen::MatrixXcd dynamicOut;
            if (this->getUnitaryMatrixDynamic(dynamicOut)) {
              if (dynamicOut.rows() == MatrixType::RowsAtCompileTime &&
                  dynamicOut.cols() == MatrixType::ColsAtCompileTime) {
                out = dynamicOut;
                result = true;
              }
            }
          }

          if (result) {
            return out;
          }
          return std::nullopt;
        }
    }];
}

#endif // QCO_INTERFACES
