// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QCO_IR_QCOUNITARYMATRIXINTERFACES_TD
#define MLIR_DIALECT_QCO_IR_QCOUNITARYMATRIXINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// UnitaryMatrixOpInterface
//===----------------------------------------------------------------------===//

def UnitaryMatrixOpInterface : OpInterface<"UnitaryMatrixOpInterface"> {
  let description = [{
        This interface provides a unified API for all operations in the QCO
        dialect that can expose their unitary matrix representation.

        This interface is intentionally separate from `UnitaryOpInterface` to
        avoid propagating the Eigen dependency to all unitary ops.
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

  let methods =
      [InterfaceMethod<"Populates the given 1x1 unitary matrix if possible.",
                       "bool", "getUnitaryMatrix1x1",
                       (ins "Eigen::Matrix<std::complex<double>, 1, 1>&":$out),
                       unitaryMatrixMethodBody>,
       InterfaceMethod<"Populates the given 2x2 unitary matrix if possible.",
                       "bool", "getUnitaryMatrix2x2",
                       (ins "Eigen::Matrix2cd&":$out), unitaryMatrixMethodBody>,
       InterfaceMethod<"Populates the given 4x4 unitary matrix if possible.",
                       "bool", "getUnitaryMatrix4x4",
                       (ins "Eigen::Matrix4cd&":$out), unitaryMatrixMethodBody>,
       InterfaceMethod<"Populates the given dynamic unitary matrix.", "bool",
                       "getUnitaryMatrixDynamic",
                       (ins "Eigen::MatrixXcd&":$out),
                       unitaryMatrixMethodBody>];

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

#endif // MLIR_DIALECT_QCO_IR_QCOUNITARYMATRIXINTERFACES_TD
