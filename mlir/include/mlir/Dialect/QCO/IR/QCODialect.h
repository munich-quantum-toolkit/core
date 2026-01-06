/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include <mlir/Interfaces/InferTypeOpInterface.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <Eigen/Core>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <string>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>
#include <variant>

#define DIALECT_NAME_QCO "qco"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QCO/IR/QCOOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h.inc"
#include "mlir/Dialect/QCO/IR/QCOOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::qco {

/**
 * @brief Retrieve C++ type of static mlir::Value.
 * @details The returned float attribute can be used to get the value of the
 *          given parameter as a C++ type.
 */
[[nodiscard]] inline std::optional<double>
tryGetParameterAsDouble(UnitaryOpInterface op, size_t i);
/**
 * @brief Trait for operations with a fixed number of target qubits and
 * parameters
 * @details This trait indicates that an operation has a fixed number of target
 * qubits and parameters, specified by the template parameters T and P. This is
 * helpful for defining operations with known arities, allowing for static
 * verification and code generation optimizations.
 * @tparam T The target arity.
 * @tparam P The parameter arity.
 * @tparam MatrixDefinition A function returning the matrix definition of the
 *                          operation. The operation will be provided as the
 *                          only argument of the function. If the operation does
 *                          not have a matrix definition, set this value to
 *                          nullptr.
 */
template <size_t T, size_t P,
          Eigen::Matrix<std::complex<double>, (1 << T), (1 << T)> (
              *MatrixDefinition)(UnitaryOpInterface)>
class TargetAndParameterArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    static size_t getNumQubits() { return T; }
    static size_t getNumTargets() { return T; }
    static size_t getNumControls() { return 0; }

    Value getInputQubit(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have qubits");
      }
      if (i >= T) {
        llvm::reportFatalUsageError("Qubit index out of bounds");
      }
      return this->getOperation()->getOperand(i);
    }
    Value getOutputQubit(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have qubits");
      }
      if (i >= T) {
        llvm::reportFatalUsageError("Qubit index out of bounds");
      }
      return this->getOperation()->getResult(i);
    }

    Value getInputTarget(const size_t i) { return getInputQubit(i); }
    Value getOutputTarget(const size_t i) { return getOutputQubit(i); }

    static Value getInputControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }
    static Value getOutputControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }

    static size_t getNumParams() { return P; }

    Value getParameter(const size_t i) {
      if (i >= P) {
        llvm::reportFatalUsageError("Parameter index out of bounds");
      }
      return this->getOperation()->getOperand(T + i);
    }

    /**
     * @brief Retrieve mlir::FloatAttr of static mlir::Value.
     * @details The returned float attribute can be used to get the value of the
     *          given parameter as a C++ type or perform other mlir operations.
     */
    [[nodiscard]] static FloatAttr getStaticParameter(Value param) {
      auto constantOp = param.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        return nullptr;
      }
      return dyn_cast<FloatAttr>(constantOp.getValue());
    }

    using UnitaryMatrixType =
        Eigen::Matrix<std::complex<double>, 1 << T, 1 << T>;

    [[nodiscard]] UnitaryMatrixType getUnitaryMatrixDefinition() const
      requires(MatrixDefinition != nullptr)
    {
      const auto* op = this->getConstOperation();
      return MatrixDefinition(llvm::dyn_cast<UnitaryOpInterface>(op));
    }

    Value getInputForOutput(Value output) {
      const auto& op = this->getOperation();
      for (size_t i = 0; i < T; ++i) {
        if (output == op->getResult(i)) {
          return op->getOperand(i);
        }
      }
      llvm::reportFatalUsageError(
          "Given qubit is not an output of the operation");
    }
    Value getOutputForInput(Value input) {
      const auto& op = this->getOperation();
      for (size_t i = 0; i < T; ++i) {
        if (input == op->getOperand(i)) {
          return op->getResult(i);
        }
      }
      llvm::reportFatalUsageError(
          "Given qubit is not an input of the operation");
    }

  protected:
    [[nodiscard]] const Operation* getConstOperation() const {
      auto* concrete = static_cast<const ConcreteType*>(this);
      // use dereference operator instead of getOperation() of mlir::Op; the
      // operator provides a const overload, getOperation() does not
      return *concrete;
    }
  };
};

} // namespace mlir::qco

// #include "mlir/Dialect/QCO/IR/QCOInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations Helpers
//===----------------------------------------------------------------------===//

namespace mlir::qco {

[[nodiscard]] inline std::optional<double>
tryGetParameterAsDouble(UnitaryOpInterface op, size_t i) {
  using DummyArityType = TargetAndParameterArityTrait<0, 0, nullptr>;
  const auto param = op.getParameter(i);
  const auto floatAttr =
      DummyArityType::Impl<arith::ConstantOp>::getStaticParameter(param);
  if (!floatAttr) {
    return std::nullopt;
  }
  return floatAttr.getValueAsDouble();
}

[[nodiscard]] inline Eigen::MatrixXcd getBlockMatrix(size_t dim, mlir::Region& region) {
  assert(dim == 1); // TODO: remove once permutations are properly handled

  Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(1 << dim, 1 << dim);
  for (auto&& block : region) {
    for (auto&& op : block) {
      auto unitaryOp = llvm::dyn_cast<mlir::qco::UnitaryOpInterface>(op);
      if (unitaryOp) {
        return result;
      }
      auto matrix = unitaryOp.getUnitaryMatrix();
      size_t matrixDim = matrix.cols();
      if (matrixDim < dim) {
        // TODO: adjust according to permutation (position in targets?)
        auto paddingDim = dim - matrixDim;
        auto padding = Eigen::MatrixXcd::Identity(1 << paddingDim, 1 << paddingDim);
        matrix = Eigen::kroneckerProduct(matrix, padding);
      }
      result = matrix * result;
    }
  }
  return result;
}

} // namespace mlir::qco

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/QCO/IR/QCOOps.h.inc" // IWYU pragma: export
