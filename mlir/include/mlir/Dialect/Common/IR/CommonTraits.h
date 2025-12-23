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

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <stdexcept>

namespace mqt::ir::common {

template <std::size_t NumQubits> struct DefinitionMatrix {
  static constexpr std::size_t MatrixSize = 1 << NumQubits;

  template <typename T>
  using MatrixType = std::array<T, MatrixSize * MatrixSize>;

  MatrixType<double (*)(mlir::ValueRange)> matrix;

  static constexpr std::size_t index(std::size_t x, std::size_t y) {
    return (y * MatrixSize) + x;
  }

  constexpr MatrixType<double> getMatrix(mlir::ValueRange params) {
    // TODO? lazy-initialized cache
    MatrixType<double> result;
    static_assert(result.size() == matrix.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = matrix[i](params);
    }
    return result;
  }
};

template <size_t N, DefinitionMatrix<N> Matrix> class TargetArityTrait {
public:
  template <typename ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
  public:
    [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
      auto unitaryOp = mlir::cast<ConcreteOp>(op);
      if (const auto size = unitaryOp.getInQubits().size(); size != N) {
        return op->emitError()
               << "number of input qubits (" << size << ") must be " << N;
      }
      return mlir::success();
    }

    [[nodiscard]] static auto getDefinitionMatrix() { return Matrix; }
    [[nodiscard]] static auto getDefinitionMatrix(mlir::Operation* op) {
      auto concreteOp = mlir::cast<ConcreteOp>(op);
      return Matrix.getMatrix(concreteOp.getParams());
    }
    [[nodiscard]] static double getDefinitionMatrixElement(mlir::Operation* op,
                                                           std::size_t x,
                                                           std::size_t y) {
      return getDefinitionMatrix(op).at(DefinitionMatrix<N>::index(x, y));
    }
  };
};

template <size_t N> class ParameterArityTrait {
public:
  template <typename ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
  public:
    [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
      auto paramOp = mlir::cast<ConcreteOp>(op);
      const auto& params = paramOp.getParams();
      const auto& staticParams = paramOp.getStaticParams();
      const auto numParams =
          params.size() + (staticParams.has_value() ? staticParams->size() : 0);
      if (numParams != N) {
        return op->emitError() << "operation expects exactly " << N
                               << " parameters but got " << numParams;
      }
      const auto& paramsMask = paramOp.getParamsMask();
      if (!params.empty() && staticParams.has_value() &&
          !paramsMask.has_value()) {
        return op->emitError() << "operation has mixed dynamic and static "
                                  "parameters but no parameter mask";
      }
      if (paramsMask.has_value() && paramsMask->size() != N) {
        return op->emitError() << "operation expects exactly " << N
                               << " parameters but has a parameter mask with "
                               << paramsMask->size() << " entries";
      }
      if (paramsMask.has_value()) {
        const auto trueEntries = static_cast<std::size_t>(std::count_if(
            paramsMask->begin(), paramsMask->end(), [](bool b) { return b; }));
        if ((!staticParams.has_value() || staticParams->empty()) &&
            trueEntries != 0) {
          return op->emitError() << "operation has no static parameter but has "
                                    "a parameter mask with "
                                 << trueEntries << " true entries";
        }
        if (const auto size = staticParams->size(); size != trueEntries) {
          return op->emitError()
                 << "operation has " << size
                 << " static parameter(s) but has a parameter mask with "
                 << trueEntries << " true entries";
        }
      }
      return mlir::success();
    }
  };
};

template <typename ConcreteOp>
class NoControlTrait
    : public mlir::OpTrait::TraitBase<ConcreteOp, NoControlTrait> {
public:
  [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
    if (auto unitaryOp = mlir::cast<ConcreteOp>(op); unitaryOp.isControlled()) {
      return op->emitOpError()
             << "Gate marked as NoControl should not have control qubits";
    }
    return mlir::success();
  }
};

template <typename ConcreteOp>
class UniqueSizeDefinitionTrait
    : public mlir::OpTrait::TraitBase<ConcreteOp, UniqueSizeDefinitionTrait> {
public:
  [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
    auto castOp = mlir::cast<ConcreteOp>(op);
    const auto hasAttr = op->hasAttr("size_attr");
    const auto hasOperand = castOp.getSize() != nullptr;
    if (!(hasAttr ^ hasOperand)) {
      return op->emitOpError()
             << "exactly one attribute ("
             << (hasAttr ? std::to_string(
                               op->getAttrOfType<mlir::IntegerAttr>("size_attr")
                                   .getInt())
                         : "undefined")
             << ") or operand (" << castOp.getSize()
             << ") must be provided for 'size'";
    }
    return mlir::success();
  }
};

template <typename ConcreteOp>
class UniqueIndexDefinitionTrait
    : public mlir::OpTrait::TraitBase<ConcreteOp, UniqueIndexDefinitionTrait> {
public:
  [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
    auto castOp = mlir::cast<ConcreteOp>(op);
    const auto hasAttr = op->hasAttr("index_attr");
    const auto hasOperand = castOp.getIndex() != nullptr;
    if (!(hasAttr ^ hasOperand)) {
      return op->emitOpError()
             << "exactly one attribute ("
             << (hasAttr ? std::to_string(op->getAttrOfType<mlir::IntegerAttr>(
                                                "index_attr")
                                              .getInt())
                         : "undefined")
             << ") or operand (" << castOp.getIndex()
             << ") must be provided for 'index'";
    }
    return mlir::success();
  }
};

} // namespace mqt::ir::common
