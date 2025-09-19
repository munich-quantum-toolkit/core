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
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <stdexcept>

namespace mqt::ir::common {
struct DefinitionMatrixElement {
  enum class Type {
    Value,
    ParameterIndex,
  };
  enum class Transformation {
    Identity,
    Sin,
    Cos,
  };

  double value;
  Type type;
  Transformation transformation = Transformation::Identity;

  double operator()() {
    if (type == Type::Value) {
      switch (transformation) {
      case Transformation::Identity:
        return value;
      case Transformation::Sin:
        return std::sin(value);
      case Transformation::Cos:
        return std::cos(value);
      }
      return value;
    } else {
      // TODO
    }
  }
};

template <std::size_t NumQubits> struct DefinitionMatrix {
  static constexpr std::size_t MatrixSize = 1 << NumQubits;
  std::array<double(*)(mlir::ValueRange), MatrixSize * MatrixSize> matrix;
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
    [[nodiscard]] static double getDefinitionMatrix(mlir::Operation* op, int x,
                                                    int y) {
      return 0.0;
      // return Matrix[y * (1 << N) + x]();
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
