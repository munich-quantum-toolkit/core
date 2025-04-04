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

#include <cstddef>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::common {
template <size_t N> class TargetArity {
public:
  template <typename ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
    [[nodiscard]] static mlir::LogicalResult verifyTrait(mlir::Operation* op) {
      auto unitaryOp = mlir::cast<ConcreteOp>(op);
      if (const auto size = unitaryOp.getInQubits().size(); size != N) {
        return op->emitError()
               << "number of input qubits (" << size << ") must be " << N;
      }
      return mlir::success();
    }
  };
};

template <size_t N> class ParameterArity {
public:
  template <typename ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
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

} // namespace mqt::ir::common
