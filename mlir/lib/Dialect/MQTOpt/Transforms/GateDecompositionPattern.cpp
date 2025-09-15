/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Helpers.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <cstddef>
#include <iterator>
#include <llvm/ADT/STLExtras.h>
#include <map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to cancel consecutive self-inverse operations.
 */
struct GateDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit GateDecompositionPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  dd::TwoQubitGateMatrix twoQubitIdentity = {
      {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto series = getTwoQubitSeries(op);
    if (series.size() <= 3) {
      return mlir::failure();
    }

    dd::TwoQubitGateMatrix unitaryMatrix = dd::opToTwoQubitGateMatrix(qc::I);
    for (auto&& gate : series) {
      if (auto gateMatrix = helpers::getUnitaryMatrix(gate)) {
        unitaryMatrix = helpers::multiply(unitaryMatrix, *gateMatrix);
      }
    }

    twoQubitDecompose(unitaryMatrix);

    return mlir::success();
  }

  [[nodiscard]] static llvm::SmallVector<UnitaryInterface>
  getTwoQubitSeries(UnitaryInterface op) {
    llvm::SmallVector<mlir::Value, 2> qubits(2);
    llvm::SmallVector<UnitaryInterface> result;

    if (helpers::isSingleQubitOperation(op)) {
      qubits = {op->getResult(0), mlir::Value{}};
    } else if (helpers::isTwoQubitOperation(op)) {
      qubits = op->getResults();
    } else {
      return result;
    }
    while (true) {
      for (auto&& user : op->getUsers()) {
        auto userUnitary = llvm::cast<UnitaryInterface>(user);
        if (helpers::isSingleQubitOperation(userUnitary)) {
          auto&& operand = userUnitary->getOperand(0);
          auto* it = llvm::find(qubits, operand);
          if (it == qubits.end()) {
            return result;
          }
          *it = userUnitary->getResult(0);

          result.push_back(userUnitary);
        } else if (helpers::isTwoQubitOperation(userUnitary)) {
          auto&& firstOperand = userUnitary->getOperand(0);
          auto&& secondOperand = userUnitary->getOperand(1);
          auto* firstQubitIt = llvm::find(qubits, firstOperand);
          auto* secondQubitIt = llvm::find(qubits, secondOperand);
          if (firstQubitIt == qubits.end() || secondQubitIt == qubits.end()) {
            return result;
          }
          *firstQubitIt = userUnitary->getResult(0);
          *secondQubitIt = userUnitary->getResult(1);

          result.push_back(userUnitary);
        } else {
          return result;
        }
      }
      return result;
    }
  }

  void twoQubitDecompose(dd::TwoQubitGateMatrix unitaryMatrix) {

  }
};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<GateDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
