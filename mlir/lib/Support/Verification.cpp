/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Support/Verification.h"

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <utility>

namespace mlir::mqt {

LogicalResult verifyProgramParameters(Operation* root) {
  if (failed(verify(root))) {
    return failure();
  }
  DenseMap<Value, std::optional<Attribute>> cache;
  SmallVector<std::pair<Value, bool>> worklist;
  const auto result = root->walk([&](Operation* operation) -> WalkResult {
    SmallVector<Value> parameters;
    if (auto gate = dyn_cast<qc::UnitaryOpInterface>(operation)) {
      parameters = gate.getParameters();
    } else if (auto gate = dyn_cast<qco::UnitaryOpInterface>(operation)) {
      parameters = gate.getParameters();
    }
    for (auto [index, parameter] : llvm::enumerate(parameters)) {
      if (parameter.getDefiningOp() == nullptr) {
        continue;
      }
      worklist.emplace_back(parameter, false);
      while (!worklist.empty()) {
        auto [value, operandsVisited] = worklist.pop_back_val();
        if (!operandsVisited) {
          if (!cache.try_emplace(value, std::nullopt).second) {
            continue;
          }
          worklist.emplace_back(value, true);
          auto* definingOp = value.getDefiningOp();
          if (definingOp != nullptr && definingOp->getNumRegions() == 0 &&
              isPure(definingOp)) {
            for (auto operand : definingOp->getOperands()) {
              worklist.emplace_back(operand, false);
            }
          }
          continue;
        }
        /// Fold after visiting operands so deep expressions do not
        /// recurse.
        cache.erase(value);
        if (auto constant = valueToConstantAttr(value, cache)) {
          if (auto floating = dyn_cast<FloatAttr>(*constant);
              floating && !floating.getValue().isFinite()) {
            operation->emitOpError()
                << "constant parameter expression at index " << index
                << " must be finite";
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

} // namespace mlir::mqt
