/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Common.h"

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"

#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <utility>

namespace mqt::ir::opt {
namespace {
/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ENTRY_POINT_ATTR{"entry_point"};

/**
 * @brief Attribute to forward function-level attributes to LLVM IR.
 */
constexpr llvm::StringLiteral PASSTHROUGH_ATTR{"passthrough"};
} // namespace

bool isEntryPoint(mlir::func::FuncOp op) {
  const auto passthroughAttr =
      op->getAttrOfType<mlir::ArrayAttr>(PASSTHROUGH_ATTR);
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const mlir::Attribute attr) {
    return isa<mlir::StringAttr>(attr) &&
           cast<mlir::StringAttr>(attr) == ENTRY_POINT_ATTR;
  });
}

bool isTwoQubitGate(UnitaryInterface u) {
  return (u.getInQubits().size() + u.getPosCtrlInQubits().size() +
          u.getNegCtrlInQubits().size()) == 2;
}

[[nodiscard]] std::pair<mlir::Value, mlir::Value> getIns(UnitaryInterface op) {
  assert(isTwoQubitGate(op));

  const auto target = op.getInQubits();
  const auto targetSize = target.size();

  if (targetSize == 2) {
    return {target[0], target[1]};
  }

  const auto posCtrl = op.getPosCtrlInQubits();
  return (posCtrl.size() == 1)
             ? std::pair{target[0], posCtrl[0]}
             : std::pair{target[0], op.getNegCtrlInQubits()[0]};
}

[[nodiscard]] std::pair<mlir::Value, mlir::Value> getOuts(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const auto target = op.getOutQubits();
  const auto targetSize = target.size();

  if (targetSize == 2) {
    return {target[0], target[1]};
  }

  const auto posCtrl = op.getPosCtrlOutQubits();
  return (posCtrl.size() == 1)
             ? std::pair{target[0], posCtrl[0]}
             : std::pair{target[0], op.getNegCtrlOutQubits()[0]};
}
} // namespace mqt::ir::opt
