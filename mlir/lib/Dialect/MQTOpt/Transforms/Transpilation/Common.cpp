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

/**
 * @brief Check if a unitary acts on two qubits.
 * @param u A unitary.
 * @returns True iff the qubit gate acts on two qubits.
 */
bool isTwoQubitGate(UnitaryInterface u) {
  return u.getAllInQubits().size() == 2;
}

/**
 * @brief Return input qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second in-qubits.
 */
[[nodiscard]] std::pair<mlir::Value, mlir::Value> getIns(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const std::vector<mlir::Value> inQubits = op.getAllInQubits();
  return {inQubits[0], inQubits[1]};
}

/**
 * @brief Return output qubit pair for a two-qubit unitary.
 * @param u A two-qubit unitary.
 * @return Pair of SSA values consisting of the first and second out-qubits.
 */
[[nodiscard]] std::pair<mlir::Value, mlir::Value> getOuts(UnitaryInterface op) {
  assert(isTwoQubitGate(op));
  const std::vector<mlir::Value> outQubits = op.getAllOutQubits();
  return {outQubits[0], outQubits[1]};
}
} // namespace mqt::ir::opt
