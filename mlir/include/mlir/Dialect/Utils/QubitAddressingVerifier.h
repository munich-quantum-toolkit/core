/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/Support/Casting.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Visitors.h>

namespace mlir::detail {

/**
 * @brief Verify that two qubit addressing modes are not mixed within a scope.
 *
 * @details
 * This helper is intended for operation-local verifiers (e.g., `qc.alloc` and
 * `qc.static`). It finds the nearest enclosing operation with the
 * `OpTrait::IsIsolatedFromAbove` trait (typically a `func.func`) and checks
 * whether an operation of type @p OppositeOp exists anywhere inside that scope.
 * The walk interrupts early on the first match.
 *
 * @tparam OppositeOp The operation type that must not appear in the same scope.
 * @tparam ThisOp The operation type emitting the diagnostic.
 *
 * @param op The current operation instance.
 * @param thisMnemonic A human-readable description of @p op's mode.
 * @param oppositeMnemonic A human-readable description of the opposite mode.
 * @param thisSpelling A string representation of @p op (e.g., "`qc.alloc`").
 * @param oppositeSpelling A string representation of @p OppositeOp
 * (e.g., "`qc.static`").
 * @return `success()` if no conflict was found, otherwise emits an error on
 * @p op and returns `failure()`.
 */
template <typename OppositeOp, typename ThisOp>
inline ::mlir::LogicalResult verifyNoMixedQubitAddressingModes(
    ThisOp op, const char* thisMnemonic, const char* oppositeMnemonic,
    const char* thisSpelling, const char* oppositeSpelling) {
  ::mlir::Operation* scope =
      op->template getParentWithTrait<::mlir::OpTrait::IsIsolatedFromAbove>();
  if (scope == nullptr) {
    return op.emitOpError(
        "must be nested within an operation with IsIsolatedFromAbove");
  }

  bool foundOpposite = false;
  (void)scope->walk([&](::mlir::Operation* nestedOp) -> ::mlir::WalkResult {
    if (nestedOp != scope &&
        nestedOp->hasTrait<::mlir::OpTrait::IsIsolatedFromAbove>()) {
      return ::mlir::WalkResult::skip();
    }
    if (::mlir::isa<OppositeOp>(nestedOp)) {
      foundOpposite = true;
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  if (foundOpposite) {
    return op.emitOpError()
           << "cannot mix " << thisMnemonic << " qubits (" << thisSpelling
           << ") with " << oppositeMnemonic << " qubits (" << oppositeSpelling
           << ") within the same isolated region";
  }
  return ::mlir::success();
}

} // namespace mlir::detail
