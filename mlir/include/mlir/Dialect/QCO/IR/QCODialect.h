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

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Visitors.h>

#define DIALECT_NAME_QCO "qco"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QCO/IR/QCOOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QCO/IR/QCOOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::qco {

/**
 * @brief Verify that two qubit addressing modes are not mixed within a scope.
 *
 * @details
 * This helper is intended for operation-local verifiers (e.g., `qco.alloc` and
 * `qco.static`). It finds the nearest enclosing operation with the
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
 * @param thisSpelling A string representation of @p op (e.g., "`qco.alloc`").
 * @param oppositeSpelling A string representation of @p OppositeOp
 * (e.g., "`qco.static`").
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
    scope = op->getParentOp();
  }
  if (scope == nullptr) {
    scope = op.getOperation();
  }

  bool foundOpposite = false;
  (void)scope->walk([&](OppositeOp) {
    foundOpposite = true;
    return ::mlir::WalkResult::interrupt();
  });
  if (foundOpposite) {
    return op.emitOpError()
           << "cannot mix " << thisMnemonic << " qubits (" << thisSpelling
           << ") with " << oppositeMnemonic << " qubits (" << oppositeSpelling
           << ") within the same isolated region";
  }
  return ::mlir::success();
}

/**
 * @brief Trait for operations with a fixed number of target qubits and
 * parameters
 * @details This trait indicates that an operation has a fixed number of target
 * qubits and parameters, specified by the template parameters T and P. This is
 * helpful for defining operations with known arities, allowing for static
 * verification and code generation optimizations.
 * @tparam T The target arity.
 * @tparam P The parameter arity.
 */
template <size_t T, size_t P> class TargetAndParameterArityTrait {
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
    OperandRange getInputQubits() {
      return this->getOperation()->getOperands().slice(0, T);
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
    ResultRange getOutputQubits() { return this->getOperation()->getResults(); }

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
  };
};

} // namespace mlir::qco
