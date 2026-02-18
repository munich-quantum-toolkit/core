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

#define DIALECT_NAME_QC "qc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QC/IR/QCOpsDialect.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/QC/IR/QCOpsTypes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::qc {

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

    Value getQubit(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have qubits");
      }
      if (i >= T) {
        llvm::reportFatalUsageError("Qubit index out of bounds");
      }
      return this->getOperation()->getOperand(i);
    }
    Value getTarget(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have targets");
      }
      if (i >= T) {
        llvm::reportFatalUsageError("Target index out of bounds");
      }
      return this->getOperation()->getOperand(i);
    }

    static Value getControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }

    static size_t getNumParams() { return P; }

    Value getParameter(const size_t i) {
      if (i >= P) {
        llvm::reportFatalUsageError("Parameter index out of bounds");
      }
      return this->getOperation()->getOperand(T + i);
    }
  };
};

} // namespace mlir::qc
