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

// Suppress warnings about ambiguous reversed operators in MLIR
// (see https://github.com/llvm/llvm-project/issues/45853)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#endif
#include <mlir/Interfaces/InferTypeOpInterface.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMOps.h.inc>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <string>
#include <variant>

#define DIALECT_NAME_QUARTZ "quartz"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::quartz {

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
    static size_t getNumPosControls() { return 0; }
    static size_t getNumNegControls() { return 0; }

    Value getQubit(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have qubits");
      }
      return this->getOperation()->getOperand(i);
    }
    Value getTarget(size_t i) {
      if constexpr (T == 0) {
        llvm::reportFatalUsageError("Operation does not have targets");
      }
      return this->getOperation()->getOperand(i);
    }

    static Value getPosControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }

    static Value getNegControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }

    static size_t getNumParams() { return P; }

    Value getParameter(const size_t i) {
      if (i >= P) {
        llvm::reportFatalUsageError("Parameter index out of bounds");
      }
      return this->getOperation()->getOperand(T + i);
    }

    [[nodiscard]] static FloatAttr getStaticParameter(const Value param) {
      auto constantOp = param.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        return nullptr;
      }
      return dyn_cast<FloatAttr>(constantOp.getValue());
    }

    bool hasStaticUnitary() {
      if constexpr (P == 0) {
        return true;
      }
      const auto& op = this->getOperation();
      for (size_t i = 0; i < P; ++i) {
        if (!getStaticParameter(op->getOperand(T + i))) {
          return false;
        }
      }
      return true;
    }
  };
};

} // namespace mlir::quartz

#include "mlir/Dialect/Quartz/IR/QuartzInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOps.h.inc" // IWYU pragma: export
