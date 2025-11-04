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

#include "mlir/Dialect/Utils/ParameterDescriptor.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <string>
#include <variant>

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

#define DIALECT_NAME_FLUX "flux"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Flux/IR/FluxOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

namespace mlir::flux {

template <size_t n> class TargetArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    size_t getNumQubits() { return n; }
    size_t getNumTargets() { return n; }
    size_t getNumControls() { return 0; }
    size_t getNumPosControls() { return 0; }
    size_t getNumNegControls() { return 0; }

    Value getInputQubit(size_t i) {
      return this->getOperation()->getOperand(i);
    }

    Value getOutputQubit(size_t i) {
      return this->getOperation()->getResult(i);
    }

    Value getInputTarget(size_t i) { return getInputQubit(i); }

    Value getOutputTarget(size_t i) { return getOutputQubit(i); }

    Value getInputPosControl(size_t i) {
      llvm::report_fatal_error("Operation does not have controls");
    }

    Value getOutputPosControl(size_t i) {
      llvm::report_fatal_error("Operation does not have controls");
    }

    Value getInputNegControl(size_t i) {
      llvm::report_fatal_error("Operation does not have controls");
    }

    Value getOutputNegControl(size_t i) {
      llvm::report_fatal_error("Operation does not have controls");
    }

    Value getInputForOutput(Value output) {
      switch (n) {
      case 1:
        if (output == this->getOperation()->getResult(0)) {
          return this->getOperation()->getOperand(0);
        }
        llvm::report_fatal_error(
            "Given qubit is not an output of the operation");
      case 2:
        if (output == this->getOperation()->getResult(0)) {
          return this->getOperation()->getOperand(0);
        }
        if (output == this->getOperation()->getResult(1)) {
          return this->getOperation()->getOperand(1);
        }
        llvm::report_fatal_error(
            "Given qubit is not an output of the operation");
      default:
        llvm::report_fatal_error("Unsupported number of qubits");
      }
    }

    Value getOutputForInput(Value input) {
      switch (n) {
      case 1:
        if (input == this->getOperation()->getOperand(0)) {
          return this->getOperation()->getResult(0);
        }
        llvm::report_fatal_error(
            "Given qubit is not an input of the operation");
      case 2:
        if (input == this->getOperation()->getOperand(0)) {
          return this->getOperation()->getResult(0);
        }
        if (input == this->getOperation()->getOperand(1)) {
          return this->getOperation()->getResult(1);
        }
        llvm::report_fatal_error(
            "Given qubit is not an input of the operation");
      default:
        llvm::report_fatal_error("Unsupported number of qubits");
      }
    }
  };
};

LogicalResult foldParameterArityTrait(Operation* op);

template <size_t n> class ParameterArityTrait {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
  public:
    size_t getNumParams() { return n; }

    static LogicalResult foldTrait(Operation* op, ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult>& results) {
      return foldParameterArityTrait(op);
    }
  };
};

} // namespace mlir::flux

#include "mlir/Dialect/Flux/IR/FluxInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.h.inc" // IWYU pragma: export
