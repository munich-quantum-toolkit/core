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

#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LLVM.h>

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
    OperandRange getInputTargets() { return getInputQubits(); }
    Value getOutputTarget(const size_t i) { return getOutputQubit(i); }
    ResultRange getOutputTargets() { return getOutputQubits(); }

    static Value getInputControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }
    static OperandRange getInputControls() { return {nullptr, 0}; }
    static Value getOutputControl([[maybe_unused]] size_t i) {
      llvm::reportFatalUsageError("Operation does not have controls");
    }
    static ResultRange getOutputControls() { return {nullptr, 0}; }

    static size_t getNumParams() { return P; }

    Value getParameter(const size_t i) {
      if (i >= P) {
        llvm::reportFatalUsageError("Parameter index out of bounds");
      }
      return this->getOperation()->getOperand(T + i);
    }
    OperandRange getParameters() {
      return this->getOperation()->getOperands().slice(T, P);
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

    [[nodiscard]] bool hasCompileTimeKnownUnitaryMatrix() {
      if constexpr (P == 0) {
        return true;
      } else {
        return llvm::all_of(this->getParameters(), [](Value param) {
          return utils::valueToDouble(param).has_value();
        });
      }
    }
  };
};

/**
 * @brief Find the entry point function with entry_point attribute
 *
 * @details
 * Searches for the function marked with the "entry_point" attribute in
 * the passthrough attributes. If multiple functions are marked, returns the
 * first one encountered.
 *
 * @param op The module operation to search in.
 * @returns the entry point function, or nullptr if not found.
 */
inline func::FuncOp getEntryPoint(ModuleOp op) {
  static constexpr StringRef PASSTHROUGH_LABEL = "passthrough";
  static constexpr StringRef ENTRY_POINT_LABEL = "entry_point";

  const auto isEntry = [](Attribute attr) {
    const auto strAttr = dyn_cast<StringAttr>(attr);
    return strAttr && strAttr.getValue() == ENTRY_POINT_LABEL;
  };

  for (auto func : op.getOps<func::FuncOp>()) {
    if (const auto passthrough =
            func->getAttrOfType<ArrayAttr>(PASSTHROUGH_LABEL);
        passthrough && llvm::any_of(passthrough, isEntry)) {
      return func;
    }
  }

  return nullptr;
}

} // namespace mlir::qco
