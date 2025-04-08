/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <algorithm>
#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mqt::ir::opt {
/// Analysis pattern that filters out all quantum operations from a given
/// program and creates a quantum computation from them.
struct ToQuantumComputationPattern final : mlir::OpRewritePattern<AllocOp> {
  qc::QuantumComputation& circuit; // NOLINT(*-avoid-const-or-ref-data-members)

  explicit ToQuantumComputationPattern(mlir::MLIRContext* context,
                                       qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  // clang-tidy false positive
  // NOLINTNEXTLINE(*-convert-member-functions-to-static)
  [[nodiscard]] mlir::LogicalResult match(const AllocOp op) const override {
    return (op->hasAttr("to_replace") || op->hasAttr("mqt_core"))
               ? mlir::failure()
               : mlir::success();
  }

  /**
   * @brief Finds the index of a qubit in the list of previously defined qubit
   * variables.
   *
   * In particular, this function checks if two value definitions are the same,
   * and, in case of array-style variables, also checks if the result index is
   * the same.
   *
   * @param input The qubit to find.
   * @param currentQubitVariables The list of previously defined qubit
   * variables.
   *
   * @return The index of the qubit in the list of previously defined qubit
   * variables.
   */
  static size_t
  findQubitIndex(mlir::Value input,
                 std::vector<mlir::Value>& currentQubitVariables) {
    size_t arrayIndex = 0;
    if (const auto opResult = llvm::dyn_cast<mlir::OpResult>(input)) {
      arrayIndex = opResult.getResultNumber();
    } else {
      throw std::runtime_error(
          "Operand is not an operation result. This should never happen!");
    }
    for (size_t i = 0; i < currentQubitVariables.size(); i++) {
      size_t qubitArrayIndex = 0;
      if (auto opResult =
              llvm::dyn_cast<mlir::OpResult>(currentQubitVariables[i])) {
        qubitArrayIndex = opResult.getResultNumber();
      } else {
        throw std::runtime_error(
            "Qubit is not an operation result. This should never happen!");
      }

      if (currentQubitVariables[i] == input && arrayIndex == qubitArrayIndex) {
        return i;
      }
    }

    throw std::runtime_error(
        "Qubit was not found in list of previously defined qubits");
  }

  /**
   * @brief Converts a measurement to an operation on the
   * `qc::QuantumComputation` and updates the `currentQubitVariables`.
   *
   * @param op The operation to convert.
   * @param currentQubitVariables The list of previously defined qubit
   * variables.
   */
  void handleMeasureOp(MeasureOp op,
                       std::vector<mlir::Value>& currentQubitVariables) const {
    const auto ins = op.getInQubits();
    const auto outs = op.getOutQubits();

    std::vector<size_t> insIndices(ins.size());
    std::transform(ins.begin(), ins.end(), insIndices.begin(),
                   [&currentQubitVariables](const mlir::Value val) {
                     return findQubitIndex(val, currentQubitVariables);
                   });

    for (size_t i = 0; i < insIndices.size(); i++) {
      currentQubitVariables[insIndices[i]] = outs[i];
      circuit.measure(insIndices[i], insIndices[i]);
    }
  }

  /**
   * @brief Converts a unitary operation to an operation on the
   * `qc::QuantumComputation` and updates the `currentQubitVariables`.
   *
   * @param op The operation to convert.
   * @param currentQubitVariables The list of previously defined qubit
   * variables.
   */
  void handleUnitaryOp(UnitaryInterface op,
                       std::vector<mlir::Value>& currentQubitVariables) const {

    // Add the operation to the QuantumComputation.
    qc::OpType opType;
    if (llvm::isa<HOp>(op)) {
      opType = qc::OpType::H;
    } else if (llvm::isa<XOp>(op)) {
      opType = qc::OpType::X;
    } else {
      throw std::runtime_error("Unsupported operation type!");
    }

    const auto in = op.getInQubits()[0];
    const auto ctrlIns = op.getCtrlQubits();
    const auto outs = op.getOutQubits();

    // Get the qubit index of every control qubit.
    std::vector<size_t> ctrlInsIndices(ctrlIns.size());
    std::transform(ctrlIns.begin(), ctrlIns.end(), ctrlInsIndices.begin(),
                   [&currentQubitVariables](const mlir::Value val) {
                     return findQubitIndex(val, currentQubitVariables);
                   });

    // Get the qubit index of the target qubit.
    const size_t targetIndex = findQubitIndex(in, currentQubitVariables);

    // Update `currentQubitVariables` with the new qubit values.
    for (size_t i = 0; i < ctrlInsIndices.size(); i++) {
      currentQubitVariables[ctrlInsIndices[i]] = outs[i + 1];
    }
    currentQubitVariables[targetIndex] = outs[0];

    // Add the operation to the QuantumComputation.
    auto operation = qc::StandardOperation(
        qc::Controls{ctrlInsIndices.cbegin(), ctrlInsIndices.cend()},
        targetIndex, opType);
    circuit.push_back(operation);
  }

  /**
   * @brief Recursively deletes an operation and all its defining operations if
   * they have no users.
   *
   * This procedure cleans up the AST so that only the base `alloc` operation
   * remains. Operations that still have users are ignored so that their users
   * can be handled first in a later step.
   *
   * @param op The operation to delete.
   * @param rewriter The pattern rewriter to use for deleting the operation.
   */
  static void deleteRecursively(mlir::Operation* op,
                                mlir::PatternRewriter& rewriter) {
    if (llvm::isa<AllocOp>(op)) {
      return; // Do not delete extract operations.
    }
    if (!op->getUsers().empty()) {
      return; // Do not delete operations with users.
    }

    rewriter.eraseOp(op);
    for (auto operand : op->getOperands()) {
      if (auto* defOp = operand.getDefiningOp()) {
        deleteRecursively(defOp, rewriter);
      }
    }
  }

  /**
   * @brief Updates the inputs of non MQTOpt-operations that use
   * MQTOpt-operations as inputs.
   *
   * Currently, such an operation should only be the return operation, but this
   * function is compatible with any operation that uses MQTOpt-operations as
   * inputs. Only Quregs and classical values may be used as inputs to non
   * MQTOpt-operations, Qubits are not supported!
   *
   * @param op The operation to update.
   * @param rewriter The pattern rewriter to use.
   * @param qureg The new Qureg to replace old Qureg uses with.
   * @param measureCount The number of measurements in the quantum circuit.
   */
  static void updateMQTOptInputs(mlir::Operation* op,
                                 mlir::PatternRewriter& rewriter,
                                 const mlir::Value qureg,
                                 const size_t measureCount) {
    size_t i = 0;
    auto* const cloned = rewriter.clone(*op);
    rewriter.setInsertionPoint(cloned);
    for (auto operand : op->getOperands()) {
      i++;
      const auto type = operand.getType();
      if (llvm::isa<QubitType>(type)) {
        throw std::runtime_error(
            "Interleaving of qubits with non MQTOpt-operations not supported "
            "by round-trip pass!");
      }
      if (llvm::isa<QubitRegisterType>(type)) {
        // Operations that used the old `qureg` will now use the new one
        // instead.
        cloned->setOperand(i - 1, qureg);
      }
      if (llvm::isa<mlir::IntegerType>(type)) {
        // Operations that used `i1` values (i.e. classical measurement results)
        // will now use a constant value of `false`.
        auto newInput = rewriter.create<mlir::arith::ConstantOp>(
            op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        cloned->setOperand(i - 1, newInput.getResult());
      }
    }

    // The return operation MUST use all measurement results as inputs.
    if (i != measureCount + 1) {
      throw std::runtime_error(
          "Measure count does not match number of return operands!");
    }
    rewriter.replaceOp(op, cloned);
  }

  void rewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::outs() << "\n-----------------GENERAL----------------\n";

    const auto& sizeAttr = op.getSizeAttr();
    if (sizeAttr.has_value()) {
      llvm::outs() << "Allocating " << *sizeAttr << " qubits\n";
    } else {
      throw std::runtime_error(
          "Qubit allocation only supported with attr size!");
    }

    // First, we create a new `AllocOp` that will replace the old one. It
    // includes the flag `to_replace`.
    auto newAlloc = rewriter.create<AllocOp>(
        op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    newAlloc->setAttr("to_replace", rewriter.getUnitAttr());

    size_t measureCount = 0;
    const std::size_t numQubits = *sizeAttr;
    // `currentQubitVariables` holds the current `Value` representation of each
    // qubit from the original register.
    std::vector<mlir::Value> currentQubitVariables(numQubits);
    std::vector<mlir::Operation*> toVisit{op};

    std::string regName;
    llvm::raw_string_ostream os(regName);
    op.getResult().print(os);

    circuit.addQubitRegister(numQubits, regName);
    circuit.addClassicalRegister(numQubits);

    std::set<mlir::Operation*> visited{};

    // Visit all operations in the AST using Breadth-First Search.
    while (!toVisit.empty()) {
      mlir::Operation* current = *toVisit.begin();
      toVisit.erase(toVisit.begin());
      if (visited.find(current) != visited.end()) {
        continue;
      }
      visited.insert(current);

      if (llvm::isa<XOp>(current)) {
        auto xOp = llvm::dyn_cast<XOp>(current);
        handleUnitaryOp(xOp, currentQubitVariables);
      } else if (llvm::isa<HOp>(current)) {
        HOp hOp = mlir::dyn_cast<HOp>(current);
        handleUnitaryOp(hOp, currentQubitVariables);
      } else if (llvm::isa<ExtractOp>(current)) {
        auto extractOp = llvm::dyn_cast<ExtractOp>(current);
        mlir::Value indexValue =
            extractOp.getIndex(); // assuming getIndex() returns the Value

        // Check if the index is defined by an arith.constant
        if (auto constOp =
                indexValue.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto intAttr =
                  llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            size_t index = intAttr.getInt();
            currentQubitVariables[index] = extractOp.getOutQubit();
          } else {
            extractOp.emitError("Expected integer constant for qubit index.");
            throw std::runtime_error(
                "Qubit extraction only supported with attr index!");
          }
        } else {
          extractOp.emitError("Qubit extraction index must be a constant.");
          throw std::runtime_error(
              "Qubit extraction only supported with attr index!");
        }
      } else if (llvm::isa<InsertOp>(current)) {
        auto insertOp = llvm::dyn_cast<InsertOp>(current);
        llvm::outs() << *current << "\n";
      } else if (llvm::isa<AllocOp>(current)) {
        // Do nothing for now, may change later.
      } else if (llvm::isa<MeasureOp>(current)) {
        // We count the number of measurements and add a measurement operation
        // to the QuantumComputation.
        measureCount++;
        auto measureOp = llvm::dyn_cast<MeasureOp>(current);
        handleMeasureOp(measureOp, currentQubitVariables);
      } else {
        llvm::outs() << "Skipping unsupported operation: " << *current << "\n";
        continue;
      }

      for (mlir::Operation* user : current->getUsers()) {
        if (visited.find(user) != visited.end() ||
            toVisit.end() != std::find(toVisit.begin(), toVisit.end(), user)) {
          continue;
        }
        toVisit.push_back(user);
      }
    }

    llvm::outs() << "----------------------------------------\n\n";

    llvm::outs() << "-------------------QC-------------------\n";
    std::stringstream ss{};
    circuit.print(ss);
    const auto circuitString = ss.str();
    llvm::outs() << circuitString << "\n";
    llvm::outs() << "----------------------------------------\n\n";

    // Update the inputs of all non-mqtopt operations that use mqtopt operations
    // as inputs, as these will be deleted later.
    for (auto* operation : llvm::make_early_inc_range(visited)) {
      if (operation->getDialect()->getNamespace() != DIALECT_NAME_MQTOPT) {
        visited.erase(operation); // safe deletion as op will be erased next
        updateMQTOptInputs(operation, rewriter, newAlloc.getQureg(),
                           measureCount);
      }
    }

    // Delete all operations that are part of the mqtopt dialect (except for
    // `AllocOp`).
    visited.erase(op); // erase alloc op
    while (!visited.empty()) {
      for (auto* operation : llvm::make_early_inc_range(visited)) {
        operation->emitRemark() << "HERE";
        if (operation->getDialect()->getNamespace() == DIALECT_NAME_MQTOPT) {
          if (operation->getUsers().empty()) {
            visited.erase(operation);
            rewriter.eraseOp(
                operation); // deletes all definitions of defining OPs
          }
        }
      }
    }

    rewriter.replaceOp(op, newAlloc);

    llvm::outs() << "--------------END-----------------------\n\n";
  }
};

/**
 * @brief Populates the given pattern set with the
 * `ToQuantumComputationPattern`.
 *
 * @param patterns The pattern set to populate.
 * @param circuit The quantum computation to create MLIR instructions from.
 */
void populateToQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                          qc::QuantumComputation& circuit) {
  patterns.add<ToQuantumComputationPattern>(patterns.getContext(), circuit);
}

} // namespace mqt::ir::opt
