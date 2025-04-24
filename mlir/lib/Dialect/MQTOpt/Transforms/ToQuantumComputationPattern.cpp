/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
   * variables. Throws an exception if the qubit is not found.
   */
  static size_t
  findQubitIndex(const mlir::Value& input,
                 std::vector<mlir::Value>& currentQubitVariables) {
    size_t arrayIndex = 0;

    // Check if the input is an operation result
    if (const auto opResult = llvm::dyn_cast<mlir::OpResult>(input)) {
      arrayIndex = opResult.getResultNumber();
    } else {
      throw std::runtime_error(
          "Operand is not an operation result. This should never happen!");
    }

    // Find the qubit in the list of previously defined qubit variables
    for (size_t i = 0; i < currentQubitVariables.size(); i++) {
      size_t qubitArrayIndex = 0;
      if (currentQubitVariables[i] != nullptr) {
        auto opResult =
            llvm::dyn_cast<mlir::OpResult>(currentQubitVariables[i]);
        // E.g., index `1` if it comes from an extract op
        qubitArrayIndex = opResult.getResultNumber();
      } else {
        // Qubit is not (yet) in currentQubitVariables
        throw std::runtime_error(
            "Qubit was not found in list of previously defined qubits");
      }

      if (currentQubitVariables[i] == input && arrayIndex == qubitArrayIndex) {
        return i;
      }
    }

    // Qubit is not (yet) in currentQubitVariables
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
   *
   * @return True if the operation was successfully handled, false if the
   * operation defining the qubit operand was not yet added to the
   * currentQubitVariables.
   */
  bool handleMeasureOp(MeasureOp& op,
                       std::vector<mlir::Value>& currentQubitVariables) const {
    const auto ins = op.getInQubits();
    const auto outs = op.getOutQubits();

    std::vector<size_t> insIndices(ins.size());

    try {
      std::transform(ins.begin(), ins.end(), insIndices.begin(),
                     [&currentQubitVariables](const mlir::Value val) {
                       return findQubitIndex(val, currentQubitVariables);
                     });
    } catch (const std::runtime_error& e) {
      if (strcmp(e.what(),
                 "Qubit was not found in list of previously defined qubits") ==
          0) {
        // Try again later when all qubits are available
        return false;
      }
      throw; // Rethrow the exception if it's not the expected one.
    }

    for (size_t i = 0; i < insIndices.size(); i++) {
      currentQubitVariables[insIndices[i]] = outs[i];
      circuit.measure(insIndices[i], insIndices[i]);
    }

    return true;
  }

  /**
   * @brief Converts a unitary operation to an operation on the
   * `qc::QuantumComputation` and updates the `currentQubitVariables`.
   *
   * @param op The operation to convert.
   * @param currentQubitVariables The list of previously defined qubit
   * variables.
   *
   * @return True if the operation was successfully handled, false if the
   * operation defining the qubit operand was not yet added to the
   * currentQubitVariables.
   */
  bool handleUnitaryOp(UnitaryInterface op,
                       std::vector<mlir::Value>& currentQubitVariables) const {

    // Add the operation to the QuantumComputation.
    qc::OpType opType = qc::OpType::H; // init placeholder-"H" overwritten next
    if (llvm::isa<HOp>(op)) {
      opType = qc::OpType::H;
    } else if (llvm::isa<XOp>(op)) {
      opType = qc::OpType::X;
    } else { // TODO: support for more operations
      throw std::runtime_error("Unsupported operation type!");
    }

    const auto in = op.getInQubits()[0];
    const auto ctrlIns = op.getAllCtrlInQubits();
    const auto outs = op.getAllOutQubits();

    // Get the qubit index of every control qubit.
    std::vector<size_t> ctrlInsIndices(ctrlIns.size());
    size_t targetIndex = 0; // Placeholder
    try {
      std::transform(ctrlIns.begin(), ctrlIns.end(), ctrlInsIndices.begin(),
                     [&currentQubitVariables](const mlir::Value val) {
                       return findQubitIndex(val, currentQubitVariables);
                     });
      // Get the qubit index of the target qubit (if already collected).
      targetIndex = findQubitIndex(in, currentQubitVariables);
    } catch (const std::runtime_error& e) {
      if (strcmp(e.what(),
                 "Qubit was not found in list of previously defined qubits") ==
          0) {
        // Try again later when all qubits are available
        return false;
      }
      throw; // Rethrow the exception if it's not the expected one.
    }
    // Update `currentQubitVariables` with the new qubit values.
    for (size_t i = 0; i < ctrlInsIndices.size(); i++) {
      currentQubitVariables[ctrlInsIndices[i]] = outs[i + 1];
    }
    currentQubitVariables[targetIndex] = outs[0];

    // Add the operation to the QuantumComputation.
    circuit.emplace_back<qc::StandardOperation>(
        qc::Controls{ctrlInsIndices.cbegin(), ctrlInsIndices.cend()},
        targetIndex, opType);

    return true;
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
  static void deleteRecursively(mlir::Operation& op,
                                mlir::PatternRewriter& rewriter) {
    if (llvm::isa<AllocOp>(op)) {
      return; // Do not delete extract operations.
    }
    if (!op.getUsers().empty()) {
      return; // Do not delete operations with users.
    }

    rewriter.eraseOp(&op);
    for (auto operand : op.getOperands()) {
      if (auto* defOp = operand.getDefiningOp()) {
        deleteRecursively(*defOp, rewriter);
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
   */
  static void updateMQTOptInputs(mlir::Operation& op,
                                 mlir::PatternRewriter& rewriter,
                                 const mlir::Value& qureg) {
    size_t i = 0;
    auto* const cloned = rewriter.clone(op);
    rewriter.setInsertionPoint(cloned);
    for (auto operand : op.getOperands()) {
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
            op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        cloned->setOperand(i - 1, newInput.getResult());
      }
    }

    rewriter.replaceOp(&op, cloned);
  }

  static std::optional<size_t> getExtractIndex(ExtractOp extractOp) {
    // Case 1: Static attribute index
    if (const auto indexAttr = extractOp.getIndexAttr();
        indexAttr.has_value()) {
      return static_cast<size_t>(*indexAttr);
    }

    // Case 2: Dynamic index via operand
    if (const mlir::Value index = extractOp.getIndex()) {
      // Case 2a: Direct constant
      if (auto constOp = index.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          return static_cast<size_t>(intAttr.getInt());
        }
      }
      // Case 2b: Extract from tensor computation
      if (auto tensorExtract = index.getDefiningOp<mlir::tensor::ExtractOp>()) {
        return std::nullopt;
      }
      // Case 2c: unknown or unsupported dynamic index pattern
      return std::nullopt;
    }

    // No index present at all
    return std::nullopt;
  }

  void rewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    const auto& sizeAttr = op.getSizeAttr();

    // First, we create a new `AllocOp` that will replace the old one. It
    // includes the flag `to_replace`.
    auto newAlloc = rewriter.create<AllocOp>(
        op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    newAlloc->setAttr("to_replace", rewriter.getUnitAttr());

    if (!sizeAttr) {
      throw std::runtime_error(
          "Failed to resolve size of qubit register in alloc operation");
    }
    const std::size_t numQubits = *sizeAttr;
    // `currentQubitVariables` holds the current `Value` representation of each
    // qubit from the original register.
    std::vector<mlir::Value> currentQubitVariables(numQubits);

    std::string regName;
    llvm::raw_string_ostream os(regName);
    op.getResult().print(os);

    circuit.addQubitRegister(numQubits, regName);
    circuit.addClassicalRegister(numQubits);

    std::unordered_set<mlir::Operation*> visited;
    // `set` to prevent 'nondeterministic iteration of pointers'
    std::set<mlir::Operation*> mqtUsers;

    mlir::Operation* current = op;
    while (current != nullptr) {
      // no need to visit non-mqtopt operations
      if (visited.find(current) != visited.end() ||
          current->getDialect()->getNamespace() != DIALECT_NAME_MQTOPT) {
        current = current->getNextNode();
        continue;
      }
      visited.insert(current);

      // collect all non-mqtopt users of the current operation
      for (mlir::Operation* user : current->getUsers()) {
        if (user->getDialect()->getNamespace() != DIALECT_NAME_MQTOPT) {
          mqtUsers.insert(user);
        }
      }

      if (llvm::isa<XOp>(current) || llvm::isa<HOp>(current)) {
        auto unitaryOp = llvm::dyn_cast<UnitaryInterface>(current);
        handleUnitaryOp(unitaryOp, currentQubitVariables);
      } else if (auto extractOp = llvm::dyn_cast<ExtractOp>(current)) {
        auto maybeIndex = getExtractIndex(extractOp);
        if (!maybeIndex.has_value()) {
          throw std::runtime_error(
              "Failed to resolve index in extractQubit operation");
        }
        const size_t index = *maybeIndex;
        currentQubitVariables[index] = extractOp.getOutQubit();
      } else if (llvm::isa<InsertOp>(current) || llvm::isa<AllocOp>(current)) {
        // Do nothing for now, may change later.
      } else if (llvm::isa<MeasureOp>(current)) {
        // We count the number of measurements and add a measurement operation
        // to the QuantumComputation.
        auto measureOp = llvm::dyn_cast<MeasureOp>(current);
        handleMeasureOp(measureOp, currentQubitVariables);
      } else {
        llvm::outs() << "Skipping unsupported operation: " << *current << "\n";
        continue;
      }

      current = current->getNextNode();
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
    // NOLINTNEXTLINE(readability-qualified-auto)
    for (auto* operation : mqtUsers) {
      updateMQTOptInputs(*operation, rewriter, newAlloc.getQureg());
    }

    // Remove all dead mqtopt operations except AllocOp.
    visited.erase(op); // Skip the original AllocOp
    bool progress = true;
    while (progress) {
      progress = false;

      for (auto it = visited.begin(); it != visited.end(); /* no increment */) {
        mlir::Operation* operation = *it;

        // Only erase ops with no remaining users
        if (operation->getUsers().empty()) {
          it = visited.erase(it);      // remove from visited
          rewriter.eraseOp(operation); // erase from IR
          progress = true;
        } else {
          ++it;
        }
      }
    }

    rewriter.replaceOp(op, newAlloc);
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
