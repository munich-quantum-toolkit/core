/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <cstddef>
#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mqt::ir::opt {

namespace {
bool isQubitType(const mlir::MemRefType type) {
  return llvm::isa<QubitType>(type.getElementType());
}

bool isQubitType(mlir::memref::DeallocOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<mlir::MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

bool isQubitType(mlir::memref::LoadOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<mlir::MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

bool isQubitType(mlir::memref::StoreOp op) {
  const auto& memRef = op.getMemref();
  const auto& memRefType = llvm::cast<mlir::MemRefType>(memRef.getType());
  return isQubitType(memRefType);
}

bool isSupportedMemRefOp(mlir::Operation* op) {
  if (auto deallocOp = llvm::dyn_cast<mlir::memref::DeallocOp>(op)) {
    return isQubitType(deallocOp);
  }
  if (auto loadOp = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
    return isQubitType(loadOp);
  }
  if (auto storeOp = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
    return isQubitType(storeOp);
  }
  return false;
}
} // namespace

/// Analysis pattern that filters out all quantum operations from a given
/// program and creates a quantum computation from them.
struct ToQuantumComputationPattern final
    : mlir::OpRewritePattern<mlir::memref::AllocOp> {
  qc::QuantumComputation& circuit; // NOLINT(*-avoid-const-or-ref-data-members)

  explicit ToQuantumComputationPattern(mlir::MLIRContext* context,
                                       qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

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
   *
   * @throws `std::runtime_error` if the qubit is not found.
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
        // E.g., index `1` if it comes from a load op
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
    const auto in = op.getInQubit();
    const auto out = op.getOutQubit();

    size_t inIndex = 0;

    try {
      inIndex = findQubitIndex(in, currentQubitVariables);
    } catch (const std::runtime_error& e) {
      if (strcmp(e.what(),
                 "Qubit was not found in list of previously defined qubits") ==
          0) {
        // Try again later when all qubits are available
        return false;
      }
      throw; // Rethrow the exception if it's not the expected one.
    }

    currentQubitVariables[inIndex] = out;
    circuit.measure(inIndex, inIndex);
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
    qc::OpType opType = qc::OpType::None;

    try {
      const std::string type = op->getName().stripDialect().str();
      opType = qc::opTypeFromString(type);
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error("Unsupported operation type: " +
                               op->getName().getStringRef().str());
    }

    const auto in = op.getInQubits();
    const auto posCtrlIns = op.getPosCtrlInQubits();
    const auto negCtrlIns = op.getNegCtrlInQubits();
    const auto outs = op.getAllOutQubits();

    // Get the qubit index of every control qubit.
    std::vector<size_t> posCtrlInsIndices;
    std::vector<size_t> negCtrlInsIndices;
    std::vector<size_t> targetIndex(2); // adapted for two-target gates
    try {
      for (const auto& val : posCtrlIns) {
        posCtrlInsIndices.emplace_back(
            findQubitIndex(val, currentQubitVariables));
      }
      for (const auto& val : negCtrlIns) {
        negCtrlInsIndices.emplace_back(
            findQubitIndex(val, currentQubitVariables));
      }
      // Get the qubit index of the target qubit (if already collected).
      targetIndex[0] = findQubitIndex(in[0], currentQubitVariables);
      if (in.size() > 1) {
        targetIndex[1] = findQubitIndex(in[1], currentQubitVariables);
      }
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
    for (size_t i = 0; i < posCtrlInsIndices.size(); i++) {
      currentQubitVariables[posCtrlInsIndices[i]] = outs[i + 1];
    }
    for (size_t i = 0; i < negCtrlInsIndices.size(); i++) {
      currentQubitVariables[negCtrlInsIndices[i]] =
          outs[i + 1 + posCtrlInsIndices.size()];
    }
    currentQubitVariables[targetIndex[0]] = outs[0];
    if (op.getOutQubits().size() > 1) {
      currentQubitVariables[targetIndex[1]] = outs[1];
    }

    // Add the operation to the QuantumComputation.
    qc::Controls controls;
    for (const auto& index : posCtrlInsIndices) {
      controls.emplace(static_cast<qc::Qubit>(index), qc::Control::Type::Pos);
    }
    for (const auto& index : negCtrlInsIndices) {
      controls.emplace(static_cast<qc::Qubit>(index), qc::Control::Type::Neg);
    }

    std::vector<double> parameters;
    if (const auto staticParamsAttr =
            op->getAttrOfType<mlir::DenseF64ArrayAttr>("static_params")) {
      parameters.reserve(staticParamsAttr.size());
      for (const auto& param : staticParamsAttr.asArrayRef()) {
        parameters.emplace_back(param);
      }
    }

    if (op.getOutQubits().size() > 1) {
      circuit.emplace_back<qc::StandardOperation>(
          controls, targetIndex[0], targetIndex[1], opType, parameters);
    } else {
      circuit.emplace_back<qc::StandardOperation>(controls, targetIndex[0],
                                                  opType, parameters);
    }

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
    if (llvm::isa<mlir::memref::AllocOp>(op)) {
      return; // Do not delete alloc operations.
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
   * inputs. Only qregs and classical values may be used as inputs to non
   * MQTOpt-operations, Qubits are not supported!
   *
   * @param op The operation to update.
   * @param rewriter The pattern rewriter to use.
   * @param qreg The new qreg to replace old qreg uses with.
   */
  static void updateMQTOptInputs(mlir::Operation& op,
                                 mlir::PatternRewriter& rewriter,
                                 const mlir::Value& qreg) {
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
      if (llvm::isa<mlir::MemRefType>(type)) {
        // Operations that used the old `qreg` will now use the new one
        // instead.
        cloned->setOperand(i - 1, qreg);
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

  static std::optional<size_t> getLoadIndex(mlir::memref::LoadOp loadOp) {
    if (const auto index = loadOp.getIndices().front()) {
      if (auto constOp = index.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          return static_cast<size_t>(intAttr.getInt());
        }
      }
    }
    return std::nullopt;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp op,
                  mlir::PatternRewriter& rewriter) const override {
    // Skip if already marked
    if (op->hasAttr("to_replace") || op->hasAttr("mqt_core")) {
      return mlir::failure();
    }

    if (!op.getType().hasStaticShape()) {
      throw std::runtime_error(
          "Failed to resolve size of qubit register in alloc operation");
    }
    auto size = op.getType().getShape().front();

    // First, we create a new `AllocOp` that will replace the old one. It
    // includes the flag `to_replace`.
    // Create a new qubit register with the correct number of qubits.
    const auto& qubitType = QubitType::get(rewriter.getContext());
    const auto& memRefType = mlir::MemRefType::get({0}, qubitType);
    auto newAlloc = rewriter.create<mlir::memref::AllocOp>(
        op.getLoc(), memRefType, mlir::ValueRange{});
    newAlloc->setAttr("to_replace", rewriter.getUnitAttr());

    const std::size_t numQubits = size;
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
      if (visited.contains(current) ||
          (current->getDialect()->getNamespace() != DIALECT_NAME_MQTOPT &&
           !isSupportedMemRefOp(current))) {
        current = current->getNextNode();
        continue;
      }
      visited.insert(current);

      // collect all non-mqtopt users of the current operation
      for (mlir::Operation* user : current->getUsers()) {
        if (user->getDialect()->getNamespace() != DIALECT_NAME_MQTOPT &&
            !isSupportedMemRefOp(user)) {
          mqtUsers.insert(user);
        }
      }

      if (llvm::isa<UnitaryInterface>(current)) {
        auto unitaryOp = llvm::dyn_cast<UnitaryInterface>(current);
        handleUnitaryOp(unitaryOp, currentQubitVariables);
      } else if (auto loadOp = llvm::dyn_cast<mlir::memref::LoadOp>(current)) {
        auto maybeIndex = getLoadIndex(loadOp);
        if (!maybeIndex.has_value()) {
          throw std::runtime_error(
              "Failed to resolve index in extractQubit operation");
        }
        const size_t index = *maybeIndex;
        currentQubitVariables[index] = loadOp.getResult();
      } else if (llvm::isa<mlir::memref::StoreOp>(current) ||
                 llvm::isa<mlir::memref::DeallocOp>(current)) {
        // Do nothing for now, may (and probably should) change later.
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
    for (auto* operation : mqtUsers) {
      updateMQTOptInputs(*operation, rewriter, newAlloc.getMemref());
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
    return mlir::success();
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
