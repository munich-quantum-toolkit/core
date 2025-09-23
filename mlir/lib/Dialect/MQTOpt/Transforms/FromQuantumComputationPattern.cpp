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
#include "ir/operations/OpType.hpp"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <stdexcept>
#include <vector>

namespace mqt::ir::opt {
/// Analysis pattern that creates MLIR instructions from a given
/// qc::QuantumComputation. These instructions replace an existing `AllocOp`
/// that has the `to_replace` attribute set.
struct FromQuantumComputationPattern final
    : mlir::OpRewritePattern<mlir::memref::AllocOp> {
  qc::QuantumComputation& circuit; // NOLINT(*-avoid-const-or-ref-data-members)

  explicit FromQuantumComputationPattern(mlir::MLIRContext* context,
                                         qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  /**
   * @brief Creates a LoadOp for loading a qubit.
   *
   * @param reg The register to load the qubit from.
   * @param index The index of the qubit to load.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created LoadOp.
   */
  static mlir::memref::LoadOp createLoadOp(mlir::Value reg, const size_t index,
                                           mlir::PatternRewriter& rewriter) {
    auto indexValue = rewriter.create<mlir::arith::ConstantIndexOp>(
        reg.getLoc(), static_cast<int64_t>(index));
    return rewriter.create<mlir::memref::LoadOp>(reg.getLoc(), reg,
                                                 mlir::ValueRange{indexValue});
  }

  /**
   * @brief Creates a StoreOp for storing a qubit.
   *
   * @param qubit The qubit to store into the register.
   * @param reg The register into which the qubit will be stored.
   * @param index The index at which to store the qubit.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created StoreOp.
   */
  static mlir::memref::StoreOp createStoreOp(mlir::Value qubit, mlir::Value reg,
                                             const size_t index,
                                             mlir::PatternRewriter& rewriter) {
    auto indexValue = rewriter.create<mlir::arith::ConstantIndexOp>(
        reg.getLoc(), static_cast<int64_t>(index));
    return rewriter.create<mlir::memref::StoreOp>(reg.getLoc(), qubit, reg,
                                                  mlir::ValueRange{indexValue});
  }

#define CREATE_OP_CASE(opType)                                                 \
  case qc::OpType::opType:                                                     \
    return rewriter.create<opType##Op>(                                        \
        loc, outQubitTypes, controlQubitsPositive.getType(),                   \
        controlQubitsNegative.getType(), denseParamsAttr, nullptr,             \
        mlir::ValueRange{}, inQubits, controlQubitsPositive,                   \
        controlQubitsNegative);

  /**
   * @brief Creates a unitary operation on a given qubit with any number of
   * positive or negative controls.
   *
   * @param loc The location of the operation.
   * @param type The type of the unitary operation.
   * @param inQubits The qubits to apply the unitary operation to.
   * @param controlQubitsPositive The positive control qubits.
   * @param controlQubitsNegative The negative control qubits.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created UnitaryOp.
   */
  static UnitaryInterface
  createUnitaryOp(const mlir::Location loc, const qc::OpType type,
                  const llvm::SmallVector<mlir::Value>& inQubits,
                  mlir::ValueRange controlQubitsPositive,
                  mlir::ValueRange controlQubitsNegative,
                  mlir::PatternRewriter& rewriter,
                  const std::vector<double>& parameters = {}) {
    // Create result types for all output qubits
    auto qubitType = QubitType::get(rewriter.getContext());
    const llvm::SmallVector<mlir::Type> outQubitTypes(inQubits.size(),
                                                      qubitType);

    // For all parametric gates, turn parameters vector into DenseF64ArrayAttr
    auto denseParamsAttr =
        parameters.empty()
            ? nullptr
            : mlir::DenseF64ArrayAttr::get(rewriter.getContext(), parameters);

    switch (type) {
      CREATE_OP_CASE(I)
      CREATE_OP_CASE(H)
      CREATE_OP_CASE(X)
      CREATE_OP_CASE(Y)
      CREATE_OP_CASE(Z)
      CREATE_OP_CASE(S)
      CREATE_OP_CASE(Sdg)
      CREATE_OP_CASE(T)
      CREATE_OP_CASE(Tdg)
      CREATE_OP_CASE(V)
      CREATE_OP_CASE(Vdg)
      CREATE_OP_CASE(U)
      CREATE_OP_CASE(U2)
      CREATE_OP_CASE(P)
      CREATE_OP_CASE(SX)
      CREATE_OP_CASE(SXdg)
      CREATE_OP_CASE(RX)
      CREATE_OP_CASE(RY)
      CREATE_OP_CASE(RZ)
      CREATE_OP_CASE(SWAP)
      CREATE_OP_CASE(iSWAP)
      CREATE_OP_CASE(iSWAPdg)
      CREATE_OP_CASE(Peres)
      CREATE_OP_CASE(Peresdg)
      CREATE_OP_CASE(DCX)
      CREATE_OP_CASE(ECR)
      CREATE_OP_CASE(RXX)
      CREATE_OP_CASE(RYY)
      CREATE_OP_CASE(RZZ)
      CREATE_OP_CASE(RZX)
      CREATE_OP_CASE(XXminusYY)
      CREATE_OP_CASE(XXplusYY)
    default:
      throw std::runtime_error("Unsupported operation type");
    }
  }

  /**
   * @brief Creates a MeasureOp on a given qubit.
   *
   * @param loc The location of the operation.
   * @param targetQubit The qubit to measure.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created MeasureOp.
   */
  static MeasureOp createMeasureOp(const mlir::Location loc,
                                   mlir::Value targetQubit,
                                   mlir::PatternRewriter& rewriter) {
    return rewriter.create<MeasureOp>(
        loc,
        mlir::TypeRange{QubitType::get(rewriter.getContext()),
                        rewriter.getI1Type()},
        targetQubit);
  }

  /**
   * @brief Updates the inputs of the function's `return` operation.
   *
   * The previous operation used constant `false` values for the returns of type
   * `i1`. After this update, the measurement results are used instead.
   *
   * @param returnOperation The `return` operation to update.
   * @param reg The register to use as the new return value.
   * @param measurementValues The values to use as the new return values.
   * @param rewriter The pattern rewriter to use.
   */
  static void
  updateReturnOperation(mlir::Operation* returnOperation, mlir::Value reg,
                        const std::vector<mlir::Value>& measurementValues,
                        mlir::PatternRewriter& rewriter) {
    auto* const cloned = rewriter.clone(*returnOperation);
    cloned->setOperand(0, reg);
    for (size_t i = 0; i < measurementValues.size(); i++) {
      cloned->setOperand(i + 1, measurementValues[i]);
    }
    rewriter.replaceOp(returnOperation, cloned);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp op,
                  mlir::PatternRewriter& rewriter) const override {
    if (op->hasAttr("to_replace")) {
      const std::size_t numQubits = circuit.getNqubits();

      // Prepare list of measurement results for later use.
      std::vector<mlir::Value> measurementValues(numQubits);

      // Create a new qubit register with the correct number of qubits.
      const auto& qubitType = QubitType::get(rewriter.getContext());
      const auto& memRefType = mlir::MemRefType::get(
          {static_cast<std::int64_t>(numQubits)}, qubitType);
      auto newAlloc = rewriter.create<mlir::memref::AllocOp>(
          op.getLoc(), memRefType, mlir::ValueRange{});
      newAlloc->setAttr("mqt_core", rewriter.getUnitAttr());

      // We start by first extracting each qubit from the register. The current
      // `Value` representations of each qubit are stored in the
      // `currentQubitVariables` vector.
      auto reg = newAlloc.getResult();
      std::vector<mlir::Value> currentQubitVariables(numQubits);
      for (size_t i = 0; i < numQubits; i++) {
        auto loadOp = createLoadOp(reg, i, rewriter);
        currentQubitVariables[i] = loadOp.getResult();
      }

      // Iterate over each operation in the circuit and create the corresponding
      // MLIR operations.
      for (const auto& o : circuit) {
        // Collect the positive and negative control qubits for the operation in
        // separate vectors.
        std::vector<int> controlQubitIndicesPositive;
        std::vector<int> controlQubitIndicesNegative;
        std::vector<mlir::Value> controlQubitsPositive;
        std::vector<mlir::Value> controlQubitsNegative;
        for (const auto& control : o->getControls()) {
          if (control.type == qc::Control::Type::Pos) {
            controlQubitIndicesPositive.emplace_back(control.qubit);
            controlQubitsPositive.emplace_back(
                currentQubitVariables[control.qubit]);
          } else {
            controlQubitIndicesNegative.emplace_back(control.qubit);
            controlQubitsNegative.emplace_back(
                currentQubitVariables[control.qubit]);
          }
        }

        if (o->isUnitary() && !o->isCompoundOperation()) {
          // For unitary operations, we call the `createUnitaryOp` function. We
          // then have to update the `currentQubitVariables` vector with the new
          // qubit values.
          llvm::SmallVector<mlir::Value> inQubits(o->getTargets().size());

          for (size_t i = 0; i < o->getTargets().size(); i++) {
            inQubits[i] = currentQubitVariables[o->getTargets()[i]];
          }

          UnitaryInterface newUnitaryOp = createUnitaryOp(
              op->getLoc(), o->getType(), inQubits, controlQubitsPositive,
              controlQubitsNegative, rewriter, o->getParameter());

          const size_t numTargets = o->getTargets().size();
          auto outs = newUnitaryOp.getAllOutQubits();

          // targets
          for (size_t i = 0; i < numTargets; ++i) {
            currentQubitVariables[o->getTargets()[i]] = outs[i];
          }

          // controls
          size_t base = numTargets;
          for (size_t i = 0; i < controlQubitsPositive.size(); ++i) {
            currentQubitVariables[controlQubitIndicesPositive[i]] =
                outs[base + i];
          }

          base += controlQubitsPositive.size();
          for (size_t i = 0; i < controlQubitsNegative.size(); ++i) {
            currentQubitVariables[controlQubitIndicesNegative[i]] =
                outs[base + i];
          }
        } else if (o->getType() == qc::OpType::Measure) {
          // For measurement operations, we call the `createMeasureOp` function.
          // We then update the `currentQubitVariables` and `measurementValues`
          // vectors.
          MeasureOp newMeasureOp = createMeasureOp(
              op->getLoc(), currentQubitVariables[o->getTargets()[0]],
              rewriter);
          currentQubitVariables[o->getTargets()[0]] =
              newMeasureOp.getOutQubit();
          measurementValues[o->getTargets()[0]] = newMeasureOp.getOutBit();
        } else {
          llvm::outs() << "ERROR: Unsupported operation type " << o->getType()
                       << "\n";
        }
      }

      // Now insert all the qubits back into the registers they were extracted
      // from.
      for (size_t i = 0; i < numQubits; i++) {
        createStoreOp(currentQubitVariables[i], reg, i, rewriter);
      }

      // Finally, the return operation needs to be updated with the measurement
      // results and then replace the original `alloc` operation with the
      // updated one.
      const auto returnIt =
          llvm::find_if(op->getUsers(), [](mlir::Operation* user) {
            return llvm::isa<mlir::func::ReturnOp>(user);
          });

      if (returnIt != op->getUsers().end()) {
        updateReturnOperation(*returnIt, reg, measurementValues, rewriter);
      }

      rewriter.replaceOp(op, newAlloc);
      return mlir::success();
    }
    return mlir::failure();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `FromQuantumComputationPattern`.
 *
 * @param patterns The pattern set to populate.
 * @param circuit The quantum computation to create MLIR instructions from.
 */
void populateFromQuantumComputationPatterns(mlir::RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit) {
  patterns.add<FromQuantumComputationPattern>(patterns.getContext(), circuit);
}

} // namespace mqt::ir::opt
