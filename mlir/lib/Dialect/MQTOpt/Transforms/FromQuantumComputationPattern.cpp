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
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <stdexcept>
#include <vector>

namespace mqt::ir::opt {
/// Analysis pattern that creates MLIR instructions from a given
/// qc::QuantumComputation. These instructions replace an existing `AllocOp`
/// that has the `to_replace` attribute set.
struct FromQuantumComputationPattern final : mlir::OpRewritePattern<AllocOp> {
  qc::QuantumComputation& circuit; // NOLINT(*-avoid-const-or-ref-data-members)

  explicit FromQuantumComputationPattern(mlir::MLIRContext* context,
                                         qc::QuantumComputation& qc)
      : OpRewritePattern(context), circuit(qc) {}

  /**
   * @brief Creates an ExtractOp that extracts the qubit at the given index from
   * the given register.
   *
   * @param reg The register to extract the qubit from.
   * @param index The index of the qubit to extract.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created ExtractOp.
   */
  static ExtractOp createRegisterAccess(mlir::Value reg, const size_t index,
                                        mlir::PatternRewriter& rewriter) {
    return rewriter.create<ExtractOp>(
        reg.getLoc(),
        mlir::TypeRange{QubitRegisterType::get(rewriter.getContext()),
                        QubitType::get(rewriter.getContext())},
        reg, nullptr,
        rewriter.getI64IntegerAttr(static_cast<std::int64_t>(index)));
  }

  /**
   * @brief Creates an InsertOp that inserts the given qubit into the given
   * register at the specified index.
   *
   * @param allocOp The AllocOp that defined the original register.
   * @param reg The register into which the qubit will be inserted.
   * @param qubit The qubit to insert into the register.
   * @param index The index at which to insert the qubit.
   * @param rewriter The pattern rewriter to use.
   *
   * @return The created InsertOp.
   */
  static InsertOp createRegisterInsert(AllocOp allocOp,
                                       mlir::TypedValue<QubitRegisterType> reg,
                                       mlir::Value qubit, const size_t index,
                                       mlir::PatternRewriter& rewriter) {
    auto insert = rewriter.create<InsertOp>(
        allocOp.getLoc(),
        QubitRegisterType::get(rewriter.getContext()), // Result type
        reg,            // The register to insert into
        qubit, nullptr, // The qubit to insert
        rewriter.getI64IntegerAttr(static_cast<std::int64_t>(index)) // Index
    );
    return insert;
  }

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
                  const std::vector<mlir::Value>& inQubits,
                  mlir::ValueRange controlQubitsPositive,
                  mlir::ValueRange controlQubitsNegative,
                  mlir::PatternRewriter& rewriter) {
    switch (type) {
    case qc::OpType::I:
      return rewriter.create<IOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::H:
      return rewriter.create<HOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::X:
      return rewriter.create<XOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Y:
      return rewriter.create<YOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Z:
      return rewriter.create<ZOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::S:
      return rewriter.create<SOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Sdg:
      return rewriter.create<SdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::T:
      return rewriter.create<TOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Tdg:
      return rewriter.create<TdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::V:
      return rewriter.create<VOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Vdg:
      return rewriter.create<VdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::SX:
      return rewriter.create<SXOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::SXdg:
      return rewriter.create<SXdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::SWAP:
      return rewriter.create<SWAPOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::iSWAP:
      return rewriter.create<iSWAPOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::iSWAPdg:
      return rewriter.create<iSWAPdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Peres:
      return rewriter.create<PeresOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::Peresdg:
      return rewriter.create<PeresdgOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::DCX:
      return rewriter.create<DCXOp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
    case qc::OpType::ECR:
      return rewriter.create<ECROp>(
          loc, inQubits[0].getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, mlir::ValueRange{},
          mlir::ValueRange{inQubits[0], inQubits[1]}, controlQubitsPositive,
          controlQubitsNegative);
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
   * @param register The register to use as the new return value.
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
  matchAndRewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    if (op->hasAttr("to_replace")) {

      const std::size_t numQubits = circuit.getNqubits();

      // Prepare list of measurement results for later use.
      std::vector<mlir::Value> measurementValues(numQubits);

      // Create a new qubit register with the correct number of qubits.
      auto newAlloc = rewriter.create<AllocOp>(
          op.getLoc(), QubitRegisterType::get(rewriter.getContext()), nullptr,
          rewriter.getIntegerAttr(rewriter.getI64Type(),
                                  static_cast<std::int64_t>(numQubits)));
      newAlloc->setAttr("mqt_core", rewriter.getUnitAttr());

      // We start by first extracting each qubit from the register. The current
      // `Value` representations of each qubit are stored in the
      // `currentQubitVariables` vector.
      auto currentRegister = newAlloc.getResult();
      std::vector<mlir::Value> currentQubitVariables(numQubits);
      for (size_t i = 0; i < numQubits; i++) {
        auto newRegisterAccess =
            createRegisterAccess(currentRegister, i, rewriter);
        currentQubitVariables[i] = newRegisterAccess.getOutQubit();
        currentRegister = newRegisterAccess.getOutQreg();
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

        if (o->getType() == qc::OpType::I || o->getType() == qc::OpType::X ||
            o->getType() == qc::OpType::H || o->getType() == qc::OpType::Y ||
            o->getType() == qc::OpType::Z || o->getType() == qc::OpType::S ||
            o->getType() == qc::OpType::Sdg || o->getType() == qc::OpType::T ||
            o->getType() == qc::OpType::Tdg || o->getType() == qc::OpType::V ||
            o->getType() == qc::OpType::Vdg || o->getType() == qc::OpType::SX ||
            o->getType() == qc::OpType::SXdg ||
            o->getType() == qc::OpType::SWAP ||
            o->getType() == qc::OpType::iSWAP ||
            o->getType() == qc::OpType::iSWAPdg ||
            o->getType() == qc::OpType::Peres ||
            o->getType() == qc::OpType::Peresdg ||
            o->getType() == qc::OpType::DCX ||
            o->getType() == qc::OpType::ECR) {
          // For unitary operations, we call the `createUnitaryOp` function. We
          // then have to update the `currentQubitVariables` vector with the new
          // qubit values.
          std::vector<mlir::Value> inQubits(2);
          if (o->getTargets().size() < 2) {
            llvm::errs() << "Operation type: " << qc::toString(o->getType())
                         << "\n"
                         << "Number of Qubits: " << o->getNqubits() << "\n"
                         << "Targets size:" << o->getTargets().size() << "\n";
            inQubits[0] = currentQubitVariables[o->getTargets()[0]];
          } else {
            inQubits[0] = currentQubitVariables[o->getTargets()[0]];
            inQubits[1] = currentQubitVariables[o->getTargets()[1]];
          }
          UnitaryInterface newUnitaryOp = createUnitaryOp(
              op->getLoc(), o->getType(), inQubits, controlQubitsPositive,
              controlQubitsNegative, rewriter);
          currentQubitVariables[o->getTargets()[0]] =
              newUnitaryOp.getAllOutQubits()[0];
          if (o->getTargets().size() > 1) {
            currentQubitVariables[o->getTargets()[1]] =
                newUnitaryOp.getAllOutQubits()[1];
          }
          for (size_t i = 0; i < controlQubitsPositive.size(); i++) {
            currentQubitVariables[controlQubitIndicesPositive[i]] =
                newUnitaryOp.getAllOutQubits()[i + 1];
          }
          for (size_t i = 0; i < controlQubitsNegative.size(); i++) {
            currentQubitVariables[controlQubitIndicesNegative[i]] =
                newUnitaryOp
                    .getAllOutQubits()[i + 1 + controlQubitsPositive.size()];
          }
        } else if (o->getType() == qc::OpType::Measure) {
          // For measurement operations, we call the `createMeasureOp` function.
          // We then update the `currentQubitVariables` and `measurementValues`
          // vectors.
          MeasureOp newMeasureOp = createMeasureOp(
              op->getLoc(), currentQubitVariables[o->getTargets()[0]],
              rewriter);
          currentQubitVariables[o->getTargets()[0]] =
              newMeasureOp.getOutQubits()[0];
          measurementValues[o->getTargets()[0]] = newMeasureOp.getOutBits()[0];
        } else {
          llvm::outs() << "ERROR: Unsupported operation type " << o->getType()
                       << "\n";
        }
      }

      // Now insert all the qubits back into the registers they were extracted
      // from.
      for (size_t i = 0; i < numQubits; i++) {
        auto insertOp = createRegisterInsert(
            newAlloc, currentRegister, currentQubitVariables[i], i, rewriter);
        // Keep track of qubit register access
        currentRegister = insertOp.getOutQreg();
      }

      // Finally, the return operation needs to be updated with the measurement
      // results and then replace the original `alloc` operation with the
      // updated one.
      const auto returnIt =
          llvm::find_if(op->getUsers(), [](mlir::Operation* user) {
            return llvm::isa<mlir::func::ReturnOp>(user);
          });

      if (returnIt != op->getUsers().end()) {
        updateReturnOperation(*returnIt, currentRegister, measurementValues,
                              rewriter);
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
