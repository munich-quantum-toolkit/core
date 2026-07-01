/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/UnionTable.hpp"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

#include <iostream>

namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

#define CREATE_OP_CASE_NO_PARAMS(opType)                                       \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), targetInput);                \
  })

#define CREATE_OP_CASE_ONE_PARAM(opType)                                       \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), resultTypes, qubitIn[0],     \
                          params[0]);                                          \
  })

#define CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(opType)                            \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), resultTypes, qubitIn[0],     \
                          qubitIn[1], params[0]);                              \
  })

#define CREATE_OP_CASE_TWO_PARAMS(opType)                                      \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), resultTypes, qubitIn[0],     \
                          params[0], params[1]);                               \
  })

#define CREATE_OP_CASE_TWO_PARAMS_TWO_QUBITS(opType)                           \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), resultTypes, qubitIn[0],     \
                          qubitIn[1], params[0], params[1]);                   \
  })

#define CREATE_OP_CASE_THREE_PARAMS(opType)                                    \
  .Case<opType>([&](auto) {                                                    \
    return opType::create(rewriter, op->getLoc(), resultTypes, qubitIn[0],     \
                          params[0], params[1], params[2]);                    \
  })

/**
 * @brief Result of checking how do modify a controlled gate.
 */
struct controlsToModify {
  llvm::DenseSet<Value> quantumCtrlsToRemove;
  llvm::DenseSet<Value> classicalPosCtrlsToAdd;
  llvm::DenseSet<Value> classicalNegCtrlsToAdd;
};

/**
 * This method checks whether the func::FuncOp is an entry point to the program.
 *
 * @param op The func::FuncOp to be checked.
 * @return Whether the operation is an entry point to the program.
 */
bool isEntryPoint(const func::FuncOp op) {
  const auto passthroughAttr = op->getAttrOfType<ArrayAttr>("passthrough");
  if (!passthroughAttr) {
    return false;
  }

  return llvm::any_of(passthroughAttr, [](const Attribute attr) {
    return mlir::isa<StringAttr>(attr) &&
           mlir::cast<StringAttr>(attr) == "entry_point";
  });
}

/**
 * This method moves all measurements as far to the front as possible, in order
 * to execute constant propagation more efficiently.
 *
 * @param module The module which contains the operations
 * @param ctx The MLIR context
 */
bool moveMeasurementsToFront(ModuleOp module, MLIRContext* ctx) {
  bool changed = false;
  PatternRewriter rewriter(ctx);
  module.walk([&](MeasureOp op) {
    Operation* previousInstruction = op.getQubitIn().getDefiningOp();
    Operation* previousNode = op->getPrevNode();
    while (isa<MeasureOp>(previousNode) &&
           previousInstruction != previousNode) {
      previousNode = previousNode->getPrevNode();
    }
    if (previousNode != previousInstruction) {
      rewriter.moveOpAfter(op, previousInstruction);
      changed = true;
    }
  });

  return changed;
}

/**
 * Removes a UnitaryOpInterface from the mlir context.
 *
 * @param op The qco::UnitaryOpInterface to be removed.
 * @param rewriter The used rewriter.
 */
void removeOperation(UnitaryOpInterface* op, PatternRewriter& rewriter) {
  for (const auto outQubit : op->getOutputQubits()) {
    rewriter.replaceAllUsesWith(outQubit, op->getInputForOutput(outQubit));
  }
  rewriter.eraseOp(*op);
}

/**
 * Removes a CtrlOp from the mlir context.
 *
 * @param op The qco::CtrlOp to be removed.
 * @param rewriter The used rewriter.
 */
void removeCtrlOperation(CtrlOp* op, PatternRewriter& rewriter) {
  for (const auto outQubit : op->getOutputQubits()) {
    rewriter.replaceAllUsesWith(outQubit, op->getInputForOutput(outQubit));
  }
  rewriter.eraseOp(*op);
}

/**
 * Handles a constant operation, meaning it is propagated through the union
 * table.
 *
 * @param ut Union table which contains the current quantum state
 * @param op The arith::ConstantOp which is propagated.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @return Whether the handling was successfully or interrupted.
 */
WalkResult handleConstant(UnionTable* ut, arith::ConstantOp op,
                          const std::span<Value> posClassicalCtrls,
                          const std::span<Value> negClassicalCtrls) {
  if (!posClassicalCtrls.empty() || !negClassicalCtrls.empty()) {
    throw std::logic_error("Cannot handle constant operation in conditional "
                           "branches during constant propagation.");
  }

  Value const res = op.getResult();
  auto attr = op.getValue();
  if (const auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    ut->propagateIntAlloc(res, intAttr.getInt());
  }
  if (const auto doubleAttr = dyn_cast<FloatAttr>(attr)) {
    ut->propagateDoubleAlloc(res, doubleAttr.getValueAsDouble());
  }
  if (const auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    if (boolAttr.getValue()) {
      ut->propagateIntAlloc(res, 1);
    } else {
      ut->propagateIntAlloc(res, 0);
    }
  }
  return WalkResult::advance();
}

/**
 * Checks if only a global phase is added to the quantum machine state. If yes
 * and if the global phase is not = 1, it adds a global phase gate.
 *
 * @param ut Union table which contains the current quantum state
 * @param op The qco::UnitaryOpInterface which is propagated.
 * @param ctrlsQuantum The quantum control values considered in the operation.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @param rewriter The used rewriter.
 * @param targetValues The target values (non-empty if the method was called via
 * a quantum control (qco::CtrlOp).
 * @return Whether there is only a global phase added.
 */
bool addsOnlyGlobalPhase(UnionTable* ut, UnitaryOpInterface* op,
                         const std::span<Value> ctrlsQuantum,
                         const std::span<Value> posClassicalCtrls,
                         const std::span<Value> negClassicalCtrls,
                         PatternRewriter& rewriter,
                         const std::span<Value> targetValues) {
  bool addsGlobalPhase = false;
  if (isa<IdOp>(op) || isa<ZOp>(op) || isa<SOp>(op) || isa<SdgOp>(op) ||
      isa<TOp>(op) || isa<TdgOp>(op)) {
    const auto inputQubit =
        targetValues.empty() ? op->getInputQubit(0) : targetValues[0];
    const auto addedGlobalPhase = ut->globalPhaseThatIsAdded(
        *op, inputQubit, ctrlsQuantum, posClassicalCtrls, negClassicalCtrls);
    if (addedGlobalPhase.has_value()) {
      addsGlobalPhase = true;
      const auto phase = addedGlobalPhase.value();
      if (std::norm(phase) > 1e-4) {
        GPhaseOp::create(rewriter, op->getLoc(), phase);
      }
    }
  }
  return addsGlobalPhase;
}

/**
 * Removes the given quantum controls from a CtrlOp, potentially removing all
 * controls and only leaving the (formerly controlled) gate in the body.
 *
 * @param op The qco::CtrlOp whose controls are removed.
 * @param ctrlsToRemove The controls which should be removed from the CtrlOp.
 * @param rewriter The used rewriter
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return The operation without given controls.
 */
Operation* removeCtrlsOfGate(CtrlOp* op,
                             const llvm::DenseSet<Value>& ctrlsToRemove,
                             PatternRewriter& rewriter,
                             std::span<Operation*>& worklist) {
  for (const auto& qubitCtrl : ctrlsToRemove) {
    rewriter.replaceAllUsesWith(op->getOutputForInput(qubitCtrl), qubitCtrl);
  }
  if (ctrlsToRemove.size() == op->getNumControls()) {
    // Remove Ctrl completely
    const auto targetInput = op->getInputTargets();
    const TypeRange resultTypes(op->getOutputTargets());
    const auto paramsRange = op->getParameters();
    const std::vector<Value> qubitIn = {targetInput.begin(), targetInput.end()};
    const std::vector<Value> params = {paramsRange.begin(), paramsRange.end()};
    const auto newOp =
        mlir::TypeSwitch<Operation*, Operation*>(op->getBodyUnitary()) CREATE_OP_CASE_NO_PARAMS(
            IdOp) CREATE_OP_CASE_NO_PARAMS(HOp) CREATE_OP_CASE_NO_PARAMS(XOp)
            CREATE_OP_CASE_NO_PARAMS(YOp) CREATE_OP_CASE_NO_PARAMS(
                ZOp) CREATE_OP_CASE_NO_PARAMS(SOp) CREATE_OP_CASE_NO_PARAMS(SdgOp)
                CREATE_OP_CASE_NO_PARAMS(TOp) CREATE_OP_CASE_NO_PARAMS(
                    TdgOp) CREATE_OP_CASE_NO_PARAMS(SXOp) CREATE_OP_CASE_NO_PARAMS(SXdgOp)
                    CREATE_OP_CASE_ONE_PARAM(RXOp) CREATE_OP_CASE_ONE_PARAM(
                        RYOp) CREATE_OP_CASE_ONE_PARAM(RZOp)
                        CREATE_OP_CASE_ONE_PARAM(POp) CREATE_OP_CASE_TWO_PARAMS(
                            ROp) CREATE_OP_CASE_TWO_PARAMS(U2Op)
                            CREATE_OP_CASE_THREE_PARAMS(UOp) CREATE_OP_CASE_NO_PARAMS(
                                SWAPOp) CREATE_OP_CASE_NO_PARAMS(iSWAPOp)
                                CREATE_OP_CASE_NO_PARAMS(
                                    DCXOp) CREATE_OP_CASE_NO_PARAMS(ECROp)
                                    CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                        RXXOp) CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(RYYOp)
                                        CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                            RZXOp)
                                            CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                                RZZOp)
                                                CREATE_OP_CASE_TWO_PARAMS_TWO_QUBITS(
                                                    XXPlusYYOp)
                                                    CREATE_OP_CASE_TWO_PARAMS_TWO_QUBITS(
                                                        XXMinusYYOp)
                                                        .Default(
                                                            [&](auto)
                                                                -> Operation* {
                                                              throw std::
                                                                  runtime_error(
                                                                      "Unsu"
                                                                      "ppor"
                                                                      "ted "
                                                                      "oper"
                                                                      "atio"
                                                                      "n");
                                                            });
    auto newUnitary = static_cast<UnitaryOpInterface>(newOp);
    for (const auto inTarget : newUnitary.getInputQubits()) {
      rewriter.replaceAllUsesWith(op->getOutputForInput(inTarget),
                                  newUnitary.getOutputForInput(inTarget));
    }
    for (const auto ctrlQubit : op->getOutputControls()) {
      rewriter.replaceAllUsesWith(ctrlQubit, op->getInputForOutput(ctrlQubit));
    }
    rewriter.eraseOp(*op);
    std::ranges::replace(worklist, *op, newOp);

    return newOp;
  }
  std::vector<Value> newControlIn;
  for (const auto& ctrls : op->getInputControls()) {
    if (!ctrlsToRemove.contains(ctrls)) {
      newControlIn.push_back(ctrls);
    }
  }
  const auto newCtrl =
      CtrlOp::create(rewriter, op->getLoc(), newControlIn, op->getTargetsIn(),
                     [&](const ValueRange target) {
                       return SmallVector<Value>{
                           XOp::create(rewriter, op->getLoc(), target[0])};
                     });

  auto newUnitary = static_cast<UnitaryOpInterface>(newCtrl);
  for (const auto inTarget : newUnitary.getInputQubits()) {
    rewriter.replaceAllUsesWith(op->getOutputForInput(inTarget),
                                newUnitary.getOutputForInput(inTarget));
  }
  for (const auto ctrlQubit : op->getOutputControls()) {
    rewriter.replaceAllUsesWith(ctrlQubit, op->getInputForOutput(ctrlQubit));
  }
  rewriter.eraseOp(*op);
  std::ranges::replace(worklist, *op, newUnitary);

  return newUnitary;
}

/**
 * Handles a unitary gate, meaning it is propagated through the union table.
 *
 * @param ut Union table which contains the current quantum state
 * @param op The qco::UnitaryOpInterface which is propagated.
 * @param ctrlsQuantum The quantum control values considered in the operation.
 * @param newCtrlsQuantum The values the quantum control values become after
 * the operation.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @param targetValues The target values (non-empty if the method was called via
 * a quantum control (qco::CtrlOp).
 * @param resultValues The values the target values become after the operation.
 * @return Whether the handling was successfully or interrupted.
 */
WalkResult handleUnitary(UnionTable* ut, UnitaryOpInterface* op,
                         const std::span<Value> ctrlsQuantum,
                         const std::span<Value> newCtrlsQuantum,
                         const std::span<Value> posClassicalCtrls,
                         const std::span<Value> negClassicalCtrls,
                         const std::span<Value> targetValues = {},
                         const std::span<Value> resultValues = {}) {

  const auto params = op->getParameters();
  std::vector<Value> paramValues = {params.begin(), params.end()};
  if (targetValues.empty() && resultValues.empty()) {
    const auto targets = op->getInputTargets();
    const auto results = op->getOutputTargets();
    std::vector<Value> targetVecs = {targets.begin(), targets.end()};
    std::vector<Value> resultVecs = {results.begin(), results.end()};
    ut->propagateGate(*op, targetVecs, resultVecs, ctrlsQuantum,
                      newCtrlsQuantum, posClassicalCtrls, negClassicalCtrls,
                      paramValues);
  } else if (targetValues.size() == resultValues.size()) {
    ut->propagateGate(*op, targetValues, resultValues, ctrlsQuantum,
                      newCtrlsQuantum, posClassicalCtrls, negClassicalCtrls,
                      paramValues);
  } else {
    throw std::invalid_argument(
        "Given targetValues and resultValues need to be of same size.");
  }

  return WalkResult::advance();
}

/**
 * Handles an uncontrolled unitary gate. First, it is checked whether the gate
 * only adds a global phase in case it is a diagonal gate. Then the gate is
 * either propagated on the union table or removed/replaced by a gloal phase
 * gate.
 *
 * @param ut Union table which contains the current quantum state
 * @param op The qco::UnitaryOpInterface which is propagated.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @param rewriter The used rewriter
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return Whether the handling was successfully or interrupted.
 */
WalkResult handleUncontrolledUnitary(UnionTable* ut, UnitaryOpInterface* op,
                                     const std::span<Value> posClassicalCtrls,
                                     const std::span<Value> negClassicalCtrls,
                                     PatternRewriter& rewriter,
                                     std::span<Operation*>& worklist) {
  const auto targets = op->getInputTargets();
  std::vector<Value> targetVecs = {targets.begin(), targets.end()};

  // Check if a diagonal gate only adds a global phase
  const bool addsGlobalPhase = addsOnlyGlobalPhase(
      ut, op, {}, posClassicalCtrls, negClassicalCtrls, rewriter, targetVecs);
  if (addsGlobalPhase) {
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    removeOperation(op, rewriter);
    return WalkResult::advance();
  }

  return handleUnitary(ut, op, {}, {}, posClassicalCtrls, negClassicalCtrls);
}

/**
 * Handles a CtrlOp. First, it is checked whether the CtrlOp is executable
 * considering the current quantum machine state. Then the CtrlOp is either
 * propagated on the union table or removed/replaced by an unconditional gate.
 * If it is replaced by an unconditional gate, the gate is propagated.
 *
 * @param ut Union table which contains the current quantum state
 * @param op The qco::CtrlOp which is propagated.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @param rewriter The used rewriter
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return Whether the handling was successfully or interrupted.
 */
WalkResult handleCtrlOp(UnionTable* ut, CtrlOp* op,
                        const std::span<Value> posClassicalCtrls,
                        const std::span<Value> negClassicalCtrls,
                        PatternRewriter& rewriter,
                        std::span<Operation*>& worklist) {
  // Avoid to address body twice
  op->walk<WalkOrder::PreOrder>([&](Operation* bodyOp) {
    std::ranges::replace(worklist, bodyOp, static_cast<Operation*>(nullptr));
  });

  const auto inputCtrls = op->getInputControls();
  std::vector<Value> inCtrlValues = {inputCtrls.begin(), inputCtrls.end()};

  // Check if gate is executable
  const auto satisfiable = ut->areThereSatisfiableCombinations(
      inCtrlValues, posClassicalCtrls, negClassicalCtrls);
  const auto superfluousCtrls = ut->getSuperfluousControls(
      inCtrlValues, posClassicalCtrls, negClassicalCtrls);
  if (superfluousCtrls.completelySuperfluous || !satisfiable) {
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    removeCtrlOperation(op, rewriter);
    return WalkResult::advance();
  }

  // Collect quantum values to remove and classical values to add
  controlsToModify ctrlsToMod;
  ctrlsToMod.quantumCtrlsToRemove = superfluousCtrls.superfluousQubits;
  for (const auto superfluousQ : ctrlsToMod.quantumCtrlsToRemove) {
    std::erase(inCtrlValues, superfluousQ);
  }

  for (const auto qCtrl : inCtrlValues) {
    auto qCtrlValuesWithoutCurrent = inCtrlValues;
    std::erase(qCtrlValuesWithoutCurrent, qCtrl);
    if (ut->isQubitImplied(qCtrl, qCtrlValuesWithoutCurrent, posClassicalCtrls,
                           negClassicalCtrls)) {
      std::erase(inCtrlValues, qCtrl);
      ctrlsToMod.quantumCtrlsToRemove.insert(qCtrl);
    } else if (auto v = ut->getValueThatIsEquivalentToQubit(qCtrl);
               !v.empty()) {
      for (const auto& [value, b] : v) {
        if (b) {
          ctrlsToMod.classicalPosCtrlsToAdd.insert(value);
        } else {
          ctrlsToMod.classicalNegCtrlsToAdd.insert(value);
        }
        break;
      }
    }
  }

  if (!ctrlsToMod.quantumCtrlsToRemove.empty()) {
    auto newOp = removeCtrlsOfGate(op, ctrlsToMod.quantumCtrlsToRemove,
                                   rewriter, worklist);
    return WalkResult::advance();
  }

  auto body = op->getBodyUnitary();
  // Make sure that the right qubits in the right order are passed to the
  // propagation of the body unitary
  std::vector<Value> targetQubits;
  const auto numTargets = op->getNumTargets();
  targetQubits.reserve(numTargets);
  const auto arguments = op->getRegion().getArguments();
  for (unsigned int argIndex = 0; argIndex < arguments.size(); ++argIndex) {
    for (unsigned int i = 0; i < numTargets; ++i) {
      if (arguments[i] == body.getInputTarget(i)) {
        targetQubits.insert(targetQubits.begin() + i,
                            op->getInputTarget(argIndex));
        break;
      }
    }
  }

  std::vector<Value> resultQubits;
  resultQubits.reserve(numTargets);
  const auto yieldOP = cast<YieldOp>(*op->getBody()->rbegin());
  for (unsigned int uOpOutIndex = 0; uOpOutIndex < numTargets; ++uOpOutIndex) {
    for (unsigned int i = 0; i < numTargets; ++i) {
      if (yieldOP->getOperand(i) == body.getOutputTarget(uOpOutIndex)) {
        resultQubits.insert(resultQubits.begin() + i,
                            op->getOutputTarget(uOpOutIndex));
        break;
      }
    }
  }

  const auto outputCtrls = op->getOutputControls();
  std::vector<Value> outCtrlValues = {outputCtrls.begin(), outputCtrls.end()};

  // Check if a diagonal gate only adds a global phase
  const bool addsGlobalPhase =
      addsOnlyGlobalPhase(ut, &body, inCtrlValues, posClassicalCtrls,
                          negClassicalCtrls, rewriter, targetQubits);
  if (addsGlobalPhase) {
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    removeCtrlOperation(op, rewriter);
    return WalkResult::advance();
  }

  return handleUnitary(ut, &body, inCtrlValues, outCtrlValues,
                       posClassicalCtrls, negClassicalCtrls, targetQubits,
                       resultQubits);
}

/**
 * Iterates through the worklist of operators and propagates the quantum machine
 * state through the union table. The iteration can be called with specific
 * classical controls which are considered in every propagation.
 *
 * @param rewriter The used rewriter
 * @param ut The union table which contains the current quantum machine state.
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @param posClassicalCtrls The positive classical controls considered in every
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in every
 * operation.
 * @return Whether the iteration was successfully or interrupted.
 */
LogicalResult iterateThroughWorklist(PatternRewriter& rewriter, UnionTable* ut,
                                     std::span<Operation*>& worklist,
                                     const std::span<Value> posClassicalCtrls,
                                     const std::span<Value> negClassicalCtrls) {
  /// Iterate work-list.
  bool addedAtLeastOneQubit = false;
  for (Operation* curr : worklist) {
    if (addedAtLeastOneQubit && ut->areStatesAllTop()) {
      return success();
    }
    if (curr == nullptr) {
      continue; // Skip erased ops.
    }
    auto n = curr->getName().stripDialect().str();
    std::string oName =
        "Op: " + curr->getName().getStringRef().str() +
        " dialect: " + curr->getName().getDialectNamespace().str();

    rewriter.setInsertionPoint(curr);

    const auto res =
        TypeSwitch<Operation*, WalkResult>(curr)
            /// qco Dialect
            .Case<CtrlOp>([&](CtrlOp op) {
              return handleCtrlOp(ut, &op, posClassicalCtrls, negClassicalCtrls,
                                  rewriter, worklist);
            })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
              return handleUncontrolledUnitary(ut, &op, posClassicalCtrls,
                                               negClassicalCtrls, rewriter,
                                               worklist);
            })
            .Case<ResetOp>([&](const ResetOp op) {
              ut->propagateReset(op->getOperand(0), op->getResult(0),
                                 posClassicalCtrls, negClassicalCtrls);
              return WalkResult::advance();
            })
            .Case<MeasureOp>([&](const MeasureOp op) {
              ut->propagateMeasurement(op->getOperand(0), op->getResult(0),
                                       op->getResult(1), posClassicalCtrls,
                                       negClassicalCtrls);
              return WalkResult::advance();
            })
            .Case<AllocOp>([&](const AllocOp op) {
              addedAtLeastOneQubit = true;
              ut->propagateQubitAlloc(op->getOperand(0));
              return WalkResult::advance();
            })
            //         .Case<StaticOp>(
            //             [&](const StaticOp op) { return handleStaticOp(ut,
            //             op); })
            .Case<SinkOp>([&]([[maybe_unused]] SinkOp op) {
              return WalkResult::advance();
            })
            //         .Case<IfOp>([&](const IfOp op) {
            //           return handleIf(qcp, op, worklist, posClassicalCtrls,
            //                           negClassicalCtrls, rewriter);
            //         })
            //         .Case<YieldOp>([&]([[maybe_unused]] YieldOp op) {
            //           return WalkResult::advance();
            //         })
            //         /// built-in Dialect
            //         .Case<ModuleOp>([&]([[maybe_unused]] ModuleOp op) {
            //           return WalkResult::advance();
            //         })
            // qtensor dialect
            .Case<qtensor::AllocOp>([&]([[maybe_unused]] qtensor::AllocOp op) {
              return WalkResult::advance();
            })
            .Case<qtensor::DeallocOp>(
                [&]([[maybe_unused]] qtensor::DeallocOp op) {
                  return WalkResult::advance();
                })
            .Case<qtensor::ExtractOp>([&](const qtensor::ExtractOp op) {
              addedAtLeastOneQubit = true;
              ut->propagateQubitAlloc(op->getResult(1));
              return WalkResult::advance();
            })
            .Case<qtensor::InsertOp>(
                [&]([[maybe_unused]] qtensor::InsertOp op) {
                  return WalkResult::advance();
                })
            // memref Dialect
            // .Case<memref::AllocOp>([&](const memref::AllocOp op) {
            //   addedAtLeastOneQubit = true;
            //   ut->propagateQubitAlloc(op->getOpResult(0));
            //   return WalkResult::advance();
            // })
            //         .Case<memref::AllocaOp>([&](const memref::AllocaOp op) {
            //           return handleAlloca(ut, op);
            //         })
            //         .Case<memref::DeallocOp>(
            //             [&]([[maybe_unused]] const memref::DeallocOp op) {
            //               return WalkResult::advance();
            //             })
            //         .Case<memref::LoadOp>([&](const memref::LoadOp op) {
            //           addedAtLeastOneQubit = true;
            //           return handleLoad(ut, op);
            //         })
            //         .Case<memref::StoreOp>([&](const memref::StoreOp op) {
            //           return handleStore(ut, op, posClassicalCtrls,
            //           negClassicalCtrls);
            //         })
            // arith dialect
            .Case<arith::ConstantOp>([&](const arith::ConstantOp op) {
              return handleConstant(ut, op, posClassicalCtrls,
                                    negClassicalCtrls);
            })
            //         .Case<arith::XOrIOp>(
            //             [&](const arith::XOrIOp op) { return
            //             handleXOrIOp(qcp, op);
            //             })
            //         .Case<arith::AndIOp>(
            //             [&](const arith::AndIOp op) { return
            //             handleAndIOp(qcp, op);
            //             })
            // func Dialect
            .Case<func::FuncOp>([&](const func::FuncOp op) {
              if (!isEntryPoint(op)) {
                throw std::domain_error(
                    "Constant propagation does not support nested functions.");
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            })
            .Case<func::ReturnOp>([&]([[maybe_unused]] func::ReturnOp op) {
              return WalkResult::advance();
            })
            .Default([](auto) {
              throw std::runtime_error("Unsupported operation");
              return WalkResult::interrupt();
            });

    if (res.wasInterrupted()) {
      return failure();
    }
  }
  return success();
}

/**
 * @brief Do constant propagation.
 *
 * @details
 * Collects all functions marked with the 'entry_point' attribute, builds
 * a preorder worklist of their operations, and processes that list.
 *
 * @note
 * We consciously avoid MLIR pattern drivers: Idiomatic MLIR
 * transformation patterns are independent and order-agnostic. Since we
 * require state-sharing between patterns for the transformation we
 * violate this assumption. Essentially this is also the reason why we
 * can't utilize MLIR's `applyPatternsGreedily` function. Moreover, we
 * require pre-order traversal which current drivers of MLIR don't
 * support. However, even if such a driver would exist, it would probably
 * not return logical results which we require for error-handling
 * (similarly to `walkAndApplyPatterns`). Consequently, a custom driver
 * would be required in any case, which adds unnecessary code to maintain.
 *
 * @param module The module which contains the operations
 * @param ctx The MLIR context
 * @return Success if constant propagation has been applied successfully
 */
LogicalResult applyCP(ModuleOp module, MLIRContext* ctx) {
  PatternRewriter rewriter(ctx);

  /// Prepare work-list.
  std::vector<Operation*> worklist;

  for (const auto func : module.getOps<func::FuncOp>()) {

    if (!isEntryPoint(func)) {
      continue; // Ignore non entry_point functions for now.
    }
    func->walk<WalkOrder::PreOrder>(
        [&](Operation* op) { worklist.push_back(op); });
  }

  // TODO: Take maximum from params
  auto ut = UnionTable(16, 4);

  std::span wl = {worklist.begin(), worklist.end()};

  return iterateThroughWorklist(rewriter, &ut, wl, {}, {});
}

/**
 * This pass applies constant propagation to a circuit. It assumes that all
 * states start in |0> and removes quantum instructions that are superfluous
 * when the current state is considered. It also replaces quantum resources by
 * classical resources.
 */
struct ConstantPropagation final
    : impl::ConstantPropagationBase<ConstantPropagation> {
  using ConstantPropagationBase::ConstantPropagationBase;

  void runOnOperation() override {
    if (failed(applyCP(getOperation(), &getContext()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
