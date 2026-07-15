/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Qua#include
 * "mlir/Dialect/QCO/IR/QCOInterfaces.h"ntum Software Company GmbH All rights
 * reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Optimizations/ConstantPropagation/UnionTable.hpp"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
/**
 * @brief Result of checking how do modify a controlled gate.
 */
struct ControlsToModify {
  llvm::DenseSet<mlir::Value> quantumCtrlsToRemove;
  llvm::DenseSet<mlir::Value> classicalPosCtrlsToAdd;
  llvm::DenseSet<mlir::Value> classicalNegCtrlsToAdd;
};
} // namespace

namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

#define CREATE_OP_CASE_NO_PARAMS(opType)                                       \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0]);               \
  })

#define CREATE_OP_CASE_NO_PARAMS_TWO_QUBITS(opType)                            \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0], qubitsIn[1]);  \
  })

#define CREATE_OP_CASE_ONE_PARAM(opType)                                       \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0],                \
                          gate.getTheta());                                    \
  })

#define CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(opType)                            \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0], qubitsIn[1],   \
                          gate.getTheta());                                    \
  })

#define CREATE_OP_CASE_TWO_PARAMS(opType)                                      \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0],                \
                          gate.getTheta(), gate.getPhi());                     \
  })

#define CREATE_OP_CASE_TWO_PARAMS_TWO_QUBITS(opType)                           \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0], qubitsIn[1],   \
                          gate.getTheta(), gate.getPhi());                     \
  })

#define CREATE_OP_CASE_PLUS_MINUS_OPS(opType)                                  \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0], qubitsIn[1],   \
                          gate.getTheta(), gate.getBeta());                    \
  })

#define CREATE_OP_CASE_THREE_PARAMS(opType)                                    \
  .Case<opType>([&](opType gate) {                                             \
    return opType::create(rewriter, gate.getLoc(), qubitsIn[0],                \
                          gate.getTheta(), gate.getPhi(), gate.getLambda());   \
  })

static LogicalResult iterateThroughWorklist(PatternRewriter& rewriter,
                                            UnionTable* ut,
                                            std::span<Operation*>& worklist,
                                            std::span<Value> posClassicalCtrls,
                                            std::span<Value> negClassicalCtrls);

/**
 * This method checks whether the func::FuncOp is an entry point to the program.
 *
 * @param op The func::FuncOp to be checked.
 * @return Whether the operation is an entry point to the program.
 */
static bool isEntryPoint(const func::FuncOp op) {
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
static void moveMeasurementsToFront(ModuleOp module, MLIRContext* ctx) {
  bool changed = false;
  do {
    changed = false;
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
  } while (changed);
}

/**
 * Removes a UnitaryOpInterface from the mlir context.
 *
 * @param op The qco::UnitaryOpInterface to be removed.
 * @param rewriter The used rewriter.
 */
static void removeOperation(UnitaryOpInterface* op, PatternRewriter& rewriter) {
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
static void removeCtrlOperation(CtrlOp* op, PatternRewriter& rewriter) {
  for (const auto outQubit : op->getOutputQubits()) {
    rewriter.replaceAllUsesWith(outQubit, op->getInputForOutput(outQubit));
  }
  rewriter.eraseOp(*op);
}

/**
 * Moves either the then or else block of an if operation out of the operation.
 * Removes the terminator of the block both from the block and the worklist.
 *
 * @param ifOperation The operation whose then or else block gets inlined.
 * @param block The block that gets inlined.
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @param rewriter The used rewriter.
 */
static void inlineBranchBlock(IfOp* ifOperation, Block* block,
                              std::span<Operation*>& worklist,
                              PatternRewriter& rewriter) {
  auto* const operation = ifOperation->getOperation();
  Operation* terminator = block->getTerminator();
  const auto results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, operation, ifOperation->getQubits());
  rewriter.replaceOp(operation, results);
  rewriter.eraseOp(terminator);
  std::ranges::replace(worklist, terminator, static_cast<Operation*>(nullptr));
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
static WalkResult handleConstant(UnionTable* ut, arith::ConstantOp op,
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
static bool addsOnlyGlobalPhase(UnionTable* ut, UnitaryOpInterface* op,
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
 * Creates a new gate at the location of the given gate and of the type of the
 * given gate.
 *
 * @param op The operation whose type and location is used.
 * @param rewriter The used rewriter.
 * @param qubitsIn A span of target inputs.
 * @return The newly created gate.
 */
static Operation*
createOperationFromUnitaryOperation(Operation* op, PatternRewriter& rewriter,
                                    const std::span<Value> qubitsIn) {
  auto* const newOp =
      mlir::TypeSwitch<Operation*, Operation*>(op)
          .Case<U2Op>([&](U2Op gate) {
            return U2Op::create(rewriter, gate.getLoc(), qubitsIn[0],
                                gate.getPhi(), gate.getLambda());
          }) CREATE_OP_CASE_NO_PARAMS(IdOp) CREATE_OP_CASE_NO_PARAMS(HOp)
              CREATE_OP_CASE_NO_PARAMS(XOp) CREATE_OP_CASE_NO_PARAMS(
                  YOp) CREATE_OP_CASE_NO_PARAMS(ZOp) CREATE_OP_CASE_NO_PARAMS(SOp)
                  CREATE_OP_CASE_NO_PARAMS(SdgOp) CREATE_OP_CASE_NO_PARAMS(
                      TOp) CREATE_OP_CASE_NO_PARAMS(TdgOp)
                      CREATE_OP_CASE_NO_PARAMS(SXOp) CREATE_OP_CASE_NO_PARAMS(
                          SXdgOp) CREATE_OP_CASE_ONE_PARAM(RXOp)
                          CREATE_OP_CASE_ONE_PARAM(RYOp) CREATE_OP_CASE_ONE_PARAM(
                              RZOp) CREATE_OP_CASE_ONE_PARAM(POp)
                              CREATE_OP_CASE_TWO_PARAMS(ROp) CREATE_OP_CASE_THREE_PARAMS(
                                  UOp) CREATE_OP_CASE_NO_PARAMS_TWO_QUBITS(SWAPOp)
                                  CREATE_OP_CASE_NO_PARAMS_TWO_QUBITS(
                                      iSWAPOp) CREATE_OP_CASE_NO_PARAMS_TWO_QUBITS(DCXOp)
                                      CREATE_OP_CASE_NO_PARAMS_TWO_QUBITS(
                                          ECROp) CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(RXXOp)
                                          CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                              RYYOp)
                                              CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                                  RZXOp)
                                                  CREATE_OP_CASE_ONE_PARAM_TWO_QUBITS(
                                                      RZZOp)
                                                      CREATE_OP_CASE_PLUS_MINUS_OPS(
                                                          XXPlusYYOp)
                                                          CREATE_OP_CASE_PLUS_MINUS_OPS(
                                                              XXMinusYYOp)
          .Default([&](auto) -> Operation* {
            throw std::runtime_error("Unsu"
                                     "ppor"
                                     "ted "
                                     "oper"
                                     "atio"
                                     "n");
          });

  return newOp;
}

/**
 * Removes all controls from a gate and returns an uncontrolled operation.
 *
 * @param op The qco::CtrlOp whose controls are removed.
 * @param rewriter The used rewriter
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return The operation without the controls.
 */
static UnitaryOpInterface
removeAllCtrlsOfGate(CtrlOp* op, PatternRewriter& rewriter,
                     std::span<Operation*>& worklist) {
  for (const auto& qubitCtrl : op->getInputQubits()) {
    rewriter.replaceAllUsesWith(op->getOutputForInput(qubitCtrl), qubitCtrl);
  }

  const auto innerUnitary =
      utils::getSoleBodyUnitary<UnitaryOpInterface>(*op->getBody());

  const auto targetInput = op->getInputTargets();
  std::vector<Value> qubitsIn = {targetInput.begin(), targetInput.end()};
  auto* const newOp =
      createOperationFromUnitaryOperation(innerUnitary, rewriter, qubitsIn);
  auto newUnitary = static_cast<UnitaryOpInterface>(newOp);
  for (const auto inTarget : newUnitary.getInputQubits()) {
    rewriter.replaceAllUsesExcept(
        inTarget, newUnitary.getOutputForInput(inTarget), newUnitary);
  }
  rewriter.eraseOp(*op);
  std::ranges::replace(worklist, *op, newOp);

  return newUnitary;
}

/**
 * Removes the given quantum controls from a CtrlOp, but not removing all
 * controls and only leaving the (formerly controlled) gate in the body.
 *
 * @param op The qco::CtrlOp whose controls are removed.
 * @param ctrlsToRemove The controls which should be removed from the CtrlOp.
 * @param rewriter The used rewriter
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return The operation without given controls.
 */
static CtrlOp removeCtrlsOfGate(CtrlOp* op,
                                const llvm::DenseSet<Value>& ctrlsToRemove,
                                PatternRewriter& rewriter,
                                std::span<Operation*>& worklist) {
  for (const auto& qubitCtrl : ctrlsToRemove) {
    rewriter.replaceAllUsesWith(op->getOutputForInput(qubitCtrl), qubitCtrl);
  }
  if (ctrlsToRemove.size() == op->getNumControls()) {
    throw std::runtime_error("Cannot remove all controls of a CtrlOp");
  }
  std::vector<Value> newControlIn;
  for (const auto& ctrls : op->getInputControls()) {
    if (!ctrlsToRemove.contains(ctrls)) {
      newControlIn.push_back(ctrls);
    }
  }
  const auto innerUnitary =
      utils::getSoleBodyUnitary<UnitaryOpInterface>(*op->getBody());
  CtrlOp newCtrl = CtrlOp::create(
      rewriter, op->getLoc(), newControlIn, op->getTargetsIn(),
      [&](const ValueRange target) {
        std::vector<Value> qubitsIn = {target.begin(), target.end()};
        auto* const newOp = createOperationFromUnitaryOperation(
            innerUnitary, rewriter, qubitsIn);
        return SmallVector<Value>{newOp->getResults()};
      });

  for (const auto inTarget : newCtrl.getInputQubits()) {
    rewriter.replaceAllUsesWith(op->getOutputForInput(inTarget),
                                newCtrl.getOutputForInput(inTarget));
  }
  for (const auto ctrlQubit : op->getOutputControls()) {
    rewriter.replaceAllUsesWith(ctrlQubit, op->getInputForOutput(ctrlQubit));
  }
  rewriter.eraseOp(*op);
  std::ranges::replace(worklist, *op, newCtrl);

  return newCtrl;
}

/**
 * Handles classical branching. Iterates through the body of the branching in a
 * new loop and removes the body from the current iteration.
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
static WalkResult handleIfOp(UnionTable* ut, IfOp* op,
                             const std::span<Value> posClassicalCtrls,
                             const std::span<Value> negClassicalCtrls,
                             PatternRewriter& rewriter,
                             std::span<Operation*>& worklist) {
  const Value condition = op->getCondition();

  // Remove branching if value is always or never true
  if (ut->isClassicalValueAlwaysTrue(condition)) {
    op->elseBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      std::ranges::replace(worklist, innerOp, static_cast<Operation*>(nullptr));
    });

    Block* block = &op->getThenRegion().front();
    inlineBranchBlock(op, block, worklist, rewriter);

    return WalkResult::advance();
  }
  if (ut->isClassicalValueAlwaysFalse(condition)) {
    op->thenBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      std::ranges::replace(worklist, innerOp, static_cast<Operation*>(nullptr));
    });

    Block* block = &op->getElseRegion().front();
    inlineBranchBlock(op, block, worklist, rewriter);

    return WalkResult::advance();
  }

  Block* thenBlock = op->thenBlock();
  Block* elseBlock = op->elseBlock();
  bool thenEmpty = thenBlock->getOperations().size() <= 1;
  bool elseEmpty = elseBlock->getOperations().size() <= 1;

  const auto targetQubits = op->getQubits();
  std::vector<Value> targets = {targetQubits.begin(), targetQubits.end()};
  std::vector<Value> thenArgs;
  std::vector<Value> elseArgs;

  // propagate through then and else block
  if (!thenEmpty) {
    for (const Value arg : thenBlock->getArguments()) {
      thenArgs.push_back(arg);
    }
    ut->replaceValuesGlobally(targets, thenArgs);
    std::vector<Operation*> newWorklist;

    // Create a new worklist to iterate over the inner instructions
    op->thenBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      newWorklist.push_back(innerOp);
      std::ranges::replace(worklist, innerOp, static_cast<Operation*>(nullptr));
    });
    std::span wl(newWorklist.data(), newWorklist.size());
    std::vector<Value> newPosClassicalCtrls = {posClassicalCtrls.begin(),
                                               posClassicalCtrls.end()};
    newPosClassicalCtrls.push_back(condition);
    const auto resThen = iterateThroughWorklist(
        rewriter, ut, wl, newPosClassicalCtrls, negClassicalCtrls);

    if (resThen.failed()) {
      return WalkResult::interrupt();
    }
    op->thenBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      const auto input = innerOp->getOperands();
      const auto output = innerOp->getResults();
      for (unsigned int i = 0; i < std::min(input.size(), output.size()); ++i) {
        std::ranges::replace(thenArgs, input[i], output[i]);
      }
    });
  }
  if (!elseEmpty) {
    for (const Value arg : elseBlock->getArguments()) {
      elseArgs.push_back(arg);
    }
    ut->replaceValuesGlobally(thenArgs.empty() ? targets : thenArgs, elseArgs);
    std::vector<Operation*> newWorklist;

    op->elseBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      newWorklist.push_back(innerOp);
      std::ranges::replace(worklist, innerOp, static_cast<Operation*>(nullptr));
    });
    std::span wl(newWorklist.data(), newWorklist.size());
    std::vector<Value> newNegClassicalCtrls = {negClassicalCtrls.begin(),
                                               negClassicalCtrls.end()};
    newNegClassicalCtrls.push_back(condition);
    const auto resElse = iterateThroughWorklist(
        rewriter, ut, wl, posClassicalCtrls, newNegClassicalCtrls);
    std::cout << "Iterated through else branch..." << std::endl;

    if (resElse.failed()) {
      return WalkResult::interrupt();
    }
    op->elseBlock()->walk<WalkOrder::PreOrder>([&](Operation* innerOp) {
      // Propagating values in order to assign the right values to the right
      // result values
      const auto input = innerOp->getOperands();
      const auto output = innerOp->getResults();
      for (unsigned int i = 0; i < std::min(input.size(), output.size()); ++i) {
        std::ranges::replace(elseArgs, input[i], output[i]);
      }
    });
    std::cout << "Walked through else branch..." << std::endl;
  }
  const auto resultQubits = op->getResults();
  std::vector<Value> results = {resultQubits.begin(), resultQubits.end()};

  thenBlock = op->thenBlock();
  elseBlock = op->elseBlock();
  thenEmpty = thenBlock->getOperations().size() <= 1;
  elseEmpty = elseBlock->getOperations().size() <= 1;

  // Remove if operation completely if both branches are empty after propagation
  if (thenEmpty && elseEmpty) {
    std::cout << "If and else empty" << std::endl;
    // Check that there is no implicit swap in one branch by re-ordered yield
    // operands and get order of returned qubits
    std::vector<unsigned int> order;
    bool implicitSwap = false;
    if (!thenArgs.empty()) {
      std::cout << "Then args not empty..." << std::endl;
      for (unsigned int i = 0; i < thenBlock->getArguments().size(); ++i) {
        auto it = std::ranges::find(thenArgs, thenBlock->getArguments()[i]);
        if (it != thenArgs.end()) {
          const unsigned int pos = std::distance(thenArgs.begin(), it);
          order.push_back(pos);
        }
      }
    }
    if (!elseArgs.empty()) {
      std::cout << "Else args not empty..." << std::endl;
      for (unsigned int i = 0; i < elseBlock->getArguments().size(); ++i) {
        auto it = std::ranges::find(elseArgs, elseBlock->getArguments()[i]);
        if (it != elseArgs.end()) {
          const unsigned int pos = std::distance(elseArgs.begin(), it);
          if (!thenArgs.empty()) {
            implicitSwap |= order.at(i) == pos;
          } else {
            order.push_back(pos);
          }
        }
      }
    }
    if (implicitSwap) {
      throw std::runtime_error("Constant propagation does not allow implicit "
                               "swapping of qubits in branching.");
    }
    std::cout << "Implicit swap checked..." << std::endl;
    // remove if Op and replace the values in the module and union table
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    for (unsigned int inputQubitIndex = 0;
         inputQubitIndex < op->getQubits().size(); ++inputQubitIndex) {
      rewriter.replaceAllUsesWith(op->getResults()[order.at(inputQubitIndex)],
                                  op->getQubits()[inputQubitIndex]);
    }

    std::cout << "Replaced input with output after if removal..." << std::endl;
    std::vector<Value> inputQubitVec = {op->getQubits().begin(),
                                        op->getQubits().end()};
    ut->replaceValuesGlobally(elseArgs.empty() ? thenArgs : elseArgs,
                              inputQubitVec);
    std::cout << "Replaced values globally..." << std::endl;
    rewriter.eraseOp(*op);
    std::cout << "Erased If Op..." << std::endl;
  } else {
    ut->replaceValuesGlobally(elseArgs.empty() ? thenArgs : elseArgs, results);
  }

  return WalkResult::advance();
}

/**
 * Puts the given operation into a classical branch and propagates the branch.
 * Only supports one new condition, i.e. only considers the first positive or
 * negative new controls.
 *
 * @param ut Union table which contains the current quantum state.
 * @param op The operation to be put in a classical branch.
 * @param posClassicalCtrls The positive classical controls considered in the
 * operation.
 * @param negClassicalCtrls The negative classical controls considered in the
 * operation.
 * @param ctrlsToMod The controls which need to be used in the branch.
 * @param rewriter The used rewriter.
 * @param worklist The worklist which contains the operations that are iterated
 * through.
 * @return Whether the propagation of the new operations in a branch succeeded.
 */
static WalkResult
putOperationIntoBranch(UnionTable* ut, UnitaryOpInterface op,
                       const std::span<Value> posClassicalCtrls,
                       const std::span<Value> negClassicalCtrls,
                       ControlsToModify ctrlsToMod, PatternRewriter& rewriter,
                       std::span<Operation*>& worklist) {
  const bool createThenBranch = !ctrlsToMod.classicalPosCtrlsToAdd.empty();
  const Value condition = createThenBranch
                              ? *ctrlsToMod.classicalPosCtrlsToAdd.begin()
                              : *ctrlsToMod.classicalNegCtrlsToAdd.begin();
  ValueRange insertedQubits = op.getInputQubits();
  const SmallVector locs(insertedQubits.size(), op->getLoc());
  auto newIfOp =
      IfOp::create(rewriter, op->getLoc(), condition, insertedQubits);

  IRMapping map;

  if (createThenBranch) {
    auto* thenBlock = rewriter.createBlock(&newIfOp.getThenRegion(), {},
                                           newIfOp->getResultTypes(), locs);
    rewriter.setInsertionPointToStart(thenBlock);
    for (auto [originalInput, ifArgs] :
         llvm::zip(insertedQubits, thenBlock->getArguments())) {
      map.map(originalInput, ifArgs);
    }
    auto* thenClone = rewriter.clone(*op.getOperation(), map);
    YieldOp::create(rewriter, op->getLoc(), thenClone->getResults());

    auto* elseBlock = rewriter.createBlock(&newIfOp.getElseRegion(), {},
                                           newIfOp->getResultTypes(), locs);
    YieldOp::create(rewriter, op->getLoc(), elseBlock->getArguments());
  } else {
    auto* elseBlock = rewriter.createBlock(&newIfOp.getElseRegion(), {},
                                           newIfOp->getResultTypes(), locs);
    rewriter.setInsertionPointToStart(elseBlock);
    for (auto [originalInput, ifArgs] :
         llvm::zip(insertedQubits, elseBlock->getArguments())) {
      map.map(originalInput, ifArgs);
    }
    auto* elseClone = rewriter.clone(*op.getOperation(), map);
    YieldOp::create(rewriter, op->getLoc(), elseClone->getResults());

    auto* thenBlock = rewriter.createBlock(&newIfOp.getThenRegion(), {},
                                           newIfOp->getResultTypes(), locs);
    YieldOp::create(rewriter, op->getLoc(), thenBlock->getArguments());
  }

  rewriter.replaceAllUsesWith(op.getOutputQubits(), newIfOp.getResults());
  rewriter.replaceOp(op.getOperation(), newIfOp.getResults());

  return handleIfOp(ut, &newIfOp, posClassicalCtrls, negClassicalCtrls,
                    rewriter, worklist);
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
static WalkResult handleUnitary(UnionTable* ut, UnitaryOpInterface* op,
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
static WalkResult
handleUncontrolledUnitary(UnionTable* ut, UnitaryOpInterface* op,
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
static WalkResult handleCtrlOp(UnionTable* ut, CtrlOp* op,
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
  if (!satisfiable) {
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    removeCtrlOperation(op, rewriter);
    return WalkResult::advance();
  }
  const auto superfluousCtrls = ut->getSuperfluousControls(
      inCtrlValues, posClassicalCtrls, negClassicalCtrls);
  if (superfluousCtrls.completelySuperfluous) {
    std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
    removeCtrlOperation(op, rewriter);
    return WalkResult::advance();
  }

  // Collect quantum values to remove and classical values to add
  ControlsToModify ctrlsToMod;
  ctrlsToMod.quantumCtrlsToRemove = superfluousCtrls.superfluousQubits;
  for (const auto superfluousQ : ctrlsToMod.quantumCtrlsToRemove) {
    std::erase(inCtrlValues, superfluousQ);
  }

  const auto ctrlCandidates = inCtrlValues;
  for (const auto qCtrl : ctrlCandidates) {
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
        ctrlsToMod.quantumCtrlsToRemove.insert(qCtrl);
        break;
      }
    }
  }

  if (!ctrlsToMod.quantumCtrlsToRemove.empty()) {
    if (ctrlsToMod.quantumCtrlsToRemove.size() == op->getNumControls()) {
      auto newOp = removeAllCtrlsOfGate(op, rewriter, worklist);
      if (!ctrlsToMod.classicalPosCtrlsToAdd.empty() ||
          !ctrlsToMod.classicalNegCtrlsToAdd.empty()) {
        return putOperationIntoBranch(ut, newOp, posClassicalCtrls,
                                      negClassicalCtrls, ctrlsToMod, rewriter,
                                      worklist);
      }
      return handleUncontrolledUnitary(ut, &newOp, posClassicalCtrls,
                                       negClassicalCtrls, rewriter, worklist);
    }
    *op = removeCtrlsOfGate(op, ctrlsToMod.quantumCtrlsToRemove, rewriter,
                            worklist);
    if (!ctrlsToMod.classicalPosCtrlsToAdd.empty() ||
        !ctrlsToMod.classicalNegCtrlsToAdd.empty()) {
      return putOperationIntoBranch(ut, *op, posClassicalCtrls,
                                    negClassicalCtrls, ctrlsToMod, rewriter,
                                    worklist);
    }
  }

  auto body = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op->getBody());
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
static LogicalResult
iterateThroughWorklist(PatternRewriter& rewriter, UnionTable* ut,
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
            .Case<ResetOp>([&](ResetOp op) {
              ut->propagateReset(op.getOperand(), op.getResult(),
                                 posClassicalCtrls, negClassicalCtrls);
              return WalkResult::advance();
            })
            .Case<MeasureOp>([&](const MeasureOp op) {
              ut->propagateMeasurement(op->getOperand(0), op->getResult(0),
                                       op->getResult(1), posClassicalCtrls,
                                       negClassicalCtrls);
              return WalkResult::advance();
            })
            .Case<AllocOp>([&](AllocOp op) {
              addedAtLeastOneQubit = true;
              ut->propagateQubitAlloc(op.getResult());
              return WalkResult::advance();
            })
            .Case<SinkOp>([&]([[maybe_unused]] SinkOp op) {
              return WalkResult::advance();
            })
            .Case<IfOp>([&](IfOp op) {
              return handleIfOp(ut, &op, posClassicalCtrls, negClassicalCtrls,
                                rewriter, worklist);
            })
            .Case<YieldOp>([&]([[maybe_unused]] YieldOp op) {
              return WalkResult::advance();
            })
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
            // arith dialect
            .Case<arith::ConstantOp>([&](const arith::ConstantOp op) {
              return handleConstant(ut, op, posClassicalCtrls,
                                    negClassicalCtrls);
            })
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
            .Default([ut, posClassicalCtrls, negClassicalCtrls](Operation* op) {
              if (llvm::isa<arith::ArithDialect>(op->getDialect())) {
                std::vector<Value> operands = {op->getOperands().begin(),
                                               op->getOperands().end()};
                std::vector<Value> results = {op->getResults().begin(),
                                              op->getResults().end()};
                ut->propagateClassicalOperation(op, operands, results,
                                                posClassicalCtrls,
                                                negClassicalCtrls);
                return WalkResult::advance();
              }

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
 * @param maxNonzeroAmplitudes The maximum number of non-zero amplitudes in the
 * tracted quantum states before reaching top.
 * @param maxHybridStates The maximum number of hybrid states which have a
 * non-zero probability.
 * @return Success if constant propagation has been applied successfully
 */
static LogicalResult applyCP(ModuleOp module, MLIRContext* ctx,
                             const size_t maxNonzeroAmplitudes,
                             const size_t maxHybridStates) {
  moveMeasurementsToFront(module, ctx);

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

  auto ut = UnionTable(maxNonzeroAmplitudes, maxHybridStates);

  std::span wl(worklist.data(), worklist.size());

  return iterateThroughWorklist(rewriter, &ut, wl, {}, {});
}

} // namespace mlir::qco

namespace {
/**
 * This pass applies constant propagation to a circuit. It assumes that all
 * states start in |0> and removes quantum instructions that are superfluous
 * when the current state is considered. It also replaces quantum resources by
 * classical resources.
 */
struct ConstantPropagation final
    : mlir::qco::impl::ConstantPropagationBase<ConstantPropagation> {
  using ConstantPropagationBase::ConstantPropagationBase;

protected:
  void runOnOperation() override {
    if (mlir::failed(mlir::qco::applyCP(getOperation(), &getContext(),
                                        maximumNonzeroAmplitudes,
                                        maximumHybridStates))) {
      signalPassFailure();
    }
  }
};
} // namespace
