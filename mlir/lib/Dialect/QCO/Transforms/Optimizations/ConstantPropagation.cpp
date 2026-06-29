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

namespace mlir::qco {

#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

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

void removeOperation(UnitaryOpInterface* op, PatternRewriter& rewriter) {
  for (const auto outQubit : op->getOutputQubits()) {
    rewriter.replaceAllUsesWith(outQubit, op->getInputForOutput(outQubit));
  }
  rewriter.eraseOp(*op);
}

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
    const bool v = boolAttr.getValue();
    if (v) {
      ut->propagateIntAlloc(res, 1);
    } else {
      ut->propagateIntAlloc(res, 0);
    }
  }
  return WalkResult::advance();
}

WalkResult handleUnitary(UnionTable* ut, UnitaryOpInterface* op,
                         const std::span<Value> ctrlsQuantum,
                         const std::span<Value> newCtrlsQuantum,
                         const std::span<Value> posClassicalCtrls,
                         const std::span<Value> negClassicalCtrls,
                         PatternRewriter& rewriter,
                         std::span<Operation*>& worklist) {
  if (isa<IdOp>(op) || isa<ZOp>(op) || isa<SOp>(op) || isa<SdgOp>(op) ||
      isa<TOp>(op) || isa<TdgOp>(op)) {
    const auto addedGlobalPhase =
        ut->globalPhaseThatIsAdded(*op, op->getInputQubit(0), ctrlsQuantum,
                                   posClassicalCtrls, negClassicalCtrls);
    if (addedGlobalPhase.has_value()) {
      std::ranges::replace(worklist, *op, static_cast<Operation*>(nullptr));
      const auto phase = addedGlobalPhase.value();
      if (std::norm(phase) > 1e-4) {
        GPhaseOp::create(rewriter, op->getLoc(), phase);
      }
      removeOperation(op, rewriter);
    }
  }

  const auto targets = op->getInputTargets();
  const auto results = op->getOutputTargets();
  const auto params = op->getParameters();
  std::vector<Value> targetValues = {targets.begin(), targets.end()};
  std::vector<Value> resultValues = {results.begin(), results.end()};
  std::vector<Value> paramValues = {params.begin(), params.end()};
  ut->propagateGate(*op, targetValues, resultValues, ctrlsQuantum,
                    newCtrlsQuantum, posClassicalCtrls, negClassicalCtrls,
                    paramValues);

  return WalkResult::advance();
}

WalkResult handleCtrlOp(UnionTable* ut, CtrlOp* op,
                        const std::span<Value> posClassicalCtrls,
                        const std::span<Value> negClassicalCtrls,
                        PatternRewriter& rewriter,
                        std::span<Operation*>& worklist) {
  const auto inputCtrs = op->getInputControls();
  std::vector<Value> inCtrlValues = {inputCtrs.begin(), inputCtrs.end()};
  // TODO: Check if gate is executable
  // TODO: Work on the right qubits in body

  const auto outputCtrls = op->getOutputControls();
  std::vector<Value> outCtrlValues = {outputCtrls.begin(), outputCtrls.end()};

  auto body = op->getBodyUnitary();
  return handleUnitary(ut, &body, inCtrlValues, outCtrlValues,
                       posClassicalCtrls, negClassicalCtrls, rewriter,
                       worklist);
}

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
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
              return handleUnitary(ut, &op, {}, {}, posClassicalCtrls,
                                   negClassicalCtrls, rewriter, worklist);
            })
            //         .Case<ResetOp>([&](const ResetOp op) {
            //           return handleReset(ut, op, posClassicalCtrls,
            //           negClassicalCtrls);
            //         })
            //         .Case<MeasureOp>([&](const MeasureOp op) {
            //           return handleMeasure(ut, op, posClassicalCtrls,
            //                                negClassicalCtrls);
            //         })
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
 * @brief This pass applies constant propagation to a circuit. It assumes that
 * all states start in |0> and removes quantum instructions that are superfluous
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
