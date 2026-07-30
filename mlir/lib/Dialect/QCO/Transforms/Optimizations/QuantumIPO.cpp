/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

//
// Created by damian on 1/21/26.
//

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mlir::qco {

namespace {
void updateSpecializedCall(func::CallOp callOp, func::FuncOp newCallee,
                           PatternRewriter& rewriter) {
  rewriter.modifyOpInPlace(callOp,
                           [&] { callOp.setCallee(newCallee.getName()); });
}

func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName,
                          PatternRewriter& rewriter) {
  const OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(funcOp);

  auto newFunc = funcOp.clone();

  rewriter.modifyOpInPlace(newFunc, [&] { newFunc.setName(newName.str()); });

  return newFunc;
}

} // namespace

#define GEN_PASS_DEF_QUANTUMIPO
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

struct PreviousSpecializations {
  std::map<std::pair<std::string, uint32_t>, func::FuncOp> zeroSpecializations;
  std::map<std::pair<std::string, uint32_t>, func::FuncOp> plusSpecializations;
  std::map<std::tuple<std::string, uint32_t, double>, func::FuncOp>
      rotationSpecializations;
};

/**
 * @brief This pattern attempts to perform context-sensitive specialization.
 */
struct ContextSensitiveSpecializationPattern final
    : OpRewritePattern<func::CallOp> {

  SymbolTable& symbolTable;
  PreviousSpecializations& previousSpecializations;

  constexpr static const auto ANGLES_TO_SPECIALIZE =
      std::array<double, 5>{0.0, M_PI, M_PI_2, M_PI_2 + M_PI, 2 * M_PI};

  static bool operationIsNopOnZero(Operation* op, Value zeroArgument) {
    if (auto ctrl = dyn_cast<CtrlOp>(op)) {
      return std::find(ctrl.getControlsIn().begin(), ctrl.getControlsIn().end(),
                       zeroArgument) != ctrl.getControlsIn().end();
    }
    return isa<ZOp>(op) || isa<SOp>(op) || isa<ResetOp>(op); // TODO more ops?
  }

  static bool operationIsNopOnPlus(Operation* op) { return isa<XOp>(op); }

  explicit ContextSensitiveSpecializationPattern(MLIRContext* context,
                                                 SymbolTable& symbolTable,
                                                 PreviousSpecializations& prev)
      : OpRewritePattern(context), symbolTable(symbolTable),
        previousSpecializations(prev) {}

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter& rewriter) const override {
    auto found = false;
    for (auto i = 0U; i < callOp.getArgOperands().size(); ++i) {
      if (trySpecialize(callOp, i, rewriter)) {
        found = true;
      }
    }
    return LogicalResult::success(found);
  }

  bool trySpecialize(func::CallOp callOp, unsigned operand,
                     PatternRewriter& rewriter) const {
    const auto argValue = callOp.getArgOperands()[operand];

    auto calleeName = callOp.getCallee();
    auto funcOp = symbolTable.lookup<func::FuncOp>(calleeName);

    if (!funcOp || funcOp.isExternal()) {
      return false;
    }

    auto* definingOp = argValue.getDefiningOp();

    if (!definingOp) {
      return false;
    }

    if (argValue.getType() == QubitType::get(rewriter.getContext())) {
      // CSS for qubit types.
      if (isa<AllocOp>(definingOp) || isa<ResetOp>(definingOp)) {
        return trySpecializeZero(callOp, funcOp, operand, rewriter);
      }
      if (isa<HOp>(definingOp)) {
        const auto* precedingOp = definingOp->getOperand(0).getDefiningOp();
        if (precedingOp &&
            (isa<AllocOp>(precedingOp) || isa<ResetOp>(precedingOp))) {
          return trySpecializePlus(callOp, funcOp, operand, rewriter);
        }
      }
    }
    if (argValue.getType() == Float64Type::get(rewriter.getContext())) {
      // CSS for double types.
      if (isa<arith::ConstantOp>(definingOp)) {
        auto constOp = cast<arith::ConstantOp>(definingOp);
        return trySpecializeRotationArguments(
            callOp, funcOp,
            cast<FloatAttr>(constOp.getValue()).getValueAsDouble(), operand,
            rewriter);
      }
    }

    return false;
  }

  bool trySpecializeZero(func::CallOp callOp, func::FuncOp funcOp,
                         unsigned operand, PatternRewriter& rewriter) const {
    auto parameter = funcOp.getArgument(operand);
    if (!parameter.hasOneUse()) {
      return false;
    }
    if (!operationIsNopOnZero(*parameter.getUsers().begin(), parameter)) {
      return false;
    }

    auto key = std::make_pair(funcOp.getName().str(), operand);
    if (previousSpecializations.zeroSpecializations.contains(key)) {
      updateSpecializedCall(callOp,
                            previousSpecializations.zeroSpecializations.at(key),
                            rewriter);
      return true;
    }

    auto newFunc = copyFunction(funcOp,
                                funcOp.getName().str() + "_spec_zero_arg_" +
                                    std::to_string(operand),
                                rewriter);
    symbolTable.insert(newFunc);
    previousSpecializations.zeroSpecializations.insert({key, newFunc});

    auto newParameter = newFunc.getArgument(operand);
    while (
        newParameter.hasOneUse() &&
        operationIsNopOnZero(*newParameter.getUsers().begin(), newParameter)) {
      auto newUser =
          dyn_cast<UnitaryOpInterface>(*newParameter.getUsers().begin());
      for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
        // TODO-DAMIAN use getOutputQubit/Input again (at current version, this
        // seems to use the output of the inner op)
        rewriter.replaceAllUsesWith(newUser->getResult(i),
                                    newUser->getOperand(i));
      }
      rewriter.eraseOp(newUser);
      break;
    }

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }

  bool trySpecializePlus(func::CallOp callOp, func::FuncOp funcOp,
                         unsigned operand, PatternRewriter& rewriter) const {
    auto parameter = funcOp.getArgument(operand);
    if (!parameter.hasOneUse()) {
      return false;
    }
    if (!operationIsNopOnPlus(*parameter.getUsers().begin())) {
      return false;
    }

    auto key = std::make_pair(funcOp.getName().str(), operand);
    if (previousSpecializations.plusSpecializations.contains(key)) {
      updateSpecializedCall(callOp,
                            previousSpecializations.plusSpecializations.at(key),
                            rewriter);
      return true;
    }

    auto newFunc = copyFunction(funcOp,
                                funcOp.getName().str() + "_spec_plus_arg_" +
                                    std::to_string(operand),
                                rewriter);
    symbolTable.insert(newFunc);
    previousSpecializations.plusSpecializations.insert({key, newFunc});

    auto newParameter = newFunc.getArgument(operand);
    while (newParameter.hasOneUse() &&
           operationIsNopOnPlus(*newParameter.getUsers().begin())) {
      auto newUser =
          dyn_cast<UnitaryOpInterface>(*newParameter.getUsers().begin());
      for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
        rewriter.replaceAllUsesWith(newUser.getOutputQubit(i),
                                    newUser.getInputQubit(i));
      }
      rewriter.eraseOp(newUser);
    }

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }

  bool trySpecializeRotationArguments(func::CallOp callOp, func::FuncOp funcOp,
                                      double angle, unsigned operand,
                                      PatternRewriter& rewriter) const {
    if (std::ranges::none_of(ANGLES_TO_SPECIALIZE, [angle](double a) {
          return std::abs(a - angle) < 1e-9;
        })) {
      return false;
    }

    const std::string suffix = "_spec_fixed_angle_" + std::to_string(operand);
    if (funcOp.getName().contains(suffix)) {
      // Already specialized
      return false;
    }

    auto key = std::make_tuple(funcOp.getName().str(), operand, angle);
    if (previousSpecializations.rotationSpecializations.contains(key)) {
      updateSpecializedCall(
          callOp, previousSpecializations.rotationSpecializations.at(key),
          rewriter);
      return true;
    }

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + suffix, rewriter);
    symbolTable.insert(newFunc);
    previousSpecializations.rotationSpecializations.insert({key, newFunc});

    auto newParameter = newFunc.getArgument(operand);
    rewriter.setInsertionPointToStart(&*newFunc.getBody().getBlocks().begin());
    auto constant = rewriter.create<arith::ConstantOp>(
        newFunc.getBody().getLoc(),
        rewriter.getFloatAttr(Float64Type::get(rewriter.getContext()), angle));
    rewriter.replaceAllUsesWith(newParameter, constant.getResult());

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }
};

/**
 * @brief Populates the given pattern set with the different IPO patterns.
 *
 * @param patterns The pattern set to populate.
 */
static void
populateQuantumIPOPatterns(RewritePatternSet& patterns,
                           SymbolTable& symbolTable,
                           PreviousSpecializations& previousSpecializations) {
  patterns.add<ContextSensitiveSpecializationPattern>(
      patterns.getContext(), symbolTable, previousSpecializations);
}

/**
 * @brief This pass performs quantum inter-procedural optimizations (IPO).
 */
struct QuantumIPO final : impl::QuantumIPOBase<QuantumIPO> {
  using impl::QuantumIPOBase<QuantumIPO>::QuantumIPOBase;

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();
    SymbolTable symbolTable(op);

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    PreviousSpecializations previousSpecializations;
    populateQuantumIPOPatterns(patterns, symbolTable, previousSpecializations);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }

    runQuantumArgumentPromotion(op, symbolTable);
    runAncillaHoisting(op, symbolTable);
    runQuantumFunctionBoundaryCommutation(op, symbolTable);
    runQuantumFunctionBoundaryCommutation(op, symbolTable);
  }
};

} // namespace mlir::qco
