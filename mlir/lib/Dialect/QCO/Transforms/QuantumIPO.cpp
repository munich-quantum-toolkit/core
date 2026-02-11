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
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <algorithm>
#include <array>
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
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
void updateSpecializedCall(mlir::func::CallOp callOp,
                           mlir::func::FuncOp newCallee,
                           mlir::PatternRewriter& rewriter) {
  rewriter.modifyOpInPlace(callOp,
                           [&] { callOp.setCallee(newCallee.getName()); });
}

mlir::func::FuncOp copyFunction(mlir::func::FuncOp funcOp,
                                mlir::StringRef newName,
                                mlir::PatternRewriter& rewriter) {
  const mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(funcOp);

  auto newFunc = funcOp.clone();

  rewriter.modifyOpInPlace(newFunc, [&] { newFunc.setName(newName.str()); });

  return newFunc;
}

} // end anonymous namespace

namespace mlir::qco {

#define GEN_PASS_DEF_QUANTUMIPO
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief This pattern attempts to perform context-sensitive specialization.
 */
struct ContextSensitiveSpecializationPattern final
    : mlir::OpRewritePattern<func::CallOp> {

  SymbolTable& symbolTable;

  constexpr static const auto ANGLES_TO_SPECIALIZE =
      std::array<double, 5>{0.0, M_PI, M_PI_2, M_PI_2 + M_PI, 2 * M_PI};

  static bool operationIsNopOnZero(mlir::Operation* op,
                                   mlir::Value zeroArgument) {
    if (auto ctrl = mlir::dyn_cast<qco::CtrlOp>(op)) {
      return std::find(ctrl.getControlsIn().begin(), ctrl.getControlsIn().end(),
                       zeroArgument) != ctrl.getControlsIn().end();
    }
    return mlir::isa<qco::ZOp>(op) || mlir::isa<qco::SOp>(op) ||
           mlir::isa<qco::ResetOp>(op); // TODO more ops?
  }

  static bool operationIsNopOnPlus(mlir::Operation* op) {
    return mlir::isa<qco::XOp>(op);
  }

  explicit ContextSensitiveSpecializationPattern(mlir::MLIRContext* context,
                                                 SymbolTable& symbolTable)
      : OpRewritePattern(context), symbolTable(symbolTable) {}

  mlir::LogicalResult
  matchAndRewrite(func::CallOp callOp,
                  mlir::PatternRewriter& rewriter) const override {
    auto found = false;
    for (auto i = 0U; i < callOp.getArgOperands().size(); ++i) {
      if (trySpecialize(callOp, i, rewriter)) {
        found = true;
      }
    }
    return mlir::LogicalResult::success(found);
  }

  bool trySpecialize(func::CallOp callOp, unsigned operand,
                     mlir::PatternRewriter& rewriter) const {
    const auto argValue = callOp.getArgOperands()[operand];

    auto calleeName = callOp.getCallee();
    auto funcOp = symbolTable.lookup<func::FuncOp>(calleeName);

    if (!funcOp || funcOp.isExternal()) {
      return false;
    }

    auto* definingOp = argValue.getDefiningOp();
    if (argValue.getType() == qco::QubitType::get(rewriter.getContext())) {
      // CSS for qubit types.
      if (mlir::isa<qco::AllocOp>(definingOp) ||
          mlir::isa<qco::ResetOp>(definingOp)) {
        return trySpecializeZero(callOp, funcOp, operand, rewriter);
      }
      if (mlir::isa<qco::HOp>(definingOp)) {
        const auto* precedingOp = definingOp->getOperand(0).getDefiningOp();
        if (precedingOp && (mlir::isa<qco::AllocOp>(precedingOp) ||
                            mlir::isa<qco::ResetOp>(precedingOp))) {
          return trySpecializePlus(callOp, funcOp, operand, rewriter);
        }
      }
    }
    if (argValue.getType() == mlir::Float64Type::get(rewriter.getContext())) {
      // CSS for double types.
      if (mlir::isa<arith::ConstantOp>(definingOp)) {
        auto constOp = mlir::cast<arith::ConstantOp>(definingOp);
        return trySpecializeRotationArguments(
            callOp, funcOp,
            mlir::cast<mlir::FloatAttr>(constOp.getValue()).getValueAsDouble(),
            operand, rewriter);
        // TODO we still need a canonicalization for this(?)
      }
    }

    return false;
  }

  bool trySpecializeZero(func::CallOp callOp, func::FuncOp funcOp,
                         unsigned operand,
                         mlir::PatternRewriter& rewriter) const {
    auto parameter = funcOp.getArgument(operand);
    if (!parameter.hasOneUse()) {
      return false;
    }
    if (!operationIsNopOnZero(*parameter.getUsers().begin(), parameter)) {
      return false;
    }

    auto newFunc = copyFunction(funcOp,
                                funcOp.getName().str() + "_spec_zero_arg_" +
                                    std::to_string(operand),
                                rewriter);
    symbolTable.insert(newFunc);

    auto newParameter = newFunc.getArgument(operand);
    while (newParameter.hasOneUse() &&
           operationIsNopOnZero(*newParameter.getUsers().begin(), parameter)) {
      auto newUser = mlir::dyn_cast<qco::UnitaryOpInterface>(
          *newParameter.getUsers().begin());
      for (auto i = 0U; i < newUser.getNumQubits(); ++i) {
        rewriter.replaceAllUsesWith(newUser.getOutputQubit(i),
                                    newUser.getInputQubit(i));
      }
      rewriter.eraseOp(newUser);
    }

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }

  bool trySpecializePlus(func::CallOp callOp, func::FuncOp funcOp,
                         unsigned operand,
                         mlir::PatternRewriter& rewriter) const {
    auto parameter = funcOp.getArgument(operand);
    if (!parameter.hasOneUse()) {
      return false;
    }
    if (!operationIsNopOnPlus(*parameter.getUsers().begin())) {
      return false;
    }

    auto newFunc = copyFunction(funcOp,
                                funcOp.getName().str() + "_spec_plus_arg_" +
                                    std::to_string(operand),
                                rewriter);
    symbolTable.insert(newFunc);

    auto newParameter = newFunc.getArgument(operand);
    while (newParameter.hasOneUse() &&
           operationIsNopOnPlus(*newParameter.getUsers().begin())) {
      auto newUser = mlir::dyn_cast<qco::UnitaryOpInterface>(
          *newParameter.getUsers().begin());
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
                                      mlir::PatternRewriter& rewriter) const {
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

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + suffix, rewriter);
    symbolTable.insert(newFunc);

    auto newParameter = newFunc.getArgument(operand);
    rewriter.setInsertionPointToStart(&*newFunc.getBody().getBlocks().begin());
    auto constant = rewriter.create<mlir::arith::ConstantOp>(
        newFunc.getBody().getLoc(),
        rewriter.getFloatAttr(mlir::Float64Type::get(rewriter.getContext()),
                              angle));
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
static void populateQuantumIPOPatterns(mlir::RewritePatternSet& patterns,
                                       SymbolTable& symbolTable) {
  patterns.add<ContextSensitiveSpecializationPattern>(patterns.getContext(),
                                                      symbolTable);
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
    mlir::RewritePatternSet patterns(ctx);
    populateQuantumIPOPatterns(patterns, symbolTable);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }

    runQuantumArgumentPromotion(op, symbolTable);
    runAncillaHoisting(op, symbolTable);
  }
};

} // namespace mlir::qco
