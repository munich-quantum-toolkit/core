/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
// IWYU pragma: begin_keep (Passes.h.inc)
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
// IWYU pragma: end_keep

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <string>
#include <tuple>
#include <utility>

namespace mlir::qco {

/**
 * @brief Redirect a call to a specialized copy of its callee.
 *
 * @param callOp The call to redirect.
 * @param newCallee The specialization the call should target.
 * @param rewriter The rewriter driving the pattern application.
 */
static void updateSpecializedCall(func::CallOp callOp, func::FuncOp newCallee,
                                  PatternRewriter& rewriter) {
  rewriter.modifyOpInPlace(callOp,
                           [&] { callOp.setCallee(newCallee.getName()); });
}

/**
 * @brief Create a detached copy of a function under a new name.
 *
 * @details
 * The copy is not inserted into the module; the caller is responsible for
 * adding it to a symbol table.
 *
 * @param funcOp The function to copy.
 * @param newName The name of the copy.
 * @return The detached copy.
 */
static func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName) {
  auto newFunc = funcOp.clone();
  newFunc.setName(newName.str());
  return newFunc;
}

#define GEN_PASS_DEF_QUANTUMIPO
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/// Caches the specializations already created for a callee, so that call sites
/// sharing the same context reuse one copy instead of cloning it repeatedly.
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  SymbolTable& symbolTable;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  PreviousSpecializations& previousSpecializations;

  constexpr static const auto ANGLES_TO_SPECIALIZE =
      std::array<double, 5>{0.0, std::numbers::pi, std::numbers::pi / 2,
                            1.5 * std::numbers::pi, 2 * std::numbers::pi};

  /**
   * @brief Check whether an operation leaves a qubit in the |0> state alone.
   *
   * @param op The operation applied to the argument.
   * @param zeroArgument The argument known to be in the |0> state.
   * @return True if @p op has no effect given that state.
   */
  static bool operationIsNopOnZero(Operation* op, Value zeroArgument) {
    if (auto ctrl = dyn_cast<CtrlOp>(op)) {
      return llvm::is_contained(ctrl.getControlsIn(), zeroArgument);
    }
    return isa<ZOp>(op) || isa<SOp>(op) || isa<ResetOp>(op); // TODO more ops?
  }

  /**
   * @brief Check whether an operation leaves a qubit in the |+> state alone.
   *
   * @param op The operation applied to the argument.
   * @return True if @p op has no effect given that state.
   */
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

  /**
   * @brief Try to specialize the callee for what is known about one argument.
   *
   * @param callOp The call whose callee may be specialized.
   * @param operand The index of the argument to reason about.
   * @param rewriter The rewriter driving the pattern application.
   * @return True if a specialization was applied.
   */
  bool trySpecialize(func::CallOp callOp, unsigned operand,
                     PatternRewriter& rewriter) const {
    const auto argValue = callOp.getArgOperands()[operand];

    auto calleeName = callOp.getCallee();
    auto funcOp = symbolTable.lookup<func::FuncOp>(calleeName);

    if (!funcOp || funcOp.isExternal()) {
      return false;
    }

    auto* definingOp = argValue.getDefiningOp();

    if (definingOp == nullptr) {
      return false;
    }

    if (argValue.getType() == QubitType::get(rewriter.getContext())) {
      // CSS for qubit types.
      if (isa<AllocOp>(definingOp) || isa<ResetOp>(definingOp)) {
        return trySpecializeZero(callOp, funcOp, operand, rewriter);
      }
      if (isa<HOp>(definingOp)) {
        const auto* precedingOp = definingOp->getOperand(0).getDefiningOp();
        if (precedingOp != nullptr &&
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

  /**
   * @brief Specialize a callee for an argument known to be in the |0> state.
   *
   * @param callOp The call to redirect.
   * @param funcOp The current callee.
   * @param operand The index of the |0> argument.
   * @param rewriter The rewriter driving the pattern application.
   * @return True if a specialization was applied.
   */
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

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + "_spec_zero_arg_" +
                                 std::to_string(operand));
    symbolTable.insert(newFunc);
    previousSpecializations.zeroSpecializations.insert({key, newFunc});

    auto newParameter = newFunc.getArgument(operand);
    if (newParameter.hasOneUse() &&
        operationIsNopOnZero(*newParameter.getUsers().begin(), newParameter)) {
      auto* newUser = *newParameter.getUsers().begin();

      // A reset does not implement `UnitaryOpInterface`, so it has to be
      // forwarded explicitly instead of going through the qubit accessors.
      if (auto resetOp = dyn_cast<ResetOp>(newUser)) {
        rewriter.replaceAllUsesWith(resetOp.getQubitOut(),
                                    resetOp.getQubitIn());
        rewriter.eraseOp(resetOp);
      } else if (auto unitaryOp = dyn_cast<UnitaryOpInterface>(newUser)) {
        for (auto i = 0U; i < unitaryOp.getNumQubits(); ++i) {
          // TODO-DAMIAN use getOutputQubit/Input again (at current version,
          // this seems to use the output of the inner op)
          rewriter.replaceAllUsesWith(unitaryOp->getResult(i),
                                      unitaryOp->getOperand(i));
        }
        rewriter.eraseOp(unitaryOp);
      }
    }

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }

  /**
   * @brief Specialize a callee for an argument known to be in the |+> state.
   *
   * @param callOp The call to redirect.
   * @param funcOp The current callee.
   * @param operand The index of the |+> argument.
   * @param rewriter The rewriter driving the pattern application.
   * @return True if a specialization was applied.
   */
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

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + "_spec_plus_arg_" +
                                 std::to_string(operand));
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

  /**
   * @brief Specialize a callee for a rotation angle known at compile time.
   *
   * @details
   * Only a small set of distinguished angles is specialized, because every
   * specialization costs a copy of the callee. The parameter stays in the
   * signature; only its uses inside the copy are replaced by a constant.
   *
   * @param callOp The call to redirect.
   * @param funcOp The current callee.
   * @param angle The constant angle passed at the call site.
   * @param operand The index of the angle argument.
   * @param rewriter The rewriter driving the pattern application.
   * @return True if a specialization was applied.
   */
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

    auto newFunc = copyFunction(funcOp, funcOp.getName().str() + suffix);
    symbolTable.insert(newFunc);
    previousSpecializations.rotationSpecializations.insert({key, newFunc});

    auto newParameter = newFunc.getArgument(operand);
    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&*newFunc.getBody().getBlocks().begin());
    auto constant = arith::ConstantOp::create(
        rewriter, newFunc.getBody().getLoc(),
        rewriter.getFloatAttr(Float64Type::get(rewriter.getContext()), angle));
    rewriter.replaceAllUsesWith(newParameter, constant.getResult());

    updateSpecializedCall(callOp, newFunc, rewriter);
    return true;
  }
};

} // namespace

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

namespace {

/**
 * @brief This pass performs quantum inter-procedural optimizations (IPO).
 */
struct QuantumIPO final : impl::QuantumIPOBase<QuantumIPO> {
  using impl::QuantumIPOBase<QuantumIPO>::QuantumIPOBase;

protected:
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
      return;
    }

    runQuantumArgumentPromotion(op);
    runAuxiliaryQubitHoisting(op);
    runQuantumFunctionBoundaryCommutation(op, symbolTable);
  }
};

} // namespace

} // namespace mlir::qco
