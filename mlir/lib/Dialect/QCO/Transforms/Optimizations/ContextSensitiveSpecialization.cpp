/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "IPOUtils.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h" // IWYU pragma: keep (Passes.h.inc)

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
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
#include <numbers>
#include <string>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_CONTEXTSENSITIVESPECIALIZATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

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

namespace {

/// One cached rotation specialization: the parameter it was created for, the
/// angle baked into it, and the resulting copy of the callee.
struct RotationSpecialization {
  uint32_t operand;
  double angle;
  func::FuncOp func;
};

/// Caches the specializations already created for a callee, so that call sites
/// sharing the same context reuse one copy instead of cloning it repeatedly.
///
/// All three are keyed by callee name first, which lets lookups take a
/// `StringRef` without building a temporary `std::string`.
struct PreviousSpecializations {
  /// Keyed by callee name, then by the specialized parameter index.
  llvm::StringMap<DenseMap<uint32_t, func::FuncOp>> zeroSpecializations;
  llvm::StringMap<DenseMap<uint32_t, func::FuncOp>> plusSpecializations;
  /// Keyed by callee name; angles are compared with the same tolerance the
  /// pass uses elsewhere rather than for exact equality.
  llvm::StringMap<SmallVector<RotationSpecialization>> rotationSpecializations;

  /// Every callee a call was redirected away from, and every specialization
  /// created. These are exactly the functions the pass may have left without
  /// callers, so they are the only ones it is entitled to clean up.
  SmallVector<func::FuncOp> touchedFunctions;
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

  /// Tolerance used when comparing rotation angles.
  constexpr static double ANGLE_TOLERANCE = 1e-9;

  /// Discardable attribute recording which rotation specializations a function
  /// already represents. Each entry carries the specialized operand index and
  /// the angle that was baked into the body, which mirrors the identity used by
  /// `PreviousSpecializations::rotationSpecializations`.
  constexpr static llvm::StringLiteral ROTATION_SPECIALIZATION_ATTR =
      "qco.rotation_specializations";
  /// Key of the operand index inside one specialization entry.
  constexpr static llvm::StringLiteral ROTATION_SPECIALIZATION_OPERAND =
      "operand";
  /// Key of the baked-in angle inside one specialization entry.
  constexpr static llvm::StringLiteral ROTATION_SPECIALIZATION_ANGLE = "angle";

  /**
   * @brief Compare two rotation angles up to the specialization tolerance.
   *
   * @param lhs The first angle.
   * @param rhs The second angle.
   * @return True if the angles are considered equal.
   */
  static bool anglesAreEqual(double lhs, double rhs) {
    return std::abs(lhs - rhs) < ANGLE_TOLERANCE;
  }

  /**
   * @brief Check whether a function already is the rotation specialization for
   * the given operand and angle.
   *
   * @details
   * Uses a discardable attribute rather than the function name, so that a
   * specialization for one operand or angle does not block specialization for
   * another, and so that user-chosen names cannot be mistaken for markers.
   *
   * @param funcOp The candidate callee.
   * @param operand The index of the angle argument.
   * @param angle The angle passed at the call site.
   * @return True if @p funcOp already bakes in that angle for that operand.
   */
  static bool hasRotationSpecialization(func::FuncOp funcOp, unsigned operand,
                                        double angle) {
    const auto marker =
        funcOp->getAttrOfType<ArrayAttr>(ROTATION_SPECIALIZATION_ATTR);
    if (!marker) {
      return false;
    }
    return llvm::any_of(
        marker.getAsRange<DictionaryAttr>(), [&](DictionaryAttr entry) {
          const auto operandAttr =
              entry.getAs<IntegerAttr>(ROTATION_SPECIALIZATION_OPERAND);
          const auto angleAttr =
              entry.getAs<FloatAttr>(ROTATION_SPECIALIZATION_ANGLE);
          return operandAttr && angleAttr && operandAttr.getInt() == operand &&
                 anglesAreEqual(angleAttr.getValueAsDouble(), angle);
        });
  }

  /**
   * @brief Record on a cloned function which rotation specialization it is.
   *
   * @details
   * Existing markers are kept, so a function specialized for several operands
   * accumulates one entry per operand.
   *
   * @param funcOp The clone to mark.
   * @param operand The index of the angle argument.
   * @param angle The angle baked into the body.
   * @param builder Builder used to create the attributes.
   */
  static void markRotationSpecialization(func::FuncOp funcOp, unsigned operand,
                                         double angle, OpBuilder& builder) {
    SmallVector<Attribute> entries;
    if (const auto existing =
            funcOp->getAttrOfType<ArrayAttr>(ROTATION_SPECIALIZATION_ATTR)) {
      entries.assign(existing.begin(), existing.end());
    }
    entries.emplace_back(builder.getDictionaryAttr(
        {builder.getNamedAttr(
             ROTATION_SPECIALIZATION_OPERAND,
             builder.getI32IntegerAttr(static_cast<int32_t>(operand))),
         builder.getNamedAttr(ROTATION_SPECIALIZATION_ANGLE,
                              builder.getF64FloatAttr(angle))}));
    funcOp->setAttr(ROTATION_SPECIALIZATION_ATTR,
                    builder.getArrayAttr(entries));
  }

  /**
   * @brief Check whether an operation leaves a qubit in the |0> state alone.
   *
   * @details
   * Only operations that fix |0> *exactly* qualify. That covers a controlled
   * operation whose control is the |0> qubit, a reset, and the diagonal gates
   * whose first diagonal entry is one: `id`, `z`, `s`, `sdg`, `t`, `tdg`, and
   * `p` for any angle, since `p = diag(1, exp(i * theta))`.
   *
   * `rz` is deliberately excluded: `rz = diag(exp(-i * theta / 2),
   * exp(i * theta / 2))` maps |0> to a phase multiple of itself, so removing it
   * would silently change the global phase of the program. Admitting it would
   * require emitting a compensating `qco.gphase`.
   *
   * @param op The operation applied to the argument.
   * @param zeroArgument The argument known to be in the |0> state.
   * @return True if @p op has no effect given that state.
   */
  static bool operationIsNopOnZero(Operation* op, Value zeroArgument) {
    if (auto ctrl = dyn_cast<CtrlOp>(op)) {
      return llvm::is_contained(ctrl.getControlsIn(), zeroArgument);
    }
    return isa<IdOp, ZOp, SOp, SdgOp, TOp, TdgOp, POp, ResetOp>(op);
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

    const auto calleeName = funcOp.getName();
    if (const auto it =
            previousSpecializations.zeroSpecializations.find(calleeName);
        it != previousSpecializations.zeroSpecializations.end()) {
      if (const auto cached = it->second.find(operand);
          cached != it->second.end()) {
        previousSpecializations.touchedFunctions.emplace_back(funcOp);
        updateSpecializedCall(callOp, cached->second, rewriter);
        return true;
      }
    }

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + "_spec_zero_arg_" +
                                 std::to_string(operand));
    symbolTable.insert(newFunc);
    previousSpecializations.zeroSpecializations[calleeName][operand] = newFunc;
    previousSpecializations.touchedFunctions.emplace_back(funcOp);
    previousSpecializations.touchedFunctions.emplace_back(newFunc);

    // Drop the whole run of operations that leave the |0> state alone, not just
    // the first one. Every iteration erases an operation, so this terminates.
    auto newParameter = newFunc.getArgument(operand);
    while (
        newParameter.hasOneUse() &&
        operationIsNopOnZero(*newParameter.getUsers().begin(), newParameter)) {
      auto* newUser = *newParameter.getUsers().begin();

      // A reset does not implement `UnitaryOpInterface`, so it has to be
      // forwarded explicitly instead of going through the qubit accessors.
      if (auto resetOp = dyn_cast<ResetOp>(newUser)) {
        rewriter.replaceAllUsesWith(resetOp.getQubitOut(),
                                    resetOp.getQubitIn());
        rewriter.eraseOp(resetOp);
        continue;
      }

      auto unitaryOp = dyn_cast<UnitaryOpInterface>(newUser);
      if (!unitaryOp) {
        break;
      }
      // Use the qubit accessors rather than raw operand and result indices.
      // The gates currently accepted above happen to list their qubits first,
      // but the modifier operations do not: `qco.pow` takes its exponent as
      // operand 0, so raw indexing would forward that exponent into a qubit
      // result. The accessors skip parameters wherever they sit.
      for (auto i = 0U; i < unitaryOp.getNumQubits(); ++i) {
        rewriter.replaceAllUsesWith(unitaryOp.getOutputQubit(i),
                                    unitaryOp.getInputQubit(i));
      }
      rewriter.eraseOp(unitaryOp);
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

    const auto calleeName = funcOp.getName();
    if (const auto it =
            previousSpecializations.plusSpecializations.find(calleeName);
        it != previousSpecializations.plusSpecializations.end()) {
      if (const auto cached = it->second.find(operand);
          cached != it->second.end()) {
        previousSpecializations.touchedFunctions.emplace_back(funcOp);
        updateSpecializedCall(callOp, cached->second, rewriter);
        return true;
      }
    }

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + "_spec_plus_arg_" +
                                 std::to_string(operand));
    symbolTable.insert(newFunc);
    previousSpecializations.plusSpecializations[calleeName][operand] = newFunc;
    previousSpecializations.touchedFunctions.emplace_back(funcOp);
    previousSpecializations.touchedFunctions.emplace_back(newFunc);

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
          return anglesAreEqual(a, angle);
        })) {
      return false;
    }

    if (funcOp.getArgument(operand).use_empty()) {
      // Nothing inside the callee reads the angle, so a specialization would be
      // an exact copy of what it was cloned from.
      return false;
    }

    if (hasRotationSpecialization(funcOp, operand, angle)) {
      // This callee already is the specialization for that operand and angle,
      // so specializing it again would clone it forever.
      return false;
    }

    const auto calleeName = funcOp.getName();
    if (const auto it =
            previousSpecializations.rotationSpecializations.find(calleeName);
        it != previousSpecializations.rotationSpecializations.end()) {
      const auto* const cached =
          llvm::find_if(it->second, [&](const RotationSpecialization& entry) {
            return entry.operand == operand &&
                   anglesAreEqual(entry.angle, angle);
          });
      if (cached != it->second.end()) {
        previousSpecializations.touchedFunctions.emplace_back(funcOp);
        updateSpecializedCall(callOp, cached->func, rewriter);
        return true;
      }
    }

    auto newFunc =
        copyFunction(funcOp, funcOp.getName().str() + "_spec_fixed_angle_" +
                                 std::to_string(operand));
    markRotationSpecialization(newFunc, operand, angle, rewriter);
    symbolTable.insert(newFunc);
    previousSpecializations.touchedFunctions.emplace_back(funcOp);
    previousSpecializations.touchedFunctions.emplace_back(newFunc);
    previousSpecializations.rotationSpecializations[calleeName].emplace_back(
        RotationSpecialization{
            .operand = operand, .angle = angle, .func = newFunc});

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
 * @brief Populate the pattern set with the specialization pattern.
 *
 * @param patterns The pattern set to populate.
 * @param symbolTable The symbol table specializations are inserted into.
 * @param previousSpecializations Cache of already-created specializations.
 */
static void populateSpecializationPatterns(
    RewritePatternSet& patterns, SymbolTable& symbolTable,
    PreviousSpecializations& previousSpecializations) {
  patterns.add<ContextSensitiveSpecializationPattern>(
      patterns.getContext(), symbolTable, previousSpecializations);
}

namespace {
/// Specializes callees for what is known at their call sites.
struct ContextSensitiveSpecialization final
    : impl::ContextSensitiveSpecializationBase<ContextSensitiveSpecialization> {
  using impl::ContextSensitiveSpecializationBase<
      ContextSensitiveSpecialization>::ContextSensitiveSpecializationBase;

protected:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    RewritePatternSet patterns(moduleOp.getContext());
    PreviousSpecializations previousSpecializations;
    populateSpecializationPatterns(patterns, symbolTable,
                                   previousSpecializations);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      llvm::reportFatalInternalError(
          "failed to apply context-sensitive specialization patterns");
    }

    // Drop the callees this pass left without callers.
    eraseOrphanedSpecializations(symbolTable,
                                 previousSpecializations.touchedFunctions);
  }
};
} // namespace

} // namespace mlir::qco
