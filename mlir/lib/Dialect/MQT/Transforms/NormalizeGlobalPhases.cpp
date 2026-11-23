/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::mqt {

#define GEN_PASS_DEF_NORMALIZEGLOBALPHASES
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct Add final {};
struct Negate final {};

// Keep the low bits until the sum has been reduced to the principal interval.
struct ConstantPhase {
  double angle;
  double error = 0.0;

  void add(ConstantPhase other) {
    if (other.angle == -other.error) {
      return;
    }
    const double sum = angle + other.angle;
    // TwoSum recovers the low bits lost when adding the two leading terms.
    const double rhs = sum - angle;
    const double residual = (angle - (sum - rhs)) + (other.angle - rhs);
    error += other.error + residual;
    angle = sum;
  }

  [[nodiscard]] bool isZero() const {
    if (angle == -error) {
      return true;
    }
    // Elide encoded whole turns only while their phase factor is still unity
    // within tolerance. A large multiple of binary64 2π can lose phase.
    return error == 0.0 && std::abs(angle) >= 2.0 * std::numbers::pi &&
           normalizeAngle(angle) == 0.0 &&
           classifyPhaseGate(angle) == PhaseGate::Identity;
  }

  [[nodiscard]] double reduced() const {
    const auto wrap = [](double radians) {
      if (std::abs(radians) > std::numbers::pi) {
        // Let libm reduce the argument using its extended-precision π.
        radians = 2.0 * std::atan(std::tan(radians / 2.0));
      }
      return radians == -std::numbers::pi ? std::numbers::pi : radians;
    };
    return wrap(wrap(angle) + error);
  }
};

using PhaseInstruction = std::variant<Value, Add, Negate>;

/// A constant offset and a postfix expression for the symbolic terms.
///
/// Keeping modifier transformations symbolic avoids repeatedly walking and
/// moving an ever-growing SSA arithmetic chain through nested modifiers. The
/// expression is materialized exactly once at the scope where the phase stops
/// bubbling.
class PhaseExpression final {
public:
  PhaseExpression(Value angle, std::optional<double> constant)
      : constant_{constant.value_or(0.0)} {
    if (!constant) {
      instructions_.emplace_back(angle);
    }
  }

  [[nodiscard]] bool isZero() const {
    return instructions_.empty() && constant_.isZero();
  }

  [[nodiscard]] bool needsNormalization() const {
    return instructions_.empty() &&
           (constant_.angle == 0.0 || constant_.angle <= -std::numbers::pi ||
            constant_.angle > std::numbers::pi);
  }

  void add(PhaseExpression& other) {
    constant_.add(other.constant_);
    if (other.instructions_.empty()) {
      return;
    }
    if (instructions_.empty()) {
      instructions_ = std::move(other.instructions_);
      return;
    }
    instructions_.append(other.instructions_.begin(),
                         other.instructions_.end());
    instructions_.emplace_back(Add{});
  }

  void negate() {
    constant_ = {-constant_.angle, -constant_.error};
    if (!instructions_.empty()) {
      instructions_.emplace_back(Negate{});
    }
  }

  [[nodiscard]] Value materialize(RewriterBase& rewriter,
                                  const Location loc) const {
    if (instructions_.empty()) {
      return constantFromScalar(rewriter, loc, constant_.reduced());
    }
    SmallVector<Value> stack;
    bool hasSum = false;
    for (const auto& instruction : instructions_) {
      if (const auto* value = std::get_if<Value>(&instruction)) {
        stack.push_back(*value);
        continue;
      }
      if (std::holds_alternative<Add>(instruction)) {
        hasSum = true;
        assert(stack.size() >= 2);
        auto rhs = stack.pop_back_val();
        auto lhs = stack.pop_back_val();
        stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs));
        continue;
      }
      assert(!stack.empty());
      auto operand = stack.pop_back_val();
      stack.push_back(rewriter.createOrFold<arith::NegFOp>(loc, operand));
    }
    assert(stack.size() == 1);
    Value result = stack.front();
    const bool hasOffset = !constant_.isZero();
    if (hasSum || hasOffset) {
      // Reduce the symbolic subtotal before adding a possibly tiny offset.
      rewriter.getContext()->getOrLoadDialect<math::MathDialect>();
      const FloatExpression angle(rewriter, loc, result);
      const auto two = FloatExpression::constant(rewriter, loc, 2.0);
      result = (two * (angle / two).tan().atan()).getValue();
    }
    if (hasOffset) {
      auto offset = constantFromScalar(rewriter, loc, constant_.reduced());
      result = rewriter.createOrFold<arith::AddFOp>(loc, result, offset);
    }
    return result;
  }

private:
  ConstantPhase constant_;
  SmallVector<PhaseInstruction> instructions_;
};

enum class PhaseDialect : std::uint8_t { QC, QCO };

struct PhaseContribution final {
  PhaseDialect dialect;
  Location loc;
  PhaseExpression expression;

  void add(PhaseContribution& other) {
    assert(dialect == other.dialect &&
           "QC and QCO operations cannot occur in the same program");
    expression.add(other.expression);
  }
};

class GlobalPhaseNormalizer final {
public:
  explicit GlobalPhaseNormalizer(RewriterBase& rewriter) : rewriter(rewriter) {}

  void normalize(Region& region) { normalizeRegion(region); }

private:
  [[nodiscard]] std::optional<double> getConstantAngle(Value angle) {
    if (const auto constant = valueToDouble(angle)) {
      return constant;
    }
    return attributeToDouble(valueToConstantAttr(angle, constants_));
  }

  [[nodiscard]] std::optional<PhaseContribution>
  normalizeOperation(Operation* op, Operation* hoistBefore) {
    if (hoistBefore == nullptr) {
      hoistBefore = op;
    }
    if (auto inv = dyn_cast<qc::InvOp>(op)) {
      return factorInverse(inv, hoistBefore);
    }
    if (auto inv = dyn_cast<qco::InvOp>(op)) {
      return factorInverse(inv, hoistBefore);
    }
    // Phase rounding inside a power can be amplified by an arbitrary exponent.
    if (isa<qc::PowOp, qco::PowOp>(op)) {
      return std::nullopt;
    }
    if (auto ctrl = dyn_cast<qc::CtrlOp>(op)) {
      return factorControl(ctrl, hoistBefore);
    }
    if (auto ctrl = dyn_cast<qco::CtrlOp>(op)) {
      return factorControl(ctrl, hoistBefore);
    }
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      GlobalPhaseNormalizer isolated(rewriter);
      for (auto& nested : op->getRegions()) {
        isolated.normalizeRegion(nested);
      }
    } else {
      for (auto& nested : op->getRegions()) {
        normalizeRegion(nested);
      }
    }
    return std::nullopt;
  }

  template <typename InvOp>
  [[nodiscard]] std::optional<PhaseContribution>
  factorInverse(InvOp op, Operation* hoistBefore) {
    auto phase = normalizeBlock(*op.getBody(), hoistBefore);
    if (phase) {
      phase->expression.negate();
      if (llvm::hasSingleElement(*op.getBody())) {
        inlineModifierBody(op, *op.getBody(), op->getOperands(), rewriter);
      }
    }
    return phase;
  }

  [[nodiscard]] std::optional<PhaseContribution>
  factorControl(qc::CtrlOp op, Operation* hoistBefore) {
    auto phase = normalizeBlock(*op.getBody(), hoistBefore);
    const bool releasePhase = phase.has_value() && op.getNumControls() == 0;
    if (phase && !releasePhase && !phase->expression.isZero()) {
      rewriter.setInsertionPoint(hoistBefore);
      auto angle = phase->expression.materialize(rewriter, phase->loc);
      rewriter.setInsertionPointAfter(op);
      if (op.getNumControls() == 1) {
        qc::POp::create(rewriter, phase->loc, op.getControl(0), angle);
      } else {
        auto controls = op.getControls();
        qc::CtrlOp::create(rewriter, phase->loc, controls.drop_back(),
                           controls.back(), [&](Value target) {
                             qc::POp::create(rewriter, phase->loc, target,
                                             angle);
                           });
      }
    }
    if (llvm::hasSingleElement(*op.getBody())) {
      rewriter.eraseOp(op);
    }
    return releasePhase ? std::move(phase) : std::nullopt;
  }

  [[nodiscard]] std::optional<PhaseContribution>
  factorControl(qco::CtrlOp op, Operation* hoistBefore) {
    auto phase = normalizeBlock(*op.getBody(), hoistBefore);
    const bool releasePhase = phase.has_value() && op.getNumControls() == 0;
    if (phase && !releasePhase && !phase->expression.isZero()) {
      rewriter.setInsertionPoint(hoistBefore);
      auto angle = phase->expression.materialize(rewriter, phase->loc);
      rewriter.setInsertionPointAfter(op);
      SmallVector<Value> oldControls(op.getOutputControls());
      SmallVector<Value> newControls;
      Operation* relativePhase = nullptr;
      if (op.getNumControls() == 1) {
        auto p =
            qco::POp::create(rewriter, phase->loc, oldControls.front(), angle);
        newControls.push_back(p.getOutputTarget(0));
        relativePhase = p;
      } else {
        auto relative = qco::CtrlOp::create(
            rewriter, phase->loc, ValueRange(oldControls).drop_back(),
            oldControls.back(), [&](Value target) {
              return qco::POp::create(rewriter, phase->loc, target, angle)
                  .getOutputTarget(0);
            });
        llvm::append_range(newControls, relative.getOutputQubits());
        relativePhase = relative;
      }

      for (auto [oldControl, newControl] :
           llvm::zip_equal(oldControls, newControls)) {
        rewriter.replaceAllUsesExcept(oldControl, newControl, relativePhase);
      }
    }
    if (llvm::hasSingleElement(*op.getBody())) {
      SmallVector<Value> replacements(op.getInputControls());
      for (auto yielded : op.getBody()->getTerminator()->getOperands()) {
        replacements.push_back(
            getValueFromBlockArgument(yielded, op.getInputTargets()));
      }
      rewriter.replaceOp(op, replacements);
    }
    return releasePhase ? std::move(phase) : std::nullopt;
  }

  void normalizeRegion(Region& region) {
    for (auto& block : region) {
      static_cast<void>(normalizeBlock(block, nullptr));
    }
  }

  [[nodiscard]] std::optional<PhaseContribution>
  normalizeBlock(Block& block, Operation* extractionBoundary) {
    SmallVector<PhaseContribution> contributions;
    size_t phaseCount = 0;
    const auto mergeLastPair = [&] {
      contributions[contributions.size() - 2].add(contributions.back());
      contributions.pop_back();
    };
    SmallVector<Operation*> directPhases;
    auto insertionPoint = block.end();

    for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
      auto* current = &op;
      auto next = std::next(op.getIterator());
      const bool atInsertionPoint = insertionPoint == op.getIterator();
      if (extractionBoundary != nullptr &&
          !isa<qc::UnitaryOpInterface, qco::UnitaryOpInterface>(op)) {
        // Hoist once, in source order, across the full eager modifier nest.
        rewriter.moveOpBefore(&op, extractionBoundary);
        continue;
      }
      std::optional<PhaseContribution> phase;
      if (isa<qc::GPhaseOp, qco::GPhaseOp>(op)) {
        auto angle = op.getOperand(0);
        const auto constant = getConstantAngle(angle);
        // Leave invalid expressions for dead-code removal or diagnostics.
        if (constant && !isValidGlobalPhaseAngle(*constant)) {
          continue;
        }
        phase.emplace(isa<qc::GPhaseOp>(op) ? PhaseDialect::QC
                                            : PhaseDialect::QCO,
                      op.getLoc(), PhaseExpression(angle, constant));
        directPhases.push_back(&op);
      } else if (op.getNumRegions() != 0) {
        phase = normalizeOperation(&op, extractionBoundary);
      }
      // A following modifier may disappear or emit a relative phase.
      if (atInsertionPoint &&
          (next == block.begin() || &*std::prev(next) != current)) {
        insertionPoint = next;
      }
      if (!phase) {
        continue;
      }
      insertionPoint = next;
      contributions.push_back(std::move(*phase));
      // Merge adjacent groups of equal size. Each set bit in the count
      // represents one remaining subtotal.
      for (size_t n = ++phaseCount; n % 2 == 0; n /= 2) {
        mergeLastPair();
      }
    }

    if (contributions.empty()) {
      return std::nullopt;
    }
    while (contributions.size() > 1) {
      mergeLastPair();
    }
    auto aggregate = std::move(contributions.front());
    if (extractionBoundary != nullptr) {
      for (auto* phase : directPhases) {
        rewriter.eraseOp(phase);
      }
      return aggregate;
    }

    // Preserve a single normalized phase, including dynamic angles.
    if (phaseCount == 1 && directPhases.size() == 1 &&
        !aggregate.expression.needsNormalization()) {
      return std::nullopt;
    }

    for (auto* phase : directPhases) {
      rewriter.eraseOp(phase);
    }
    if (aggregate.expression.isZero()) {
      return std::nullopt;
    }
    rewriter.setInsertionPoint(&block, insertionPoint);
    auto angle = aggregate.expression.materialize(rewriter, aggregate.loc);
    if (aggregate.dialect == PhaseDialect::QC) {
      qc::GPhaseOp::create(rewriter, aggregate.loc, angle);
    } else {
      qco::GPhaseOp::create(rewriter, aggregate.loc, angle);
    }
    return std::nullopt;
  }

  RewriterBase& rewriter;
  // Scalar values retain their meaning; each isolated scope owns its cache.
  DenseMap<Value, Attribute> constants_;
};

struct NormalizeGlobalPhases final
    : impl::NormalizeGlobalPhasesBase<NormalizeGlobalPhases> {
  using NormalizeGlobalPhasesBase::NormalizeGlobalPhasesBase;

protected:
  void runOnOperation() override {
    if (failed(normalizeGlobalPhases(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult normalizeGlobalPhases(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  return normalizeGlobalPhases(moduleOp, rewriter);
}

LogicalResult normalizeGlobalPhases(ModuleOp moduleOp, RewriterBase& rewriter) {
  GlobalPhaseNormalizer normalizer(rewriter);
  normalizer.normalize(moduleOp.getRegion());
  return success();
}

} // namespace mlir::mqt
