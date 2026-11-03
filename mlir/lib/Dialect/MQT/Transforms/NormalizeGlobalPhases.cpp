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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::mqt {

#define GEN_PASS_DEF_NORMALIZEGLOBALPHASES
#include "mqt/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct Add final {};
struct Negate final {};

using PhaseInstruction = std::variant<double, Value, Add, Negate>;

/// A postfix phase expression.
///
/// Keeping modifier transformations symbolic avoids repeatedly walking and
/// moving an ever-growing SSA arithmetic chain through nested modifiers. The
/// expression is materialized exactly once at the scope where the phase stops
/// bubbling.
class PhaseExpression final {
public:
  PhaseExpression(Value angle, std::optional<double> constant) {
    if (constant) {
      instructions.emplace_back(normalizeAngle(*constant));
    } else {
      instructions.emplace_back(angle);
    }
  }

  [[nodiscard]] bool isZero() const {
    const auto constant = getConstant();
    return constant && *constant == 0.0;
  }

  void add(PhaseExpression&& other) {
    if (isZero()) {
      *this = std::move(other);
      return;
    }
    if (other.isZero()) {
      return;
    }
    const auto lhs = getConstant();
    const auto rhs = other.getConstant();
    if (lhs && rhs) {
      instructions.clear();
      instructions.emplace_back(normalizeAngle(*lhs + *rhs));
      return;
    }
    instructions.append(std::make_move_iterator(other.instructions.begin()),
                        std::make_move_iterator(other.instructions.end()));
    instructions.emplace_back(Add{});
  }

  void negate() {
    if (const auto constant = getConstant()) {
      instructions.front() = normalizeAngle(-*constant);
      return;
    }
    instructions.emplace_back(Negate{});
  }

  [[nodiscard]] Value materialize(RewriterBase& rewriter,
                                  const Location loc) const {
    SmallVector<Value> stack;
    bool hasSum = false;
    for (const auto& instruction : instructions) {
      if (const auto* constant = std::get_if<double>(&instruction)) {
        stack.push_back(constantFromScalar(rewriter, loc, *constant));
        continue;
      }
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
    if (hasSum) {
      // Keep dynamic sums within the phase range using exporter-supported ops.
      rewriter.getContext()->getOrLoadDialect<math::MathDialect>();
      const FloatExpression angle(rewriter, loc, result);
      const auto two = FloatExpression::constant(rewriter, loc, 2.0);
      return (two * (angle / two).tan().atan()).getValue();
    }
    return result;
  }

private:
  [[nodiscard]] std::optional<double> getConstant() const {
    if (instructions.size() == 1) {
      if (const auto* constant = std::get_if<double>(&instructions.front())) {
        return *constant;
      }
    }
    return std::nullopt;
  }

  SmallVector<PhaseInstruction> instructions;
};

enum class PhaseDialect : std::uint8_t { QC, QCO };

struct PhaseContribution final {
  PhaseDialect dialect;
  Location loc;
  PhaseExpression expression;

  void add(PhaseContribution other) {
    assert(dialect == other.dialect &&
           "QC and QCO operations cannot occur in the same program");
    expression.add(std::move(other.expression));
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
    SmallVector<Operation*> directPhases;
    Operation* insertionBefore = nullptr;

    for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
      auto* current = &op;
      auto* next = op.getNextNode();
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
      if (insertionBefore == current && next->getPrevNode() != current) {
        insertionBefore = next;
      }
      if (!phase) {
        continue;
      }
      insertionBefore = next;
      contributions.push_back(std::move(*phase));
    }

    if (contributions.empty()) {
      return std::nullopt;
    }
    // Keep accumulated phase depth logarithmic for symbolic exporters.
    for (size_t stride = 1; stride < contributions.size(); stride *= 2) {
      for (size_t i = 0; i + stride < contributions.size(); i += 2 * stride) {
        contributions[i].add(std::move(contributions[i + stride]));
      }
    }
    auto aggregate = std::move(contributions.front());
    if (extractionBoundary != nullptr) {
      for (auto* phase : directPhases) {
        rewriter.eraseOp(phase);
      }
      return aggregate;
    }

    // Preserve a single normalized phase, including dynamic angles.
    if (contributions.size() == 1 && directPhases.size() == 1) {
      const auto constant =
          getConstantAngle(directPhases.front()->getOperand(0));
      if (!constant ||
          (normalizeAngle(*constant) == *constant && *constant != 0.0)) {
        return std::nullopt;
      }
    }

    for (auto* phase : directPhases) {
      rewriter.eraseOp(phase);
    }
    if (aggregate.expression.isZero()) {
      return std::nullopt;
    }
    rewriter.setInsertionPoint(insertionBefore);
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
