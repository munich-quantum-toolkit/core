/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/MQT/Utils/Angles.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/GatePowering.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::mqt {

#define GEN_PASS_DEF_NORMALIZEGLOBALPHASES
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

namespace {

struct Add final {};
struct Negate final {};
struct Scale final {
  double factor;
};

using PhaseInstruction = std::variant<double, Value, Add, Negate, Scale>;

/// A postfix phase expression. Keeping modifier transformations symbolic avoids
/// repeatedly walking and moving an ever-growing SSA arithmetic chain through
/// nested modifiers. The expression is materialized exactly once at the scope
/// where the phase stops bubbling.
class PhaseExpression final {
public:
  explicit PhaseExpression(Value angle) {
    if (const auto constant = valueToConstantDouble(angle)) {
      instructions.emplace_back(normalizeAngle(*constant));
    } else {
      instructions.emplace_back(angle);
      leaves.push_back(angle);
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
      leaves.clear();
      return;
    }
    instructions.append(std::make_move_iterator(other.instructions.begin()),
                        std::make_move_iterator(other.instructions.end()));
    leaves.append(std::make_move_iterator(other.leaves.begin()),
                  std::make_move_iterator(other.leaves.end()));
    instructions.emplace_back(Add{});
  }

  void negate() {
    if (const auto constant = getConstant()) {
      instructions.front() = normalizeAngle(-*constant);
      return;
    }
    instructions.emplace_back(Negate{});
  }

  void scale(const double factor) {
    if (factor == 0.0) {
      instructions.clear();
      instructions.emplace_back(0.0);
      leaves.clear();
      return;
    }
    if (factor == 1.0) {
      return;
    }
    if (const auto constant = getConstant()) {
      instructions.front() = normalizeAngle(*constant * factor);
      return;
    }
    instructions.emplace_back(Scale{factor});
  }

  void forEachValue(const llvm::function_ref<void(Value)> callback) const {
    for (auto value : leaves) {
      callback(value);
    }
  }

  [[nodiscard]] Value materialize(RewriterBase& rewriter,
                                  const Location loc) const {
    SmallVector<Value, 4> stack;
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
        assert(stack.size() >= 2);
        auto rhs = stack.pop_back_val();
        auto lhs = stack.pop_back_val();
        stack.push_back(rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs));
        continue;
      }
      assert(!stack.empty());
      auto operand = stack.pop_back_val();
      if (std::holds_alternative<Negate>(instruction)) {
        stack.push_back(rewriter.createOrFold<arith::NegFOp>(loc, operand));
        continue;
      }
      const auto factor = std::get<Scale>(instruction).factor;
      auto factorValue = constantFromScalar(rewriter, loc, factor);
      stack.push_back(
          rewriter.createOrFold<arith::MulFOp>(loc, factorValue, operand));
    }
    assert(stack.size() == 1);
    Value result = stack.front();
    // Fold pure constant arith trees back to a single normalized angle so
    // merged exit phases stay within the GPhase verifier contract.
    if (const auto constant = valueToConstantDouble(result)) {
      return constantFromScalar(rewriter, loc, normalizeAngle(*constant));
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

  SmallVector<PhaseInstruction, 4> instructions;
  SmallVector<Value, 2> leaves;
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

} // namespace

/// Collect a pure, body-local dependency slice in topological order.
static bool collectHoistableSlice(Value value, Block& body,
                                  SmallPtrSetImpl<Operation*>& visiting,
                                  SmallPtrSetImpl<Operation*>& collected,
                                  SmallVectorImpl<Operation*>& ordered) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    return blockArg.getOwner() != &body;
  }

  auto* definingOp = value.getDefiningOp();
  if (definingOp == nullptr || definingOp->getBlock() != &body) {
    return true;
  }
  if (collected.contains(definingOp)) {
    return true;
  }
  if (!visiting.insert(definingOp).second || definingOp->getNumRegions() != 0 ||
      !isPure(definingOp) || !isSpeculatable(definingOp)) {
    return false;
  }
  for (auto operand : definingOp->getOperands()) {
    if (!collectHoistableSlice(operand, body, visiting, collected, ordered)) {
      return false;
    }
  }
  visiting.erase(definingOp);
  collected.insert(definingOp);
  ordered.push_back(definingOp);
  return true;
}

/// Make all dynamic leaves of @p expression available before @p modifier.
static bool hoistExpressionBefore(const PhaseExpression& expression,
                                  Block& body, Operation* modifier,
                                  RewriterBase& rewriter) {
  SmallPtrSet<Operation*, 8> visiting;
  SmallPtrSet<Operation*, 8> collected;
  SmallVector<Operation*, 8> ordered;
  bool hoistable = true;
  expression.forEachValue([&](Value value) {
    if (hoistable &&
        !collectHoistableSlice(value, body, visiting, collected, ordered)) {
      hoistable = false;
    }
  });
  if (hoistable) {
    for (auto* op : ordered) {
      rewriter.moveOpBefore(op, modifier);
    }
  }
  return hoistable;
}

namespace {

class GlobalPhaseNormalizer final {
public:
  explicit GlobalPhaseNormalizer(MLIRContext* context) : rewriter(context) {}

  void normalize(Region& region) { normalizeRegion(region); }

private:
  [[nodiscard]] std::optional<PhaseContribution>
  normalizeOperation(Operation* op) {
    if (auto inv = dyn_cast<qc::InvOp>(op)) {
      return factorInverse(inv);
    }
    if (auto inv = dyn_cast<qco::InvOp>(op)) {
      return factorInverse(inv);
    }
    if (auto pow = dyn_cast<qc::PowOp>(op)) {
      return factorPower(pow);
    }
    if (auto pow = dyn_cast<qco::PowOp>(op)) {
      return factorPower(pow);
    }
    if (auto ctrl = dyn_cast<qc::CtrlOp>(op)) {
      return factorControl(ctrl);
    }
    if (auto ctrl = dyn_cast<qco::CtrlOp>(op)) {
      return factorControl(ctrl);
    }
    for (auto& nested : op->getRegions()) {
      normalizeRegion(nested);
    }
    return std::nullopt;
  }

  template <typename InvOp>
  [[nodiscard]] std::optional<PhaseContribution> factorInverse(InvOp op) {
    auto phase = normalizeBlock(*op.getBody(), op);
    if (phase) {
      phase->expression.negate();
    }
    return phase;
  }

  template <typename PowOp>
  [[nodiscard]] std::optional<PhaseContribution> factorPower(PowOp op) {
    const auto exponent = op.getExponentValue();
    if (!exponent || !isIntegerExponent(*exponent)) {
      normalizeRegion(op->getRegion(0));
      return std::nullopt;
    }
    auto phase = normalizeBlock(*op.getBody(), op);
    if (phase) {
      phase->expression.scale(*exponent);
    }
    return phase;
  }

  [[nodiscard]] std::optional<PhaseContribution> factorControl(qc::CtrlOp op) {
    auto phase = normalizeBlock(*op.getBody(), op);
    if (!phase || op.getNumControls() == 0) {
      return phase;
    }
    if (phase->expression.isZero()) {
      return std::nullopt;
    }

    rewriter.setInsertionPoint(op);
    auto angle = phase->expression.materialize(rewriter, phase->loc);
    rewriter.setInsertionPointAfter(op);
    if (op.getNumControls() == 1) {
      qc::POp::create(rewriter, phase->loc, op.getControl(0), angle);
      return std::nullopt;
    }
    auto controls = op.getControls();
    qc::CtrlOp::create(rewriter, phase->loc, controls.drop_back(),
                       controls.back(), [&](Value target) {
                         qc::POp::create(rewriter, phase->loc, target, angle);
                       });
    return std::nullopt;
  }

  [[nodiscard]] std::optional<PhaseContribution> factorControl(qco::CtrlOp op) {
    auto phase = normalizeBlock(*op.getBody(), op);
    if (!phase || op.getNumControls() == 0) {
      return phase;
    }
    if (phase->expression.isZero()) {
      return std::nullopt;
    }

    rewriter.setInsertionPoint(op);
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
    return std::nullopt;
  }

  void normalizeRegion(Region& region) {
    for (auto& block : region) {
      static_cast<void>(normalizeBlock(block, nullptr));
    }
  }

  [[nodiscard]] std::optional<PhaseContribution>
  normalizeBlock(Block& block, Operation* extractionBoundary) {
    std::optional<PhaseContribution> aggregate;
    SmallVector<Operation*, 4> directPhases;
    bool hasNestedContribution = false;

    for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
      std::optional<PhaseContribution> phase;
      if (auto gphase = dyn_cast<qc::GPhaseOp>(&op)) {
        phase.emplace(PhaseDialect::QC, gphase.getLoc(),
                      PhaseExpression(gphase.getTheta()));
        directPhases.push_back(gphase);
      } else if (auto gphase = dyn_cast<qco::GPhaseOp>(&op)) {
        phase.emplace(PhaseDialect::QCO, gphase.getLoc(),
                      PhaseExpression(gphase.getTheta()));
        directPhases.push_back(gphase);
      } else {
        phase = normalizeOperation(&op);
        hasNestedContribution |= phase.has_value();
      }
      if (!phase) {
        continue;
      }
      if (aggregate) {
        aggregate->add(std::move(*phase));
      } else {
        aggregate = std::move(phase);
      }
    }

    if (!aggregate) {
      return std::nullopt;
    }
    if (extractionBoundary != nullptr &&
        hoistExpressionBefore(aggregate->expression, block, extractionBoundary,
                              rewriter)) {
      for (auto* phase : directPhases) {
        rewriter.eraseOp(phase);
      }
      return aggregate;
    }

    // Preserve already-normalized exit phases, including dynamic angles.
    if (extractionBoundary == nullptr && !hasNestedContribution &&
        directPhases.size() == 1 &&
        directPhases.front()->getNextNode() == block.getTerminator()) {
      auto angle = dyn_cast<qc::GPhaseOp>(directPhases.front())
                       ? cast<qc::GPhaseOp>(directPhases.front()).getTheta()
                       : cast<qco::GPhaseOp>(directPhases.front()).getTheta();
      const auto constant = valueToConstantDouble(angle);
      if (!constant ||
          (normalizeAngle(*constant) == *constant && *constant != 0.0)) {
        return std::nullopt;
      }
    }

    for (auto* phase : directPhases) {
      rewriter.eraseOp(phase);
    }
    if (aggregate->expression.isZero()) {
      return std::nullopt;
    }
    rewriter.setInsertionPoint(block.getTerminator());
    auto angle = aggregate->expression.materialize(rewriter, aggregate->loc);
    if (aggregate->dialect == PhaseDialect::QC) {
      qc::GPhaseOp::create(rewriter, aggregate->loc, angle);
    } else {
      qco::GPhaseOp::create(rewriter, aggregate->loc, angle);
    }
    return std::nullopt;
  }

  IRRewriter rewriter;
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
  GlobalPhaseNormalizer normalizer(moduleOp.getContext());
  normalizer.normalize(moduleOp.getRegion());
  return success();
}

} // namespace mlir::mqt
