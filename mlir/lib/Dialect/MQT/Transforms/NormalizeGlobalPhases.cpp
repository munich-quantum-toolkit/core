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

#include <llvm/ADT/DenseMap.h>
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

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
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

  [[nodiscard]] bool isConstant() const { return getConstant().has_value(); }

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
      instructions.front() = scaleAngleByInteger(*constant, factor);
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
        stack.push_back(normalizeAngle(rewriter, loc, *value));
        continue;
      }
      if (std::holds_alternative<Add>(instruction)) {
        assert(stack.size() >= 2);
        auto rhs = stack.pop_back_val();
        auto lhs = stack.pop_back_val();
        auto sum = rewriter.createOrFold<arith::AddFOp>(loc, lhs, rhs);
        stack.push_back(normalizeAngle(rewriter, loc, sum));
        continue;
      }
      assert(!stack.empty());
      auto operand = stack.pop_back_val();
      if (std::holds_alternative<Negate>(instruction)) {
        auto negated = rewriter.createOrFold<arith::NegFOp>(loc, operand);
        stack.push_back(normalizeAngle(rewriter, loc, negated));
        continue;
      }
      const auto factor = std::get<Scale>(instruction).factor;
      auto factorValue = constantFromScalar(rewriter, loc, factor);
      auto scaled =
          rewriter.createOrFold<arith::MulFOp>(loc, factorValue, operand);
      stack.push_back(normalizeAngle(rewriter, loc, scaled));
    }
    assert(stack.size() == 1);
    Value result = stack.front();
    // Fold pure constant arith trees back to a single normalized angle so
    // merged exit phases stay within the GPhase verifier contract.
    if (const auto constant = valueToConstantDouble(result)) {
      return constantFromScalar(rewriter, loc, normalizeAngle(*constant));
    }
    return normalizeAngle(rewriter, loc, result);
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
};

using PhaseContributions = std::array<std::optional<PhaseContribution>, 2>;

} // namespace

[[nodiscard]] static constexpr std::size_t
getDialectIndex(PhaseDialect dialect) {
  return static_cast<std::size_t>(dialect);
}

static void addContribution(PhaseContributions& contributions,
                            PhaseContribution contribution) {
  auto& aggregate = contributions[getDialectIndex(contribution.dialect)];
  if (aggregate) {
    aggregate->expression.add(std::move(contribution.expression));
    return;
  }
  aggregate = std::move(contribution);
}

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

  void normalize(Region& root) { normalizeRegion(root, nullptr); }

private:
  void normalizeRegion(Region& region, Operation* extractionBoundary) {
    for (auto& block : region) {
      for (auto& op : block) {
        auto* nestedBoundary = getExtractionBoundary(&op);
        for (auto& nested : op.getRegions()) {
          normalizeRegion(nested, nestedBoundary);
        }
      }
      auto contributions = normalizeBlock(block, extractionBoundary);
      if (extractionBoundary != nullptr) {
        applyExtractionBoundary(extractionBoundary, std::move(contributions));
      }
    }
  }

  [[nodiscard]] static Operation* getExtractionBoundary(Operation* op) {
    if (isa<qc::InvOp, qco::InvOp, qc::CtrlOp, qco::CtrlOp>(op)) {
      return op;
    }
    if (auto pow = dyn_cast<qc::PowOp>(op)) {
      const auto exponent = pow.getExponentValue();
      return exponent && isIntegerExponent(*exponent) ? op : nullptr;
    }
    if (auto pow = dyn_cast<qco::PowOp>(op)) {
      const auto exponent = pow.getExponentValue();
      return exponent && isIntegerExponent(*exponent) ? op : nullptr;
    }
    return nullptr;
  }

  static bool canExtract(Operation* boundary, PhaseDialect dialect) {
    if (isa<qc::CtrlOp>(boundary)) {
      return dialect == PhaseDialect::QC;
    }
    if (isa<qco::CtrlOp>(boundary)) {
      return dialect == PhaseDialect::QCO;
    }
    return true;
  }

  static bool canExtractExpression(Operation* boundary,
                                   const PhaseExpression& expression) {
    std::optional<double> exponent;
    if (auto pow = dyn_cast<qc::PowOp>(boundary)) {
      exponent = pow.getExponentValue();
    } else if (auto pow = dyn_cast<qco::PowOp>(boundary)) {
      exponent = pow.getExponentValue();
    }
    return !exponent || expression.isConstant() || std::abs(*exponent) <= 1.0;
  }

  void factorControl(qc::CtrlOp op, PhaseContribution phase) {
    if (phase.expression.isZero()) {
      return;
    }
    rewriter.setInsertionPoint(op);
    auto angle = phase.expression.materialize(rewriter, phase.loc);
    rewriter.setInsertionPointAfter(op);
    if (op.getNumControls() == 1) {
      qc::POp::create(rewriter, phase.loc, op.getControl(0), angle);
      return;
    }
    auto controls = op.getControls();
    qc::CtrlOp::create(rewriter, phase.loc, controls.drop_back(),
                       controls.back(), [&](Value target) {
                         qc::POp::create(rewriter, phase.loc, target, angle);
                       });
  }

  void factorControl(qco::CtrlOp op, PhaseContribution phase) {
    if (phase.expression.isZero()) {
      return;
    }
    rewriter.setInsertionPoint(op);
    auto angle = phase.expression.materialize(rewriter, phase.loc);
    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> oldControls(op.getOutputControls());
    SmallVector<Value> newControls;
    Operation* relativePhase = nullptr;
    if (op.getNumControls() == 1) {
      auto p =
          qco::POp::create(rewriter, phase.loc, oldControls.front(), angle);
      newControls.push_back(p.getOutputTarget(0));
      relativePhase = p;
    } else {
      auto relative = qco::CtrlOp::create(
          rewriter, phase.loc, ValueRange(oldControls).drop_back(),
          oldControls.back(), [&](Value target) {
            return qco::POp::create(rewriter, phase.loc, target, angle)
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

  void recordContributions(Operation* op, PhaseContributions contributions) {
    auto& recorded = contributionsByOperation[op];
    for (auto& contribution : contributions) {
      if (contribution) {
        addContribution(recorded, std::move(*contribution));
      }
    }
  }

  void applyExtractionBoundary(Operation* op,
                               PhaseContributions contributions) {
    if (isa<qc::InvOp, qco::InvOp>(op)) {
      for (auto& contribution : contributions) {
        if (contribution) {
          contribution->expression.negate();
        }
      }
      recordContributions(op, std::move(contributions));
      return;
    }

    std::optional<double> exponent;
    if (auto pow = dyn_cast<qc::PowOp>(op)) {
      exponent = pow.getExponentValue();
    } else if (auto pow = dyn_cast<qco::PowOp>(op)) {
      exponent = pow.getExponentValue();
    }
    if (exponent) {
      for (auto& contribution : contributions) {
        if (contribution) {
          contribution->expression.scale(*exponent);
        }
      }
      recordContributions(op, std::move(contributions));
      return;
    }

    if (auto ctrl = dyn_cast<qc::CtrlOp>(op)) {
      if (ctrl.getNumControls() == 0) {
        recordContributions(op, std::move(contributions));
        return;
      }
      auto& phase = contributions[getDialectIndex(PhaseDialect::QC)];
      if (phase) {
        factorControl(ctrl, std::move(*phase));
        phase.reset();
      }
    } else if (auto ctrl = dyn_cast<qco::CtrlOp>(op)) {
      if (ctrl.getNumControls() == 0) {
        recordContributions(op, std::move(contributions));
        return;
      }
      auto& phase = contributions[getDialectIndex(PhaseDialect::QCO)];
      if (phase) {
        factorControl(ctrl, std::move(*phase));
        phase.reset();
      }
    }
    recordContributions(op, std::move(contributions));
  }

  [[nodiscard]] static bool isAtBlockExit(Operation* phase,
                                          Operation* terminator) {
    for (auto* next = phase->getNextNode(); next != terminator;
         next = next->getNextNode()) {
      if (next == nullptr || !isa<qc::GPhaseOp, qco::GPhaseOp>(next)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] PhaseContributions
  normalizeBlock(Block& block, Operation* extractionBoundary) {
    PhaseContributions aggregates;
    PhaseContributions extracted;
    std::array<SmallVector<Operation*, 4>, 2> directPhases;
    std::array<bool, 2> hasNestedContribution{};
    Operation* terminator =
        block.mightHaveTerminator() ? block.getTerminator() : nullptr;

    for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
      if (auto gphase = dyn_cast<qc::GPhaseOp>(&op)) {
        addContribution(aggregates, {PhaseDialect::QC, gphase.getLoc(),
                                     PhaseExpression(gphase.getTheta())});
        directPhases[getDialectIndex(PhaseDialect::QC)].push_back(gphase);
      } else if (auto gphase = dyn_cast<qco::GPhaseOp>(&op)) {
        addContribution(aggregates, {PhaseDialect::QCO, gphase.getLoc(),
                                     PhaseExpression(gphase.getTheta())});
        directPhases[getDialectIndex(PhaseDialect::QCO)].push_back(gphase);
      } else if (auto it = contributionsByOperation.find(&op);
                 it != contributionsByOperation.end()) {
        for (std::size_t i = 0; i < it->second.size(); ++i) {
          auto& contribution = it->second[i];
          if (contribution) {
            hasNestedContribution[i] = true;
            addContribution(aggregates, std::move(*contribution));
          }
        }
        contributionsByOperation.erase(it);
      } else {
        continue;
      }
    }

    for (std::size_t i = 0; i < aggregates.size(); ++i) {
      auto& aggregate = aggregates[i];
      if (!aggregate) {
        continue;
      }
      if (extractionBoundary != nullptr &&
          canExtract(extractionBoundary, aggregate->dialect) &&
          canExtractExpression(extractionBoundary, aggregate->expression) &&
          hoistExpressionBefore(aggregate->expression, block,
                                extractionBoundary, rewriter)) {
        for (auto* phase : directPhases[i]) {
          rewriter.eraseOp(phase);
        }
        extracted[i] = std::move(aggregate);
        continue;
      }

      // Preserve already-normalized exit phases, including dynamic angles.
      if (extractionBoundary == nullptr && !hasNestedContribution[i] &&
          directPhases[i].size() == 1 &&
          isAtBlockExit(directPhases[i].front(), terminator)) {
        auto* phase = directPhases[i].front();
        auto angle = dyn_cast<qc::GPhaseOp>(phase)
                         ? cast<qc::GPhaseOp>(phase).getTheta()
                         : cast<qco::GPhaseOp>(phase).getTheta();
        const auto constant = valueToConstantDouble(angle);
        if (!constant ||
            (normalizeAngle(*constant) == *constant && *constant != 0.0)) {
          continue;
        }
      }

      for (auto* phase : directPhases[i]) {
        rewriter.eraseOp(phase);
      }
      if (aggregate->expression.isZero()) {
        continue;
      }
      if (terminator != nullptr) {
        rewriter.setInsertionPoint(terminator);
      } else {
        rewriter.setInsertionPointToEnd(&block);
      }
      auto angle = aggregate->expression.materialize(rewriter, aggregate->loc);
      if (aggregate->dialect == PhaseDialect::QC) {
        qc::GPhaseOp::create(rewriter, aggregate->loc, angle);
      } else {
        qco::GPhaseOp::create(rewriter, aggregate->loc, angle);
      }
    }
    return extracted;
  }

  IRRewriter rewriter;
  DenseMap<Operation*, PhaseContributions> contributionsByOperation;
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
