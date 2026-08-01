/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/Utils/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <variant>

namespace mlir::quantum {

#define GEN_PASS_DEF_NORMALIZEGLOBALPHASES
#include "mlir/Dialect/Utils/Transforms/Passes.h.inc"

namespace {

using PhaseTerm = std::variant<double, Value>;

/// Collect a pure, body-local dependency slice in topological order.
static bool collectHoistableSlice(Value value, Block& body,
                                  SmallPtrSetImpl<Operation*>& visiting,
                                  SmallPtrSetImpl<Operation*>& collected,
                                  SmallVectorImpl<Operation*>& ordered) {
  if (const auto blockArg = dyn_cast<BlockArgument>(value)) {
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
  for (const auto operand : definingOp->getOperands()) {
    if (!collectHoistableSlice(operand, body, visiting, collected, ordered)) {
      return false;
    }
  }
  visiting.erase(definingOp);
  collected.insert(definingOp);
  ordered.push_back(definingOp);
  return true;
}

/// Make @p angle available before @p modifier without moving impure operations.
static bool hoistAngleBefore(Value angle, Block& body, Operation* modifier,
                             RewriterBase& rewriter) {
  auto* definingOp = angle.getDefiningOp();
  if (definingOp == nullptr || definingOp->getBlock() != &body) {
    if (const auto blockArg = dyn_cast<BlockArgument>(angle)) {
      return blockArg.getOwner() != &body;
    }
    return true;
  }

  SmallPtrSet<Operation*, 8> visiting;
  SmallPtrSet<Operation*, 8> collected;
  SmallVector<Operation*, 8> ordered;
  if (!collectHoistableSlice(angle, body, visiting, collected, ordered)) {
    return false;
  }
  for (auto* op : ordered) {
    rewriter.moveOpBefore(op, modifier);
  }
  return true;
}

template <typename GPhaseOp> static GPhaseOp getExitPhase(Block& block) {
  auto* terminator = block.getTerminator();
  if (terminator == nullptr) {
    return {};
  }
  auto* previous = terminator->getPrevNode();
  return previous == nullptr ? GPhaseOp{} : dyn_cast<GPhaseOp>(previous);
}

static Value negateAngle(Value angle, Location loc, RewriterBase& rewriter) {
  return rewriter.createOrFold<arith::NegFOp>(loc, angle);
}

static Value scaleAngle(Value angle, Value exponent, Location loc,
                        RewriterBase& rewriter) {
  return rewriter.createOrFold<arith::MulFOp>(loc, exponent, angle);
}

static void factorInverse(qc::InvOp op, RewriterBase& rewriter) {
  auto phase = getExitPhase<qc::GPhaseOp>(*op.getBody());
  if (!phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);
  qc::GPhaseOp::create(rewriter, loc, negateAngle(angle, loc, rewriter));
}

static void factorInverse(qco::InvOp op, RewriterBase& rewriter) {
  auto phase = getExitPhase<qco::GPhaseOp>(*op.getBody());
  if (!phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);
  qco::GPhaseOp::create(rewriter, loc, negateAngle(angle, loc, rewriter));
}

static void factorPower(qc::PowOp op, RewriterBase& rewriter) {
  const auto exponent = op.getExponentValue();
  auto phase = getExitPhase<qc::GPhaseOp>(*op.getBody());
  if (!exponent || !utils::isIntegerExponent(*exponent) || !phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);
  qc::GPhaseOp::create(rewriter, loc,
                       scaleAngle(angle, op.getExponent(), loc, rewriter));
}

static void factorPower(qco::PowOp op, RewriterBase& rewriter) {
  const auto exponent = op.getExponentValue();
  auto phase = getExitPhase<qco::GPhaseOp>(*op.getBody());
  if (!exponent || !utils::isIntegerExponent(*exponent) || !phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);
  qco::GPhaseOp::create(rewriter, loc,
                        scaleAngle(angle, op.getExponent(), loc, rewriter));
}

static void factorControl(qc::CtrlOp op, RewriterBase& rewriter) {
  auto phase = getExitPhase<qc::GPhaseOp>(*op.getBody());
  if (!phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);

  if (op.getNumControls() == 0) {
    qc::GPhaseOp::create(rewriter, loc, angle);
    return;
  }
  if (op.getNumControls() == 1) {
    qc::POp::create(rewriter, loc, op.getControl(0), angle);
    return;
  }
  const auto controls = op.getControls();
  qc::CtrlOp::create(
      rewriter, loc, controls.drop_back(), controls.back(),
      [&](Value target) { qc::POp::create(rewriter, loc, target, angle); });
}

static void factorControl(qco::CtrlOp op, RewriterBase& rewriter) {
  auto phase = getExitPhase<qco::GPhaseOp>(*op.getBody());
  if (!phase ||
      !hoistAngleBefore(phase.getTheta(), *op.getBody(), op, rewriter)) {
    return;
  }
  const auto loc = phase.getLoc();
  const auto angle = phase.getTheta();
  rewriter.eraseOp(phase);
  rewriter.setInsertionPointAfter(op);

  if (op.getNumControls() == 0) {
    qco::GPhaseOp::create(rewriter, loc, angle);
    return;
  }
  SmallVector<Value> oldControls(op.getOutputControls());
  SmallVector<Value> newControls;
  Operation* relativePhase = nullptr;
  if (op.getNumControls() == 1) {
    auto p = qco::POp::create(rewriter, loc, oldControls.front(), angle);
    newControls.push_back(p.getOutputTarget(0));
    relativePhase = p;
  } else {
    auto relative = qco::CtrlOp::create(
        rewriter, loc, ValueRange(oldControls).drop_back(), oldControls.back(),
        [&](Value target) {
          return qco::POp::create(rewriter, loc, target, angle)
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

static void factorModifier(Operation* op, RewriterBase& rewriter) {
  TypeSwitch<Operation*>(op)
      .Case<qc::InvOp, qco::InvOp>(
          [&](auto modifier) { factorInverse(modifier, rewriter); })
      .Case<qc::PowOp, qco::PowOp>(
          [&](auto modifier) { factorPower(modifier, rewriter); })
      .Case<qc::CtrlOp, qco::CtrlOp>(
          [&](auto modifier) { factorControl(modifier, rewriter); });
}

static void flushConstant(std::optional<double>& constant,
                          SmallVectorImpl<PhaseTerm>& terms) {
  if (!constant) {
    return;
  }
  const auto normalized = utils::normalizeAngle(*constant);
  if (normalized != 0.0) {
    terms.emplace_back(normalized);
  }
  constant.reset();
}

template <typename GPhaseOp>
static void normalizeBlockPhases(Block& block, RewriterBase& rewriter) {
  auto phases = llvm::to_vector(block.getOps<GPhaseOp>());
  if (phases.empty()) {
    return;
  }
  if (phases.size() == 1 &&
      phases.front()->getNextNode() == block.getTerminator()) {
    const auto value = utils::valueToDouble(phases.front().getTheta());
    if (!value || !std::isfinite(*value) ||
        (utils::normalizeAngle(*value) == *value && *value != 0.0)) {
      return;
    }
  }

  SmallVector<PhaseTerm> terms;
  std::optional<double> constant;
  for (auto phase : phases) {
    const auto value = utils::valueToDouble(phase.getTheta());
    if (!value || !std::isfinite(*value)) {
      flushConstant(constant, terms);
      terms.emplace_back(phase.getTheta());
      continue;
    }
    if (!constant) {
      constant = utils::normalizeAngle(*value);
      continue;
    }
    // Reduce each literal before adding it. This keeps the accumulator bounded
    // even when the original finite f64 literals would overflow when added.
    constant = utils::normalizeAngle(*constant + utils::normalizeAngle(*value));
  }
  flushConstant(constant, terms);

  const auto loc = phases.front().getLoc();
  for (auto phase : phases) {
    rewriter.eraseOp(phase);
  }
  if (terms.empty()) {
    return;
  }

  rewriter.setInsertionPoint(block.getTerminator());
  Value total;
  for (const auto& term : terms) {
    Value value;
    if (const auto* scalar = std::get_if<double>(&term)) {
      value = utils::constantFromScalar(rewriter, loc, *scalar);
    } else {
      value = std::get<Value>(term);
    }
    total = total ? arith::AddFOp::create(rewriter, loc, total, value) : value;
  }
  GPhaseOp::create(rewriter, loc, total);
}

static void normalizeRegion(Region& region, RewriterBase& rewriter) {
  for (auto& block : region) {
    for (auto& op : llvm::make_early_inc_range(block.without_terminator())) {
      for (auto& nested : op.getRegions()) {
        normalizeRegion(nested, rewriter);
      }
      factorModifier(&op, rewriter);
    }
    normalizeBlockPhases<qc::GPhaseOp>(block, rewriter);
    normalizeBlockPhases<qco::GPhaseOp>(block, rewriter);
  }
}

struct NormalizeGlobalPhases final
    : impl::NormalizeGlobalPhasesBase<NormalizeGlobalPhases> {
  using NormalizeGlobalPhasesBase::NormalizeGlobalPhasesBase;

  void runOnOperation() override {
    if (failed(normalizeGlobalPhases(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult normalizeGlobalPhases(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  normalizeRegion(module.getRegion(), rewriter);
  return success();
}

} // namespace mlir::quantum
