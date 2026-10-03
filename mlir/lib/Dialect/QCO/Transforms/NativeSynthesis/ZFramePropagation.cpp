/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/NativeSynthesis/SingleQubitFusion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>

namespace mlir::qco::decomposition {

static bool isZero(Value value) {
  const auto constant = mqt::valueToConstantDouble(value);
  return constant && std::abs(*constant) <= mqt::PARAMETER_COMPARISON_TOLERANCE;
}

namespace {

/// Binary accumulation keeps exported expression depth logarithmic.
struct ZFrame {
  SmallVector<Value, 4> sums;
  size_t order = 0;

  void append(OpBuilder& builder, Location loc, Value angle) {
    if (isZero(angle)) {
      return;
    }
    for (auto& partial : sums) {
      if (!partial) {
        partial = angle;
        return;
      }
      angle = builder.createOrFold<arith::AddFOp>(loc, partial, angle);
      partial = {};
    }
    sums.push_back(angle);
  }

  Value value(OpBuilder& builder, Location loc) const {
    Value result;
    for (auto partial : llvm::reverse(sums)) {
      if (partial) {
        result = result
                     ? builder.createOrFold<arith::AddFOp>(loc, result, partial)
                     : partial;
      }
    }
    return result ? result : mqt::constantFromScalar(builder, loc, 0.);
  }
};

} // namespace

static bool commutesWithZFrames(UnitaryOpInterface gate) {
  if (auto controlled = dyn_cast<CtrlOp>(gate.getOperation())) {
    auto body =
        mqt::getSoleBodyUnitary<UnitaryOpInterface>(*controlled.getBody());
    return body && isa<ZOp, RZOp, POp>(body.getOperation());
  }
  return isa<RZZOp>(gate.getOperation());
}

LogicalResult synthesizeEquatorialGates(RewriterBase& rewriter,
                                        ModuleOp moduleOp,
                                        const CompilerTarget& target,
                                        const GreedyRewriteConfig& config) {
  if (failed(fuseSingleQubitUnitaryRuns(
          moduleOp, {.singleQubit = SingleQubitBasis::U}, &target, config))) {
    return failure();
  }
  const bool nativeRZ = llvm::all_of(target.siteIds(), [&](auto site) {
    return target.supports(CompilerTarget::GateKind::RZ, ArrayRef(&site, 1));
  });
  const CompilerTarget::SynthesisBasis basis{
      .singleQubit = SingleQubitBasis::R,
  };
  const auto result =
      moduleOp->walk<WalkOrder::PreOrder>([&](Operation* parent) {
        /// Modifier bodies and their phases belong to the enclosing unitary.
        if (isa<UnitaryOpInterface>(parent)) {
          return WalkResult::skip();
        }
        for (auto& region : parent->getRegions()) {
          for (auto& block : region) {
            DenseMap<Value, ZFrame> frames;
            size_t nextOrder = 0;
            const auto take = [&](Value wire) {
              auto found = frames.find(wire);
              if (found == frames.end()) {
                return ZFrame{.order = nextOrder++};
              }
              auto frame = std::move(found->second);
              frames.erase(found);
              return frame;
            };
            const auto flush = [&](Value wire, Location loc) {
              auto found = frames.find(wire);
              if (found == frames.end()) {
                return;
              }
              Value angle = found->second.value(rewriter, loc);
              if (!isZero(angle)) {
                auto& use = *wire.use_begin();
                Value output;
                if (nativeRZ) {
                  output = RZOp::create(rewriter, loc, wire, angle);
                } else if (auto last = wire.getDefiningOp<ROp>();
                           last &&
                           last->getBlock() == rewriter.getInsertionBlock()) {
                  auto angles = *zyzAnglesFromOperation(rewriter, loc, last);
                  angles[1] = rewriter.createOrFold<arith::AddFOp>(
                      loc, mqt::variantToValue(rewriter, loc, angles[1]),
                      angle);
                  output = emitParameterizedEulerAngles(
                      rewriter, loc, last.getQubitIn(), angles, basis);
                  rewriter.modifyOpInPlace(use.getOwner(),
                                           [&] { use.set(output); });
                  rewriter.eraseOp(last);
                  frames.erase(found);
                  return;
                } else {
                  output = emitParameterizedEulerAngles(
                      rewriter, loc, wire, {0., 0., angle, 0.}, basis);
                }
                rewriter.modifyOpInPlace(use.getOwner(),
                                         [&] { use.set(output); });
              }
              frames.erase(found);
            };
            for (auto& operation : llvm::make_early_inc_range(block)) {
              rewriter.setInsertionPoint(&operation);
              auto loc = operation.getLoc();
              const auto normalized = [&](Value angle) {
                return mqt::variantToValue(
                    rewriter, loc,
                    normalizeRotationParameter(rewriter, loc, angle));
              };
              auto gate = dyn_cast<UnitaryOpInterface>(operation);
              if (gate && commutesWithZFrames(gate)) {
                for (auto [input, output] : llvm::zip_equal(
                         gate.getInputQubits(), gate.getOutputQubits())) {
                  if (frames.contains(input)) {
                    frames.try_emplace(output, take(input));
                  }
                }
                continue;
              }
              if (gate && gate.isSingleQubit() && !isa<BarrierOp>(operation)) {
                auto wire = gate.getInputQubit(0);
                if (isa<ROp>(operation) && !frames.contains(wire)) {
                  continue;
                }
                auto frame = take(wire);
                if (auto rotation = dyn_cast<RZOp>(operation)) {
                  frame.append(rewriter, loc, normalized(rotation.getTheta()));
                } else if (auto rotation = dyn_cast<ROp>(operation)) {
                  auto phi = rewriter.createOrFold<arith::SubFOp>(
                      loc, normalized(rotation.getPhi()),
                      frame.value(rewriter, loc));
                  wire = ROp::create(rewriter, loc, wire, rotation.getTheta(),
                                     phi);
                } else {
                  auto angles = zyzAnglesFromOperation(rewriter, loc, gate);
                  if (!angles) {
                    operation.emitError("equatorial synthesis requires a known "
                                        "single-qubit unitary");
                    return WalkResult::interrupt();
                  }
                  auto [theta, phi, lambda, phase] = *angles;
                  Value polar = mqt::variantToValue(rewriter, loc, theta);
                  Value azimuth = mqt::variantToValue(rewriter, loc, phi);
                  Value outer = mqt::variantToValue(rewriter, loc, lambda);
                  if (!isZero(polar)) {
                    auto axis = rewriter.createOrFold<arith::SubFOp>(
                        loc,
                        mqt::constantFromScalar(rewriter, loc,
                                                std::numbers::pi / 2.),
                        rewriter.createOrFold<arith::AddFOp>(
                            loc, outer, frame.value(rewriter, loc)));
                    wire = ROp::create(rewriter, loc, wire, polar, axis);
                  }
                  frame.append(rewriter, loc,
                               rewriter.createOrFold<arith::AddFOp>(
                                   loc, azimuth, outer));
                  auto phaseValue = mqt::variantToValue(rewriter, loc, phase);
                  if (!isZero(phaseValue)) {
                    GPhaseOp::create(rewriter, loc, phaseValue);
                  }
                }
                rewriter.replaceOp(&operation, wire);
                if (!frame.sums.empty()) {
                  frames.try_emplace(wire, std::move(frame));
                }
                continue;
              }
              if (operation.getNumRegions() != 0) {
                SmallVector<std::pair<size_t, Value>> pending;
                for (const auto& [wire, frame] : frames) {
                  pending.emplace_back(frame.order, wire);
                }
                llvm::sort(pending, [](const auto& a, const auto& b) {
                  return a.first < b.first;
                });
                for (const auto& [order, wire] : pending) {
                  flush(wire, loc);
                }
              } else {
                for (auto operand : llvm::to_vector(operation.getOperands())) {
                  flush(operand, loc);
                }
              }
            }
            assert(frames.empty() && "all frames must reach a wire boundary");
          }
        }
        return WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

} // namespace mlir::qco::decomposition
