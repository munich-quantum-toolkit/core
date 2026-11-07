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
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

using mqt::FloatExpression;

static bool isZero(Value value) {
  const auto constant = mqt::valueToConstantDouble(value);
  return constant && std::abs(*constant) <= mqt::PARAMETER_COMPARISON_TOLERANCE;
}

namespace {

/// Binary accumulation keeps exported expression depth logarithmic.
struct ZFrame {
  // Keep per-wire map entries compact.
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

LogicalResult propagateZFrames(RewriterBase& rewriter, ModuleOp moduleOp,
                               const CompilerTarget& target) {
  const bool equatorial =
      target.synthesisBasis() &&
      target.synthesisBasis()->singleQubit == SingleQubitBasis::R;
  // Reuse cached coverage before checking heterogeneous site capabilities.
  const bool nativeRZ =
      target.findOperation("rz", 1, {std::nullopt}, std::nullopt).capability !=
          nullptr ||
      (target.hasDeviceConstraints()
           ? llvm::all_of(target.siteIds(),
                          [&](auto site) {
                            return target.supports(CompilerTarget::GateKind::RZ,
                                                   ArrayRef(&site, 1));
                          })
           : target.supports(CompilerTarget::GateKind::RZ));
  if (!equatorial && !nativeRZ) {
    return success();
  }
  DominanceInfo dominance;
  const WalkResult result = moduleOp->walk<
      WalkOrder::PreOrder>([&](Operation* parent) {
    // Modifier bodies and their phases belong to the enclosing unitary.
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
          OpBuilder::InsertionGuard guard(rewriter);
          ROp last;
          if (!nativeRZ) {
            auto anchor = wire;
            auto gate = anchor.getDefiningOp<UnitaryOpInterface>();
            while (gate && gate->getBlock() == &block &&
                   commutesWithZFrames(gate)) {
              anchor = gate.getInputForOutput(anchor);
              gate = anchor.getDefiningOp<UnitaryOpInterface>();
            }
            last = anchor.getDefiningOp<ROp>();
            if (last && last->getBlock() == &block && wire == anchor) {
              // The adjacent R can be replaced at the current boundary.
            } else if (last && last->getBlock() == &block &&
                       llvm::all_of(found->second.sums, [&](Value angle) {
                         return !angle ||
                                dominance.properlyDominates(angle, last) ||
                                mqt::valueToConstantDouble(angle);
                       })) {
              // Absorb before a diagonal segment when its scalar inputs
              // are already available at the preceding equatorial gate.
              wire = anchor;
              rewriter.setInsertionPoint(last);
              for (auto& partial : found->second.sums) {
                if (partial && !dominance.properlyDominates(partial, last)) {
                  partial = mqt::constantFromScalar(
                      rewriter, loc, *mqt::valueToConstantDouble(partial));
                }
              }
            } else {
              last = {};
            }
          }
          Value angle = found->second.value(rewriter, loc);
          if (!isZero(angle)) {
            auto& use = *wire.use_begin();
            Value output;
            if (nativeRZ) {
              output = RZOp::create(rewriter, loc, wire, angle);
            } else {
              const auto zero = FloatExpression::constant(rewriter, loc, 0.);
              const auto pi =
                  FloatExpression::constant(rewriter, loc, std::numbers::pi);
              const FloatExpression axis(
                  rewriter, loc, last ? last.getPhi() : zero.getValue());
              const FloatExpression theta(
                  rewriter, loc,
                  last ? normalizeRotationParameter(rewriter, loc,
                                                    last.getTheta())
                       : mqt::FloatParameter{zero.getValue()});
              // RZ(a) R(t,p) = R(pi,p+a/2) R(t-pi,p), including its phase.
              auto first =
                  ROp::create(rewriter, loc, last ? last.getQubitIn() : wire,
                              (theta - pi).getValue(), axis.getValue());
              const auto half = FloatExpression(rewriter, loc, angle) *
                                FloatExpression::constant(rewriter, loc, 0.5);
              output = ROp::create(rewriter, loc, first, pi.getValue(),
                                   (axis + half).getValue());
            }
            rewriter.modifyOpInPlace(use.getOwner(), [&] { use.set(output); });
            if (last) {
              rewriter.eraseOp(last);
            }
          }
          frames.erase(found);
        };
        for (auto iterator = block.begin(); iterator != block.end();) {
          auto& operation = *iterator++;
          rewriter.setInsertionPoint(&operation);
          auto loc = operation.getLoc();
          if (isa<MeasureOp, ResetOp>(operation)) {
            // A Z frame changes only the phase of each measurement
            // branch; reset discards the previous state altogether.
            frames.erase(operation.getOperand(0));
            continue;
          }
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
          if (gate && gate.isSingleQubit() && !isa<BarrierOp>(operation) &&
              (equatorial || isa<RZOp>(operation))) {
            auto wire = gate.getInputQubit(0);
            if (!equatorial && !frames.contains(wire)) {
              auto next = dyn_cast<UnitaryOpInterface>(
                  *gate.getOutputQubit(0).user_begin());
              // Leave isolated native rotations and their angles intact.
              if (!next || !commutesWithZFrames(next)) {
                continue;
              }
            }
            auto tail = gate;
            std::optional<std::array<RotationParameter, 4>> fusedAngles;
            if (equatorial) {
              if (auto composed = gate.getUnitaryMatrix<Matrix2x2>()) {
                auto next = dyn_cast<UnitaryOpInterface>(
                    *tail.getOutputQubit(0).user_begin());
                while (next && next->getBlock() == &block &&
                       next.isSingleQubit() && !isa<BarrierOp>(next)) {
                  const auto matrix = next.getUnitaryMatrix<Matrix2x2>();
                  if (!matrix) {
                    break;
                  }
                  composed->premultiplyBy(*matrix);
                  tail = next;
                  next = dyn_cast<UnitaryOpInterface>(
                      *tail.getOutputQubit(0).user_begin());
                }
                if (tail != gate) {
                  const auto angles =
                      anglesFromUnitary(*composed, SingleQubitBasis::ZYZ);
                  // Matrix resynthesis uses the same tolerance as Euler
                  // emission.
                  fusedAngles = {
                      std::abs(angles.theta) <= MATRIX_TOLERANCE ? 0.
                                                                 : angles.theta,
                      angles.phi,
                      angles.lambda,
                      angles.phase,
                  };
                }
              }
            }
            if (!fusedAngles && isa<ROp>(operation) && !frames.contains(wire)) {
              continue;
            }
            auto frame = take(wire);
            if (auto rotation = dyn_cast<RZOp>(operation);
                rotation && !fusedAngles) {
              frame.append(rewriter, loc, normalized(rotation.getTheta()));
            } else if (auto rotation = dyn_cast<ROp>(operation);
                       rotation && !fusedAngles) {
              auto phi = rewriter.createOrFold<arith::SubFOp>(
                  loc, normalized(rotation.getPhi()),
                  frame.value(rewriter, loc));
              wire = ROp::create(rewriter, loc, wire, rotation.getTheta(), phi);
            } else {
              auto angles = fusedAngles
                                ? fusedAngles
                                : zyzAnglesFromOperation(rewriter, loc, gate);
              if (!angles) {
                operation.emitError("equatorial synthesis requires a known "
                                    "single-qubit unitary");
                return WalkResult::interrupt();
              }
              auto [theta, phi, lambda, phase] = *angles;
              Value polar = mqt::variantToValue(rewriter, loc, theta);
              const FloatExpression azimuth(rewriter, loc, phi);
              const FloatExpression outer(rewriter, loc, lambda);
              if (!isZero(polar)) {
                const auto halfPi = FloatExpression::constant(
                    rewriter, loc, std::numbers::pi / 2.);
                const FloatExpression frameAngle(rewriter, loc,
                                                 frame.value(rewriter, loc));
                const auto axis = halfPi - (outer + frameAngle);
                wire = ROp::create(rewriter, loc, wire, polar, axis.getValue());
              }
              frame.append(rewriter, loc, (azimuth + outer).getValue());
              auto phaseValue = mqt::variantToValue(rewriter, loc, phase);
              if (!isZero(phaseValue)) {
                GPhaseOp::create(rewriter, loc, phaseValue);
              }
            }
            if (tail != gate) {
              rewriter.replaceAllUsesWith(tail.getOutputQubit(0), wire);
              while (tail != gate) {
                auto previous =
                    tail.getInputQubit(0).getDefiningOp<UnitaryOpInterface>();
                // The next block operation can be a member of this wire's run.
                if (iterator != block.end() &&
                    &*iterator == tail.getOperation()) {
                  ++iterator;
                }
                rewriter.eraseOp(tail);
                tail = previous;
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
