/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Angles.h"

#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cmath>
#include <numbers>

namespace mlir::mqt {

double normalizeAngle(double theta) {
  const double twoPi = 2.0 * std::numbers::pi;
  theta = std::fmod(theta, twoPi);
  if (theta > std::numbers::pi) {
    theta -= twoPi;
  }
  if (theta <= -std::numbers::pi) {
    theta += twoPi;
  }
  return theta;
}

Value normalizeAngle(RewriterBase& rewriter, Location loc, Value theta) {
  if (const auto constant = valueToConstantDouble(theta)) {
    return constantFromScalar(rewriter, loc, normalizeAngle(*constant));
  }

  const auto pi = constantFromScalar(rewriter, loc, std::numbers::pi);
  const auto negativePi = constantFromScalar(rewriter, loc, -std::numbers::pi);
  const auto twoPi = constantFromScalar(rewriter, loc, 2.0 * std::numbers::pi);
  auto remainder = arith::RemFOp::create(rewriter, loc, theta, twoPi);
  auto abovePi = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                       remainder, pi);
  auto belowOrAtNegativePi = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLE, remainder, negativePi);
  auto subtractTurn = arith::SubFOp::create(rewriter, loc, remainder, twoPi);
  auto addTurn = arith::AddFOp::create(rewriter, loc, remainder, twoPi);
  auto upperBounded =
      arith::SelectOp::create(rewriter, loc, abovePi, subtractTurn, remainder);
  return arith::SelectOp::create(rewriter, loc, belowOrAtNegativePi, addTurn,
                                 upperBounded)
      .getResult();
}

double scaleAngleByInteger(double theta, double factor) {
  assert(std::isfinite(factor) && factor == std::floor(factor));
  double remaining = std::abs(factor);
  double multiple = normalizeAngle(theta);
  double result = 0.0;
  while (remaining >= 1.0) {
    if (std::fmod(remaining, 2.0) == 1.0) {
      result = normalizeAngle(result + multiple);
    }
    remaining = std::floor(remaining / 2.0);
    if (remaining >= 1.0) {
      multiple = normalizeAngle(multiple + multiple);
    }
  }
  return factor < 0.0 ? normalizeAngle(-result) : result;
}

bool isValidGlobalPhaseAngle(const double theta) {
  return std::isfinite(theta) && std::abs(theta) <= MAX_GLOBAL_PHASE_ANGLE;
}

LogicalResult verifyGlobalPhaseAngle(Operation* operation, Value angle) {
  const auto constant = valueToConstantDouble(angle);
  if (!constant) {
    return success();
  }
  if (!std::isfinite(*constant)) {
    return operation->emitOpError("constant angle must be finite");
  }
  if (!isValidGlobalPhaseAngle(*constant)) {
    return operation->emitOpError()
           << "constant angle must have magnitude at most "
           << MAX_GLOBAL_PHASE_ANGLE << " radians";
  }
  return success();
}

} // namespace mlir::mqt
