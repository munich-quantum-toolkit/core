/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>

namespace mlir::qco::decomposition {

/// Emit local ZYZ angles with fixed pulses, returning the phase correction.
/// Numeric and SSA callers supply constants and emitters for their angle type.
/// Callers handle a statically zero theta by emitting phi + lambda directly.
template <typename Angle>
double
emitFixedRotationSequence(const CompilerTarget::FixedRotationBasis& basis,
                          Angle theta, Angle phi, Angle lambda,
                          std::optional<double> constantTheta, auto constant,
                          auto emitFree, auto emitPulse) {
  constexpr double pi = std::numbers::pi;
  constexpr double halfPi = pi / 2.;
  const auto matchesTheta = [&](double value) {
    return constantTheta && std::abs(*constantTheta - value) <=
                                mqt::PARAMETER_COMPARISON_TOLERANCE;
  };
  const auto quarterTurn = [&] {
    emitFree(constant(basis.quarterTurnAngles.front()));
    for (size_t i = 1; i < basis.quarterTurnAngles.size(); ++i) {
      emitPulse(basis.angle);
      emitFree(constant(basis.quarterTurnAngles[i]));
    }
  };
  if (matchesTheta(halfPi)) {
    emitFree(lambda - constant(halfPi));
    quarterTurn();
    emitFree(phi + constant(halfPi));
    return 0.;
  }
  if (matchesTheta(pi) && basis.halfTurnAngle) {
    const double axis = basis.gate == basis.axes()[0] ? 0. : halfPi;
    emitFree(lambda + constant(axis));
    emitPulse(*basis.halfTurnAngle);
    emitFree(phi + constant(pi) - constant(axis));
    return *basis.halfTurnAngle < 0. ? pi : 0.;
  }
  emitFree(lambda);
  quarterTurn();
  emitFree(theta + constant(pi));
  quarterTurn();
  emitFree(phi + constant(pi));
  return pi;
}

/// Emit GPI2/GPI/GPI2, or four GPI2 pulses, and return the phase correction.
/// The emitter takes a GPI/GPI2 selector and an angle in turns.
template <typename Angle>
double emitGPISequence(Angle theta, Angle phi, Angle lambda, bool useGPI,
                       auto constant, auto emit) {
  constexpr double pi = std::numbers::pi;
  const auto twoPi = constant(2. * pi);
  const auto middle = (phi - lambda - theta) / constant(4. * pi);
  emit(false, -lambda / twoPi);
  if (useGPI) {
    emit(true, middle);
  } else {
    emit(false, middle);
    emit(false, middle);
  }
  emit(false, phi / twoPi);
  return useGPI ? pi / 2. : pi;
}

} // namespace mlir::qco::decomposition
