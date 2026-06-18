/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Operation.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>

namespace mlir::qco::helpers {

decomposition::GateKind getGateKind(UnitaryOpInterface op) {
  Operation* raw = op.getOperation();
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(raw)) {
    // Controlled operations encode the physical gate in the body region.
    raw = ctrl.getBodyUnitary(0).getOperation();
  }
  return llvm::TypeSwitch<Operation*, decomposition::GateKind>(raw)
      .Case<IdOp>([](auto) { return decomposition::GateKind::I; })
      .Case<HOp>([](auto) { return decomposition::GateKind::H; })
      .Case<POp>([](auto) { return decomposition::GateKind::P; })
      .Case<UOp>([](auto) { return decomposition::GateKind::U; })
      .Case<U2Op>([](auto) { return decomposition::GateKind::U2; })
      .Case<XOp>([](auto) { return decomposition::GateKind::X; })
      .Case<YOp>([](auto) { return decomposition::GateKind::Y; })
      .Case<ZOp>([](auto) { return decomposition::GateKind::Z; })
      .Case<SXOp>([](auto) { return decomposition::GateKind::SX; })
      .Case<RXOp>([](auto) { return decomposition::GateKind::RX; })
      .Case<RYOp>([](auto) { return decomposition::GateKind::RY; })
      .Case<RZOp>([](auto) { return decomposition::GateKind::RZ; })
      .Case<ROp>([](auto) { return decomposition::GateKind::R; })
      .Case<RXXOp>([](auto) { return decomposition::GateKind::RXX; })
      .Case<RYYOp>([](auto) { return decomposition::GateKind::RYY; })
      .Case<RZZOp>([](auto) { return decomposition::GateKind::RZZ; })
      .Case<GPhaseOp>([](auto) { return decomposition::GateKind::GPhase; })
      .Default([](Operation*) -> decomposition::GateKind {
        llvm::reportFatalInternalError(
            "Unsupported QCO unitary operation kind");
        llvm_unreachable("unsupported gate kind");
      });
}

bool isUnitaryMatrix(const Matrix2x2& matrix, double tolerance) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

bool isUnitaryMatrix(const Matrix4x4& matrix, double tolerance) {
  return (matrix.adjoint() * matrix).isIdentity(tolerance);
}

double remEuclid(double a, double b) {
  if (b == 0.0) {
    llvm::reportFatalInternalError(
        "remEuclid expects non-zero divisor; callers like mod2pi pass positive "
        "constants");
  }
  auto r = std::fmod(a, b);
  return (r < 0.0) ? r + std::abs(b) : r;
}

double mod2pi(double angle, double angleZeroEpsilon) {
  // remEuclid() isn't exactly the same as Python's % operator, but
  // because the RHS here is a constant and positive it is effectively
  // equivalent for this case
  auto wrapped = remEuclid(angle + std::numbers::pi, 2 * std::numbers::pi) -
                 std::numbers::pi;
  if (std::abs(wrapped - std::numbers::pi) < angleZeroEpsilon) {
    // Canonicalize the upper endpoint back to -pi so callers always receive a
    // half-open interval [-pi, pi).
    return -std::numbers::pi;
  }
  return wrapped;
}

double traceToFidelity(const std::complex<double>& x) {
  // Average two-qubit process fidelity given the Hilbert-Schmidt overlap
  // `x = tr(U_target^dag * U_actual)`. For a 4x4 unitary the general formula is
  // `F_avg = (d + |tr|^2) / (d * (d + 1))` with `d = 4`, which reduces to the
  // `(4 + |x|^2) / 20` expression below. See e.g. Horodecki/Nielsen.
  auto xAbs = std::abs(x);
  return (4.0 + (xAbs * xAbs)) / 20.0;
}

std::size_t getComplexity(decomposition::GateKind type,
                          std::size_t numOfQubits) {
  if (numOfQubits > 1) {
    // Multi-qubit operations dominate the heuristic cost model.
    constexpr std::size_t multiQubitFactor = 10;
    return (numOfQubits - 1) * multiQubitFactor;
  }
  if (type == decomposition::GateKind::GPhase) {
    return 0;
  }
  return 1;
}

std::complex<double> globalPhaseFactor(double globalPhase) {
  return std::exp(std::complex<double>{0, 1} * globalPhase);
}

} // namespace mlir::qco::helpers
