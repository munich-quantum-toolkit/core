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
    raw = ctrl.getBodyUnitary().getOperation();
  }
  if (llvm::isa<IdOp>(raw)) {
    return decomposition::GateKind::I;
  }
  if (llvm::isa<HOp>(raw)) {
    return decomposition::GateKind::H;
  }
  if (llvm::isa<POp>(raw)) {
    return decomposition::GateKind::P;
  }
  if (llvm::isa<UOp>(raw)) {
    return decomposition::GateKind::U;
  }
  if (llvm::isa<U2Op>(raw)) {
    return decomposition::GateKind::U2;
  }
  if (llvm::isa<XOp>(raw)) {
    return decomposition::GateKind::X;
  }
  if (llvm::isa<YOp>(raw)) {
    return decomposition::GateKind::Y;
  }
  if (llvm::isa<ZOp>(raw)) {
    return decomposition::GateKind::Z;
  }
  if (llvm::isa<SXOp>(raw)) {
    return decomposition::GateKind::SX;
  }
  if (llvm::isa<RXOp>(raw)) {
    return decomposition::GateKind::RX;
  }
  if (llvm::isa<RYOp>(raw)) {
    return decomposition::GateKind::RY;
  }
  if (llvm::isa<RZOp>(raw)) {
    return decomposition::GateKind::RZ;
  }
  if (llvm::isa<ROp>(raw)) {
    return decomposition::GateKind::R;
  }
  if (llvm::isa<RXXOp>(raw)) {
    return decomposition::GateKind::RXX;
  }
  if (llvm::isa<RYYOp>(raw)) {
    return decomposition::GateKind::RYY;
  }
  if (llvm::isa<RZZOp>(raw)) {
    return decomposition::GateKind::RZZ;
  }
  if (llvm::isa<GPhaseOp>(raw)) {
    return decomposition::GateKind::GPhase;
  }
  llvm::reportFatalInternalError("Unsupported QCO unitary operation kind");
}

double remEuclid(double a, double b) {
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
