/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/BasisDecomposer.h"

#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Helpers.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/UnitaryMatrices.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco::decomposition {

using namespace std::complex_literals;

TwoQubitBasisDecomposer TwoQubitBasisDecomposer::create(const Gate& basisGate,
                                                        double basisFidelity) {
  const Matrix2x2 k12RArr = Matrix2x2::fromElements(
      1i * FRAC1_SQRT2, FRAC1_SQRT2, -FRAC1_SQRT2, -1i * FRAC1_SQRT2);
  const Matrix2x2 k12LArr =
      Matrix2x2::fromElements(Complex{0.5, 0.5}, Complex{0.5, 0.5},
                              Complex{-0.5, 0.5}, Complex{0.5, -0.5});

  // The Shende-Markov-Bullock 3-CX sandwich (and its 1/2-CX reductions) used
  // below is derived for a basis CX whose 4x4 matrix is the Qiskit/LSB form
  // `[[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]`, i.e. "control on the LSB
  // factor, target on the MSB factor" of the tensor product. MQT's wider
  // convention places operand 0 on the MSB factor, so `getTwoQubitMatrix` for
  // the same logical CX gives the SWAP-conjugate
  // `[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]`.
  //
  // Because `SWAP * C(a,b,c) * SWAP = C(a,b,c)` but
  // `SWAP * (K1l ⊗ K1r) * SWAP = (K1r ⊗ K1l)`, feeding the MSB matrix directly
  // into the Weyl decomposer would swap the roles of `k1l`/`k1r` (and `k2l`/
  // `k2r`) relative to the hard-coded constants above. To keep the SMB algebra
  // self-consistent we SWAP-conjugate the basis matrix here (restoring the
  // Qiskit/LSB 4x4) and then absorb the resulting "left/right" relabeling at
  // the emission boundary in `decomp{0,1,2,3}` below. This reproduces the
  // pre-flip gate counts without having to re-derive every SMB constant for
  // the MSB basis -- the two routes are algebraically equivalent.
  const Matrix4x4 basisMatrixLsb =
      swapGate() * getTwoQubitMatrix(basisGate) * swapGate();
  const auto basisDecomposer = decomposition::TwoQubitWeylDecomposition::create(
      basisMatrixLsb, basisFidelity);
  const auto isSuperControlled =
      relativeEq(basisDecomposer.a(), std::numbers::pi / 4.0, 1e-13, 1e-09) &&
      relativeEq(basisDecomposer.c(), 0.0, 1e-13, 1e-09);

  // Create some useful matrices U1, U2, U3 are equivalent to the basis,
  // expand as Ui = Ki1.Ubasis.Ki2
  auto b = basisDecomposer.b();
  Complex temp{0.5, -0.5};
  const Matrix2x2 k11l = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-1i * b)), temp * std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * -std::exp(1i * b));
  const Matrix2x2 k11r = Matrix2x2::fromElements(
      FRAC1_SQRT2 * (1i * std::exp(-1i * b)), FRAC1_SQRT2 * -std::exp(-1i * b),
      FRAC1_SQRT2 * std::exp(1i * b), FRAC1_SQRT2 * (-1i * std::exp(1i * b)));
  const Matrix2x2 k32lK21l =
      Matrix2x2::fromElements(FRAC1_SQRT2 * Complex{1., std::cos(2. * b)},
                              FRAC1_SQRT2 * (1i * std::sin(2. * b)),
                              FRAC1_SQRT2 * (1i * std::sin(2. * b)),
                              FRAC1_SQRT2 * Complex{1., -std::cos(2. * b)});
  temp = Complex{0.5, 0.5};
  const Matrix2x2 k21r = Matrix2x2::fromElements(
      temp * (-1i * std::exp(-2i * b)), temp * std::exp(-2i * b),
      temp * (1i * std::exp(2i * b)), temp * std::exp(2i * b));
  const Matrix2x2 k22l = Matrix2x2::fromElements(FRAC1_SQRT2, -FRAC1_SQRT2,
                                                 FRAC1_SQRT2, FRAC1_SQRT2);
  const Matrix2x2 k22r = Matrix2x2::fromElements(0, 1, -1, 0);
  const Matrix2x2 k31l = Matrix2x2::fromElements(
      FRAC1_SQRT2 * std::exp(-1i * b), FRAC1_SQRT2 * std::exp(-1i * b),
      FRAC1_SQRT2 * -std::exp(1i * b), FRAC1_SQRT2 * std::exp(1i * b));
  const Matrix2x2 k31r = Matrix2x2::fromElements(1i * std::exp(1i * b), 0, 0,
                                                 -1i * std::exp(-1i * b));
  const Matrix2x2 k32r = Matrix2x2::fromElements(
      temp * std::exp(1i * b), temp * -std::exp(-1i * b),
      temp * (-1i * std::exp(1i * b)), temp * (-1i * std::exp(-1i * b)));
  auto k1lDagger = basisDecomposer.k1l().adjoint();
  auto k1rDagger = basisDecomposer.k1r().adjoint();
  auto k2lDagger = basisDecomposer.k2l().adjoint();
  auto k2rDagger = basisDecomposer.k2r().adjoint();
  // Pre-build the fixed parts of the matrices used in 3-part decomposition
  auto u0l = k31l * k1lDagger;
  auto u0r = k31r * k1rDagger;
  auto u1l = k2lDagger * k32lK21l * k1lDagger;
  auto u1ra = k2rDagger * k32r;
  auto u1rb = k21r * k1rDagger;
  auto u2la = k2lDagger * k22l;
  auto u2lb = k11l * k1lDagger;
  auto u2ra = k2rDagger * k22r;
  auto u2rb = k11r * k1rDagger;
  auto u3l = k2lDagger * k12LArr;
  auto u3r = k2rDagger * k12RArr;
  // Pre-build the fixed parts of the matrices used in the 2-part decomposition
  auto q0l = k12LArr.adjoint() * k1lDagger;
  auto q0r = k12RArr.adjoint() * ipz() * k1rDagger;
  auto q1la = k2lDagger * k11l.adjoint();
  auto q1lb = k11l * k1lDagger;
  auto q1ra = k2rDagger * ipz() * k11r.adjoint();
  auto q1rb = k11r * k1rDagger;
  auto q2l = k2lDagger * k12LArr;
  auto q2r = k2rDagger * k12RArr;

  return TwoQubitBasisDecomposer{
      basisGate,
      basisFidelity,
      basisDecomposer,
      isSuperControlled,
      u0l,
      u0r,
      u1l,
      u1ra,
      u1rb,
      u2la,
      u2lb,
      u2ra,
      u2rb,
      u3l,
      u3r,
      q0l,
      q0r,
      q1la,
      q1lb,
      q1ra,
      q1rb,
      q2l,
      q2r,
  };
}

std::optional<TwoQubitNativeDecomposition>
TwoQubitBasisDecomposer::twoQubitDecompose(
    const decomposition::TwoQubitWeylDecomposition& targetDecomposition,
    std::optional<std::uint8_t> numBasisGateUses) const {
  auto traces = this->traces(targetDecomposition);
  auto getDefaultNbasis = [&]() -> std::uint8_t {
    // Pick the number of basis gate uses `i ∈ {0, 1, 2, 3}` that maximizes
    //   expected_fidelity(i) = traceToFidelity(traces[i]) * basisFidelity^i.
    auto bestValue = std::numeric_limits<double>::lowest();
    auto bestIndex = -1;
    for (int i = 0; std::cmp_less(i, traces.size()); ++i) {
      auto value =
          helpers::traceToFidelity(traces[i]) * std::pow(basisFidelity, i);
      if (std::isnan(value)) {
        continue;
      }
      if (value > bestValue) {
        bestIndex = i;
        bestValue = value;
      }
    }
    if (bestIndex < 0) {
      llvm::reportFatalInternalError("Unable to select basis-gate count: all "
                                     "candidate fidelities are NaN");
    }
    return static_cast<std::uint8_t>(bestIndex);
  };
  // number of basis gates that need to be used in the decomposition
  auto bestNbasis = numBasisGateUses.value_or(getDefaultNbasis());
  if (bestNbasis > 1 && !isSuperControlled) {
    // cannot reliably decompose with more than one basis gate and a
    // non-super-controlled basis gate
    return std::nullopt;
  }
  auto chooseDecomposition = [&]() {
    if (bestNbasis == 0) {
      return decomp0(targetDecomposition);
    }
    if (bestNbasis == 1) {
      return decomp1(targetDecomposition);
    }
    if (bestNbasis == 2) {
      return decomp2Supercontrolled(targetDecomposition);
    }
    if (bestNbasis == 3) {
      return decomp3Supercontrolled(targetDecomposition);
    }
    llvm::reportFatalInternalError(
        "Invalid number of basis gates to use in basis decomposition (" +
        llvm::Twine(bestNbasis) + ")!");
    llvm_unreachable("");
  };
  TwoQubitLocalUnitaryList factors = chooseDecomposition();
#ifndef NDEBUG
  for (const auto& factor : factors) {
    assert(helpers::isUnitaryMatrix(factor));
  }
#endif

  double globalPhase = targetDecomposition.globalPhase();
  globalPhase -= bestNbasis * basisDecomposer.globalPhase();
  if (bestNbasis == 2) {
    // The two-basis (2x CX/CZ) template in `decomp2Supercontrolled` produces
    // a sequence whose global phase is off by `pi` relative to the target;
    // compensate here so the emitted sequence reproduces the target unitary
    // exactly, not just up to sign.
    globalPhase += std::numbers::pi;
  }
  // large global phases can be generated by the decomposition, thus limit
  // it to [0, +2*pi)
  globalPhase = helpers::remEuclid(globalPhase, 2.0 * std::numbers::pi);

  return TwoQubitNativeDecomposition{
      .numBasisUses = bestNbasis,
      .singleQubitFactors = std::move(factors),
      .globalPhase = globalPhase,
  };
}

// Ported SMB helpers assume Qiskit Weyl k-factor layout; QCO 4x4 input order
// swaps l/r vs that port. Swap k1l<->k1r and k2l<->k2r when reading ``target``,
// and swap adjacent pairs in each return vector so the emission boundary maps
// matrices to the same wires as the upstream decomposer. ``decomp0`` cancels to
// the unswapped formula.
TwoQubitLocalUnitaryList
TwoQubitBasisDecomposer::decomp0(const TwoQubitWeylDecomposition& target) {
  return TwoQubitLocalUnitaryList{
      target.k1r() * target.k2r(),
      target.k1l() * target.k2l(),
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp1(
    const TwoQubitWeylDecomposition& target) const {
  // may not work for z != 0 and c != 0 (not always in Weyl chamber)
  return TwoQubitLocalUnitaryList{
      basisDecomposer.k2l().adjoint() * target.k2r(),
      basisDecomposer.k2r().adjoint() * target.k2l(),
      target.k1r() * basisDecomposer.k1l().adjoint(),
      target.k1l() * basisDecomposer.k1r().adjoint(),
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp2Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with two basis gates");
  }
  return TwoQubitLocalUnitaryList{
      q2l * target.k2r(),
      q2r * target.k2l(),
      q1la * rzMatrix(-2. * target.a()) * q1lb,
      q1ra * rzMatrix(2. * target.b()) * q1rb,
      target.k1r() * q0l,
      target.k1l() * q0r,
  };
}

TwoQubitLocalUnitaryList TwoQubitBasisDecomposer::decomp3Supercontrolled(
    const TwoQubitWeylDecomposition& target) const {
  if (!isSuperControlled) {
    llvm::reportFatalInternalError(
        "Basis gate of TwoQubitBasisDecomposer is not super-controlled "
        "- no guarantee for exact decomposition with three basis gates");
  }
  return TwoQubitLocalUnitaryList{
      u3l * target.k2r(),
      u3r * target.k2l(),
      u2la * rzMatrix(-2. * target.a()) * u2lb,
      u2ra * rzMatrix(2. * target.b()) * u2rb,
      u1l,
      u1ra * rzMatrix(-2. * target.c()) * u1rb,
      target.k1r() * u0l,
      target.k1l() * u0r,
  };
}

std::array<std::complex<double>, 4>
TwoQubitBasisDecomposer::traces(const TwoQubitWeylDecomposition& target) const {
  // Returns the Hilbert-Schmidt traces between the target canonical gate and
  // the best candidate reachable with `0, 1, 2, 3` uses of the basis gate,
  // respectively. Fed into `traceToFidelity` by `getDefaultNbasis` to pick
  // the best basis-gate count. The closed-form expressions specialize
  // `TwoQubitWeylDecomposition::getTrace(a, b, c, ap, bp, cp)` for:
  //   i == 0: no basis gate       (ap == bp == cp == 0)
  //   i == 1: one basis use       (ap == pi/4, bp == basis.b, cp == 0)
  //   i == 2: two basis uses      (ap == 0, bp == 0, cp == -target.c)
  //   i == 3: three basis uses    (target reachable exactly -> trace == 4)
  // so the array has length 4 and is indexed by the number of basis uses.
  return {
      4. * std::complex<double>{std::cos(target.a()) * std::cos(target.b()) *
                                    std::cos(target.c()),
                                std::sin(target.a()) * std::sin(target.b()) *
                                    std::sin(target.c())},
      4. *
          std::complex<double>{std::cos((std::numbers::pi / 4.0) - target.a()) *
                                   std::cos(basisDecomposer.b() - target.b()) *
                                   std::cos(target.c()),
                               std::sin((std::numbers::pi / 4.0) - target.a()) *
                                   std::sin(basisDecomposer.b() - target.b()) *
                                   std::sin(target.c())},
      std::complex<double>{4. * std::cos(target.c()), 0.},
      std::complex<double>{4., 0.},
  };
}

bool TwoQubitBasisDecomposer::relativeEq(double lhs, double rhs, double epsilon,
                                         double maxRelative) {
  // Handle same infinities
  if (lhs == rhs) {
    return true;
  }

  // Handle remaining infinities
  if (std::isinf(lhs) || std::isinf(rhs)) {
    return false;
  }

  auto absDiff = std::abs(lhs - rhs);

  // For when the numbers are really close together
  if (absDiff <= epsilon) {
    return true;
  }

  auto absLhs = std::abs(lhs);
  auto absRhs = std::abs(rhs);
  if (absRhs > absLhs) {
    return absDiff <= absRhs * maxRelative;
  }
  return absDiff <= absLhs * maxRelative;
}

} // namespace mlir::qco::decomposition
