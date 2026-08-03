/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Decomposition/SynthesisBasis.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/ErrorHandling.h>

#include <numbers>

namespace mlir::qco::decomposition {

using GateKind = CompilerTarget::GateKind;

[[nodiscard]] static EulerBasis
toEulerBasis(const CompilerTarget::SingleQubitBasis basis) {
  switch (basis) {
  case CompilerTarget::SingleQubitBasis::U:
    return EulerBasis::U;
  case CompilerTarget::SingleQubitBasis::ZSXX:
    return EulerBasis::ZSXX;
  case CompilerTarget::SingleQubitBasis::R:
    return EulerBasis::R;
  case CompilerTarget::SingleQubitBasis::XZX:
    return EulerBasis::XZX;
  case CompilerTarget::SingleQubitBasis::XYX:
    return EulerBasis::XYX;
  case CompilerTarget::SingleQubitBasis::ZYZ:
    return EulerBasis::ZYZ;
  }
  llvm_unreachable("unhandled compiler target single-qubit basis");
}

static constexpr Matrix4x4 CANONICAL_CONTROLLED_X =
    Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0,  // row 0
                            0.0, 1.0, 0.0, 0.0,  // row 1
                            0.0, 0.0, 0.0, 1.0,  // row 2
                            0.0, 0.0, 1.0, 0.0); // row 3

static constexpr Matrix4x4 CANONICAL_CONTROLLED_Z =
    Matrix4x4::fromDiagonal(1., 1., 1., -1.);

static const TwoQubitBasisDecomposer&
cachedNativeBasisDecomposer(const GateKind entangler) {
  switch (entangler) {
  case GateKind::RXX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RXXOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case GateKind::RYY: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RYYOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case GateKind::RZX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RZXOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case GateKind::RZZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(
            RZZOp::unitaryMatrix(std::numbers::pi / 2.0), 1.0);
    return DECOMPOSER;
  }
  case GateKind::ISWAP: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(iSWAPOp::getUnitaryMatrix(), 1.0);
    return DECOMPOSER;
  }
  case GateKind::CZ: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_Z, 1.0);
    return DECOMPOSER;
  }
  case GateKind::CX: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(CANONICAL_CONTROLLED_X, 1.0);
    return DECOMPOSER;
  }
  case GateKind::ECR: {
    static const TwoQubitBasisDecomposer DECOMPOSER =
        TwoQubitBasisDecomposer::create(ECROp::getUnitaryMatrix(), 1.0);
    return DECOMPOSER;
  }
  default:
    llvm_unreachable(
        "only RXX/RYY/RZX/RZZ/ISWAP/CZ/CX/ECR are valid entanglers");
  }
}

TwoQubitNativeDecomposition
NativeSynthesisBasis::decomposeTarget(const Matrix4x4& target) const {
  const auto decomposition =
      cachedNativeBasisDecomposer(entangler).decomposeTarget(target);
  if (!decomposition) {
    llvm::reportFatalInternalError(
        "target-selected entangler failed to decompose a two-qubit unitary");
  }
  return *decomposition;
}

NativeSynthesisBasis NativeSynthesisBasis::fromCompilerTarget(
    const CompilerTarget::SynthesisBasis basis) {
  return NativeSynthesisBasis{.singleQubit = toEulerBasis(basis.singleQubit),
                              .entangler = basis.entangler};
}

} // namespace mlir::qco::decomposition
