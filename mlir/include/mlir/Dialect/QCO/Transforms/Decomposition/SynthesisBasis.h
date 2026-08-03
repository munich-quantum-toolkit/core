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

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

namespace mlir::qco::decomposition {

struct TwoQubitNativeDecomposition;

/**
 * @brief Typed Euler/Weyl basis adapted from one compiler target.
 *
 * This value contains only the transformation-specific adaptation needed by
 * the decomposers. Gate capabilities and basis selection remain owned by
 * @ref CompilerTarget.
 */
struct NativeSynthesisBasis {
  EulerBasis singleQubit;
  CompilerTarget::GateKind entangler;

  /// Adapt one target-selected synthesis basis to the decomposition layer.
  [[nodiscard]] static NativeSynthesisBasis
  fromCompilerTarget(CompilerTarget::SynthesisBasis basis);

  /**
   * @brief Basis decomposition of @p target under this synthesis basis.
   */
  [[nodiscard]] TwoQubitNativeDecomposition
  decomposeTarget(const Matrix4x4& target) const;
};

} // namespace mlir::qco::decomposition
