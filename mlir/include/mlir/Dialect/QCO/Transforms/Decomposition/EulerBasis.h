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

#include "GateKind.h"

#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace mlir::qco::decomposition {
/**
 * Default absolute tolerance used to treat small Euler angles as zero during
 * simplification.
 */
inline constexpr auto DEFAULT_ATOL = 1e-12;

/**
 * Supported single-qubit Euler-style output bases.
 *
 * The listed values describe the gate alphabet that `EulerDecomposition`
 * targets when converting a 2x2 unitary into a `OneQubitGateSequence`.
 * Several entries share the angle-extraction routine and only differ in how
 * the final circuit is emitted (e.g. `U3` vs `U321`, or `ZSX` vs `ZSXX`).
 */
enum class EulerBasis : std::uint8_t {
  U3 = 0,   ///< Single `u(theta, phi, lambda)` gate.
  U321 = 1, ///< `u1`/`u2`/`u3` family — picks the smallest form per angles.
  U = 2,    ///< Same ZYZ angle extraction as `U3`, emitted as a single `u`.
  ZYZ = 3,  ///< `rz · ry · rz`.
  ZXZ = 4,  ///< `rz · rx · rz`.
  XZX = 5,  ///< `rx · rz · rx`.
  XYX = 6,  ///< `rx · ry · rx`.
  ZSXX = 7, ///< `rz · sx` chain, with `sx · rz(+/- pi) · sx` collapsed to `x`.
  ZSX = 8,  ///< Like `ZSXX` but without the `x` shortcut.
};

/**
 * Return the gate types that may appear in a circuit emitted for `eulerBasis`.
 *
 * The result describes the basis alphabet, not the exact gate count. Some
 * decompositions emit fewer than three gates after simplification.
 */
[[nodiscard]] llvm::SmallVector<GateKind, 3>
getGateTypesForEulerBasis(EulerBasis eulerBasis);

} // namespace mlir::qco::decomposition
