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

#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/DenseSet.h>

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::qco::decomposition {

/**
 * @brief Gate token in a comma-separated native gateset (e.g. `"u,cx"`).
 */
enum class NativeGateKind : std::uint8_t {
  U,
  X,
  SX,
  RZ,
  RX,
  RY,
  R,
  RXX,
  RYY,
  RZX,
  RZZ,
  ISWAP,
  CZ,
  CX,
  ECR,
};

struct TwoQubitNativeDecomposition;

/**
 * @brief Resolved native gateset for two-qubit Weyl synthesis.
 *
 * Use @ref parse to obtain a gateset with @p eulerBasis and @p entangler
 * resolved from @p gates. When several entanglers appear, preference is
 * **RXX > RYY > RZX > RZZ > iSWAP > CZ > CX > ECR** (alphabetic among two-qubit
 * rotations; then discrete named gates; ECR last). Weyl synthesis emits
 * `rxx`/`ryy`/`rzx`/`rzz` at a fixed angle of π/2.
 */
struct NativeGateset {
  llvm::DenseSet<NativeGateKind> gates;
  std::optional<EulerBasis> eulerBasis;
  std::optional<NativeGateKind> entangler;

  /**
   * @brief Parses a comma-separated native gateset (e.g. `"u,cx"`).
   *
   * @param nativeGates Comma-separated gate tokens.
   * @return Parsed gateset, or `std::nullopt` when the gateset is unsupported.
   */
  [[nodiscard]] static std::optional<NativeGateset>
  parse(StringRef nativeGates);

  /**
   * @brief Basis decomposition of @p target under this gateset, if supported.
   */
  [[nodiscard]] std::optional<TwoQubitNativeDecomposition>
  decomposeTarget(const Matrix4x4& target) const;

  /**
   * @brief Whether @p op is already on this native gateset.
   *
   * `qco.barrier` and `qco.gphase` are always allowed. Single-qubit primitives
   * are checked against @p gates. Single-control, single-target `qco.ctrl`
   * shells with an `X`/`Z` body are accepted when `cx`/`cz` is present.
   * `qco.rxx`, `qco.ryy`, `qco.rzx`, and `qco.rzz` are accepted when the
   * corresponding token is present, including runtime-parameterized forms.
   * Bare `qco.iswap` and `qco.ecr` are accepted when the corresponding token is
   * present. All other ops are rejected.
   */
  [[nodiscard]] bool allowsOp(Operation* op) const;
};

} // namespace mlir::qco::decomposition
