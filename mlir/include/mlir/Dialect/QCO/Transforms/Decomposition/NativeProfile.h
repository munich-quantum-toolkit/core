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
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>

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
  CX,
  CZ,
};

/**
 * @brief Resolved native gateset for two-qubit Weyl synthesis.
 *
 * @p gates is the parsed gateset. Euler decomposition and entangler choice are
 * derived from it with fixed priority (see @ref NativeProfileSpec::eulerBasis
 * and @ref NativeProfileSpec::parse). Gatesets must include a supported
 * single-qubit strategy and at least one of `cx` or `cz` (when both are
 * present, `cx` is preferred).
 */
struct NativeProfileSpec {
  llvm::DenseSet<NativeGateKind> gates;

  /**
   * @brief Preferred single-qubit Euler basis for synthesis in this gateset.
   *
   * Only valid for specs returned by @ref parse.
   */
  [[nodiscard]] EulerBasis eulerBasis() const;

  /**
   * @brief Parses a comma-separated native gateset (e.g. `"u,cx"`).
   *
   * @param nativeGates Comma-separated gate tokens.
   * @return Parsed profile, or `std::nullopt` when the gateset is unsupported.
   */
  [[nodiscard]] static std::optional<NativeProfileSpec>
  parse(StringRef nativeGates);
};

/** @brief Synthesizes a two-qubit unitary as gates allowed by @p spec. */
[[nodiscard]] LogicalResult
synthesizeUnitary2QWeyl(OpBuilder& builder, Location loc, Value qubit0,
                        Value qubit1, const Matrix4x4& target,
                        const NativeProfileSpec& spec, Value& outQubit0,
                        Value& outQubit1);

/**
 * @brief Entangling basis gates needed to synthesize @p target under @p spec.
 *
 * @return Count for @p spec, or `std::nullopt` when synthesis is impossible.
 */
[[nodiscard]] std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec);

/** @brief Returns true when @p op is already in the resolved native gateset. */
[[nodiscard]] bool allowsOp(Operation* op, const NativeProfileSpec& spec);

} // namespace mlir::qco::decomposition
