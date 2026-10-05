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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::mqt {

/// Absolute tolerance used when comparing static operation parameters.
inline constexpr double PARAMETER_COMPARISON_TOLERANCE = 1e-15;

/// A host double or an existing f64 SSA value. This alias preserves the
/// scalar-or-SSA argument type used by program builders.
using FloatParameter = std::variant<double, Value>;

/// Evaluate a host double or a constant SSA expression with the shared folder.
[[nodiscard]] std::optional<double>
parameterToConstantDouble(const FloatParameter& parameter);

/// Fold a known constant parameter to a host double, preserving unknown SSA
/// values. This does not emit IR or apply angle policies.
[[nodiscard]] FloatParameter foldParameter(const FloatParameter& parameter);

/// Fold scalar arithmetic without angle wrapping or approximate identities.
/// Known operands stay in host arithmetic; unknown operands use createOrFold.
/// SSA operands must dominate the builder insertion point.
[[nodiscard]] FloatParameter addParameters(OpBuilder& builder, Location loc,
                                           const FloatParameter& lhs,
                                           const FloatParameter& rhs);

/// Scale a parameter with the same folding and dominance contract as
/// addParameters. A zero factor does not discard an unknown operand.
[[nodiscard]] FloatParameter scaleParameter(OpBuilder& builder, Location loc,
                                            const FloatParameter& parameter,
                                            double scale);

/// Build f64 SSA arithmetic with MLIR's local operation folders.
///
/// The borrowed builder must outlive the expressions. Operations use its
/// current insertion point; operands must dominate it and use the same builder.
/// Load the arith and math dialects before emitting their operations. This
/// wrapper neither traverses expression DAGs nor applies angle policies.
class FloatExpression {
  Value value_;
  OpBuilder* builder_;
  Location loc_;

  [[nodiscard]] FloatExpression withValue(Value value) const;
  void assertSameBuilder(FloatExpression rhs) const;

public:
  FloatExpression(OpBuilder& builder, Location loc, Value value);
  /// Materialize a host scalar or preserve an existing f64 SSA value.
  FloatExpression(OpBuilder& builder, Location loc,
                  const FloatParameter& parameter);

  [[nodiscard]] static FloatExpression constant(OpBuilder& builder,
                                                Location loc, double value);
  [[nodiscard]] Value getValue() const { return value_; }
  [[nodiscard]] OpBuilder& getBuilder() const { return *builder_; }
  [[nodiscard]] Location getLoc() const { return loc_; }

  [[nodiscard]] FloatExpression operator+(FloatExpression rhs) const;
  [[nodiscard]] FloatExpression operator-(FloatExpression rhs) const;
  [[nodiscard]] FloatExpression operator*(FloatExpression rhs) const;
  [[nodiscard]] FloatExpression operator/(FloatExpression rhs) const;
  [[nodiscard]] FloatExpression operator-() const;
  [[nodiscard]] FloatExpression sin() const;
  [[nodiscard]] FloatExpression cos() const;
  [[nodiscard]] FloatExpression tan() const;
  [[nodiscard]] FloatExpression atan() const;
  [[nodiscard]] FloatExpression abs() const;
  [[nodiscard]] FloatExpression floor() const;
  [[nodiscard]] FloatExpression sqrt() const;
  [[nodiscard]] FloatExpression pow(FloatExpression exponent) const;
  [[nodiscard]] FloatExpression atan2(FloatExpression x) const;

  /// Ordered floating-point comparisons, returning i1 SSA values.
  [[nodiscard]] Value oge(FloatExpression rhs) const;
  [[nodiscard]] Value olt(FloatExpression rhs) const;
  [[nodiscard]] static FloatExpression select(Value condition,
                                              FloatExpression trueValue,
                                              FloatExpression falseValue);
};

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       double value);

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       int64_t value);

/// Materialize a scalar as an arithmetic constant.
[[nodiscard]] Value constantFromScalar(OpBuilder& builder, Location loc,
                                       bool value);

/// Preserve an existing SSA value, or materialize the scalar with the caller's
/// dialect-specific constant builder. Existing values do not invoke
/// materialize.
template <typename T, typename Materialize>
[[nodiscard]] Value variantToValue(const std::variant<T, Value>& parameter,
                                   Materialize materialize) {
  if (const auto* value = std::get_if<Value>(&parameter)) {
    return *value;
  }
  return materialize(std::get<T>(parameter));
}

/// Convert a scalar or existing SSA value to an arithmetic SSA value.
template <typename T>
[[nodiscard]] Value variantToValue(OpBuilder& builder, Location loc,
                                   const std::variant<T, Value>& parameter) {
  return variantToValue(parameter, [&](T value) {
    return constantFromScalar(builder, loc, value);
  });
}

/// Verify that direct floating-point constant parameters are finite.
[[nodiscard]] LogicalResult
verifyFiniteConstantParameters(Operation* operation, ValueRange parameters);

} // namespace mlir::mqt
