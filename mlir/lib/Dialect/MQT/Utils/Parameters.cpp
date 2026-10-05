/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::mqt {

std::optional<double>
parameterToConstantDouble(const FloatParameter& parameter) {
  if (const auto* scalar = std::get_if<double>(&parameter)) {
    return *scalar;
  }
  auto value = std::get<Value>(parameter);
  assert(value && isa<Float64Type>(value.getType()) && "expected an f64 value");
  return valueToConstantDouble(value);
}

FloatParameter foldParameter(const FloatParameter& parameter) {
  if (const auto value = parameterToConstantDouble(parameter)) {
    return *value;
  }
  return parameter;
}

FloatParameter addParameters(OpBuilder& builder, Location loc,
                             const FloatParameter& lhs,
                             const FloatParameter& rhs) {
  const auto foldedLhs = foldParameter(lhs);
  const auto foldedRhs = foldParameter(rhs);
  const auto* a = std::get_if<double>(&foldedLhs);
  const auto* b = std::get_if<double>(&foldedRhs);
  if (a != nullptr && b != nullptr) {
    return *a + *b;
  }
  return (FloatExpression(builder, loc, foldedLhs) +
          FloatExpression(builder, loc, foldedRhs))
      .getValue();
}

FloatParameter scaleParameter(OpBuilder& builder, Location loc,
                              const FloatParameter& parameter, double scale) {
  const auto folded = foldParameter(parameter);
  if (const auto* value = std::get_if<double>(&folded)) {
    return *value * scale;
  }
  return (FloatExpression(builder, loc, folded) *
          FloatExpression::constant(builder, loc, scale))
      .getValue();
}

FloatExpression::FloatExpression(OpBuilder& builder, Location loc, Value value)
    : value_(value), builder_(&builder), loc_(loc) {
  assert(value && isa<Float64Type>(value.getType()) && "expected an f64 value");
  assert(value.getContext() == builder.getContext() &&
         "expected a value from the builder's context");
}

FloatExpression::FloatExpression(OpBuilder& builder, Location loc,
                                 const FloatParameter& parameter)
    : FloatExpression(builder, loc, variantToValue(builder, loc, parameter)) {}

FloatExpression FloatExpression::withValue(Value value) const {
  return {*builder_, loc_, value};
}

void FloatExpression::assertSameBuilder(FloatExpression rhs) const {
  assert(builder_ == rhs.builder_ &&
         "expected expressions from the same builder");
}

FloatExpression FloatExpression::constant(OpBuilder& builder, Location loc,
                                          double value) {
  return {builder, loc, constantFromScalar(builder, loc, value)};
}

FloatExpression FloatExpression::operator+(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return withValue(
      builder_->createOrFold<arith::AddFOp>(loc_, value_, rhs.value_));
}

FloatExpression FloatExpression::operator-(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return withValue(
      builder_->createOrFold<arith::SubFOp>(loc_, value_, rhs.value_));
}

FloatExpression FloatExpression::operator*(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return withValue(
      builder_->createOrFold<arith::MulFOp>(loc_, value_, rhs.value_));
}

FloatExpression FloatExpression::operator/(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return withValue(
      builder_->createOrFold<arith::DivFOp>(loc_, value_, rhs.value_));
}

FloatExpression FloatExpression::operator-() const {
  return withValue(builder_->createOrFold<arith::NegFOp>(loc_, value_));
}

FloatExpression FloatExpression::sin() const {
  return withValue(builder_->createOrFold<math::SinOp>(loc_, value_));
}

FloatExpression FloatExpression::cos() const {
  return withValue(builder_->createOrFold<math::CosOp>(loc_, value_));
}

FloatExpression FloatExpression::tan() const {
  return withValue(builder_->createOrFold<math::TanOp>(loc_, value_));
}

FloatExpression FloatExpression::atan() const {
  return withValue(builder_->createOrFold<math::AtanOp>(loc_, value_));
}

FloatExpression FloatExpression::abs() const {
  return withValue(builder_->createOrFold<math::AbsFOp>(loc_, value_));
}

FloatExpression FloatExpression::floor() const {
  return withValue(builder_->createOrFold<math::FloorOp>(loc_, value_));
}

FloatExpression FloatExpression::sqrt() const {
  return withValue(builder_->createOrFold<math::SqrtOp>(loc_, value_));
}

FloatExpression FloatExpression::pow(FloatExpression exponent) const {
  assertSameBuilder(exponent);
  return withValue(
      builder_->createOrFold<math::PowFOp>(loc_, value_, exponent.value_));
}

FloatExpression FloatExpression::atan2(FloatExpression x) const {
  assertSameBuilder(x);
  return withValue(
      builder_->createOrFold<math::Atan2Op>(loc_, value_, x.value_));
}

Value FloatExpression::oge(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return builder_->createOrFold<arith::CmpFOp>(loc_, arith::CmpFPredicate::OGE,
                                               value_, rhs.value_);
}

Value FloatExpression::olt(FloatExpression rhs) const {
  assertSameBuilder(rhs);
  return builder_->createOrFold<arith::CmpFOp>(loc_, arith::CmpFPredicate::OLT,
                                               value_, rhs.value_);
}

FloatExpression FloatExpression::select(Value condition,
                                        FloatExpression trueValue,
                                        FloatExpression falseValue) {
  trueValue.assertSameBuilder(falseValue);
  assert(condition && condition.getType().isInteger(1) &&
         "expected an i1 value");
  assert(condition.getContext() == trueValue.builder_->getContext() &&
         "expected a condition from the builder's context");
  return trueValue.withValue(trueValue.builder_->createOrFold<arith::SelectOp>(
      trueValue.loc_, condition, trueValue.value_, falseValue.value_));
}

Value constantFromScalar(OpBuilder& builder, Location loc, const double value) {
  return arith::ConstantOp::create(builder, loc,
                                   builder.getF64FloatAttr(value));
}

Value constantFromScalar(OpBuilder& builder, Location loc,
                         const int64_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexAttr(value));
}

Value constantFromScalar(OpBuilder& builder, Location loc, const bool value) {
  return arith::ConstantOp::create(builder, loc, builder.getBoolAttr(value));
}

LogicalResult verifyFiniteConstantParameters(Operation* operation,
                                             ValueRange parameters) {
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    Attribute constant;
    if (matchPattern(parameter, m_Constant(&constant))) {
      if (auto floating = dyn_cast<FloatAttr>(constant);
          floating && !floating.getValue().isFinite()) {
        return operation->emitOpError()
               << "constant parameter expression at index " << index
               << " must be finite";
      }
    }
  }
  return success();
}

} // namespace mlir::mqt
