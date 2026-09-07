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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>

namespace mlir {
class RewritePatternSet;
namespace mqt {

/// Expand bounded integer rotations and population count for targets without
/// them.
void populateIntegerExpansionPatterns(RewritePatternSet& patterns);

/// Maps signed ordering to the corresponding unsigned predicate.
inline arith::CmpIPredicate unsignedPredicate(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::uge;
  default:
    return predicate;
  }
}

/// Builds a logical shift with unsigned distance and zero on overshift.
/// Check before narrowing the distance and keep even unselected shifts valid
/// for interpreters that evaluate SSA operations eagerly.
inline Value buildZeroFillingShift(OpBuilder& builder, Location location,
                                   Value value, Value distance, bool left) {
  auto type = cast<IntegerType>(value.getType());
  auto distanceType = cast<IntegerType>(distance.getType());
  const auto width = type.getWidth();
  const auto constant = [&](const llvm::APInt& bits) -> Value {
    return arith::ConstantOp::create(builder, location,
                                     builder.getIntegerAttr(type, bits));
  };
  const auto shift = [&](Value amount) -> Value {
    if (left) {
      return arith::ShLIOp::create(builder, location, value, amount);
    }
    return arith::ShRUIOp::create(builder, location, value, amount);
  };
  llvm::APInt amount;
  if (matchPattern(distance, m_ConstantInt(&amount))) {
    if (amount.uge(width)) {
      return constant(llvm::APInt(width, 0));
    }
    if (amount.isZero()) {
      return value;
    }
    return shift(constant(amount.zextOrTrunc(width)));
  }
  Value inRange;
  if (llvm::APInt::getMaxValue(distanceType.getWidth()).uge(width)) {
    auto limit = arith::ConstantOp::create(
        builder, location, builder.getIntegerAttr(distanceType, width));
    inRange = arith::CmpIOp::create(builder, location,
                                    arith::CmpIPredicate::ult, distance, limit);
  }
  if (distanceType.getWidth() < width) {
    distance = arith::ExtUIOp::create(builder, location, type, distance);
  } else if (distanceType.getWidth() > width) {
    distance = arith::TruncIOp::create(builder, location, type, distance);
  }
  if (!inRange) {
    return shift(distance);
  }
  auto zero = constant(llvm::APInt(width, 0));
  auto safe =
      arith::SelectOp::create(builder, location, inRange, distance, zero);
  return arith::SelectOp::create(builder, location, inRange, shift(safe), zero);
}

} // namespace mqt
} // namespace mlir
