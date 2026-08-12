/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Utils/AngleConversion.h"

#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::mqt::angle {

static constexpr double TWO_PI = 2.0 * std::numbers::pi;
static constexpr uint64_t TWO_PI_BITS = 0x401921FB54442D18ULL;
static constexpr uint64_t TWO_PI_ODD_SIGNIFICAND = 0x3243F6A8885A3ULL;
static constexpr uint64_t DOUBLE_FRACTION_MASK = (uint64_t{1} << 52U) - 1U;
static constexpr uint64_t DOUBLE_MAGNITUDE_MASK =
    std::numeric_limits<int64_t>::max();
static constexpr uint64_t DOUBLE_EXPONENT_MASK = 0x7FFU;
static constexpr uint32_t DIVISION_CHUNK_WIDTH = 13;
static constexpr uint32_t DIVISION_CHUNK_COUNT = 5;
static constexpr StringLiteral FLOAT_TO_BITS_ATTR =
    "mqt.openqasm.float_to_angle";

static_assert(std::bit_cast<uint64_t>(TWO_PI) == TWO_PI_BITS);

[[nodiscard]] static Value constantF64(OpBuilder& builder, Location loc,
                                       const double value) {
  return arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                        llvm::APFloat(value));
}

[[nodiscard]] static bool isExpectedScale(Value value,
                                          const uint32_t bitWidth) {
  llvm::APFloat scale(0.0);
  if (!matchPattern(value, m_ConstantFloat(&scale))) {
    return false;
  }
  return scale.convertToDouble() ==
         std::ldexp(TWO_PI, -static_cast<int>(bitWidth));
}

[[nodiscard]] static bool constantEquals(const Value value,
                                         const llvm::APInt& expected) {
  llvm::APInt actual;
  return matchPattern(value, m_ConstantInt(&actual)) &&
         actual == expected.zextOrTrunc(actual.getBitWidth());
}

bool isSupportedWidth(const uint32_t bitWidth) {
  return bitWidth >= 1 && bitWidth <= MACHINE_WIDTH;
}

[[nodiscard]] static llvm::APInt
roundUnsignedQuotient(const llvm::APInt& numerator,
                      const llvm::APInt& denominator) {
  const auto quotient = numerator.udiv(denominator);
  const auto remainder = numerator.urem(denominator);
  const auto twiceRemainder = remainder.shl(1U);
  const auto roundUp = twiceRemainder.ugt(denominator) ||
                       (twiceRemainder == denominator && quotient[0]);
  return roundUp ? quotient + 1U : quotient;
}

[[nodiscard]] static uint64_t quantizeMagnitude(const uint64_t significand,
                                                const int32_t binaryExponent,
                                                const uint32_t bitWidth) {
  constexpr unsigned workingWidth = 128;
  const llvm::APInt modulus(workingWidth, TWO_PI_ODD_SIGNIFICAND);
  const llvm::APInt magnitude(workingWidth, significand);
  llvm::APInt rounded(workingWidth, 0);

  if (binaryExponent >= 0) {
    auto power = llvm::APInt(workingWidth, 2);
    auto factor = llvm::APInt(workingWidth, 1);
    auto exponent = static_cast<uint32_t>(binaryExponent);
    while (exponent != 0) {
      if ((exponent & 1U) != 0) {
        factor = (factor * power).urem(modulus);
      }
      power = (power * power).urem(modulus);
      exponent >>= 1U;
    }
    const auto remainder = (magnitude.urem(modulus) * factor).urem(modulus);
    rounded = roundUnsignedQuotient(remainder.shl(bitWidth), modulus);
  } else {
    const auto scaledExponent = binaryExponent + static_cast<int32_t>(bitWidth);
    if (scaledExponent >= 0) {
      rounded = roundUnsignedQuotient(
          magnitude.shl(static_cast<unsigned>(scaledExponent)), modulus);
    } else {
      const auto denominatorShift = static_cast<unsigned>(-scaledExponent);
      // The numerator has at most 53 bits and the odd denominator has 50.
      // Five or more additional denominator bits are already strictly below
      // the half-way point, including the largest binary64 significand.
      if (denominatorShift >= 5U) {
        return 0;
      }
      rounded = roundUnsignedQuotient(magnitude, modulus.shl(denominatorShift));
    }
  }

  return rounded.trunc(bitWidth).getZExtValue();
}

std::optional<uint64_t> quantize(const double radians,
                                 const uint32_t bitWidth) {
  if (!isSupportedWidth(bitWidth) || !std::isfinite(radians)) {
    return std::nullopt;
  }

  const auto representation = std::bit_cast<uint64_t>(radians);
  const auto exponent =
      static_cast<uint32_t>((representation >> 52U) & DOUBLE_EXPONENT_MASK);
  const auto fraction = representation & DOUBLE_FRACTION_MASK;
  if (exponent == 0 && fraction == 0) {
    return 0;
  }

  const auto significand =
      exponent == 0 ? fraction : fraction | (uint64_t{1} << 52U);
  // TWO_PI_ODD_SIGNIFICAND * 2^-47 is exactly the binary64 value of
  // 2*pi. A normal input is significand * 2^(exponent-1075), while a
  // subnormal input is fraction * 2^-1074.
  const auto binaryExponent =
      exponent == 0 ? -1027 : static_cast<int32_t>(exponent) - 1028;
  auto result = quantizeMagnitude(significand, binaryExponent, bitWidth);
  if ((representation >> 63U) != 0) {
    result = uint64_t{0} - result;
    if (bitWidth != MACHINE_WIDTH) {
      result &= (uint64_t{1} << bitWidth) - 1U;
    }
  }
  return result;
}

double toRadians(const uint64_t bits, const uint32_t bitWidth) {
  assert(isSupportedWidth(bitWidth) && "unsupported angle bit width");
  return static_cast<double>(bits) *
         std::ldexp(TWO_PI, -static_cast<int>(bitWidth));
}

uint64_t resize(const uint64_t bits, const uint32_t sourceWidth,
                const uint32_t targetWidth) {
  assert(isSupportedWidth(sourceWidth) && isSupportedWidth(targetWidth) &&
         "angle resize requires supported widths");
  const auto mask = [](const uint32_t width) {
    return width == MACHINE_WIDTH ? std::numeric_limits<uint64_t>::max()
                                  : (uint64_t{1} << width) - 1U;
  };
  if (sourceWidth == targetWidth) {
    return bits & mask(targetWidth);
  }
  if (sourceWidth < targetWidth) {
    return (bits << (targetWidth - sourceWidth)) & mask(targetWidth);
  }
  const auto discardedWidth = sourceWidth - targetWidth;
  auto retained = bits >> discardedWidth;
  const auto discarded = bits & ((uint64_t{1} << discardedWidth) - 1U);
  const auto halfway = uint64_t{1} << (discardedWidth - 1U);
  if (discarded > halfway || (discarded == halfway && (retained & 1U) != 0)) {
    ++retained;
  }
  return retained & mask(targetWidth);
}

[[nodiscard]] static Value integerConstant(OpBuilder& builder,
                                           const Location loc,
                                           const IntegerType type,
                                           const uint64_t value) {
  return arith::ConstantOp::create(
      builder, loc,
      IntegerAttr::get(type, llvm::APInt(type.getWidth(), value)));
}

namespace {

struct UnsignedDivision {
  Value quotient;
  Value remainder;
};

} // namespace

[[nodiscard]] static Value
roundUnsignedDivision(OpBuilder& builder, const Location loc,
                      const UnsignedDivision& division,
                      const Value denominator) {
  const auto type = cast<IntegerType>(division.quotient.getType());
  auto one = integerConstant(builder, loc, type, 1);
  auto twiceRemainder =
      arith::ShLIOp::create(builder, loc, division.remainder, one);
  auto greater = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                       twiceRemainder, denominator);
  auto equal = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                     twiceRemainder, denominator);
  auto quotientLsb =
      arith::AndIOp::create(builder, loc, division.quotient, one);
  auto odd =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, quotientLsb,
                            integerConstant(builder, loc, type, 0));
  auto tiedOdd = arith::AndIOp::create(builder, loc, equal, odd);
  auto roundUp = arith::OrIOp::create(builder, loc, greater, tiedOdd);
  auto increment = arith::ExtUIOp::create(builder, loc, type, roundUp);
  return arith::AddIOp::create(builder, loc, division.quotient, increment);
}

[[nodiscard]] static Value roundUnsignedQuotient(OpBuilder& builder,
                                                 const Location loc,
                                                 const Value numerator,
                                                 const Value denominator) {
  return roundUnsignedDivision(
      builder, loc,
      UnsignedDivision{.quotient = arith::DivUIOp::create(
                           builder, loc, numerator, denominator),
                       .remainder = arith::RemUIOp::create(
                           builder, loc, numerator, denominator)},
      denominator);
}

[[nodiscard]] static UnsignedDivision
appendQuotientBits(OpBuilder& builder, const Location loc,
                   const UnsignedDivision& division, const Value modulus,
                   const Value scaledExponent, const uint32_t offset) {
  const auto type = cast<IntegerType>(division.quotient.getType());
  const auto zero = integerConstant(builder, loc, type, 0);
  Value remaining =
      arith::SubIOp::create(builder, loc, scaledExponent,
                            integerConstant(builder, loc, type, offset));
  auto positive = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                        remaining, zero);
  remaining = arith::SelectOp::create(builder, loc, positive, remaining, zero);
  auto fitsChunk = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, remaining,
      integerConstant(builder, loc, type, DIVISION_CHUNK_WIDTH));
  auto shift = arith::SelectOp::create(
      builder, loc, fitsChunk, remaining,
      integerConstant(builder, loc, type, DIVISION_CHUNK_WIDTH));
  auto partialNumerator =
      arith::ShLIOp::create(builder, loc, division.remainder, shift);
  auto digit = arith::DivUIOp::create(builder, loc, partialNumerator, modulus);
  auto remainder =
      arith::RemUIOp::create(builder, loc, partialNumerator, modulus);
  auto quotient = arith::OrIOp::create(
      builder, loc,
      arith::ShLIOp::create(builder, loc, division.quotient, shift), digit);
  return {.quotient = quotient, .remainder = remainder};
}

Value buildFloatToBits(OpBuilder& builder, const Location loc, Value radians,
                       const uint32_t bitWidth) {
  assert(isSupportedWidth(bitWidth) && "unsupported angle bit width");
  assert(radians.getType().isF64() && "angle source must be f64");
  if (const auto constant = utils::valueToConstantDouble(radians)) {
    const auto bits = quantize(*constant, bitWidth);
    assert(bits && "non-finite constants must be rejected before lowering");
    return arith::ConstantOp::create(
        builder, loc,
        IntegerAttr::get(builder.getIntegerType(bitWidth),
                         llvm::APInt(bitWidth, *bits, /*isSigned=*/false)));
  }

  const auto i64Type = builder.getI64Type();
  const auto zero64 = integerConstant(builder, loc, i64Type, 0);
  auto originalRepresentation =
      arith::BitcastOp::create(builder, loc, i64Type, radians);
  auto magnitudeRepresentation = arith::AndIOp::create(
      builder, loc, originalRepresentation,
      integerConstant(builder, loc, i64Type, DOUBLE_MAGNITUDE_MASK));
  auto magnitudeRadians = arith::BitcastOp::create(
      builder, loc, builder.getF64Type(), magnitudeRepresentation);
  // Binary64 2*pi is M*2^-47 for the 50-bit odd integer M above. Reducing the
  // magnitude first is exact: the remainder is smaller than 2*pi and remains
  // representable as a binary64 dyadic fraction. It also bounds the subsequent
  // integer quotient to 64 bits.
  auto reducedRadians = arith::RemFOp::create(
      builder, loc, magnitudeRadians, constantF64(builder, loc, TWO_PI));
  auto representation =
      arith::BitcastOp::create(builder, loc, i64Type, reducedRadians);
  auto exponent = arith::AndIOp::create(
      builder, loc,
      arith::ShRUIOp::create(builder, loc, representation,
                             integerConstant(builder, loc, i64Type, 52)),
      integerConstant(builder, loc, i64Type, DOUBLE_EXPONENT_MASK));
  auto fraction = arith::AndIOp::create(
      builder, loc, representation,
      integerConstant(builder, loc, i64Type, DOUBLE_FRACTION_MASK));
  auto isSubnormal = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, exponent, zero64);
  auto normalSignificand = arith::OrIOp::create(
      builder, loc, fraction,
      integerConstant(builder, loc, i64Type, uint64_t{1} << 52U));
  auto significand = arith::SelectOp::create(builder, loc, isSubnormal,
                                             fraction, normalSignificand);
  auto normalExponent = arith::SubIOp::create(
      builder, loc, exponent, integerConstant(builder, loc, i64Type, 1028));
  auto binaryExponent = arith::SelectOp::create(
      builder, loc, isSubnormal,
      integerConstant(builder, loc, i64Type,
                      static_cast<uint64_t>(int64_t{-1027})),
      normalExponent);
  const auto modulus =
      integerConstant(builder, loc, i64Type, TWO_PI_ODD_SIGNIFICAND);
  UnsignedDivision positiveDivision{
      .quotient = arith::DivUIOp::create(builder, loc, significand, modulus),
      .remainder = arith::RemUIOp::create(builder, loc, significand, modulus)};

  auto scaledExponent =
      arith::AddIOp::create(builder, loc, binaryExponent,
                            integerConstant(builder, loc, i64Type, bitWidth));
  auto scaledExponentIsNonnegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, scaledExponent, zero64);
  // The reduced value needs at most 61 appended quotient bits. Five 13-bit
  // radix steps suffice, and (remainder << 13) stays below 2^63 because the
  // modulus is smaller than 2^50.
  for (uint32_t chunk = 0; chunk < DIVISION_CHUNK_COUNT; ++chunk) {
    positiveDivision =
        appendQuotientBits(builder, loc, positiveDivision, modulus,
                           scaledExponent, chunk * DIVISION_CHUNK_WIDTH);
  }
  auto positiveRounded =
      roundUnsignedDivision(builder, loc, positiveDivision, modulus);

  Value denominatorShift =
      arith::SubIOp::create(builder, loc, zero64, scaledExponent);
  auto denominatorShiftAtMostFour = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ule, denominatorShift,
      integerConstant(builder, loc, i64Type, 4));
  denominatorShift = arith::SelectOp::create(
      builder, loc, denominatorShiftAtMostFour, denominatorShift,
      integerConstant(builder, loc, i64Type, 4));
  auto denominator =
      arith::ShLIOp::create(builder, loc, modulus, denominatorShift);
  Value negativeRounded =
      roundUnsignedQuotient(builder, loc, significand, denominator);
  negativeRounded = arith::SelectOp::create(
      builder, loc, denominatorShiftAtMostFour, negativeRounded, zero64);
  auto magnitude =
      arith::SelectOp::create(builder, loc, scaledExponentIsNonnegative,
                              positiveRounded, negativeRounded);

  auto sign = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne,
      arith::AndIOp::create(
          builder, loc, originalRepresentation,
          integerConstant(builder, loc, i64Type, uint64_t{1} << 63U)),
      zero64);
  auto signedMagnitude = arith::SelectOp::create(
      builder, loc, sign,
      arith::SubIOp::create(builder, loc, zero64, magnitude), magnitude);
  Value result = signedMagnitude;
  if (bitWidth != MACHINE_WIDTH) {
    result = arith::TruncIOp::create(
        builder, loc, builder.getIntegerType(bitWidth), signedMagnitude);
  }
  result.getDefiningOp()->setAttr(FLOAT_TO_BITS_ATTR, builder.getUnitAttr());
  return result;
}

Value buildBitsToRadians(OpBuilder& builder, const Location loc, Value bits) {
  const auto integerType = dyn_cast<IntegerType>(bits.getType());
  assert(integerType && isSupportedWidth(integerType.getWidth()) &&
         "angle bits must use a supported integer width");
  auto asFloat =
      arith::UIToFPOp::create(builder, loc, builder.getF64Type(), bits);
  auto scale = constantF64(
      builder, loc,
      std::ldexp(TWO_PI, -static_cast<int>(integerType.getWidth())));
  return arith::MulFOp::create(builder, loc, asFloat, scale);
}

Value buildResize(OpBuilder& builder, const Location loc, Value bits,
                  const uint32_t targetWidth) {
  const auto sourceType = cast<IntegerType>(bits.getType());
  const auto sourceWidth = sourceType.getWidth();
  assert(isSupportedWidth(sourceWidth) && isSupportedWidth(targetWidth) &&
         "angle resize requires supported widths");
  if (sourceWidth == targetWidth) {
    return bits;
  }
  llvm::APInt constantBits;
  if (matchPattern(bits, m_ConstantInt(&constantBits))) {
    const auto resized =
        resize(constantBits.getZExtValue(), sourceWidth, targetWidth);
    return arith::ConstantOp::create(
        builder, loc,
        IntegerAttr::get(builder.getIntegerType(targetWidth),
                         llvm::APInt(targetWidth, resized)));
  }
  if (sourceWidth < targetWidth) {
    const auto targetType = builder.getIntegerType(targetWidth);
    auto extended = arith::ExtUIOp::create(builder, loc, targetType, bits);
    auto shift = arith::ConstantIntOp::create(
        builder, loc, targetWidth - sourceWidth, targetWidth);
    return arith::ShLIOp::create(builder, loc, extended, shift);
  }

  const auto discardedWidth = sourceWidth - targetWidth;
  auto shift =
      arith::ConstantIntOp::create(builder, loc, discardedWidth, sourceWidth);
  auto retained = arith::ShRUIOp::create(builder, loc, bits, shift);
  auto discardedMask = arith::ConstantIntOp::create(
      builder, loc, sourceType,
      llvm::APInt::getLowBitsSet(sourceWidth, discardedWidth));
  auto discarded = arith::AndIOp::create(builder, loc, bits, discardedMask);
  auto halfway = arith::ConstantIntOp::create(
      builder, loc, sourceType,
      llvm::APInt::getOneBitSet(sourceWidth, discardedWidth - 1U));
  auto greater = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                       discarded, halfway);
  auto equal = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                     discarded, halfway);
  auto one = arith::ConstantIntOp::create(builder, loc, 1, sourceWidth);
  auto leastSignificant = arith::AndIOp::create(builder, loc, retained, one);
  auto odd = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, leastSignificant,
      arith::ConstantIntOp::create(builder, loc, 0, sourceWidth));
  auto roundedTie = arith::AndIOp::create(builder, loc, equal, odd);
  auto shouldRound = arith::OrIOp::create(builder, loc, greater, roundedTie);
  auto increment =
      arith::ExtUIOp::create(builder, loc, sourceType, shouldRound);
  auto rounded = arith::AddIOp::create(builder, loc, retained, increment);
  return arith::TruncIOp::create(builder, loc,
                                 builder.getIntegerType(targetWidth), rounded);
}

Value buildQuantizedRadians(OpBuilder& builder, const Location loc,
                            Value radians, const uint32_t bitWidth) {
  assert(isSupportedWidth(bitWidth) && "unsupported angle bit width");
  if (const auto existing = matchQuantizedRadians(radians)) {
    return buildBitsToRadians(
        builder, loc, buildResize(builder, loc, existing->bits, bitWidth));
  }
  return buildBitsToRadians(builder, loc,
                            buildFloatToBits(builder, loc, radians, bitWidth));
}

std::optional<QuantizedRadians> matchQuantizedRadians(Value radians) {
  auto scaleOp = radians.getDefiningOp<arith::MulFOp>();
  if (!scaleOp) {
    return std::nullopt;
  }

  Value converted;
  Value scale;
  if (scaleOp.getLhs().getDefiningOp<arith::UIToFPOp>()) {
    converted = scaleOp.getLhs();
    scale = scaleOp.getRhs();
  } else if (scaleOp.getRhs().getDefiningOp<arith::UIToFPOp>()) {
    converted = scaleOp.getRhs();
    scale = scaleOp.getLhs();
  } else {
    return std::nullopt;
  }
  auto conversion = converted.getDefiningOp<arith::UIToFPOp>();
  Value bits = conversion.getIn();
  const auto integerType = dyn_cast<IntegerType>(bits.getType());
  if (!integerType || !isSupportedWidth(integerType.getWidth()) ||
      !isExpectedScale(scale, integerType.getWidth())) {
    return std::nullopt;
  }
  return QuantizedRadians{.bits = bits, .bitWidth = integerType.getWidth()};
}

std::optional<Value> matchFloatToBits(Value bits) {
  const auto bitType = dyn_cast<IntegerType>(bits.getType());
  if (!bitType || !isSupportedWidth(bitType.getWidth())) {
    return std::nullopt;
  }
  auto* marker = bits.getDefiningOp();
  if (marker == nullptr || !marker->hasAttr(FLOAT_TO_BITS_ATTR)) {
    return std::nullopt;
  }
  SmallVector<Value> worklist(marker->getOperands());
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    const auto value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    if (auto remainder = value.getDefiningOp<arith::RemFOp>()) {
      llvm::APFloat modulus(0.0);
      if (matchPattern(remainder.getRhs(), m_ConstantFloat(&modulus)) &&
          modulus.convertToDouble() == TWO_PI) {
        auto magnitude = remainder.getLhs().getDefiningOp<arith::BitcastOp>();
        auto masked = magnitude
                          ? magnitude.getIn().getDefiningOp<arith::AndIOp>()
                          : arith::AndIOp{};
        if (masked) {
          Value sourceBits;
          const auto magnitudeMask = llvm::APInt(64, DOUBLE_MAGNITUDE_MASK);
          if (constantEquals(masked.getLhs(), magnitudeMask)) {
            sourceBits = masked.getRhs();
          } else if (constantEquals(masked.getRhs(), magnitudeMask)) {
            sourceBits = masked.getLhs();
          }
          if (auto source = sourceBits.getDefiningOp<arith::BitcastOp>();
              source && source.getIn().getType().isF64() &&
              sourceBits.getType().isInteger(64)) {
            return source.getIn();
          }
        }
      }
    }
    if (auto* definingOp = value.getDefiningOp()) {
      llvm::append_range(worklist, definingOp->getOperands());
    }
  }
  return std::nullopt;
}

std::optional<AngleResize> matchResize(Value bits) {
  const auto targetType = dyn_cast<IntegerType>(bits.getType());
  if (!targetType) {
    return std::nullopt;
  }
  const auto targetWidth = targetType.getWidth();
  if (auto shift = bits.getDefiningOp<arith::ShLIOp>()) {
    auto extension = shift.getLhs().getDefiningOp<arith::ExtUIOp>();
    const auto distance = getConstantIntValue(shift.getRhs());
    if (!extension || !distance || *distance <= 0) {
      return std::nullopt;
    }
    const auto source = extension.getIn();
    const auto sourceType = dyn_cast<IntegerType>(source.getType());
    if (!sourceType || sourceType.getWidth() >= targetWidth ||
        sourceType.getWidth() + static_cast<uint64_t>(*distance) !=
            targetWidth) {
      return std::nullopt;
    }
    return AngleResize{
        .source = source,
        .sourceWidth = sourceType.getWidth(),
        .targetWidth = targetWidth,
        .operations = {extension.getOperation(), shift.getOperation()}};
  }

  auto truncation = bits.getDefiningOp<arith::TruncIOp>();
  if (!truncation) {
    return std::nullopt;
  }
  auto rounded = truncation.getIn().getDefiningOp<arith::AddIOp>();
  if (!rounded) {
    return std::nullopt;
  }
  arith::ShRUIOp retained;
  arith::ExtUIOp increment;
  for (const auto operand : rounded->getOperands()) {
    if (!retained) {
      retained = operand.getDefiningOp<arith::ShRUIOp>();
    }
    if (!increment) {
      increment = operand.getDefiningOp<arith::ExtUIOp>();
    }
  }
  if (!retained || !increment || !increment.getIn().getType().isInteger(1)) {
    return std::nullopt;
  }
  const auto source = retained.getLhs();
  const auto sourceType = dyn_cast<IntegerType>(source.getType());
  const auto discardedWidth = getConstantIntValue(retained.getRhs());
  if (!sourceType || !discardedWidth || *discardedWidth <= 0 ||
      sourceType.getWidth() - static_cast<uint64_t>(*discardedWidth) !=
          targetWidth) {
    return std::nullopt;
  }

  auto shouldRound = increment.getIn().getDefiningOp<arith::OrIOp>();
  if (!shouldRound) {
    return std::nullopt;
  }
  arith::CmpIOp greater;
  arith::AndIOp roundedTie;
  for (const auto operand : shouldRound->getOperands()) {
    if (!greater) {
      greater = operand.getDefiningOp<arith::CmpIOp>();
    }
    if (!roundedTie) {
      roundedTie = operand.getDefiningOp<arith::AndIOp>();
    }
  }
  if (!greater || greater.getPredicate() != arith::CmpIPredicate::ugt ||
      !roundedTie) {
    return std::nullopt;
  }
  auto discarded = greater.getLhs().getDefiningOp<arith::AndIOp>();
  if (!discarded || discarded.getLhs() != source) {
    return std::nullopt;
  }
  const auto sourceWidth = sourceType.getWidth();
  const auto discardedBits = static_cast<unsigned>(*discardedWidth);
  if (!constantEquals(discarded.getRhs(),
                      llvm::APInt::getLowBitsSet(sourceWidth, discardedBits)) ||
      !constantEquals(greater.getRhs(), llvm::APInt::getOneBitSet(
                                            sourceWidth, discardedBits - 1U))) {
    return std::nullopt;
  }

  arith::CmpIOp equal;
  arith::CmpIOp odd;
  for (const auto operand : roundedTie->getOperands()) {
    auto comparison = operand.getDefiningOp<arith::CmpIOp>();
    if (!comparison) {
      continue;
    }
    if (comparison.getPredicate() == arith::CmpIPredicate::eq) {
      equal = comparison;
    } else if (comparison.getPredicate() == arith::CmpIPredicate::ne) {
      odd = comparison;
    }
  }
  if (!equal || equal.getLhs() != discarded.getResult() ||
      equal.getRhs() != greater.getRhs() || !odd ||
      !constantEquals(odd.getRhs(), llvm::APInt(sourceWidth, 0))) {
    return std::nullopt;
  }
  auto leastSignificant = odd.getLhs().getDefiningOp<arith::AndIOp>();
  if (!leastSignificant || leastSignificant.getLhs() != retained.getResult() ||
      !constantEquals(leastSignificant.getRhs(), llvm::APInt(sourceWidth, 1))) {
    return std::nullopt;
  }

  return AngleResize{
      .source = source,
      .sourceWidth = sourceWidth,
      .targetWidth = targetWidth,
      .operations = {retained.getOperation(), discarded.getOperation(),
                     greater.getOperation(), equal.getOperation(),
                     leastSignificant.getOperation(), odd.getOperation(),
                     roundedTie.getOperation(), shouldRound.getOperation(),
                     increment.getOperation(), rounded.getOperation(),
                     truncation.getOperation()}};
}

} // namespace mlir::mqt::angle
