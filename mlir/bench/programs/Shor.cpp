/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Shor.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"

#include "ModularArithmetic.h"
#include "Programs.h"
#include "QFTUtils.h"
#include "ShorMultiplier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace mqt::bench {
using namespace mlir;

// Extended Euclid on coprime residues below 2^31.
[[nodiscard]] static uint64_t inverseModulo(uint64_t value, uint64_t modulus) {
  auto remainder = static_cast<int64_t>(modulus);
  auto nextRemainder = static_cast<int64_t>(value);
  int64_t coefficient = 0;
  int64_t nextCoefficient = 1;
  while (nextRemainder != 0) {
    const auto quotient = remainder / nextRemainder;
    const auto reduced = remainder - quotient * nextRemainder;
    remainder = nextRemainder;
    nextRemainder = reduced;
    const auto updated = coefficient - quotient * nextCoefficient;
    coefficient = nextCoefficient;
    nextCoefficient = updated;
  }
  if (coefficient < 0) {
    coefficient += static_cast<int64_t>(modulus);
  }
  return static_cast<uint64_t>(coefficient);
}

namespace detail {

func::FuncOp createInPlaceMultiplier(qc::QCProgramBuilder& builder,
                                     int64_t bits,
                                     RankedTensorType anglesType) {
  auto qubitType = qc::QubitType::get(builder.getContext());
  SmallVector<Type> types{
      qubitType,
      MemRefType::get({bits}, qubitType),
      MemRefType::get({bits + 1}, qubitType),
      qubitType,
      anglesType,
      builder.getIndexType(),
  };
  const auto createAccumulator = [&](StringRef name, bool inverse) {
    return builder.createFunction(name, types, [&](ValueRange arguments) {
      multiplyAccumulate(builder, arguments[0], arguments[1], arguments[2],
                         arguments[3], arguments[4], arguments[5], bits,
                         inverse);
      return SmallVector<Value>{};
    });
  };
  auto accumulate = createAccumulator("shor_accumulate", false);
  auto subtract = createAccumulator("shor_uncompute", true);
  return builder.createFunction(
      "shor_multiply", types, [&](ValueRange arguments) {
        builder.call(accumulate, arguments);
        builder.scfFor(0, bits, 1, [&](Value index) {
          builder.cswap(arguments[0], builder.loadQubit(arguments[1], index),
                        builder.loadQubit(arguments[2], index));
        });
        auto inverseOffset = arith::AddIOp::create(
            builder, arguments[5],
            builder.indexConstant((bits + 1) * (bits + 1)));
        SmallVector<Value> inverseArguments(arguments);
        inverseArguments.back() = inverseOffset;
        builder.call(subtract, inverseArguments);
        return SmallVector<Value>{};
      });
}

} // namespace detail

SmallVector<Value> shor(qc::QCProgramBuilder& builder, const Shor& benchmark) {
  const auto& options = benchmark.options();
  const auto bits = static_cast<int64_t>(std::bit_width(options.number));
  const auto precision = 2 * bits;
  const auto width = static_cast<unsigned>(bits + 1);
  const auto blockSize = (bits + 1) * (bits + 1);

  SmallVector<double> angles;
  angles.reserve(static_cast<size_t>(2 * precision * blockSize));
  auto power = options.base;
  const llvm::APInt modulus(width, options.number);
  for (int64_t round = 0; round < precision; ++round) {
    detail::appendModularPhaseAngles(angles, llvm::APInt(width, power),
                                     modulus);
    detail::appendModularPhaseAngles(
        angles, llvm::APInt(width, inverseModulo(power, options.number)),
        modulus);
    power = (power * power) % options.number;
  }
  auto anglesType = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                          builder.getF64Type());
  auto multiply = detail::createInPlaceMultiplier(builder, bits, anglesType);
  auto phases = arith::ConstantOp::create(
      builder, DenseElementsAttr::get(anglesType, ArrayRef<double>(angles)));

  auto query = builder.allocQubit();
  auto value = builder.allocQubitRegisterStorage(bits, "value");
  auto accumulator = builder.allocQubitRegisterStorage(bits + 1, "accumulator");
  auto work = builder.allocQubit();
  builder.x(builder.loadQubit(value, builder.indexConstant(0)));
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);

  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(precision - 1);
  auto stride = builder.indexConstant(2 * blockSize);
  auto firstCorrection = builder.floatConstant(-std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);
  builder.scfFor(0, precision, 1, [&](Value round) {
    auto powerIndex = arith::SubIOp::create(builder, last, round);
    auto offset = arith::MulIOp::create(builder, powerIndex, stride);
    builder.h(query);
    builder.call(multiply, {query, value, accumulator, work, phases, offset});
    auto previous = arith::SubIOp::create(builder, round, one);
    detail::phaseRotationLoop(
        builder, zero, round, one, firstCorrection, half,
        [&](Value angle, Value distance) {
          auto bit = arith::SubIOp::create(builder, previous, distance);
          builder.scfIf(result, bit, [&] { builder.p(angle, query); });
        });
    builder.h(query);
    builder.measure(query, result, round);
    builder.reset(query);
  });
  return {result};
}

} // namespace mqt::bench
