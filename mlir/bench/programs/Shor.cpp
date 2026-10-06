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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace mqt::bench {
using namespace mlir;

/// Extended Euclid on coprime residues below 2^31.
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
                                     int64_t bits) {
  auto qubitType = qc::QubitType::get(builder.getContext());
  SmallVector<Type> types{
      qubitType,
      MemRefType::get({bits}, qubitType),
      MemRefType::get({bits + 1}, qubitType),
      qubitType,
      builder.getI64Type(),
      builder.getI64Type(),
      builder.getI64Type(),
  };
  return builder.createFunction(
      "shor_multiply", types, [&](ValueRange arguments) {
        multiplyAccumulate(builder, arguments[0], arguments[1], arguments[2],
                           arguments[3], arguments[4], arguments[6], bits);
        builder.scfFor(0, bits, 1, [&](Value index) {
          builder.cswap(arguments[0], builder.loadQubit(arguments[1], index),
                        builder.loadQubit(arguments[2], index));
        });
        multiplyAccumulate(builder, arguments[0], arguments[1], arguments[2],
                           arguments[3], arguments[5], arguments[6], bits,
                           true);
        return SmallVector<Value>{};
      });
}

} // namespace detail

SmallVector<Value> shor(qc::QCProgramBuilder& builder, const Shor& benchmark) {
  const auto& options = benchmark.options();
  const auto bits = static_cast<int64_t>(std::bit_width(options.number));
  const auto precision = 2 * bits;
  SmallVector<int64_t> powers;
  powers.reserve(static_cast<size_t>(2 * precision));
  auto power = options.base;
  auto inverse = inverseModulo(power, options.number);
  for (int64_t round = 0; round < precision; ++round) {
    powers.push_back(static_cast<int64_t>(power));
    powers.push_back(static_cast<int64_t>(inverse));
    power = (power * power) % options.number;
    inverse = (inverse * inverse) % options.number;
  }
  auto powersType =
      RankedTensorType::get({2 * precision}, builder.getI64Type());
  auto multiply = detail::createInPlaceMultiplier(builder, bits);
  auto values = arith::ConstantOp::create(
      builder, DenseElementsAttr::get(powersType, ArrayRef<int64_t>(powers)));
  auto modulus = builder.intConstant(static_cast<int64_t>(options.number));

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
  auto stride = builder.indexConstant(2);
  auto firstCorrection = builder.floatConstant(-std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);
  builder.scfFor(0, precision, 1, [&](Value round) {
    auto powerIndex = arith::SubIOp::create(builder, last, round);
    auto offset = arith::MulIOp::create(builder, powerIndex, stride);
    auto inverseOffset = arith::AddIOp::create(builder, offset, one);
    auto multiplier =
        tensor::ExtractOp::create(builder, values, ValueRange{offset});
    auto inverse =
        tensor::ExtractOp::create(builder, values, ValueRange{inverseOffset});
    builder.h(query);
    builder.call(multiply, {
                               query,
                               value,
                               accumulator,
                               work,
                               multiplier,
                               inverse,
                               modulus,
                           });
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
