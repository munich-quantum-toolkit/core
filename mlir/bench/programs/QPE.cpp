/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QPE.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"
#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"

#include <bit>
#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static Value doubleResidue(qc::QCProgramBuilder& builder,
                                         Value residue, Value denominator) {
  auto complement = arith::SubIOp::create(builder, denominator, residue);
  auto wraps = arith::CmpIOp::create(builder, arith::CmpIPredicate::uge,
                                     residue, complement);
  auto wrapped = arith::SubIOp::create(builder, residue, complement);
  auto doubled = arith::AddIOp::create(builder, residue, residue);
  return arith::SelectOp::create(builder, wraps, wrapped, doubled);
}

[[nodiscard]] static Value controlledPhaseAngle(qc::QCProgramBuilder& builder,
                                                Value residue,
                                                uint64_t denominator) {
  auto numerator =
      arith::UIToFPOp::create(builder, builder.getF64Type(), residue);
  auto turns = arith::DivFOp::create(
      builder, numerator,
      builder.floatConstant(static_cast<double>(denominator)));
  return arith::MulFOp::create(builder, turns,
                               builder.floatConstant(2. * std::numbers::pi));
}

[[nodiscard]] static SmallVector<Value>
iterativeQPE(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubit();
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.x(ancilla);

  auto lower = builder.indexConstant(0);
  auto upper = builder.indexConstant(precision);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(precision - 1);
  const auto& phase = benchmark.options().phase;
  const auto denominator = phase.denominator();

  /// Compute one exact starting residue in O(log precision). Products of
  /// reduced uint64_t values fit in 128 bits.
  const llvm::APInt modulus(128, denominator);
  llvm::APInt initial(128, phase.numerator());
  llvm::APInt factor(128, 2);
  for (auto exponent = static_cast<uint64_t>(precision - 1); exponent != 0;
       exponent >>= 1U) {
    if ((exponent & 1U) != 0) {
      initial = (initial * factor).urem(modulus);
    }
    factor = (factor * factor).urem(modulus);
  }
  auto initialResidue =
      builder.intConstant(static_cast<int64_t>(initial.getZExtValue()));

  /// For d=2^s*m with odd m, high powers reverse by halving and adding
  /// ceil(d/2) when (residue>>s) is odd. Pack the lost wrap bits for powers at
  /// most s.
  const auto shift = static_cast<unsigned>(
      std::countr_zero(denominator)); /// spellchecker:disable-line
  uint64_t wraps = 0;
  auto residue = phase.numerator();
  for (unsigned power = 1; power <= shift; ++power) {
    if (residue >= denominator - residue) {
      wraps |= uint64_t{1} << power;
      residue -= denominator - residue;
    } else {
      residue += residue;
    }
  }
  auto integerOne = builder.intConstant(1);
  auto integerZero = builder.intConstant(0);
  auto halfDenominator = builder.intConstant(
      static_cast<int64_t>((denominator >> 1U) + (denominator & 1U)));
  auto firstCorrection = builder.floatConstant(-std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);

  auto loop = scf::ForOp::create(builder, lower, upper, one,
                                 ValueRange{initialResidue});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    auto index = loop.getInductionVar();
    auto current = loop.getRegionIterArg(0);
    auto angle = controlledPhaseAngle(builder, current, denominator);
    builder.h(query);
    builder.cp(angle, query, ancilla);

    auto previous = arith::SubIOp::create(builder, index, one);
    detail::phaseRotationLoop(
        builder, lower, index, one, firstCorrection, half,
        [&](Value correction, Value distance) {
          auto bit = arith::SubIOp::create(builder, previous, distance);
          builder.scfIf(result, bit, [&] { builder.p(correction, query); });
        });

    builder.h(query);
    builder.measure(query, result, index);
    builder.reset(query);

    Value wrap = current;
    if (shift != 0) {
      auto wrapBits = builder.intConstant(static_cast<int64_t>(wraps));
      auto shiftValue = builder.intConstant(shift);
      auto power = arith::SubIOp::create(builder, last, index);
      auto powerValue =
          arith::IndexCastOp::create(builder, builder.getI64Type(), power);
      /// Both select operands execute, so even the unused shift must stay
      /// below 64.
      auto isLowPower = arith::CmpIOp::create(
          builder, arith::CmpIPredicate::ule, powerValue, shiftValue);
      auto boundedPower =
          arith::SelectOp::create(builder, isLowPower, powerValue, shiftValue);
      auto lowWrap = arith::ShRUIOp::create(builder, wrapBits, boundedPower);
      auto highWrap = arith::ShRUIOp::create(builder, current, shiftValue);
      wrap = arith::SelectOp::create(builder, isLowPower, lowWrap, highWrap);
    }
    auto carried = arith::TruncIOp::create(builder, builder.getI1Type(), wrap);
    auto correction =
        arith::SelectOp::create(builder, carried, halfDenominator, integerZero);
    auto halved = arith::ShRUIOp::create(builder, current, integerOne);
    auto next = arith::AddIOp::create(builder, halved, correction);
    scf::YieldOp::create(builder, ValueRange{next});
  }
  return {result};
}

[[nodiscard]] static SmallVector<Value>
standardQPE(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubitRegisterStorage(precision, "query");
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.scfFor(0, precision, 1, [&](Value index) {
    builder.h(builder.loadQubit(query, index));
  });
  builder.x(ancilla);

  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto upper = builder.indexConstant(precision);
  auto last = builder.indexConstant(precision - 1);
  const auto& phase = benchmark.options().phase;
  auto denominator =
      builder.intConstant(static_cast<int64_t>(phase.denominator()));
  auto initialResidue =
      builder.intConstant(static_cast<int64_t>(phase.numerator()));
  auto loop =
      scf::ForOp::create(builder, zero, upper, one, ValueRange{initialResidue});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    auto index = loop.getInductionVar();
    auto residue = loop.getRegionIterArg(0);
    auto angle = controlledPhaseAngle(builder, residue, phase.denominator());
    auto control = arith::SubIOp::create(builder, last, index);
    builder.cp(angle, builder.loadQubit(query, control), ancilla);
    auto next = doubleResidue(builder, residue, denominator);
    scf::YieldOp::create(builder, ValueRange{next});
  }

  detail::inverseQFT(builder, query, precision);
  builder.measureQubitRegister(query, result, precision);
  return {result};
}

SmallVector<Value> qpe(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  return benchmark.options().method == QPEMethod::Standard
             ? standardQPE(builder, benchmark)
             : iterativeQPE(builder, benchmark);
}

} // namespace mqt::bench
