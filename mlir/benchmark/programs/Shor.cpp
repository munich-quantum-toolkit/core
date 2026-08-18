/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The base of the modular exponentiation.
constexpr int64_t SHOR_BASE = 3;
/// The modulus that the exponentiation reduces by.
constexpr int64_t SHOR_MODULUS = 5;

/// Applies a quantum Fourier transform, or its inverse, to @p reg.
void fourier(qc::QCProgramBuilder& b, Value reg, const int64_t size,
             const double sign) {
  auto one = arith::ConstantIndexOp::create(b, 1);
  auto last = arith::ConstantIndexOp::create(b, size - 1);

  b.scfFor(0, size, 1, [&](Value step) {
    // The inverse runs the same rotations in the opposite order.
    auto i = sign > 0.0 ? step : Value{arith::SubIOp::create(b, last, step)};
    if (sign < 0.0) {
      b.h(b.loadQubit(reg, i));
    }

    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = arith::ConstantIndexOp::create(b, size);
    auto start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(sign * std::numbers::pi / 2.0));
    auto half =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

    auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      b.cp(angle, b.loadQubit(reg, loop.getInductionVar()),
           b.loadQubit(reg, i));
      auto next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }

    if (sign > 0.0) {
      b.h(b.loadQubit(reg, i));
    }
  });
}
} // namespace

SmallVector<Value> shor(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = (static_cast<int64_t>(n) - 2) / 2;
  const auto steps = 2 * size;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(size, "x");
  auto acc = b.allocQubitRegister(size, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(steps, "c");

  b.reset(ctrl);
  b.scfFor(0, size, 1, [&](Value i) { b.reset(b.loadQubit(x.value, i)); });
  b.scfFor(0, size, 1, [&](Value i) { b.reset(b.loadQubit(acc.value, i)); });
  b.reset(anc);

  // The work register starts in the neutral element of the multiplication.
  b.x(x[0]);

  auto zero = arith::ConstantIndexOp::create(b, 0);
  auto one = arith::ConstantIndexOp::create(b, 1);
  auto width = arith::ConstantIndexOp::create(b, size);
  auto rounds = arith::ConstantIndexOp::create(b, steps);
  auto two =
      arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(2.0));
  auto half =
      arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

  // Phase estimation reads one control qubit over and over. Each round
  // multiplies the accumulator by the next power of the base, corrects the
  // control from the results of the earlier rounds, and then reuses the qubit.
  auto first = arith::ConstantFloatOp::create(
      b, b.getF64Type(),
      llvm::APFloat(std::numbers::pi * static_cast<double>(SHOR_BASE)));
  auto estimation = scf::ForOp::create(b, zero, rounds, one, ValueRange{first});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(estimation.getBody());
    auto k = estimation.getInductionVar();
    auto power = estimation.getRegionIterArg(0);

    b.reset(ctrl);
    b.h(ctrl);

    // The multiplication runs in the Fourier basis, so each multiplier qubit
    // contributes a phase to every accumulator qubit. The phase doubles from
    // one multiplier qubit to the next and halves down the accumulator.
    fourier(b, acc.value, size, 1.0);
    auto multiply = scf::ForOp::create(b, zero, width, one, ValueRange{power});
    {
      OpBuilder::InsertionGuard multiplyGuard(b);
      b.setInsertionPointToStart(multiply.getBody());
      auto shifted = multiply.getRegionIterArg(0);
      const SmallVector<Value> controls{
          ctrl, b.loadQubit(x.value, multiply.getInductionVar())};

      auto row = scf::ForOp::create(b, zero, width, one, ValueRange{shifted});
      {
        OpBuilder::InsertionGuard rowGuard(b);
        b.setInsertionPointToStart(row.getBody());
        auto angle = row.getRegionIterArg(0);
        b.mcp(angle, controls, b.loadQubit(acc.value, row.getInductionVar()));
        auto next = arith::MulFOp::create(b, angle, half);
        scf::YieldOp::create(b, ValueRange{next});
      }

      auto doubled = arith::MulFOp::create(b, shifted, two);
      scf::YieldOp::create(b, ValueRange{doubled});
    }

    // The product is reduced modulo the modulus. The number of rounds depends
    // on the product, so the loop bound is only known at runtime.
    b.scfWhile(
        [&] {
          fourier(b, acc.value, size, -1.0);
          b.reset(anc);
          b.cx(acc[size - 1], anc);
          fourier(b, acc.value, size, 1.0);
          auto overflow = b.measure(anc);
          b.scfCondition(overflow);
        },
        [&] {
          auto start = arith::ConstantFloatOp::create(
              b, b.getF64Type(),
              llvm::APFloat(-std::numbers::pi *
                            static_cast<double>(SHOR_MODULUS)));
          auto loop =
              scf::ForOp::create(b, zero, width, one, ValueRange{start});
          OpBuilder::InsertionGuard reduceGuard(b);
          b.setInsertionPointToStart(loop.getBody());
          auto angle = loop.getRegionIterArg(0);
          b.p(angle, b.loadQubit(acc.value, loop.getInductionVar()));
          auto next = arith::MulFOp::create(b, angle, half);
          scf::YieldOp::create(b, ValueRange{next});
        });
    fourier(b, acc.value, size, -1.0);

    // The inverse transform of the phase estimation runs on the classical
    // side: every result measured so far rotates the control qubit, and the
    // rotation halves the further back the result lies.
    auto correction = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(-std::numbers::pi / 2.0));
    auto feedback = scf::ForOp::create(b, zero, k, one, ValueRange{correction});
    {
      OpBuilder::InsertionGuard feedbackGuard(b);
      b.setInsertionPointToStart(feedback.getBody());
      auto angle = feedback.getRegionIterArg(0);
      auto previous = arith::SubIOp::create(b, arith::SubIOp::create(b, k, one),
                                            feedback.getInductionVar());
      b.scfIf(c, previous, [&] { b.p(angle, ctrl); });
      auto next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }

    b.h(ctrl);
    b.measure(ctrl, c, k);

    auto squared = arith::MulFOp::create(b, power, two);
    scf::YieldOp::create(b, ValueRange{squared});
  }

  return {c};
}

} // namespace mqt::benchmark
