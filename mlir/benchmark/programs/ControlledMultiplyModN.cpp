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
/// The multiplier of the modular multiplication.
constexpr int64_t MULTIPLIER = 3;
/// The modulus of the modular multiplication.
constexpr int64_t MODULUS = 5;

/// Applies a quantum Fourier transform, or its inverse, to @p reg.
void fourierTransform(qc::QCProgramBuilder& b, Value reg, const int64_t size,
                      const double sign) {
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(size - 1);

  b.scfFor(0, size, 1, [&](Value step) {
    // The inverse runs the same rotations in the opposite order.
    auto i = sign > 0.0 ? step : Value{arith::SubIOp::create(b, last, step)};
    if (sign < 0.0) {
      b.h(b.loadQubit(reg, i));
    }

    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = b.indexConstant(size);
    auto start = b.floatConstant(sign * std::numbers::pi / 2.0);
    auto half = b.floatConstant(0.5);

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

SmallVector<Value> controlledMultiplyModN(qc::QCProgramBuilder& b,
                                          const uint64_t n) {
  const auto total = static_cast<int64_t>(n) - 2;
  const auto factor = total / 2;
  const auto size = total - factor;
  auto ctrl = b.allocQubit();
  auto x = b.allocQubitRegister(factor, "x");
  auto acc = b.allocQubitRegister(size, "acc");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(size, "c");

  b.reset(ctrl);
  b.scfFor(0, factor, 1, [&](Value i) { b.reset(b.loadQubit(x.value, i)); });
  b.scfFor(0, size, 1, [&](Value i) { b.reset(b.loadQubit(acc.value, i)); });
  b.reset(anc);

  b.h(ctrl);
  b.scfFor(0, factor, 1, [&](Value i) { b.h(b.loadQubit(x.value, i)); });

  fourierTransform(b, acc.value, size, 1.0);

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto factors = b.indexConstant(factor);
  auto width = b.indexConstant(size);
  auto two = b.floatConstant(2.0);
  auto half = b.floatConstant(0.5);

  // Each multiplier qubit adds a shifted copy of the multiplier into the
  // accumulator, and every addition is a layer of phases in the Fourier basis.
  // The shift doubles the phase from one multiplier qubit to the next, and the
  // phase halves down the accumulator, so both angles come from the loops.
  auto base =
      b.floatConstant(std::numbers::pi * static_cast<double>(MULTIPLIER));
  auto outer = scf::ForOp::create(b, zero, factors, one, ValueRange{base});
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outer.getBody());
    auto shifted = outer.getRegionIterArg(0);
    auto i = outer.getInductionVar();
    const SmallVector<Value> controls{ctrl, b.loadQubit(x.value, i)};

    auto inner = scf::ForOp::create(b, zero, width, one, ValueRange{shifted});
    {
      OpBuilder::InsertionGuard innerGuard(b);
      b.setInsertionPointToStart(inner.getBody());
      auto angle = inner.getRegionIterArg(0);
      b.mcp(angle, controls, b.loadQubit(acc.value, inner.getInductionVar()));
      auto next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }

    auto doubled = arith::MulFOp::create(b, shifted, two);
    scf::YieldOp::create(b, ValueRange{doubled});
  }

  // The product is reduced modulo the modulus. Each round tests the top qubit
  // of the accumulator and subtracts the modulus once when it is set, so the
  // number of rounds depends on the product.
  b.scfWhile(
      [&] {
        // The top qubit only carries the overflow in the computational basis,
        // so the accumulator leaves and re-enters the Fourier basis around the
        // copy onto the ancilla.
        fourierTransform(b, acc.value, size, -1.0);
        b.reset(anc);
        b.cx(acc[size - 1], anc);
        fourierTransform(b, acc.value, size, 1.0);
        auto overflow = b.measure(anc);
        b.scfCondition(overflow);
      },
      [&] {
        auto start =
            b.floatConstant(-std::numbers::pi * static_cast<double>(MODULUS));
        auto loop = scf::ForOp::create(b, zero, width, one, ValueRange{start});
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(loop.getBody());
        auto angle = loop.getRegionIterArg(0);
        b.p(angle, b.loadQubit(acc.value, loop.getInductionVar()));
        auto next = arith::MulFOp::create(b, angle, half);
        scf::YieldOp::create(b, ValueRange{next});
      });

  fourierTransform(b, acc.value, size, -1.0);

  b.scfFor(0, size, 1,
           [&](Value i) { b.measure(b.loadQubit(acc.value, i), c, i); });

  return {c};
}

} // namespace mqt::benchmark
