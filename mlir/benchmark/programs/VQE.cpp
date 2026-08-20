/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/BenchmarkUtils.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The number of ansatz layers in one energy evaluation.
constexpr int64_t VQE_LAYERS = 2;
/// The angle the optimizer starts from.
constexpr double VQE_INITIAL_ANGLE = std::numbers::pi / 2.0;
/// The factor by which the optimizer shrinks the angle after a round.
constexpr double VQE_DECAY = 0.5;
} // namespace

SmallVector<Value> vqe(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto pairs = b.indexConstant(size - 1);
  auto initial = b.floatConstant(VQE_INITIAL_ANGLE);
  auto decay = b.floatConstant(VQE_DECAY);
  // The chain has at most `size - 1` disagreeing pairs, so the first round
  // improves on this value whatever it measures.
  auto worst = b.intConstant(size);

  // A round prepares the ansatz at the current angle, reads the register, and
  // estimates the energy of an Ising chain from the measured bits. The
  // optimizer shrinks the angle and runs another round only while the energy
  // improves on the round before it, so the rounds follow from the
  // measurements.
  scf::WhileOp::create(
      b, TypeRange{b.getF64Type(), b.getI64Type()}, ValueRange{initial, worst},
      [&](OpBuilder&, Location, ValueRange args) {
        auto angle = args[0];
        auto previous = args[1];
        resetRegister(b, q.value, size);
        b.scfFor(0, VQE_LAYERS, 1, [&](Value) {
          b.scfFor(0, size, 1,
                   [&](Value i) { b.ry(angle, b.loadQubit(q.value, i)); });
          b.scfFor(0, size - 1, 1, [&](Value i) {
            auto next = arith::AddIOp::create(b, i, one);
            b.cx(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
          });
        });
        measureRegister(b, q.value, size, c);

        // The energy of the chain counts the neighbouring pairs that disagree.
        auto sum = scf::ForOp::create(b, zero, pairs, one,
                                      ValueRange{b.intConstant(0)});
        {
          const OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(sum.getBody());
          auto accumulated = sum.getRegionIterArg(0);
          auto i = sum.getInductionVar();
          auto next = arith::AddIOp::create(b, i, one);
          auto differs = arith::XOrIOp::create(
              b, b.loadClassicalBit(c, i), b.loadClassicalBit(c, Value{next}));
          auto term = arith::ExtUIOp::create(b, b.getI64Type(), differs);
          scf::YieldOp::create(
              b, ValueRange{arith::AddIOp::create(b, accumulated, term)});
        }
        auto energy = sum.getResult(0);

        auto improved = arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                              energy, previous);
        scf::ConditionOp::create(b, improved, ValueRange{angle, energy});
      },
      [&](OpBuilder&, Location, ValueRange args) {
        auto next = arith::MulFOp::create(b, args[0], decay);
        scf::YieldOp::create(b, ValueRange{next, args[1]});
      });

  measureRegister(b, q.value, size, c);

  return {c};
}

} // namespace mqt::benchmark
