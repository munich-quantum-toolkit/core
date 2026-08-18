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

  auto one = b.indexConstant(1);
  auto initial = b.floatConstant(VQE_INITIAL_ANGLE);
  auto decay = b.floatConstant(VQE_DECAY);

  // One round evaluates the ansatz at the current angle and reads one qubit.
  // The optimizer shrinks the angle and repeats until a round reports that the
  // energy no longer improves, so the round count is only known at runtime.
  scf::WhileOp::create(
      b, TypeRange{b.getF64Type()}, ValueRange{initial},
      [&](OpBuilder&, Location, ValueRange args) {
        auto angle = args[0];
        b.scfFor(0, size, 1,
                 [&](Value i) { b.reset(b.loadQubit(q.value, i)); });
        b.scfFor(0, VQE_LAYERS, 1, [&](Value) {
          b.scfFor(0, size, 1,
                   [&](Value i) { b.ry(angle, b.loadQubit(q.value, i)); });
          b.scfFor(0, size - 1, 1, [&](Value i) {
            auto next = arith::AddIOp::create(b, i, one);
            b.cx(b.loadQubit(q.value, i), b.loadQubit(q.value, next));
          });
        });
        auto improved = b.measure(q[0]);
        scf::ConditionOp::create(b, improved, ValueRange{angle});
      },
      [&](OpBuilder&, Location, ValueRange args) {
        auto next = arith::MulFOp::create(b, args[0], decay);
        scf::YieldOp::create(b, ValueRange{next});
      });

  b.scfFor(0, size, 1,
           [&](Value i) { b.measure(b.loadQubit(q.value, i), c, i); });

  return {c};
}

} // namespace mqt::benchmark
