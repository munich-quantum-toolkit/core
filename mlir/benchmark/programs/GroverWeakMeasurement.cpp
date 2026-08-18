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

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The coupling of the probe qubit to the search register.
constexpr double PROBE_ANGLE = std::numbers::pi / 3.0;
} // namespace

SmallVector<Value> groverWeakMeasurement(qc::QCProgramBuilder& b,
                                         const uint64_t n) {
  const auto search = static_cast<int64_t>(n) - 2;
  auto q = b.allocQubitRegister(search, "q");
  auto flag = b.allocQubit();
  auto probe = b.allocQubit();
  auto c = b.allocClassicalBitRegister(search, "c");

  b.scfFor(0, search, 1, [&](Value i) { b.reset(b.loadQubit(q.value, i)); });
  b.reset(flag);
  b.reset(probe);

  b.scfFor(0, search, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
  b.x(flag);

  const llvm::ArrayRef<Value> oracleControls(q.qubits);
  const auto diffusionControls = oracleControls.drop_back();

  // The search repeats until the probe reports the marked state. The probe is
  // coupled weakly, so it reports late and the round count follows from the
  // measurements rather than from the size of the search space.
  b.scfWhile(
      [&] {
        b.mcz(oracleControls, flag);

        b.scfFor(0, search, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });
        b.scfFor(0, search, 1, [&](Value i) { b.x(b.loadQubit(q.value, i)); });
        b.mcz(diffusionControls, q[search - 1]);
        b.scfFor(0, search, 1, [&](Value i) { b.x(b.loadQubit(q.value, i)); });
        b.scfFor(0, search, 1, [&](Value i) { b.h(b.loadQubit(q.value, i)); });

        // A fresh probe takes a small rotation from the marked state alone.
        b.reset(probe);
        b.mcry(PROBE_ANGLE, oracleControls, probe);
        auto reported = b.measure(probe);
        const Value unreported = arith::XOrIOp::create(
            b, reported, arith::ConstantIntOp::create(b, 1, 1));
        b.scfCondition(unreported);
      },
      [] {});

  b.scfFor(0, search, 1,
           [&](Value i) { b.measure(b.loadQubit(q.value, i), c, i); });

  return {c};
}

} // namespace mqt::benchmark
