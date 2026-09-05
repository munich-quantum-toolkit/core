/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdderQuantum.hpp"

#include "Programs.h"
#include "QFTAdderUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

static void addQuantumRegister(qc::QCProgramBuilder& builder, Value addend,
                               Value sum, int64_t qubits) {
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(qubits - 1);
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step).getResult();
    auto upper = arith::AddIOp::create(builder, target, one).getResult();
    auto firstAngle = builder.floatConstant(std::numbers::pi);
    auto half = builder.floatConstant(0.5);
    auto loop =
        scf::ForOp::create(builder, zero, upper, one, ValueRange{firstAngle});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    auto control =
        arith::SubIOp::create(builder, target, loop.getInductionVar())
            .getResult();
    builder.cp(angle, builder.loadQubit(addend, control),
               builder.loadQubit(sum, target));
    auto next = arith::MulFOp::create(builder, angle, half).getResult();
    scf::YieldOp::create(builder, ValueRange{next});
  });
}

SmallVector<Value> qftAdderQuantum(qc::QCProgramBuilder& builder,
                                   const QFTAdderQuantum& benchmark) {
  const auto qubits = static_cast<int64_t>(benchmark.options().qubits);
  auto addend = builder.allocQubitRegisterStorage(qubits, "addend");
  auto sum = builder.allocQubitRegisterStorage(qubits, "sum");
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  builder.scfFor(0, qubits, 1, [&](Value index) {
    builder.h(builder.loadQubit(addend, index));
  });
  auto zero = builder.indexConstant(0);
  builder.x(builder.loadQubit(sum, zero));

  detail::forwardQFT(builder, sum, qubits);
  addQuantumRegister(builder, addend, sum, qubits);
  detail::inverseQFT(builder, sum, qubits);

  builder.scfFor(0, qubits, 1, [&](Value index) {
    builder.measure(builder.loadQubit(sum, index), result, index);
  });
  auto resultOffset = builder.indexConstant(qubits);
  builder.scfFor(0, qubits, 1, [&](Value index) {
    auto resultIndex =
        arith::AddIOp::create(builder, resultOffset, index).getResult();
    builder.measure(builder.loadQubit(addend, index), result, resultIndex);
  });
  return {result};
}

} // namespace mqt::bench
