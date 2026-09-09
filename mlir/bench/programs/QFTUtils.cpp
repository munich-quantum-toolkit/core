/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QFTUtils.h"

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string_view>

namespace mqt::bench::detail {

using namespace mlir;

void prepareRegister(qc::QCProgramBuilder& builder, Value reg,
                     std::string_view bits) {
  for (size_t first = 0; first < bits.size();) {
    auto last = bits.find_first_not_of(bits[first], first);
    if (last == std::string_view::npos) {
      last = bits.size();
    }
    if (bits[first] != '0') {
      builder.scfFor(static_cast<int64_t>(bits.size() - last),
                     static_cast<int64_t>(bits.size() - first), 1,
                     [&](Value index) {
                       auto qubit = builder.loadQubit(reg, index);
                       if (bits[first] == '+') {
                         builder.h(qubit);
                       } else {
                         builder.x(qubit);
                       }
                     });
    }
    first = last;
  }
}

void phaseRotationLoop(
    qc::QCProgramBuilder& builder, Value lower, Value upper, Value step,
    Value initialAngle, Value scale,
    const function_ref<void(Value angle, Value index)>& body) {
  auto loop =
      scf::ForOp::create(builder, lower, upper, step, ValueRange{initialAngle});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  auto next = arith::MulFOp::create(builder, angle, scale).getResult();
  scf::YieldOp::create(builder, ValueRange{next});
}

void forwardQFT(qc::QCProgramBuilder& builder, Value qubitRegister,
                int64_t qubits) {
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(qubits - 1);
  auto firstAngle = builder.floatConstant(std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step).getResult();
    builder.h(builder.loadQubit(qubitRegister, target));

    auto previous = arith::SubIOp::create(builder, target, one).getResult();
    phaseRotationLoop(
        builder, zero, target, one, firstAngle, half,
        [&](Value angle, Value distance) {
          auto control =
              arith::SubIOp::create(builder, previous, distance).getResult();
          builder.cp(angle, builder.loadQubit(qubitRegister, control),
                     builder.loadQubit(qubitRegister, target));
        });
  });
}

void inverseQFT(qc::QCProgramBuilder& builder, Value qubitRegister,
                int64_t qubits) {
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto upper = builder.indexConstant(qubits);
  auto firstAngle = builder.floatConstant(-std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);
  builder.scfFor(zero, upper, 1, [&](Value target) {
    auto previous = arith::SubIOp::create(builder, target, one).getResult();
    phaseRotationLoop(
        builder, zero, target, one, firstAngle, half,
        [&](Value angle, Value distance) {
          auto control =
              arith::SubIOp::create(builder, previous, distance).getResult();
          builder.cp(angle, builder.loadQubit(qubitRegister, control),
                     builder.loadQubit(qubitRegister, target));
        });
    builder.h(builder.loadQubit(qubitRegister, target));
  });
}

} // namespace mqt::bench::detail
