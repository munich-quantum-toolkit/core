/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QFTAdderUtils.h"

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <numbers>

namespace mqt::bench::detail {

using namespace mlir;

static void
phaseRotationLoop(qc::QCProgramBuilder& builder, Value upper,
                  Value initialAngle, double factor,
                  const function_ref<void(Value angle, Value index)>& body) {
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto scale = builder.floatConstant(factor);
  auto loop =
      scf::ForOp::create(builder, zero, upper, one, ValueRange{initialAngle});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  auto next = arith::MulFOp::create(builder, angle, scale).getResult();
  scf::YieldOp::create(builder, ValueRange{next});
}

void forwardQFT(qc::QCProgramBuilder& builder, Value qubitRegister,
                int64_t qubits) {
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(qubits - 1);
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step).getResult();
    builder.h(builder.loadQubit(qubitRegister, target));

    auto previous = arith::SubIOp::create(builder, target, one).getResult();
    auto firstAngle = builder.floatConstant(std::numbers::pi / 2.);
    phaseRotationLoop(
        builder, target, firstAngle, 0.5, [&](Value angle, Value distance) {
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
  auto firstAngle = builder.floatConstant(-std::numbers::pi);
  auto half = builder.floatConstant(0.5);
  auto loop =
      scf::ForOp::create(builder, zero, upper, one, ValueRange{firstAngle});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());

  auto target = loop.getInductionVar();
  auto initialAngle = loop.getRegionIterArg(0);
  phaseRotationLoop(
      builder, target, initialAngle, 2., [&](Value angle, Value control) {
        builder.cp(angle, builder.loadQubit(qubitRegister, control),
                   builder.loadQubit(qubitRegister, target));
      });
  builder.h(builder.loadQubit(qubitRegister, target));

  auto next = arith::MulFOp::create(builder, initialAngle, half).getResult();
  scf::YieldOp::create(builder, ValueRange{next});
}

} // namespace mqt::bench::detail
