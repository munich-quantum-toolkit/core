/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ModularArithmetic.h"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallVector.h"

namespace mqt::bench::detail {
using namespace mlir;

static void phaseAdd(qc::QCProgramBuilder& builder, Value accumulator,
                     int64_t width, Value addend, ValueRange controls,
                     bool inverse) {
  auto one = builder.intConstant(1);
  auto zero = builder.intConstant(0);
  phaseAdditionLoop(
      builder, width,
      [&](Value target) -> Value {
        auto shift =
            arith::IndexCastUIOp::create(builder, builder.getI64Type(), target);
        auto shifted = arith::ShRUIOp::create(builder, addend, shift);
        auto bit = arith::AndIOp::create(builder, shifted, one);
        return arith::CmpIOp::create(builder, arith::CmpIPredicate::ne, bit,
                                     zero);
      },
      [&](Value angle, Value target) {
        if (inverse) {
          angle =
              arith::MulFOp::create(builder, angle, builder.floatConstant(-1.));
        }
        auto qubit = builder.loadQubit(accumulator, target);
        if (controls.empty()) {
          builder.p(angle, qubit);
        } else {
          builder.mcp(angle, controls, qubit);
        }
      });
}

static void modularAdd(qc::QCProgramBuilder& builder, Value accumulator,
                       int64_t width, Value addend, Value modulus,
                       ValueRange controls, Value work, bool inverse) {
  auto overflowIndex = builder.indexConstant(width - 1);

  if (inverse) {
    /// Reverse the modular-adder operations and every phase rotation.
    phaseAdd(builder, accumulator, width, addend, controls, true);

    inverseQFT(builder, accumulator, width);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    forwardQFT(builder, accumulator, width);

    phaseAdd(builder, accumulator, width, addend, controls, false);
    phaseAdd(builder, accumulator, width, modulus, work, true);

    inverseQFT(builder, accumulator, width);
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    forwardQFT(builder, accumulator, width);

    phaseAdd(builder, accumulator, width, modulus, {}, false);
    phaseAdd(builder, accumulator, width, addend, controls, true);
    return;
  }
  phaseAdd(builder, accumulator, width, addend, controls, false);
  phaseAdd(builder, accumulator, width, modulus, {}, true);

  inverseQFT(builder, accumulator, width);
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  forwardQFT(builder, accumulator, width);

  phaseAdd(builder, accumulator, width, modulus, work, false);
  phaseAdd(builder, accumulator, width, addend, controls, true);

  inverseQFT(builder, accumulator, width);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  forwardQFT(builder, accumulator, width);

  phaseAdd(builder, accumulator, width, addend, controls, false);
}

void multiplyAccumulate(qc::QCProgramBuilder& builder, Value control,
                        Value multiplicand, Value accumulator, Value work,
                        Value multiplier, Value modulus, int64_t bits,
                        bool inverse) {
  const auto width = bits + 1;
  forwardQFT(builder, accumulator, width);
  auto loop =
      scf::ForOp::create(builder, builder.getLoc(), builder.indexConstant(0),
                         builder.indexConstant(bits), builder.indexConstant(1),
                         ValueRange{multiplier});
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    auto residue = loop.getRegionIterArgs().front();
    SmallVector<Value, 2> controls{
        control,
        builder.loadQubit(multiplicand, loop.getInductionVar()),
    };
    /// Controlled modular translations commute on the clean-workspace domain,
    /// so their inverses can consume the residues in the same forward order.
    modularAdd(builder, accumulator, width, residue, modulus, controls, work,
               inverse);
    /// Residues are below 2^63, so doubling fits unsigned i64.
    auto doubled =
        arith::ShLIOp::create(builder, residue, builder.intConstant(1));
    auto next = arith::RemUIOp::create(builder, doubled, modulus);
    scf::YieldOp::create(builder, builder.getLoc(), ValueRange{next});
  }
  inverseQFT(builder, accumulator, width);
}

} // namespace mqt::bench::detail
