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
  phaseAdditionLoop(
      builder, width,
      [&](Value target) -> Value {
        auto shift =
            arith::IndexCastUIOp::create(builder, builder.getI64Type(), target);
        auto shifted = arith::ShRUIOp::create(builder, addend, shift);
        return arith::TruncIOp::create(builder, builder.getI1Type(), shifted);
      },
      [&](Value angle, Value target) {
        auto qubit = builder.loadQubit(accumulator, target);
        if (controls.empty()) {
          builder.p(angle, qubit);
        } else {
          builder.mcp(angle, controls, qubit);
        }
      },
      inverse);
}

static void modularAdd(qc::QCProgramBuilder& builder, Value accumulator,
                       int64_t width, Value addend, Value modulus,
                       ValueRange controls, Value work, bool inverse) {
  auto overflowIndex = builder.indexConstant(width - 1);

  const auto toggleOverflow = [&](bool complement) {
    inverseQFT(builder, accumulator, width);
    if (complement) {
      builder.x(builder.loadQubit(accumulator, overflowIndex));
    }
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    if (complement) {
      builder.x(builder.loadQubit(accumulator, overflowIndex));
    }
    forwardQFT(builder, accumulator, width);
  };

  phaseAdd(builder, accumulator, width, addend, controls, inverse);
  if (!inverse) {
    phaseAdd(builder, accumulator, width, modulus, {}, true);
  }
  toggleOverflow(inverse);
  // These diagonal phase additions commute in the Fourier basis.
  phaseAdd(builder, accumulator, width, addend, controls, !inverse);
  phaseAdd(builder, accumulator, width, modulus, work, inverse);
  toggleOverflow(!inverse);
  if (inverse) {
    phaseAdd(builder, accumulator, width, modulus, {}, false);
  }
  phaseAdd(builder, accumulator, width, addend, controls, inverse);
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
    // Controlled modular translations commute on the clean-workspace domain,
    // so their inverses can consume the residues in the same forward order.
    modularAdd(builder, accumulator, width, residue, modulus, controls, work,
               inverse);
    // Residues are below 2^63, so doubling fits unsigned i64.
    auto doubled =
        arith::ShLIOp::create(builder, residue, builder.intConstant(1));
    auto next = arith::RemUIOp::create(builder, doubled, modulus);
    scf::YieldOp::create(builder, builder.getLoc(), ValueRange{next});
  }
  inverseQFT(builder, accumulator, width);
}

} // namespace mqt::bench::detail
