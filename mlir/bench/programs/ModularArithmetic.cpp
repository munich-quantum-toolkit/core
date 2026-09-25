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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallVector.h"

#include <numbers>

namespace mqt::bench::detail {
using namespace mlir;

namespace {

struct PhaseData {
  int64_t width;
  Value angles;
  Value modulusOffset;
};

} // namespace

static void appendPhaseAngles(SmallVectorImpl<double>& angles,
                              const llvm::APInt& value) {
  long double angle = 0.L;
  for (unsigned bit = 0; bit < value.getBitWidth(); ++bit) {
    angle /= 2.L;
    if (value[bit]) {
      angle += std::numbers::pi_v<long double>;
    }
    angles.push_back(static_cast<double>(angle));
  }
}

void appendModularPhaseAngles(SmallVectorImpl<double>& angles,
                              llvm::APInt multiplier,
                              const llvm::APInt& modulus) {
  const auto bits = modulus.getBitWidth() - 1;
  for (unsigned bit = 0; bit < bits; ++bit) {
    appendPhaseAngles(angles, multiplier);
    multiplier = multiplier.shl(1).urem(modulus);
  }
  appendPhaseAngles(angles, modulus);
}

static void phaseAdd(qc::QCProgramBuilder& builder, Value accumulator,
                     const PhaseData& data, Value offset, ValueRange controls,
                     const bool inverse) {
  builder.scfFor(0, data.width, 1, [&](Value target) {
    auto angleIndex =
        arith::AddIOp::create(builder, offset, target).getResult();
    auto angle =
        tensor::ExtractOp::create(builder, data.angles, ValueRange{angleIndex})
            .getResult();
    if (inverse) {
      angle = arith::MulFOp::create(builder, angle, builder.floatConstant(-1.))
                  .getResult();
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
                       const PhaseData& data, Value addendOffset,
                       ValueRange controls, Value work, bool inverse) {
  auto overflowIndex = builder.indexConstant(data.width - 1);

  if (inverse) {
    // Reverse the modular-adder operations and every phase rotation.
    phaseAdd(builder, accumulator, data, addendOffset, controls, true);

    inverseQFT(builder, accumulator, data.width);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    forwardQFT(builder, accumulator, data.width);

    phaseAdd(builder, accumulator, data, addendOffset, controls, false);
    phaseAdd(builder, accumulator, data, data.modulusOffset, work, true);

    inverseQFT(builder, accumulator, data.width);
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    forwardQFT(builder, accumulator, data.width);

    phaseAdd(builder, accumulator, data, data.modulusOffset, {}, false);
    phaseAdd(builder, accumulator, data, addendOffset, controls, true);
    return;
  }
  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
  phaseAdd(builder, accumulator, data, data.modulusOffset, {}, true);

  inverseQFT(builder, accumulator, data.width);
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  forwardQFT(builder, accumulator, data.width);

  phaseAdd(builder, accumulator, data, data.modulusOffset, work, false);
  phaseAdd(builder, accumulator, data, addendOffset, controls, true);

  inverseQFT(builder, accumulator, data.width);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  forwardQFT(builder, accumulator, data.width);

  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
}

void multiplyAccumulate(qc::QCProgramBuilder& builder, Value control,
                        Value multiplicand, Value accumulator, Value work,
                        Value angles, Value offset, int64_t bits,
                        bool inverse) {
  const auto width = bits + 1;
  auto stride = builder.indexConstant(width);
  auto modulusRow = builder.indexConstant(bits * width);
  const PhaseData phases{
      .width = width,
      .angles = angles,
      .modulusOffset = arith::AddIOp::create(builder, offset, modulusRow),
  };
  forwardQFT(builder, accumulator, width);
  builder.scfFor(0, bits, 1, [&](Value index) {
    auto bit = index;
    if (inverse) {
      bit = arith::SubIOp::create(builder, builder.indexConstant(bits - 1),
                                  index);
    }
    auto row = arith::MulIOp::create(builder, bit, stride);
    auto addendOffset = arith::AddIOp::create(builder, offset, row);
    SmallVector<Value, 2> controls{
        control,
        builder.loadQubit(multiplicand, bit),
    };
    modularAdd(builder, accumulator, phases, addendOffset, controls, work,
               inverse);
  });
  inverseQFT(builder, accumulator, width);
}

} // namespace mqt::bench::detail
