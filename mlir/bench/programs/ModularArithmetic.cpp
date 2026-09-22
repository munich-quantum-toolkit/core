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
#include "mqt/Dialect/QC/IR/QCDialect.h"

#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <numbers>

namespace mqt::bench::detail {
using namespace mlir;

namespace {

struct PhaseData {
  int64_t width;
  Value angles;
  Value modulusOffset;
  Value negativeOne;
};

} // namespace

static void appendPhaseAngles(SmallVectorImpl<double>& angles,
                              const llvm::APInt& value,
                              std::optional<size_t> cutoff) {
  long double angle = 0.L;
  for (unsigned bit = 0; bit < value.getBitWidth(); ++bit) {
    angle /= 2.L;
    if (value[bit]) {
      angle += std::numbers::pi_v<long double>;
    }
    if (cutoff && bit > *cutoff && value[bit - *cutoff - 1]) {
      angle -= std::ldexp(std::numbers::pi_v<long double>,
                          -static_cast<int>(*cutoff + 1));
    }
    angles.push_back(static_cast<double>(angle));
  }
}

void appendModularPhaseAngles(SmallVectorImpl<double>& angles,
                              llvm::APInt multiplier,
                              const llvm::APInt& modulus,
                              std::optional<size_t> cutoff) {
  const auto bits = modulus.getBitWidth() - 1;
  for (unsigned bit = 0; bit < bits; ++bit) {
    appendPhaseAngles(angles, multiplier, cutoff);
    multiplier = multiplier.shl(1).urem(modulus);
  }
  appendPhaseAngles(angles, modulus, cutoff);
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
      angle =
          arith::MulFOp::create(builder, angle, data.negativeOne).getResult();
    }
    auto qubit = builder.loadQubit(accumulator, target);
    if (controls.empty()) {
      builder.p(angle, qubit);
    } else if (controls.size() == 1U) {
      builder.cp(angle, controls.front(), qubit);
    } else {
      builder.mcp(angle, controls, qubit);
    }
  });
}

static void modularAdd(qc::QCProgramBuilder& builder, Value accumulator,
                       const PhaseData& data, Value addendOffset,
                       ValueRange controls, Value work, bool inverse,
                       std::optional<size_t> cutoff) {
  auto overflowIndex = builder.indexConstant(data.width - 1);

  if (inverse) {
    /// Reverse the modular-adder operations and every phase rotation.
    phaseAdd(builder, accumulator, data, addendOffset, controls, true);
    inverseQFT(builder, accumulator, data.width, cutoff);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    builder.x(builder.loadQubit(accumulator, overflowIndex));
    forwardQFT(builder, accumulator, data.width, cutoff);
    phaseAdd(builder, accumulator, data, addendOffset, controls, false);
    phaseAdd(builder, accumulator, data, data.modulusOffset, work, true);
    inverseQFT(builder, accumulator, data.width, cutoff);
    builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
    forwardQFT(builder, accumulator, data.width, cutoff);
    phaseAdd(builder, accumulator, data, data.modulusOffset, {}, false);
    phaseAdd(builder, accumulator, data, addendOffset, controls, true);
    return;
  }
  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
  phaseAdd(builder, accumulator, data, data.modulusOffset, {}, true);

  inverseQFT(builder, accumulator, data.width, cutoff);
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  forwardQFT(builder, accumulator, data.width, cutoff);

  phaseAdd(builder, accumulator, data, data.modulusOffset, work, false);
  phaseAdd(builder, accumulator, data, addendOffset, controls, true);

  inverseQFT(builder, accumulator, data.width, cutoff);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  forwardQFT(builder, accumulator, data.width, cutoff);

  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
}

void multiplyAccumulate(qc::QCProgramBuilder& builder, Value control,
                        Value multiplicand, Value accumulator, Value work,
                        Value angles, Value offset, int64_t bits, bool inverse,
                        std::optional<size_t> cutoff) {
  const auto width = bits + 1;
  auto stride = builder.indexConstant(width);
  auto modulusRow = builder.indexConstant(bits * width);
  const PhaseData phases{
      .width = width,
      .angles = angles,
      .modulusOffset = arith::AddIOp::create(builder, offset, modulusRow),
      .negativeOne = builder.floatConstant(-1.),
  };
  forwardQFT(builder, accumulator, width, cutoff);
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
               inverse, cutoff);
  });
  inverseQFT(builder, accumulator, width, cutoff);
}

func::FuncOp createInPlaceMultiplier(qc::QCProgramBuilder& builder,
                                     int64_t bits, RankedTensorType anglesType,
                                     std::optional<size_t> cutoff) {
  auto qubitType = qc::QubitType::get(builder.getContext());
  SmallVector<Type> types{
      qubitType,
      MemRefType::get({bits}, qubitType),
      MemRefType::get({bits + 1}, qubitType),
      qubitType,
      anglesType,
      builder.getIndexType(),
  };
  const auto createAccumulator = [&](StringRef name, bool inverse) {
    return builder.createFunction(name, types, [&](ValueRange arguments) {
      multiplyAccumulate(builder, arguments[0], arguments[1], arguments[2],
                         arguments[3], arguments[4], arguments[5], bits,
                         inverse, cutoff);
      return SmallVector<Value>{};
    });
  };
  auto accumulate = createAccumulator("shor_accumulate", false);
  auto subtract = createAccumulator("shor_uncompute", true);
  return builder.createFunction(
      "shor_multiply", types, [&](ValueRange arguments) {
        builder.call(accumulate, arguments);
        builder.scfFor(0, bits, 1, [&](Value index) {
          builder.cswap(arguments[0], builder.loadQubit(arguments[1], index),
                        builder.loadQubit(arguments[2], index));
        });
        auto inverseOffset = arith::AddIOp::create(
            builder, arguments[5],
            builder.indexConstant((bits + 1) * (bits + 1)));
        SmallVector<Value> inverseArguments(arguments);
        inverseArguments.back() = inverseOffset;
        builder.call(subtract, inverseArguments);
        return SmallVector<Value>{};
      });
}

} // namespace mqt::bench::detail
