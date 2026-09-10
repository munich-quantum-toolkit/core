/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ModularMultiplier.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"
#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

namespace {

struct PhaseData {
  int64_t width;
  Value angles;
  Value rowStride;
  Value modulusOffset;
  Value negativeOne;
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

[[nodiscard]] static PhaseData phaseData(qc::QCProgramBuilder& builder,
                                         const StringRef multiplier,
                                         const StringRef modulus) {
  const auto bits = multiplier.size();
  const auto width = static_cast<unsigned>(bits + 1U);
  auto addend = llvm::APInt(width, multiplier, /*radix=*/2);
  const auto modulusValue = llvm::APInt(width, modulus, /*radix=*/2);

  SmallVector<double> angles;
  angles.reserve((bits + 1U) * width);
  for (size_t bit = 0; bit < bits; ++bit) {
    appendPhaseAngles(angles, addend);
    addend = addend.shl(1).urem(modulusValue);
  }
  appendPhaseAngles(angles, modulusValue);

  const auto type = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                          builder.getF64Type());
  const auto value = DenseElementsAttr::get(type, ArrayRef<double>(angles));
  return {
      .width = static_cast<int64_t>(width),
      .angles = arith::ConstantOp::create(builder, value).getResult(),
      .rowStride = builder.indexConstant(static_cast<int64_t>(width)),
      .modulusOffset =
          builder.indexConstant(static_cast<int64_t>(bits * width)),
      .negativeOne = builder.floatConstant(-1.),
  };
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
                       ValueRange controls, Value work) {
  auto overflowIndex = builder.indexConstant(data.width - 1);

  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
  phaseAdd(builder, accumulator, data, data.modulusOffset, {}, true);

  detail::inverseQFT(builder, accumulator, data.width);
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  detail::forwardQFT(builder, accumulator, data.width);

  phaseAdd(builder, accumulator, data, data.modulusOffset, work, false);
  phaseAdd(builder, accumulator, data, addendOffset, controls, true);

  detail::inverseQFT(builder, accumulator, data.width);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  builder.cx(builder.loadQubit(accumulator, overflowIndex), work);
  builder.x(builder.loadQubit(accumulator, overflowIndex));
  detail::forwardQFT(builder, accumulator, data.width);

  phaseAdd(builder, accumulator, data, addendOffset, controls, false);
}

SmallVector<Value> modularMultiplier(qc::QCProgramBuilder& builder,
                                     const ModularMultiplier& benchmark) {
  const auto& options = benchmark.options();
  const auto bits = static_cast<int64_t>(options.modulus.size());
  const auto width = bits + 1;

  auto control = builder.allocQubit();
  auto multiplicand = builder.allocQubitRegisterStorage(bits, "multiplicand");
  auto accumulator = builder.allocQubitRegisterStorage(width, "accumulator");
  auto work = builder.allocQubit();
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  if (options.control == '+') {
    builder.h(control);
  } else if (options.control == '1') {
    builder.x(control);
  }
  detail::prepareRegister(builder, multiplicand, options.multiplicand);

  detail::forwardQFT(builder, accumulator, width);

  const auto phases = phaseData(builder, options.multiplier, options.modulus);
  builder.scfFor(0, bits, 1, [&](Value index) {
    auto addendOffset =
        arith::MulIOp::create(builder, index, phases.rowStride).getResult();
    SmallVector<Value, 2> controls{
        control,
        builder.loadQubit(multiplicand, index),
    };
    modularAdd(builder, accumulator, phases, addendOffset, controls, work);
  });

  detail::inverseQFT(builder, accumulator, width);

  builder.measureQubitRegister(accumulator, result, width);
  auto multiplicandOffset = builder.indexConstant(width);
  builder.scfFor(0, bits, 1, [&](Value index) {
    auto resultIndex =
        arith::AddIOp::create(builder, multiplicandOffset, index).getResult();
    builder.measure(builder.loadQubit(multiplicand, index), result,
                    resultIndex);
  });
  builder.measure(control, result, 2 * bits + 1);
  return {result};
}

} // namespace mqt::bench
