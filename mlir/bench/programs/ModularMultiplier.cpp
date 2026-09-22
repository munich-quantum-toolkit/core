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

#include "ModularArithmetic.h"
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

  SmallVector<double> angles;
  detail::appendModularPhaseAngles(
      angles, llvm::APInt(static_cast<unsigned>(width), options.multiplier, 2),
      llvm::APInt(static_cast<unsigned>(width), options.modulus, 2));
  auto type = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                    builder.getF64Type());
  auto phases = arith::ConstantOp::create(
      builder, DenseElementsAttr::get(type, ArrayRef<double>(angles)));
  detail::multiplyAccumulate(builder, control, multiplicand, accumulator, work,
                             phases, builder.indexConstant(0), bits);

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
