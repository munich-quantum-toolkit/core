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
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

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

  auto multiplier = arith::ConstantOp::create(
      builder, builder.getIntegerAttr(builder.getI64Type(),
                                      llvm::APInt(64, options.multiplier, 2)));
  auto modulus = arith::ConstantOp::create(
      builder, builder.getIntegerAttr(builder.getI64Type(),
                                      llvm::APInt(64, options.modulus, 2)));
  detail::multiplyAccumulate(builder, control, multiplicand, accumulator, work,
                             multiplier, modulus, bits);

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
