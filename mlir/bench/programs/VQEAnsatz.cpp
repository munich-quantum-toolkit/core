/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/VQEAnsatz.hpp"

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

SmallVector<Value> vqeAnsatz(qc::QCProgramBuilder& builder,
                             const VQEAnsatz& benchmark) {
  const auto& options = benchmark.options();
  const auto size = static_cast<int64_t>(options.qubits);
  const auto layers = static_cast<int64_t>(options.depth);
  auto entry = cast<func::FuncOp>(builder.getInsertionBlock()->getParentOp());
  const auto anglesType = RankedTensorType::get(
      {static_cast<int64_t>(benchmark.numParameters())}, builder.getF64Type());
  entry.insertArgument(
      entry.getNumArguments(), anglesType,
      builder.getDictionaryAttr({
          builder.getNamedAttr(
              mlir::mqt::MQTDialect::InputNameAttrHelper::getNameStr(),
              builder.getStringAttr("thetas")),
      }),
      builder.getLoc());
  auto thetas = entry.getArgument(0);
  auto qubits = builder.allocQubitRegisterStorage(size, "qubits");
  auto result =
      builder.allocClassicalBitRegister(size, benchmark.output().name);
  auto width = builder.indexConstant(size);
  auto one = builder.indexConstant(1);
  const auto rotate = [&](Value layer) {
    auto offset = arith::MulIOp::create(builder, layer, width);
    builder.scfFor(0, size, 1, [&](Value iv) {
      auto index = arith::AddIOp::create(builder, offset, iv);
      auto theta =
          tensor::ExtractOp::create(builder, thetas, ValueRange{index});
      builder.ry(theta, builder.loadQubit(qubits, iv));
    });
  };
  builder.scfFor(0, layers, 1, [&](Value layer) {
    rotate(layer);
    builder.scfFor(0, size - 1, 1, [&](Value iv) {
      auto next = arith::AddIOp::create(builder, iv, one);
      builder.cx(builder.loadQubit(qubits, iv),
                 builder.loadQubit(qubits, next));
    });
  });
  rotate(builder.indexConstant(layers));
  builder.measureQubitRegister(qubits, result, size);
  return {result};
}

} // namespace mqt::bench
