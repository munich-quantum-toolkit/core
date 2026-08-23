/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/BV.hpp"

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::bench {

using namespace mlir;

namespace {

[[nodiscard]] Value hiddenBits(qc::QCProgramBuilder& builder,
                               const BV& benchmark) {
  SmallVector<bool> bits;
  bits.reserve(benchmark.options().hiddenBitstring.size());
  for (auto bit = benchmark.options().hiddenBitstring.rbegin();
       bit != benchmark.options().hiddenBitstring.rend(); ++bit) {
    bits.push_back(*bit == '1');
  }
  const auto type = RankedTensorType::get(
      {static_cast<int64_t>(bits.size())}, builder.getI1Type());
  const auto value = DenseElementsAttr::get(type, ArrayRef<bool>(bits));
  return arith::ConstantOp::create(builder, value).getResult();
}

} // namespace

SmallVector<Value> bv(qc::QCProgramBuilder& builder, const BV& benchmark) {
  const auto width = static_cast<int64_t>(benchmark.output().width);
  auto flag = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(width, benchmark.output().name);
  auto hidden = hiddenBits(builder, benchmark);
  builder.x(flag);

  if (benchmark.options().method == BVMethod::Dynamic) {
    auto query = builder.allocQubit();
    builder.scfFor(0, width, 1, [&](Value index) {
      builder.h(query);
      auto bit =
          tensor::ExtractOp::create(builder, hidden, ValueRange{index})
              .getResult();
      builder.scfIf(bit, [&] { builder.cz(query, flag); });
      builder.h(query);
      builder.measure(query, result, index);
      builder.reset(query);
    });
    return {result};
  }

  auto query = builder.allocQubitRegisterStorage(width, "query");
  builder.scfFor(0, width, 1,
                 [&](Value index) { builder.h(builder.loadQubit(query, index)); });
  builder.scfFor(0, width, 1, [&](Value index) {
    auto bit = tensor::ExtractOp::create(builder, hidden, ValueRange{index})
                   .getResult();
    builder.scfIf(
        bit, [&] { builder.cz(builder.loadQubit(query, index), flag); });
  });
  builder.scfFor(0, width, 1,
                 [&](Value index) { builder.h(builder.loadQubit(query, index)); });
  builder.measureQubitRegister(query, result, width);
  return {result};
}

} // namespace mqt::bench
