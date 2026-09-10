/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>

namespace mqt::bench::test {

template <class Benchmark>
[[nodiscard]] std::optional<mlir::QCOProgram>
generateQCO(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  if (!program) {
    return std::nullopt;
  }
  auto compiled = mlir::runDefaultPipeline(
      mlir::CompilerInput{std::move(*program)}, mlir::ProgramFormat::QCO);
  if (!compiled) {
    return std::nullopt;
  }
  return std::get<mlir::QCOProgram>(std::move(*compiled));
}

template <class Benchmark>
void expectSamplingMatchesReference(const Benchmark& benchmark,
                                    double tolerance = 0.03) {
  auto program = generateQCO(benchmark);
  ASSERT_TRUE(program);
  constexpr size_t shots = 16'384;
  auto counts =
      mlir::qco::sample(mlir::mqt::getEntryPoint(program->module()), shots, 17);
  ASSERT_TRUE(mlir::succeeded(counts));
  EXPECT_LT(benchmark.evaluate(*counts).totalVariationDistance, tolerance);
}

[[nodiscard]] inline mlir::DenseElementsAttr
angleTable(mlir::ModuleOp moduleOp) {
  mlir::DenseElementsAttr result;
  moduleOp.walk([&](mlir::arith::ConstantOp op) {
    if (auto table = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue())) {
      EXPECT_FALSE(result);
      result = table;
    }
  });
  return result;
}

template <class Op> [[nodiscard]] size_t countOps(mlir::ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&count](Op /*unused*/) { ++count; });
  return count;
}

[[nodiscard]] inline size_t countOperations(mlir::ModuleOp moduleOp) {
  size_t count = 0;
  moduleOp.walk([&count](mlir::Operation* /*unused*/) { ++count; });
  return count;
}

inline void expectJeffRoundTrip(mlir::QCProgram&& program) {
  auto compiled = mlir::runDefaultPipeline(
      mlir::CompilerInput{std::move(program)}, mlir::ProgramFormat::Jeff);
  ASSERT_TRUE(compiled);
  ASSERT_TRUE(std::holds_alternative<mlir::JeffProgram>(*compiled));
  auto& jeff = std::get<mlir::JeffProgram>(*compiled);
  const auto bytes = jeff.toBytes();
  ASSERT_FALSE(bytes.empty());
  auto restored = mlir::JeffProgram::fromBytes(bytes);
  ASSERT_TRUE(restored);
  EXPECT_EQ(restored->toBytes(), bytes);
}

} // namespace mqt::bench::test
