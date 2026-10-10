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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <utility>
#include <variant>

namespace mqt::bench::test {

/// Fold emitted arithmetic with concrete loop arguments, without running
/// the quantum circuit.
[[nodiscard]] inline mlir::Attribute evaluateArithmetic(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Attribute>& arguments) {
  if (const auto found = arguments.find(value); found != arguments.end()) {
    return found->second;
  }
  auto* operation = value.getDefiningOp();
  if (operation == nullptr) {
    ADD_FAILURE() << "Missing concrete arithmetic argument";
    return {};
  }
  llvm::SmallVector<mlir::Attribute> operands;
  for (auto operand : operation->getOperands()) {
    auto attribute = evaluateArithmetic(operand, arguments);
    if (!attribute) {
      return {};
    }
    operands.push_back(attribute);
  }
  llvm::SmallVector<mlir::OpFoldResult> results;
  if (mlir::failed(operation->fold(operands, results)) || results.size() != 1) {
    ADD_FAILURE() << "Cannot fold "
                  << operation->getName().getStringRef().str();
    return {};
  }
  if (auto attribute = llvm::dyn_cast<mlir::Attribute>(results.front())) {
    return attribute;
  }
  return evaluateArithmetic(llvm::cast<mlir::Value>(results.front()),
                            arguments);
}

template <class Benchmark>
[[nodiscard]] mlir::FailureOr<mlir::QCOProgram>
generateQCO(const Benchmark& benchmark) {
  auto program = generate(benchmark);
  if (mlir::failed(program)) {
    return mlir::failure();
  }
  auto compiled = mlir::runDefaultPipeline(
      mlir::CompilerInput{std::move(*program)}, mlir::ProgramFormat::QCO);
  if (mlir::failed(compiled)) {
    return mlir::failure();
  }
  return std::get<mlir::QCOProgram>(std::move(*compiled));
}

template <class Benchmark>
void expectSamplingMatchesReference(const Benchmark& benchmark,
                                    double tolerance = 0.03) {
  auto program = generateQCO(benchmark);
  ASSERT_TRUE(mlir::succeeded(program));
  constexpr size_t shots = 16'384;
  auto counts =
      mlir::qco::sample(mlir::mqt::getEntryPoint(program->module()), shots, 17);
  ASSERT_TRUE(mlir::succeeded(counts));
  EXPECT_LT(benchmark.evaluate(*counts).totalVariationDistance, tolerance);
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
  ASSERT_TRUE(mlir::succeeded(compiled));
  ASSERT_TRUE(std::holds_alternative<mlir::JeffProgram>(*compiled));
  auto& jeff = std::get<mlir::JeffProgram>(*compiled);
  const auto bytes = jeff.toBytes();
  ASSERT_FALSE(bytes.empty());
  auto restored = mlir::JeffProgram::fromBytes(bytes);
  ASSERT_TRUE(mlir::succeeded(restored));
  EXPECT_EQ(restored->toBytes(), bytes);
}

} // namespace mqt::bench::test
