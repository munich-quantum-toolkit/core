/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/MagicStateDistillation.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mqt::bench {

using namespace mlir;

static void expectDistillationCounts(QCProgram program,
                                     const std::string& expected,
                                     size_t shots = 4) {
  auto compiled =
      runDefaultPipeline(CompilerInput{std::move(program)}, ProgramFormat::QCO);
  ASSERT_TRUE(compiled);
  auto& qcoProgram = std::get<QCOProgram>(*compiled);
  auto counts =
      qco::sample(mlir::mqt::getEntryPoint(qcoProgram.module()), shots, 17);
  ASSERT_TRUE(succeeded(counts));
  EXPECT_EQ(*counts, (Counts{{expected, shots}}));
}

static void expectDistillationFaults(size_t levels, uint32_t errors,
                                     const std::string& expected,
                                     size_t shots = 4) {
  auto program = generate(MagicStateDistillation({.levels = levels}));
  ASSERT_TRUE(program);
  auto entryPoint = mlir::mqt::getEntryPoint(program->module());
  SmallVector<qc::TOp> rotations;
  program->module().walk([&](qc::TOp op) {
    if (op->getParentOfType<func::FuncOp>() != entryPoint) {
      rotations.push_back(op);
    }
  });
  ASSERT_EQ(rotations.size(), 15U);
  for (size_t i = 0; i < rotations.size(); ++i) {
    if ((errors & (1U << i)) == 0) {
      continue;
    }
    // A Z fault between T and parity uncomputation is the P Pauli fault
    // on the corresponding π/8 rotation in Litinski's error model.
    auto op = rotations[i];
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    qc::ZOp::create(builder, op.getLoc(), op.getQubit(0));
  }
  expectDistillationCounts(std::move(*program), expected, shots);
}

TEST(GenerateProgramTest, SamplesMagicStateDistillation) {
  for (const size_t levels : {1U, 2U}) {
    SCOPED_TRACE(levels);
    expectDistillationFaults(levels, 0, "00", 16);
  }
}

TEST(GenerateProgramTest, KeepsConcatenatedMagicStateDistillationCompact) {
  for (const size_t levels : {1U, 2U, 3U, 4U, 8U}) {
    SCOPED_TRACE(levels);
    auto program = generate(MagicStateDistillation({.levels = levels}));
    ASSERT_TRUE(program);
    auto moduleOp = program->module();
    EXPECT_EQ(test::countOps<memref::AllocOp>(moduleOp), 1U);
    moduleOp.walk([&](memref::AllocOp op) {
      EXPECT_EQ(op.getType().getNumElements(),
                static_cast<int64_t>(5 * levels));
    });
    auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                       ProgramFormat::QCO);
    ASSERT_TRUE(compiled);
    auto& qcoProgram = std::get<QCOProgram>(*compiled);
    EXPECT_LT(test::countOperations(qcoProgram.module()), 500U * levels);

    auto jeff = std::move(qcoProgram).intoJeff();
    ASSERT_TRUE(jeff);
    const auto bytes = jeff->toBytes();
    ASSERT_FALSE(bytes.empty());
    auto restored = JeffProgram::fromBytes(bytes);
    ASSERT_TRUE(restored);
    EXPECT_EQ(restored->toBytes(), bytes);
  }
}

TEST(GenerateProgramTest,
     DistillationRejectsRotationErrorsAndDetectsLogicalErrors) {
  // Columns of Litinski's Fig. 3, encoded with the output qubit as bit 0.
  // XOR gives the root Z error and the four X-check syndromes independently.
  constexpr std::array<uint32_t, 15> columns{
      2, 4, 8, 16, 14, 7, 11, 13, 25, 19, 21, 31, 28, 26, 22,
  };
  size_t undetectedTriples = 0;
  for (uint32_t errors = 0; errors < (1U << 15U); ++errors) {
    const auto weight = std::popcount(errors);
    if (weight < 1 || weight > 3) {
      continue;
    }
    uint32_t pauliError = 0;
    for (size_t i = 0; i < columns.size(); ++i) {
      if ((errors & (1U << i)) != 0) {
        pauliError ^= columns[i];
      }
    }
    const auto syndrome = pauliError >> 1U;
    if (weight == 3 && syndrome != 0) {
      continue;
    }
    SCOPED_TRACE(errors);
    undetectedTriples += weight == 3 ? 1U : 0U;
    const std::string expected{
        syndrome == 0 ? '0' : '1',
        (pauliError & 1U) == 0 ? '0' : '1',
    };
    expectDistillationFaults(1, errors, expected);
  }
  EXPECT_EQ(undetectedTriples, 35U);
}

TEST(GenerateProgramTest, ConcatenatedDistillationConsumesRetainedStates) {
  // Reject only the first child, retaining its ideal output state, as with a
  // check-only fault. Subsequent accepting children must not clear rejection.
  auto program = generate(MagicStateDistillation({.levels = 2}));
  ASSERT_TRUE(program);
  auto entryPoint = mlir::mqt::getEntryPoint(program->module());
  const auto injected = program->module().walk([&](func::CallOp op) {
    if (op->getParentOfType<func::FuncOp>() == entryPoint) {
      return WalkResult::advance();
    }
    OpBuilder builder(op);
    auto rejected = arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
    op.getResult(0).replaceAllUsesWith(rejected);
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(injected.wasInterrupted());
  expectDistillationCounts(std::move(*program), "10");

  // The undetected triple at rotations 5, 11, 14 flips every lower output.
  // The resulting 15 faulty higher-level rotations also leave a root Z error.
  constexpr auto errors = (1U << 4U) | (1U << 10U) | (1U << 13U);
  expectDistillationFaults(2, errors, "01");
}

} // namespace mqt::bench
