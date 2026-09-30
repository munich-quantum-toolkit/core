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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace mqt::bench {

using namespace mlir;

TEST(GenerateProgramTest, SamplesMagicStateDistillation) {
  auto program = test::generateQCO(MagicStateDistillation{});
  ASSERT_TRUE(program);
  auto counts =
      qco::sample(mlir::mqt::getEntryPoint(program->module()), 16, 17);
  ASSERT_TRUE(succeeded(counts));
  EXPECT_EQ(*counts, (Counts{{"00", 16}}));
}

TEST(GenerateProgramTest, KeepsConcatenatedMagicStateDistillationCompact) {
  int64_t qubitCount = 1;
  for (const size_t levels : {1U, 2U, 3U, 4U}) {
    qubitCount *= 15;
    auto program = generate(MagicStateDistillation({.levels = levels}));
    ASSERT_TRUE(program);
    auto moduleOp = program->module();
    EXPECT_EQ(test::countOps<memref::AllocOp>(moduleOp), 1U);
    moduleOp.walk([&](memref::AllocOp op) {
      EXPECT_EQ(op.getType().getNumElements(), qubitCount);
    });
    auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                       ProgramFormat::QCO);
    ASSERT_TRUE(compiled);
    auto& qcoProgram = std::get<QCOProgram>(*compiled);
    EXPECT_LT(test::countOperations(qcoProgram.module()), 2000U);
  }
}

TEST(GenerateProgramTest,
     DistillationRejectsInputErrorsAndDetectsLogicalErrors) {
  // Independently, the four X-check syndromes are the XOR of the erroneous
  // columns 1..15; the logical Z error is their weight modulo two.
  for (uint32_t errors = 0; errors < (1U << 15U); ++errors) {
    const auto weight = std::popcount(errors);
    if (weight < 1 || weight > 3) {
      continue;
    }
    uint32_t syndrome = 0;
    for (uint32_t i = 0; i < 15; ++i) {
      if ((errors & (1U << i)) != 0) {
        syndrome ^= i + 1;
      }
    }
    if (weight == 3 && syndrome != 0) {
      continue;
    }
    SCOPED_TRACE(errors);
    auto program = generate(MagicStateDistillation{});
    ASSERT_TRUE(program);
    // Insert faults in the leaf preparation only. The private block and public
    // rejection readout remain exactly those emitted for the benchmark.
    program->module().walk([&](qc::TOp op) {
      auto qubit = op.getQubit(0);
      auto index = qubit.getDefiningOp<memref::LoadOp>().getIndices().front();
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      auto loc = op.getLoc();
      for (uint32_t i = 0; i < 15; ++i) {
        if ((errors & (1U << i)) == 0) {
          continue;
        }
        auto position = arith::ConstantIndexOp::create(builder, loc, i);
        auto condition = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, index, position);
        scf::IfOp::create(builder, loc, condition,
                          [&](OpBuilder& bodyBuilder, Location bodyLoc) {
                            qc::ZOp::create(bodyBuilder, bodyLoc, qubit);
                            scf::YieldOp::create(bodyBuilder, bodyLoc);
                          });
      }
    });
    auto compiled = runDefaultPipeline(CompilerInput{std::move(*program)},
                                       ProgramFormat::QCO);
    ASSERT_TRUE(compiled);
    auto& qcoProgram = std::get<QCOProgram>(*compiled);
    auto counts =
        qco::sample(mlir::mqt::getEntryPoint(qcoProgram.module()), 4, 17);
    ASSERT_TRUE(succeeded(counts));
    const std::string expected{
        syndrome == 0 ? '0' : '1',
        weight % 2 == 0 ? '0' : '1',
    };
    EXPECT_EQ(*counts, (Counts{{expected, 4}}));
  }
}

} // namespace mqt::bench
