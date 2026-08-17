/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "programs/qco_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <memory>
#include <string>

static mlir::Value unobservedH(mlir::qco::QCOProgramBuilder& builder) {
  auto qubit = builder.allocQubit();
  qubit = builder.h(qubit);
  builder.sink(qubit);
  return builder.intConstant(0);
}

namespace {

using namespace mlir;
using namespace mlir::qco;

struct RemoveDeadGatesTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<QCOProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<QCOProgramBuilder> referenceBuilder;
};

class RemoveDeadGatesTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    qtensor::QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static LogicalResult runRemoveDeadGates(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createRemoveDeadGates());
    return pm.run(module);
  }
};

class RemoveDeadGatesParameterizedTest
    : public RemoveDeadGatesTest,
      public testing::WithParamInterface<RemoveDeadGatesTestCase> {};

TEST_P(RemoveDeadGatesParameterizedTest, ProducesExpectedProgram) {
  const auto& testCase = GetParam();
  auto program =
      ::mqt::test::buildMLIRProgram(context.get(), testCase.programBuilder);
  ASSERT_TRUE(program);
  ASSERT_TRUE(runRemoveDeadGates(*program).succeeded());
  ASSERT_TRUE(runQCOCleanupPipeline(*program).succeeded());

  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), testCase.referenceBuilder);
  ASSERT_TRUE(reference);
  ASSERT_TRUE(runQCOCleanupPipeline(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

TEST_F(RemoveDeadGatesTest, QCOCleanupPreservesDeadGates) {
  auto program = ::mqt::test::buildMLIRProgram(context.get(),
                                               MQT_NAMED_BUILDER(unobservedH));
  ASSERT_TRUE(program);
  ASSERT_TRUE(runQCOCleanupPipeline(*program).succeeded());
  const auto countH = [&program] {
    size_t count = 0;
    program->walk([&](HOp) { ++count; });
    return count;
  };
  EXPECT_EQ(countH(), 1U);

  ASSERT_TRUE(runRemoveDeadGates(*program).succeeded());
  EXPECT_EQ(countH(), 0U);
}

INSTANTIATE_TEST_SUITE_P(
    DeadGateRemoval, RemoveDeadGatesParameterizedTest,
    testing::Values(RemoveDeadGatesTestCase{"Sink",
                                            MQT_NAMED_BUILDER(deadGatesProgram),
                                            MQT_NAMED_BUILDER(alloc2Qubits)},
                    RemoveDeadGatesTestCase{
                        "Reset", MQT_NAMED_BUILDER(deadGatesResetProgram),
                        MQT_NAMED_BUILDER(allocQubit)},
                    RemoveDeadGatesTestCase{
                        "IfOp", MQT_NAMED_BUILDER(deadGatesWithIfOpProgram),
                        MQT_NAMED_BUILDER(deadGatesWithIfOpSimplified)}),
    [](const testing::TestParamInfo<RemoveDeadGatesTestCase>& info) {
      return info.param.name;
    });

} // namespace
