/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <iterator>
#include <llvm/Support/Debug.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <utility>

using namespace mlir;

namespace {
class WireIteratorTest : public testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<qco::QCODialect, arith::ArithDialect, cf::ControlFlowDialect,
                func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;
};
} // namespace

TEST_P(WireIteratorTest, MixedUse) {
  const bool isDynamic = GetParam();

  // Build circuit.
  qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto q00 = isDynamic ? builder.allocQubit() : builder.staticQubit(0);
  const auto q10 = isDynamic ? builder.allocQubit() : builder.staticQubit(1);
  const auto q01 = builder.h(q00);
  const auto [q02, q11] = builder.cx(q01, q10);
  const auto [q03, c0] = builder.measure(q02);
  const auto q04 = builder.reset(q03);
  builder.dealloc(q04);
  builder.dealloc(q11);

  // Setup WireIterator.
  auto module = builder.finalize();
  qco::WireIterator it(q00);

  //
  // Test: Forward Iteration
  //

  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc
  ASSERT_EQ(it.qubit(), q00);

  ++it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q01);

  ++it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q02);

  ++it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q03);

  ++it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q04);

  ++it;
  ASSERT_EQ(it.operation(), *(q04.getUsers().begin())); // qco.dealloc
  ASSERT_EQ(it.qubit(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  //
  // Test: Backward Iteration
  //

  --it;
  ASSERT_EQ(it.operation(), *(q04.getUsers().begin())); // qco.dealloc
  ASSERT_EQ(it.qubit(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), q04.getDefiningOp()); // qco.reset
  ASSERT_EQ(it.qubit(), q04);

  --it;
  ASSERT_EQ(it.operation(), q03.getDefiningOp()); // qco.measure
  ASSERT_EQ(it.qubit(), q03);

  --it;
  ASSERT_EQ(it.operation(), q02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(it.qubit(), q02);

  --it;
  ASSERT_EQ(it.operation(), q01.getDefiningOp()); // qco.h
  ASSERT_EQ(it.qubit(), q01);

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc or qco.static
  ASSERT_EQ(it.qubit(), q00);
}

INSTANTIATE_TEST_SUITE_P(DynamicAndStatic, WireIteratorTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Dynamic" : "Static";
                         });
