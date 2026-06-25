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
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;

namespace {
class WireIteratorTest : public testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qco::QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  std::unique_ptr<MLIRContext> context;
};
} // namespace

TEST_P(WireIteratorTest, Traversal) {
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

  Value iterQ00;
  Value iterQ01;
  Value iterQ02;
  Value iterQ10;
  Value iterQ11;

  const auto loopOut =
      builder.scfFor(1, 4, 1, {q04, q11}, [&](Value, ValueRange iterArgs) {
        iterQ00 = iterArgs[0];
        iterQ10 = iterArgs[1];
        iterQ01 = builder.h(iterQ00);
        std::tie(iterQ02, iterQ11) = builder.cx(iterQ01, iterQ10);
        return SmallVector{iterQ02, iterQ11};
      });
  const auto q05 = loopOut[0];
  const auto q12 = loopOut[1];
  const auto ifOut = builder.qcoIf(
      true, {q05, q12},
      [&](ValueRange args) { return SmallVector{args[0], args[1]}; },
      [&](ValueRange args) { return SmallVector{args[0], args[1]}; });
  const auto q06 = ifOut[0];
  const auto q13 = ifOut[1];
  builder.sink(q06);
  builder.sink(q13);
  [[maybe_unused]] auto module = builder.finalize();

  // Setup WireIterator.
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
  ASSERT_EQ(it.operation(), q05.getDefiningOp()); // scf.for
  ASSERT_EQ(it.qubit(), q05);

  ++it;
  ASSERT_EQ(it.operation(), q06.getDefiningOp()); // qco.if
  ASSERT_EQ(it.qubit(), q06);

  ++it;
  ASSERT_EQ(it.operation(), *(q06.getUsers().begin())); // qco.sink
  ASSERT_EQ(it.qubit(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  //
  // Test: Backward Iteration
  //

  --it;
  ASSERT_EQ(it.operation(), *(q06.getUsers().begin())); // qco.sink
  ASSERT_EQ(it.qubit(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), q06.getDefiningOp()); // qco.if
  ASSERT_EQ(it.qubit(), q06);

  --it;
  ASSERT_EQ(it.operation(), q05.getDefiningOp()); // scf.for
  ASSERT_EQ(it.qubit(), q05);

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

  --it;
  ASSERT_EQ(it.operation(), q00.getDefiningOp()); // qco.alloc or qco.static
  ASSERT_EQ(it.qubit(), q00);

  //
  // Test: Recursive use with block-argument.
  //

  qco::WireIterator recIt(iterQ00);
  ASSERT_EQ(recIt.operation(), nullptr); // Blockargument
  ASSERT_EQ(recIt.qubit(), iterQ00);

  ++recIt;
  ASSERT_EQ(recIt.operation(), iterQ01.getDefiningOp()); // qco.h
  ASSERT_EQ(recIt.qubit(), iterQ01);

  ++recIt;
  ASSERT_EQ(recIt.operation(), iterQ02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(recIt.qubit(), iterQ02);

  ++recIt;
  ASSERT_EQ(recIt.operation(), *(iterQ02.getUsers().begin())); // scf.yield
  ASSERT_EQ(recIt.qubit(), nullptr);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  --recIt;
  ASSERT_EQ(recIt.operation(), *(iterQ02.getUsers().begin())); // scf.yield
  ASSERT_EQ(recIt.qubit(), nullptr);

  --recIt;
  ASSERT_EQ(recIt.operation(), iterQ02.getDefiningOp()); // qco.ctrl
  ASSERT_EQ(recIt.qubit(), iterQ02);

  --recIt;
  ASSERT_EQ(recIt.operation(), iterQ01.getDefiningOp()); // qco.h
  ASSERT_EQ(recIt.qubit(), iterQ01);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr); // Blockargument
  ASSERT_EQ(recIt.qubit(), iterQ00);
}

INSTANTIATE_TEST_SUITE_P(DynamicAndStatic, WireIteratorTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Dynamic" : "Static";
                         });
