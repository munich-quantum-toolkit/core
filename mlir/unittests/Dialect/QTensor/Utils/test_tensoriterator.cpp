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
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>

using namespace mlir;
using namespace mlir::qtensor;
using namespace mlir::qco;

namespace {

class TensorIteratorTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    scf::SCFDialect, QTensorDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};
} // namespace

TEST_F(TensorIteratorTest, Traversal) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  constexpr int64_t n = 3;
  auto tensor0 = builder.qtensorAlloc(n);
  auto [tensor1, q00] = builder.qtensorExtract(tensor0, 0);
  auto q01 = builder.h(q00);
  auto tensor2 = builder.qtensorInsert(q01, tensor1, 0);
  auto [tensor3, q02] = builder.qtensorExtract(tensor2, 0);
  auto [tensor4, q10] = builder.qtensorExtract(tensor3, 1);
  auto [q03, q11] = builder.cx(q02, q10);
  auto tensor5 = builder.qtensorInsert(q03, tensor4, 0);
  auto tensor6 = builder.qtensorInsert(q11, tensor5, 1);
  auto tensor7 = builder.scfFor(
      1, n, 1, {tensor6}, [&builder](Value iv, ValueRange iterArgs) {
        Value loopTensor = iterArgs[0];
        Value q;
        std::tie(loopTensor, q) = builder.qtensorExtract(loopTensor, iv);
        q = builder.h(q);
        loopTensor = builder.qtensorInsert(q, loopTensor, 0);
        return SmallVector{loopTensor};
      })[0];

  Value tensorThen0;
  Value tensorThen1;
  Value tensorThen2;

  Value tensorElse0;
  Value tensorElse1;
  Value tensorElse2;

  auto tensor8 = builder.qcoIf(
      false, tensor7,
      [&](ValueRange args) -> SmallVector<Value> {
        Value q;
        tensorThen0 = args[0];
        std::tie(tensorThen1, q) = builder.qtensorExtract(tensorThen0, 0);
        q = builder.h(q);
        tensorThen2 = builder.qtensorInsert(q, tensorThen1, 0);
        return SmallVector{tensorThen2};
      },
      [&](ValueRange args) -> SmallVector<Value> {
        Value q;
        tensorElse0 = args[0];
        std::tie(tensorElse1, q) = builder.qtensorExtract(tensorElse0, 0);
        q = builder.t(q);
        tensorElse2 = builder.qtensorInsert(q, tensorElse1, 0);
        return SmallVector{tensorElse2};
      })[0];
  const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
  const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
      identity};
  const auto tensor9 = builder.qcoIndexSwitch(
      0, tensor8, SmallVector<int64_t>{0}, caseBodies, identity)[0];
  builder.qtensorDealloc(tensor9);
  [[maybe_unused]] auto m = builder.finalize();

  TensorIterator it(cast<TypedValue<RankedTensorType>>(tensor0));

  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  ++it;
  ASSERT_EQ(it.operation(), tensor1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor1);

  ++it;
  ASSERT_EQ(it.operation(), tensor2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor2);

  ++it;
  ASSERT_EQ(it.operation(), tensor3.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor3);

  ++it;
  ASSERT_EQ(it.operation(), tensor4.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor4);

  ++it;
  ASSERT_EQ(it.operation(), tensor5.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor5);

  ++it;
  ASSERT_EQ(it.operation(), tensor6.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor6);

  ++it;
  ASSERT_EQ(it.operation(), tensor7.getDefiningOp()); // scf.for
  ASSERT_EQ(it.tensor(), tensor7);

  ++it;
  ASSERT_EQ(it.operation(), tensor8.getDefiningOp()); // qco.if
  ASSERT_EQ(it.tensor(), tensor8);

  ++it;
  ASSERT_EQ(it.operation(), tensor9.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.tensor(), tensor9);

  ++it;
  ASSERT_EQ(it.operation(), *(tensor9.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_EQ(it.operation(), *(tensor9.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

  --it;
  ASSERT_EQ(it.operation(), tensor9.getDefiningOp()); // qco.index_switch
  ASSERT_EQ(it.tensor(), tensor9);

  --it;
  ASSERT_EQ(it.operation(), tensor8.getDefiningOp()); // qco.if
  ASSERT_EQ(it.tensor(), tensor8);

  --it;
  ASSERT_EQ(it.operation(), tensor7.getDefiningOp()); // scf.for
  ASSERT_EQ(it.tensor(), tensor7);

  --it;
  ASSERT_EQ(it.operation(), tensor6.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor6);

  --it;
  ASSERT_EQ(it.operation(), tensor5.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(it.tensor(), tensor5);

  --it;
  ASSERT_EQ(it.operation(), tensor4.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor4);

  --it;
  ASSERT_EQ(it.operation(), tensor3.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor3);

  --it;
  ASSERT_EQ(it.operation(), tensor2.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor2);

  --it;
  ASSERT_EQ(it.operation(), tensor1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(it.tensor(), tensor1);

  --it;
  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  --it;
  ASSERT_EQ(it.operation(), tensor0.getDefiningOp()); // qtensor.alloc
  ASSERT_EQ(it.tensor(), tensor0);

  //
  // Test recursive use with block-argument.
  //

  TensorIterator recIt(cast<TypedValue<RankedTensorType>>(tensorElse0));

  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);

  ++recIt;
  ASSERT_EQ(recIt.operation(), tensorElse1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(recIt.tensor(), tensorElse1);

  ++recIt;
  ASSERT_EQ(recIt.operation(), tensorElse2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(recIt.tensor(), tensorElse2);

  ++recIt;
  ASSERT_EQ(recIt.operation(), *(tensorElse2.user_begin())); // qco.yield
  ASSERT_EQ(recIt.tensor(), nullptr);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  ++recIt;
  ASSERT_EQ(recIt, std::default_sentinel);

  --recIt;
  ASSERT_EQ(recIt.operation(), *(tensorElse2.user_begin())); // qco.yield
  ASSERT_EQ(recIt.tensor(), nullptr);

  --recIt;
  ASSERT_EQ(recIt.operation(), tensorElse2.getDefiningOp()); // qtensor.insert
  ASSERT_EQ(recIt.tensor(), tensorElse2);

  --recIt;
  ASSERT_EQ(recIt.operation(), tensorElse1.getDefiningOp()); // qtensor.extract
  ASSERT_EQ(recIt.tensor(), tensorElse1);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);

  --recIt;
  ASSERT_EQ(recIt.operation(), nullptr);
  ASSERT_EQ(recIt.tensor(), tensorElse0);
}
