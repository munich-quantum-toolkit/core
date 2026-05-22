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
  qco::QCOProgramBuilder builder(context.get());
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
  builder.qtensorDealloc(tensor7);
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
  ASSERT_EQ(it.operation(), *(tensor7.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  ++it;
  ASSERT_EQ(it, std::default_sentinel);

  --it;
  ASSERT_EQ(it.operation(), *(tensor7.user_begin())); // qtensor.dealloc
  ASSERT_EQ(it.tensor(), nullptr);

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
}
