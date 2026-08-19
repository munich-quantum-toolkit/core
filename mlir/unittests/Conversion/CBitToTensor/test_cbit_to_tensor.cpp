/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CBitToTensor/CBitToTensor.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cstddef>
#include <utility>

using namespace mlir;

TEST(CBitToTensorTest, ConvertsInitializationLoadsAndStores) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                      tensor::TensorDialect>();
  OpBuilder builder(&context);
  const auto loc = builder.getUnknownLoc();
  auto moduleOp = ModuleOp::create(builder, loc);
  auto function = func::FuncOp::create(
      builder, loc, "main", builder.getFunctionType({}, builder.getI1Type()));
  moduleOp.getBody()->push_back(function);
  auto* block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);

  const auto registerType = cbit::RegisterType::get(&context, 2);
  auto zero = cbit::AllocOp::create(builder, loc, registerType,
                                    cbit::Initialization::Zero, StringAttr{});
  cbit::AllocOp::create(builder, loc, registerType,
                        cbit::Initialization::Undefined, StringAttr{});
  auto index = arith::ConstantIndexOp::create(builder, loc, 1);
  auto value =
      arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
  cbit::StoreOp::create(builder, loc, value, zero, index);
  auto loaded =
      cbit::LoadOp::create(builder, loc, builder.getI1Type(), zero, index);
  func::ReturnOp::create(builder, loc, loaded.getResult());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  cbit::addCBitToTensorTypeConversion(typeConverter);
  cbit::CBitToTensorState state;
  state.recordRegisterUses(moduleOp);
  RewritePatternSet patterns(&context);
  cbit::populateCBitToTensorConversionPatterns(typeConverter, patterns, state);
  ConversionTarget target(context);
  target.addIllegalDialect<cbit::CBitDialect>();
  target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                         tensor::TensorDialect>();

  ASSERT_TRUE(
      succeeded(applyPartialConversion(moduleOp, target, std::move(patterns))));

  size_t emptyOps = 0;
  size_t insertOps = 0;
  size_t extractOps = 0;
  moduleOp.walk([&](tensor::EmptyOp) { ++emptyOps; });
  moduleOp.walk([&](tensor::InsertOp) { ++insertOps; });
  moduleOp.walk([&](tensor::ExtractOp) { ++extractOps; });
  EXPECT_EQ(emptyOps, 2);
  EXPECT_EQ(insertOps, 3);
  EXPECT_EQ(extractOps, 1);
}
