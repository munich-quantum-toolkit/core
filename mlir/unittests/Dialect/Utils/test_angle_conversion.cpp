/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Utils/AngleConversion.h"

#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/Pass.h> // IWYU pragma: keep (factory return type)
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <utility>

[[nodiscard]] static mlir::func::FuncOp
buildRuntimeQuantizer(mlir::ModuleOp module, const uint32_t width) {
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  const auto location = builder.getUnknownLoc();
  const auto name = (llvm::Twine("quantize_") + llvm::Twine(width)).str();
  auto function = mlir::func::FuncOp::create(
      builder, location, name,
      builder.getFunctionType({builder.getF64Type()}, {builder.getI64Type()}));
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  auto* block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  auto bits = mlir::mqt::angle::buildFloatToBits(builder, location,
                                                 block->getArgument(0), width);
  if (width != mlir::mqt::angle::MACHINE_WIDTH) {
    bits = mlir::arith::ExtUIOp::create(builder, location, builder.getI64Type(),
                                        bits);
  }
  mlir::func::ReturnOp::create(builder, location, bits);
  return function;
}

namespace {

TEST(AngleConversionTest, QuantizesBinary64ExactlyBeyondItsSignificand) {
  struct Case {
    double radians;
    uint32_t width;
    uint64_t expected;
  };
  constexpr std::array<Case, 5> cases{{
      {.radians = 1.0, .width = 53, .expected = 0x517CC1B727221ULL},
      {.radians = 1.0, .width = 54, .expected = 0xA2F9836E4E441ULL},
      {.radians = 1.0, .width = 64, .expected = 0x28BE60DB939105BDULL},
      {.radians = -1.0, .width = 64, .expected = 0xD7419F246C6EFA43ULL},
      {.radians = 1.0e300, .width = 64, .expected = 0xE28662C4184B3DE4ULL},
  }};
  for (const auto& testCase : cases) {
    EXPECT_EQ(mlir::mqt::angle::quantize(testCase.radians, testCase.width),
              testCase.expected);
  }
}

TEST(AngleConversionTest, QuantizesBoundaryBinary64Values) {
  constexpr auto twoPi = 2.0 * std::numbers::pi;
  EXPECT_EQ(mlir::mqt::angle::quantize(twoPi, 64), 0U);
  EXPECT_EQ(mlir::mqt::angle::quantize(std::nextafter(twoPi, 0.0), 64),
            0xFFFFFFFFFFFFF5D0ULL);
  EXPECT_EQ(
      mlir::mqt::angle::quantize(std::numeric_limits<double>::denorm_min(), 64),
      0U);
  EXPECT_FALSE(
      mlir::mqt::angle::quantize(std::numeric_limits<double>::infinity(), 64));
  EXPECT_FALSE(
      mlir::mqt::angle::quantize(std::numeric_limits<double>::quiet_NaN(), 64));
}

TEST(AngleConversionTest, ResizesBitPatternsExactly) {
  EXPECT_EQ(mlir::mqt::angle::resize(165, 8, 53), 0x14A00000000000ULL);
  EXPECT_EQ(mlir::mqt::angle::resize(0b1010101, 7, 4), 0b1011U);
  EXPECT_EQ(mlir::mqt::angle::resize(0b1001000, 7, 4), 0b1001U);
  EXPECT_EQ(mlir::mqt::angle::resize(0b1010100, 7, 4), 0b1010U);
  EXPECT_EQ(mlir::mqt::angle::resize(0b1011100, 7, 4), 0b1100U);
}

TEST(AngleConversionTest, ConvertsBitsToCanonicalBinary64Radians) {
  EXPECT_EQ(mlir::mqt::angle::toRadians(0, 64), 0.0);
  EXPECT_EQ(mlir::mqt::angle::toRadians(128, 8), std::numbers::pi);
  EXPECT_EQ(mlir::mqt::angle::toRadians(15, 8),
            15.0 * std::ldexp(2.0 * std::numbers::pi, -8));
}

TEST(AngleConversionTest, RecognizesRuntimeFloatToBits) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  for (const auto width : std::array<uint32_t, 2>{8, 64}) {
    auto function = buildRuntimeQuantizer(module, width);
    auto bits = function.getBody().front().getTerminator()->getOperand(0);
    if (auto extension = bits.getDefiningOp<mlir::arith::ExtUIOp>()) {
      bits = extension.getIn();
    }
    ASSERT_EQ(bits.getType().getIntOrFloatBitWidth(), width);
    const auto source = mlir::mqt::angle::matchFloatToBits(bits);
    ASSERT_TRUE(source);
    EXPECT_EQ(*source, function.getArgument(0));
  }
}

TEST(AngleConversionTest, ExecutesRuntimeFloatToBitsExactly) {
  ASSERT_FALSE(llvm::InitializeNativeTarget());
  ASSERT_FALSE(llvm::InitializeNativeTargetAsmPrinter());

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::LLVM::LLVMDialect>();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  for (const auto width : std::array<uint32_t, 4>{1, 8, 53, 64}) {
    buildRuntimeQuantizer(module, width);
  }
  module.walk([](mlir::Operation* operation) {
    for (const auto result : operation->getResults()) {
      const auto type = mlir::dyn_cast<mlir::IntegerType>(result.getType());
      if (type) {
        EXPECT_LE(type.getWidth(), 64)
            << "runtime angle conversion emitted an i" << type.getWidth()
            << " value";
      }
    }
  });

  mlir::PassManager passManager(&context);
  passManager.addPass(mlir::createArithToLLVMConversionPass());
  passManager.addPass(mlir::createConvertFuncToLLVMPass());
  passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
  ASSERT_TRUE(succeeded(passManager.run(module)));

  auto engineOrError = mlir::ExecutionEngine::create(module);
  if (!engineOrError) {
    FAIL() << llvm::toString(engineOrError.takeError());
  }
  auto engine = std::move(*engineOrError);
  const auto execute = [&](const double radians, const uint32_t width) {
    uint64_t actual = 0;
    const auto function = (llvm::Twine("quantize_") + llvm::Twine(width)).str();
    if (auto error = engine->invoke(function, radians,
                                    mlir::ExecutionEngine::result(actual))) {
      ADD_FAILURE() << llvm::toString(std::move(error));
    }
    return actual;
  };

  struct Case {
    double radians;
    uint32_t width;
    uint64_t expected;
  };
  const auto twoPi = 2.0 * std::numbers::pi;
  const std::array cases{
      Case{.radians = 0.0, .width = 64, .expected = 0},
      Case{.radians = -0.0, .width = 64, .expected = 0},
      Case{.radians = std::numbers::pi, .width = 1, .expected = 1},
      Case{.radians = 1.0, .width = 8, .expected = 41},
      Case{.radians = twoPi * (127.0 / 512.0), .width = 8, .expected = 64},
      Case{.radians = 1.0, .width = 53, .expected = 0x517CC1B727221ULL},
      Case{.radians = 1.0, .width = 64, .expected = 0x28BE60DB939105BDULL},
      Case{.radians = -1.0, .width = 64, .expected = 0xD7419F246C6EFA43ULL},
      Case{.radians = std::nextafter(twoPi, 0.0),
           .width = 64,
           .expected = 0xFFFFFFFFFFFFF5D0ULL},
      Case{.radians = 1.0e300, .width = 64, .expected = 0xE28662C4184B3DE4ULL},
      Case{.radians = std::numeric_limits<double>::denorm_min(),
           .width = 64,
           .expected = 0},
  };
  for (const auto& testCase : cases) {
    const auto actual = execute(testCase.radians, testCase.width);
    EXPECT_EQ(actual, testCase.expected)
        << "width " << testCase.width << ", radians " << testCase.radians;
  }

  std::mt19937_64 generator(0xA1161EULL);
  for (size_t sample = 0; sample < 256; ++sample) {
    const auto radians = std::bit_cast<double>(generator());
    if (!std::isfinite(radians)) {
      continue;
    }
    for (const auto width : std::array<uint32_t, 4>{1, 8, 53, 64}) {
      const auto expected = mlir::mqt::angle::quantize(radians, width);
      ASSERT_TRUE(expected);
      EXPECT_EQ(execute(radians, width), *expected)
          << "width " << width << ", radians " << radians;
    }
  }
}

} // namespace
