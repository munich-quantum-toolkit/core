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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/AngleConversion.h"

#include <gtest/gtest.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <tuple>

[[nodiscard]] static mlir::LogicalResult quantize(mlir::ModuleOp moduleOp,
                                                  const uint32_t width) {
  mlir::qco::QuantizeGateAnglesOptions options;
  options.precisionBits = width;
  mlir::PassManager manager(moduleOp.getContext());
  manager.addPass(mlir::qco::createQuantizeGateAngles(options));
  return manager.run(moduleOp);
}

[[nodiscard]] static mlir::Value addDynamicF64Input(mlir::ModuleOp moduleOp) {
  auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>("main");
  assert(funcOp && "QCOProgramBuilder must create @main");
  funcOp.insertArgument(0, mlir::Float64Type::get(moduleOp.getContext()), {},
                        funcOp.getLoc());
  return funcOp.getArgument(0);
}

namespace {

using namespace mlir;
using namespace mlir::qco;

TEST(QCOQuantizeGateAnglesTest, QuantizesConstantsAndIsIdempotent) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rx(-std::numbers::pi / 2.0, q);
    return b.intConstant(0);
  });

  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  RXOp rotation;
  module->walk([&](RXOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  const auto precision = module->getOperation()->getAttrOfType<IntegerAttr>(
      mqt::angle::FINAL_QUANTIZATION_ATTR);
  ASSERT_TRUE(precision);
  EXPECT_EQ(precision.getInt(), 8);
  const auto converted = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted->bitWidth, 8);
  llvm::APInt bits;
  ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
  EXPECT_EQ(bits.getZExtValue(), 192);

  const auto parameter = rotation.getTheta();
  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  EXPECT_EQ(rotation.getTheta(), parameter);
}

TEST(QCOQuantizeGateAnglesTest, FoldsConstantPrecisionComposition) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    auto sourceBits = arith::ConstantOp::create(
        b, b.getUnknownLoc(),
        IntegerAttr::get(b.getI64Type(), llvm::APInt(64, uint64_t{1} << 63U)));
    auto radians = mqt::angle::buildBitsToRadians(b, b.getUnknownLoc(),
                                                  sourceBits.getResult());
    q = b.rz(radians, q);
    return b.intConstant(0);
  });

  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  RZOp rotation;
  module->walk([&](RZOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  const auto converted = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(converted);
  llvm::APInt bits;
  ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
  EXPECT_EQ(bits.getZExtValue(), 128);
}

TEST(QCOQuantizeGateAnglesTest, WidensConstantPrecisionCompositionExactly) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    auto sourceBits = arith::ConstantOp::create(
        b, b.getUnknownLoc(),
        IntegerAttr::get(b.getI8Type(), llvm::APInt(8, 165)));
    auto radians = mqt::angle::buildBitsToRadians(b, b.getUnknownLoc(),
                                                  sourceBits.getResult());
    q = b.rz(radians, q);
    return b.intConstant(0);
  });

  ASSERT_TRUE(succeeded(quantize(*module, 53)));
  RZOp rotation;
  module->walk([&](RZOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  const auto converted = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(converted);
  llvm::APInt bits;
  ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
  EXPECT_EQ(bits.getZExtValue(), 0x14A00000000000ULL);
}

TEST(QCOQuantizeGateAnglesTest, QuantizesLargeFiniteConstantsExactly) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rx(1.0e20, q);
    return b.intConstant(0);
  });

  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  RXOp rotation;
  module->walk([&](RXOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  const auto converted = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(converted);
  llvm::APInt bits;
  ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
  EXPECT_EQ(bits.getZExtValue(), 77);
}

TEST(QCOQuantizeGateAnglesTest, QuantizesDynamicAndNestedGateParametersOnly) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.pow(0.75, q, [&](Value nested) { return b.rz(0.25, nested); });
    return b.intConstant(0);
  });

  PowOp power;
  RZOp rotation;
  module->walk([&](PowOp op) { power = op; });
  module->walk([&](RZOp op) { rotation = op; });
  ASSERT_TRUE(power);
  ASSERT_TRUE(rotation);
  const auto dynamicInput = addDynamicF64Input(*module);
  rotation.getThetaMutable().assign(dynamicInput);
  const auto exponent = power.getExponent();
  ASSERT_TRUE(succeeded(quantize(*module, 53)));
  EXPECT_EQ(power.getExponent(), exponent);
  const auto converted = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted->bitWidth, 53);
  EXPECT_TRUE(converted->bits.getDefiningOp<arith::TruncIOp>());
  EXPECT_EQ(mqt::angle::matchFloatToBits(converted->bits), dynamicInput);
}

TEST(QCOQuantizeGateAnglesTest, ReusesDynamicConversionWithinOneBlock) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rx(0.25, q);
    q = b.rz(0.5, q);
    return b.intConstant(0);
  });

  SmallVector<UnitaryOpInterface> rotations;
  module->walk([&](UnitaryOpInterface op) {
    if (isa<RXOp, RZOp>(op.getOperation())) {
      rotations.push_back(op);
    }
  });
  ASSERT_EQ(rotations.size(), 2);
  const auto dynamicInput = addDynamicF64Input(*module);
  for (auto rotation : rotations) {
    rotation->setOperand(rotation.getParameters().getBeginOperandIndex(),
                         dynamicInput);
  }

  ASSERT_TRUE(succeeded(quantize(*module, 64)));
  EXPECT_EQ(rotations[0].getParameters().front(),
            rotations[1].getParameters().front());
}

TEST(QCOQuantizeGateAnglesTest, QuantizesGlobalPhaseAtBoundaryPrecisions) {
  struct Case {
    uint32_t width;
    double radians;
  };
  constexpr auto cases = std::to_array<Case>({
      {.width = 1, .radians = std::numbers::pi},
      {.width = 8, .radians = -9.0 * std::numbers::pi / 2.0},
      {.width = 64, .radians = std::numbers::pi / 2.0},
  });
  for (const auto& testCase : cases) {
    MLIRContext context;
    context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
    const auto multiTurnRadians = testCase.radians + (10.0 * std::numbers::pi);
    auto module = QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& b) {
      b.gphase(multiTurnRadians);
      return b.intConstant(0);
    });

    ASSERT_TRUE(succeeded(quantize(*module, testCase.width)));
    GPhaseOp phase;
    module->walk([&](GPhaseOp op) { phase = op; });
    ASSERT_TRUE(phase);
    const auto converted = mqt::angle::matchQuantizedRadians(phase.getTheta());
    ASSERT_TRUE(converted);
    EXPECT_EQ(converted->bitWidth, testCase.width);
    llvm::APInt bits;
    ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
    const auto expected =
        mqt::angle::quantize(multiTurnRadians, testCase.width);
    ASSERT_TRUE(expected);
    EXPECT_EQ(bits.getZExtValue(), *expected);
  }
}

TEST(QCOQuantizeGateAnglesTest, VisitsControlAndInverseRegions) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto control = b.allocQubit();
    auto target = b.allocQubit();
    target = b.inv(target, [&](Value nested) {
      b.gphase(0.125);
      return b.rx(0.25, nested);
    });
    std::tie(control, target) = b.ctrl(
        control, target, [&](Value nested) { return b.ry(0.5, nested); });
    std::tie(control, target) = b.rzz(0.75, control, target);
    return b.intConstant(0);
  });

  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  size_t quantizedParameters = 0;
  module->walk([&](UnitaryOpInterface unitary) {
    if (isa<PowOp>(unitary.getOperation())) {
      return;
    }
    for (const auto parameter : unitary.getParameters()) {
      const auto converted = mqt::angle::matchQuantizedRadians(parameter);
      ASSERT_TRUE(converted);
      EXPECT_EQ(converted->bitWidth, 8);
      ++quantizedParameters;
    }
  });
  EXPECT_EQ(quantizedParameters, 4);
}

TEST(QCOQuantizeGateAnglesTest, DifferentPrecisionsComposeConversions) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rz(0.25, q);
    return b.intConstant(0);
  });

  RZOp rotation;
  module->walk([&](RZOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  rotation.getThetaMutable().assign(addDynamicF64Input(*module));

  ASSERT_TRUE(succeeded(quantize(*module, 8)));
  ASSERT_TRUE(succeeded(quantize(*module, 53)));
  const auto outer = mqt::angle::matchQuantizedRadians(rotation.getTheta());
  ASSERT_TRUE(outer);
  EXPECT_EQ(outer->bitWidth, 53);
  const auto resize = mqt::angle::matchResize(outer->bits);
  ASSERT_TRUE(resize);
  EXPECT_EQ(resize->sourceWidth, 8);
  EXPECT_EQ(resize->targetWidth, 53);
  const auto original = mqt::angle::matchFloatToBits(resize->source);
  ASSERT_TRUE(original);
}

TEST(QCOQuantizeGateAnglesTest, RejectsOversizedPrecisionMetadata) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rz(0.1, q);
    return b.intConstant(0);
  });
  module->getOperation()->setAttr(
      mqt::angle::FINAL_QUANTIZATION_ATTR,
      IntegerAttr::get(IntegerType::get(&context, 128),
                       llvm::APInt::getOneBitSet(128, 100)));

  EXPECT_TRUE(failed(quantize(*module, 53)));
}

TEST(QCOQuantizeGateAnglesTest, RejectsInvalidPrecisions) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  for (const auto width : {0U, 65U}) {
    auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
      auto q = b.allocQubit();
      q = b.rz(0.5, q);
      return b.intConstant(0);
    });
    EXPECT_TRUE(failed(quantize(*module, width)));
  }
}

TEST(QCOQuantizeGateAnglesTest, RejectsNonFiniteConstantsBeforeMutation) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rx(0.5, q);
    q = b.rz(std::numeric_limits<double>::infinity(), q);
    return b.intConstant(0);
  });
  RXOp rotation;
  module->walk([&](RXOp op) { rotation = op; });
  ASSERT_TRUE(rotation);
  const auto originalParameter = rotation.getTheta();

  EXPECT_TRUE(failed(quantize(*module, 8)));
  EXPECT_EQ(rotation.getTheta(), originalParameter);
}

TEST(QCOQuantizeGateAnglesTest,
     RejectsFoldableNonFiniteExpressionsBeforeMutation) {
  MLIRContext context;
  context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  auto module = QCOProgramBuilder::build(&context, [](QCOProgramBuilder& b) {
    auto q = b.allocQubit();
    q = b.rx(0.5, q);
    q = b.rz(0.25, q);
    return b.intConstant(0);
  });
  RXOp first;
  RZOp invalid;
  module->walk([&](RXOp op) { first = op; });
  module->walk([&](RZOp op) { invalid = op; });
  ASSERT_TRUE(first);
  ASSERT_TRUE(invalid);
  const auto originalParameter = first.getTheta();

  OpBuilder builder(invalid);
  auto infinity = arith::ConstantFloatOp::create(
      builder, invalid.getLoc(), builder.getF64Type(),
      llvm::APFloat(std::numeric_limits<double>::infinity()));
  auto negativeInfinity =
      arith::NegFOp::create(builder, invalid.getLoc(), infinity);
  invalid.getThetaMutable().assign(negativeInfinity);

  EXPECT_TRUE(failed(quantize(*module, 8)));
  EXPECT_EQ(first.getTheta(), originalParameter);
  EXPECT_EQ(invalid.getTheta(), negativeInfinity.getResult());
}

} // namespace
