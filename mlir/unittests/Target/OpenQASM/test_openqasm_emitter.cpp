/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "OpenQASMTestUtils.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/OpenQASMAttributes.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/Utils/AngleConversion.h"
#include "mlir/Support/Passes.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "qasm_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

using namespace mlir;
using namespace mlir::oq3::test;

namespace {

TEST(OpenQASMTargetTest, LowersAnglesToBuiltinIntegersAndRoundTripsMetadata) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
input angle[8] theta;
output angle[8] result;
result = theta + angle[8](pi / 2.0);
qubit q;
rx(result) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_EQ(function.getNumArguments(), 1);
  ASSERT_EQ(function.getNumResults(), 1);
  EXPECT_TRUE(function.getArgumentTypes().front().isInteger(8));
  EXPECT_TRUE(function.getResultTypes().front().isInteger(8));
  EXPECT_TRUE(function.getArgAttr(0, qc::openqasm::SCALAR_ATTR));
  EXPECT_TRUE(function.getResultAttr(0, qc::openqasm::SCALAR_ATTR));

  size_t angleAdds = 0;
  size_t angleBridges = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto add = dyn_cast<arith::AddIOp>(operation)) {
      angleAdds += add.getType().isInteger(8);
    }
    if (auto conversion = dyn_cast<arith::UIToFPOp>(operation)) {
      angleBridges += conversion.getIn().getType().isInteger(64);
    }
  });
  EXPECT_EQ(angleAdds, 1);
  EXPECT_EQ(angleBridges, 1);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("input angle[8] theta;"), std::string::npos);
  EXPECT_NE(emitted->find("output angle[8] result;"), std::string::npos);
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << "\n"
                        << *emitted;
}

TEST(OpenQASMTargetTest, RoundTripsEveryScalarInterfaceKind) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input bool input_bool;
input int input_int;
input uint[8] input_uint;
input float input_float;
input angle[8] input_angle;
output bool output_bool;
output int output_int;
output uint[8] output_uint;
output float output_float;
output angle[8] output_angle;
output_bool = bool(input_angle);
output_int = int(input_uint);
output_uint = uint[8](input_int);
output_float = float(input_int);
output_angle = angle[8](input_float);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_EQ(function.getNumArguments(), 5);
  ASSERT_EQ(function.getNumResults(), 5);
  for (const auto index : llvm::seq<size_t>(0, 5)) {
    EXPECT_TRUE(function.getArgAttr(index, qc::openqasm::SCALAR_ATTR));
    EXPECT_TRUE(function.getResultAttr(index, qc::openqasm::SCALAR_ATTR));
  }

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  for (const llvm::StringRef declaration :
       {"input bool input_bool;", "input int input_int;",
        "input uint[8] input_uint;", "input float input_float;",
        "input angle[8] input_angle;", "output bool output_bool;",
        "output int output_int;", "output uint[8] output_uint;",
        "output float output_float;", "output angle[8] output_angle;"}) {
    EXPECT_NE(emitted->find(declaration), std::string::npos) << *emitted;
  }

  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
  auto reimported = qc::translateQASM3ToQC(*emitted, &context);
  ASSERT_TRUE(reimported);
  EXPECT_TRUE(succeeded(verify(*reimported)));
}

TEST(OpenQASMTargetTest, EmitsCanonicalRuntimeFloatToAngleConversion) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input float theta;
output angle[8] result;
result = theta;
qubit q;
gphase(result);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t truncations = 0;
  size_t exactConversions = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto truncation = dyn_cast<arith::TruncIOp>(operation)) {
      truncations += truncation.getType().isInteger(8);
      exactConversions += mqt::angle::matchFloatToBits(truncation).has_value();
    }
  });
  EXPECT_EQ(truncations, 1);
  EXPECT_EQ(exactConversions, 1);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("angle[8](theta)"), std::string::npos);
  EXPECT_EQ(emitted->find("_mqt_roundeven"), std::string::npos) << *emitted;
}

TEST(OpenQASMTargetTest, RecognizesCommutedCanonicalAngleConversions) {
  MLIRContext context;
  context
      .loadDialect<arith::ArithDialect, func::FuncDialect, math::MathDialect>();
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  const auto loc = builder.getUnknownLoc();
  auto function =
      func::FuncOp::create(builder, loc, "conversion_matchers",
                           builder.getFunctionType({builder.getF64Type()}, {}));
  module.push_back(function);
  auto* block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);

  auto bits = arith::ConstantIntOp::create(builder, loc, 64, 8).getResult();
  EXPECT_EQ(mqt::angle::buildResize(builder, loc, bits, 8), bits);

  auto radians = mqt::angle::buildBitsToRadians(builder, loc, bits);
  auto scaledRadians = radians.getDefiningOp<arith::MulFOp>();
  ASSERT_TRUE(scaledRadians);
  const auto converted = scaledRadians.getLhs();
  const auto scale = scaledRadians.getRhs();
  scaledRadians->setOperand(0, scale);
  scaledRadians->setOperand(1, converted);
  const auto quantized = mqt::angle::matchQuantizedRadians(radians);
  ASSERT_TRUE(quantized);
  EXPECT_EQ(quantized->bits, bits);

  auto one =
      arith::ConstantOp::create(builder, loc, builder.getF64FloatAttr(1.0))
          .getResult();
  auto nonCanonical = arith::MulFOp::create(builder, loc, one, one).getResult();
  EXPECT_FALSE(mqt::angle::matchQuantizedRadians(nonCanonical));
  auto invalidScale =
      arith::MulFOp::create(builder, loc, converted, one).getResult();
  EXPECT_FALSE(mqt::angle::matchQuantizedRadians(invalidScale));

  const auto input = block->getArgument(0);
  auto encoded = mqt::angle::buildFloatToBits(builder, loc, input, 8);
  auto truncated = encoded.getDefiningOp<arith::TruncIOp>();
  ASSERT_TRUE(truncated);
  const auto decoded = mqt::angle::matchFloatToBits(encoded);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(*decoded, input);
}

TEST(OpenQASMTargetTest, LowersAndRoundTripsScalarAngleBitBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
input angle[8] theta;
input uint[8] bits;
input int distance;
output angle[8] rotated_angle;
output uint[8] rotated_uint;
output uint count;
rotated_angle = rotl(theta, distance);
rotated_uint = rotr(bits, distance);
count = popcount(rotated_angle);
qubit q;
rz(rotated_angle) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  const auto countOperations = [](ModuleOp module) {
    std::array<size_t, 3> counts{};
    module.walk([&](Operation* operation) {
      counts[0] += isa<LLVM::FshlOp>(operation);
      counts[1] += isa<LLVM::FshrOp>(operation);
      counts[2] += isa<math::CtPopOp>(operation);
    });
    return counts;
  };
  EXPECT_EQ(countOperations(*moduleOp), (std::array<size_t, 3>{1, 1, 1}));

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("rotated_angle = rotl(theta, int("),
            std::string::npos)
      << *emitted;
  EXPECT_NE(emitted->find("rotated_uint = rotr(bits, int("), std::string::npos)
      << *emitted;
  EXPECT_NE(emitted->find("count = uint[64](popcount(rotl(theta, int("),
            std::string::npos)
      << *emitted;

  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
  auto reimported = qc::translateQASM3ToQC(*emitted, &context);
  ASSERT_TRUE(reimported);
  ASSERT_TRUE(succeeded(verify(*reimported)));
  const auto reimportedCounts = countOperations(*reimported);
  EXPECT_GE(reimportedCounts[0], 1);
  EXPECT_EQ(reimportedCounts[1], 1);
  EXPECT_EQ(reimportedCounts[2], 1);
}

TEST(OpenQASMTargetTest, EmitsCanonicalRuntimeFloatToAngle64Conversion) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input float theta;
output angle[64] result;
result = theta;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("angle[64](theta)"), std::string::npos) << *emitted;
  EXPECT_EQ(emitted->find("roundeven"), std::string::npos) << *emitted;
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(*emitted));
}

TEST(OpenQASMTargetTest, EmitsUnsignedHighBitPatternsExactly) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
output uint[8] byte;
output angle[64] angle_bits;
byte = uint[8](255);
angle_bits = angle[64](bit[64](uint[64](maximum)));
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("byte = 255;"), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("18446744073709551615"), std::string::npos)
      << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, PreservesAngleBitPatternCastsInUintContexts) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[8] theta;
output uint[8] bits;
output bool high_half;
bits = uint[8](bit[8](theta));
high_half = bits >= uint[8](128);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("uint[8](bit[8](theta))"), std::string::npos)
      << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, KeepsCustomGateParametersAsAngleBits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate half(theta) q {
  rz(theta / 2) q;
}
input angle[64] theta;
qubit q;
half(theta) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t radiansBridges = 0;
  size_t floatToAngleConversions = 0;
  moduleOp->walk([&](Operation* operation) {
    radiansBridges += isa<arith::UIToFPOp>(operation);
    for (const auto result : operation->getResults()) {
      floatToAngleConversions +=
          mqt::angle::matchFloatToBits(result).has_value();
    }
  });
  EXPECT_EQ(radiansBridges, 1);
  EXPECT_EQ(floatToAngleConversions, 0);
}

TEST(OpenQASMTargetTest, LowersDynamicAngleAndUintBooleanCasts) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[8] theta;
input uint[8] count;
input float value;
output bool has_theta;
output bool has_count;
output bool has_value;
has_theta = bool(theta);
has_count = bool(count);
has_value = bool(value);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("theta != angle[8](bit[8](uint[8](0)))"),
            std::string::npos)
      << *emitted;
  EXPECT_NE(emitted->find("count != 0"), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("value != 0.0"), std::string::npos) << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, CollapsesSafeDynamicAngleShifts) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
input angle[8] theta;
input uint distance;
output angle[8] left;
output angle[8] right;
left = theta << distance;
right = theta >> distance;
qubit q;
rx(left) q;
rz(right) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("theta << distance"), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("theta >> distance"), std::string::npos) << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, CollapsesCanonicalAngleWideningAndNarrowing) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[4] narrow;
input angle[8] wide;
output angle[8] widened;
output angle[4] narrowed;
widened = angle[8](narrow);
narrowed = angle[4](wide);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("angle[8](narrow)"), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("angle[4](wide)"), std::string::npos) << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, CarriesBuiltinIntegerAnglesThroughStructuredControl) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
input angle[8] theta;
input uint[8] factor;
input bool repeat;
input int selector;
output angle[8] result;
result = theta;
if (factor != uint[8](0)) { result = result * factor; }
for int i in [0:1] {
  result = result + angle[8](pi / 4.0);
}
while (repeat) {
  result = -result;
  repeat = false;
}
switch (selector) {
  case 1 { result = result + angle[8](pi / 2.0); }
  default { result = result + angle[8](pi / 4.0); }
}
qubit q;
rx(result) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t angleMultiplications = 0;
  size_t angleAdditions = 0;
  size_t angleNegations = 0;
  size_t angleIfResults = 0;
  size_t angleForResults = 0;
  size_t angleWhileResults = 0;
  size_t angleSwitchResults = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto multiplication = dyn_cast<arith::MulIOp>(operation)) {
      angleMultiplications += multiplication.getType().isInteger(8);
    }
    if (auto addition = dyn_cast<arith::AddIOp>(operation)) {
      angleAdditions += addition.getType().isInteger(8);
    }
    if (auto subtraction = dyn_cast<arith::SubIOp>(operation)) {
      angleNegations += subtraction.getType().isInteger(8);
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      angleIfResults +=
          llvm::count_if(conditional.getResultTypes(),
                         [](Type type) { return type.isInteger(8); });
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      angleForResults += llvm::count_if(
          loop.getResultTypes(), [](Type type) { return type.isInteger(8); });
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      angleWhileResults += llvm::count_if(
          loop.getResultTypes(), [](Type type) { return type.isInteger(8); });
    }
    if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(operation)) {
      angleSwitchResults +=
          llvm::count_if(switchOp.getResultTypes(),
                         [](Type type) { return type.isInteger(8); });
    }
  });
  EXPECT_EQ(angleMultiplications, 1);
  EXPECT_GE(angleAdditions, 1);
  EXPECT_EQ(angleNegations, 1);
  EXPECT_GE(angleIfResults, 1);
  EXPECT_GE(angleForResults, 1);
  EXPECT_GE(angleWhileResults, 1);
  EXPECT_GE(angleSwitchResults, 1);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  moduleOp->print(stream);
  EXPECT_EQ(ir.find("!quant."), std::string::npos);
  EXPECT_EQ(ir.find("mqt.angle"), std::string::npos);

  ASSERT_TRUE(succeeded(runQCCleanupPipeline(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("angle[8] _mqt_s"), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("while ("), std::string::npos) << *emitted;
  EXPECT_NE(emitted->find("switch (selector)"), std::string::npos) << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, UsesUnsignedIntegerAngleOperations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[8] lhs;
input angle[8] rhs;
input uint[8] divisor;
output angle[8] quotient;
output uint[8] ratio;
output angle[8] bits;
output angle[8] conjunction;
output angle[8] disjunction;
output angle[8] negated;
output bool less;
quotient = lhs / divisor;
ratio = lhs / rhs;
bits = ~(lhs ^ rhs);
conjunction = lhs & rhs;
disjunction = lhs | rhs;
negated = -lhs;
less = lhs < rhs;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t unsignedDivisions = 0;
  size_t bitwiseAnd = 0;
  size_t bitwiseOrs = 0;
  size_t bitwiseXors = 0;
  size_t unsignedComparisons = 0;
  moduleOp->walk([&](Operation* operation) {
    unsignedDivisions += isa<arith::DivUIOp>(operation);
    if (auto bitwise = dyn_cast<arith::AndIOp>(operation)) {
      bitwiseAnd += bitwise.getType().isInteger(8);
    }
    if (auto bitwise = dyn_cast<arith::OrIOp>(operation)) {
      bitwiseOrs += bitwise.getType().isInteger(8);
    }
    if (auto bitwise = dyn_cast<arith::XOrIOp>(operation)) {
      bitwiseXors += bitwise.getType().isInteger(8);
    }
    if (auto comparison = dyn_cast<arith::CmpIOp>(operation)) {
      unsignedComparisons +=
          comparison.getPredicate() == arith::CmpIPredicate::ult;
    }
  });
  EXPECT_EQ(unsignedDivisions, 2);
  EXPECT_EQ(bitwiseAnd, 1);
  EXPECT_EQ(bitwiseOrs, 1);
  EXPECT_EQ(bitwiseXors, 2);
  EXPECT_EQ(unsignedComparisons, 1);

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, PreservesAngleOperandsOfComparisonsAndAngleDivision) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[8] theta;
output bool less_than_pi;
output uint[8] ratio_to_pi;
output bool less_than_computed;
less_than_pi = theta < angle[8](pi);
ratio_to_pi = theta / angle[8](pi);
less_than_computed = theta < (theta + angle[8](pi));
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;

  bool sawConstantComparison = false;
  bool sawAngleDivision = false;
  bool sawComputedComparison = false;
  for (const auto statementId : reparsed.program->body) {
    const auto* assignment =
        std::get_if<oq3::frontend::ScalarAssignmentStatement>(
            &reparsed.program->statements[statementId].data);
    if (assignment == nullptr) {
      continue;
    }
    const auto& scalar = reparsed.program->scalars[assignment->scalar];
    if (scalar.name == "ratio_to_pi") {
      ASSERT_TRUE(assignment->value);
      const auto& division = reparsed.program->expressions[*assignment->value];
      ASSERT_EQ(division.kind, oq3::frontend::ExpressionKind::Divide);
      const auto& denominator = reparsed.program->expressions[division.rhs];
      EXPECT_EQ(denominator.type, oq3::frontend::ScalarType::Angle);
      ASSERT_EQ(denominator.kind, oq3::frontend::ExpressionKind::Constant);
      EXPECT_EQ(std::get<uint64_t>(denominator.constant), 128);
      sawAngleDivision = true;
      continue;
    }
    ASSERT_TRUE(assignment->condition);
    const auto& comparison =
        reparsed.program->conditions[*assignment->condition];
    ASSERT_EQ(comparison.kind, oq3::frontend::ConditionKind::Comparison);
    const auto& rhs = reparsed.program->expressions[comparison.comparisonRhs];
    EXPECT_EQ(rhs.type, oq3::frontend::ScalarType::Angle);
    if (scalar.name == "less_than_pi") {
      ASSERT_EQ(rhs.kind, oq3::frontend::ExpressionKind::Constant);
      EXPECT_EQ(std::get<uint64_t>(rhs.constant), 128);
      sawConstantComparison = true;
    } else if (scalar.name == "less_than_computed") {
      EXPECT_EQ(rhs.kind, oq3::frontend::ExpressionKind::Add);
      EXPECT_EQ(rhs.type, oq3::frontend::ScalarType::Angle);
      sawComputedComparison = true;
    }
  }
  EXPECT_TRUE(sawConstantComparison);
  EXPECT_TRUE(sawAngleDivision);
  EXPECT_TRUE(sawComputedComparison);
}

TEST(OpenQASMTargetTest, EmitsVerifiedQCDirectly) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t gates = 0;
  moduleOp->walk([&](qc::HOp) { ++gates; });
  EXPECT_EQ(gates, 2);
}

TEST(OpenQASMTargetTest, ProductionTranslationUsesTheStagedPipeline) {
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  bool hasQuantumOperation = false;
  moduleOp->walk([&](Operation* operation) {
    hasQuantumOperation |= isa<qc::HOp>(operation);
  });
  EXPECT_TRUE(hasQuantumOperation);
}

TEST(OpenQASMTargetTest, EmitsTypedGateAngleExpressions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shifted(theta) q {
  rx(theta + angle(1.0)) q;
}
qubit q;
shifted(0.5) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, NormalizesDynamicSignedScalarBitIndices) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input angle[8] value;
input int index;
output bit selected;
selected = value[index];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawNormalization = false;
  bool sawBoundsCheck = false;
  moduleOp->walk([&](arith::SelectOp) { sawNormalization = true; });
  moduleOp->walk([&](cf::AssertOp assertion) {
    sawBoundsCheck |= assertion.getMsg() == "scalar bit index out of bounds";
  });
  EXPECT_TRUE(sawNormalization);
  EXPECT_TRUE(sawBoundsCheck);
}

TEST(OpenQASMTargetTest, EmitsScalarMathFunctions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
float value = 0.5;
qubit q;
rx(sin(value) + cos(value) + tan(value) + exp(value) + log(value) +
   sqrt(value)) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t functions = 0;
  moduleOp->walk([&](Operation* operation) {
    functions += isa<math::SinOp, math::CosOp, math::TanOp, math::ExpOp,
                     math::LogOp, math::SqrtOp>(operation);
  });
  EXPECT_EQ(functions, 6);
}

TEST(OpenQASMTargetTest, EmitsInverseTrigFunctionsWithNumericConversions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
uint unsigned_value = 1;
int signed_value = 1;
float float_value = 0.5;
qubit q;
rx(arccos(unsigned_value) + arcsin(signed_value) + arctan(float_value)) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t arcCosines = 0;
  size_t arcSines = 0;
  size_t arcTangents = 0;
  size_t unsignedConversions = 0;
  size_t signedConversions = 0;
  moduleOp->walk([&](Operation* operation) {
    arcCosines += isa<math::AcosOp>(operation);
    arcSines += isa<math::AsinOp>(operation);
    arcTangents += isa<math::AtanOp>(operation);
    if (auto conversion = dyn_cast<arith::UIToFPOp>(operation)) {
      unsignedConversions +=
          llvm::any_of(conversion->getUsers(),
                       [](Operation* user) { return isa<math::AcosOp>(user); });
    }
    if (auto conversion = dyn_cast<arith::SIToFPOp>(operation)) {
      signedConversions +=
          llvm::any_of(conversion->getUsers(),
                       [](Operation* user) { return isa<math::AsinOp>(user); });
    }
  });
  EXPECT_EQ(arcCosines, 1);
  EXPECT_EQ(arcSines, 1);
  EXPECT_EQ(arcTangents, 1);
  EXPECT_EQ(unsignedConversions, 1);
  EXPECT_EQ(signedConversions, 1);
}

TEST(OpenQASMTargetTest, EmitsRuntimeScalarConversionMatrix) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
float source_float = 2.5;
uint uint_from_float = source_float;
int int_from_float = source_float;
bool source_bool = true;
float float_from_bool = source_bool;
int int_from_bool = source_bool;
uint source_uint = 2;
float float_from_uint = source_uint;
int source_int = 2;
float float_from_int = source_int;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t floatToUnsigned = 0;
  size_t floatToSigned = 0;
  size_t boolToInteger = 0;
  size_t unsignedToFloat = 0;
  size_t signedToFloat = 0;
  moduleOp->walk([&](Operation* operation) {
    floatToUnsigned += isa<arith::FPToUIOp>(operation);
    floatToSigned += isa<arith::FPToSIOp>(operation);
    boolToInteger += isa<arith::ExtUIOp>(operation);
    unsignedToFloat += isa<arith::UIToFPOp>(operation);
    signedToFloat += isa<arith::SIToFPOp>(operation);
  });
  EXPECT_EQ(floatToUnsigned, 1);
  EXPECT_EQ(floatToSigned, 1);
  EXPECT_EQ(boolToInteger, 1);
  EXPECT_EQ(unsignedToFloat, 2);
  EXPECT_EQ(signedToFloat, 1);
}

TEST(OpenQASMTargetTest, EmitsEveryRuntimeComparisonPredicate) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
float float_left = 1.0;
float float_right = 2.0;
bool float_equal = float_left == float_right;
bool float_not_equal = float_left != float_right;
bool float_less = float_left < float_right;
bool float_less_equal = float_left <= float_right;
bool float_greater = float_left > float_right;
bool float_greater_equal = float_left >= float_right;
int signed_left = 1;
int signed_right = 2;
bool signed_equal = signed_left == signed_right;
bool signed_not_equal = signed_left != signed_right;
bool signed_less = signed_left < signed_right;
bool signed_less_equal = signed_left <= signed_right;
bool signed_greater = signed_left > signed_right;
bool signed_greater_equal = signed_left >= signed_right;
uint unsigned_left = 1;
uint unsigned_right = 2;
bool unsigned_equal = unsigned_left == unsigned_right;
bool unsigned_not_equal = unsigned_left != unsigned_right;
bool unsigned_less = unsigned_left < unsigned_right;
bool unsigned_less_equal = unsigned_left <= unsigned_right;
bool unsigned_greater = unsigned_left > unsigned_right;
bool unsigned_greater_equal = unsigned_left >= unsigned_right;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  std::array<size_t, 6> floatPredicates{};
  std::array<size_t, 10> integerPredicates{};
  moduleOp->walk([&](Operation* operation) {
    if (auto comparison = dyn_cast<arith::CmpFOp>(operation)) {
      switch (comparison.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
        ++floatPredicates[0];
        break;
      case arith::CmpFPredicate::UNE:
        ++floatPredicates[1];
        break;
      case arith::CmpFPredicate::OLT:
        ++floatPredicates[2];
        break;
      case arith::CmpFPredicate::OLE:
        ++floatPredicates[3];
        break;
      case arith::CmpFPredicate::OGT:
        ++floatPredicates[4];
        break;
      case arith::CmpFPredicate::OGE:
        ++floatPredicates[5];
        break;
      default:
        break;
      }
    }
    if (auto comparison = dyn_cast<arith::CmpIOp>(operation)) {
      switch (comparison.getPredicate()) {
      case arith::CmpIPredicate::eq:
        ++integerPredicates[0];
        break;
      case arith::CmpIPredicate::ne:
        ++integerPredicates[1];
        break;
      case arith::CmpIPredicate::slt:
        ++integerPredicates[2];
        break;
      case arith::CmpIPredicate::sle:
        ++integerPredicates[3];
        break;
      case arith::CmpIPredicate::sgt:
        ++integerPredicates[4];
        break;
      case arith::CmpIPredicate::sge:
        ++integerPredicates[5];
        break;
      case arith::CmpIPredicate::ult:
        ++integerPredicates[6];
        break;
      case arith::CmpIPredicate::ule:
        ++integerPredicates[7];
        break;
      case arith::CmpIPredicate::ugt:
        ++integerPredicates[8];
        break;
      case arith::CmpIPredicate::uge:
        ++integerPredicates[9];
        break;
      default:
        break;
      }
    }
  });
  EXPECT_EQ(floatPredicates, (std::array<size_t, 6>{1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(integerPredicates,
            (std::array<size_t, 10>{2, 2, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(OpenQASMTargetTest, FoldsAndEmitsCeilingAndFloor) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gate rounded(theta) q {
  rx(sin(theta) + cos(theta)) q;
}
float runtime_value = 0.5;
qubit q;
rx(ceiling(1.25) + floor(-1.25)) q;
rx(ceiling(runtime_value) + floor(runtime_value)) q;
rounded(0.5) q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const auto& constantApplication = std::get<oq3::frontend::GateApplication>(
      analyzed.program->statements[analyzed.program->body[2]].data);
  const auto& parameter =
      analyzed.program->expressions.at(constantApplication.parameters.front());
  ASSERT_EQ(parameter.kind, oq3::frontend::ExpressionKind::Constant);
  EXPECT_EQ(parameter.type, oq3::frontend::ScalarType::Angle);
  EXPECT_EQ(std::get<uint64_t>(parameter.constant), 0);

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t ceilings = 0;
  size_t floors = 0;
  moduleOp->walk([&](Operation* operation) {
    ceilings += isa<math::CeilOp>(operation);
    if (auto floor = dyn_cast<math::FloorOp>(operation)) {
      floors += floor.getOperand().getDefiningOp<arith::DivFOp>() == nullptr;
    }
  });
  EXPECT_EQ(ceilings, 1);
  EXPECT_EQ(floors, 1);
}

TEST(OpenQASMTargetTest, NestsAlternatingControlsAndFlipsPolarityOutside) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(2) @ negctrl @ inv @ ctrl @ x q[0], q[1], q[2], q[3], q[4];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<size_t> controlArities;
  size_t outerPolarityFlips = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
      controlArities.push_back(control.getNumControls());
    }
    if (isa<qc::XOp>(operation) &&
        operation->getParentOfType<qc::CtrlOp>() == nullptr &&
        operation->getParentOfType<qc::InvOp>() == nullptr) {
      ++outerPolarityFlips;
    }
  });
  llvm::sort(controlArities);
  EXPECT_EQ(controlArities, (SmallVector<size_t>{1, 1, 2}));
  EXPECT_EQ(outerPolarityFlips, 2);
}

TEST(OpenQASMTargetTest, EmitsOneControlRegionForNegCtrlArity) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
negctrl(3) @ x q[0], q[1], q[2], q[3];
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t controlRegions = 0;
  size_t polarityFlips = 0;
  moduleOp->walk([&](qc::CtrlOp control) {
    ++controlRegions;
    EXPECT_EQ(control.getNumControls(), 3);
  });
  moduleOp->walk([&](qc::XOp operation) {
    polarityFlips += operation->getParentOfType<qc::CtrlOp>() == nullptr;
  });
  EXPECT_EQ(controlRegions, 1);
  EXPECT_EQ(polarityFlips, 6);
}

TEST(OpenQASMTargetTest, LowersDynamicPowerModifiersToQC) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
float exponent = 0.5;
qubit q;
pow(exponent) @ x q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<qc::PowOp> powers;
  moduleOp->walk([&](qc::PowOp op) { powers.push_back(op); });
  ASSERT_EQ(powers.size(), 1U);
  ASSERT_TRUE(powers.front().getExponentValue().has_value());
  EXPECT_DOUBLE_EQ(*powers.front().getExponentValue(), 0.5);
}

TEST(OpenQASMTargetTest, GuardsRuntimeIntegerPowerModifierExactness) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(
OPENQASM 3.1;
qubit q;
uint exponent = 9007199254740993;
pow(exponent) @ x q;
)qasm",
      R"qasm(
OPENQASM 3.1;
qubit q;
int exponent = 9007199254740992;
bit choose = measure q;
if (choose) { exponent = 9007199254740993; }
pow(exponent) @ x q;
)qasm",
      R"qasm(
OPENQASM 3.1;
qubit q;
uint exponent = 9007199254740993;
bit repeat = measure q;
while (repeat) {
  exponent -= 1;
  repeat = measure q;
}
pow(exponent) @ x q;
)qasm",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    SmallVector<qc::PowOp> powers;
    size_t exactnessAssertions = 0;
    moduleOp->walk([&](Operation* operation) {
      if (auto power = dyn_cast<qc::PowOp>(operation)) {
        powers.push_back(power);
      }
      if (auto assertion = dyn_cast<cf::AssertOp>(operation);
          assertion &&
          assertion.getMsg().contains(
              "power modifier exponent cannot be represented exactly")) {
        ++exactnessAssertions;
      }
    });
    ASSERT_EQ(powers.size(), 1U);
    EXPECT_FALSE(powers.front().getExponentValue().has_value());
    EXPECT_EQ(exactnessAssertions, 1U);
  }
}

TEST(OpenQASMTargetTest,
     LowersCustomGatesConditionalsAndQuantumRuntimeOperations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate pair(theta) left, right {
  rx(theta) left;
  cx left, right;
}
qubit[2] q;
bit c = measure q[0];
if (!c) {
  pair(0.5) q[0], q[1];
} else {
  reset q[1];
}
barrier q;
output bit[2] out;
out = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t conditionals = 0;
  moduleOp->walk([&](Operation* operation) {
    conditionals += operation->getName().getStringRef() == "scf.if";
  });
  EXPECT_EQ(conditionals, 1);

  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t resets = 0;
  size_t barriers = 0;
  moduleOp->walk([&](Operation* operation) {
    auto name = operation->getName().getStringRef();
    resets += name == "qc.reset";
    barriers += name == "qc.barrier";
  });
  EXPECT_EQ(resets, 1);
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, ResolvesManyCustomGateDefinitionsThroughTheIndex) {
  constexpr size_t definitionCount = 2048;
  std::string source = "OPENQASM 3.1;\nqubit q;\n";
  source.reserve(definitionCount * 40);
  for (size_t index = 0; index < definitionCount; ++index) {
    source += "gate g" + std::to_string(index) + " target { x target; }\n";
  }
  for (size_t index = 0; index < definitionCount; ++index) {
    source += "g" + std::to_string(index) + " q;\n";
  }

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t xGates = 0;
  moduleOp->walk(
      [&](Operation* operation) { xGates += isa<qc::XOp>(operation); });
  EXPECT_EQ(xGates, definitionCount);
}

TEST(OpenQASMTargetTest, LowersOpenQASM2ControlledGateCompatibilityPrefixes) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
cccx q[0], q[1], q[2], q[3];
measure q -> c;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t controls = 0;
  moduleOp->walk(
      [&](Operation* operation) { controls += isa<qc::CtrlOp>(operation); });
  EXPECT_EQ(controls, 1);
}

TEST(OpenQASMTargetTest, LowersLanguageBuiltinsOnHardwareQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gphase(pi / 2);
x $3;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t globalPhases = 0;
  size_t xGates = 0;
  moduleOp->walk([&](Operation* operation) {
    globalPhases += isa<qc::GPhaseOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(globalPhases, 1);
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, RejectsMixedQubitAllocationAtTheQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
x q;
x $0;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(moduleOp);
  EXPECT_NE(diagnostic.find("mixing physical and declared qubits"),
            std::string::npos)
      << diagnostic;
  EXPECT_NE(diagnostic.find("QC target"), std::string::npos) << diagnostic;
}

TEST(OpenQASMTargetTest, AllocatesClassicalStorageOnlyForBitOutputs) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit local = measure q;
if (local) {
  x q;
}
output bit result;
result = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t classicalAllocations = 0;
  size_t classicalLoads = 0;
  size_t classicalStores = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto allocation = dyn_cast<memref::AllocOp>(operation);
        allocation && allocation.getType().getElementType().isInteger(1)) {
      ++classicalAllocations;
    }
    if (auto load = dyn_cast<memref::LoadOp>(operation);
        load && load.getMemRefType().getElementType().isInteger(1)) {
      ++classicalLoads;
    }
    if (auto store = dyn_cast<memref::StoreOp>(operation);
        store && store.getMemRefType().getElementType().isInteger(1)) {
      ++classicalStores;
    }
  });
  EXPECT_EQ(classicalAllocations, 1);
  EXPECT_EQ(classicalLoads, 0);
  EXPECT_EQ(classicalStores, 1);
}

TEST(OpenQASMTargetTest, PreservesOrderedScalarAndRegisterOutputs) {
  constexpr llvm::StringLiteral implicitSource = R"qasm(
OPENQASM 3.1;
int count = 1;
bit[2] bits;
bits[0] = true;
bits[1] = false;
float ratio = 2.0;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(implicitSource);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->outputs.size(), 3);
  EXPECT_EQ(analyzed.program->outputs[0].kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(analyzed.program->scalars[analyzed.program->outputs[0].symbol].name,
            "count");
  EXPECT_EQ(analyzed.program->outputs[1].kind,
            oq3::frontend::OutputKind::BitRegister);
  EXPECT_EQ(
      analyzed.program->registers[analyzed.program->outputs[1].symbol].name,
      "bits");
  EXPECT_EQ(analyzed.program->outputs[2].kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(analyzed.program->scalars[analyzed.program->outputs[2].symbol].name,
            "ratio");

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(implicitSource, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  func::ReturnOp result;
  moduleOp->walk([&](func::ReturnOp operation) { result = operation; });
  ASSERT_TRUE(result);
  ASSERT_EQ(result.getNumOperands(), 3);
  EXPECT_TRUE(result.getOperand(0).getType().isInteger(64));
  const auto bitType = dyn_cast<MemRefType>(result.getOperand(1).getType());
  ASSERT_TRUE(bitType);
  EXPECT_EQ(bitType.getShape(), ArrayRef<int64_t>{2});
  EXPECT_TRUE(bitType.getElementType().isInteger(1));
  EXPECT_TRUE(result.getOperand(2).getType().isF64());

  constexpr llvm::StringLiteral explicitSource = R"qasm(
OPENQASM 3.1;
int internal = 1;
output int result;
result = internal + 1;
)qasm";
  auto explicitAnalysis = oq3::frontend::analyzeOpenQASM(explicitSource);
  ASSERT_TRUE(explicitAnalysis) << explicitAnalysis.diagnostics.front().message;
  ASSERT_EQ(explicitAnalysis.program->outputs.size(), 1);
  EXPECT_EQ(explicitAnalysis.program->outputs.front().kind,
            oq3::frontend::OutputKind::Scalar);
  EXPECT_EQ(explicitAnalysis.program
                ->scalars[explicitAnalysis.program->outputs.front().symbol]
                .name,
            "result");
}

TEST(OpenQASMTargetTest, SupportsLargeDirectDynamicQubitAccess) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[3163] q;
qubit[3163] aux;
int i = 0;
  int j = 1;
cx q[i], aux[j];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t loads = 0;
  size_t controls = 0;
  size_t switches = 0;
  moduleOp->walk([&](memref::LoadOp load) {
    loads += isa<qc::QubitType>(load.getType());
  });
  moduleOp->walk([&](qc::CtrlOp) { ++controls; });
  moduleOp->walk([&](scf::IndexSwitchOp) { ++switches; });
  EXPECT_EQ(loads, 2);
  EXPECT_EQ(controls, 1);
  EXPECT_EQ(switches, 0);
}

TEST(OpenQASMTargetTest, RejectsExcessiveCustomGateExpansion) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate g0 q { x q; }\n";
  for (size_t level = 1; level <= 24; ++level) {
    source += "gate g" + std::to_string(level) + " q { g" +
              std::to_string(level - 1) + " q; g" + std::to_string(level - 1) +
              " q; }\n";
  }
  source += "qubit q;\ng24 q;\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(moduleOp);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, AccountsForEachLabelInSwitchCaseBudgets) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate g0 q { x q; }\n";
  for (size_t level = 1; level <= 23; ++level) {
    source += "gate g" + std::to_string(level) + " q { g" +
              std::to_string(level - 1) + " q; g" + std::to_string(level - 1) +
              " q; }\n";
  }
  source += "qubit q;\n"
            "int selector = 0;\n"
            "switch (selector) {\n"
            "  case 0, 1 { g23 q; }\n"
            "  default { }\n"
            "}\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(moduleOp);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos)
      << diagnostic;
}

TEST(OpenQASMTargetTest, DoesNotMultiplyCustomGatesByRegisterWidth) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate expanded a, b {\n";
  for (size_t operation = 0; operation < 25; ++operation) {
    source += operation % 2 == 0 ? "  x a;\n" : "  x b;\n";
  }
  source += "}\n"
            "qubit[640] q;\n"
            "qubit[640] aux;\n"
            "int i = 0;\n"
            "int j = 1;\n"
            "expanded q[i], aux[j];\n";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t loads = 0;
  size_t xGates = 0;
  moduleOp->walk([&](memref::LoadOp load) {
    loads += isa<qc::QubitType>(load.getType());
  });
  moduleOp->walk([&](qc::XOp) { ++xGates; });
  EXPECT_EQ(loads, 2);
  EXPECT_EQ(xGates, 25);
}

TEST(OpenQASMTargetTest, BudgetsRepresentativeOperationConstruction) {
  constexpr auto baseBodies = std::to_array<llvm::StringLiteral>({
      "rx(theta + theta) q;",
      "U(theta, theta, theta) q;",
      "pow(2) @ x q;",
      "for int i in [0:1] { x q; }",
  });
  for (const auto baseBody : baseBodies) {
    std::string source = "OPENQASM 3.1;\n"
                         "include \"stdgates.inc\";\n"
                         "gate g0(theta) q { " +
                         baseBody.str() + " }\n";
    for (size_t level = 1; level <= 24; ++level) {
      source += "gate g" + std::to_string(level) + "(theta) q { g" +
                std::to_string(level - 1) + "(theta) q; g" +
                std::to_string(level - 1) + "(theta) q; }\n";
    }
    source += "qubit q;\ng24(0.5) q;\n";

    MLIRContext context;
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
      diagnostic = value.str();
      return success();
    });
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    EXPECT_FALSE(moduleOp);
    EXPECT_NE(diagnostic.find("projected emitted operation count"),
              std::string::npos)
        << diagnostic;
  }
}

TEST(OpenQASMTargetTest, LowersGateBodyLoopsAndBuiltinConstants) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate repeated(theta) q {
  for int i in [0:2] {
    rx(theta + angle(pi + float(i))) q;
  }
  while (false) { x q; }
}
qubit q;
repeated(0.5) q;
bit result = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t forLoops = 0;
  size_t whileLoops = 0;
  moduleOp->walk([&](Operation* operation) {
    forLoops += isa<scf::ForOp>(operation);
    whileLoops += isa<scf::WhileOp>(operation);
  });
  EXPECT_EQ(forLoops, 1);
  EXPECT_EQ(whileLoops, 1);

  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, GateDefinitionsCaptureGlobalConstants) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const float theta = pi / 2;
gate g q { rx(theta) q; }
qubit q;
g q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  qc::RXOp rotation;
  moduleOp->walk([&](qc::RXOp application) { rotation = application; });
  ASSERT_TRUE(rotation);
  const auto converted =
      mqt::angle::matchQuantizedRadians(rotation.getParameter(0));
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted->bitWidth, 64);
  APInt bits;
  ASSERT_TRUE(matchPattern(converted->bits, m_ConstantInt(&bits)));
  EXPECT_EQ(bits, APInt(64, 1).shl(62));
}

TEST(OpenQASMTargetTest, SupportsWholeBitRegisterAssignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[2] source = measure q;
output bit[2] target;
target = source;
if (target[0] || target[1]) { x q[0]; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const oq3::frontend::BitVectorAssignmentStatement* assignment = nullptr;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* current =
            std::get_if<oq3::frontend::BitVectorAssignmentStatement>(
                &statement.data)) {
      ASSERT_EQ(assignment, nullptr);
      assignment = current;
    }
  }
  ASSERT_NE(assignment, nullptr);
  ASSERT_EQ(analyzed.program->registers.size(), 3);
  EXPECT_EQ(analyzed.program->registers[assignment->target].name, "target");
  const auto& value =
      analyzed.program->bitVectorExpressions.at(assignment->value);
  EXPECT_EQ(value.kind, oq3::frontend::BitVectorExpressionKind::Register);
  EXPECT_EQ(analyzed.program->registers[value.reg].name, "source");
  EXPECT_EQ(value.width, 2);

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<Value> measured;
  moduleOp->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  const auto returned = returnedBitValues(*moduleOp);
  ASSERT_EQ(measured.size(), 2);
  ASSERT_EQ(returned.size(), 2);
  EXPECT_EQ(returned[0], measured[0]);
  EXPECT_EQ(returned[1], measured[1]);
}

TEST(OpenQASMTargetTest, LowersTypedBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
output bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
uint count = popcount(value);
value = rotl(value, 0);
value = rotr(value, -7);
qubit q;
if (count == 3) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_FALSE(analyzed.program->bitVectorExpressions.empty());
  EXPECT_TRUE(
      llvm::any_of(analyzed.program->expressions, [](const auto& expression) {
        return expression.kind == oq3::frontend::ExpressionKind::PopCount &&
               expression.type == oq3::frontend::ScalarType::Uint;
      }));

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t populationCounts = 0;
  size_t funnelShifts = 0;
  moduleOp->walk([&](Operation* operation) {
    populationCounts += isa<math::CtPopOp>(operation);
    funnelShifts += isa<LLVM::FshlOp, LLVM::FshrOp>(operation);
  });
  EXPECT_EQ(populationCounts, 1);
  // Both rotation distances are constant and therefore only permute SSA values.
  EXPECT_EQ(funnelShifts, 0);
}

TEST(OpenQASMTargetTest, ReusesPackedNestedDynamicRotations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
int distance = -7;
uint count = popcount(rotl(rotr(value, distance), 1));
qubit q;
if (count == 3) { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t leftShifts = 0;
  size_t rightShifts = 0;
  size_t populationCounts = 0;
  size_t unpackingTruncations = 0;
  moduleOp->walk([&](Operation* operation) {
    leftShifts += isa<LLVM::FshlOp>(operation);
    rightShifts += isa<LLVM::FshrOp>(operation);
    populationCounts += isa<math::CtPopOp>(operation);
    if (auto truncation = dyn_cast<arith::TruncIOp>(operation);
        truncation && truncation.getOut().getType().isInteger(1)) {
      ++unpackingTruncations;
    }
  });
  EXPECT_EQ(leftShifts, 1);
  EXPECT_EQ(rightShifts, 1);
  EXPECT_EQ(populationCounts, 1);
  // The nested packed value reaches popcount without an unpack/repack cycle.
  EXPECT_EQ(unpackingTruncations, 0);
}

TEST(OpenQASMTargetTest, CarriesAtomicRotationsThroughControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
output bit[5] value;
value[0] = true;
value[1] = false;
value[2] = true;
value[3] = false;
value[4] = true;
qubit q;
bit condition = measure q;
if (condition) {
  value = rotl(value, 1);
}
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  bool carriedWholeRegister = false;
  moduleOp->walk([&](scf::IfOp conditional) {
    carriedWholeRegister |= conditional.getNumResults() == 5;
  });
  EXPECT_TRUE(carriedWholeRegister);
}

TEST(OpenQASMTargetTest, SelfRotationSnapshotsTheWholeRegister) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[5] q;
output bit[5] result;
result = measure q;
result = rotl(result, 2);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<Value> measured;
  moduleOp->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  const auto returned = returnedBitValues(*moduleOp);
  ASSERT_EQ(measured.size(), 5);
  ASSERT_EQ(returned.size(), 5);
  EXPECT_EQ(returned[0], measured[3]);
  EXPECT_EQ(returned[1], measured[4]);
  EXPECT_EQ(returned[2], measured[0]);
  EXPECT_EQ(returned[3], measured[1]);
  EXPECT_EQ(returned[4], measured[2]);
}

TEST(OpenQASMTargetTest, SupportsWidthOneBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
output bit[1] value;
value[0] = measure q;
int distance = -3;
value = rotl(value, distance);
value = rotr(value, 4);
uint count = popcount(value);
rx(count) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t populationCounts = 0;
  size_t leftShifts = 0;
  size_t rightShifts = 0;
  moduleOp->walk([&](Operation* operation) {
    populationCounts += isa<math::CtPopOp>(operation);
    leftShifts += isa<LLVM::FshlOp>(operation);
    rightShifts += isa<LLVM::FshrOp>(operation);
  });
  EXPECT_EQ(populationCounts, 1);
  EXPECT_EQ(leftShifts, 1);
  EXPECT_EQ(rightShifts, 0);
}

TEST(OpenQASMTargetTest, SupportsWideBitVectorBuiltins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[65] q;
bit[65] value = measure q;
int distance = 3;
value = rotl(value, distance);
uint count = popcount(value);
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t wideFunnelShifts = 0;
  size_t populationCounts = 0;
  size_t narrowedCounts = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto shift = dyn_cast<LLVM::FshlOp>(operation)) {
      wideFunnelShifts += shift.getResult().getType().isInteger(65);
    }
    if (auto count = dyn_cast<math::CtPopOp>(operation)) {
      populationCounts += count.getResult().getType().isInteger(65);
    }
    if (auto truncation = dyn_cast<arith::TruncIOp>(operation)) {
      narrowedCounts += truncation.getIn().getType().isInteger(65) &&
                        truncation.getOut().getType().isInteger(64);
    }
  });
  EXPECT_EQ(wideFunnelShifts, 1);
  EXPECT_EQ(populationCounts, 1);
  EXPECT_EQ(narrowedCounts, 1);
}

TEST(OpenQASMTargetTest, RotationsProduceSpecifiedBitResults) {
  constexpr std::array input{true, false, true, true, false};
  constexpr std::array<int64_t, 5> distances{0, 2, -2, 7, -7};
  std::string source = "OPENQASM 3.1;\n";
  std::vector<std::vector<bool>> expectedResults;
  size_t resultIndex = 0;
  for (const bool runtime : {false, true}) {
    for (const auto distance : distances) {
      for (const bool left : {true, false}) {
        const auto resultName = "result" + std::to_string(resultIndex);
        source += "output bit[5] " + resultName + ";\n";
        for (size_t bit = 0; bit < input.size(); ++bit) {
          source += resultName + "[" + std::to_string(bit) +
                    "] = " + (input[bit] ? "true;\n" : "false;\n");
        }
        std::string distanceExpression = std::to_string(distance);
        if (runtime) {
          const auto distanceName = "distance" + std::to_string(resultIndex);
          source.append("int ")
              .append(distanceName)
              .append(" = ")
              .append(distanceExpression)
              .append(";\n");
          distanceExpression = distanceName;
        }
        source.append(resultName)
            .append(" = ")
            .append(left ? "rotl(" : "rotr(")
            .append(resultName)
            .append(", ")
            .append(distanceExpression)
            .append(");\n");
        expectedResults.push_back(rotateBits(input, distance, left));
        ++resultIndex;
      }
    }
  }

  const auto outputs = canonicalizedBitOutputs(source);
  ASSERT_EQ(outputs.size(), expectedResults.size() * input.size());
  std::vector<std::vector<bool>> actualResults;
  actualResults.reserve(expectedResults.size());
  for (size_t result = 0; result < expectedResults.size(); ++result) {
    const auto begin =
        outputs.begin() + static_cast<ptrdiff_t>(result * input.size());
    actualResults.emplace_back(begin, begin + input.size());
    EXPECT_EQ(actualResults.back(), expectedResults[result])
        << "rotation result " << result;
  }

  for (size_t runtime = 0; runtime < 2; ++runtime) {
    for (size_t distance = 0; distance < distances.size(); ++distance) {
      // libc++ uses a pointer here, whereas MSVC uses an iterator class.
      const auto opposite = // NOLINT(readability-qualified-auto)
          std::ranges::find(distances, -distances[distance]);
      ASSERT_NE(opposite, distances.end());
      const auto oppositeIndex =
          static_cast<size_t>(opposite - distances.begin());
      const auto left = ((runtime * distances.size()) + distance) * 2;
      const auto oppositeRight =
          (((runtime * distances.size()) + oppositeIndex) * 2) + 1;
      EXPECT_EQ(actualResults[left], actualResults[oppositeRight])
          << "rotl(a, n) differs from rotr(a, -n) for n = "
          << distances[distance];
    }
  }
}

TEST(OpenQASMTargetTest, PopcountProducesSpecifiedResult) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit[5] source;
source[0] = true;
source[1] = false;
source[2] = true;
source[3] = true;
source[4] = false;
output bit[6] result;
result[0] = false;
result[1] = false;
result[2] = false;
result[3] = false;
result[4] = false;
result[5] = false;
result[popcount(source)] = true;
)qasm";

  EXPECT_EQ(canonicalizedBitOutputs(source),
            (std::vector<bool>{false, false, false, true, false, false}));
}

TEST(OpenQASMTargetTest, SupportsOpenQASM2RegisterConditions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q -> c;
if (c == 1) x q[0];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t conditionals = 0;
  moduleOp->walk([&](scf::IfOp) { ++conditionals; });
  // The register equality and the source-level branch each short-circuit
  // through their own structured conditional.
  EXPECT_EQ(conditionals, 2);
}

TEST(OpenQASMTargetTest, ZeroInitializesUnmeasuredOpenQASM2Registers) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[2];
if (c == 0) x q[0];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  const auto returned = returnedBitValues(*moduleOp);
  ASSERT_EQ(returned.size(), 2);
  for (const auto value : returned) {
    const auto constant = evaluateConstantInteger(value);
    ASSERT_TRUE(constant);
    EXPECT_TRUE(constant->isZero());
  }
  size_t conditionals = 0;
  size_t xGates = 0;
  moduleOp->walk([&](scf::IfOp) { ++conditionals; });
  moduleOp->walk([&](qc::XOp) { ++xGates; });
  // The register equality and source branch retain their structured form;
  // the initialized storage provides their runtime value.
  EXPECT_EQ(conditionals, 2);
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, ZeroInitializesUntouchedOpenQASM2RegisterBits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[2];
h q[0];
measure q[0] -> c[0];
if (c == 1) x q[0];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  const auto returned = returnedBitValues(*moduleOp);
  ASSERT_EQ(returned.size(), 2);
  const auto untouchedBit = evaluateConstantInteger(returned[1]);
  ASSERT_TRUE(untouchedBit);
  EXPECT_TRUE(untouchedBit->isZero());

  size_t conditionals = 0;
  moduleOp->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_GT(conditionals, 0);
}

TEST(OpenQASMTargetTest, SelectsFloatingPowForNegativeSignedExponent) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int base = 4;
float result = pow(base, -2);
qubit q;
if (result == 0.0625) { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  bool foundResult = false;
  moduleOp->walk([&](arith::ConstantFloatOp constant) {
    foundResult |= constant.value().convertToDouble() == 0.0625;
  });
  EXPECT_TRUE(foundResult);
}

TEST(OpenQASMTargetTest, PreservesSignedPowResultWithUnsignedExponent) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int base = -2;
uint exponent = 3;
output int result;
result = pow(base, exponent);
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const oq3::frontend::ScalarAssignmentStatement* assignment = nullptr;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* current =
            std::get_if<oq3::frontend::ScalarAssignmentStatement>(
                &statement.data);
        current != nullptr &&
        analyzed.program->scalars[current->scalar].name == "result") {
      assignment = current;
    }
  }
  ASSERT_NE(assignment, nullptr);
  ASSERT_TRUE(assignment->value);
  const auto& power = analyzed.program->expressions[*assignment->value];
  EXPECT_EQ(power.kind, oq3::frontend::ExpressionKind::Power);
  EXPECT_EQ(power.type, oq3::frontend::ScalarType::Int);
  EXPECT_EQ(analyzed.program->expressions[power.lhs].type,
            oq3::frontend::ScalarType::Int);
  EXPECT_EQ(analyzed.program->expressions[power.rhs].type,
            oq3::frontend::ScalarType::Uint);

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t integerPowers = 0;
  size_t floatConversions = 0;
  moduleOp->walk([&](scf::WhileOp) { ++integerPowers; });
  moduleOp->walk([&](Operation* operation) {
    floatConversions += isa<arith::SIToFPOp, arith::UIToFPOp>(operation);
  });
  EXPECT_EQ(integerPowers, 1);
  EXPECT_EQ(floatConversions, 0);
}

TEST(OpenQASMTargetTest, SupportsBitMeasurementReassignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
bit measured;
measured = measure q;
if (measured) { x q; }
measured = measure q;
if (!measured) { h q; }
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t measurements = 0;
  moduleOp->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 2);
}

TEST(OpenQASMTargetTest, EmitsStructuredDiagnosticsWithIncludeStacks) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "OPENQASM 3.1;\ninclude \"stdgates.inc\";\n"
                                   "include \"outer.inc\";\n",
                                   "main.qasm"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"nested.inc\";\n", "outer.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "qubit q;\npow(9007199254740993) @ x q;\n", "nested.inc"),
      llvm::SMLoc());

  MLIRContext context;
  std::string message;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    message = diagnostic.str();
    location = diagnostic.getLocation();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(sourceMgr, &context);
  EXPECT_FALSE(moduleOp);
  EXPECT_NE(message.find("cannot be represented exactly"), std::string::npos);
  const auto mainCall = dyn_cast<CallSiteLoc>(location);
  ASSERT_TRUE(mainCall);
  const auto mainLocation = dyn_cast<FileLineColLoc>(mainCall.getCaller());
  ASSERT_TRUE(mainLocation);
  EXPECT_EQ(mainLocation.getFilename(), "main.qasm");
  EXPECT_EQ(mainLocation.getLine(), 3);
  const auto outerCall = dyn_cast<CallSiteLoc>(mainCall.getCallee());
  ASSERT_TRUE(outerCall);
  const auto outerLocation = dyn_cast<FileLineColLoc>(outerCall.getCaller());
  ASSERT_TRUE(outerLocation);
  EXPECT_EQ(outerLocation.getFilename(), "outer.inc");
  EXPECT_EQ(outerLocation.getLine(), 1);
  const auto nestedLocation = dyn_cast<FileLineColLoc>(outerCall.getCallee());
  ASSERT_TRUE(nestedLocation);
  EXPECT_EQ(nestedLocation.getFilename(), "nested.inc");
  EXPECT_EQ(nestedLocation.getLine(), 2);
}

TEST(OpenQASMTargetTest, NormalizesNegativeIndicesAndChecksDynamicAliases) {
  constexpr llvm::StringLiteral indexSource = R"qasm(
OPENQASM 3.1;
qubit[3] q;
x q[-1];
bit[3] c = measure q;
if (c[-1]) { h q[-1]; }
int i = -1;
x q[i];
c[i] = measure q[i];
if (c[i]) { x q[0]; }
output bit[3] result;
result = measure q;
)qasm";
  MLIRContext indexContext;
  auto indexed = qc::translateQASM3ToQC(indexSource, &indexContext);
  ASSERT_TRUE(indexed);
  ASSERT_TRUE(succeeded(verify(*indexed)));
  size_t indexSelections = 0;
  indexed->walk([&](arith::SelectOp) { ++indexSelections; });
  EXPECT_GE(indexSelections, 3);

  constexpr llvm::StringLiteral aliasSource = R"qasm(
OPENQASM 3.1;
qubit[2] q;
int i = 0;
cx q[i], q[i];
bit[2] result = measure q;
)qasm";
  MLIRContext aliasContext;
  auto aliased = qc::translateQASM3ToQC(aliasSource, &aliasContext);
  ASSERT_TRUE(aliased);
  ASSERT_TRUE(succeeded(verify(*aliased)));
  size_t aliasAssertions = 0;
  aliased->walk([&](cf::AssertOp) { ++aliasAssertions; });
  EXPECT_GE(aliasAssertions, 3);
}

TEST(OpenQASMTargetTest, ExtendsNarrowUnsignedDynamicIndices) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
input uint[8] index;
qubit[3] q;
x q[index];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t indexExtensions = 0;
  moduleOp->walk([&](arith::ExtUIOp extension) {
    indexExtensions += extension.getIn().getType().isInteger(8) &&
                       extension.getType().isInteger(64);
  });
  EXPECT_EQ(indexExtensions, 1);
}

TEST(OpenQASMTargetTest, LoadsDynamicQubitGatesDirectly) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
x q[i];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t switches = 0;
  size_t loads = 0;
  moduleOp->walk([&](scf::IndexSwitchOp) { ++switches; });
  moduleOp->walk([&](memref::LoadOp load) {
    if (!isa<qc::QubitType>(load.getType())) {
      return;
    }
    ++loads;
    EXPECT_TRUE(load.getIndices().front().getType().isIndex());
  });
  EXPECT_EQ(switches, 0);
  EXPECT_EQ(loads, 1);
}

TEST(OpenQASMTargetTest, LowersNativeSwitchWithCasesAndCarriedState) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
int selector = 2;
int result = 0;
switch (selector) {
  case 1, 3 {
    result = 10;
    x q;
  }
  case 2 {
    result = 20;
    h q;
  }
  default {
    result = 30;
  }
}
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t switches = 0;
  moduleOp->walk([&](scf::IndexSwitchOp switchOp) {
    ++switches;
    EXPECT_EQ(switchOp.getCases(), ArrayRef<int64_t>({1, 3, 2}));
    ASSERT_EQ(switchOp.getNumResults(), 1);
    EXPECT_TRUE(switchOp.getResult(0).getType().isInteger(64));
  });
  EXPECT_EQ(switches, 1);
}

TEST(OpenQASMTargetTest, LoadsDynamicQubitMeasurementsDirectly) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit c;
int i = 0;
c = measure q[i];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t switches = 0;
  size_t measurements = 0;
  moduleOp->walk([&](scf::IndexSwitchOp) { ++switches; });
  moduleOp->walk([&](qc::MeasureOp measurement) {
    ++measurements;
    auto load = measurement.getQubit().getDefiningOp<memref::LoadOp>();
    ASSERT_TRUE(load);
    EXPECT_TRUE(load.getIndices().front().getType().isIndex());
  });
  EXPECT_EQ(switches, 0);
  EXPECT_EQ(measurements, 1);
}

TEST(OpenQASMTargetTest, HandlesWidthOneAndMultipleDynamicQubitLoads) {
  constexpr llvm::StringLiteral widthOneSource = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[1] q;
int i = 0;
x q[i];
)qasm";
  MLIRContext widthOneContext;
  auto widthOneModule =
      qc::translateQASM3ToQC(widthOneSource, &widthOneContext);
  ASSERT_TRUE(widthOneModule);
  size_t widthOneSwitches = 0;
  size_t widthOneLoads = 0;
  widthOneModule->walk([&](scf::IndexSwitchOp) { ++widthOneSwitches; });
  widthOneModule->walk([&](memref::LoadOp load) {
    widthOneLoads += isa<qc::QubitType>(load.getType());
  });
  EXPECT_EQ(widthOneSwitches, 0);
  EXPECT_EQ(widthOneLoads, 1);

  constexpr llvm::StringLiteral nestedSource = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] left;
qubit[2] right;
int i = 0;
int j = 1;
cx left[i], right[j];
)qasm";
  MLIRContext nestedContext;
  auto nestedModule = qc::translateQASM3ToQC(nestedSource, &nestedContext);
  ASSERT_TRUE(nestedModule);
  ASSERT_TRUE(succeeded(verify(*nestedModule)));
  size_t switches = 0;
  size_t controls = 0;
  nestedModule->walk([&](scf::IndexSwitchOp) { ++switches; });
  nestedModule->walk([&](qc::CtrlOp) { ++controls; });
  size_t loads = 0;
  nestedModule->walk([&](memref::LoadOp load) {
    loads += isa<qc::QubitType>(load.getType());
  });
  EXPECT_EQ(switches, 0);
  EXPECT_EQ(controls, 1);
  EXPECT_EQ(loads, 2);
}

TEST(OpenQASMTargetTest, LoadsEveryDynamicQuantumStatementDirectly) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
int i = 0;
int j = 1;
negctrl @ x q[i], q[j];
bit measured = measure q[i];
reset q[j];
barrier q[i], q[j];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t switches = 0;
  size_t loads = 0;
  size_t measurements = 0;
  size_t resets = 0;
  size_t barriers = 0;
  size_t distinctnessAssertions = 0;
  moduleOp->walk([&](scf::IndexSwitchOp) { ++switches; });
  moduleOp->walk([&](memref::LoadOp load) {
    if (isa<qc::QubitType>(load.getType())) {
      ++loads;
      EXPECT_TRUE(load.getIndices().front().getType().isIndex());
    }
  });
  moduleOp->walk([&](qc::MeasureOp) { ++measurements; });
  moduleOp->walk([&](qc::ResetOp) { ++resets; });
  moduleOp->walk([&](qc::BarrierOp) { ++barriers; });
  moduleOp->walk([&](cf::AssertOp assertion) {
    distinctnessAssertions += assertion.getMsg().ends_with(
        "operands must not reference the same qubit");
  });
  EXPECT_EQ(switches, 0);
  EXPECT_EQ(loads, 6);
  EXPECT_EQ(measurements, 1);
  EXPECT_EQ(resets, 1);
  EXPECT_EQ(barriers, 1);
  EXPECT_EQ(distinctnessAssertions, 2);
}

TEST(OpenQASMTargetTest, QuantumEmissionDoesNotScaleWithRegisterWidth) {
  const auto operationCount = [](const int64_t width) {
    const auto source = "OPENQASM 3.1;\ninclude \"stdgates.inc\";\nqubit[" +
                        std::to_string(width) + "] q;\nint i = 0;\nh q[i];\n";
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    EXPECT_TRUE(moduleOp);
    if (!moduleOp) {
      return size_t{0};
    }
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    size_t operations = 0;
    moduleOp->walk([&](Operation*) { ++operations; });
    return operations;
  };

  EXPECT_EQ(operationCount(2), operationCount(100'000));
}

TEST(OpenQASMTargetTest, LargeStaticBarrierAvoidsAliasChecks) {
  constexpr size_t width = 10'000;
  const auto source =
      "OPENQASM 3.1;\nqubit[" + std::to_string(width) + "] q;\nbarrier q;\n";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  size_t loads = 0;
  size_t barriers = 0;
  size_t assertions = 0;
  moduleOp->walk([&](memref::LoadOp load) {
    if (isa<qc::QubitType>(load.getType())) {
      ++loads;
    }
  });
  moduleOp->walk([&](qc::BarrierOp) { ++barriers; });
  moduleOp->walk([&](cf::AssertOp) { ++assertions; });
  EXPECT_EQ(loads, width);
  EXPECT_EQ(barriers, 1);
  EXPECT_EQ(assertions, 0);
}

TEST(OpenQASMTargetTest, SupportsOrdinaryBitInitializationAndAssignment) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit enabled = false;
enabled = true;
bit[2] flags;
flags[0] = enabled;
flags[1] = !enabled;
if (flags[0] && !flags[1]) { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t xGates = 0;
  moduleOp->walk([&](qc::XOp) { ++xGates; });
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, SupportsTargetlessMeasurements) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t measurements = 0;
  moduleOp->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 1);
}

TEST(OpenQASMTargetTest, PromotesMixedRangeEndpointsBeforeIteration) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const uint start = 0;
const int stop = -1;
qubit q;
for uint i in [start:-1:stop] { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  size_t loops = 0;
  size_t xGates = 0;
  moduleOp->walk([&](Operation* operation) {
    loops += isa<scf::ForOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(loops, 0);
  EXPECT_EQ(xGates, 0);
}

TEST(OpenQASMTargetTest, ThreadsGateParametersIntoWhileConditions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate conditional(theta) q {
  while (theta > 0.0) { x q; }
}
qubit q;
conditional(0.0) q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t loops = 0;
  moduleOp->walk([&](scf::WhileOp) { ++loops; });
  EXPECT_EQ(loops, 1);
}

TEST(OpenQASMTargetTest, RejectsModifiersOnStructuredCustomGatesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped(theta) q {
  for int i in [0:0] { p(theta) q; }
}
qubit q;
inv @ looped(pi / 2) q;
)qasm";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(moduleOp);
  EXPECT_NE(diagnostic.find("structured control flow"), std::string::npos);
}

TEST(OpenQASMTargetTest,
     RejectsModifiersOnTransitivelyStructuredCustomGatesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped q { for int i in [0:0] { x q; } }
gate wrapper q { looped q; }
qubit q;
inv @ wrapper q;
)qasm";

  MLIRContext context;
  std::string diagnostic;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    location = value.getLocation();
    return success();
  });
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(moduleOp);
  const auto fileLocation = dyn_cast<FileLineColLoc>(location);
  ASSERT_TRUE(fileLocation);
  EXPECT_EQ(fileLocation.getFilename(), "<input>");
  EXPECT_EQ(fileLocation.getLine(), 7);
  EXPECT_EQ(fileLocation.getColumn(), 1);
  EXPECT_NE(diagnostic.find("structured control flow"), std::string::npos);
}

TEST(OpenQASMTargetTest, IgnoresUnreachableStructuredCustomGates) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped q { for int i in [0:0] { x q; } }
gate wrapper q { looped q; }
qubit q;
x q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, LowersRuntimeDynamicIndicesWithBoundsChecks) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit choose = measure q[0];
if (choose) { i = 1; }
x q[i];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t assertions = 0;
  moduleOp->walk([&](cf::AssertOp) { ++assertions; });
  EXPECT_GE(assertions, 1);
}

TEST(OpenQASMTargetTest, LowersRuntimeIndicesAcrossStatementKinds) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(OPENQASM 3.1; include "stdgates.inc"; qubit[2] q;
bit[2] c = measure q;
int i = 0; bit choose = measure q[0]; if (choose) { i = 1; }
if (c[i]) { x q[0]; })qasm",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } bool value = c[i];",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bool value = false; "
      "bit choose = measure q[0]; if (choose) { i = 1; } value = c[i];",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } c[i] = true;",
      "OPENQASM 3.1; qubit[2] q; bit[2] c = measure q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } "
      "c[i] = measure q[0];",
      "OPENQASM 3.1; qubit[2] q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } reset q[i];",
      "OPENQASM 3.1; qubit[2] q; int i = 0; "
      "bit choose = measure q[0]; if (choose) { i = 1; } barrier q[i];",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(moduleOp);
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
  }
}

TEST(OpenQASMTargetTest, LowersLoopVariantDynamicIndicesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit repeat = measure q[0];
while (repeat) { x q[i]; i = 1; repeat = measure q[0]; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, LowersMultiIterationInductionIndicesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
for uint i in [0:2] { int x = i + 1; h q[x]; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, LowersCheckedIntegerArithmeticAtQCTarget) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
int turns = 0;
for int i in [0:2] { turns += 1; }
rx(turns) q;
)qasm",
      "OPENQASM 3.1; int value = 1; int derived = value + 1;",
      R"qasm(OPENQASM 3.1; include "stdgates.inc"; qubit q; int value = 1;
if (value + 1 > 0) { x q; })qasm",
      "OPENQASM 3.1; include \"stdgates.inc\"; qubit q; int value = 1; "
      "rx(value + 1) q;",
      "OPENQASM 3.1; int value = 1; bool result = value + 1 > 0;",
      "OPENQASM 3.1; int value = 1; bit result; result = value + 1 > 0;",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    size_t assertions = 0;
    moduleOp->walk([&](cf::AssertOp) { ++assertions; });
    EXPECT_GE(assertions, 1);
  }
}

TEST(OpenQASMTargetTest, LowersRuntimeSignedAndUnsignedIntegerOperators) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int signedValue = 7;
int signedOperand = 2;
signedValue = -signedValue;
signedValue = signedValue + signedOperand;
signedValue = signedValue - signedOperand;
signedValue = signedValue * signedOperand;
signedValue = signedValue / signedOperand;
signedValue = signedValue % signedOperand;
signedValue = signedValue ** signedOperand;
uint unsignedValue = 7;
uint unsignedOperand = 2;
unsignedValue = -unsignedValue;
unsignedValue = unsignedValue + unsignedOperand;
unsignedValue = unsignedValue - unsignedOperand;
unsignedValue = unsignedValue * unsignedOperand;
unsignedValue = unsignedValue / unsignedOperand;
unsignedValue = unsignedValue % unsignedOperand;
unsignedValue = unsignedValue ** unsignedOperand;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t assertions = 0;
  size_t powerLoops = 0;
  moduleOp->walk([&](cf::AssertOp) { ++assertions; });
  moduleOp->walk([&](scf::WhileOp) { ++powerLoops; });
  EXPECT_GE(assertions, 7);
  EXPECT_EQ(powerLoops, 2);
}

TEST(OpenQASMTargetTest, UsesConstantBoundsForStaticInclusiveRanges) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; for int i in [0:1:2] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [2:-1:0] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [3:1:0] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [7:1:7] { x q; }",
      "OPENQASM 3.1; qubit q; for int i in [0:2:3] { x q; }",
      R"qasm(OPENQASM 3.1; qubit q;
for int i in [9223372036854775806:1:9223372036854775807] { x q; })qasm",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    size_t forLoops = 0;
    size_t whileLoops = 0;
    size_t divisions = 0;
    moduleOp->walk([&](scf::ForOp) { ++forLoops; });
    moduleOp->walk([&](scf::WhileOp) { ++whileLoops; });
    moduleOp->walk([&](arith::DivUIOp) { ++divisions; });
    EXPECT_EQ(forLoops, 1);
    EXPECT_EQ(whileLoops, 0);
    EXPECT_EQ(divisions, 0);
  }
}

TEST(OpenQASMTargetTest, UsesComparisonDrivenDynamicInclusiveRanges) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
int start = 0;
int step = 1;
int stop = 2;
bit choose = measure q;
if (choose) { start = 1; }
for int i in [start:step:stop] { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t whileLoops = 0;
  size_t divisions = 0;
  size_t assertions = 0;
  moduleOp->walk([&](scf::WhileOp) { ++whileLoops; });
  moduleOp->walk([&](arith::DivUIOp) { ++divisions; });
  moduleOp->walk([&](cf::AssertOp) { ++assertions; });
  EXPECT_EQ(whileLoops, 1);
  EXPECT_EQ(divisions, 0);
  EXPECT_GE(assertions, 1);
}

TEST(OpenQASMTargetTest, PreservesStaticallySelectedIndexState) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
if (false) { i = 1; pow(2) @ x q[1]; }
x q[i];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t conditionals = 0;
  size_t indexSwitches = 0;
  size_t xGates = 0;
  size_t qubitLoads = 0;
  size_t powers = 0;
  moduleOp->walk([&](scf::IfOp) { ++conditionals; });
  moduleOp->walk([&](scf::IndexSwitchOp) { ++indexSwitches; });
  moduleOp->walk([&](qc::XOp) { ++xGates; });
  moduleOp->walk([&](memref::LoadOp load) {
    qubitLoads += isa<qc::QubitType>(load.getType());
  });
  moduleOp->walk([&](qc::PowOp) { ++powers; });
  EXPECT_EQ(conditionals, 0);
  EXPECT_EQ(indexSwitches, 0);
  EXPECT_EQ(xGates, 1);
  EXPECT_EQ(qubitLoads, 1);
  EXPECT_EQ(powers, 0);
}

TEST(OpenQASMTargetTest, PreservesEqualConstantIndexJoins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit choose = measure q[0];
if (choose) { i = 1; } else { i = 1; }
x q[i];
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, LowersShortCircuitBooleanEvaluation) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] measured = measure q;
float negative = -1.0;
float notANumber = sqrt(negative);
if ((measured[0] && measured[1]) || notANumber != notANumber) { x q[0]; }
output bit[2] result;
result = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<int64_t> firstMeasuredIndices;
  bool sawUnorderedInequality = false;
  size_t shortCircuitOperations = 0;
  size_t eagerLogicalOperations = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto comparison = dyn_cast<arith::CmpFOp>(operation)) {
      sawUnorderedInequality |=
          comparison.getPredicate() == arith::CmpFPredicate::UNE;
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      shortCircuitOperations += conditional.getNumResults() == 1;
    }
    eagerLogicalOperations += isa<arith::AndIOp, arith::OrIOp>(operation);
    auto measurement = dyn_cast<qc::MeasureOp>(operation);
    if (!measurement || firstMeasuredIndices.size() == 2) {
      return;
    }
    auto load = measurement.getQubit().getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().empty()) {
      return;
    }
    APInt index;
    if (matchPattern(load.getIndices().front(), m_ConstantInt(&index))) {
      firstMeasuredIndices.push_back(index.getSExtValue());
    }
  });
  EXPECT_EQ(firstMeasuredIndices, (SmallVector<int64_t>{0, 1}));
  EXPECT_EQ(shortCircuitOperations, 2);
  EXPECT_EQ(eagerLogicalOperations, 0);
  EXPECT_TRUE(sawUnorderedInequality);
}

TEST(OpenQASMTargetTest, EmitsStructuredLoopsWithCarriedMutableState) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
float theta = 0.0;
for int i in [0:2] {
  theta += 0.125;
  h q;
}
bit repeat = measure q;
while (repeat) {
  theta += 0.25;
  rx(theta) q;
  repeat = measure q;
}
rx(theta) q;
bit result = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::ForOp forLoop;
  scf::WhileOp whileLoop;
  moduleOp->walk([&](Operation* operation) {
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      forLoop = loop;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      whileLoop = loop;
    }
  });
  ASSERT_TRUE(forLoop);
  ASSERT_TRUE(whileLoop);
  EXPECT_EQ(forLoop.getInitArgs().size(), 1);
  EXPECT_EQ(forLoop.getNumResults(), 1);
  EXPECT_EQ(forLoop.getBody()->getTerminator()->getNumOperands(), 1);
  EXPECT_EQ(whileLoop.getInits().size(), 2);
  EXPECT_EQ(whileLoop.getNumResults(), 2);
  EXPECT_EQ(whileLoop.getBeforeBody()->getTerminator()->getNumOperands(), 3);
  EXPECT_EQ(whileLoop.getAfterBody()->getTerminator()->getNumOperands(), 2);
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  forLoop = {};
  moduleOp->walk([&](scf::ForOp loop) { forLoop = loop; });
  ASSERT_TRUE(forLoop);
  APInt lower;
  APInt upper;
  APInt step;
  ASSERT_TRUE(matchPattern(forLoop.getLowerBound(), m_ConstantInt(&lower)));
  ASSERT_TRUE(matchPattern(forLoop.getUpperBound(), m_ConstantInt(&upper)));
  ASSERT_TRUE(matchPattern(forLoop.getStep(), m_ConstantInt(&step)));
  EXPECT_EQ(lower.getSExtValue(), 0);
  EXPECT_EQ(upper.getSExtValue(), 3);
  EXPECT_EQ(step.getSExtValue(), 1);

  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(OpenQASMTargetTest, PreservesBranchAndWhileCarriedClassicalBits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bool choose = true;
bit branch;
if (choose) {
  branch = measure q[0];
} else {
  branch = measure q[1];
}
while (branch) {
  h q[0];
  branch = measure q[0];
}
if (branch) { x q[1]; }
output bit[2] result;
result = measure q;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t resultBearingConditionals = 0;
  size_t resultBearingWhiles = 0;
  moduleOp->walk([&](Operation* operation) {
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      resultBearingConditionals += conditional.getNumResults() != 0;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      resultBearingWhiles += loop.getNumResults() != 0;
    }
  });
  EXPECT_EQ(resultBearingConditionals, 1);
  EXPECT_EQ(resultBearingWhiles, 1);
  moduleOp->walk([&](scf::IfOp conditional) {
    if (conditional.getNumResults() == 0) {
      return;
    }
    EXPECT_EQ(conditional.getNumResults(), 1);
    EXPECT_TRUE(conditional.getResult(0).getType().isInteger(1));
    EXPECT_EQ(
        conditional.getThenRegion().front().getTerminator()->getNumOperands(),
        1);
    EXPECT_EQ(
        conditional.getElseRegion().front().getTerminator()->getNumOperands(),
        1);
  });
  moduleOp->walk([&](scf::WhileOp loop) {
    EXPECT_EQ(loop.getInits().size(), 1);
    EXPECT_EQ(loop.getNumResults(), 1);
    EXPECT_EQ(loop.getBeforeBody()->getTerminator()->getNumOperands(), 2);
    EXPECT_EQ(loop.getAfterBody()->getTerminator()->getNumOperands(), 1);
  });

  ASSERT_TRUE(succeeded(runQCCleanupPipeline(*moduleOp)));
  auto emitted = qc::translateQCToOpenQASM3(*moduleOp);
  ASSERT_TRUE(succeeded(emitted));
  EXPECT_NE(emitted->find("bit _mqt_s"), std::string::npos) << *emitted;
  auto reparsed = oq3::frontend::analyzeOpenQASM(*emitted);
  ASSERT_TRUE(reparsed) << reparsed.diagnostics.front().message << '\n'
                        << *emitted;
}

TEST(OpenQASMTargetTest, CarriesEveryPotentiallyMutatedDynamicBit) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
uint index = 0;
bit[2] state;
state[0] = false;
state[1] = false;
bool choose = true;
if (choose) {
  state[index] = measure q;
} else {
  state[index] = true;
}
if (state[0]) { x q; }
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::IfOp stateUpdate;
  moduleOp->walk([&](scf::IfOp conditional) {
    if (conditional.getNumResults() == 2) {
      stateUpdate = conditional;
    }
  });
  ASSERT_TRUE(stateUpdate);
  EXPECT_TRUE(llvm::all_of(stateUpdate.getResultTypes(),
                           [](Type type) { return type.isInteger(1); }));
  EXPECT_EQ(
      stateUpdate.getThenRegion().front().getTerminator()->getNumOperands(), 2);
  EXPECT_EQ(
      stateUpdate.getElseRegion().front().getTerminator()->getNumOperands(), 2);
}

TEST(OpenQASMTargetTest, HandlesTheMaximumUnsignedSingletonRange) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
qubit q;
for uint i in [maximum:maximum] { if (i == maximum) { x q; } }
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  scf::ForOp loop;
  moduleOp->walk([&](scf::ForOp current) { loop = current; });
  ASSERT_TRUE(loop);
  APInt lower;
  APInt step;
  ASSERT_TRUE(matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)));
  ASSERT_TRUE(matchPattern(loop.getStep(), m_ConstantInt(&step)));
  EXPECT_EQ(lower.getSExtValue(), 0);
  EXPECT_EQ(step.getSExtValue(), 1);

  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));
  size_t remainingLoops = 0;
  size_t xApplications = 0;
  moduleOp->walk([&](Operation* operation) {
    remainingLoops += isa<scf::ForOp>(operation);
    xApplications += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(remainingLoops, 0);
  EXPECT_EQ(xApplications, 1);
}

TEST(OpenQASMTargetTest, ExpandsAnOperandlessBarrierToAllDeclaredQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[3] q;
barrier;
bit[3] result = measure q;
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t barriers = 0;
  moduleOp->walk([&](qc::BarrierOp barrier) {
    ++barriers;
    EXPECT_EQ(barrier.getNumQubits(), 3);
  });
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, CanonicalizesVariadicCompatibilityGates) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[10] q;
mcx q[0], q[1], q[2], q[3];
mcphase(0.5) q[0], q[1], q[2];
mcx_vchain q[0], q[1], q[2], q[3], q[4], q[8], q[9];
mcx_recursive q[0], q[1], q[2], q[3], q[4], q[9];
)qasm";
  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  size_t controls = 0;
  size_t xGates = 0;
  size_t phaseGates = 0;
  moduleOp->walk([&](Operation* operation) {
    controls += isa<qc::CtrlOp>(operation);
    xGates += isa<qc::XOp>(operation);
    phaseGates += isa<qc::POp>(operation);
  });
  EXPECT_EQ(controls, 4);
  EXPECT_EQ(xGates, 3);
  EXPECT_EQ(phaseGates, 1);
}

TEST(OpenQASMTargetTest, BroadcastsRegistersAlongsideScalarQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[3] controls;
qubit target;
cx controls, target;
bit[3] left = measure controls;
bit right = measure target;
)qasm";

  MLIRContext context;
  auto moduleOp = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(moduleOp);
  size_t controls = 0;
  moduleOp->walk([&](qc::CtrlOp) { ++controls; });
  EXPECT_EQ(controls, 3);
}

TEST(OpenQASMTargetTest, PreservesImportedWhileBehavior) {
  struct OperationCounts {
    size_t h;
    size_t x;
    size_t measurements;
    size_t controls;
  };
  struct ConditionalCounts {
    size_t semantic;
    size_t dispatch;
    size_t whileMeasurements;
  };
  struct Fixture {
    llvm::StringRef name;
    llvm::StringRef source;
    SmallVector<int64_t> tripCounts;
    size_t whileLoops;
    OperationCounts operations;
    ConditionalCounts conditionals;
  };
  const auto fixtures = std::to_array<Fixture>({
      {.name = "simple-while",
       .source = qasm::simpleWhileReset,
       .tripCounts = {},
       .whileLoops = 1,
       .operations = {.h = 2, .x = 0, .measurements = 3, .controls = 0},
       .conditionals = {.semantic = 0, .dispatch = 0, .whileMeasurements = 0}},
      {.name = "condition-while-and",
       .source = qasm::conditionWhileAnd,
       .tripCounts = {},
       .whileLoops = 1,
       .operations = {.h = 3, .x = 0, .measurements = 6, .controls = 0},
       .conditionals = {.semantic = 1, .dispatch = 0, .whileMeasurements = 0}},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    MLIRContext context;
    auto moduleOp = qc::translateQASM3ToQC(fixture.source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    PassManager canonicalizer(&context);
    canonicalizer.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

    SmallVector<scf::ForOp> forLoops;
    size_t whileLoops = 0;
    bool hasQubitSelect = false;
    moduleOp->walk([&](Operation* operation) {
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        forLoops.push_back(loop);
      }
      whileLoops += isa<scf::WhileOp>(operation);
      if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        hasQubitSelect |= isa<qc::QubitType>(select.getType());
      }
    });
    ASSERT_EQ(forLoops.size(), fixture.tripCounts.size());
    EXPECT_EQ(whileLoops, fixture.whileLoops);
    EXPECT_FALSE(hasQubitSelect);
    for (const auto [loop, expectedCount] :
         llvm::zip_equal(forLoops, fixture.tripCounts)) {
      APInt lower;
      APInt upper;
      APInt step;
      ASSERT_TRUE(matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)));
      ASSERT_TRUE(matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)));
      ASSERT_TRUE(matchPattern(loop.getStep(), m_ConstantInt(&step)));
      EXPECT_EQ(lower.getSExtValue(), 0);
      EXPECT_EQ(upper.getSExtValue(), expectedCount);
      EXPECT_EQ(step.getSExtValue(), 1);
    }

    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    size_t hGates = 0;
    size_t xGates = 0;
    size_t measurements = 0;
    size_t controls = 0;
    size_t semanticConditionals = 0;
    size_t dispatchConditionals = 0;
    SmallVector<scf::ForOp> loweredForLoops;
    SmallVector<scf::WhileOp> loweredWhileLoops;
    moduleOp->walk([&](Operation* operation) {
      hGates += isa<qc::HOp>(operation);
      xGates += isa<qc::XOp>(operation);
      measurements += isa<qc::MeasureOp>(operation);
      controls += isa<qc::CtrlOp>(operation);
      if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
        size_t controlledXGates = 0;
        control->walk([&](qc::XOp) { ++controlledXGates; });
        EXPECT_EQ(controlledXGates, 1)
            << "each imported controlled-X must retain its controlled body";
      }
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        loweredForLoops.push_back(loop);
      }
      if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        loweredWhileLoops.push_back(loop);
      }
      auto conditional = dyn_cast<scf::IfOp>(operation);
      if (!conditional) {
        return;
      }
      auto comparison =
          conditional.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!comparison ||
          comparison.getPredicate() != arith::CmpIPredicate::eq) {
        ++semanticConditionals;
        return;
      }

      APInt candidate;
      const bool lhsCandidate =
          matchPattern(comparison.getLhs(), m_ConstantInt(&candidate));
      const bool rhsCandidate =
          matchPattern(comparison.getRhs(), m_ConstantInt(&candidate));
      if (lhsCandidate == rhsCandidate) {
        ++semanticConditionals;
        return;
      }
      ++dispatchConditionals;
      size_t dispatchedQuantumOperations = 0;
      conditional->walk([&](Operation* nested) {
        dispatchedQuantumOperations +=
            isa<qc::HOp, qc::XOp, qc::MeasureOp, qc::CtrlOp>(nested);
      });
      EXPECT_GT(dispatchedQuantumOperations, 0)
          << "each dynamic-index dispatch must retain quantum behavior";
    });
    EXPECT_EQ(hGates, fixture.operations.h);
    EXPECT_EQ(xGates, fixture.operations.x);
    EXPECT_EQ(measurements, fixture.operations.measurements);
    EXPECT_EQ(controls, fixture.operations.controls);
    EXPECT_EQ(semanticConditionals, fixture.conditionals.semantic);
    EXPECT_EQ(dispatchConditionals, fixture.conditionals.dispatch);

    ASSERT_EQ(loweredForLoops.size(), fixture.tripCounts.size());
    for (auto loop : loweredForLoops) {
      size_t bodyGates = 0;
      loop.getRegion().walk([&](Operation* nested) {
        bodyGates += isa<qc::HOp, qc::XOp>(nested);
      });
      EXPECT_GT(bodyGates, 0)
          << "each imported for-loop body must retain its gate behavior";
    }

    ASSERT_EQ(loweredWhileLoops.size(), fixture.whileLoops);
    for (auto loop : loweredWhileLoops) {
      size_t conditionMeasurements = 0;
      size_t bodyGates = 0;
      loop.getBefore().walk([&](qc::MeasureOp) { ++conditionMeasurements; });
      loop.getAfter().walk([&](Operation* nested) {
        bodyGates += isa<qc::HOp, qc::XOp>(nested);
      });
      EXPECT_EQ(conditionMeasurements, fixture.conditionals.whileMeasurements);
      EXPECT_GT(bodyGates, 0)
          << "each imported while-loop body must retain its gate behavior";
    }
  }
}

} // namespace
