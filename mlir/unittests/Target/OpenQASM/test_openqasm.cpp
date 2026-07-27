/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "qasm_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/VirtualFileSystem.h>
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
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral BROADCAST_PROGRAM = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q;
bit[2] c = measure q;
)qasm";

} // namespace

static std::optional<APInt> evaluateConstantInteger(const Value value) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) {
    return constant;
  }
  auto* operation = value.getDefiningOp();
  if (operation == nullptr || operation->getNumOperands() == 0) {
    return std::nullopt;
  }
  const auto operand = [&](const unsigned index) {
    return evaluateConstantInteger(operation->getOperand(index));
  };
  const auto width = cast<IntegerType>(value.getType()).getWidth();
  if (isa<arith::TruncIOp>(operation)) {
    const auto input = operand(0);
    return input ? std::optional(input->trunc(width)) : std::nullopt;
  }
  if (isa<arith::ExtUIOp>(operation)) {
    const auto input = operand(0);
    return input ? std::optional(input->zext(width)) : std::nullopt;
  }
  auto lhs = operand(0);
  if (!lhs) {
    return std::nullopt;
  }
  if (isa<math::CtPopOp>(operation)) {
    return APInt(width, lhs->popcount());
  }
  if (isa<arith::SelectOp>(operation)) {
    return evaluateConstantInteger(
        operation->getOperand(lhs->isZero() ? 2 : 1));
  }
  const auto rhs = operand(1);
  if (!rhs) {
    return std::nullopt;
  }
  if (isa<arith::AddIOp>(operation)) {
    return *lhs + *rhs;
  }
  if (isa<arith::RemSIOp>(operation)) {
    return lhs->srem(*rhs);
  }
  if (isa<arith::ShLIOp>(operation)) {
    return lhs->shl(rhs->getLimitedValue());
  }
  if (isa<arith::ShRUIOp>(operation)) {
    return lhs->lshr(rhs->getLimitedValue());
  }
  if (isa<arith::OrIOp>(operation)) {
    return *lhs | *rhs;
  }
  if (isa<LLVM::FshlOp, LLVM::FshrOp>(operation)) {
    const auto shift = evaluateConstantInteger(operation->getOperand(2));
    if (!shift) {
      return std::nullopt;
    }
    const auto amount = shift->urem(APInt(shift->getBitWidth(), width));
    const auto distance = amount.getLimitedValue();
    if (distance == 0) {
      return lhs;
    }
    if (isa<LLVM::FshlOp>(operation)) {
      return lhs->shl(distance) | rhs->lshr(width - distance);
    }
    return lhs->lshr(distance) | rhs->shl(width - distance);
  }
  return std::nullopt;
}

static std::vector<bool> canonicalizedBitOutputs(const StringRef source) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  if (!module) {
    ADD_FAILURE() << "translation failed";
    return {};
  }
  if (failed(verify(*module))) {
    ADD_FAILURE() << "translation produced an invalid module";
    return {};
  }
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  if (failed(canonicalizer.run(*module))) {
    ADD_FAILURE() << "canonicalization failed";
    return {};
  }

  func::ReturnOp result;
  module->walk([&](func::ReturnOp operation) { result = operation; });
  if (!result) {
    ADD_FAILURE() << "translated module has no return operation";
    return {};
  }
  std::vector<bool> outputs;
  outputs.reserve(result.getNumOperands());
  for (const auto operand : result.getOperands()) {
    const auto value = evaluateConstantInteger(operand);
    if (!value) {
      std::string description;
      llvm::raw_string_ostream stream(description);
      operand.print(stream);
      ADD_FAILURE() << "canonicalized output is not constant: " << description;
      return {};
    }
    outputs.push_back(!value->isZero());
  }
  return outputs;
}

static std::vector<bool> rotateBits(const std::array<bool, 5>& bits,
                                    const int64_t distance, const bool left) {
  constexpr int64_t width = 5;
  auto normalized = distance % width;
  if (normalized < 0) {
    normalized += width;
  }
  std::vector<bool> result(width);
  for (int64_t bit = 0; bit < width; ++bit) {
    const auto source =
        left ? (bit + width - normalized) % width : (bit + normalized) % width;
    result[bit] = bits[static_cast<size_t>(source)];
  }
  return result;
}

namespace {

TEST(OpenQASMFrontendTest, SemanticAnalysisIsIndependentOfMLIR) {
  auto parsed = oq3::frontend::parseOpenQASM(BROADCAST_PROGRAM);
  ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;

  auto analyzed = oq3::frontend::analyzeOpenQASM(*parsed.program);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->registers.size(), 2);
  EXPECT_EQ(analyzed.program->body.size(), 5);
  EXPECT_EQ(analyzed.program->outputs.size(), 1);
}

TEST(OpenQASMFrontendTest, TreatsOpenQASM30AsTheOpenQASM3Mode) {
  constexpr llvm::StringLiteral v31 = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
x q;
)qasm";
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(BROADCAST_PROGRAM));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(v31));
}

TEST(OpenQASMFrontendTest, PreservesExactAndOptionalVersionSemantics) {
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM("qubit q; x q;"));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM("OPENQASM 3; qubit q; x q;"));

  auto unsupported =
      oq3::frontend::analyzeOpenQASM("OPENQASM 3.10; qubit q; x q;");
  ASSERT_FALSE(unsupported);
  ASSERT_FALSE(unsupported.diagnostics.empty());
  EXPECT_NE(unsupported.diagnostics.front().message.find("3.10"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedOpenQASM3MinorVersions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.2;
qubit q;
x q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Unsupported OpenQASM"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedIntegerDeclarations) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int[32] counter;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Integer declarations"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsTooFewVariadicControlOperands) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
mcx q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("qubit operands"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsDuplicateGateQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
cx q, q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("same qubit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, CompatibilityGatePolicyIsExplicit) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
qubit[2] q;
cu3(0.1, 0.2, 0.3) q[0], q[1];
)qasm";
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(source));

  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  auto analyzed = oq3::frontend::analyzeOpenQASM(source, strict);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("No OpenQASM definition"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, StrictPolicyRequiresTheStandardLibraryInclude) {
  constexpr llvm::StringLiteral withoutInclude = R"qasm(
OPENQASM 3.0;
qubit q;
x q;
)qasm";
  constexpr llvm::StringLiteral withInclude = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
)qasm";
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;

  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(withoutInclude, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(withInclude, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(withoutInclude));
}

TEST(OpenQASMFrontendTest, PreservesStandardLibraryIdentity) {
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;

  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
phase(0.5) q[0];
u1(0.5) q[0];
CX q[0], q[1];
)qasm",
                                             strict));
  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
r(0.5, 0.25) q;
)qasm",
                                              strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "qelib1.inc";
qubit[2] q;
crz(0.5) q[0], q[1];
cu1(0.5) q[0], q[1];
)qasm",
                                             strict));
  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
cu1(0.5) q[0], q[1];
)qasm",
                                              strict));
  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "qelib1.inc";
qubit[2] q;
swap q[0], q[1];
)qasm",
                                              strict));
}

TEST(OpenQASMFrontendTest, AcceptsHybridOpenQASM2Libraries) {
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 2.0;
include "stdgates.inc";
qreg q[1];
sx q[0];
)qasm",
                                             strict));
}

TEST(OpenQASMFrontendTest, StrictPolicyAllowsUserDefinedGateNames) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
gate x q {
  U(0, 0, 0) q;
}
qubit q;
x q;
)qasm";
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(source, strict));
}

TEST(OpenQASMFrontendTest, RequiresGateDefinitionScopes) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gate unbraced q x q;
)qasm";
  auto parsed = oq3::frontend::parseOpenQASM(source);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("expected '{'"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, AcceptsTrailingGateIdentifierCommas) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate trailing(theta,) control, target, {
  ctrl @ rx(theta) control, target;
}
qubit[2] q;
trailing(0.5) q[0], q[1];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, PreservesSourceNamesInSemanticDiagnostics) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.0;\nqubit q;\nunknown q;\n", "fixture.qasm"),
      llvm::SMLoc());
  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_EQ(analyzed.diagnostics.front().location.filename, "fixture.qasm");
  EXPECT_EQ(analyzed.diagnostics.front().location.line, 3);
}

TEST(OpenQASMFrontendTest, LocatesVersionAndOutputDiagnosticsPrecisely) {
  llvm::SourceMgr versionSources;
  versionSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("OPENQASM 3.2;\nqubit q;\n",
                                           "unsupported-version.qasm"),
      llvm::SMLoc());
  auto version = oq3::frontend::analyzeOpenQASM(versionSources);
  ASSERT_FALSE(version);
  ASSERT_FALSE(version.diagnostics.empty());
  EXPECT_EQ(version.diagnostics.front().location.filename,
            "unsupported-version.qasm");
  EXPECT_EQ(version.diagnostics.front().location.line, 1);

  llvm::SourceMgr outputSources;
  outputSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\nqubit q;\noutput bit result;\n",
          "incomplete-output.qasm"),
      llvm::SMLoc());
  auto output = oq3::frontend::analyzeOpenQASM(outputSources);
  ASSERT_FALSE(output);
  ASSERT_FALSE(output.diagnostics.empty());
  EXPECT_EQ(output.diagnostics.front().location.filename,
            "incomplete-output.qasm");
  EXPECT_EQ(output.diagnostics.front().location.line, 3);
}

TEST(OpenQASMTargetTest, EmitsVerifiedQCDirectly) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t gates = 0;
  module->walk([&](qc::HOp) { ++gates; });
  EXPECT_EQ(gates, 2);
}

TEST(OpenQASMTargetTest, ProductionTranslationUsesTheStagedPipeline) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));

  bool hasQuantumOperation = false;
  module->walk([&](Operation* operation) {
    hasQuantumOperation |= isa<qc::HOp>(operation);
  });
  EXPECT_TRUE(hasQuantumOperation);
}

TEST(OpenQASMTargetTest, EmitsTypedMixedNumericGateExpressions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shifted(theta) q {
  rx(theta + 1) q;
}
qubit q;
shifted(0.5) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t numericCasts = 0;
  module->walk([&](Operation* operation) {
    numericCasts += isa<arith::SIToFPOp, arith::UIToFPOp>(operation);
  });
  EXPECT_EQ(numericCasts, 1);
}

TEST(OpenQASMTargetTest, EmitsScalarMathFunctions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shaped(theta) q {
  rx(sin(theta) + cos(theta) + tan(theta) + exp(theta) + log(theta) +
     sqrt(theta)) q;
}

qubit q;
shaped(0.5) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t functions = 0;
  module->walk([&](Operation* operation) {
    functions += isa<math::SinOp, math::CosOp, math::TanOp, math::ExpOp,
                     math::LogOp, math::SqrtOp>(operation);
  });
  EXPECT_EQ(functions, 6);
}

TEST(OpenQASMTargetTest, FoldsAndEmitsCeilingAndFloor) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gate rounded(theta) q {
  rx(ceiling(theta) + floor(theta)) q;
}
qubit q;
rx(ceiling(1.25) + floor(-1.25)) q;
rounded(0.5) q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const auto& constantApplication = std::get<oq3::frontend::GateApplication>(
      analyzed.program->statements[analyzed.program->body[1]].data);
  const auto& constant =
      analyzed.program->expressions.at(constantApplication.parameters.front());
  ASSERT_EQ(constant.kind, oq3::frontend::ExpressionKind::Constant);
  EXPECT_DOUBLE_EQ(std::get<double>(constant.constant), 0.0);

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t ceilings = 0;
  size_t floors = 0;
  module->walk([&](Operation* operation) {
    ceilings += isa<math::CeilOp>(operation);
    floors += isa<math::FloorOp>(operation);
  });
  EXPECT_EQ(ceilings, 1);
  EXPECT_EQ(floors, 1);
}

TEST(OpenQASMFrontendTest, AcceptsLogAndRejectsLnBuiltInSpelling) {
  auto log = oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; const float value = log(2.0);");
  ASSERT_TRUE(log) << log.diagnostics.front().message;

  auto nonstandardLn = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; const float value = ln(2.0);");
  ASSERT_FALSE(nonstandardLn);
  ASSERT_FALSE(nonstandardLn.diagnostics.empty());
  EXPECT_NE(
      nonstandardLn.diagnostics.front().message.find("unknown function 'ln'"),
      std::string::npos);
}

TEST(OpenQASMTargetTest, NestsAlternatingControlsAndFlipsPolarityOutside) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(2) @ negctrl @ inv @ ctrl @ x q[0], q[1], q[2], q[3], q[4];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);

  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<size_t> controlArities;
  size_t outerPolarityFlips = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t controlRegions = 0;
  size_t polarityFlips = 0;
  module->walk([&](qc::CtrlOp control) {
    ++controlRegions;
    EXPECT_EQ(control.getNumControls(), 3);
  });
  module->walk([&](qc::XOp operation) {
    polarityFlips += operation->getParentOfType<qc::CtrlOp>() == nullptr;
  });
  EXPECT_EQ(controlRegions, 1);
  EXPECT_EQ(polarityFlips, 6);
}

TEST(OpenQASMTargetTest, LowersDynamicPowerModifiersToQC) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate powered(exponent) q {
  pow(exponent) @ x q;
}
qubit q;
powered(0.5) q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<qc::PowOp> powers;
  module->walk([&](qc::PowOp op) { powers.push_back(op); });
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
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    SmallVector<qc::PowOp> powers;
    size_t exactnessAssertions = 0;
    module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t conditionals = 0;
  module->walk([&](Operation* operation) {
    conditionals += operation->getName().getStringRef() == "scf.if";
  });
  EXPECT_EQ(conditionals, 1);

  ASSERT_TRUE(succeeded(verify(*module)));

  size_t resets = 0;
  size_t barriers = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t xGates = 0;
  module->walk(
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t controls = 0;
  module->walk(
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t globalPhases = 0;
  size_t xGates = 0;
  module->walk([&](Operation* operation) {
    globalPhases += isa<qc::GPhaseOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(globalPhases, 1);
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMFrontendTest, RejectsUninitializedOutputsAndInvalidConditions) {
  constexpr llvm::StringLiteral unmeasuredOutput = R"qasm(
OPENQASM 3.1;
qubit q;
output bit result;
)qasm";
  constexpr llvm::StringLiteral unmeasuredCondition = R"qasm(
OPENQASM 3.1;
qubit q;
bit c;
if (c) { x q; }
)qasm";

  auto uninitializedOutput = oq3::frontend::analyzeOpenQASM(unmeasuredOutput);
  ASSERT_FALSE(uninitializedOutput);
  ASSERT_FALSE(uninitializedOutput.diagnostics.empty());
  EXPECT_NE(uninitializedOutput.diagnostics.front().message.find(
                "not fully initialized"),
            std::string::npos);

  auto uninitializedCondition =
      oq3::frontend::analyzeOpenQASM(unmeasuredCondition);
  ASSERT_FALSE(uninitializedCondition);
  ASSERT_FALSE(uninitializedCondition.diagnostics.empty());
  EXPECT_NE(uninitializedCondition.diagnostics.front().message.find(
                "has not been initialized"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsInvalidGateControlAndBroadcastShapes) {
  constexpr llvm::StringLiteral zeroControl = R"qasm(
OPENQASM 3.1;
qubit[2] q;
ctrl(0) @ x q[0], q[1];
)qasm";
  constexpr llvm::StringLiteral mismatchedBroadcast = R"qasm(
OPENQASM 3.1;
qubit[2] q;
qubit[3] r;
cx q, r;
)qasm";
  constexpr llvm::StringLiteral overflowingControlCount = R"qasm(
OPENQASM 3.1;
qubit q;
ctrl(9223372036854775807) @ ctrl(9223372036854775807) @ ctrl(2) @ x q;
)qasm";
  constexpr llvm::StringLiteral excessiveDynamicDispatch = R"qasm(
OPENQASM 3.1;
qubit[16] q;
int a = 0;
int b = 1;
int c = 2;
int d = 3;
mcx q[a], q[b], q[c], q[d];
)qasm";

  auto zeroControlResult = oq3::frontend::analyzeOpenQASM(zeroControl);
  ASSERT_FALSE(zeroControlResult);
  ASSERT_FALSE(zeroControlResult.diagnostics.empty());
  EXPECT_NE(
      zeroControlResult.diagnostics.front().message.find("must be positive"),
      std::string::npos);

  auto mismatchedBroadcastResult =
      oq3::frontend::analyzeOpenQASM(mismatchedBroadcast);
  ASSERT_FALSE(mismatchedBroadcastResult);
  ASSERT_FALSE(mismatchedBroadcastResult.diagnostics.empty());
  EXPECT_NE(
      mismatchedBroadcastResult.diagnostics.front().message.find("same width"),
      std::string::npos);

  auto overflowingControlCountResult =
      oq3::frontend::analyzeOpenQASM(overflowingControlCount);
  ASSERT_FALSE(overflowingControlCountResult);
  ASSERT_FALSE(overflowingControlCountResult.diagnostics.empty());
  EXPECT_NE(overflowingControlCountResult.diagnostics.front().message.find(
                "Invalid number of qubit operands"),
            std::string::npos);

  auto excessiveDispatch =
      oq3::frontend::analyzeOpenQASM(excessiveDynamicDispatch);
  EXPECT_TRUE(excessiveDispatch);
}

TEST(OpenQASMFrontendTest, TracksLexicalScopeAndEnclosingAssignments) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int value = 1;
if (true) {
  int value = 2;
  value += 3;
} else {
  value = 4;
}
value += 5;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);

  size_t outerAssignments = 0;
  size_t innerAssignments = 0;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* assignment =
            std::get_if<oq3::frontend::ScalarAssignmentStatement>(
                &statement.data)) {
      outerAssignments += static_cast<size_t>(assignment->scalar == 0);
      innerAssignments += static_cast<size_t>(assignment->scalar == 1);
    }
  }
  EXPECT_EQ(outerAssignments, 2);
  EXPECT_EQ(innerAssignments, 1);
}

TEST(OpenQASMTargetTest, RejectsExcessiveDynamicDispatch) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[317] q;
qubit[317] aux;
int i = 0;
  int j = 1;
cx q[i], aux[j];
)qasm";

  MLIRContext context;
  std::string diagnostic;
  Location location = UnknownLoc::get(&context);
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    location = value.getLocation();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  const auto fileLocation = dyn_cast<FileLineColLoc>(location);
  ASSERT_TRUE(fileLocation);
  EXPECT_EQ(fileLocation.getFilename(), "<input>");
  EXPECT_EQ(fileLocation.getLine(), 8);
  EXPECT_EQ(fileLocation.getColumn(), 1);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, RejectsExcessiveCustomGateExpansion) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate g0 q { x q; }\n";
  for (size_t level = 1; level <= 17; ++level) {
    source += "gate g" + std::to_string(level) + " q { g" +
              std::to_string(level - 1) + " q; g" + std::to_string(level - 1) +
              " q; }\n";
  }
  source += "qubit q;\ng17 q;\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, ComposesDispatchAndCustomGateExpansionBudgets) {
  std::string source = "OPENQASM 3.1;\n"
                       "include \"stdgates.inc\";\n"
                       "gate expanded a, b {\n";
  for (size_t operation = 0; operation < 25; ++operation) {
    source += operation % 2 == 0 ? "  x a;\n" : "  x b;\n";
  }
  source += "}\n"
            "qubit[64] q;\n"
            "qubit[64] aux;\n"
            "int i = 0;\n"
            "int j = 1;\n"
            "expanded q[i], aux[j];\n";

  MLIRContext context;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    diagnostic = value.str();
    return success();
  });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
  EXPECT_NE(diagnostic.find("projected emitted operation count"),
            std::string::npos);
}

TEST(OpenQASMTargetTest, RejectsWideConstructionBeforeEmittingOperations) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1;\nqubit[50001] q;\n",
      "OPENQASM 3.1;\nbit[40000] c;\nint i = 0;\n"
      "output bit value;\nvalue = false;\n"
      "c[i] = value;\n",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    MLIRContext context;
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
      diagnostic = value.str();
      return success();
    });
    auto module = qc::translateQASM3ToQC(source, &context);
    EXPECT_FALSE(module);
    EXPECT_NE(diagnostic.find("projected emitted operation count"),
              std::string::npos)
        << diagnostic;
  }
}

TEST(OpenQASMTargetTest, BudgetsScalarAndStructuredOperationConstruction) {
  std::string expressionSource =
      "OPENQASM 3.1;\nint operand = 1;\nint result = ";
  std::vector<std::string> expressions(16384, "operand");
  while (expressions.size() > 1) {
    std::vector<std::string> next;
    next.reserve(expressions.size() / 2);
    for (size_t expression = 0; expression < expressions.size();
         expression += 2) {
      next.push_back("(" + expressions[expression] + " + " +
                     expressions[expression + 1] + ")");
    }
    expressions = std::move(next);
  }
  expressionSource += expressions.front();
  expressionSource += ";\n";

  std::string controlFlowSource = "OPENQASM 3.1;\nint operand = 1;\n";
  constexpr size_t conditionals = 12000;
  for (size_t conditional = 0; conditional < conditionals; ++conditional) {
    controlFlowSource += "if (operand > 0) {}\n";
  }

  std::string phaseSource = "OPENQASM 3.1;\nqubit q;\n";
  // Setup costs eight operations and each OpenQASM 3 U application costs
  // three parameter constants plus four phase-aware lowering operations.
  constexpr size_t phaseGates = ((100000 - 8) / 7) + 1;
  static_assert(8 + ((phaseGates - 1) * 7) <= 100000);
  static_assert(8 + (phaseGates * 7) > 100000);
  for (size_t gate = 0; gate < phaseGates; ++gate) {
    phaseSource += "U(0.1, 0.2, 0.3) q;\n";
  }

  std::string powerSource = "OPENQASM 3.1;\nqubit q;\nint exponent = 1;\n";
  constexpr size_t powerGates = 6000;
  for (size_t gate = 0; gate < powerGates; ++gate) {
    powerSource += "pow(exponent) @ x q;\n";
  }

  for (const auto* source :
       {&expressionSource, &controlFlowSource, &phaseSource, &powerSource}) {
    MLIRContext context;
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
      diagnostic = value.str();
      return success();
    });
    auto module = qc::translateQASM3ToQC(*source, &context);
    EXPECT_FALSE(module);
    EXPECT_NE(diagnostic.find("projected emitted operation count"),
              std::string::npos)
        << diagnostic;
  }
}

TEST(OpenQASMTargetTest, BudgetsLinearBitVectorPackingWork) {
  constexpr size_t width = 12501;
  std::string source =
      "OPENQASM 3.1;\noutput bit[" + std::to_string(width) + "] value;\n";
  for (size_t bit = 0; bit < width; ++bit) {
    source += "value[" + std::to_string(bit) + "] = false;\n";
  }
  source += "int distance = 1;\nvalue = rotl(value, distance);\n";

  MLIRContext context;
  ScopedDiagnosticHandler diagnostics(&context,
                                      [](Diagnostic&) { return success(); });
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
}

TEST(OpenQASMTargetTest, LowersGateBodyLoopsAndBuiltinConstants) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate repeated(theta) q {
  for int i in [0:2] { rx(theta + pi + i) q; }
  while (false) { x q; }
}
qubit q;
repeated(0.5) q;
bit result = measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t forLoops = 0;
  size_t whileLoops = 0;
  module->walk([&](Operation* operation) {
    forLoops += isa<scf::ForOp>(operation);
    whileLoops += isa<scf::WhileOp>(operation);
  });
  EXPECT_EQ(forLoops, 1);
  EXPECT_EQ(whileLoops, 1);

  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMFrontendTest, RejectsMutableGlobalCapturesInGateBodies) {
  constexpr auto fixtures =
      std::to_array<std::pair<llvm::StringLiteral, llvm::StringLiteral>>({
          {"mutable-capture",
           "OPENQASM 3.1; float theta = 0.5; gate g q { rx(theta) q; }"},
          {"declaration", "OPENQASM 3.1; gate g q { int i = 0; }"},
          {"measurement", "OPENQASM 3.1; bit c; gate g q { measure q -> c; }"},
          {"reset", "OPENQASM 3.1; gate g q { reset q; }"},
          {"conditional", "OPENQASM 3.1; gate g q { if (true) { x q; } }"},
      });
  for (const auto& [name, source] : fixtures) {
    SCOPED_TRACE(name.str());
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
    auto analyzed = oq3::frontend::analyzeOpenQASM(*parsed.program);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
  }
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  qc::RXOp rotation;
  module->walk([&](qc::RXOp application) { rotation = application; });
  ASSERT_TRUE(rotation);
  FloatAttr angle;
  EXPECT_TRUE(matchPattern(rotation.getParameter(0), m_Constant(&angle)));
  EXPECT_DOUBLE_EQ(angle.getValueAsDouble(), std::numbers::pi / 2);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<Value> measured;
  SmallVector<Value> returned;
  module->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  module->walk([&](func::ReturnOp operation) {
    returned.assign(operation.getOperands().begin(),
                    operation.getOperands().end());
  });
  ASSERT_EQ(measured.size(), 2);
  ASSERT_EQ(returned.size(), 2);
  EXPECT_EQ(returned[0], measured[0]);
  EXPECT_EQ(returned[1], measured[1]);
}

TEST(OpenQASMFrontendTest, RetainsScalarAndWidthOneBitTypes) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit scalar = true;
bit[1] vector;
vector[0] = scalar;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->registers.size(), 2);
  EXPECT_EQ(analyzed.program->registers[0].width, 1);
  EXPECT_TRUE(analyzed.program->registers[0].isScalar);
  EXPECT_EQ(analyzed.program->registers[1].width, 1);
  EXPECT_FALSE(analyzed.program->registers[1].isScalar);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t populationCounts = 0;
  size_t funnelShifts = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  size_t leftShifts = 0;
  size_t rightShifts = 0;
  size_t populationCounts = 0;
  size_t unpackingTruncations = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool carriedWholeRegister = false;
  module->walk([&](scf::IfOp conditional) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<Value> measured;
  SmallVector<Value> returned;
  module->walk([&](qc::MeasureOp measurement) {
    measured.push_back(measurement.getResult());
  });
  module->walk([&](func::ReturnOp operation) {
    returned.assign(operation.getOperands().begin(),
                    operation.getOperands().end());
  });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t populationCounts = 0;
  size_t leftShifts = 0;
  size_t rightShifts = 0;
  module->walk([&](Operation* operation) {
    populationCounts += isa<math::CtPopOp>(operation);
    leftShifts += isa<LLVM::FshlOp>(operation);
    rightShifts += isa<LLVM::FshrOp>(operation);
  });
  EXPECT_EQ(populationCounts, 1);
  EXPECT_EQ(leftShifts, 1);
  EXPECT_EQ(rightShifts, 0);
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
      const auto* const opposite =
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

TEST(OpenQASMFrontendTest, RejectsInvalidBitVectorBuiltinUses) {
  const std::vector<llvm::StringLiteral> invalidSources{
      "OPENQASM 3.1; qubit q; uint n = popcount(q);",
      "OPENQASM 3.1; bit value = true; uint n = popcount(value);",
      "OPENQASM 3.1; bit value = true; value = rotl(value, 1);",
      "OPENQASM 3.1; bit value = true; value = rotr(value, -1);",
      R"qasm(OPENQASM 3.1;
bit[2] value;
value[0] = false;
value[1] = true;
uint distance = 1;
value = rotl(value, distance);
)qasm",
      "OPENQASM 3.1; bit[2] value; value = rotl(value, 1);",
      R"qasm(OPENQASM 3.1;
bit[2] source;
source[0] = false;
source[1] = true;
bit[3] target;
target = rotr(source, 1);
)qasm"};
  for (const auto source : invalidSources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    EXPECT_FALSE(analyzed) << source.str();
  }

  EXPECT_FALSE(oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; bit[2] value; value = rotl(value);"));
  EXPECT_FALSE(oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; bit[2] value; uint n = popcount(value, 1);"));
}

TEST(OpenQASMFrontendTest, InvalidatesPopcountIndexFactsOnBitMutation) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
bit[2] source;
source[0] = false;
source[1] = true;
bit[2] target;
target[popcount(source)] = true;
source[0] = true;
qubit q;
if (target[popcount(source)]) { x q; }
output bit out;
out = true;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("uninitialized bit"),
            std::string::npos);

  constexpr llvm::StringLiteral noMutation = R"qasm(
OPENQASM 3.1;
bit[2] source;
source[0] = false;
source[1] = true;
bit[2] target;
target[popcount(source)] = true;
qubit q;
if (target[popcount(source)]) { x q; }
output bit out;
out = true;
)qasm";
  auto preserved = oq3::frontend::analyzeOpenQASM(noMutation);
  EXPECT_TRUE(preserved) << preserved.diagnostics.front().message;
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t conditionals = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  // The register equality and the source-level branch each short-circuit
  // through their own structured conditional.
  EXPECT_EQ(conditionals, 2);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool foundResult = false;
  module->walk([&](arith::ConstantFloatOp constant) {
    foundResult |= constant.value().convertToDouble() == 0.0625;
  });
  EXPECT_TRUE(foundResult);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 2);
}

TEST(OpenQASMFrontendTest, RejectsBoolMeasurementTargetsInAllSourceModes) {
  constexpr auto sourcePrograms = std::to_array<llvm::StringLiteral>({
      "qubit q; bool measured = measure q;",
      "OPENQASM 2.0; qubit q; bool measured = measure q;",
      "OPENQASM 3.0; qubit q; bool measured = measure q;",
      "OPENQASM 3.1; qubit q; bool measured = measure q;",
      "OPENQASM 3.1; qubit q; bool measured; measured = measure q;",
  });
  for (const auto source : sourcePrograms) {
    SCOPED_TRACE(source.str());
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find(
                  "measurement results have type 'bit'"),
              std::string::npos);
  }

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\nqubit q;\nbool measured;\n"
          "measured = measure q;\n",
          "bool-measurement.qasm"),
      llvm::SMLoc());
  auto located = oq3::frontend::analyzeOpenQASM(sourceManager);
  ASSERT_FALSE(located);
  ASSERT_FALSE(located.diagnostics.empty());
  EXPECT_EQ(located.diagnostics.front().location.filename,
            "bool-measurement.qasm");
  EXPECT_EQ(located.diagnostics.front().location.line, 4);
  EXPECT_EQ(located.diagnostics.front().location.column, 1);
}

TEST(OpenQASMFrontendTest, InvalidatesDynamicBitFactsOnIndexChanges) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] c;
int i = 0;
c[i] = measure q[i];
i = 1;
if (c[i]) { x q[i]; }
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("uninitialized bit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsAConstantZeroRangeStep) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
for int i in [0:0:3] {}
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("must not be zero"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, OwnsAndAnalyzesProvidedIncludeBuffers) {
  oq3::frontend::ParseResult parsed;
  {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "custom.inc";
qubit q;
custom q;
bit result = measure q;
)qasm",
                                             "main.qasm"),
        llvm::SMLoc());
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
gate custom q { x q; }
)qasm",
                                             "custom.inc"),
        llvm::SMLoc());
    parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  }

  ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
  auto analyzed = oq3::frontend::analyzeOpenQASM(*parsed.program);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->gates.size(), 1);
  EXPECT_EQ(analyzed.program->gates.front().name, "custom");
  EXPECT_EQ(analyzed.program->gates.front().location.filename, "custom.inc");
}

TEST(OpenQASMFrontendTest, ExpandsNestedIncludesAtTheirSourceLocations) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "outer.inc";
int result = outer + nested;
)qasm",
                                           "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
int outer = 1;
include "nested.inc";
int after = nested;
)qasm",
                                           "outer.inc"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("int nested = 2;\n", "nested.inc"),
      llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 4);
  EXPECT_EQ(analyzed.program->scalars[0].name, "outer");
  EXPECT_EQ(analyzed.program->scalars[1].name, "nested");
  EXPECT_EQ(analyzed.program->scalars[2].name, "after");
  EXPECT_EQ(analyzed.program->scalars[3].name, "result");
}

TEST(OpenQASMFrontendTest, PreservesNestedIncludeStacksInDiagnostics) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\ninclude \"outer.inc\";\n", "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"nested.inc\";\n", "outer.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "int value = missing;\n", "nested.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_EQ(analyzed.diagnostics.size(), 1);
  const auto& location = analyzed.diagnostics.front().location;
  EXPECT_EQ(location.filename, "nested.inc");
  EXPECT_EQ(location.line, 1);
  ASSERT_EQ(location.includeStack.size(), 2);
  EXPECT_EQ(location.includeStack[0].filename, "outer.inc");
  EXPECT_EQ(location.includeStack[0].line, 1);
  EXPECT_EQ(location.includeStack[1].filename, "main.qasm");
  EXPECT_EQ(location.includeStack[1].line, 2);
}

TEST(OpenQASMFrontendTest, PreservesDistinctProvenanceForRepeatedIncludes) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1;\ninclude \"a.inc\";\ninclude \"b.inc\";\n",
          "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"shared.inc\";\n", "a.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "include \"shared.inc\";\n", "b.inc"),
                               llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "int duplicate = 1;\n", "shared.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_FALSE(analyzed);
  ASSERT_EQ(analyzed.diagnostics.size(), 1);
  const auto& location = analyzed.diagnostics.front().location;
  EXPECT_EQ(location.filename, "shared.inc");
  ASSERT_EQ(location.includeStack.size(), 2);
  EXPECT_EQ(location.includeStack[0].filename, "b.inc");
  EXPECT_EQ(location.includeStack[0].line, 1);
  EXPECT_EQ(location.includeStack[1].filename, "main.qasm");
  EXPECT_EQ(location.includeStack[1].line, 3);
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
  auto module = qc::translateQASM3ToQC(sourceMgr, &context);
  EXPECT_FALSE(module);
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

TEST(OpenQASMFrontendTest, RejectsRecursiveIncludesResolvedThroughSearchPaths) {
  auto fileSystem = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(fileSystem->addFile(
      "/includes/recursive.inc", 0,
      llvm::MemoryBuffer::getMemBuffer("include \"recursive.inc\";")));

  llvm::SourceMgr sourceMgr;
  sourceMgr.setVirtualFileSystem(fileSystem);
  sourceMgr.setIncludeDirs({"/includes"});
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"recursive.inc\";", "main.qasm"),
      llvm::SMLoc());

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("recursive include"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, LimitsIncludeNesting) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"depth-0.inc\";", "main.qasm"),
      llvm::SMLoc());
  for (size_t index = 0; index <= 64; ++index) {
    std::string source;
    if (index == 64) {
      source = "int leaf = 1;";
    } else {
      source = "include \"depth-" + std::to_string(index + 1) + ".inc\";";
    }
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(
            source, "depth-" + std::to_string(index) + ".inc"),
        llvm::SMLoc());
  }

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("include nesting"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, LimitsTextualIncludeExpansion) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(
          "OPENQASM 3.1; include \"level-0.inc\";", "main.qasm"),
      llvm::SMLoc());
  for (size_t index = 0; index < 18; ++index) {
    std::string source;
    if (index == 17) {
      source = "int leaf = 1;";
    } else {
      const auto next = "level-" + std::to_string(index + 1) + ".inc";
      source.append("include \"")
          .append(next)
          .append("\"; include \"")
          .append(next)
          .append("\";");
    }
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(
            source, "level-" + std::to_string(index) + ".inc"),
        llvm::SMLoc());
  }

  auto parsed = oq3::frontend::parseOpenQASM(sourceMgr);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("statement limit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, BoundsRegisterStorageBeforeAllocation) {
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM("OPENQASM 3.1; qubit[100000] q;"));

  for (const auto* const source : {
           "OPENQASM 3.1; qubit[100001] q;",
           "OPENQASM 3.1; qubit[18446744073709551615] q;",
           "OPENQASM 3.1; qubit[60000] q; bit[40001] c;",
       }) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source;
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("limit"),
              std::string::npos);
    EXPECT_EQ(analyzed.diagnostics.front().location.filename, "<input>");
    EXPECT_EQ(analyzed.diagnostics.front().location.line, 1);
  }
}

TEST(OpenQASMFrontendTest, BoundsExpressionAndBlockDepth) {
  std::string flatExpression = "OPENQASM 3.1; int value = 1";
  for (size_t index = 0; index < 256; ++index) {
    flatExpression += " + 1";
  }
  flatExpression += ";";
  auto flat = oq3::frontend::analyzeOpenQASM(flatExpression);
  ASSERT_FALSE(flat);
  ASSERT_FALSE(flat.diagnostics.empty());
  EXPECT_NE(flat.diagnostics.front().message.find("expression depth"),
            std::string::npos);

  std::string nestedExpression = "OPENQASM 3.1; int value = ";
  nestedExpression.append(257, '(');
  nestedExpression += "1";
  nestedExpression.append(257, ')');
  nestedExpression += ";";
  auto nested = oq3::frontend::parseOpenQASM(nestedExpression);
  ASSERT_FALSE(nested);
  ASSERT_FALSE(nested.diagnostics.empty());
  EXPECT_NE(nested.diagnostics.front().message.find("expression nesting"),
            std::string::npos);

  std::string nestedBlocks = "OPENQASM 3.1;";
  for (size_t depth = 0; depth < 65; ++depth) {
    nestedBlocks += "if (true) {";
  }
  nestedBlocks += "int value = 0;";
  nestedBlocks.append(65, '}');
  auto blocks = oq3::frontend::parseOpenQASM(nestedBlocks);
  ASSERT_FALSE(blocks);
  ASSERT_FALSE(blocks.diagnostics.empty());
  EXPECT_NE(blocks.diagnostics.front().message.find("block depth"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, BoundsModifiersAndGateDependencies) {
  std::string modifiers = "OPENQASM 3.1; qubit[66] q;";
  for (size_t depth = 0; depth < 65; ++depth) {
    modifiers += "ctrl @ ";
  }
  modifiers += "x q;";
  auto parsed = oq3::frontend::parseOpenQASM(modifiers);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("modifier depth"),
            std::string::npos);

  std::string gates = "OPENQASM 3.1; gate g0 q { U(0, 0, 0) q; }\n";
  for (size_t depth = 1; depth < 65; ++depth) {
    gates += "gate g" + std::to_string(depth) + " q { g" +
             std::to_string(depth - 1) + " q; }\n";
  }
  auto analyzed = oq3::frontend::analyzeOpenQASM(gates);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("dependency depth"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, EnforcesUnicodeIdentifierCategoriesAndUtf8) {
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; int θ = 1; int Ångström = θ;"));

  auto symbol = oq3::frontend::analyzeOpenQASM("OPENQASM 3.1; int 💥 = 1;");
  ASSERT_FALSE(symbol);
  ASSERT_FALSE(symbol.diagnostics.empty());

  std::string invalid = "OPENQASM 3.1; int ";
  invalid.push_back(static_cast<char>(0xC3));
  invalid += " = 1;";
  auto malformed = oq3::frontend::analyzeOpenQASM(invalid);
  ASSERT_FALSE(malformed);
  ASSERT_FALSE(malformed.diagnostics.empty());
}

TEST(OpenQASMFrontendTest, RejectsShadowingBuiltInConstants) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; int pi = 0;",
      "OPENQASM 3.1; if (true) { int tau = 0; }",
      "OPENQASM 3.1; gate g(euler) q { U(euler, 0, 0) q; }",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("already declared"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, PropagatesDynamicBitFactsThroughKnownControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] c;
int i = 1;
if (true) { c[i] = measure q[i]; }
if (c[i]) { x q[0]; }
output bit result;
result = measure q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, SelectsKnownBranchStateForWideRegisters) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
bit[99998] c;
int i = 99997;
int selected;
if (true) {
  c[i] = measure q;
  selected = 1;
} else {
  i = 0;
}
if (c[i] && selected == 1) {}
if (false) {
  i = 0;
} else {
  c[i] = measure q;
}
if (c[i]) {}
output bit result;
result = measure q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, DiagnosesUnselectedKnownBranches) {
  auto analyzed = oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
if (true) {
  int valid = 1;
} else {
  int invalid = missing;
}
)qasm");
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find(
                "unknown scalar identifier 'missing'"),
            std::string::npos);
  EXPECT_EQ(analyzed.diagnostics.front().location.line, 6);
}

TEST(OpenQASMFrontendTest, ResolvesIncludedNamesWithoutBasenameAliasing) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
include "a/defs.inc";
include "b/defs.inc";
counter += 1;
qubit q;
if (enabled) { x q; }
bit result = measure q;
)qasm",
                                           "main.qasm"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("int counter = 0;\n", "a/defs.inc"),
      llvm::SMLoc());
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   "bool enabled = true;\n", "b/defs.inc"),
                               llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);
  EXPECT_EQ(analyzed.program->scalars[0].name, "counter");
  EXPECT_EQ(analyzed.program->scalars[1].name, "enabled");
}

TEST(OpenQASMFrontendTest, ExpandsEveryTextualIncludeOccurrence) {
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(R"qasm(
OPENQASM 3.1;
qubit q;
include "operations.inc";
include "operations.inc";
bit result = measure q;
)qasm",
                                                                  "main.qasm"),
                             llvm::SMLoc());
  sources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("x q;\n", "operations.inc"),
      llvm::SMLoc());

  auto analyzed = oq3::frontend::analyzeOpenQASM(sources);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  size_t applications = 0;
  for (const auto& statement : analyzed.program->statements) {
    applications += static_cast<size_t>(
        std::holds_alternative<oq3::frontend::GateApplication>(statement.data));
  }
  EXPECT_EQ(applications, 2);
}

TEST(OpenQASMFrontendTest, RejectsRecursiveAndRepeatedStandardIncludes) {
  llvm::SourceMgr recursiveSources;
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("OPENQASM 3.1; include \"a.inc\";",
                                           "main.qasm"),
      llvm::SMLoc());
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("include \"b.inc\";", "a.inc"),
      llvm::SMLoc());
  recursiveSources.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy("include \"a.inc\";", "b.inc"),
      llvm::SMLoc());
  auto recursive = oq3::frontend::parseOpenQASM(recursiveSources);
  ASSERT_FALSE(recursive);
  ASSERT_FALSE(recursive.diagnostics.empty());
  EXPECT_NE(recursive.diagnostics.front().message.find("recursive"),
            std::string::npos);

  auto repeated = oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; include \"stdgates.inc\"; include "
      "\"stdgates.inc\";");
  ASSERT_FALSE(repeated);
  ASSERT_FALSE(repeated.diagnostics.empty());
  EXPECT_NE(repeated.diagnostics.front().message.find("more than once"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, ActivatesStandardGatesSequentially) {
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  auto beforeInclude = oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
qubit q;
x q;
include "stdgates.inc";
)qasm",
                                                      strict);
  ASSERT_FALSE(beforeInclude);

  auto afterInclude = oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
x q;
)qasm",
                                                     strict);
  ASSERT_TRUE(afterInclude) << afterInclude.diagnostics.front().message;

  auto collision = oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
gate x q { U(0, 0, 0) q; }
include "stdgates.inc";
)qasm");
  ASSERT_FALSE(collision);
  ASSERT_FALSE(collision.diagnostics.empty());
  EXPECT_NE(collision.diagnostics.front().message.find("already declared"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsIncludesInsideBlocks) {
  auto parsed = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; if (true) { include \"nested.inc\"; }");
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("only allowed globally"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, AcceptsBothIncludeStringQuoteStyles) {
  EXPECT_TRUE(
      oq3::frontend::parseOpenQASM("OPENQASM 3.1; include \"stdgates.inc\";"));
  EXPECT_TRUE(
      oq3::frontend::parseOpenQASM("OPENQASM 3.1; include 'stdgates.inc';"));
}

TEST(OpenQASMFrontendTest, RejectsInvalidIncludeStringsAtTheOffendingByte) {
  struct InvalidInclude {
    llvm::StringRef source;
    size_t line{};
    size_t column{};
  };
  constexpr auto includes = std::to_array<InvalidInclude>({
      {.source = "include \"\";", .line = 1, .column = 10},
      {.source = "include \"bad\tname.inc\";", .line = 1, .column = 13},
      {.source = "include \"bad\nname.inc\";", .line = 1, .column = 13},
      {.source = "include \"bad\rname.inc\";", .line = 1, .column = 13},
  });

  for (const auto& include : includes) {
    SCOPED_TRACE(include.source.str());
    llvm::SourceMgr sources;
    sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                   include.source, "invalid-include.qasm"),
                               llvm::SMLoc());
    auto parsed = oq3::frontend::parseOpenQASM(sources);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_EQ(parsed.diagnostics.front().location.filename,
              "invalid-include.qasm");
    EXPECT_EQ(parsed.diagnostics.front().location.line, include.line);
    EXPECT_EQ(parsed.diagnostics.front().location.column, include.column);
  }
}

TEST(OpenQASMFrontendTest, CollectsMultipleRecoverableSyntaxDiagnostics) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit ;
bit ;
)qasm";
  auto parsed = oq3::frontend::parseOpenQASM(source);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.diagnostics.size(), 2);
  EXPECT_EQ(parsed.diagnostics[0].location.line, 3);
  EXPECT_EQ(parsed.diagnostics[1].location.line, 4);
}

TEST(OpenQASMFrontendTest, RejectsMisplacedVersionsAndRecursiveGates) {
  constexpr llvm::StringLiteral misplacedVersion = R"qasm(
qubit q;
OPENQASM 3.1;
)qasm";
  constexpr llvm::StringLiteral recursiveGates = R"qasm(
OPENQASM 3.1;
gate first q { first q; }
qubit q;
first q;
bit result = measure q;
)qasm";

  auto misplaced = oq3::frontend::analyzeOpenQASM(misplacedVersion);
  ASSERT_FALSE(misplaced);
  ASSERT_FALSE(misplaced.diagnostics.empty());
  EXPECT_NE(misplaced.diagnostics.front().message.find("must be the first"),
            std::string::npos);

  auto recursive = oq3::frontend::analyzeOpenQASM(recursiveGates);
  ASSERT_FALSE(recursive);
  ASSERT_FALSE(recursive.diagnostics.empty());
  EXPECT_NE(recursive.diagnostics.front().message.find("recursive"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, DiagnosesMalformedLexicalAndGrammarFamilies) {
  struct InvalidSource {
    llvm::StringRef name;
    llvm::StringRef source;
  };
  const auto fixtures = std::to_array<InvalidSource>({
      {.name = "unterminated-comment", .source = "OPENQASM 3.1; /*"},
      {.name = "unterminated-string",
       .source = "OPENQASM 3.1; include \"missing.inc;"},
      {.name = "missing-include",
       .source = "OPENQASM 3.1; include \"missing.inc\";"},
      {.name = "invalid-hardware-qubit",
       .source = "OPENQASM 3.1; qubit q; x $;"},
      {.name = "integer-overflow",
       .source = "OPENQASM 3.1; int value = 999999999999999999999999999999;"},
      {.name = "float-overflow",
       .source = "OPENQASM 3.1; float value = 1e99999;"},
      {.name = "unsupported-angle", .source = "OPENQASM 3.1; angle theta;"},
      {.name = "unsupported-duration",
       .source = "OPENQASM 3.1; duration delay;"},
      {.name = "unsupported-opaque",
       .source = "OPENQASM 3.1; opaque custom q;"},
      {.name = "output-qubit", .source = "OPENQASM 3.1; output qubit q;"},
      {.name = "const-qubit", .source = "OPENQASM 3.1; const qubit q;"},
      {.name = "duplicate-version", .source = "OPENQASM 3.1; OPENQASM 3.1;"},
      {.name = "non-string-include",
       .source = "OPENQASM 3.1; include stdgates.inc;"},
      {.name = "gate-designator",
       .source = "OPENQASM 3.1; gate custom[2] q {}"},
      {.name = "missing-range-members",
       .source = "OPENQASM 3.1; for int i in [:] {}"},
      {.name = "missing-while-condition",
       .source = "OPENQASM 3.1; while () {}"},
      {.name = "const-without-initializer",
       .source = "OPENQASM 3.1; const int value;"},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto parsed = oq3::frontend::parseOpenQASM(fixture.source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_FALSE(parsed.diagnostics.front().message.empty());
  }
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedReservedWordsAsIdentifiers) {
  constexpr auto reservedWords = std::to_array<llvm::StringLiteral>({
      "defcalgrammar", "def",      "cal",        "defcal",   "extern",
      "box",           "let",      "break",      "continue", "end",
      "return",        "switch",   "case",       "default",  "pragma",
      "input",         "readonly", "mutable",    "complex",  "array",
      "void",          "stretch",  "durationof", "delay",    "im",
      "#dim",          "#pragma",
  });
  for (const auto keyword : reservedWords) {
    SCOPED_TRACE(keyword.str());
    const std::string source = "OPENQASM 3.1; int " + keyword.str() + " = 0;";
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_NE(parsed.diagnostics.front().message.find("reserved keyword"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, DiagnosesUnsupportedReservedFeatureSyntax) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; input int value;",
      "OPENQASM 3.1; const complex value = 0;",
      "OPENQASM 3.1; output array[int, 2] values;",
      "OPENQASM 3.1; for complex value in [0:1] {}",
      "OPENQASM 3.1; int value = durationof({});",
  });
  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_NE(parsed.diagnostics.front().message.find("reserved keyword"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, EnforcesNumericSeparatorPlacement) {
  constexpr auto invalidLiterals = std::to_array<llvm::StringLiteral>({
      "1e+_2",
      "1e-_2",
      "1_e2",
      "1._2",
      "1e_2",
      "0xA__B",
      "0b_1",
      "0o7_",
  });
  for (const auto literal : invalidLiterals) {
    SCOPED_TRACE(literal.str());
    const std::string source =
        "OPENQASM 3.1; float value = " + literal.str() + ";";
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
  }

  auto valid = oq3::frontend::parseOpenQASM(
      "OPENQASM 3.1; int hex = 0xA_B; float value = 1_2.3_4e+5_6;");
  ASSERT_TRUE(valid) << valid.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, SourceManagerOverloadsPreserveParseFailures) {
  llvm::SourceMgr sources;
  sources.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(
                                 "OPENQASM 3.1; qubit ;", "broken.qasm"),
                             llvm::SMLoc());

  auto parsed = oq3::frontend::parseOpenQASM(sources);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_EQ(parsed.diagnostics.front().location.filename, "broken.qasm");

  auto analyzed = oq3::frontend::analyzeOpenQASM(sources);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_EQ(analyzed.diagnostics.front().location.filename, "broken.qasm");
}

TEST(OpenQASMFrontendTest, FoldsTypedConstantExpressionFamilies) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const float circle = pi + tau + euler;
const bool logic = !false && true || false;
const bool bool_equal = true == true;
const bool bool_not_equal = true != false;
const bool equal = 1 == 1;
const bool not_equal = 1 != 2;
const bool less = -1 < 9223372036854775808;
const bool less_equal = 1 <= 1;
const bool greater = 2 > 1;
const bool greater_equal = 2 >= 2;
const float float_arithmetic =
    (1.5 + 2.5) - (3.0 * 0.5) + (4.0 / 2.0) + mod(5.0, 2.0) + pow(2.0, 3.0);
const int integer_arithmetic =
    (1 + 2) - (3 * 1) + (8 / 2) + (5 % 2) + pow(2, 3);
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(analyzed.program->body.empty());
}

TEST(OpenQASMFrontendTest, AppliesC99SignedUnsignedConstantPromotion) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
const uint wrapped_add = maximum + 1;
const uint one = 1;
const uint wrapped_negation = -one;
const bool mixed_order = -1 < maximum;
qubit q;
rx(wrapped_add) q;
if (mixed_order) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;

  bool sawWrappedParameter = false;
  bool sawFalseCondition = false;
  for (const auto statement : analyzed.program->body) {
    const auto& data = analyzed.program->statements[statement].data;
    if (const auto* application =
            std::get_if<oq3::frontend::GateApplication>(&data);
        application != nullptr && application->callee == "rx") {
      ASSERT_EQ(application->parameters.size(), 1);
      const auto& parameter =
          analyzed.program->expressions[application->parameters.front()];
      sawWrappedParameter = parameter.type == oq3::frontend::ScalarType::Uint &&
                            std::get<uint64_t>(parameter.constant) == 0;
    }
    if (const auto* conditional =
            std::get_if<oq3::frontend::IfStatement>(&data)) {
      const auto& condition =
          analyzed.program->conditions[conditional->condition];
      sawFalseCondition =
          condition.kind == oq3::frontend::ConditionKind::Literal &&
          !condition.literal;
    }
  }
  EXPECT_TRUE(sawWrappedParameter);
  EXPECT_TRUE(sawFalseCondition);
}

TEST(OpenQASMFrontendTest, PromotesReleasedConstInitializerSubset) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const bool bool_value = true;
const bool bool_copy = bool_value;
const int int_from_bool = bool_value;
const uint uint_from_bool = bool_value;
const float float_from_bool = bool_value;
const int int_value = 4;
const int int_copy = int_value;
const uint uint_from_int = int_value;
const float float_from_int = int_value;
const uint u = 4;
const uint uint_copy = u;
const int int_from_uint = u;
const uint largest_representable_uint = 9223372036854775807;
const int largest_representable_int = largest_representable_uint;
const float float_from_uint = u;
const float float_value = 4.0;
const float float_copy = float_value;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(analyzed.program->body.empty());
}

TEST(OpenQASMFrontendTest, RejectsInvalidConstInitializerPromotions) {
  struct InvalidPromotion {
    llvm::StringRef source;
    llvm::StringRef diagnostic;
  };
  constexpr auto promotions = std::to_array<InvalidPromotion>({
      {.source = "OPENQASM 3.1; const bool value = 1;",
       .diagnostic = "'int' cannot"},
      {.source = "OPENQASM 3.1; const bool value = 1.0;",
       .diagnostic = "'float' cannot"},
      {.source = "OPENQASM 3.1; const int value = 1.0;",
       .diagnostic = "'float' cannot"},
      {.source = "OPENQASM 3.1; const uint value = 1.0;",
       .diagnostic = "'float' cannot"},
      {.source = "OPENQASM 3.1; const uint value = -1;",
       .diagnostic = "'int' cannot"},
      {.source =
           "OPENQASM 3.1; const uint source = 9223372036854775808; const int "
           "value = source;",
       .diagnostic = "'uint' cannot"},
  });

  for (const auto& promotion : promotions) {
    SCOPED_TRACE(promotion.source.str());
    auto analyzed = oq3::frontend::analyzeOpenQASM(promotion.source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find(promotion.diagnostic),
              std::string::npos);
  }

  auto nonConstant = oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; int source = 1; const int value = source;");
  ASSERT_FALSE(nonConstant);
  ASSERT_FALSE(nonConstant.diagnostics.empty());
  EXPECT_NE(nonConstant.diagnostics.front().message.find(
                "requires a constant initializer"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, AppliesScalarAssignmentConversions) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
int mutable_int = 2.5;
bool mutable_bool = 3;
float mutable_float = true;
uint mutable_uint = -1;
mutable_int = mutable_float;
mutable_bool = mutable_int;
mutable_float = mutable_uint;
mutable_uint = mutable_bool;
qubit q;
rx(mutable_int + mutable_float) q;
if (mutable_bool) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, RejectsInvalidProgramsAcrossSemanticFamilies) {
  struct InvalidSource {
    llvm::StringRef name;
    llvm::StringRef source;
  };
  const auto fixtures = std::to_array<InvalidSource>({
      {.name = "duplicate-scalar",
       .source = "OPENQASM 3.1; int value; int value;"},
      {.name = "unknown-assignment", .source = "OPENQASM 3.1; value = 1;"},
      {.name = "const-assignment",
       .source = "OPENQASM 3.1; const int value = 1; value = 2;"},
      {.name = "negated-signed-minimum",
       .source =
           "OPENQASM 3.1; const int minimum = -9223372036854775808; const int "
           "value = -minimum;"},
      {.name = "negation-overflow",
       .source = "OPENQASM 3.1; const int value = -9223372036854775809;"},
      {.name = "constant-bool-arithmetic",
       .source = "OPENQASM 3.1; const bool value = true + false;"},
      {.name = "mixed-bool-comparison",
       .source = "OPENQASM 3.1; const bool value = 1 < true;"},
      {.name = "non-finite-constant-math",
       .source = "OPENQASM 3.1; const float value = sqrt(-1.0);"},
      {.name = "float-division-by-zero",
       .source = "OPENQASM 3.1; const float value = 1.0 / 0.0;"},
      {.name = "float-modulo-by-zero",
       .source = "OPENQASM 3.1; const float value = 1.0 % 0.0;"},
      {.name = "float-percent",
       .source = "OPENQASM 3.1; const float value = 5.0 % 2.0;"},
      {.name = "integer-division-by-zero",
       .source = "OPENQASM 3.1; const int value = 1 / 0;"},
      {.name = "integer-division-overflow",
       .source = "OPENQASM 3.1; const int value = -9223372036854775808 / -1;"},
      {.name = "integer-modulo-by-zero",
       .source = "OPENQASM 3.1; const int value = 1 % 0;"},
      {.name = "integer-modulo-overflow",
       .source = "OPENQASM 3.1; const int value = -9223372036854775808 % -1;"},
      {.name = "negative-integer-power",
       .source = "OPENQASM 3.1; const int value = 2 ** -1;"},
      {.name = "integer-add-overflow",
       .source = "OPENQASM 3.1; const int value = 9223372036854775807 + 1;"},
      {.name = "integer-subtract-overflow",
       .source = "OPENQASM 3.1; const int value = -9223372036854775808 - 1;"},
      {.name = "integer-multiply-overflow",
       .source = "OPENQASM 3.1; const int value = 9223372036854775807 * 2;"},
      {.name = "bool-ordering",
       .source =
           "OPENQASM 3.1; bool left = true; bool right = false; if (left < "
           "right) {}"},
      {.name = "zero-register", .source = "OPENQASM 3.1; qubit[0] q;"},
      {.name = "negative-register", .source = "OPENQASM 3.1; qubit[-1] q;"},
      {.name = "dynamic-register-size",
       .source = "OPENQASM 3.1; int size = 2; qubit[size] q;"},
      {.name = "float-register-size", .source = "OPENQASM 3.1; qubit[1.5] q;"},
      {.name = "out-of-bounds-qubit",
       .source = "OPENQASM 3.1; qubit[2] q; x q[2];"},
      {.name = "float-qubit-index",
       .source = "OPENQASM 3.1; qubit[2] q; x q[1.0];"},
      {.name = "measurement-width",
       .source = "OPENQASM 3.1; qubit[2] q; bit[3] c; c = measure q;"},
      {.name = "unknown-reset", .source = "OPENQASM 3.1; reset missing;"},
      {.name = "unknown-barrier", .source = "OPENQASM 3.1; barrier missing;"},
      {.name = "duplicate-gate-parameter",
       .source = "OPENQASM 3.1; gate custom(a, a) q { U(a, 0, 0) q; }"},
      {.name = "duplicate-gate-qubit",
       .source = "OPENQASM 3.1; gate custom q, q { cx q, q; }"},
      {.name = "duplicate-custom-gate",
       .source = "OPENQASM 3.1; gate custom q {} gate custom q {}"},
      {.name = "custom-gate-conflicts-with-catalog",
       .source = "OPENQASM 3.1; gate x q {}"},
      {.name = "wrong-gate-parameter-count",
       .source = "OPENQASM 3.1; qubit q; rx(1, 2) q;"},
      {.name = "wrong-gate-qubit-count",
       .source = "OPENQASM 3.1; qubit q; cx q;"},
      {.name = "negative-control-count",
       .source = "OPENQASM 3.1; qubit[2] q; ctrl(-1) @ x q[0], q[1];"},
      {.name = "dynamic-control-count",
       .source =
           "OPENQASM 3.1; int n = 1; qubit[2] q; ctrl(n) @ x q[0], q[1];"},
      {.name = "non-integer-range",
       .source = "OPENQASM 3.1; for int i in [0.0:1.0] {}"},
      {.name = "non-bool-condition",
       .source = "OPENQASM 3.1; int value = 1; if (value) {}"},
      {.name = "bool-compound-assignment",
       .source = "OPENQASM 3.1; bool value = true; value += false;"},
      {.name = "unsupported-bitwise-not",
       .source = "OPENQASM 3.1; int value = ~1;"},
      {.name = "unsupported-bitwise-and",
       .source = "OPENQASM 3.1; int value = 1 & 2;"},
      {.name = "unsupported-bitwise-or",
       .source = "OPENQASM 3.1; int value = 1 | 2;"},
      {.name = "unsupported-bitwise-xor",
       .source = "OPENQASM 3.1; int value = 1 ^ 2;"},
      {.name = "unsupported-shift-left",
       .source = "OPENQASM 3.1; int value = 1 << 2;"},
      {.name = "unsupported-shift-right",
       .source = "OPENQASM 3.1; int value = 2 >> 1;"},
      {.name = "uninitialized-scalar",
       .source = "OPENQASM 3.1; int x; int y = x + 1;"},
      {.name = "self-initialization", .source = "OPENQASM 3.1; int x = x + 1;"},
      {.name = "uninitialized-condition",
       .source = "OPENQASM 3.1; bool ready; if (ready) {}"},
      {.name = "partially-initialized-branch",
       .source =
           "OPENQASM 3.1; qubit q; bit choose = measure q; int x; if (choose) "
           "{ x = 1; } int y = x;"},
      {.name = "forward-gate-call",
       .source = "OPENQASM 3.1; qubit q; later q; gate later a { x a; }"},
      {.name = "forward-gate-in-definition",
       .source =
           "OPENQASM 3.1; gate first q { second q; } gate second q { x q; }"},
      {.name = "hardware-qubit-in-gate",
       .source = "OPENQASM 3.1; gate invalid q { x $0; }"},
      {.name = "negative-index-out-of-bounds",
       .source = "OPENQASM 3.1; qubit[2] q; x q[-3];"},
      {.name = "bool-gate-parameter",
       .source = "OPENQASM 3.1; bool value = true; qubit q; rx(value) q;"},
      {.name = "local-qubit",
       .source = "OPENQASM 3.1; bool value = true; if (value) { qubit q; }"},
      {.name = "local-output",
       .source =
           "OPENQASM 3.1; bool value = true; if (value) { output bit c; }"},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto parsed = oq3::frontend::parseOpenQASM(fixture.source);
    ASSERT_TRUE(parsed) << parsed.diagnostics.front().message;
    auto analyzed = oq3::frontend::analyzeOpenQASM(*parsed.program);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_FALSE(analyzed.diagnostics.front().message.empty());
  }
}

TEST(OpenQASMFrontendTest, TracksDefiniteStateAndBlockLocalBits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
int fromTrueBranch;
if (true) { fromTrueBranch = 1; }
int fromSelectedElse;
if (false) { fromSelectedElse = 1; } else { fromSelectedElse = 2; }
bit choose = measure q[0];
int fromBothBranches;
if (choose) { fromBothBranches = 1; } else { fromBothBranches = 2; }
int fromNonemptyLoop;
for int iteration in [0:0] { fromNonemptyLoop = iteration; }
int combined = fromTrueBranch + fromSelectedElse + fromBothBranches +
               fromNonemptyLoop;
if (true) {
  bit local = measure q[0];
  if (local) { x q[1]; }
}
bit branch;
if (true) { branch = measure q[0]; }
bit loop;
for int i in [0:0] { loop = measure q[i]; }
if (branch && loop && combined >= 4) { h q[1]; }
output bit[2] result;
result = measure q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->outputs.size(), 1);
  EXPECT_EQ(analyzed.program->registers[analyzed.program->outputs.front()].name,
            "result");
}

TEST(OpenQASMFrontendTest, RejectsSignedMinimumDivisionAndModuloOverflow) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; const int minimum = -9223372036854775808; "
      "const int value = minimum / -1;",
      "OPENQASM 3.1; const int minimum = -9223372036854775808; "
      "const int value = minimum % -1;",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("overflows"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, AcceptsAllScalarOperatorsAndComparisonPredicates) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
int signedValue = 7;
uint unsignedValue = 8;
float realValue = 0.5;
bool enabled = true;
signedValue = -signedValue + 2 * 3 - 1;
signedValue /= 2;
signedValue %= 3;
signedValue = signedValue ** 2;
unsignedValue = unsignedValue + 1;
unsignedValue /= 2;
unsignedValue %= 3;
unsignedValue = unsignedValue ** 2;
realValue = -realValue + 2.0 * 3.0 - 1.0;
realValue /= 2.0;
realValue = mod(realValue, 3.0);
realValue = realValue ** 2.0;
float functions = arccos(realValue) + arcsin(realValue) + arctan(realValue) +
                  sin(signedValue) + cos(unsignedValue) + tan(realValue) +
                  exp(realValue) + log(realValue) + sqrt(realValue);
enabled = signedValue != 0 && realValue >= 0.0;
if (signedValue == 0) { x q[0]; }
if (signedValue != 0) { x q[0]; }
if (signedValue < 0) { x q[0]; }
if (signedValue <= 0) { x q[0]; }
if (signedValue > 0) { x q[0]; }
if (signedValue >= 0) { x q[0]; }
if (unsignedValue == 0) { x q[1]; }
if (unsignedValue != 0) { x q[1]; }
if (unsignedValue < 1) { x q[1]; }
if (unsignedValue <= 1) { x q[1]; }
if (unsignedValue > 1) { x q[1]; }
if (unsignedValue >= 1) { x q[1]; }
if (signedValue < unsignedValue) { x q[1]; }
if (realValue == 0.0) { x q[2]; }
if (realValue != 0.0) { x q[2]; }
if (realValue < 0.0) { x q[2]; }
if (realValue <= 0.0) { x q[2]; }
if (realValue > 0.0) { x q[2]; }
if (realValue >= 0.0) { x q[2]; }
bit[3] scratch;
scratch[0] = measure q[0];
scratch[1] = measure q[1];
scratch[2] = measure q[2];
for uint i in [0:2] {
  scratch[i] = measure q[i];
  if (scratch[i] || !enabled) { h q[3]; }
}
rx(functions) q[3];
output bit[4] result;
result = measure q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
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

TEST(OpenQASMTargetTest, DispatchesDynamicQubitGatesWithStructuredControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
x q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t switches = 0;
  size_t conditionals = 0;
  module->walk([&](scf::IndexSwitchOp switchOp) {
    ++switches;
    EXPECT_EQ(switchOp.getNumCases(), 1);
    EXPECT_EQ(switchOp.getNumResults(), 0);
  });
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(switches, 1);
  EXPECT_EQ(conditionals, 0);
}

TEST(OpenQASMTargetTest,
     DispatchesDynamicQubitMeasurementsWithStructuredControlFlow) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit c;
int i = 0;
c = measure q[i];
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t switches = 0;
  size_t conditionals = 0;
  module->walk([&](scf::IndexSwitchOp switchOp) {
    ++switches;
    EXPECT_EQ(switchOp.getNumCases(), 1);
    ASSERT_EQ(switchOp.getNumResults(), 1);
    EXPECT_TRUE(switchOp.getResult(0).getType().isInteger(1));
  });
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(switches, 1);
  EXPECT_EQ(conditionals, 0);
}

TEST(OpenQASMTargetTest, HandlesWidthOneAndNestedDynamicQubitDispatch) {
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
  widthOneModule->walk([&](scf::IndexSwitchOp switchOp) {
    EXPECT_EQ(switchOp.getNumCases(), 0);
  });

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
  EXPECT_EQ(switches, 3);
  EXPECT_EQ(controls, 4);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t xGates = 0;
  module->walk([&](qc::XOp) { ++xGates; });
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMTargetTest, SupportsTargetlessMeasurements) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit q;
measure q;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  size_t loops = 0;
  size_t xGates = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t loops = 0;
  module->walk([&](scf::WhileOp) { ++loops; });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  EXPECT_FALSE(module);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t assertions = 0;
  module->walk([&](cf::AssertOp) { ++assertions; });
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
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    EXPECT_TRUE(succeeded(verify(*module)));
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, LowersMultiIterationInductionIndicesAtQCTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
for uint i in [0:2] { int x = i + 1; h q[x]; }
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
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
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    size_t assertions = 0;
    module->walk([&](cf::AssertOp) { ++assertions; });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t assertions = 0;
  size_t powerLoops = 0;
  module->walk([&](cf::AssertOp) { ++assertions; });
  module->walk([&](scf::WhileOp) { ++powerLoops; });
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
    auto module = qc::translateQASM3ToQC(source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    size_t forLoops = 0;
    size_t whileLoops = 0;
    size_t divisions = 0;
    module->walk([&](scf::ForOp) { ++forLoops; });
    module->walk([&](scf::WhileOp) { ++whileLoops; });
    module->walk([&](arith::DivUIOp) { ++divisions; });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t whileLoops = 0;
  size_t divisions = 0;
  size_t assertions = 0;
  module->walk([&](scf::WhileOp) { ++whileLoops; });
  module->walk([&](arith::DivUIOp) { ++divisions; });
  module->walk([&](cf::AssertOp) { ++assertions; });
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t conditionals = 0;
  size_t indexSwitches = 0;
  size_t xGates = 0;
  size_t powers = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  module->walk([&](scf::IndexSwitchOp) { ++indexSwitches; });
  module->walk([&](qc::XOp) { ++xGates; });
  module->walk([&](qc::PowOp) { ++powers; });
  EXPECT_EQ(conditionals, 0);
  EXPECT_EQ(indexSwitches, 1);
  EXPECT_EQ(xGates, 2);
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<int64_t> firstMeasuredIndices;
  bool sawUnorderedInequality = false;
  size_t shortCircuitOperations = 0;
  size_t eagerLogicalOperations = 0;
  module->walk([&](Operation* operation) {
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

TEST(OpenQASMFrontendTest, ConstantEvaluationShortCircuitsLogicalOperands) {
  auto analyzed = oq3::frontend::analyzeOpenQASM(R"qasm(
OPENQASM 3.1;
bool andValue = false && (1 / 0 == 0);
bool orValue = true || (1 / 0 == 0);
)qasm");
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);
  ASSERT_EQ(analyzed.program->conditions.size(), 2);
  EXPECT_EQ(analyzed.program->conditions[0].kind,
            oq3::frontend::ConditionKind::Literal);
  EXPECT_FALSE(analyzed.program->conditions[0].literal);
  EXPECT_EQ(analyzed.program->conditions[1].kind,
            oq3::frontend::ConditionKind::Literal);
  EXPECT_TRUE(analyzed.program->conditions[1].literal);

  auto invalid = oq3::frontend::analyzeOpenQASM(
      "OPENQASM 3.1; const bool value = false && (1 / 0);");
  ASSERT_FALSE(invalid);
  ASSERT_FALSE(invalid.diagnostics.empty());
  EXPECT_NE(invalid.diagnostics.front().message.find(
                "logical operators require bool operands"),
            std::string::npos);
  EXPECT_EQ(invalid.diagnostics.front().message.find("division by zero"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsMeasurementsInGeneralExpressions) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; if (measure q) {}",
      "OPENQASM 3.1; qubit q; bool value = measure q && true;",
      "OPENQASM 3.1; qubit q; output bit value = measure q;",
      "OPENQASM 3.1; qubit q; bit value; value += measure q;",
  });
  for (const auto source : sources) {
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed) << source.str();
    ASSERT_FALSE(parsed.diagnostics.empty());
  }
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::ForOp forLoop;
  scf::WhileOp whileLoop;
  module->walk([&](Operation* operation) {
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
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  forLoop = {};
  module->walk([&](scf::ForOp loop) { forLoop = loop; });
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

  EXPECT_TRUE(succeeded(verify(*module)));
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t resultBearingConditionals = 0;
  size_t resultBearingWhiles = 0;
  module->walk([&](Operation* operation) {
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      resultBearingConditionals += conditional.getNumResults() != 0;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      resultBearingWhiles += loop.getNumResults() != 0;
    }
  });
  EXPECT_EQ(resultBearingConditionals, 1);
  EXPECT_EQ(resultBearingWhiles, 1);
  module->walk([&](scf::IfOp conditional) {
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
  module->walk([&](scf::WhileOp loop) {
    EXPECT_EQ(loop.getInits().size(), 1);
    EXPECT_EQ(loop.getNumResults(), 1);
    EXPECT_EQ(loop.getBeforeBody()->getTerminator()->getNumOperands(), 2);
    EXPECT_EQ(loop.getAfterBody()->getTerminator()->getNumOperands(), 1);
  });
}

TEST(OpenQASMFrontendTest, SupportsRequiredLiteralFormsAndOperatorPrecedence) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const int binary = 0b1010;
const int octal = 0o12;
const int hexadecimal = 0xA;
const int separated = 1_0;
const float fraction = .5;
const float trailing = 1.;
const float separated_float = 1_0.5_0;
const bool precedence = 1 < 2 == true;
int powered = 2;
powered **= 3;
qubit q;
if (precedence && powered == binary && binary == octal && octal == hexadecimal &&
    hexadecimal == separated && fraction + trailing + separated_float > 0.0) {
  x q;
}
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMTargetTest, HandlesTheMaximumUnsignedSingletonRange) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
qubit q;
for uint i in [maximum:maximum] { if (i == maximum) { x q; } }
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  scf::ForOp loop;
  module->walk([&](scf::ForOp current) { loop = current; });
  ASSERT_TRUE(loop);
  APInt lower;
  APInt step;
  ASSERT_TRUE(matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)));
  ASSERT_TRUE(matchPattern(loop.getStep(), m_ConstantInt(&step)));
  EXPECT_EQ(lower.getSExtValue(), 0);
  EXPECT_EQ(step.getSExtValue(), 1);

  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  size_t remainingLoops = 0;
  size_t xApplications = 0;
  module->walk([&](Operation* operation) {
    remainingLoops += isa<scf::ForOp>(operation);
    xApplications += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(remainingLoops, 0);
  EXPECT_EQ(xApplications, 1);
}

TEST(OpenQASMFrontendTest, DiagnosesMixedPhysicalAndDeclaredQubits) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; x q; x $0;",
      "OPENQASM 3.1; x $0; qubit q; x q;",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("mixing physical"),
              std::string::npos);
  }
}

TEST(OpenQASMTargetTest, ExpandsAnOperandlessBarrierToAllDeclaredQubits) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
qubit[3] q;
barrier;
bit[3] result = measure q;
)qasm";
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t barriers = 0;
  module->walk([&](qc::BarrierOp barrier) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  size_t controls = 0;
  size_t xGates = 0;
  size_t phaseGates = 0;
  module->walk([&](Operation* operation) {
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
  auto module = qc::translateQASM3ToQC(source, &context);
  ASSERT_TRUE(module);
  size_t controls = 0;
  module->walk([&](qc::CtrlOp) { ++controls; });
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
       .conditionals = {.semantic = 0, .dispatch = 0, .whileMeasurements = 0}},
  });

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    MLIRContext context;
    auto module = qc::translateQASM3ToQC(fixture.source, &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    PassManager canonicalizer(&context);
    canonicalizer.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(canonicalizer.run(*module)));

    SmallVector<scf::ForOp> forLoops;
    size_t whileLoops = 0;
    bool hasQubitSelect = false;
    module->walk([&](Operation* operation) {
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

    ASSERT_TRUE(succeeded(verify(*module)));
    size_t hGates = 0;
    size_t xGates = 0;
    size_t measurements = 0;
    size_t controls = 0;
    size_t semanticConditionals = 0;
    size_t dispatchConditionals = 0;
    SmallVector<scf::ForOp> loweredForLoops;
    SmallVector<scf::WhileOp> loweredWhileLoops;
    module->walk([&](Operation* operation) {
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
