/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/OQ3/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "mlir/Target/OpenQASM/OpenQASM.h"

#include <gtest/gtest.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <memory>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral BROADCAST_PROGRAM = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q;
bit[2] c = measure q;
)qasm";

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
  constexpr llvm::StringLiteral V31 = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
x q;
)qasm";
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(BROADCAST_PROGRAM));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(V31));
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedOpenQASM3MinorVersions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.2;
qubit q;
x q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Unsupported OpenQASM"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedIntegerDeclarations) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
int[32] counter;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Integer declarations"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsTooFewVariadicControlOperands) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit q;
mcx q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("qubit operands"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsDuplicateGateQubits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit q;
cx q, q;
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("same qubit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, CompatibilityGatePolicyIsExplicit) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
qubit[2] q;
cu3(0.1, 0.2, 0.3) q[0], q[1];
)qasm";
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(SOURCE));

  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE, strict);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("No OpenQASM definition"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, StrictPolicyRequiresTheStandardLibraryInclude) {
  constexpr llvm::StringLiteral WITHOUT_INCLUDE = R"qasm(
OPENQASM 3.0;
qubit q;
x q;
)qasm";
  constexpr llvm::StringLiteral WITH_INCLUDE = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
)qasm";
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;

  EXPECT_FALSE(oq3::frontend::analyzeOpenQASM(WITHOUT_INCLUDE, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(WITH_INCLUDE, strict));
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(WITHOUT_INCLUDE));
}

TEST(OpenQASMFrontendTest, StrictPolicyAllowsUserDefinedGateNames) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
gate x q {
  U(0, 0, 0) q;
}
qubit q;
x q;
)qasm";
  oq3::frontend::FrontendOptions strict;
  strict.gatePolicy = oq3::frontend::GatePolicy::Strict;
  EXPECT_TRUE(oq3::frontend::analyzeOpenQASM(SOURCE, strict));
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

TEST(OpenQASMTargetTest, EmitsVerifiedOQ3BeforeTargetLowering) {
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(BROADCAST_PROGRAM, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t applications = 0;
  module->walk([&](oq3::ApplyGateOp) { ++applications; });
  EXPECT_EQ(applications, 2);

  PassManager manager(&context);
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(module->getOps<oq3::GateDeclOp>().empty());
  EXPECT_TRUE(module->getOps<oq3::GateOp>().empty());
  EXPECT_TRUE(module->getOps<oq3::ForOp>().empty());
  EXPECT_TRUE(module->getOps<oq3::ApplyGateOp>().empty());
}

TEST(OpenQASMTargetTest, ProductionTranslationUsesTheStagedPipeline) {
  MLIRContext context;
  auto module = qc::translateQASM3ToQC(BROADCAST_PROGRAM, &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));

  bool hasOQ3Operation = false;
  bool hasQuantumOperation = false;
  module->walk([&](Operation* operation) {
    hasOQ3Operation |= operation->getDialect() != nullptr &&
                       operation->getDialect()->getNamespace() == "oq3";
    hasQuantumOperation |= isa<qc::HOp>(operation);
  });
  EXPECT_FALSE(hasOQ3Operation);
  EXPECT_TRUE(hasQuantumOperation);
}

TEST(OpenQASMTargetTest, EmitsTypedMixedNumericGateExpressions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shifted(theta) q {
  rx(theta + 1) q;
}
qubit q;
shifted(0.5) q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t numericCasts = 0;
  module->walk([&](Operation* operation) {
    numericCasts += isa<arith::SIToFPOp, arith::UIToFPOp>(operation);
  });
  EXPECT_EQ(numericCasts, 1);
}

TEST(OpenQASMTargetTest, EmitsScalarMathFunctions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate shaped(theta) q {
  rx(sin(theta) + cos(theta) + tan(theta) + exp(theta) + ln(theta) +
     sqrt(theta)) q;
}
qubit q;
shaped(0.5) q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t functions = 0;
  module->walk([&](Operation* operation) {
    functions += isa<math::SinOp, math::CosOp, math::TanOp, math::ExpOp,
                     math::LogOp, math::SqrtOp>(operation);
  });
  EXPECT_EQ(functions, 6);
}

TEST(OpenQASMTargetTest, NestsAlternatingControlsAndFlipsPolarityOutside) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
ctrl(2) @ negctrl @ inv @ ctrl @ x q[0], q[1], q[2], q[3], q[4];
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);

  PassManager manager(&context);
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t controls = 0;
  std::size_t outerPolarityFlips = 0;
  module->walk([&](Operation* operation) {
    if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
      ++controls;
      EXPECT_EQ(control.getNumControls(), 1);
    }
    if (isa<qc::XOp>(operation) &&
        operation->getParentOfType<qc::CtrlOp>() == nullptr &&
        operation->getParentOfType<qc::InvOp>() == nullptr) {
      ++outerPolarityFlips;
    }
  });
  EXPECT_EQ(controls, 4);
  EXPECT_EQ(outerPolarityFlips, 2);
}

TEST(OpenQASMTargetTest, PreservesPowerUntilTargetLowering) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.0;
include "stdgates.inc";
gate powered(exponent) q {
  pow(exponent) @ x q;
}
qubit q;
powered(0.5) q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& value) {
    llvm::raw_string_ostream(diagnostic) << value;
    return success();
  });
  PassManager manager(&context);
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(*module)));
  EXPECT_NE(diagnostic.find("pow gate modifiers are preserved in OQ3"),
            std::string::npos);
}

TEST(OpenQASMTargetTest,
     LowersCustomGatesConditionalsAndQuantumRuntimeOperations) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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
output bit[2] out = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t customGates = 0;
  std::size_t conditionals = 0;
  module->walk([&](Operation* operation) {
    customGates += isa<oq3::GateOp>(operation);
    conditionals += operation->getName().getStringRef() == "scf.if";
  });
  EXPECT_EQ(customGates, 1);
  EXPECT_EQ(conditionals, 1);

  PassManager manager(&context);
  manager.addPass(oq3::createLowerOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t resets = 0;
  std::size_t barriers = 0;
  module->walk([&](Operation* operation) {
    const StringRef name = operation->getName().getStringRef();
    resets += name == "qc.reset";
    barriers += name == "qc.barrier";
  });
  EXPECT_EQ(resets, 1);
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, LowersOpenQASM2ControlledGateCompatibilityPrefixes) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
cccx q[0], q[1], q[2], q[3];
measure q -> c;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(SOURCE, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t controls = 0;
  module->walk(
      [&](Operation* operation) { controls += isa<qc::CtrlOp>(operation); });
  EXPECT_EQ(controls, 3);
}

TEST(OpenQASMTargetTest, LowersLanguageBuiltinsOnHardwareQubits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
gphase(pi / 2);
x $3;
)qasm";

  MLIRContext context;
  auto module = qc::translateQASM3ToQC(SOURCE, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t globalPhases = 0;
  std::size_t xGates = 0;
  module->walk([&](Operation* operation) {
    globalPhases += isa<qc::GPhaseOp>(operation);
    xGates += isa<qc::XOp>(operation);
  });
  EXPECT_EQ(globalPhases, 1);
  EXPECT_EQ(xGates, 1);
}

TEST(OpenQASMFrontendTest, RejectsUnmeasuredOutputsAndInvalidConditions) {
  constexpr llvm::StringLiteral UNMEASURED_OUTPUT = R"qasm(
OPENQASM 3.1;
qubit q;
output bit result;
)qasm";
  constexpr llvm::StringLiteral UNMEASURED_CONDITION = R"qasm(
OPENQASM 3.1;
qubit q;
bit c;
if (c) { x q; }
)qasm";

  auto unmeasuredOutput = oq3::frontend::analyzeOpenQASM(UNMEASURED_OUTPUT);
  ASSERT_FALSE(unmeasuredOutput);
  ASSERT_FALSE(unmeasuredOutput.diagnostics.empty());
  EXPECT_NE(
      unmeasuredOutput.diagnostics.front().message.find("not fully measured"),
      std::string::npos);

  auto unmeasuredCondition =
      oq3::frontend::analyzeOpenQASM(UNMEASURED_CONDITION);
  ASSERT_FALSE(unmeasuredCondition);
  ASSERT_FALSE(unmeasuredCondition.diagnostics.empty());
  EXPECT_NE(unmeasuredCondition.diagnostics.front().message.find(
                "has not been measured"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsInvalidGateControlAndBroadcastShapes) {
  constexpr llvm::StringLiteral ZERO_CONTROL = R"qasm(
OPENQASM 3.1;
qubit[2] q;
ctrl(0) @ x q[0], q[1];
)qasm";
  constexpr llvm::StringLiteral MIXED_BROADCAST = R"qasm(
OPENQASM 3.1;
qubit[2] q;
qubit r;
cx q, r;
)qasm";

  auto zeroControl = oq3::frontend::analyzeOpenQASM(ZERO_CONTROL);
  ASSERT_FALSE(zeroControl);
  ASSERT_FALSE(zeroControl.diagnostics.empty());
  EXPECT_NE(zeroControl.diagnostics.front().message.find("must be positive"),
            std::string::npos);

  auto mixedBroadcast = oq3::frontend::analyzeOpenQASM(MIXED_BROADCAST);
  ASSERT_FALSE(mixedBroadcast);
  ASSERT_FALSE(mixedBroadcast.diagnostics.empty());
  EXPECT_NE(mixedBroadcast.diagnostics.front().message.find("not a mixture"),
            std::string::npos);
}

} // namespace
