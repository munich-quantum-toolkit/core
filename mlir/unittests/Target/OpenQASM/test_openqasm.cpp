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

} // namespace
