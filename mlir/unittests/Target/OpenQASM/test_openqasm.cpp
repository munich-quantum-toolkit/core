/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/OQ3ToQC/OQ3ToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/OQ3/IR/OQ3Ops.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "mlir/Target/OpenQASM/OpenQASM.h"
#include "qasm_programs.h"

#include <gtest/gtest.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <numbers>
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

TEST(OpenQASMFrontendTest, RequiresGateDefinitionScopes) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
gate unbraced q x q;
)qasm";
  auto parsed = oq3::frontend::parseOpenQASM(SOURCE);
  ASSERT_FALSE(parsed);
  ASSERT_FALSE(parsed.diagnostics.empty());
  EXPECT_NE(parsed.diagnostics.front().message.find("expected '{'"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, AcceptsTrailingGateIdentifierCommas) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate trailing(theta,) control, target, {
  ctrl @ rx(theta) control, target;
}
qubit[2] q;
trailing(0.5) q[0], q[1];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
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

TEST(OpenQASMTargetTest, EmitsVerifiedOQ3BeforeTargetLowering) {
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(BROADCAST_PROGRAM, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t applications = 0;
  module->walk([&](oq3::ApplyGateOp) { ++applications; });
  EXPECT_EQ(applications, 2);

  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
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
  manager.addPass(oq3::createOQ3ToQCPass());
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
  manager.addPass(oq3::createOQ3ToQCPass());
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
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t resets = 0;
  std::size_t barriers = 0;
  module->walk([&](Operation* operation) {
    auto name = operation->getName().getStringRef();
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

TEST(OpenQASMFrontendTest, RejectsUninitializedOutputsAndInvalidConditions) {
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

  auto uninitializedOutput = oq3::frontend::analyzeOpenQASM(UNMEASURED_OUTPUT);
  ASSERT_FALSE(uninitializedOutput);
  ASSERT_FALSE(uninitializedOutput.diagnostics.empty());
  EXPECT_NE(uninitializedOutput.diagnostics.front().message.find(
                "not fully initialized"),
            std::string::npos);

  auto uninitializedCondition =
      oq3::frontend::analyzeOpenQASM(UNMEASURED_CONDITION);
  ASSERT_FALSE(uninitializedCondition);
  ASSERT_FALSE(uninitializedCondition.diagnostics.empty());
  EXPECT_NE(uninitializedCondition.diagnostics.front().message.find(
                "has not been initialized"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsInvalidGateControlAndBroadcastShapes) {
  constexpr llvm::StringLiteral ZERO_CONTROL = R"qasm(
OPENQASM 3.1;
qubit[2] q;
ctrl(0) @ x q[0], q[1];
)qasm";
  constexpr llvm::StringLiteral MISMATCHED_BROADCAST = R"qasm(
OPENQASM 3.1;
qubit[2] q;
qubit[3] r;
cx q, r;
)qasm";
  constexpr llvm::StringLiteral OVERFLOWING_CONTROL_COUNT = R"qasm(
OPENQASM 3.1;
qubit q;
ctrl(9223372036854775807) @ ctrl(9223372036854775807) @ ctrl(2) @ x q;
)qasm";
  constexpr llvm::StringLiteral EXCESSIVE_DYNAMIC_DISPATCH = R"qasm(
OPENQASM 3.1;
qubit[16] q;
int a = 0;
int b = 1;
int c = 2;
int d = 3;
mcx q[a], q[b], q[c], q[d];
)qasm";

  auto zeroControl = oq3::frontend::analyzeOpenQASM(ZERO_CONTROL);
  ASSERT_FALSE(zeroControl);
  ASSERT_FALSE(zeroControl.diagnostics.empty());
  EXPECT_NE(zeroControl.diagnostics.front().message.find("must be positive"),
            std::string::npos);

  auto mismatchedBroadcast =
      oq3::frontend::analyzeOpenQASM(MISMATCHED_BROADCAST);
  ASSERT_FALSE(mismatchedBroadcast);
  ASSERT_FALSE(mismatchedBroadcast.diagnostics.empty());
  EXPECT_NE(mismatchedBroadcast.diagnostics.front().message.find("same width"),
            std::string::npos);

  auto overflowingControlCount =
      oq3::frontend::analyzeOpenQASM(OVERFLOWING_CONTROL_COUNT);
  ASSERT_FALSE(overflowingControlCount);
  ASSERT_FALSE(overflowingControlCount.diagnostics.empty());
  EXPECT_NE(overflowingControlCount.diagnostics.front().message.find(
                "Invalid number of qubit operands"),
            std::string::npos);

  auto excessiveDispatch =
      oq3::frontend::analyzeOpenQASM(EXCESSIVE_DYNAMIC_DISPATCH);
  ASSERT_FALSE(excessiveDispatch);
  ASSERT_FALSE(excessiveDispatch.diagnostics.empty());
  EXPECT_NE(excessiveDispatch.diagnostics.front().message.find(
                "structured-dispatch expansion budget"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, TracksLexicalScopeAndEnclosingAssignments) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->scalars.size(), 2);

  std::size_t outerAssignments = 0;
  std::size_t innerAssignments = 0;
  for (const auto& statement : analyzed.program->statements) {
    if (const auto* assignment =
            std::get_if<oq3::frontend::ScalarAssignmentStatement>(
                &statement.data)) {
      outerAssignments += assignment->scalar == 0;
      innerAssignments += assignment->scalar == 1;
    }
  }
  EXPECT_EQ(outerAssignments, 2);
  EXPECT_EQ(innerAssignments, 1);
}

TEST(OpenQASMTargetTest, RevalidatesDynamicDispatchBudgetForTypedPrograms) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
qubit[2] aux;
int i = 0;
int j = 1;
cx q[i], aux[j];
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  for (auto& declaration : analyzed.program->registers) {
    if (declaration.kind == oq3::frontend::RegisterKind::Qubit) {
      declaration.width = 65;
    }
  }

  MLIRContext context;
  EXPECT_FALSE(oq3::emitOQ3(*analyzed.program, context));
}

TEST(OpenQASMTargetTest, LowersGateBodyLoopsAndBuiltinConstants) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t forLoops = 0;
  std::size_t whileLoops = 0;
  module->walk([&](Operation* operation) {
    forLoops += isa<scf::ForOp>(operation);
    whileLoops += isa<scf::WhileOp>(operation);
  });
  EXPECT_EQ(forLoops, 1);
  EXPECT_EQ(whileLoops, 1);

  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(module->getOps<oq3::GateOp>().empty());
}

TEST(OpenQASMFrontendTest, RejectsMutableGlobalCapturesInGateBodies) {
  constexpr std::pair<llvm::StringLiteral, llvm::StringLiteral> fixtures[] = {
      {"mutable-capture",
       "OPENQASM 3.1; float theta = 0.5; gate g q { rx(theta) q; }"},
      {"declaration", "OPENQASM 3.1; gate g q { int i = 0; }"},
      {"measurement", "OPENQASM 3.1; bit c; gate g q { measure q -> c; }"},
      {"reset", "OPENQASM 3.1; gate g q { reset q; }"},
      {"conditional", "OPENQASM 3.1; gate g q { if (true) { x q; } }"},
  };
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
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const float theta = pi / 2;
gate g q { rx(theta) q; }
qubit q;
g q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  oq3::ApplyGateOp rotation;
  module->walk([&](oq3::ApplyGateOp application) {
    if (application.getCallee() == "rx") {
      rotation = application;
    }
  });
  ASSERT_TRUE(rotation);
  ASSERT_EQ(rotation.getParameters().size(), 1);
  FloatAttr angle;
  EXPECT_TRUE(
      matchPattern(rotation.getParameters().front(), m_Constant(&angle)));
  EXPECT_DOUBLE_EQ(angle.getValueAsDouble(), std::numbers::pi / 2);
}

TEST(OpenQASMTargetTest, SupportsWholeBitRegisterAssignment) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[2] source = measure q;
bit[2] target = measure q;
target = source;
if (target[0] || target[1]) { x q[0]; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  std::size_t assignments = 0;
  for (const auto& statement : analyzed.program->statements) {
    assignments +=
        std::holds_alternative<oq3::frontend::BitAssignmentStatement>(
            statement.data);
  }
  EXPECT_EQ(assignments, 2);

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, SupportsOpenQASM2RegisterConditions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q -> c;
if (c == 1) x q[0];
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t conditionals = 0;
  module->walk([&](scf::IfOp) { ++conditionals; });
  EXPECT_EQ(conditionals, 1);
}

TEST(OpenQASMTargetTest, SelectsFloatingPowForNegativeSignedExponent) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
int base = 4;
float result = pow(base, -2);
qubit q;
if (result == 0.0625) { x q; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool foundResult = false;
  module->walk([&](arith::ConstantFloatOp constant) {
    foundResult |= constant.value().convertToDouble() == 0.0625;
  });
  EXPECT_TRUE(foundResult);
}

TEST(OpenQASMTargetTest, SupportsScalarMeasurementReassignment) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit q;
bool measured;
measured = measure q;
if (measured) { x q; }
measured = measure q;
if (!measured) { h q; }
)qasm";
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 2);
}

TEST(OpenQASMFrontendTest, InvalidatesDynamicBitFactsOnIndexChanges) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] c;
int i = 0;
c[i] = measure q[i];
i = 1;
if (c[i]) { x q[i]; }
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("uninitialized bit"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsAConstantZeroRangeStep) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
for int i in [0:0:3] {}
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
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
  for (std::size_t index = 0; index <= 64; ++index) {
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
  for (std::size_t index = 0; index < 18; ++index) {
    std::string source;
    if (index == 17) {
      source = "int leaf = 1;";
    } else {
      const auto next = "level-" + std::to_string(index + 1) + ".inc";
      source = "include \"" + next + "\"; include \"" + next + "\";";
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
  constexpr llvm::StringLiteral SOURCES[] = {
      "OPENQASM 3.1; int pi = 0;",
      "OPENQASM 3.1; if (true) { int tau = 0; }",
      "OPENQASM 3.1; gate g(euler) q { U(euler, 0, 0) q; }",
  };
  for (const auto source : SOURCES) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("already declared"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, PropagatesDynamicBitFactsThroughKnownControlFlow) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit[2] c;
int i = 1;
if (true) { c[i] = measure q[i]; }
if (c[i]) { x q[0]; }
output bit result = measure q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
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
  std::size_t applications = 0;
  for (const auto& statement : analyzed.program->statements) {
    applications +=
        std::holds_alternative<oq3::frontend::GateApplication>(statement.data);
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
gate r(theta, phi) q { U(theta, phi, 0) q; }
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

TEST(OpenQASMFrontendTest, CollectsMultipleRecoverableSyntaxDiagnostics) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit ;
bit ;
)qasm";
  auto parsed = oq3::frontend::parseOpenQASM(SOURCE);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.diagnostics.size(), 2);
  EXPECT_EQ(parsed.diagnostics[0].location.line, 3);
  EXPECT_EQ(parsed.diagnostics[1].location.line, 4);
}

TEST(OpenQASMFrontendTest, RejectsMisplacedVersionsAndRecursiveGates) {
  constexpr llvm::StringLiteral MISPLACED_VERSION = R"qasm(
qubit q;
OPENQASM 3.1;
)qasm";
  constexpr llvm::StringLiteral RECURSIVE_GATES = R"qasm(
OPENQASM 3.1;
gate first q { first q; }
qubit q;
first q;
bit result = measure q;
)qasm";

  auto misplaced = oq3::frontend::analyzeOpenQASM(MISPLACED_VERSION);
  ASSERT_FALSE(misplaced);
  ASSERT_FALSE(misplaced.diagnostics.empty());
  EXPECT_NE(misplaced.diagnostics.front().message.find("must be the first"),
            std::string::npos);

  auto recursive = oq3::frontend::analyzeOpenQASM(RECURSIVE_GATES);
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
  const InvalidSource fixtures[] = {
      {"unterminated-comment", "OPENQASM 3.1; /*"},
      {"unterminated-string", "OPENQASM 3.1; include \"missing.inc;"},
      {"missing-include", "OPENQASM 3.1; include \"missing.inc\";"},
      {"invalid-hardware-qubit", "OPENQASM 3.1; qubit q; x $;"},
      {"integer-overflow",
       "OPENQASM 3.1; int value = 999999999999999999999999999999;"},
      {"float-overflow", "OPENQASM 3.1; float value = 1e99999;"},
      {"unsupported-angle", "OPENQASM 3.1; angle theta;"},
      {"unsupported-duration", "OPENQASM 3.1; duration delay;"},
      {"unsupported-opaque", "OPENQASM 3.1; opaque custom q;"},
      {"output-qubit", "OPENQASM 3.1; output qubit q;"},
      {"const-qubit", "OPENQASM 3.1; const qubit q;"},
      {"duplicate-version", "OPENQASM 3.1; OPENQASM 3.1;"},
      {"non-string-include", "OPENQASM 3.1; include stdgates.inc;"},
      {"gate-designator", "OPENQASM 3.1; gate custom[2] q {}"},
      {"missing-range-members", "OPENQASM 3.1; for int i in [:] {}"},
      {"missing-while-condition", "OPENQASM 3.1; while () {}"},
      {"const-without-initializer", "OPENQASM 3.1; const int value;"},
  };

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    auto parsed = oq3::frontend::parseOpenQASM(fixture.source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_FALSE(parsed.diagnostics.front().message.empty());
  }
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedReservedWordsAsIdentifiers) {
  constexpr llvm::StringLiteral RESERVED_WORDS[] = {
      "defcalgrammar", "def",      "cal",        "defcal",   "extern",
      "box",           "let",      "break",      "continue", "end",
      "return",        "switch",   "case",       "default",  "pragma",
      "input",         "readonly", "mutable",    "complex",  "array",
      "void",          "stretch",  "durationof", "delay",    "im",
      "#dim",          "#pragma",
  };
  for (const auto keyword : RESERVED_WORDS) {
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
  constexpr llvm::StringLiteral SOURCES[] = {
      "OPENQASM 3.1; input int value;",
      "OPENQASM 3.1; const complex value = 0;",
      "OPENQASM 3.1; output array[int, 2] values;",
      "OPENQASM 3.1; for complex value in [0:1] {}",
      "OPENQASM 3.1; int value = durationof({});",
  };
  for (const auto source : SOURCES) {
    SCOPED_TRACE(source.str());
    auto parsed = oq3::frontend::parseOpenQASM(source);
    ASSERT_FALSE(parsed);
    ASSERT_FALSE(parsed.diagnostics.empty());
    EXPECT_NE(parsed.diagnostics.front().message.find("reserved keyword"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, EnforcesNumericSeparatorPlacement) {
  constexpr llvm::StringLiteral INVALID_LITERALS[] = {
      "1e+_2", "1e-_2", "1_e2", "1._2", "1e_2", "0xA__B", "0b_1", "0o7_",
  };
  for (const auto literal : INVALID_LITERALS) {
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
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(analyzed.program->body.empty());
}

TEST(OpenQASMFrontendTest, AppliesC99SignedUnsignedConstantPromotion) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
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
                            std::get<std::uint64_t>(parameter.constant) == 0;
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

TEST(OpenQASMTargetTest, AppliesC99ScalarAssignmentConversions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
const int truncated = 1.75;
const bool truthy = -2;
const int from_bool = true;
const uint wrapped = -1;
const int signed_wrap = 9223372036854775808;
int mutable_int = 2.5;
bool mutable_bool = 3;
float mutable_float = true;
uint mutable_uint = -1;
mutable_int = mutable_float;
mutable_bool = mutable_int;
mutable_float = mutable_uint;
mutable_uint = mutable_bool;
qubit q;
rx(truncated + from_bool + wrapped + signed_wrap + mutable_int + mutable_float)
    q;
if (truthy && mutable_bool) { x q; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t conversions = 0;
  module->walk([&](Operation* operation) {
    conversions +=
        isa<arith::ExtUIOp, arith::FPToSIOp, arith::SIToFPOp, arith::UIToFPOp>(
            operation);
  });
  EXPECT_GE(conversions, 4);
}

TEST(OpenQASMFrontendTest, RejectsInvalidProgramsAcrossSemanticFamilies) {
  struct InvalidSource {
    llvm::StringRef name;
    llvm::StringRef source;
  };
  const InvalidSource fixtures[] = {
      {"duplicate-scalar", "OPENQASM 3.1; int value; int value;"},
      {"unknown-assignment", "OPENQASM 3.1; value = 1;"},
      {"const-assignment", "OPENQASM 3.1; const int value = 1; value = 2;"},
      {"negated-signed-minimum",
       "OPENQASM 3.1; const int minimum = -9223372036854775808; const int "
       "value = -minimum;"},
      {"negation-overflow",
       "OPENQASM 3.1; const int value = -9223372036854775809;"},
      {"constant-bool-arithmetic",
       "OPENQASM 3.1; const bool value = true + false;"},
      {"mixed-bool-comparison", "OPENQASM 3.1; const bool value = 1 < true;"},
      {"non-finite-constant-math",
       "OPENQASM 3.1; const float value = sqrt(-1.0);"},
      {"float-division-by-zero",
       "OPENQASM 3.1; const float value = 1.0 / 0.0;"},
      {"float-modulo-by-zero", "OPENQASM 3.1; const float value = 1.0 % 0.0;"},
      {"float-percent", "OPENQASM 3.1; const float value = 5.0 % 2.0;"},
      {"integer-division-by-zero", "OPENQASM 3.1; const int value = 1 / 0;"},
      {"integer-division-overflow",
       "OPENQASM 3.1; const int value = -9223372036854775808 / -1;"},
      {"integer-modulo-by-zero", "OPENQASM 3.1; const int value = 1 % 0;"},
      {"integer-modulo-overflow",
       "OPENQASM 3.1; const int value = -9223372036854775808 % -1;"},
      {"negative-integer-power", "OPENQASM 3.1; const int value = 2 ** -1;"},
      {"integer-add-overflow",
       "OPENQASM 3.1; const int value = 9223372036854775807 + 1;"},
      {"integer-subtract-overflow",
       "OPENQASM 3.1; const int value = -9223372036854775808 - 1;"},
      {"integer-multiply-overflow",
       "OPENQASM 3.1; const int value = 9223372036854775807 * 2;"},
      {"bool-ordering",
       "OPENQASM 3.1; bool left = true; bool right = false; if (left < "
       "right) {}"},
      {"zero-register", "OPENQASM 3.1; qubit[0] q;"},
      {"negative-register", "OPENQASM 3.1; qubit[-1] q;"},
      {"dynamic-register-size", "OPENQASM 3.1; int size = 2; qubit[size] q;"},
      {"float-register-size", "OPENQASM 3.1; qubit[1.5] q;"},
      {"out-of-bounds-qubit", "OPENQASM 3.1; qubit[2] q; x q[2];"},
      {"float-qubit-index", "OPENQASM 3.1; qubit[2] q; x q[1.0];"},
      {"measurement-width",
       "OPENQASM 3.1; qubit[2] q; bit[3] c; c = measure q;"},
      {"unknown-reset", "OPENQASM 3.1; reset missing;"},
      {"unknown-barrier", "OPENQASM 3.1; barrier missing;"},
      {"duplicate-gate-parameter",
       "OPENQASM 3.1; gate custom(a, a) q { U(a, 0, 0) q; }"},
      {"duplicate-gate-qubit", "OPENQASM 3.1; gate custom q, q { cx q, q; }"},
      {"duplicate-custom-gate",
       "OPENQASM 3.1; gate custom q {} gate custom q {}"},
      {"custom-gate-conflicts-with-catalog", "OPENQASM 3.1; gate x q {}"},
      {"wrong-gate-parameter-count", "OPENQASM 3.1; qubit q; rx(1, 2) q;"},
      {"wrong-gate-qubit-count", "OPENQASM 3.1; qubit q; cx q;"},
      {"negative-control-count",
       "OPENQASM 3.1; qubit[2] q; ctrl(-1) @ x q[0], q[1];"},
      {"dynamic-control-count",
       "OPENQASM 3.1; int n = 1; qubit[2] q; ctrl(n) @ x q[0], q[1];"},
      {"non-integer-range", "OPENQASM 3.1; for int i in [0.0:1.0] {}"},
      {"non-bool-condition", "OPENQASM 3.1; int value = 1; if (value) {}"},
      {"bool-compound-assignment",
       "OPENQASM 3.1; bool value = true; value += false;"},
      {"measurement-compound-assignment",
       "OPENQASM 3.1; qubit q; bit value; value += measure q;"},
      {"unsupported-bitwise-not", "OPENQASM 3.1; int value = ~1;"},
      {"unsupported-bitwise-and", "OPENQASM 3.1; int value = 1 & 2;"},
      {"unsupported-bitwise-or", "OPENQASM 3.1; int value = 1 | 2;"},
      {"unsupported-bitwise-xor", "OPENQASM 3.1; int value = 1 ^ 2;"},
      {"unsupported-shift-left", "OPENQASM 3.1; int value = 1 << 2;"},
      {"unsupported-shift-right", "OPENQASM 3.1; int value = 2 >> 1;"},
      {"uninitialized-scalar", "OPENQASM 3.1; int x; int y = x + 1;"},
      {"self-initialization", "OPENQASM 3.1; int x = x + 1;"},
      {"uninitialized-condition", "OPENQASM 3.1; bool ready; if (ready) {}"},
      {"partially-initialized-branch",
       "OPENQASM 3.1; qubit q; bool choose = measure q; int x; if (choose) "
       "{ x = 1; } int y = x;"},
      {"forward-gate-call",
       "OPENQASM 3.1; qubit q; later q; gate later a { x a; }"},
      {"forward-gate-in-definition",
       "OPENQASM 3.1; gate first q { second q; } gate second q { x q; }"},
      {"hardware-qubit-in-gate", "OPENQASM 3.1; gate invalid q { x $0; }"},
      {"negative-index-out-of-bounds", "OPENQASM 3.1; qubit[2] q; x q[-3];"},
      {"bool-gate-parameter",
       "OPENQASM 3.1; bool value = true; qubit q; rx(value) q;"},
      {"local-qubit",
       "OPENQASM 3.1; bool value = true; if (value) { qubit q; }"},
      {"local-output",
       "OPENQASM 3.1; bool value = true; if (value) { output bit c; }"},
  };

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

TEST(OpenQASMTargetTest, TracksDefiniteStateAndBlockLocalBits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
int fromTrueBranch;
if (true) { fromTrueBranch = 1; }
int fromSelectedElse;
if (false) { fromSelectedElse = 1; } else { fromSelectedElse = 2; }
bool choose = measure q[0];
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
output bit[2] result = measure q;
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(SOURCE);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  ASSERT_EQ(analyzed.program->outputs.size(), 1);
  EXPECT_EQ(analyzed.program->registers[analyzed.program->outputs.front()].name,
            "result");

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMFrontendTest, RejectsSignedMinimumDivisionAndModuloOverflow) {
  constexpr llvm::StringLiteral sources[] = {
      "OPENQASM 3.1; const int minimum = -9223372036854775808; "
      "const int value = minimum / -1;",
      "OPENQASM 3.1; const int minimum = -9223372036854775808; "
      "const int value = minimum % -1;",
  };
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("overflows"),
              std::string::npos);
  }
}

TEST(OpenQASMTargetTest, EmitsAllScalarOperatorsAndComparisonPredicates) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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
output bit[4] result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  std::size_t integerComparisons = 0;
  std::size_t floatingComparisons = 0;
  std::size_t unsignedDivisions = 0;
  std::size_t unsignedRemainders = 0;
  std::size_t assertions = 0;
  module->walk([&](Operation* operation) {
    integerComparisons += isa<arith::CmpIOp>(operation);
    floatingComparisons += isa<arith::CmpFOp>(operation);
    unsignedDivisions += isa<arith::DivUIOp>(operation);
    unsignedRemainders += isa<arith::RemUIOp>(operation);
    assertions += isa<cf::AssertOp>(operation);
  });
  EXPECT_GE(integerComparisons, 13);
  EXPECT_GE(floatingComparisons, 6);
  EXPECT_GE(unsignedDivisions, 1);
  EXPECT_GE(unsignedRemainders, 1);
  EXPECT_GE(assertions, 2);
}

TEST(OpenQASMTargetTest, NormalizesNegativeIndicesAndChecksDynamicAliases) {
  constexpr llvm::StringLiteral INDEX_SOURCE = R"qasm(
OPENQASM 3.1;
qubit[3] q;
x q[-1];
bit[3] c = measure q;
if (c[-1]) { h q[-1]; }
int i = -1;
x q[i];
c[i] = measure q[i];
if (c[i]) { x q[0]; }
output bit[3] result = measure q;
)qasm";
  MLIRContext indexContext;
  auto indexed = oq3::translateOpenQASMToOQ3(INDEX_SOURCE, indexContext);
  ASSERT_TRUE(indexed);
  ASSERT_TRUE(succeeded(verify(*indexed)));
  std::size_t indexSelections = 0;
  indexed->walk([&](arith::SelectOp) { ++indexSelections; });
  EXPECT_GE(indexSelections, 3);

  constexpr llvm::StringLiteral ALIAS_SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
int i = 0;
cx q[i], q[i];
bit[2] result = measure q;
)qasm";
  MLIRContext aliasContext;
  auto aliased = oq3::translateOpenQASMToOQ3(ALIAS_SOURCE, aliasContext);
  ASSERT_TRUE(aliased);
  ASSERT_TRUE(succeeded(verify(*aliased)));
  std::size_t aliasAssertions = 0;
  aliased->walk([&](cf::AssertOp) { ++aliasAssertions; });
  EXPECT_GE(aliasAssertions, 3);
}

TEST(OpenQASMTargetTest, DispatchesDynamicQubitGatesWithStructuredControlFlow) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
x q[i];
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_EQ(std::distance(module->getOps<scf::IfOp>().begin(),
                          module->getOps<scf::IfOp>().end()),
            0);
  std::size_t dispatches = 0;
  module->walk([&](scf::IfOp) { ++dispatches; });
  EXPECT_EQ(dispatches, 1);
}

TEST(OpenQASMTargetTest,
     DispatchesDynamicQubitMeasurementsWithStructuredControlFlow) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bit c;
int i = 0;
c = measure q[i];
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t dispatches = 0;
  module->walk([&](scf::IfOp) { ++dispatches; });
  EXPECT_EQ(dispatches, 1);
}

TEST(OpenQASMTargetTest, SupportsOrdinaryBitInitializationAndAssignment) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t applications = 0;
  module->walk([&](oq3::ApplyGateOp) { ++applications; });
  EXPECT_EQ(applications, 1);
}

TEST(OpenQASMTargetTest, SupportsTargetlessMeasurements) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit q;
measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t measurements = 0;
  module->walk([&](qc::MeasureOp) { ++measurements; });
  EXPECT_EQ(measurements, 1);
}

TEST(OpenQASMTargetTest, SignExtendsDynamicNegativeRangeSteps) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
uint start = 3;
uint stop = 0;
int step = -1;
int count = 0;
for uint i in [start:step:stop] { count += 1; }
qubit q;
if (count == 4) { x q; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t signExtensions = 0;
  module->walk([&](arith::ExtSIOp extension) {
    if (extension.getResult().getType().isInteger(128)) {
      ++signExtensions;
    }
  });
  EXPECT_GE(signExtensions, 1);

  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  scf::ForOp loop;
  module->walk([&](scf::ForOp current) { loop = current; });
  ASSERT_TRUE(loop);
  APInt upper;
  ASSERT_TRUE(matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)));
  EXPECT_EQ(upper.getSExtValue(), 4);
}

TEST(OpenQASMTargetTest, PromotesMixedRangeEndpointsBeforeIteration) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const uint start = 0;
const int stop = -1;
qubit q;
for uint i in [start:-1:stop] { x q; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  std::size_t loops = 0;
  std::size_t applications = 0;
  module->walk([&](Operation* operation) {
    loops += isa<scf::ForOp>(operation);
    applications += isa<oq3::ApplyGateOp>(operation);
  });
  EXPECT_EQ(loops, 0);
  EXPECT_EQ(applications, 0);
}

TEST(OpenQASMTargetTest, ThreadsGateParametersIntoWhileConditions) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate conditional(theta) q {
  while (theta > 0.0) { x q; }
}
qubit q;
conditional(0.0) q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t loops = 0;
  module->walk([&](scf::WhileOp) { ++loops; });
  EXPECT_EQ(loops, 1);
}

TEST(OpenQASMTargetTest, RejectsModifiersOnStructuredCustomGatesAtQCTarget) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate looped(theta) q {
  for int i in [0:0] { p(theta) q; }
}
qubit q;
inv @ looped(pi / 2) q;
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
  manager.addPass(oq3::createOQ3ToQCPass());
  EXPECT_TRUE(failed(manager.run(*module)));
  EXPECT_NE(diagnostic.find("structured control flow"), std::string::npos);
}

TEST(OpenQASMTargetTest, DynamicQubitDispatchLowersThroughQCO) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
x q[i];
bit result = measure q[i];
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  bool sawResultDispatch = false;
  std::size_t stackAllocations = 0;
  module->walk([&](Operation* operation) {
    stackAllocations += isa<memref::AllocaOp>(operation);
    auto conditional = dyn_cast<scf::IfOp>(operation);
    if (!conditional || conditional.getNumResults() != 1 ||
        !conditional.getResult(0).getType().isInteger(1)) {
      return;
    }
    std::size_t measurements = 0;
    conditional->walk([&](qc::MeasureOp) { ++measurements; });
    sawResultDispatch |= measurements > 0;
  });
  EXPECT_EQ(stackAllocations, 0);
  EXPECT_TRUE(sawResultDispatch);
  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  manager.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  bool retainsQCReferences = false;
  module->walk([&](Operation* operation) {
    const auto isQCQubit = [](Type type) { return isa<qc::QubitType>(type); };
    retainsQCReferences |=
        llvm::any_of(operation->getOperandTypes(), isQCQubit) ||
        llvm::any_of(operation->getResultTypes(), isQCQubit);
  });
  EXPECT_FALSE(retainsQCReferences);
}

TEST(OpenQASMTargetTest,
     DynamicMeasurementsInLoopsAvoidRepeatedStackAllocation) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
for int i in [0:1] { measure q[i]; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  bool sawLoopDispatch = false;
  std::size_t stackAllocations = 0;
  module->walk([&](Operation* operation) {
    stackAllocations += isa<memref::AllocaOp>(operation);
    auto conditional = dyn_cast<scf::IfOp>(operation);
    if (!conditional || !conditional->getParentOfType<scf::ForOp>()) {
      return;
    }
    std::size_t measurements = 0;
    conditional->walk([&](qc::MeasureOp) { ++measurements; });
    sawLoopDispatch |= measurements > 0;
  });
  EXPECT_EQ(stackAllocations, 0);
  EXPECT_TRUE(sawLoopDispatch);

  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  manager.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
  bool nestedStackAllocation = false;
  module->walk([&](memref::AllocaOp allocation) {
    nestedStackAllocation |=
        allocation->getParentOfType<scf::ForOp>() != nullptr;
  });
  EXPECT_FALSE(nestedStackAllocation);
}

TEST(OpenQASMTargetTest, ClassicalControlFlowStateLowersThroughQCO) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
int total = 0;
for int i in [0:1] {
  total += i;
  h q;
}
while (total < 3) {
  total += 1;
  z q;
}
if (total == 3) {
  total += 4;
  x q;
} else {
  total += 5;
  y q;
}
if (total == 7) { s q; }
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  manager.addPass(createQCToQCO());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  bool retainsQCReferences = false;
  module->walk([&](Operation* operation) {
    const auto isQCQubit = [](Type type) { return isa<qc::QubitType>(type); };
    retainsQCReferences |=
        llvm::any_of(operation->getOperandTypes(), isQCQubit) ||
        llvm::any_of(operation->getResultTypes(), isQCQubit);
  });
  EXPECT_FALSE(retainsQCReferences);
  EXPECT_TRUE(module->getOps<oq3::GateOp>().empty());
}

TEST(OpenQASMTargetTest, PreservesBooleanEvaluationOrderAndIEEEInequality) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[2] q;
bool measured = measure q[0] && measure q[1];
float negative = -1.0;
float notANumber = sqrt(negative);
if (measured || notANumber != notANumber) { x q[0]; }
output bit[2] result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  SmallVector<std::int64_t> firstMeasuredIndices;
  bool sawUnorderedInequality = false;
  module->walk([&](Operation* operation) {
    if (auto comparison = dyn_cast<arith::CmpFOp>(operation)) {
      sawUnorderedInequality |=
          comparison.getPredicate() == arith::CmpFPredicate::UNE;
    }
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
  EXPECT_EQ(firstMeasuredIndices, (SmallVector<std::int64_t>{0, 1}));
  EXPECT_TRUE(sawUnorderedInequality);
}

TEST(OpenQASMTargetTest, EmitsStructuredLoopsWithCarriedMutableState) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
int total = 0;
for int i in [3:-1:0] {
  total += i;
  h q[i];
}
while (total > 0) {
  total -= 1;
}
if (total == 0) {
  x q[0];
}
bit[4] result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
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
  EXPECT_EQ(whileLoop.getInits().size(), 1);
  EXPECT_EQ(whileLoop.getNumResults(), 1);
  EXPECT_EQ(whileLoop.getBeforeBody()->getTerminator()->getNumOperands(), 2);
  EXPECT_EQ(whileLoop.getAfterBody()->getTerminator()->getNumOperands(), 1);
  bool sourceInductionUsesNormalizedCounter = false;
  forLoop.walk([&](arith::IndexCastOp cast) {
    sourceInductionUsesNormalizedCounter |=
        cast.getIn() == forLoop.getInductionVar() &&
        cast.getResult().getType().isInteger(64);
  });
  EXPECT_TRUE(sourceInductionUsesNormalizedCounter);

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
  EXPECT_EQ(upper.getSExtValue(), 4);
  EXPECT_EQ(step.getSExtValue(), 1);

  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, NormalizesDynamicAndEmptyInclusiveRangesSafely) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[4] q;
int step = 1;
for int i in [0:step:3] { h q[i]; }
for int i in [3:1] { x q[0]; }
bit[4] result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t loops = 0;
  std::size_t assertions = 0;
  module->walk([&](Operation* operation) {
    loops += isa<scf::ForOp>(operation);
    assertions += isa<cf::AssertOp>(operation);
  });
  EXPECT_EQ(loops, 2);
  EXPECT_GE(assertions, 2);

  PassManager canonicalizer(&context);
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*module)));
  SmallVector<scf::ForOp> remainingLoops;
  std::size_t xApplications = 0;
  module->walk([&](Operation* operation) {
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      remainingLoops.push_back(loop);
    }
    if (auto application = dyn_cast<oq3::ApplyGateOp>(operation)) {
      xApplications += application.getCallee() == "x";
    }
  });
  ASSERT_EQ(remainingLoops.size(), 1);
  EXPECT_EQ(xApplications, 0);
  APInt upper;
  ASSERT_TRUE(matchPattern(remainingLoops.front().getUpperBound(),
                           m_ConstantInt(&upper)));
  EXPECT_EQ(upper.getSExtValue(), 4);
}

TEST(OpenQASMTargetTest, PreservesBranchAndWhileCarriedClassicalBits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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
output bit[2] result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t resultBearingConditionals = 0;
  std::size_t resultBearingWhiles = 0;
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

TEST(OpenQASMTargetTest, HandlesIntegerBoundaryRangesAndOperators) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
int iterations = 0;
for int i in [9223372036854775806:9223372036854775807] {
  iterations += 1;
}
int arithmetic = 2 ** 3;
arithmetic %= 3;
const bool ready = 2 < 3;
qubit q;
if (ready && iterations == 2 && arithmetic == 2) { x q; }
bit result = measure q;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OpenQASMTargetTest, SupportsRequiredLiteralFormsAndOperatorPrecedence) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
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

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t checkedPowers = 0;
  module->walk([&](scf::WhileOp) { ++checkedPowers; });
  EXPECT_EQ(checkedPowers, 1);
}

TEST(OpenQASMTargetTest, HandlesTheMaximumUnsignedSingletonRange) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
const uint maximum = 18446744073709551615;
qubit q;
for uint i in [maximum:maximum] { if (i == maximum) { x q; } }
)qasm";
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
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
  std::size_t remainingLoops = 0;
  std::size_t xApplications = 0;
  module->walk([&](Operation* operation) {
    remainingLoops += isa<scf::ForOp>(operation);
    if (auto application = dyn_cast<oq3::ApplyGateOp>(operation)) {
      xApplications += application.getCallee() == "x";
    }
  });
  EXPECT_EQ(remainingLoops, 0);
  EXPECT_EQ(xApplications, 1);
}

TEST(OpenQASMFrontendTest, DiagnosesMixedPhysicalAndDeclaredQubits) {
  constexpr llvm::StringLiteral sources[] = {
      "OPENQASM 3.1; qubit q; x q; x $0;",
      "OPENQASM 3.1; x $0; qubit q; x q;",
  };
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("mixing physical"),
              std::string::npos);
  }
}

TEST(OpenQASMTargetTest, ExpandsAnOperandlessBarrierToAllDeclaredQubits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[3] q;
barrier;
bit[3] result = measure q;
)qasm";
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t barriers = 0;
  module->walk([&](qc::BarrierOp barrier) {
    ++barriers;
    EXPECT_EQ(barrier.getNumQubits(), 3);
  });
  EXPECT_EQ(barriers, 1);
}

TEST(OpenQASMTargetTest, CanonicalizesVariadicCompatibilityGates) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
qubit[10] q;
mcx q[0], q[1], q[2], q[3];
mcphase(0.5) q[0], q[1], q[2];
mcx_vchain q[0], q[1], q[2], q[3], q[4], q[8], q[9];
mcx_recursive q[0], q[1], q[2], q[3], q[4], q[9];
)qasm";
  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  module->walk([&](oq3::ApplyGateOp application) {
    EXPECT_TRUE(application.getCallee() == "x" ||
                application.getCallee() == "p");
  });

  PassManager manager(&context);
  manager.addPass(oq3::createOQ3ToQCPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  std::size_t controls = 0;
  std::size_t xGates = 0;
  std::size_t phaseGates = 0;
  module->walk([&](Operation* operation) {
    controls += isa<qc::CtrlOp>(operation);
    xGates += isa<qc::XOp>(operation);
    phaseGates += isa<qc::POp>(operation);
  });
  EXPECT_EQ(controls, 13);
  EXPECT_EQ(xGates, 3);
  EXPECT_EQ(phaseGates, 1);
}

TEST(OpenQASMTargetTest, BroadcastsRegistersAlongsideScalarQubits) {
  constexpr llvm::StringLiteral SOURCE = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[3] controls;
qubit target;
cx controls, target;
bit[3] left = measure controls;
bit right = measure target;
)qasm";

  MLIRContext context;
  auto module = oq3::translateOpenQASMToOQ3(SOURCE, context);
  ASSERT_TRUE(module);
  std::size_t applications = 0;
  module->walk([&](oq3::ApplyGateOp) { ++applications; });
  EXPECT_EQ(applications, 3);
}

TEST(OpenQASMTargetTest, PreservesImportedLoopAndDynamicIndexBehavior) {
  struct OperationCounts {
    std::size_t h;
    std::size_t x;
    std::size_t measurements;
    std::size_t controls;
  };
  struct ConditionalCounts {
    std::size_t semantic;
    std::size_t dispatch;
    std::size_t whileMeasurements;
  };
  struct Fixture {
    llvm::StringRef name;
    llvm::StringRef source;
    SmallVector<std::int64_t> tripCounts;
    std::size_t whileLoops;
    OperationCounts operations;
    ConditionalCounts conditionals;
  };
  const Fixture fixtures[] = {
      {"nested-if-for",
       qasm::nestedIfOpForLoop,
       {3},
       0,
       {5, 0, 2, 0},
       {1, 2, 0}},
      {"simple-while", qasm::simpleWhileReset, {}, 1, {2, 0, 2, 0}, {0, 0, 1}},
      {"simple-for", qasm::simpleForLoop, {2}, 0, {2, 0, 2, 0}, {0, 1, 0}},
      {"nested-for-if",
       qasm::nestedForLoopIfOp,
       {2},
       0,
       {3, 0, 2, 0},
       {1, 1, 0}},
      {"nested-for-while",
       qasm::nestedForLoopWhileOp,
       {2, 2},
       1,
       {4, 0, 4, 0},
       {0, 3, 2}},
      {"loop-control-separate",
       qasm::nestedForLoopCtrlOpWithSeparateQubit,
       {3},
       0,
       {4, 3, 1, 3},
       {0, 4, 0}},
      {"loop-control-extracted",
       qasm::nestedForLoopCtrlOpWithExtractedQubit,
       {3},
       0,
       {5, 3, 1, 3},
       {0, 6, 0}},
      {"expression-dynamic-index",
       qasm::expressionDynamicIntIndex,
       {3},
       0,
       {4, 0, 4, 0},
       {0, 3, 0}},
      {"expression-mod-index",
       qasm::expressionModIndex,
       {4},
       0,
       {2, 0, 2, 0},
       {0, 1, 0}},
      {"condition-while-and",
       qasm::conditionWhileAnd,
       {},
       1,
       {3, 0, 4, 0},
       {0, 0, 2}},
  };

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.name.str());
    MLIRContext context;
    auto module = oq3::translateOpenQASMToOQ3(fixture.source, context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    PassManager canonicalizer(&context);
    canonicalizer.addPass(createCanonicalizerPass());
    ASSERT_TRUE(succeeded(canonicalizer.run(*module)));

    SmallVector<scf::ForOp> forLoops;
    std::size_t whileLoops = 0;
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

    PassManager lowering(&context);
    lowering.addPass(oq3::createOQ3ToQCPass());
    ASSERT_TRUE(succeeded(lowering.run(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));
    bool retainsOQ3 = false;
    std::size_t hGates = 0;
    std::size_t xGates = 0;
    std::size_t measurements = 0;
    std::size_t controls = 0;
    std::size_t semanticConditionals = 0;
    std::size_t dispatchConditionals = 0;
    SmallVector<scf::ForOp> loweredForLoops;
    SmallVector<scf::WhileOp> loweredWhileLoops;
    module->walk([&](Operation* operation) {
      retainsOQ3 |= operation->getName().getDialectNamespace() == "oq3";
      hGates += isa<qc::HOp>(operation);
      xGates += isa<qc::XOp>(operation);
      measurements += isa<qc::MeasureOp>(operation);
      controls += isa<qc::CtrlOp>(operation);
      if (auto control = dyn_cast<qc::CtrlOp>(operation)) {
        std::size_t controlledXGates = 0;
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
      std::size_t dispatchedQuantumOperations = 0;
      conditional->walk([&](Operation* nested) {
        dispatchedQuantumOperations +=
            isa<qc::HOp, qc::XOp, qc::MeasureOp, qc::CtrlOp>(nested);
      });
      EXPECT_GT(dispatchedQuantumOperations, 0)
          << "each dynamic-index dispatch must retain quantum behavior";
    });
    EXPECT_FALSE(retainsOQ3);
    EXPECT_EQ(hGates, fixture.operations.h);
    EXPECT_EQ(xGates, fixture.operations.x);
    EXPECT_EQ(measurements, fixture.operations.measurements);
    EXPECT_EQ(controls, fixture.operations.controls);
    EXPECT_EQ(semanticConditionals, fixture.conditionals.semantic);
    EXPECT_EQ(dispatchConditionals, fixture.conditionals.dispatch);

    ASSERT_EQ(loweredForLoops.size(), fixture.tripCounts.size());
    for (auto loop : loweredForLoops) {
      std::size_t bodyGates = 0;
      loop.getRegion().walk([&](Operation* nested) {
        bodyGates += isa<qc::HOp, qc::XOp>(nested);
      });
      EXPECT_GT(bodyGates, 0)
          << "each imported for-loop body must retain its gate behavior";
    }

    ASSERT_EQ(loweredWhileLoops.size(), fixture.whileLoops);
    for (auto loop : loweredWhileLoops) {
      std::size_t conditionMeasurements = 0;
      std::size_t bodyGates = 0;
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
