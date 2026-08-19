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
#include "mlir/Dialect/QC/Translation/StandardGate.h"
#include "mlir/Target/OpenQASM/Frontend.h"
#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;
using namespace mlir::oq3::test;

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
  EXPECT_NE(analyzed.diagnostics.front().message.find("distinct qubits"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, ProvesNestedAffineQuantumIndices) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[8] q;
qubit[8] left;
qubit[8] right;
int last = 7;
x q[-1];
for int i in [0:6] {
  x q[i];
  cx q[i], q[i + 1];
  h q[last - i];
  barrier q[i], q[i + 1];
  for int j in [i + 1:last] {
    cx q[j], q[i];
    for int step in [0:j] {
      cx left[j - step], right[j];
    }
  }
}
for int i in [0:2:6] { x q[i]; }
for int i in [0:3] {
  uint even = 2 * i;
  h q[even];
}
int stride = 2;
for int i in [0:stride:6] { x q[i]; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, PreservesEquivalentAffineScalarJoins) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
int i = 0;
bit choose = measure q[0];
if (choose) { i = i + 1; } else { i = 1; }
x q[i];
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, RejectsUnprovedQuantumIndices) {
  constexpr auto rejections =
      std::to_array<std::pair<llvm::StringRef, llvm::StringRef>>({
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:3] { for int j in "
           "[0:3] { cx q[i], q[j]; } }",
           "distinct qubits"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:3] { x q[i + 1]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[2] q; bit b = measure q[0]; int i = b; "
           "x q[i];",
           "not a scalar value"},
          {"OPENQASM 3.1; qubit[16] q; for int i in [0:3] { x q[i * i]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:3] { x q[i / 1]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:3] { x q[i % 4]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:1] { x q[i ** 1]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[2] q; for int i in [0:1] { "
           "x q[(i + 9223372036854775807) - 9223372036854775807]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[2] q; for int i in "
           "[9223372036854775806:9223372036854775807] { "
           "x q[i - 9223372036854775806]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [0:1] { x q[i << 1]; }",
           "not supported yet"},
          {"OPENQASM 3.1; qubit[4] q; for int i in [3:-1:0] { x q[i]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[4] q; int i = 4; if (i < 4) { x q[i]; }",
           "cannot prove that qubit index is in bounds"},
          {"OPENQASM 3.1; qubit[2] q; int i = 0; bit repeat = measure "
           "q[0]; while (repeat) { x q[i]; i = 1; repeat = measure q[0]; }",
           "cannot prove that qubit index is in bounds"},
      });

  for (const auto& [source, diagnostic] : rejections) {
    SCOPED_TRACE(source.str());
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find(diagnostic),
              std::string::npos)
        << analyzed.diagnostics.front().message;
  }
}

TEST(OpenQASMFrontendTest, BoundsNontrivialAffineDistinctnessProofWork) {
  std::string source =
      "OPENQASM 3.1; qubit[100] q; for int i in [0:0] { barrier ";
  for (size_t index = 0; index < 100; ++index) {
    if (index != 0) {
      source += ", ";
    }
    source +=
        "q[" + std::to_string(index) + " * i + " + std::to_string(index) + "]";
  }
  source += "; }";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("distinct qubits"),
            std::string::npos);

  constexpr llvm::StringLiteral broadcastSource = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate pair q0, q1, unused { cx q0, q1; }
qubit[2] q;
qubit[2048] other;
for int i in [0:0] { pair q[i], q[i + 1], other; }
)qasm";
  auto broadcast = oq3::frontend::analyzeOpenQASM(broadcastSource);
  ASSERT_TRUE(broadcast) << broadcast.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, RejectsDuplicateBarrierQubits) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; barrier q, q;",
      "OPENQASM 3.1; qubit[2] q; barrier q[0], q[0];",
      "OPENQASM 3.1; qubit[2] q; int i = 0; barrier q, q[i];",
      "OPENQASM 3.1; barrier $0, $0;",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("same qubit"),
              std::string::npos);
  }
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

TEST(OpenQASMFrontendTest, CanonicalGateNamesRoundTripThroughTheCatalog) {
  for (const auto& entry : oq3::frontend::getGateCatalog()) {
    const auto& descriptor = qc::getStandardGateDescriptor(entry.lowering);
    const auto canonicalName = oq3::frontend::canonicalGateName(entry.lowering);
    const auto* canonicalEntry = oq3::frontend::lookupGate(canonicalName);
    ASSERT_NE(canonicalEntry, nullptr) << canonicalName.str();
    EXPECT_EQ(canonicalEntry->lowering, entry.lowering) << entry.name.str();
    EXPECT_EQ(canonicalEntry->parameterCount, descriptor.parameterCount)
        << entry.name.str();
    EXPECT_EQ(canonicalEntry->controlCount, descriptor.controlCount)
        << entry.name.str();
    EXPECT_EQ(canonicalEntry->targetCount, descriptor.targetCount)
        << entry.name.str();
  }
}

TEST(OpenQASMFrontendTest, RestrictsMathBuiltinsOnGateAngles) {
  for (const auto function : {llvm::StringRef{"exp"}, llvm::StringRef{"log"},
                              llvm::StringRef{"sqrt"}}) {
    const auto source = "OPENQASM 3.1; gate invalid(theta) q { rx(" +
                        function.str() + "(theta)) q; }";
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << function.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("angle"),
              std::string::npos)
        << function.str();
  }
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

TEST(OpenQASMFrontendTest, RejectsUninitializedScalarOutputs) {
  auto analyzed =
      oq3::frontend::analyzeOpenQASM("OPENQASM 3.1; output int result;");
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("Output scalar 'result'"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsRecursiveCustomGatesAfterBodyAnalysis) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
gate recursive q {
  recursive q;
}
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_EQ(analyzed.diagnostics.size(), 1);
  EXPECT_EQ(analyzed.diagnostics.front().message,
            "recursive custom gate definition involving 'recursive'");
  EXPECT_EQ(analyzed.diagnostics.front().location.line, 4);
  EXPECT_EQ(analyzed.diagnostics.front().location.column, 3);
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

TEST(OpenQASMFrontendTest, FoldsFixedAngleConversionsAndArithmetic) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const float two_pi = 6.283185307179586;
angle[8] halfway = angle[8](two_pi * (127.0 / 512.0));
const angle[4] negative = angle[4](-pi / 8.0);
const angle[4] multi_turn = angle[4](tau + pi / 2.0);
const angle[1] width_one = angle[1](pi);
const angle[4] a = angle[4](7.0 * pi / 8.0);
const angle[4] b = angle[4](pi / 8.0);
const angle[8] widened = angle[8](b);
const angle[8] mixed_width = widened + b;
const angle[3] tie_even_down = angle[3](b);
const angle[3] tie_even_up = angle[3](angle[4](3.0 * pi / 8.0));
const angle[4] sum = a + b;
const angle[4] difference = b - a;
const angle[4] product = 2 * b;
const angle[4] quotient = a / 2;
const angle[4] negated = -b;
angle smallest_default = angle(tau / 4503599627370496.0);
const angle[52] huge = angle[52](1e300);
const angle[52] subnormal = angle[52](5e-324);
qubit q;
rx(halfway) q;
rx(negative) q;
rx(multi_turn) q;
rx(width_one) q;
rx(widened) q;
rx(mixed_width) q;
rx(tie_even_down) q;
rx(tie_even_up) q;
rx(sum) q;
rx(difference) q;
rx(product) q;
rx(quotient) q;
rx(negated) q;
rx(smallest_default) q;
rx(huge) q;
rx(subnormal) q;
rx(sin(a)) q;
rx(sin(angle[4](pi / 8.0))) q;
if (a > b && b < a && a == 7.0 * pi / 8.0) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;

  const auto defaultStep = std::ldexp(2.0 * std::numbers::pi, -52);
  const std::array expectedParameters{
      std::numbers::pi / 2.0,
      15.0 * std::numbers::pi / 8.0,
      std::numbers::pi / 2.0,
      std::numbers::pi,
      std::numbers::pi / 8.0,
      std::numbers::pi / 4.0,
      0.0,
      std::numbers::pi / 2.0,
      std::numbers::pi,
      5.0 * std::numbers::pi / 4.0,
      std::numbers::pi / 4.0,
      3.0 * std::numbers::pi / 8.0,
      15.0 * std::numbers::pi / 8.0,
      defaultStep,
      3985068968215732.0 * defaultStep,
      0.0,
      std::sin(7.0 * std::numbers::pi / 8.0),
      std::sin(std::numbers::pi / 8.0),
  };
  size_t parameterIndex = 0;
  bool sawTrueCondition = false;
  for (const auto statement : analyzed.program->body) {
    const auto& data = analyzed.program->statements[statement].data;
    if (const auto* application =
            std::get_if<oq3::frontend::GateApplication>(&data);
        application != nullptr && application->callee == "rx") {
      ASSERT_LT(parameterIndex, expectedParameters.size());
      const auto& parameter =
          analyzed.program->expressions.at(application->parameters.front());
      ASSERT_EQ(parameter.type, oq3::frontend::ScalarType::Angle);
      const auto& value = parameter.kind == oq3::frontend::ExpressionKind::Cast
                              ? analyzed.program->expressions.at(parameter.lhs)
                              : parameter;
      ASSERT_EQ(value.kind, oq3::frontend::ExpressionKind::Constant);
      EXPECT_DOUBLE_EQ(std::get<double>(value.constant),
                       expectedParameters[parameterIndex]);
      ++parameterIndex;
    }
    if (const auto* conditional =
            std::get_if<oq3::frontend::IfStatement>(&data)) {
      const auto& condition =
          analyzed.program->conditions.at(conditional->condition);
      sawTrueCondition =
          condition.kind == oq3::frontend::ConditionKind::Literal &&
          condition.literal;
    }
  }
  EXPECT_EQ(parameterIndex, expectedParameters.size());
  EXPECT_TRUE(sawTrueCondition);
  EXPECT_TRUE(analyzed.program->scalars.empty());
  EXPECT_TRUE(analyzed.program->outputs.empty());
}

TEST(OpenQASMFrontendTest, RejectsUnsupportedFixedAnglePrograms) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; angle[0] a = pi;",
      "OPENQASM 3.1; angle[53] a = pi;",
      "OPENQASM 3.1; angle a;",
      "OPENQASM 3.1; float f = 0.5; angle[8] a = f;",
      "OPENQASM 3.1; angle[8] a = pi; a = pi / 2;",
      "OPENQASM 3.1; float f = 0.5; const angle[8] a = angle[8](f);",
      "OPENQASM 3.1; const angle[8] a = angle[8](pi); const angle[8] b = a / "
      "a;",
      "OPENQASM 3.1; const angle[8] a = angle[8](pi); const angle[8] b = a * "
      "256;",
      "OPENQASM 3.1; output angle[8] result;",
      "OPENQASM 3.1; float theta = pi; qubit q; rx(angle[8](theta)) q;",
      "OPENQASM 3.1; const angle[8] a = angle[8](pi); const bool bad = a == 1;",
      "OPENQASM 3.1; const angle[8] a = angle[8](pi); const angle[8] bad = a + "
      "pi;",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed);
    ASSERT_FALSE(analyzed.diagnostics.empty());
  }
}

TEST(OpenQASMFrontendTest, UsesSpecificationTypesForRealConstantsAndArcTrig) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
const float circle = pi + tau;
const float inverse = arccos(0.5) + arcsin(0.5) + arctan(0.5);
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
      ASSERT_EQ(parameter.kind, oq3::frontend::ExpressionKind::Cast);
      ASSERT_EQ(parameter.type, oq3::frontend::ScalarType::Angle);
      const auto& operand = analyzed.program->expressions[parameter.lhs];
      sawWrappedParameter = operand.type == oq3::frontend::ScalarType::Uint &&
                            std::get<uint64_t>(operand.constant) == 0;
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

TEST(OpenQASMFrontendTest, FoldsUnsignedAndMixedNumericConstantOperators) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
const uint operand = 9;
const uint added = operand + 4;
const uint subtracted = operand - 4;
const uint multiplied = operand * 4;
const uint divided = operand / 4;
const uint remainder = operand % 4;
const uint builtin_remainder = mod(operand, 4);
const uint powered = operand ** 2;
const uint builtin_powered = pow(operand, 2);
qubit q;
rx(added) q;
rx(subtracted) q;
rx(multiplied) q;
rx(divided) q;
rx(remainder) q;
rx(builtin_remainder) q;
rx(powered) q;
rx(builtin_powered) q;
if (1.5 < operand) { x q; }
if (operand <= 9.0) { x q; }
if (10.0 > operand) { x q; }
if (operand >= 9.0) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;

  constexpr std::array<uint64_t, 8> expectedParameters{13, 5, 36, 2,
                                                       1,  1, 81, 81};
  size_t parameterIndex = 0;
  size_t trueConditions = 0;
  for (const auto statement : analyzed.program->body) {
    const auto& data = analyzed.program->statements[statement].data;
    if (const auto* application =
            std::get_if<oq3::frontend::GateApplication>(&data);
        application != nullptr && application->callee == "rx") {
      ASSERT_LT(parameterIndex, expectedParameters.size());
      ASSERT_EQ(application->parameters.size(), 1);
      const auto& parameter =
          analyzed.program->expressions[application->parameters.front()];
      ASSERT_EQ(parameter.kind, oq3::frontend::ExpressionKind::Cast);
      ASSERT_EQ(parameter.type, oq3::frontend::ScalarType::Angle);
      const auto& operand = analyzed.program->expressions[parameter.lhs];
      ASSERT_EQ(operand.kind, oq3::frontend::ExpressionKind::Constant);
      ASSERT_EQ(operand.type, oq3::frontend::ScalarType::Uint);
      EXPECT_EQ(std::get<uint64_t>(operand.constant),
                expectedParameters[parameterIndex]);
      ++parameterIndex;
    }
    if (const auto* conditional =
            std::get_if<oq3::frontend::IfStatement>(&data)) {
      const auto& condition =
          analyzed.program->conditions[conditional->condition];
      ASSERT_EQ(condition.kind, oq3::frontend::ConditionKind::Literal);
      trueConditions += static_cast<size_t>(condition.literal);
    }
  }
  EXPECT_EQ(parameterIndex, expectedParameters.size());
  EXPECT_EQ(trueConditions, 4);
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

TEST(OpenQASMFrontendTest, RecordsResolvedConversionsInTypedExpressions) {
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
  ASSERT_GE(analyzed.program->statements.size(), 5);

  const auto& declaration = std::get<oq3::frontend::ScalarDeclarationStatement>(
      analyzed.program->statements[0].data);
  ASSERT_TRUE(declaration.initializer);
  EXPECT_EQ(analyzed.program->expressions[*declaration.initializer].kind,
            oq3::frontend::ExpressionKind::Cast);

  const auto& assignment = std::get<oq3::frontend::ScalarAssignmentStatement>(
      analyzed.program->statements[4].data);
  ASSERT_TRUE(assignment.value);
  EXPECT_EQ(analyzed.program->expressions[*assignment.value].kind,
            oq3::frontend::ExpressionKind::Cast);
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
  EXPECT_EQ(analyzed.program->outputs.front().kind,
            oq3::frontend::OutputKind::BitRegister);
  EXPECT_EQ(
      analyzed.program->registers[analyzed.program->outputs.front().symbol]
          .name,
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

TEST(OpenQASMFrontendTest, AcceptsMixedPhysicalAndDeclaredQubits) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; qubit q; x q; x $0;",
      "OPENQASM 3.1; x $0; qubit q; x q;",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    EXPECT_TRUE(analyzed) << source.str();
  }
}

TEST(OpenQASMFrontendTest, SkipsOpenQASM2StdlibGateRedefinition) {
  // OpenQASM 2.0 programs often repeat standard library gate definitions; keep
  // the standard-library entry and drop the duplicate body.
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
gate sx a { U(pi/2, -pi/2, pi/2) a; }
sx q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(llvm::none_of(analyzed.program->gates, [](const auto& gate) {
    return gate.name == "sx";
  }));
}

TEST(OpenQASMFrontendTest, RejectsOpenQASM3StdlibGateShadowing) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
gate sx a { U(pi/2, -pi/2, pi/2) a; }
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("already declared"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, PrefersMatchingCompatibilityGateCatalogEntries) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
gate r(theta, phi) q {
  x q;
}
qubit q;
r(0.5, 0.25) q;
)qasm";

  auto compatible = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(compatible) << compatible.diagnostics.front().message;
  EXPECT_TRUE(llvm::none_of(compatible.program->gates,
                            [](const auto& gate) { return gate.name == "r"; }));

  auto strict = oq3::frontend::analyzeOpenQASM(
      source, {.gatePolicy = oq3::frontend::GatePolicy::Strict});
  ASSERT_TRUE(strict) << strict.diagnostics.front().message;
  EXPECT_TRUE(llvm::any_of(strict.program->gates,
                           [](const auto& gate) { return gate.name == "r"; }));
}

TEST(OpenQASMFrontendTest, RejectsCompatibilityGateSignatureMismatch) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"qasm(OPENQASM 3.1;
gate r(theta) q {}
)qasm",
      R"qasm(OPENQASM 3.1;
gate r(theta, phi) q0, q1 {}
)qasm",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find(
                  "does not match its compatibility signature"),
              std::string::npos);
  }
}

TEST(OpenQASMFrontendTest, AcceptsOpenQASM2PartialClassicalRegisterIf) {
  // Classic OpenQASM 2.0: if (c == k) after measuring only some bits of c.
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
measure q[0] -> c[0];
if(c==1) x q[1];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
}

TEST(OpenQASMFrontendTest, AcceptsWideIntegerLiteralInOpenQASM2If) {
  // 2^70 + 9 fits in an 80-bit register but exceeds uint64_t.
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[80];
measure q[0] -> c[0];
if(c==1180591620717411303433) x q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(llvm::any_of(analyzed.program->conditions, [](const auto& c) {
    return c.kind == oq3::frontend::ConditionKind::Bit && c.bit.index == 70;
  }));
}

TEST(OpenQASMFrontendTest, AcceptsInitializedRegisterConditionInOpenQASM3) {
  std::string source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit[80] c;
)qasm";
  for (size_t index = 0; index < 80; ++index) {
    source += "c[" + std::to_string(index) + "] = false;\n";
  }
  source += R"qasm(
c[70] = measure q;
if(c==1180591620717411303424) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);

  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(llvm::any_of(analyzed.program->conditions, [](const auto& c) {
    return c.kind == oq3::frontend::ConditionKind::Bit && c.bit.index == 70;
  }));
}

TEST(OpenQASMFrontendTest, RejectsUninitializedRegisterConditionInOpenQASM3) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit[2] c;
c[0] = measure q;
if(c==1) { x q; }
)qasm";

  auto analyzed = oq3::frontend::analyzeOpenQASM(source);

  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(
      analyzed.diagnostics.front().message.find("has not been initialized"),
      std::string::npos);
}

TEST(OpenQASMFrontendTest,
     AcceptsWideIntegerLiteralWithDigitSeparatorsInOpenQASM2If) {
  // Same value as above, spelled with grammar-legal digit separators.
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[80];
measure q[0] -> c[0];
if(c==1_180_591_620_717_411_303_433) x q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  EXPECT_TRUE(llvm::any_of(analyzed.program->conditions, [](const auto& c) {
    return c.kind == oq3::frontend::ConditionKind::Bit && c.bit.index == 70;
  }));
}

TEST(OpenQASMFrontendTest,
     FoldsNonFittingWideIntegerOpenQASM2RegisterCondition) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[80];
measure q[0] -> c[0];
if(c==123456789012345678901234567890) x q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  const oq3::frontend::IfStatement* conditional = nullptr;
  for (const auto statement : analyzed.program->body) {
    conditional = std::get_if<oq3::frontend::IfStatement>(
        &analyzed.program->statements[statement].data);
    if (conditional != nullptr) {
      break;
    }
  }
  ASSERT_NE(conditional, nullptr);
  const auto& condition = analyzed.program->conditions[conditional->condition];
  ASSERT_EQ(condition.kind, oq3::frontend::ConditionKind::Literal);
  EXPECT_FALSE(condition.literal);
}

TEST(OpenQASMFrontendTest, AcceptsNarrowConstantAgainstWideOpenQASM2Register) {
  // Zero-extend a narrow constant across the full >64-bit register.
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[80];
measure q[0] -> c[0];
if(c==1) x q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_TRUE(analyzed) << analyzed.diagnostics.front().message;
  // Truncating to 64 bits would omit Not(c[79]).
  EXPECT_TRUE(llvm::any_of(analyzed.program->conditions, [&](const auto& c) {
    return c.kind == oq3::frontend::ConditionKind::Not &&
           analyzed.program->conditions[c.lhs].kind ==
               oq3::frontend::ConditionKind::Bit &&
           analyzed.program->conditions[c.lhs].bit.index == 79;
  }));
}

TEST(OpenQASMFrontendTest, RejectsNegativeOpenQASM2RegisterCondition) {
  constexpr llvm::StringLiteral source = R"qasm(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[2];
if(c==-1) x q[0];
)qasm";
  auto analyzed = oq3::frontend::analyzeOpenQASM(source);
  ASSERT_FALSE(analyzed);
  ASSERT_FALSE(analyzed.diagnostics.empty());
  EXPECT_NE(analyzed.diagnostics.front().message.find("unsigned integer"),
            std::string::npos);
}

TEST(OpenQASMFrontendTest, RejectsWideIntegerLiteralInOrdinaryConstants) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; int value = 999999999999999999999999999999;",
      "OPENQASM 3.1; int value = 999_999_999_999_999_999_999_999_999_999;",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("exceeds 64-bit"),
              std::string::npos)
        << analyzed.diagnostics.front().message;
  }
}

TEST(OpenQASMFrontendTest,
     RejectsWideIntegerLiteralInShortCircuitedConstantOperands) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      "OPENQASM 3.1; const bool value = false && "
      "(999999999999999999999999999999 == 0);",
      "OPENQASM 3.1; const bool value = true || "
      "(999999999999999999999999999999 == 0);",
  });
  for (const auto source : sources) {
    auto analyzed = oq3::frontend::analyzeOpenQASM(source);
    ASSERT_FALSE(analyzed) << source.str();
    ASSERT_FALSE(analyzed.diagnostics.empty());
    EXPECT_NE(analyzed.diagnostics.front().message.find("exceeds 64-bit"),
              std::string::npos)
        << analyzed.diagnostics.front().message;
  }
}

} // namespace
