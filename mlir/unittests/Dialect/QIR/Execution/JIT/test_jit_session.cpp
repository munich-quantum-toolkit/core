/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QIR/Execution/JIT/Session.h"
#include "mqt/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"

#include "support/TestSupport.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using mqt::test::value;

static std::string getProgram(const std::string_view file) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / file;
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    ADD_FAILURE() << "Failed to open " << path << ": "
                  << buffer.getError().message();
    return {};
  }
  return (*buffer)->getBuffer().str();
}

static void expectOpenQASMSampling(std::string_view source,
                                   std::string_view expected) {
  auto qc = mlir::QCProgram::fromOpenQASMString(source);
  MQT_ASSERT_SUCCESS(qc);
  auto qco = std::move(*qc).intoQCO();
  MQT_ASSERT_SUCCESS(qco);
  auto sampled =
      mlir::qco::sample(mlir::mqt::getEntryPoint(qco->module()), 1, 42);
  MQT_ASSERT_SUCCESS(sampled);
  ASSERT_EQ(sampled->size(), 1);
  EXPECT_EQ(sampled->begin()->first, expected);
  auto restored = std::move(*qco).intoQC();
  MQT_ASSERT_SUCCESS(restored);
  auto qir = std::move(*restored).intoQIR(mlir::QIRProfile::Adaptive);
  MQT_ASSERT_SUCCESS(qir);
  const auto ir = std::move(*qir).intoLLVMIR();
  MQT_ASSERT_SUCCESS(ir);
  auto sessionOwner = value(qir::JitSession::create(
      *ir, "openqasm-slices", qir::Execution::Sampling, 42));
  auto& session = *sessionOwner;
  session.runtime().disableOutput();
  std::vector<std::string> shots;
  ASSERT_EQ(value(session.sample(1, shots)), 0);
  // The JIT exposes recording order; DD strings put output bit zero on the
  // right.
  EXPECT_EQ(shots, std::vector<std::string>(
                       1, std::string(expected.rbegin(), expected.rend())));
}

namespace {

class JitSessionTest : public testing::Test {
protected:
  std::string sinkText;
  llvm::raw_string_ostream sink{sinkText};
};

TEST_F(JitSessionTest, BindsRuntimeForStaticConstructorsAndDestructors) {
  constexpr llvm::StringRef ir = R"(
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @setup, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @cleanup, ptr null }]
define void @setup() {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__rt__bool_record_output(i1 true, ptr null)
  ret void
}
define void @cleanup() {
  call void @__quantum__rt__bool_record_output(i1 false, ptr null)
  ret void
}
define i64 @main() #0 {
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  %measured = call i1 @__quantum__rt__read_result(ptr null)
  %value = zext i1 %measured to i64
  ret i64 %value
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare i1 @__quantum__rt__read_result(ptr)
declare void @__quantum__rt__bool_record_output(i1, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  std::async(std::launch::async, [&] {
    auto sessionOwner = value(qir::JitSession::create(
        ir, "static-lifecycle", qir::Execution::Sampling, 42, &sink));
    auto& session = *sessionOwner;
    EXPECT_EQ(value(session.run()), 1);
  }).get();
  EXPECT_EQ(sink.str(), "OUTPUT\tBOOL\ttrue\nOUTPUT\tBOOL\tfalse\n");
}

TEST(OpenQASMExecutionTest, ExecutesAffineQuantumSlices) {
  expectOpenQASMSampling(R"qasm(OPENQASM 3.1;
qubit[4] q;
qubit[4] r;
reset q;
reset r;
x q[0:2:2];
for int i in [0:1] { cx q[2*i:2*i+1], r[2*i:2*i+1]; }
output bit[4] result;
result = measure r;
)qasm",
                         "0101");
}

TEST(OpenQASMExecutionTest, ExecutesMeasurementDependentArithmetic) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"qasm(OPENQASM 3.1;
qubit q;
h q;
bit feedback = measure q;
int[64] limit = 10000;
if (feedback) { limit = 20000; }
int[64] sum = 0;
for int i in [0:limit-1] { sum += i; }
if (sum == 199990000) { x q; }
output bit result;
result = measure q;
)qasm");
  MQT_ASSERT_SUCCESS(qc);
  auto qir = std::move(*qc).intoQIR(mlir::QIRProfile::Adaptive);
  MQT_ASSERT_SUCCESS(qir);
  const auto ir = std::move(*qir).intoLLVMIR();
  MQT_ASSERT_SUCCESS(ir);
  auto sessionOwner = value(qir::JitSession::create(
      *ir, "feedback-arithmetic", qir::Execution::Sampling, 42));
  auto& session = *sessionOwner;
  session.runtime().disableOutput();
  EXPECT_FALSE(session.canSampleTerminal());
  ASSERT_TRUE(session.canSampleBranches());
  std::map<std::string, size_t> counts;
  ASSERT_EQ(value(session.sampleCounts(100000, counts)), 0);
  EXPECT_EQ(counts, (std::map<std::string, size_t>{{"0", 100000}}));
}

TEST(OpenQASMExecutionTest, PreservesBitstringAndSliceOrder) {
  expectOpenQASMSampling(R"qasm(OPENQASM 3.1;
output bit[6] literal;
output bit[3] selected;
output bit[6] copied;
literal = "00_1101";
selected = literal[3:-1:1];
copied = literal;
for int i in [0:1] {
  int next = uint[64](i) + uint[64](1);
  copied[next:next+1] = copied[i:i+1];
}
)qasm",
                         "000111011001101");
}

TEST(OpenQASMExecutionTest, ExecutesClassicalSliceExpressions) {
  expectOpenQASMSampling(R"qasm(OPENQASM 3.1;
int n = 2;
bit[6] b = "110101";
bit[3] a = b[0:n];
bit[4] partial;
partial[0:1] = "10";
output bit[3] result;
result[0] = (a == "101") && (~b[0:n] == "010") &&
    ((b[0:n] << uint(1)) == "010") &&
    (rotl(b[0:n], int[8](-1)) == "110") &&
    (rotr(b[0:n], -1) == "011") &&
    (popcount(b[0:n]) == 2) && (uint[3](b[0:n]) == 5) && bool(b[0:n]);
result[1] = (partial[0:1] == "10") && ((~0 ^ b[0:n]) == "010") &&
    ((b[0:n] & ~0) == "101");
b[n:-1:0] = "011";
result[2] = b[0:2] == "110";
b[0:n] = (~0 >> uint(n)) | rotl(1, n);
result[2] = result[2] && (b[0:2] == "101");
)qasm",
                         "111");
}

TEST(OpenQASMExecutionTest, ExecutesInclusiveRangeBoundaries) {
  const auto cases =
      std::to_array<std::tuple<std::string_view, std::string_view, int>>({
          {"[hi-1:hi]", "", 2},
          {"[lo:hi:hi]", "", 3},
          {"[hi-1+delta:hi]", "continue;", 2},
          {"[lo+1+delta:-1:lo]", "", 2},
          {"[uint(-2)+uint(delta):uint(-1)]", "", 2},
          {"[int(-1)+delta:-1:uint(-3)]", "if (count == 2) { break; }", 2},
          {"[2+delta:1]", "", 0},
      });
  for (const auto& [range, tail, count] : cases) {
    SCOPED_TRACE(range);
    const auto source =
        "OPENQASM 3.1; qubit q; reset q; bit zero = measure q; "
        "int delta = int(zero); const int hi = 9223372036854775807; "
        "const int lo = -9223372036854775807 - 1; int count = 0; "
        "for int i in " +
        std::string(range) + " { count += 1; " + std::string(tail) +
        " } "
        "output bit ok; ok = count == " +
        std::to_string(count) + ";";
    expectOpenQASMSampling(source, "1");
  }
}

TEST_F(JitSessionTest, LoadModuleFromUnterminatedMemory) {
  auto program = getProgram("BellPairStatic.ll");
  const auto size = program.size();
  program += "outside the input buffer";
  auto sessionOwner = value(qir::JitSession::create(
      llvm::StringRef(program.data(), size), "BellPairStatic.ll"));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  ASSERT_EQ(value(session.run()), 0);
  EXPECT_FALSE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, SamplingRecordsOutputs) {
  const auto program = getProgram("BellPairStatic.ll");
  // qir::Execution::Sampling is the default Execution mode
  auto sessionOwner =
      value(qir::JitSession::create(program, "BellPairStatic.ll"));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  ASSERT_EQ(value(session.run()), 0);
  EXPECT_FALSE(session.runtime().getMeasurements().empty());
  session.runtime().outputShotStart();
  EXPECT_THAT(sink.str(), ::testing::HasSubstr("METADATA\tentry_point\n"));
  EXPECT_THAT(sink.str(),
              ::testing::HasSubstr("METADATA\tqir_profiles\tbase_profile\n"));
}

TEST_F(JitSessionTest, StateExtractionLeavesNoRecordedOutputs) {
  const auto program = getProgram("BellPairStatic.ll");
  auto sessionOwner = value(qir::JitSession::create(
      program, "BellPairStatic.ll", qir::Execution::StateExtraction));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  ASSERT_EQ(value(session.run()), 0);
  EXPECT_TRUE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, StateExtractionSupportsAdaptiveControlAndLifetimes) {
  const auto ir = getProgram("StatevectorAdaptive.ll");
  auto sessionOwner = value(qir::JitSession::create(
      ir, "Adaptive.ll", qir::Execution::StateExtraction));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  for (size_t run = 0; run < 2; ++run) {
    ASSERT_EQ(value(session.run()), 0);
    EXPECT_TRUE(session.runtime().getMeasurements().empty());
    EXPECT_TRUE(sink.str().empty());
    auto state = value(session.runtime().takeState());
    EXPECT_EQ(state.numQubits, 4);
    const auto values = value(dd::getVector(state.edge));
    ASSERT_EQ(values.size(), 16);
    for (size_t i = 0; i < values.size(); ++i) {
      const auto expected = i == 4 || i == 7 ? std::polar(dd::SQRT2_2, 0.3)
                                             : std::complex<double>{};
      EXPECT_NEAR(std::abs(values[i] - expected), 0., 1e-12);
    }
    state.dd->decRef(state.edge);
  }
}

TEST_F(JitSessionTest, StateExtractionRejectsNonTerminalMeasurement) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 {
  call void @measure()
  call void @__quantum__qis__x__body(ptr null)
  ret i64 0
}
declare void @measure() #1
declare void @__quantum__qis__x__body(ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
attributes #1 = { "irreversible" }
)";
  EXPECT_THAT(
      llvm::toString(qir::JitSession::create(ir, "NonTerminal.ll",
                                             qir::Execution::StateExtraction)
                         .takeError()),
      testing::HasSubstr("terminal region"));
}

TEST_F(JitSessionTest, OutputSchemaDefaultsToLabeledWhenAttributeAbsent) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "NoOutputSchema.ll"));
  auto& session = *sessionOwner;
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Labeled);
}

TEST_F(JitSessionTest, OutputSchemaFromLabeledAttribute) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "output_labeling_schema"="labeled" }
)";
  auto sessionOwner =
      value(qir::JitSession::create(ir, "LabeledOutputSchema.ll"));
  auto& session = *sessionOwner;
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Labeled);
}

TEST_F(JitSessionTest, OutputSchemaFromOrderedAttribute) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "output_labeling_schema"="ordered" }
)";
  auto sessionOwner =
      value(qir::JitSession::create(ir, "OrderedOutputSchema.ll"));
  auto& session = *sessionOwner;
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Ordered);
}

TEST_F(JitSessionTest, ExecutesArbitrarilyNamedEntryPoint) {
  constexpr std::string_view ir = R"(
define i64 @bell_entry() #0 { ret i64 7 }
  attributes #0 = { "entry_point" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "NamedEntry.ll"));
  auto& session = *sessionOwner;
  EXPECT_EQ(value(session.run()), 7);
}

TEST_F(JitSessionTest, AcceptsFutureMinorVersionsAndRejectsOtherMajors) {
  const std::string program = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"qir_major_version", )";
  for (const auto& [major, minor] : {
           std::pair{"i32 2", "i32 99"},
           std::pair{"i32 3", "i32 0"},
           std::pair{"i32 -1", "i32 0"},
           std::pair{"!\"2\"", "i32 0"},
           std::pair{"i32 2", "i32 -1"},
           std::pair{"i32 2", "!\"1\""},
       }) {
    const std::string ir = program + major + R"(}
!1 = !{i32 1, !"qir_minor_version", )" +
                           minor + "}\n";
    auto session = qir::JitSession::create(ir, "versions.ll");
    if (llvm::StringRef(major) == "i32 2" &&
        llvm::StringRef(minor) == "i32 99") {
      ASSERT_TRUE(static_cast<bool>(session));
      EXPECT_EQ(value((*session)->run()), 0);
    } else {
      ASSERT_FALSE(static_cast<bool>(session));
      EXPECT_NE(llvm::toString(session.takeError()).find("major version 2"),
                std::string::npos);
    }
  }
}

TEST_F(JitSessionTest, RejectsMalformedModuleFlagsBeforeReadingVersions) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
!llvm.module.flags = !{!0}
!0 = !{i32 1}
)";
  auto session = qir::JitSession::create(ir, "malformed-version.ll");
  ASSERT_FALSE(static_cast<bool>(session));
  EXPECT_NE(llvm::toString(session.takeError()).find("Invalid QIR module"),
            std::string::npos);
}

TEST_F(JitSessionTest, SupportsQir21DynamicResources) {
  constexpr std::string_view ir = R"(
define i64 @adaptive() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %q = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %r = call ptr @__quantum__rt__result_allocate(ptr null)
  call void @__quantum__qis__x__body(ptr %q)
  call void @__quantum__qis__mz__body(ptr %q, ptr %r)
  call void @__quantum__rt__result_record_output(ptr %r, ptr null)
  call void @__quantum__rt__result_release(ptr %r)
  call void @__quantum__rt__qubit_release(ptr %q)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare ptr @__quantum__rt__qubit_allocate(ptr)
declare ptr @__quantum__rt__result_allocate(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
declare void @__quantum__rt__result_release(ptr)
declare void @__quantum__rt__qubit_release(ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "Qir21Resources.ll"));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  EXPECT_EQ(value(session.run()), 0);
  EXPECT_EQ(session.runtime().getMeasurements(), "1");
}

TEST_F(JitSessionTest, WritesDefaultsAndComputedValuesIntoResultArrays) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %results = alloca [2 x ptr]
  call void @__quantum__rt__result_array_allocate(i64 2, ptr %results, ptr null)
  %first = load ptr, ptr %results
  %address = getelementptr ptr, ptr %results, i64 1
  %second = load ptr, ptr %address
  call void @__quantum__rt__write_result(i1 true, ptr %first)
  call void @__quantum__qis__mz__body(ptr null, ptr %second)
  %bit = call i1 @__quantum__rt__read_result(ptr %second)
  %corrected = xor i1 %bit, true
  call void @__quantum__rt__write_result(i1 %corrected, ptr %second)
  call void @__quantum__rt__result_array_record_output(i64 2, ptr %results, ptr null)
  call void @__quantum__rt__result_array_release(i64 2, ptr %results)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__rt__result_array_allocate(i64, ptr, ptr)
declare void @__quantum__rt__result_array_release(i64, ptr)
declare void @__quantum__rt__write_result(i1, ptr)
declare i1 @__quantum__rt__read_result(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_array_record_output(i64, ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"writable_results", i1 true}
)";
  auto session = value(qir::JitSession::create(
      ir, "writable-array", qir::Execution::Sampling, 17, &sink));
  std::vector<std::string> shots;
  ASSERT_EQ(value(session->sample(8, shots)), 0);
  EXPECT_EQ(shots, std::vector<std::string>(8, "11"));
  EXPECT_THAT(sinkText, testing::HasSubstr("OUTPUT\tRESULT_ARRAY\t11"));
}

TEST_F(JitSessionTest, WritableResultsReplaceDeferredMeasurements) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__qis__h__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__rt__write_result(i1 true, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__write_result(i1 false, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__write_result(i1, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" "required_num_qubits"="1" "required_num_results"="1" }
)";
  auto session = value(qir::JitSession::create(
      ir, "overwritten-measurement", qir::Execution::Sampling, 17, &sink));
  std::vector<std::string> shots;
  ASSERT_EQ(value(session->sample(16, shots)), 0);
  EXPECT_EQ(shots, std::vector<std::string>(16, "10"));
  std::map<std::string, size_t> counts;
  bool retained = false;
  ASSERT_EQ(value(session->sampleCounts(16, counts, &retained)), 0);
  EXPECT_EQ(counts, (std::map<std::string, size_t>{{"10", 16}}));
  EXPECT_TRUE(retained);
  const auto state = value(session->runtime().takeState());
  const auto amplitudes = value(dd::getVector(state.edge));
  ASSERT_EQ(amplitudes.size(), 2);
  for (const auto amplitude : amplitudes) {
    EXPECT_NEAR(std::abs(amplitude), std::sqrt(0.5), 1e-12);
  }
}

TEST_F(JitSessionTest, SupportsNativeOneAndTwoControlExtensions) {
  constexpr std::string_view ir = R"(
define i64 @native_controls() #0 {
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__x__body(ptr inttoptr (i64 1 to ptr))
  call void @__quantum__qis__crx__body(double 3.141592653589793, ptr null, ptr inttoptr (i64 2 to ptr))
  call void @__quantum__qis__ccrx__body(double 3.141592653589793, ptr null, ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 3 to ptr))
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 2 to ptr), ptr null)
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 3 to ptr), ptr inttoptr (i64 1 to ptr))
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__crx__body(double, ptr, ptr)
declare void @__quantum__qis__ccrx__body(double, ptr, ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="4" "required_num_results"="2" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "NativeControls.ll"));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  EXPECT_EQ(value(session.run()), 0);
  EXPECT_EQ(session.runtime().getMeasurements(), "11");
}

TEST_F(JitSessionTest, RejectsObsoleteQubitAllocationSignature) {
  constexpr std::string_view ir = R"(
define i64 @legacy() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %q = call ptr @__quantum__rt__qubit_allocate()
  call void @__quantum__rt__qubit_release(ptr %q)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare ptr @__quantum__rt__qubit_allocate()
declare void @__quantum__rt__qubit_release(ptr)
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "LegacyAllocation.ll").takeError())
          .empty());
}

TEST_F(JitSessionTest, SupportsGenericControlledSpecialization) {
  constexpr std::string_view ir = R"(
define i64 @generic_controlled() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %control0 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %control1 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %control2 = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %target = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %result = call ptr @__quantum__rt__result_allocate(ptr null)
  %controls = call ptr @__quantum__rt__array_create_1d(i32 8, i64 3)
  %control_slot0 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %controls, i64 0)
  %control_slot1 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %controls, i64 1)
  %control_slot2 = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %controls, i64 2)
  store ptr %control0, ptr %control_slot0
  store ptr %control1, ptr %control_slot1
  store ptr %control2, ptr %control_slot2
  %args = call ptr @__quantum__rt__tuple_create(i64 16)
  %angle_slot = getelementptr { double, ptr }, ptr %args, i32 0, i32 0
  %target_slot = getelementptr { double, ptr }, ptr %args, i32 0, i32 1
  store double 3.141592653589793, ptr %angle_slot
  store ptr %target, ptr %target_slot
  call void @__quantum__qis__x__body(ptr %control0)
  call void @__quantum__qis__x__body(ptr %control1)
  call void @__quantum__qis__x__body(ptr %control2)
  call void @__quantum__qis__rx__ctl(ptr %controls, ptr %args)
  call void @__quantum__qis__mz__body(ptr %target, ptr %result)
  call void @__quantum__rt__result_record_output(ptr %result, ptr null)
  call void @__quantum__rt__result_release(ptr %result)
  call void @__quantum__rt__tuple_update_reference_count(ptr %args, i32 -1)
  call void @__quantum__rt__array_update_reference_count(ptr %controls, i32 -1)
  call void @__quantum__rt__qubit_release(ptr %control0)
  call void @__quantum__rt__qubit_release(ptr %control1)
  call void @__quantum__rt__qubit_release(ptr %control2)
  call void @__quantum__rt__qubit_release(ptr %target)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare ptr @__quantum__rt__qubit_allocate(ptr)
declare ptr @__quantum__rt__result_allocate(ptr)
declare ptr @__quantum__rt__array_create_1d(i32, i64)
declare ptr @__quantum__rt__array_get_element_ptr_1d(ptr, i64)
declare ptr @__quantum__rt__tuple_create(i64)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__rx__ctl(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
declare void @__quantum__rt__result_release(ptr)
declare void @__quantum__rt__tuple_update_reference_count(ptr, i32)
declare void @__quantum__rt__array_update_reference_count(ptr, i32)
declare void @__quantum__rt__qubit_release(ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  auto sessionOwner =
      value(qir::JitSession::create(ir, "ControlledRotation.ll"));
  auto& session = *sessionOwner;
  session.runtime().setOstream(sink);
  EXPECT_EQ(value(session.run()), 0);
  EXPECT_EQ(session.runtime().getMeasurements(), "1");
}

TEST_F(JitSessionTest, RejectsQirRunnerPauliRotationAbi) {
  constexpr std::string_view ir = R"(
define i64 @pauli_rotation() #0 {
  call void @__quantum__qis__r__body(i2 1, double 3.141592653589793, ptr null)
  ret i64 0
}
declare void @__quantum__qis__r__body(i2, double, ptr)
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(llvm::toString(
                   qir::JitSession::create(ir, "QirRunnerPauli.ll").takeError())
                   .empty());
}

TEST_F(JitSessionTest, SessionsExecuteIndependently) {
  const auto program = getProgram("BellPairStatic.ll");
  std::string firstSinkText;
  llvm::raw_string_ostream firstSink{firstSinkText};
  std::string secondSinkText;
  llvm::raw_string_ostream secondSink{secondSinkText};
  auto firstOwner = value(qir::JitSession::create(program, "first.ll"));
  auto& first = *firstOwner;
  auto secondOwner = value(qir::JitSession::create(program, "second.ll"));
  auto& second = *secondOwner;
  first.runtime().setOstream(firstSink);
  second.runtime().setOstream(secondSink);
  int64_t firstExit = -1;
  int64_t secondExit = -1;

  std::thread firstThread([&] { firstExit = value(first.run()); });
  std::thread secondThread([&] { secondExit = value(second.run()); });
  firstThread.join();
  secondThread.join();

  EXPECT_EQ(firstExit, 0);
  EXPECT_EQ(secondExit, 0);
  EXPECT_NE(&first.runtime(), &second.runtime());
  EXPECT_EQ(first.runtime().getMeasurements().size(), 2);
  EXPECT_EQ(second.runtime().getMeasurements().size(), 2);
  EXPECT_FALSE(firstSink.str().empty());
  EXPECT_FALSE(secondSink.str().empty());
}

TEST_F(JitSessionTest, SeedReproducesShotSequence) {
  const auto program = getProgram("BellPairStatic.ll");
  auto firstOwner = value(qir::JitSession::create(program, "first.ll"));
  auto& first = *firstOwner;
  auto secondOwner = value(qir::JitSession::create(program, "second.ll"));
  auto& second = *secondOwner;
  first.runtime().seed(42);
  second.runtime().seed(42);
  first.runtime().setOstream(sink);
  second.runtime().setOstream(sink);
  std::string firstSequence;
  std::string secondSequence;

  for (std::size_t shot = 0; shot < 16; ++shot) {
    ASSERT_EQ(value(first.run()), 0);
    ASSERT_EQ(value(second.run()), 0);
    firstSequence += first.runtime().getMeasurements().front();
    secondSequence += second.runtime().getMeasurements().front();
  }

  EXPECT_EQ(firstSequence, secondSequence);
  EXPECT_THAT(firstSequence, ::testing::HasSubstr("0"));
  EXPECT_THAT(firstSequence, ::testing::HasSubstr("1"));
}

TEST(JitSessionErrors, OwnsDiagnosticsAcrossLaterSessions) {
  auto error = [] {
    const std::string name = "owned-diagnostic.ll";
    return qir::JitSession::create("not LLVM IR", name).takeError();
  }();
  auto session =
      value(qir::JitSession::create("define i64 @main() #0 { ret i64 7 }\n"
                                    "attributes #0 = { \"entry_point\" }",
                                    "valid.ll"));
  EXPECT_EQ(value(session->run()), 7);
  EXPECT_THAT(llvm::toString(std::move(error)),
              testing::HasSubstr("owned-diagnostic.ll"));
}

TEST(JitSessionErrors, MalformedIRReturnsError) {
  constexpr std::string_view ir = R"(define i32 @main() {})";
  EXPECT_FALSE(
      llvm::toString(qir::JitSession::create(ir, "MalformedIR.ll").takeError())
          .empty());
}

TEST(JitSessionErrors, RejectsNonCompliantEntryPointSignature) {
  constexpr std::string_view ir = R"(
define i32 @main() #0 { ret i32 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "BadEntrySignature.ll").takeError())
          .empty());
}

TEST(JitSessionErrors, RejectsNonCEntryPointCallingConvention) {
  constexpr std::string_view ir = R"(
define preserve_allcc i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_THAT(llvm::toString(
                  qir::JitSession::create(ir, "entry-convention").takeError()),
              testing::HasSubstr("C calling convention"));
}

TEST(JitSessionErrors, RejectsEntryPointABIAttributes) {
  constexpr std::string_view ir = R"(
define inreg i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_THAT(llvm::toString(
                  qir::JitSession::create(ir, "entry-attribute").takeError()),
              testing::HasSubstr("unsupported ABI attribute inreg"));
}

TEST(JitSessionErrors, RejectsNonCRuntimeCallingConventions) {
  for (const auto* symbol : {"h__body", "sdg__body"}) {
    for (const auto& [declaration, call] :
         std::array<std::pair<std::string_view, std::string_view>, 3>{
             {
                 {"preserve_allcc ", "preserve_allcc "},
                 {"preserve_allcc ", ""},
                 {"", "preserve_allcc "},
             },
         }) {
      const auto ir =
          std::string("declare void @__quantum__qis__s__adj(ptr)\ndeclare ") +
          std::string(declaration) + "void @__quantum__qis__" + symbol +
          "(ptr)\n"
          "define i64 @main() #0 { call " +
          std::string(call) + "void @__quantum__qis__" + symbol +
          "(ptr null)\nret i64 0 }\n"
          "attributes #0 = { \"entry_point\" }";
      SCOPED_TRACE(ir);
      EXPECT_THAT(
          llvm::toString(
              qir::JitSession::create(ir, "runtime-convention").takeError()),
          testing::HasSubstr("C calling convention"));
    }
  }
}

TEST(JitSessionErrors, RejectsRuntimeABIAttributes) {
  for (const auto* symbol : {"h__body", "sdg__body"}) {
    for (const auto* attribute : {
             "byval(i8)",
             "byref(i8)",
             "sret(i8)",
             "inreg",
             "nest",
             "swiftself",
             "swiftasync",
         }) {
      for (const bool onDeclaration : {false, true}) {
        const auto ir =
            std::string("declare void @__quantum__qis__s__adj(ptr)\ndeclare "
                        "void @__quantum__qis__") +
            symbol + "(ptr " + (onDeclaration ? attribute : "") +
            ")\ndefine i64 @main() #0 { %q = alloca i8\n"
            "call void @__quantum__qis__" +
            symbol + "(ptr " + (onDeclaration ? "" : attribute) +
            " %q)\nret i64 0 }\n"
            "attributes #0 = { \"entry_point\" }";
        SCOPED_TRACE(ir);
        EXPECT_THAT(
            llvm::toString(
                qir::JitSession::create(ir, "runtime-attribute").takeError()),
            testing::HasSubstr("unsupported ABI attribute"));
      }
    }
  }
}

TEST(JitSessionErrors, RejectsMismatchedRuntimeCallType) {
  constexpr std::string_view ir = R"(
declare void @__quantum__qis__h__body(ptr)
define i64 @main() #0 {
  call void @__quantum__qis__h__body(double 0.0)
  ret i64 0
}
attributes #0 = { "entry_point" }
)";
  EXPECT_THAT(llvm::toString(
                  qir::JitSession::create(ir, "runtime-call-type").takeError()),
              testing::HasSubstr("must match its declaration"));
}

TEST(JitSessionErrors, RejectsInvalidModulesBeforeExecution) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 {
entry:
  br label %exit
exit:
  %x = phi i64 [0, %exit]
  ret i64 %x
}
attributes #0 = { "entry_point" }
)";
  EXPECT_THAT(
      llvm::toString(qir::JitSession::create(ir, "invalid-module").takeError()),
      testing::HasSubstr("Invalid QIR module"));
}

TEST(QIRJIT, AcceptsInternalCallingConventionsAndSemanticAttributes) {
  constexpr std::string_view ir = R"(
declare void @__quantum__rt__initialize(ptr noundef) nounwind
declare preserve_allcc void @__quantum__qis__h__body(ptr byval(i8))
define noundef i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr noundef null)
  %code = call fastcc i64 @helper()
  ret i64 %code
}
define internal fastcc i64 @helper() { ret i64 7 }
attributes #0 = { nounwind alignstack=16 "entry_point" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "internal-convention"));
  auto& session = *sessionOwner;
  EXPECT_EQ(value(session.run()), 7);
}

TEST(JitSessionErrors, RejectsMultipleEntryPoints) {
  constexpr std::string_view ir = R"(
define i64 @first() #0 { ret i64 0 }
define i64 @second() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "MultipleEntries.ll").takeError())
          .empty());
}

TEST(JitSessionErrors, RejectsMismatchedRuntimeDeclaration) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 {
  call void @__quantum__qis__x__body(double 0.0)
  ret i64 0
}
declare void @__quantum__qis__x__body(double)
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "BadRuntimeSignature.ll").takeError())
          .empty());
}

} // namespace

TEST(QIRBatchSampling, PreservesLogicalOutputOrderAndRepeatedMeasurements) {
  constexpr llvm::StringRef ir = R"(
@label = constant [6 x i8] c"items\00"
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 2 to ptr))
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 1 to ptr))
  call void @__quantum__rt__tuple_record_output(i64 4, ptr @label)
  call void @__quantum__rt__array_record_output(i64 2, ptr @label)
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr @label)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr @label)
  call void @__quantum__rt__int_record_output(i64 7, ptr @label)
  call void @__quantum__rt__double_record_output(double 0.5, ptr null)
  ret i64 0
}
declare void @__quantum__rt__int_record_output(i64, ptr)
declare void @__quantum__rt__double_record_output(double, ptr)
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
declare void @__quantum__rt__array_record_output(i64, ptr)
declare void @__quantum__rt__tuple_record_output(i64, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="3" "required_num_results"="2" }
)";
  for (const auto schema : {
           qir::Runtime::OutputSchema::Ordered,
           qir::Runtime::OutputSchema::Labeled,
       }) {
    std::string outputText;
    llvm::raw_string_ostream output{outputText};
    auto sessionOwner = value(qir::JitSession::create(
        ir, "output-order", qir::Execution::Sampling, 42, &output));
    auto& session = *sessionOwner;
    session.runtime().setOutputSchema(schema);
    std::vector<std::string> results;
    bool stateAvailable = false;
    ASSERT_EQ(value(session.sample(
                  2, [&](std::string_view shot) { results.emplace_back(shot); },
                  &stateAvailable)),
              0);
    EXPECT_TRUE(stateAvailable);
    EXPECT_EQ(results, std::vector<std::string>(2, "100"));
    const std::string label =
        schema == qir::Runtime::OutputSchema::Labeled ? "\titems" : "";
    std::string expectedRecordsStorage;
    llvm::raw_string_ostream expectedRecords(expectedRecordsStorage);
    expectedRecords << "OUTPUT\tTUPLE\t4" << label << "\n"
                    << "OUTPUT\tARRAY\t2" << label << "\n"
                    << "OUTPUT\tRESULT\t1" << label << "\n"
                    << "OUTPUT\tRESULT\t0\n"
                    << "OUTPUT\tRESULT\t0" << label << "\n"
                    << "OUTPUT\tINT\t7" << label << "\n"
                    << "OUTPUT\tDOUBLE\t0.5\nEND\t0\n";
    const auto expected = expectedRecords.str();
    const auto text = output.str();
    const auto first = text.find(expected);
    ASSERT_NE(first, std::string::npos);
    EXPECT_NE(text.find(expected, first + expected.size()), std::string::npos);
    ASSERT_EQ(value(session.sample(0, results)), 0);
    EXPECT_TRUE(results.empty());
  }
}

TEST(QIRBatchSampling, SeedReproducesBellSamples) {
  const auto ir = getProgram("BellPairStatic.ll");
  auto firstOwner =
      value(qir::JitSession::create(ir, "first", qir::Execution::Sampling, 42));
  auto& first = *firstOwner;
  auto secondOwner = value(
      qir::JitSession::create(ir, "second", qir::Execution::Sampling, 42));
  auto& second = *secondOwner;
  first.runtime().disableOutput();
  second.runtime().disableOutput();
  std::vector<std::string> a;
  std::vector<std::string> b;
  ASSERT_EQ(value(first.sample(256, a)), 0);
  ASSERT_EQ(value(second.sample(256, b)), 0);
  EXPECT_EQ(a, b);
  EXPECT_THAT(a, testing::Each(testing::AnyOf("00", "11")));
  EXPECT_THAT(a, testing::Contains("00"));
  EXPECT_THAT(a, testing::Contains("11"));
}

TEST(QIRBatchSampling, TerminalArraysPreserveProjectionAndState) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"qasm(
OPENQASM 3.0; include "stdgates.inc";
qubit[2] q; bit[3] c;
h q[0]; rz(0.6) q[0]; x q[1];
c[0] = measure q[1];
c[1] = measure q[0];
c[2] = measure q[0];
)qasm");
  MQT_ASSERT_SUCCESS(qc);
  auto qir = std::move(*qc).intoQIR(mlir::QIRProfile::Adaptive);
  MQT_ASSERT_SUCCESS(qir);
  auto ir = std::move(*qir).intoLLVMIR();
  MQT_ASSERT_SUCCESS(ir);
  auto sessionOwner = value(qir::JitSession::create(
      *ir, "terminal-arrays", qir::Execution::Sampling, 17, nullptr));
  auto& session = *sessionOwner;
  std::map<std::string, size_t> counts;
  bool retained = false;
  ASSERT_EQ(value(session.sampleCounts(32768, counts, &retained)), 0);
  ASSERT_TRUE(retained);
  ASSERT_EQ(counts.size(), 2);
  EXPECT_EQ(counts.at("100") + counts.at("111"), 32768);
  EXPECT_NEAR(static_cast<double>(counts.at("100")), 16384., 550.);
  auto state = value(session.runtime().takeState());
  const auto vector = value(dd::getVector(state.edge));
  ASSERT_EQ(vector.size(), 4);
  EXPECT_NEAR(std::abs(vector[2] - std::polar(std::sqrt(0.5), -0.3)), 0.,
              1e-12);
  EXPECT_NEAR(std::abs(vector[3] - std::polar(std::sqrt(0.5), 0.3)), 0., 1e-12);
  EXPECT_NEAR(std::abs(vector[0]) + std::abs(vector[1]), 0., 1e-12);
  state.dd->decRef(state.edge);
}

TEST(QIRBatchSampling, DeferralFallsBackForMeasuredGatesAndQubitReuse) {
  for (const bool reuse : {false, true}) {
    const auto* continuation =
        reuse ? "%next = call ptr @__quantum__rt__qubit_allocate(ptr null)\n"
                "call void @__quantum__qis__mz__body(ptr %next, ptr null)\n"
              : "call void @__quantum__qis__h__body(ptr %q)\n"
                "call void @__quantum__qis__mz__body(ptr %q, ptr null)\n";
    const std::string ir =
        R"(
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %q = call ptr @__quantum__rt__qubit_allocate(ptr null)
  call void @__quantum__qis__h__body(ptr %q)
  call void @__quantum__qis__mz__body(ptr %q, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
)" +
        std::string(reuse ? "call void @__quantum__rt__qubit_release(ptr %q)\n"
                          : "") +
        continuation + R"(
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare ptr @__quantum__rt__qubit_allocate(ptr)
declare void @__quantum__rt__qubit_release(ptr)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
    auto sessionOwner = value(qir::JitSession::create(
        ir, "deferral-fallback", qir::Execution::Sampling, 17, nullptr));
    auto& session = *sessionOwner;
    std::map<std::string, size_t> counts;
    bool retained = true;
    ASSERT_EQ(value(session.sampleCounts(32768, counts, &retained)), 0);
    EXPECT_FALSE(retained);
    EXPECT_EQ(counts.size(), reuse ? 2 : 4);
    size_t total = 0;
    for (const auto& [bits, count] : counts) {
      if (reuse) {
        EXPECT_EQ(bits.back(), '0');
      }
      EXPECT_NEAR(static_cast<double>(count), reuse ? 16384. : 8192., 550.);
      total += count;
    }
    EXPECT_EQ(total, 32768);
  }
}

TEST(QIRBatchSampling, RetainedStatePreservesPhaseOrderAndSessionLifetime) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__rz__body(double 0.6, ptr null)
  call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 1 to ptr))
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 1 to ptr))
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)
  ret i64 0
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__rz__body(double, ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" "required_num_qubits"="2" "required_num_results"="2" }
)";
  qir::Runtime::QState state;
  {
    auto sessionOwner = value(qir::JitSession::create(ir, "retained-state"));
    auto& session = *sessionOwner;
    session.runtime().disableOutput();
    std::vector<std::string> shots;
    bool available = false;
    ASSERT_EQ(value(session.sample(16, shots, &available)), 0);
    ASSERT_TRUE(available);
    state = value(session.runtime().takeState());
    ASSERT_EQ(value(session.sample(0, shots, &available)), 0);
    EXPECT_FALSE(available);
  }
  EXPECT_EQ(state.numQubits, 2);
  const auto vector = value(dd::getVector(state.edge));
  ASSERT_EQ(vector.size(), 4);
  EXPECT_NEAR(std::abs(vector[2] - std::polar(1., 0.3)), 0., 1e-12);
  EXPECT_NEAR(std::abs(vector[0]) + std::abs(vector[1]) + std::abs(vector[3]),
              0., 1e-12);
}

TEST(QIRBatchSampling, TextOutputPreservesTerminalSamplesAndState) {
  const auto ir = getProgram("BellPairStatic.ll");
  std::string outputText;
  llvm::raw_string_ostream output{outputText};
  auto sessionOwner = value(qir::JitSession::create(
      ir, "text", qir::Execution::Sampling, 42, &output));
  auto& session = *sessionOwner;
  std::vector<std::string> results;
  bool stateAvailable = false;
  ASSERT_EQ(value(session.sample(32, results, &stateAvailable)), 0);
  ASSERT_TRUE(stateAvailable);
  EXPECT_THAT(results, testing::Contains("00"));
  EXPECT_THAT(results, testing::Contains("11"));
  size_t shot = 0;
  std::string bits;
  for (auto line : llvm::split(output.str(), '\n')) {
    if (line.starts_with("OUTPUT\tRESULT\t")) {
      bits += line[std::string_view("OUTPUT\tRESULT\t").size()];
    } else if (line == "END\t0") {
      ASSERT_LT(shot, results.size());
      EXPECT_EQ(bits, results[shot++]);
      bits.clear();
    }
  }
  EXPECT_EQ(shot, results.size());
  const auto state = value(session.runtime().takeState());
  const auto vector = value(dd::getVector(state.edge));
  ASSERT_EQ(vector.size(), 4);
  EXPECT_NEAR(std::abs(vector[0]), std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(std::abs(vector[3]), std::sqrt(0.5), 1e-12);
}

TEST(QIRBatchSampling, FramesEmptyTerminalOutputs) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__tuple_record_output(i64 0, ptr null)
  ret i64 0
}
declare void @__quantum__rt__tuple_record_output(i64, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)";
  for (const bool enabled : {false, true}) {
    std::string output;
    llvm::raw_string_ostream stream(output);
    auto sessionOwner = value(
        qir::JitSession::create(ir, "empty-output", qir::Execution::Sampling,
                                42, enabled ? &stream : nullptr));
    auto& session = *sessionOwner;
    std::vector<std::string> results;
    ASSERT_EQ(
        value(session.sample(
            3, [&](std::string_view shot) { results.emplace_back(shot); })),
        0);
    EXPECT_EQ(results, std::vector<std::string>(3));
    size_t starts = 0;
    size_t ends = 0;
    for (auto line : llvm::split(output, '\n')) {
      starts += static_cast<size_t>(line == "START");
      ends += static_cast<size_t>(line == "END\t0");
    }
    EXPECT_EQ(starts, enabled ? 3U : 0U);
    EXPECT_EQ(ends, starts);
  }
}

TEST(QIRBatchSampling, ExecutesClassicalSideEffectsOnEveryShot) {
  constexpr llvm::StringRef ir = R"(
@counter = internal global i64 0
define i64 @main() #0 {
  %old = load i64, ptr @counter
  %new = add i64 %old, 1
  store i64 %new, ptr @counter
  %failed = icmp eq i64 %new, 3
  %code = zext i1 %failed to i64
  ret i64 %code
}
attributes #0 = { "entry_point" "qir_profiles"="base_profile" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "side-effects"));
  auto& session = *sessionOwner;
  EXPECT_FALSE(session.canShareCompiledCode());
  session.runtime().disableOutput();
  std::vector<std::string> results;
  qir::Runtime worker(17);
  EXPECT_FALSE(
      llvm::toString(session.sampleWithRuntime(worker, 1, results).takeError())
          .empty());
  EXPECT_EQ(value(session.sample(5, results)), 1);
  EXPECT_EQ(results.size(), 2);
  auto histogramOwner = value(qir::JitSession::create(
      ir, "histogram-side-effects", qir::Execution::Sampling, 17, nullptr));
  auto& histogram = *histogramOwner;
  std::map<std::string, size_t> counts;
  EXPECT_FALSE(histogram.canSampleBranches());
  EXPECT_EQ(value(histogram.sampleCounts(5, counts)), 1);
  EXPECT_EQ(counts, (std::map<std::string, size_t>{{"", 2}}));
}

TEST(QIRBatchSampling, HistogramIncludesUnrecordedResetAndReleaseMeasurements) {
  for (const auto* operation :
       {"__quantum__qis__reset__body", "__quantum__rt__qubit_release"}) {
    SCOPED_TRACE(operation);
    const std::string ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  %a = call ptr @__quantum__rt__qubit_allocate(ptr null)
  %b = call ptr @__quantum__rt__qubit_allocate(ptr null)
  call void @__quantum__qis__h__body(ptr %a)
  call void @__quantum__qis__cnot__body(ptr %a, ptr %b)
  call void @)" + std::string(operation) +
                           R"((ptr %a)
  call void @__quantum__qis__mz__body(ptr %b, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__int_record_output(i64 17, ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare ptr @__quantum__rt__qubit_allocate(ptr)
declare void @__quantum__rt__qubit_release(ptr)
declare void @__quantum__qis__reset__body(ptr)
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__cnot__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
declare void @__quantum__rt__int_record_output(i64, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
    std::string text;
    llvm::raw_string_ostream output(text);
    auto sessionOwner = value(qir::JitSession::create(
        ir, "hidden-measurement", qir::Execution::Sampling, 17, &output));
    auto& session = *sessionOwner;
    ASSERT_TRUE(session.canSampleBranches());
    constexpr size_t shots = 100000;
    std::map<std::string, size_t> counts;
    ASSERT_EQ(value(session.sampleCounts(shots, counts)), 0);
    EXPECT_TRUE(text.empty());
    ASSERT_EQ(counts.size(), 2);
    EXPECT_EQ(counts["0"] + counts["1"], shots);
    EXPECT_NEAR(static_cast<double>(counts["0"]), shots / 2., 1500.);
    session.runtime().seed(17);
    std::map<std::string, size_t> repeated;
    ASSERT_EQ(value(session.sampleCounts(shots, repeated)), 0);
    EXPECT_EQ(repeated, counts);
    std::vector<std::string> reference;
    ASSERT_EQ(value(session.sample(2048, reference)), 0);
    EXPECT_NEAR(static_cast<double>(std::ranges::count(reference, "0")), 1024.,
                200.);
    EXPECT_FALSE(text.empty());
  }
}

TEST(QIRBatchSampling, SharesCompiledCodeWithPrivateConcurrentRuntimes) {
  const auto ir = getProgram("BellPairAdaptive.ll");
  auto sharedOwner = value(
      qir::JitSession::create(ir, "shared", qir::Execution::Sampling, 17));
  auto& shared = *sharedOwner;
  ASSERT_TRUE(shared.canShareCompiledCode());
  std::array<std::vector<std::string>, 2> actual;
  std::array<std::future<int64_t>, 2> tasks;
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i] = std::async(std::launch::async, [&, i] {
      auto worker = value(shared.makeWorkerRuntime(21 + i));
      worker->disableOutput();
      return value(shared.sampleWithRuntime(*worker, 128, actual[i], false));
    });
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    EXPECT_EQ(tasks[i].get(), 0);
    auto referenceOwner = value(qir::JitSession::create(
        ir, "reference", qir::Execution::Sampling, 21 + i));
    auto& reference = *referenceOwner;
    reference.runtime().disableOutput();
    std::vector<std::string> expected;
    ASSERT_EQ(value(reference.sample(128, expected)), 0);
    EXPECT_EQ(actual[i], expected);
  }
}

TEST(QIRBatchSampling, ResetsProgramsWithoutInitializeBetweenShots) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  %measured = call i1 @__quantum__rt__read_result(ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  ret i64 0
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare i1 @__quantum__rt__read_result(ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "no-initialize"));
  auto& session = *sessionOwner;
  session.runtime().disableOutput();
  std::vector<std::string> results;
  ASSERT_EQ(value(session.sample(32, results)), 0);
  EXPECT_EQ(results, std::vector<std::string>(32, "1"));
}

TEST(QIRStaticResources, ExtractsDeclaredWidthIncludingUnusedQubits) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  ret i64 0
}
declare void @__quantum__qis__x__body(ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="3" }
)";
  auto sessionOwner = value(qir::JitSession::create(
      ir, "declared-width", qir::Execution::StateExtraction));
  auto& session = *sessionOwner;
  for (size_t job = 0; job < 2; ++job) {
    ASSERT_EQ(value(session.run()), 0);
    auto state = value(session.runtime().takeState());
    EXPECT_EQ(state.numQubits, 3);
    EXPECT_EQ(state.dd->qubits(), 3);
    const auto values = value(dd::getVector(state.edge));
    ASSERT_EQ(values.size(), 8);
    EXPECT_EQ(values[1], 1.);
    state.dd->decRef(state.edge);
  }
  std::vector<std::string> results;
  EXPECT_FALSE(llvm::toString(session.sample(1, results).takeError()).empty());
}

TEST(QIRStaticResources, RejectsQubitCapacityBeyondDDRange) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "required_num_qubits"="65537" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "excess-qubit-capacity").takeError())
          .empty());
}

TEST(QIRStaticResources, RejectsMalformedResourceCapacities) {
  for (const auto* attribute : {
           R"("required_num_qubits"="-1")",
           R"("required_num_results"="18446744073709551616")",
       }) {
    SCOPED_TRACE(attribute);
    const std::string ir = std::string("define i64 @main() #0 { ret i64 0 }\n"
                                       "attributes #0 = { \"entry_point\" ") +
                           attribute + " }";
    EXPECT_FALSE(
        llvm::toString(
            qir::JitSession::create(ir, "invalid-capacity").takeError())
            .empty());
  }
}

TEST(QIRStaticResources, ConfiguresRuntimeResourceBounds) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "required_num_qubits"="1" "required_num_results"="1" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "resource-bounds"));
  auto& session = *sessionOwner;
  auto& runtime = session.runtime();
  Qubit* zeroQubit = nullptr;
  Result* zeroResult = nullptr;
  value(runtime.measure(zeroQubit, zeroResult));
  EXPECT_FALSE(
      llvm::toString(
          runtime.measure(reinterpret_cast<Qubit*>(uintptr_t{1}), zeroResult))
          .empty());
  EXPECT_FALSE(
      llvm::toString(
          runtime.measure(zeroQubit, reinterpret_cast<Result*>(uintptr_t{1})))
          .empty());
}

TEST(QIRJIT, ResolvesProcessSymbolsWithDefaultGenerator) {
  constexpr llvm::StringRef ir = R"(
@text = private constant [5 x i8] c"test\00"
define i64 @main() #0 {
  %length = call i64 @strlen(ptr @text)
  ret i64 %length
}
declare i64 @strlen(ptr)
attributes #0 = { "entry_point" }
)";
  auto sessionOwner = value(qir::JitSession::create(ir, "process-symbol"));
  auto& session = *sessionOwner;
  EXPECT_EQ(value(session.run()), 4);
}

TEST(QIRJIT, RejectsTargetIncompatibleWithInProcessExecution) {
  constexpr llvm::StringRef ir = R"(
target triple = "wasm32-unknown-unknown"
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_FALSE(
      llvm::toString(
          qir::JitSession::create(ir, "incompatible-target").takeError())
          .empty());
}

TEST(QIRBatchSampling, PreservesWideSeededOutputMappings) {
  constexpr size_t width = 64;
  std::vector<std::string> reference;
  for (size_t mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<size_t> outputs;
    if (mode == 2) {
      outputs = {0, 5, 0, 2, 63};
    } else {
      for (size_t q = 0; q < width; ++q) {
        outputs.push_back(mode == 1 ? width - 1 - q : q);
      }
    }
    const auto pointer = [](size_t q) {
      return "ptr inttoptr (i64 " + std::to_string(q) + " to ptr)";
    };
    std::string irStorage;
    llvm::raw_string_ostream ir(irStorage);
    ir << "define i64 @main() #0 {\n";
    for (const size_t q : {0, 5, 63}) {
      ir << "call void @__quantum__qis__x__body(" << pointer(q) << ")\n";
    }
    ir << "call void @__quantum__qis__h__body(" << pointer(2) << ")\n";
    if (mode == 3) {
      ir << "call void @__quantum__qis__swap__body(" << pointer(0) << ", "
         << pointer(10) << ")\n";
    }
    for (size_t q = 0; q < width; ++q) {
      ir << "call void @__quantum__qis__mz__body(" << pointer(q) << ", "
         << pointer(q) << ")\n";
    }
    for (const auto q : outputs) {
      ir << "call void @__quantum__rt__result_record_output(" << pointer(q)
         << ", ptr null)\n";
    }
    ir << R"(ret i64 0
}
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" "required_num_qubits"="64" "required_num_results"="64" }
)";
    auto sessionOwner = value(qir::JitSession::create(
        ir.str(), "wide-output", qir::Execution::Sampling, 42));
    auto& session = *sessionOwner;
    session.runtime().disableOutput();
    std::vector<std::string> shots;
    ASSERT_EQ(value(session.sample(32, shots)), 0);
    ASSERT_EQ(shots.size(), 32);
    EXPECT_EQ(session.runtime().getMeasurements(), shots.back());
    if (mode == 0) {
      reference = shots;
    }
    for (size_t shot = 0; shot < shots.size(); ++shot) {
      ASSERT_EQ(shots[shot].size(), outputs.size());
      for (size_t bit = 0; bit < outputs.size(); ++bit) {
        auto q = outputs[bit];
        if (mode == 3 && (q == 0 || q == 10)) {
          q = 10 - q;
        }
        EXPECT_EQ(shots[shot][bit], reference[shot][q]);
        if (q != 2) {
          EXPECT_EQ(shots[shot][bit], q == 0 || q == 5 || q == 63 ? '1' : '0');
        }
      }
    }
    session.runtime().seed(42);
    std::vector<std::string> repeated;
    ASSERT_EQ(
        value(session.sample(
            32, [&](std::string_view shot) { repeated.emplace_back(shot); })),
        0);
    EXPECT_EQ(repeated, shots);
  }
}

TEST(QIRBatchSampling, FallsBackForConstantBooleanOutputs) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__rt__bool_record_output(i1 false, ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__bool_record_output(i1 true, ptr null)
  ret i64 0
}
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
declare void @__quantum__rt__bool_record_output(i1, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="1" "required_num_results"="1" }
)";
  std::string outputText;
  llvm::raw_string_ostream output{outputText};
  auto sessionOwner = value(
      qir::JitSession::create(ir, "boolean-output", qir::Execution::Sampling));
  auto& session = *sessionOwner;
  EXPECT_EQ(session.quantumCallSites(), 1U);
  session.runtime().disableOutput();
  std::vector<std::string> shots;
  bool available = false;
  ASSERT_EQ(value(session.sample(4, shots, &available)), 0);
  EXPECT_FALSE(available);
  EXPECT_EQ(shots, (std::vector<std::string>{"011", "011", "011", "011"}));
  EXPECT_EQ(session.runtime().getMeasurements(), shots.back());

  session.runtime().setOstream(output);
  ASSERT_EQ(value(session.sample(4, shots, &available)), 0);
  EXPECT_FALSE(available);
  EXPECT_EQ(shots, (std::vector<std::string>{"011", "011", "011", "011"}));
  EXPECT_FALSE(output.str().empty());
}

TEST(QIRBatchSampling, FallsBackForConstantBooleansWithoutQubits) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__bool_record_output(i1 true, ptr null)
  call void @__quantum__rt__bool_record_output(i1 false, ptr null)
  ret i64 0
}
declare void @__quantum__rt__bool_record_output(i1, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="0" "required_num_results"="0" }
)";
  auto sessionOwner = value(
      qir::JitSession::create(ir, "constant-output", qir::Execution::Sampling));
  auto& session = *sessionOwner;
  session.runtime().disableOutput();
  std::vector<std::string> shots;
  bool available = false;
  ASSERT_EQ(value(session.sample(3, shots, &available)), 0);
  EXPECT_FALSE(available);
  EXPECT_EQ(shots, (std::vector<std::string>{"10", "10", "10"}));
  EXPECT_EQ(session.runtime().getMeasurements(), shots.back());
}

TEST(QIRStaticResources, EmptyStatesKeepZeroCapacityAfterTransfer) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="0" }
)";
  auto sessionOwner = value(qir::JitSession::create(
      ir, "empty-state", qir::Execution::StateExtraction));
  auto& session = *sessionOwner;
  for (size_t job = 0; job < 2; ++job) {
    ASSERT_EQ(value(session.run()), 0);
    auto state = value(session.runtime().takeState());
    ASSERT_NE(state.dd, nullptr);
    EXPECT_EQ(state.dd->qubits(), 0);
    EXPECT_EQ(state.numQubits, 0);
    EXPECT_TRUE(state.edge.isOneTerminal());
  }
}

TEST(QIRAdaptiveStatevector,
     RejectsOperationsOnMeasuredWiresAfterReturningFromJIT) {
  for (const auto* operation : {
           "call void @__quantum__qis__x__body(ptr null)",
           "call void @__quantum__qis__cx__body(ptr null, ptr inttoptr (i64 1 "
           "to ptr))",
           "call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 "
           "1 to ptr))",
           "call void @helper(ptr null)",
       }) {
    SCOPED_TRACE(operation);
    const auto ir = std::string(R"(
define i64 @main() #0 {
  call void @__quantum__qis__h__body(ptr null)
  call void @__quantum__qis__mz__body(ptr null, ptr null)
)") + operation + R"(
  ret i64 0
}
define void @helper(ptr %q) {
  call void @__quantum__qis__x__body(ptr %q)
  ret void
}
declare void @__quantum__qis__h__body(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__cx__body(ptr, ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
    auto sessionOwner = value(qir::JitSession::create(
        ir, "non-terminal", qir::Execution::StateExtraction));
    auto& session = *sessionOwner;
    EXPECT_FALSE(llvm::toString(session.run().takeError()).empty());
    EXPECT_FALSE(
        llvm::toString(session.runtime().takeState().takeError()).empty());
  }
}

TEST(QIRAdaptiveStatevector,
     RejectsFeedbackResetAndUnknownEffectsBeforeExecution) {
  for (const auto* body : {
           R"(%r = call i1 @__quantum__rt__read_result(ptr null)
br i1 %r, label %left, label %right
left: ret i64 0
right: ret i64 0)",
           "call void @__quantum__qis__reset__body(ptr null)\nret i64 0",
           "call void @feedback()\nret i64 0",
           "%f = load ptr, ptr @function\ncall void %f()\nret i64 0",
           "call void @external()\nret i64 0",
           "%code = call i64 @main()\nret i64 %code",
           "call void @__quantum__rt__initialize(ptr null)\nret i64 0",
       }) {
    SCOPED_TRACE(body);
    const auto ir = std::string(R"(
@function = global ptr @external
define i64 @main() #0 {
  call void @__quantum__qis__mz__body(ptr null, ptr null)
)") + body + R"(
}
define void @feedback() {
  %r = call i1 @__quantum__rt__read_result(ptr null)
  br i1 %r, label %left, label %right
left: ret void
right: ret void
}
declare void @external()
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__reset__body(ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare i1 @__quantum__rt__read_result(ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
    EXPECT_FALSE(
        llvm::toString(qir::JitSession::create(ir, "unsupported",
                                               qir::Execution::StateExtraction)
                           .takeError())
            .empty());
  }
}

TEST(QIRAdaptiveStatevector, PreservesExitCodesAndResetsValidationBetweenRuns) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  br i1 false, label %unused, label %exit
unused:
  %q = call ptr @__quantum__rt__qubit_allocate(ptr null)
  br label %exit
exit:
  ret i64 7
}
declare ptr @__quantum__rt__qubit_allocate(ptr)
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  auto sessionOwner = value(qir::JitSession::create(
      ir, "unused-allocation", qir::Execution::StateExtraction));
  auto& session = *sessionOwner;
  ASSERT_EQ(value(session.run()), 7);
  auto state = value(session.runtime().takeState());
  EXPECT_EQ(state.numQubits, 0);
  EXPECT_TRUE(state.edge.isOneTerminal());
  const std::array<Qubit*, 1> qubits{nullptr};
  value(session.runtime().reset(qubits));
  EXPECT_FALSE(
      llvm::toString(session.runtime().takeState().takeError()).empty());
  ASSERT_EQ(value(session.run()), 7);
  value(session.runtime().takeState());
}
