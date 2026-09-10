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
#include "mlir/Dialect/QIR/Execution/JIT/Session.h"
#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

static std::string getProgram(const std::string_view file) {
  const auto path = std::filesystem::path(QIR_FILES_DIR) / file;
  std::ifstream stream(path);
  EXPECT_TRUE(stream.is_open()) << "Failed to open " << path;
  return {std::istreambuf_iterator<char>{stream}, {}};
}

namespace {

class JitSessionTest : public testing::Test {
protected:
  std::ostringstream sink;
};

TEST_F(JitSessionTest, LoadModuleFromMemory) {
  const auto program = getProgram("BellPairStatic.ll");
  qir::JitSession session(program, "BellPairStatic.ll");
  session.runtime().setOstream(sink);
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, SamplingRecordsOutputs) {
  const auto program = getProgram("BellPairStatic.ll");
  // qir::Execution::Sampling is the default Execution mode
  qir::JitSession session(program, "BellPairStatic.ll");
  session.runtime().setOstream(sink);
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(session.runtime().getMeasurements().empty());
  session.runtime().outputShotStart();
  EXPECT_THAT(sink.str(), ::testing::HasSubstr("METADATA\tentry_point\n"));
  EXPECT_THAT(sink.str(),
              ::testing::HasSubstr("METADATA\tqir_profiles\tbase_profile\n"));
}

TEST_F(JitSessionTest, StateExtractionLeavesNoRecordedOutputs) {
  const auto program = getProgram("BellPairStatic.ll");
  qir::JitSession session(program, "BellPairStatic.ll",
                          qir::Execution::StateExtraction);
  session.runtime().setOstream(sink);
  ASSERT_EQ(session.run(), 0);
  EXPECT_TRUE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, StateExtractionSupportsAdaptiveControlAndLifetimes) {
  const auto ir = getProgram("StatevectorAdaptive.ll");
  qir::JitSession session(ir, "Adaptive.ll", qir::Execution::StateExtraction);
  session.runtime().setOstream(sink);
  for (size_t run = 0; run < 2; ++run) {
    ASSERT_EQ(session.run(), 0);
    EXPECT_TRUE(session.runtime().getMeasurements().empty());
    EXPECT_TRUE(sink.str().empty());
    auto state = session.runtime().takeState();
    EXPECT_EQ(state.numQubits, 4);
    const auto values = state.edge.getVector();
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
  EXPECT_THROW(
      {
        try {
          const qir::JitSession session(ir, "NonTerminal.ll",
                                        qir::Execution::StateExtraction);
        } catch (const std::invalid_argument& error) {
          EXPECT_THAT(error.what(), ::testing::HasSubstr("terminal region"));
          throw;
        }
      },
      std::invalid_argument);
}

TEST_F(JitSessionTest, OutputSchemaDefaultsToLabeledWhenAttributeAbsent) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  qir::JitSession session(ir, "NoOutputSchema.ll");
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Labeled);
}

TEST_F(JitSessionTest, OutputSchemaFromLabeledAttribute) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "output_labeling_schema"="labeled" }
)";
  qir::JitSession session(ir, "LabeledOutputSchema.ll");
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Labeled);
}

TEST_F(JitSessionTest, OutputSchemaFromOrderedAttribute) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "output_labeling_schema"="ordered" }
)";
  qir::JitSession session(ir, "OrderedOutputSchema.ll");
  EXPECT_EQ(session.runtime().getOutputSchema(),
            qir::Runtime::OutputSchema::Ordered);
}

TEST_F(JitSessionTest, ExecutesArbitrarilyNamedEntryPoint) {
  constexpr std::string_view ir = R"(
define i64 @bell_entry() #0 { ret i64 7 }
  attributes #0 = { "entry_point" }
)";
  qir::JitSession session(ir, "NamedEntry.ll");
  EXPECT_EQ(session.run(), 7);
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
  qir::JitSession session(ir, "Qir21Resources.ll");
  session.runtime().setOstream(sink);
  EXPECT_EQ(session.run(), 0);
  EXPECT_EQ(session.runtime().getMeasurements(), "1");
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
  qir::JitSession session(ir, "NativeControls.ll");
  session.runtime().setOstream(sink);
  EXPECT_EQ(session.run(), 0);
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
  EXPECT_THROW(qir::JitSession(ir, "LegacyAllocation.ll"), std::runtime_error);
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
  qir::JitSession session(ir, "ControlledRotation.ll");
  session.runtime().setOstream(sink);
  EXPECT_EQ(session.run(), 0);
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
  EXPECT_THROW(qir::JitSession(ir, "QirRunnerPauli.ll"), std::runtime_error);
}

TEST_F(JitSessionTest, SessionsExecuteIndependently) {
  const auto program = getProgram("BellPairStatic.ll");
  qir::JitSession first(program, "first.ll");
  qir::JitSession second(program, "second.ll");
  std::ostringstream firstSink;
  std::ostringstream secondSink;
  first.runtime().setOstream(firstSink);
  second.runtime().setOstream(secondSink);
  int64_t firstExit = -1;
  int64_t secondExit = -1;

  std::thread firstThread([&] { firstExit = first.run(); });
  std::thread secondThread([&] { secondExit = second.run(); });
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
  qir::JitSession first(program, "first.ll");
  qir::JitSession second(program, "second.ll");
  first.runtime().seed(42);
  second.runtime().seed(42);
  first.runtime().setOstream(sink);
  second.runtime().setOstream(sink);
  std::string firstSequence;
  std::string secondSequence;

  for (std::size_t shot = 0; shot < 16; ++shot) {
    ASSERT_EQ(first.run(), 0);
    ASSERT_EQ(second.run(), 0);
    firstSequence += first.runtime().getMeasurements().front();
    secondSequence += second.runtime().getMeasurements().front();
  }

  EXPECT_EQ(firstSequence, secondSequence);
  EXPECT_THAT(firstSequence, ::testing::HasSubstr("0"));
  EXPECT_THAT(firstSequence, ::testing::HasSubstr("1"));
}

TEST(JitSessionErrors, MalformedIRThrows) {
  constexpr std::string_view ir = R"(define i32 @main() {})";
  EXPECT_THROW(qir::JitSession(ir, "MalformedIR.ll"), std::runtime_error);
}

TEST(JitSessionErrors, RejectsNonCompliantEntryPointSignature) {
  constexpr std::string_view ir = R"(
define i32 @main() #0 { ret i32 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_THROW(qir::JitSession(ir, "BadEntrySignature.ll"), std::runtime_error);
}

TEST(JitSessionErrors, RejectsMultipleEntryPoints) {
  constexpr std::string_view ir = R"(
define i64 @first() #0 { ret i64 0 }
define i64 @second() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_THROW(qir::JitSession(ir, "MultipleEntries.ll"), std::runtime_error);
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
  EXPECT_THROW(qir::JitSession(ir, "BadRuntimeSignature.ll"),
               std::runtime_error);
}

} // namespace

TEST(QIRBatchSampling, PreservesLogicalOutputOrderAndRepeatedMeasurements) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 {
  call void @__quantum__rt__initialize(ptr null)
  call void @__quantum__qis__x__body(ptr null)
  call void @__quantum__qis__swap__body(ptr null, ptr inttoptr (i64 2 to ptr))
  call void @__quantum__qis__mz__body(ptr null, ptr null)
  call void @__quantum__qis__mz__body(ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 1 to ptr))
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)
  call void @__quantum__rt__result_record_output(ptr null, ptr null)
  call void @__quantum__rt__result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)
  ret i64 0
}
declare void @__quantum__rt__initialize(ptr)
declare void @__quantum__qis__x__body(ptr)
declare void @__quantum__qis__swap__body(ptr, ptr)
declare void @__quantum__qis__mz__body(ptr, ptr)
declare void @__quantum__rt__result_record_output(ptr, ptr)
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="3" "required_num_results"="2" }
)";
  qir::JitSession session(ir, "output-order");
  session.runtime().disableOutput();
  std::vector<std::string> results;
  ASSERT_EQ(session.sample(32, results), 0);
  EXPECT_EQ(results, std::vector<std::string>(32, "101"));
  ASSERT_EQ(session.sample(2, results), 0);
  EXPECT_EQ(results, std::vector<std::string>(2, "101"));
  ASSERT_EQ(session.sample(0, results), 0);
  EXPECT_TRUE(results.empty());
}

TEST(QIRBatchSampling, SeedReproducesBellSamples) {
  const auto ir = getProgram("BellPairStatic.ll");
  qir::JitSession first(ir, "first", qir::Execution::Sampling, 42);
  qir::JitSession second(ir, "second", qir::Execution::Sampling, 42);
  first.runtime().disableOutput();
  second.runtime().disableOutput();
  std::vector<std::string> a;
  std::vector<std::string> b;
  ASSERT_EQ(first.sample(256, a), 0);
  ASSERT_EQ(second.sample(256, b), 0);
  EXPECT_EQ(a, b);
  EXPECT_THAT(a, testing::Each(testing::AnyOf("00", "11")));
  EXPECT_THAT(a, testing::Contains("00"));
  EXPECT_THAT(a, testing::Contains("11"));
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
    qir::JitSession session(ir, "retained-state");
    session.runtime().disableOutput();
    std::vector<std::string> shots;
    bool available = false;
    ASSERT_EQ(session.sample(16, shots, &available), 0);
    ASSERT_TRUE(available);
    state = session.runtime().takeState();
    ASSERT_EQ(session.sample(0, shots, &available), 0);
    EXPECT_FALSE(available);
  }
  EXPECT_EQ(state.numQubits, 2);
  const auto vector = state.edge.getVector();
  ASSERT_EQ(vector.size(), 4);
  EXPECT_NEAR(std::abs(vector[2] - std::polar(1., 0.3)), 0., 1e-12);
  EXPECT_NEAR(std::abs(vector[0]) + std::abs(vector[1]) + std::abs(vector[3]),
              0., 1e-12);
}

TEST(QIRBatchSampling, TextOutputKeepsPerShotRecords) {
  const auto ir = getProgram("BellPairStatic.ll");
  qir::JitSession session(ir, "text", qir::Execution::Sampling, 42);
  std::ostringstream output;
  session.runtime().setOstream(output);
  std::vector<std::string> results;
  ASSERT_EQ(session.sample(4, results), 0);
  EXPECT_EQ(results.size(), 4);
  const auto text = output.str();
  size_t ends = 0;
  for (size_t pos = text.find("END\t0\n"); pos != std::string::npos;
       pos = text.find("END\t0\n", pos + 1)) {
    ++ends;
  }
  EXPECT_EQ(ends, 4);
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
  qir::JitSession session(ir, "side-effects");
  session.runtime().disableOutput();
  std::vector<std::string> results;
  EXPECT_EQ(session.sample(5, results), 1);
  EXPECT_EQ(results.size(), 2);
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
  qir::JitSession session(ir, "no-initialize");
  session.runtime().disableOutput();
  std::vector<std::string> results;
  ASSERT_EQ(session.sample(32, results), 0);
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
  qir::JitSession session(ir, "declared-width",
                          qir::Execution::StateExtraction);
  for (size_t job = 0; job < 2; ++job) {
    ASSERT_EQ(session.run(), 0);
    auto state = session.runtime().takeState();
    EXPECT_EQ(state.numQubits, 3);
    EXPECT_EQ(state.dd->qubits(), 3);
    const auto values = state.edge.getVector();
    ASSERT_EQ(values.size(), 8);
    EXPECT_EQ(values[1], 1.);
    state.dd->decRef(state.edge);
  }
  std::vector<std::string> results;
  EXPECT_THROW(session.sample(1, results), std::logic_error);
}

TEST(QIRStaticResources, RejectsQubitCapacityBeyondDDRange) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "required_num_qubits"="65537" }
)";
  EXPECT_THROW(qir::JitSession(ir, "excess-qubit-capacity"), std::out_of_range);
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
    EXPECT_THROW(qir::JitSession(ir, "invalid-capacity"),
                 std::invalid_argument);
  }
}

TEST(QIRStaticResources, ConfiguresRuntimeResourceBounds) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "required_num_qubits"="1" "required_num_results"="1" }
)";
  qir::JitSession session(ir, "resource-bounds");
  auto& runtime = session.runtime();
  Qubit* zeroQubit = nullptr;
  Result* zeroResult = nullptr;
  EXPECT_NO_THROW(runtime.measure(zeroQubit, zeroResult));
  EXPECT_THROW(
      runtime.measure(reinterpret_cast<Qubit*>(uintptr_t{1}), zeroResult),
      std::out_of_range);
  EXPECT_THROW(
      runtime.measure(zeroQubit, reinterpret_cast<Result*>(uintptr_t{1})),
      std::out_of_range);
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
  qir::JitSession session(ir, "process-symbol");
  EXPECT_EQ(session.run(), 4);
}

TEST(QIRJIT, RejectsTargetIncompatibleWithInProcessExecution) {
  constexpr llvm::StringRef ir = R"(
target triple = "wasm32-unknown-unknown"
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" }
)";
  EXPECT_THROW(qir::JitSession(ir, "incompatible-target"),
               std::invalid_argument);
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
    std::ostringstream ir;
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
    qir::JitSession session(ir.str(), "wide-output", qir::Execution::Sampling,
                            42);
    session.runtime().disableOutput();
    std::vector<std::string> shots;
    ASSERT_EQ(session.sample(32, shots), 0);
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
    ASSERT_EQ(session.sample(32, repeated), 0);
    EXPECT_EQ(repeated, shots);
  }
}

TEST(QIRStaticResources, EmptyStatesKeepZeroCapacityAfterTransfer) {
  constexpr llvm::StringRef ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "qir_profiles"="base_profile" "required_num_qubits"="0" }
)";
  qir::JitSession session(ir, "empty-state", qir::Execution::StateExtraction);
  for (size_t job = 0; job < 2; ++job) {
    ASSERT_EQ(session.run(), 0);
    auto state = session.runtime().takeState();
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
    qir::JitSession session(ir, "non-terminal",
                            qir::Execution::StateExtraction);
    EXPECT_THROW(session.run(), std::invalid_argument);
    EXPECT_THROW(session.runtime().takeState(), std::invalid_argument);
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
    EXPECT_THROW(
        qir::JitSession(ir, "unsupported", qir::Execution::StateExtraction),
        std::invalid_argument);
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
  qir::JitSession session(ir, "unused-allocation",
                          qir::Execution::StateExtraction);
  ASSERT_EQ(session.run(), 7);
  auto state = session.runtime().takeState();
  EXPECT_EQ(state.numQubits, 0);
  EXPECT_TRUE(state.edge.isOneTerminal());
  const std::array<Qubit*, 1> qubits{nullptr};
  session.runtime().reset(qubits);
  EXPECT_THROW(session.runtime().takeState(), std::invalid_argument);
  ASSERT_EQ(session.run(), 7);
  EXPECT_NO_THROW(session.runtime().takeState());
}
