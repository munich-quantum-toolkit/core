/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/JIT/Session.h"
#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "qir/helpers/test_utils.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

class JitSessionTest : public testing::Test {
protected:
  std::ostringstream sink;
};

TEST_F(JitSessionTest, LoadModuleFromMemory) {
  const auto program = qir_test::getProgram("BellPairStatic.ll");
  qir::JitSession session(program, "BellPairStatic.ll");
  session.runtime().setOstream(sink);
  ASSERT_EQ(session.run(), 0);
  EXPECT_FALSE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, SamplingRecordsOutputs) {
  const auto program = qir_test::getProgram("BellPairStatic.ll");
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
  const auto program = qir_test::getProgram("BellPairStatic.ll");
  qir::JitSession session(program, "BellPairStatic.ll",
                          qir::Execution::StateExtraction);
  session.runtime().setOstream(sink);
  ASSERT_EQ(session.run(), 0);
  EXPECT_TRUE(session.runtime().getMeasurements().empty());
}

TEST_F(JitSessionTest, StateExtractionRejectsAdaptiveProfile) {
  constexpr std::string_view ir = R"(
define i64 @main() #0 { ret i64 0 }
attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
)";
  EXPECT_THROW(
      {
        try {
          const qir::JitSession session(ir, "Adaptive.ll",
                                        qir::Execution::StateExtraction);
        } catch (const std::invalid_argument& error) {
          EXPECT_THAT(error.what(), ::testing::HasSubstr("Base Profile"));
          throw;
        }
      },
      std::invalid_argument);
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
  const auto program = qir_test::getProgram("BellPairStatic.ll");
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
  const auto program = qir_test::getProgram("BellPairStatic.ll");
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
