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
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define SYSTEM _wsystem
#else
#define SYSTEM std::system
#endif

namespace qir {

namespace {

class QIRRuntimeTest : public testing::Test {
protected:
  std::ostringstream sink;
  void SetUp() override { Runtime::getInstance().setOstream(sink); }
  void TearDown() override {
    Runtime::getInstance().resetOstream();
    Runtime::getInstance().setOutputSchema(Runtime::OutputSchema::Labeled);
  }
};

TEST(QIRRuntimeArgumentsTest, RejectsInvalidArrayDimensions) {
  EXPECT_DEATH(__quantum__rt__array_create_1d(0, 1), ".+");
  EXPECT_DEATH(__quantum__rt__array_create_1d(sizeof(Qubit*), -1), ".+");
  EXPECT_DEATH(
      __quantum__rt__array_create_1d(2, std::numeric_limits<int64_t>::max()),
      ".+");
}

TEST(QIRRuntimeArgumentsTest, RejectsInvalidTupleSizes) {
  EXPECT_DEATH(__quantum__rt__tuple_create(-1), ".+");
  EXPECT_DEATH(__quantum__rt__tuple_create(std::numeric_limits<int64_t>::max()),
               ".+");

  auto* controls = __quantum__rt__array_create_1d(sizeof(Qubit*), 0);
  auto* tuple = __quantum__rt__tuple_create(1);
  EXPECT_DEATH(__quantum__qis__rx__ctl(controls, tuple), ".+");
  __quantum__rt__tuple_update_reference_count(tuple, -1);
  __quantum__rt__array_update_reference_count(controls, -1);
}

TEST_F(QIRRuntimeTest, RejectsInvalidDynamicResourceUse) {
  __quantum__rt__initialize(nullptr);
  auto* qubit = __quantum__rt__qubit_allocate(nullptr);
  auto* result = __quantum__rt__result_allocate(nullptr);
  __quantum__rt__qubit_release(qubit);
  __quantum__rt__result_release(result);

  EXPECT_DEATH(__quantum__qis__x__body(qubit), ".+");
  EXPECT_DEATH(__quantum__rt__read_result(result), ".+");
  EXPECT_DEATH(__quantum__rt__qubit_release(qubit), ".+");
  EXPECT_DEATH(__quantum__rt__result_release(result), ".+");
}

TEST_F(QIRRuntimeTest, RejectsMixedStaticAndDynamicResourceManagement) {
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(nullptr);
  EXPECT_DEATH(__quantum__rt__qubit_allocate(nullptr), ".+");

  __quantum__rt__initialize(nullptr);
  __quantum__qis__mz__body(nullptr, nullptr);
  EXPECT_DEATH(__quantum__rt__result_allocate(nullptr), ".+");
}

TEST_F(QIRRuntimeTest, ReportsInvalidControlArraysAndTuples) {
  __quantum__rt__initialize(nullptr);
  EXPECT_DEATH(__quantum__qis__x__ctl(nullptr, nullptr),
               "control array must not be null");
  auto* controls = __quantum__rt__array_create_1d(1, 0);
  EXPECT_DEATH(__quantum__qis__x__ctl(controls, nullptr), "qubit pointers");
  EXPECT_DEATH(__quantum__qis__rx__ctl(controls, nullptr),
               "tuple must not be null");
  auto* tuple = __quantum__rt__tuple_create(sizeof(double) + sizeof(Qubit*));
  EXPECT_DEATH(__quantum__qis__rx__ctl(nullptr, tuple),
               "control array must not be null");
  EXPECT_DEATH(__quantum__rt__array_get_size_1d(nullptr),
               "array must not be null");
  __quantum__rt__initialize(nullptr);
}

TEST_F(QIRRuntimeTest, ReportsInvalidBulkResourceArguments) {
  __quantum__rt__initialize(nullptr);
  auto& runtime = Runtime::getInstance();
  for (const int64_t size : {-1, 1}) {
    bool error = false;
    __quantum__rt__qubit_array_allocate(size, nullptr, &error);
    EXPECT_TRUE(error);

    error = false;
    __quantum__rt__result_array_allocate(size, nullptr, &error);
    EXPECT_TRUE(error);

    EXPECT_DEATH(__quantum__rt__result_array_allocate(size, nullptr, nullptr),
                 "resource array allocation");
    EXPECT_DEATH(__quantum__rt__qubit_array_release(size, nullptr),
                 "resource array release");
    EXPECT_DEATH(
        __quantum__rt__result_array_record_output(size, nullptr, nullptr),
        "result array output");
  }
  std::array<Result*, 1> results{};
  bool error = true;
  __quantum__rt__result_array_allocate(1, results.data(), &error);
  EXPECT_FALSE(error);
  ASSERT_NE(results[0], nullptr);
  EXPECT_FALSE(__quantum__rt__read_result(results[0]));
  __quantum__rt__result_array_release(1, results.data());
  EXPECT_DEATH(__quantum__rt__result_array_release(1, results.data()),
               "result");
  EXPECT_DEATH(
      __quantum__rt__result_array_record_output(1, results.data(), nullptr),
      "Result not allocated");
  EXPECT_DEATH(__quantum__rt__result_record_output(results[0], nullptr),
               "Result not allocated");
  EXPECT_TRUE(runtime.getMeasurements().empty());
}

TEST_F(QIRRuntimeTest, RollsBackFailedBulkQubitAllocation) {
  __quantum__rt__initialize(nullptr);
  std::vector<Qubit*> qubits(dd::Package::MAX_POSSIBLE_QUBITS + 1);
  bool error = false;
  __quantum__rt__qubit_array_allocate(static_cast<int64_t>(qubits.size()),
                                      qubits.data(), &error);
  EXPECT_TRUE(error);

  EXPECT_TRUE(std::ranges::all_of(
      qubits, [](const auto* qubit) { return qubit == nullptr; }));
  auto* qubit = __quantum__rt__qubit_allocate(&error);
  EXPECT_FALSE(error);
  ASSERT_NE(qubit, nullptr);
  __quantum__rt__qubit_release(qubit);

  __quantum__rt__initialize(nullptr);
}

TEST_F(QIRRuntimeTest, RejectsStaticQubitBeyondDDRange) {
  __quantum__rt__initialize(nullptr);
  auto* qubit = reinterpret_cast<Qubit*>(dd::Package::MAX_POSSIBLE_QUBITS);
  EXPECT_DEATH(__quantum__qis__x__body(qubit), ".+");

  Runtime runtime{0};
  const auto matrix = mlir::qco::XOp::getUnitaryMatrix();
  ::mqt::test::value(runtime.apply(matrix, {}, std::array<Qubit*, 1>{nullptr}));
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return runtime.apply(matrix, {}, std::array{qubit}); }),
            ::mqt::ErrorCategory::OutOfRange);
  ::mqt::test::value(runtime.measure(static_cast<Qubit*>(nullptr),
                                     static_cast<Result*>(nullptr)));
  EXPECT_TRUE(::mqt::test::value(runtime.deref(nullptr))->r);
}

TEST_F(QIRRuntimeTest, RejectsDynamicQubitBeyondDDRange) {
  Runtime runtime{0};
  for (size_t i = 0; i < dd::Package::MAX_POSSIBLE_QUBITS; ++i) {
    static_cast<void>(::mqt::test::value(runtime.qAlloc()));
  }
  EXPECT_FALSE(
      ::mqt::test::errorMessage([&] { return runtime.qAlloc(); }).empty());
}

} // namespace

/// HEADER/START/METADATA/END frame the per-shot OUTPUT block.
/// The default Labeled schema emits `labeled` in HEADER and METADATA.
TEST_F(QIRRuntimeTest, OutputFraming) {
  auto& runtime = Runtime::getInstance();
  ::mqt::test::value(runtime.outputProgramHeader());
  ::mqt::test::value(runtime.outputShotStart());
  ::mqt::test::value(runtime.outputShotEnd());
  std::ostringstream expected;
  expected << "HEADER\tschema_id\tlabeled\n"
           << "HEADER\tschema_version\t2.1\n"
           << "START\n"
           << "METADATA\toutput_labeling_schema\tlabeled\n"
           << "END\t0\n";
  EXPECT_EQ(sink.str(), expected.str());
}

// In Labeled mode:
// - the HEADER announces `labeled`,
// - the per-shot METADATA line matches the output schema, and
// - OUTPUT records carry the label column.
TEST_F(QIRRuntimeTest, OutputFramingLabeled) {
  auto& runtime = Runtime::getInstance();
  ::mqt::test::value(runtime.outputProgramHeader());
  ::mqt::test::value(runtime.outputShotStart());
  ::mqt::test::value(runtime.outputBool(true, "bool_label"));
  ::mqt::test::value(runtime.outputInt(42, "int_label"));
  ::mqt::test::value(runtime.outputFloat(3.14, "double_label"));
  ::mqt::test::value(runtime.outputTuple(2, "tuple_label"));
  ::mqt::test::value(runtime.outputArray(3, "array_label"));
  ::mqt::test::value(runtime.outputShotEnd());
  std::ostringstream expected;
  expected << "HEADER\tschema_id\tlabeled\n"
           << "HEADER\tschema_version\t2.1\n"
           << "START\n"
           << "METADATA\toutput_labeling_schema\tlabeled\n"
           << "OUTPUT\tBOOL\ttrue\tbool_label\n"
           << "OUTPUT\tINT\t42\tint_label\n"
           << "OUTPUT\tDOUBLE\t3.14\tdouble_label\n"
           << "OUTPUT\tTUPLE\t2\ttuple_label\n"
           << "OUTPUT\tARRAY\t3\tarray_label\n"
           << "END\t0\n";
  EXPECT_EQ(sink.str(), expected.str());
}

// In Ordered mode:
// - the HEADER announces `ordered`,
// - the per-shot METADATA line matches the output schema, and
// - OUTPUT records drop the label column.
TEST_F(QIRRuntimeTest, OutputFramingOrdered) {
  auto& runtime = Runtime::getInstance();
  runtime.setOutputSchema(Runtime::OutputSchema::Ordered);
  ::mqt::test::value(runtime.outputProgramHeader());
  ::mqt::test::value(runtime.outputShotStart());
  ::mqt::test::value(runtime.outputBool(true, "bool_label"));
  ::mqt::test::value(runtime.outputInt(42, "int_label"));
  ::mqt::test::value(runtime.outputFloat(3.14, "double_label"));
  ::mqt::test::value(runtime.outputTuple(2, "tuple_label"));
  ::mqt::test::value(runtime.outputArray(3, "array_label"));
  ::mqt::test::value(runtime.outputShotEnd());
  std::ostringstream expected;
  expected << "HEADER\tschema_id\tordered\n"
           << "HEADER\tschema_version\t2.1\n"
           << "START\n"
           << "METADATA\toutput_labeling_schema\tordered\n"
           << "OUTPUT\tBOOL\ttrue\n"
           << "OUTPUT\tINT\t42\n"
           << "OUTPUT\tDOUBLE\t3.14\n"
           << "OUTPUT\tTUPLE\t2\n"
           << "OUTPUT\tARRAY\t3\n"
           << "END\t0\n";
  EXPECT_EQ(sink.str(), expected.str());
}

TEST_F(QIRRuntimeTest, XGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);
}

TEST_F(QIRRuntimeTest, IdentityGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__i__body(q0);
}

TEST_F(QIRRuntimeTest, YGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__y__body(q0);
}

TEST_F(QIRRuntimeTest, ZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__z__body(q0);
}

TEST_F(QIRRuntimeTest, HGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q0);
}

TEST_F(QIRRuntimeTest, SGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__s__body(q0);
}

TEST_F(QIRRuntimeTest, SdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__s__adj(q0);
}

TEST_F(QIRRuntimeTest, SXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sx__body(q0);
}

TEST_F(QIRRuntimeTest, SXdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sx__adj(q0);
}

TEST_F(QIRRuntimeTest, TGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__t__body(q0);
}

TEST_F(QIRRuntimeTest, TdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__t__adj(q0);
}

TEST_F(QIRRuntimeTest, GlobalPhase) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__i__body(q0);
  __quantum__qis__gphase__body(dd::PI_2);

  const auto state = Runtime::getInstance().takeState();
  const auto vector = state.edge.getVector();
  ASSERT_EQ(vector.size(), 2);
  EXPECT_NEAR(vector[0].real(), 0., 1e-12);
  EXPECT_NEAR(vector[0].imag(), 1., 1e-12);
  EXPECT_EQ(vector[1], 0.);
}

TEST_F(QIRRuntimeTest, PRXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__prx__body(dd::PI_2, 0, q0);
}

TEST_F(QIRRuntimeTest, RXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rx__body(dd::PI_2, q0);
}

TEST_F(QIRRuntimeTest, RYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ry__body(dd::PI_2, q0);
}

TEST_F(QIRRuntimeTest, RZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rz__body(dd::PI_2, q0);
}

TEST_F(QIRRuntimeTest, PGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__p__body(dd::PI_2, q0);
}

TEST_F(QIRRuntimeTest, RXXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rxx__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, RYYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ryy__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, RZZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rzz__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, RZXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rzx__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, ISwapGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__iswap__body(q0, q1);
}

TEST_F(QIRRuntimeTest, DCXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__dcx__body(q0, q1);
}

TEST_F(QIRRuntimeTest, ECRGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ecr__body(q0, q1);
}

TEST_F(QIRRuntimeTest, XXPlusYYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__xx_plus_yy__body(dd::PI_2, dd::PI_4, q0, q1);
}

TEST_F(QIRRuntimeTest, XXMinusYYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__xx_minus_yy__body(dd::PI_2, dd::PI_4, q0, q1);
}

TEST_F(QIRRuntimeTest, U3Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u3__body(dd::PI_2, 0, dd::PI_4, q0);
}

TEST_F(QIRRuntimeTest, U2Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u2__body(dd::PI_2, 0, q0);
}

TEST_F(QIRRuntimeTest, CNotGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cnot__body(q0, q1);
}

TEST_F(QIRRuntimeTest, CXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cx__body(q0, q1);
}

TEST_F(QIRRuntimeTest, CYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cy__body(q0, q1);
}

TEST_F(QIRRuntimeTest, CZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cz__body(q0, q1);
}

TEST_F(QIRRuntimeTest, CHGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ch__body(q0, q1);
}

TEST_F(QIRRuntimeTest, SwapGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  __quantum__qis__x__body(q0);
  __quantum__qis__swap__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t0\tr0\n"
           << "OUTPUT\tRESULT\t1\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
}

TEST_F(QIRRuntimeTest, EmptyGenericControlsUseSwap) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);

  auto* controls = __quantum__rt__array_create_1d(sizeof(Qubit*), 0);
  struct Args {
    std::array<Qubit*, 2> targets;
  };
  const Args args{.targets = {q0, q1}};
  auto* tuple = __quantum__rt__tuple_create(sizeof(Args));
  std::memcpy(tuple, &args, sizeof(Args));
  __quantum__qis__swap__ctl(controls, tuple);

  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  EXPECT_FALSE(__quantum__rt__read_result(r0));
  EXPECT_TRUE(__quantum__rt__read_result(r1));
  __quantum__rt__tuple_update_reference_count(tuple, -1);
  __quantum__rt__array_update_reference_count(controls, -1);
}

TEST_F(QIRRuntimeTest, CSwapGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cswap__body(q0, q1, q2);
}

TEST_F(QIRRuntimeTest, CRZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__crz__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, CRYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cry__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, CRXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__crx__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, CPGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cp__body(dd::PI_2, q0, q1);
}

TEST_F(QIRRuntimeTest, CCXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ccx__body(q0, q1, q2);
}

TEST_F(QIRRuntimeTest, CCYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ccy__body(q0, q1, q2);
}

TEST_F(QIRRuntimeTest, CCZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ccz__body(q0, q1, q2);
}

TEST_F(QIRRuntimeTest, ThreeControlsUseGenericSpecialization) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  auto* target = reinterpret_cast<Qubit*>(3UL);
  auto* result = reinterpret_cast<Result*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);
  __quantum__qis__x__body(q1);
  __quantum__qis__x__body(q2);

  auto* controls = __quantum__rt__array_create_1d(sizeof(Qubit*), 3);
  const std::array controlQubits{q0, q1, q2};
  for (size_t i = 0; i < controlQubits.size(); ++i) {
    std::memcpy(__quantum__rt__array_get_element_ptr_1d(
                    controls, static_cast<int64_t>(i)),
                static_cast<const void*>(&controlQubits[i]), sizeof(Qubit*));
  }

  __quantum__qis__x__ctl(controls, target);
  __quantum__qis__mz__body(target, result);
  EXPECT_TRUE(__quantum__rt__read_result(result));
  __quantum__rt__array_update_reference_count(controls, -1);
}

TEST_F(QIRRuntimeTest, GenericControlledRotationUsesArgumentTuple) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  auto* target = reinterpret_cast<Qubit*>(3UL);
  auto* result = reinterpret_cast<Result*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);
  __quantum__qis__x__body(q1);
  __quantum__qis__x__body(q2);

  auto* controls = __quantum__rt__array_create_1d(sizeof(Qubit*), 3);
  const std::array controlQubits{q0, q1, q2};
  for (size_t i = 0; i < controlQubits.size(); ++i) {
    std::memcpy(__quantum__rt__array_get_element_ptr_1d(
                    controls, static_cast<int64_t>(i)),
                static_cast<const void*>(&controlQubits[i]), sizeof(Qubit*));
  }

  struct Args {
    double angle;
    Qubit* target;
  };
  const Args args{.angle = dd::PI, .target = target};
  auto* tuple = __quantum__rt__tuple_create(sizeof(Args));
  std::memcpy(tuple, &args, sizeof(Args));

  __quantum__qis__rx__ctl(controls, tuple);
  __quantum__qis__mz__body(target, result);
  EXPECT_TRUE(__quantum__rt__read_result(result));
  __quantum__rt__tuple_update_reference_count(tuple, -1);
  __quantum__rt__array_update_reference_count(controls, -1);
}

TEST_F(QIRRuntimeTest, RCCXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* q2 = reinterpret_cast<Qubit*>(2UL);
  __quantum__rt__initialize(nullptr);
  EXPECT_NO_THROW(__quantum__qis__rccx__body(q0, q1, q2));
}

TEST_F(QIRRuntimeTest, MzGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__mz__body(q0, r0);
}

TEST_F(QIRRuntimeTest, ResetGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);
  __quantum__qis__reset__body(q0);
  __quantum__qis__mz__body(q0, r0);
  EXPECT_FALSE(__quantum__rt__read_result(r0));
}

TEST_F(QIRRuntimeTest, Qir21BulkResourceManagement) {
  __quantum__rt__initialize(nullptr);
  std::array<Qubit*, 2> qubits{};
  std::array<Result*, 2> results{};
  bool error = true;
  __quantum__rt__qubit_array_allocate(qubits.size(), qubits.data(), &error);
  EXPECT_FALSE(error);
  __quantum__rt__result_array_allocate(results.size(), results.data(), &error);
  EXPECT_FALSE(error);

  __quantum__qis__x__body(qubits[0]);
  __quantum__qis__mz__body(qubits[0], results[0]);
  __quantum__qis__mz__body(qubits[1], results[1]);
  EXPECT_TRUE(__quantum__rt__read_result(results[0]));
  EXPECT_FALSE(__quantum__rt__read_result(results[1]));
  __quantum__rt__result_array_record_output(results.size(), results.data(),
                                            "results");
  EXPECT_THAT(sink.str(),
              ::testing::HasSubstr("OUTPUT\tRESULT_ARRAY\t10\tresults\n"));

  __quantum__rt__result_array_release(results.size(), results.data());
  __quantum__rt__qubit_array_release(qubits.size(), qubits.data());
}

TEST_F(QIRRuntimeTest, BellPairStatic) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
}

TEST_F(QIRRuntimeTest, BellPairDynamic) {
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  auto* q0 = __quantum__rt__qubit_allocate(nullptr);
  auto* q1 = __quantum__rt__qubit_allocate(nullptr);
  auto* r0 = __quantum__rt__result_allocate(nullptr);
  auto* r1 = __quantum__rt__result_allocate(nullptr);
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
  __quantum__rt__result_release(r0);
  __quantum__rt__result_release(r1);
}

TEST_F(QIRRuntimeTest, BellPairStaticReverse) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  __quantum__qis__h__body(q1);
  __quantum__qis__cx__body(q1, q0);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
}

TEST_F(QIRRuntimeTest, BellPairDynamicReverse) {
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  auto* q0 = __quantum__rt__qubit_allocate(nullptr);
  auto* q1 = __quantum__rt__qubit_allocate(nullptr);
  auto* r0 = __quantum__rt__result_allocate(nullptr);
  auto* r1 = __quantum__rt__result_allocate(nullptr);
  __quantum__qis__h__body(q1);
  __quantum__qis__cx__body(q1, q0);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
  __quantum__rt__result_release(r0);
  __quantum__rt__result_release(r1);
}

TEST_F(QIRRuntimeTest, GHZ4Static) {
  const std::array q = {
      reinterpret_cast<Qubit*>(0UL),
      reinterpret_cast<Qubit*>(1UL),
      reinterpret_cast<Qubit*>(2UL),
      reinterpret_cast<Qubit*>(3UL),
  };
  const std::array r = {
      reinterpret_cast<Result*>(0UL),
      reinterpret_cast<Result*>(1UL),
      reinterpret_cast<Result*>(2UL),
      reinterpret_cast<Result*>(3UL),
  };
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q[0]);
  __quantum__qis__cx__body(q[0], q[1]);
  __quantum__qis__cx__body(q[1], q[2]);
  __quantum__qis__cx__body(q[2], q[3]);
  __quantum__qis__mz__body(q[0], r[0]);
  __quantum__qis__mz__body(q[1], r[1]);
  __quantum__qis__mz__body(q[2], r[2]);
  __quantum__qis__mz__body(q[3], r[3]);
  const auto m0 = __quantum__rt__read_result(r[0]);
  const auto m1 = __quantum__rt__read_result(r[1]);
  const auto m2 = __quantum__rt__read_result(r[2]);
  const auto m3 = __quantum__rt__read_result(r[3]);
  EXPECT_EQ(m0, m1);
  EXPECT_EQ(m1, m2);
  EXPECT_EQ(m1, m3);
  __quantum__rt__result_record_output(r[0], "r0");
  __quantum__rt__result_record_output(r[1], "r1");
  __quantum__rt__result_record_output(r[2], "r2");
  __quantum__rt__result_record_output(r[3], "r3");
}

TEST_F(QIRRuntimeTest, GHZ4Dynamic) {
  __quantum__rt__initialize(nullptr);
  std::array<Qubit*, 4> q{};
  std::array<Result*, 4> r{};
  __quantum__rt__qubit_array_allocate(q.size(), q.data(), nullptr);
  __quantum__rt__result_array_allocate(r.size(), r.data(), nullptr);
  __quantum__qis__h__body(q[0]);
  __quantum__qis__cx__body(q[0], q[1]);
  __quantum__qis__cx__body(q[1], q[2]);
  __quantum__qis__cx__body(q[2], q[3]);
  __quantum__qis__mz__body(q[0], r[0]);
  __quantum__qis__mz__body(q[1], r[1]);
  __quantum__qis__mz__body(q[2], r[2]);
  __quantum__qis__mz__body(q[3], r[3]);
  const std::array m = {
      __quantum__rt__read_result(r[0]),
      __quantum__rt__read_result(r[1]),
      __quantum__rt__read_result(r[2]),
      __quantum__rt__read_result(r[3]),
  };
  EXPECT_EQ(m[0], m[1]);
  EXPECT_EQ(m[1], m[2]);
  EXPECT_EQ(m[2], m[3]);
  __quantum__rt__result_record_output(r[0], "r0");
  __quantum__rt__result_record_output(r[1], "r1");
  __quantum__rt__result_record_output(r[2], "r2");
  __quantum__rt__result_record_output(r[3], "r3");
  __quantum__rt__result_array_release(r.size(), r.data());
  __quantum__rt__qubit_array_release(q.size(), q.data());
}

TEST_F(QIRRuntimeTest, PackageResizeWhenEnlargingState) {
  /// Acting on qubit 32 must extend the initially empty state.
  auto* q32 = reinterpret_cast<Qubit*>(32UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q32);
}

TEST_F(QIRRuntimeTest, TakeStateReturnsStateAndResetsRuntime) {
  // Drive a small program through the runtime: H on q0.
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q0);

  auto state = Runtime::getInstance().takeState();
  EXPECT_NE(state.dd, nullptr);
  EXPECT_FALSE(state.edge.isTerminal());
  EXPECT_EQ(state.numQubits, 1);

  // After takeState the runtime is reset and usable again.
  EXPECT_NO_THROW(__quantum__rt__initialize(nullptr));
  EXPECT_NO_THROW(__quantum__qis__h__body(q0));
}

TEST_F(QIRRuntimeTest, AdaptiveRecordOutputs) {
  __quantum__rt__initialize(nullptr);
  ::mqt::test::value(Runtime::getInstance().outputProgramHeader());
  ::mqt::test::value(Runtime::getInstance().outputShotStart());
  auto* q0 = __quantum__rt__qubit_allocate(nullptr);
  auto* q1 = __quantum__rt__qubit_allocate(nullptr);
  auto* q2 = __quantum__rt__qubit_allocate(nullptr);
  auto* r0 = __quantum__rt__result_allocate(nullptr);
  auto* r1 = __quantum__rt__result_allocate(nullptr);
  auto* r2 = __quantum__rt__result_allocate(nullptr);
  __quantum__qis__h__body(q0);
  __quantum__qis__h__body(q1);
  __quantum__qis__h__body(q2);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__qis__mz__body(q2, r2);
  const auto b0 = __quantum__rt__read_result(r0);
  const auto b1 = __quantum__rt__read_result(r1);
  const auto b2 = __quantum__rt__read_result(r2);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  __quantum__rt__qubit_release(q2);

  // Classical compute: Hamming weight and its mean.
  const int64_t weight =
      static_cast<int>(b0) + static_cast<int>(b1) + static_cast<int>(b2);
  const double mean = static_cast<double>(weight) / 3.0;

  // Output: tuple of 3 elements (array of 3 bools, int weight, float mean).
  __quantum__rt__tuple_record_output(3, "outputs");
  __quantum__rt__array_record_output(3, "measurements");
  __quantum__rt__bool_record_output(b0, "m0");
  __quantum__rt__bool_record_output(b1, "m1");
  __quantum__rt__bool_record_output(b2, "m2");
  __quantum__rt__int_record_output(weight, "hamming_weight");
  __quantum__rt__double_record_output(mean, "mean");
  ::mqt::test::value(Runtime::getInstance().outputShotEnd());

  std::ostringstream expected;
  expected.setf(std::ios::boolalpha);
  expected << "OUTPUT\tTUPLE\t3\toutputs\n"
           << "OUTPUT\tARRAY\t3\tmeasurements\n"
           << "OUTPUT\tBOOL\t" << b0 << "\tm0\n"
           << "OUTPUT\tBOOL\t" << b1 << "\tm1\n"
           << "OUTPUT\tBOOL\t" << b2 << "\tm2\n"
           << "OUTPUT\tINT\t" << weight << "\thamming_weight\n"
           << "OUTPUT\tDOUBLE\t" << mean << "\tmean\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));

  __quantum__rt__result_release(r0);
  __quantum__rt__result_release(r1);
  __quantum__rt__result_release(r2);
}

namespace {

class QIRFilesTest : public ::testing::TestWithParam<std::filesystem::path> {};

} // namespace

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    QIRExecutablesTest, //< Custom instantiation name
    QIRFilesTest,       //< Test suite name
    // Parameters to test with
    ::testing::Values(TEST_EXECUTABLES),
    [](const testing::TestParamInfo<std::filesystem::path>& inf) {
      // Extract the last part of the file path
      auto filename = inf.param.stem().string();
      // replace all '-' with '_'
      std::ranges::replace(filename, '-', '_');
      return filename;
    });

TEST_P(QIRFilesTest, Executables) {
  const auto& path = GetParam();
  const auto result = SYSTEM(path.c_str());
  EXPECT_EQ(result, 0);
}
} // namespace qir

namespace qir {

TEST_F(QIRRuntimeTest, PreservesPhaseBeforeFirstQubitAndFollowingGates) {
  __quantum__rt__initialize(nullptr);
  __quantum__qis__gphase__body(0.3);
  __quantum__qis__x__body(nullptr);
  __quantum__qis__gphase__body(0.4);
  __quantum__qis__x__body(nullptr);
  auto state = Runtime::getInstance().takeState();
  state.dd->garbageCollect(true);
  const auto values = state.edge.getVector();
  ASSERT_EQ(values.size(), 2);
  EXPECT_NEAR(std::abs(values[0] - std::polar(1., 0.7)), 0., 1e-12);
  EXPECT_EQ(values[1], 0.);
  state.dd->decRef(state.edge);
  EXPECT_NO_THROW(__quantum__rt__initialize(nullptr));
}

TEST_F(QIRRuntimeTest, ExtractsLogicalOrderAfterSwapCycle) {
  __quantum__rt__initialize(nullptr);
  auto* q0 = reinterpret_cast<Qubit*>(0);
  auto* q1 = reinterpret_cast<Qubit*>(1);
  auto* q2 = reinterpret_cast<Qubit*>(2);
  __quantum__qis__x__body(q0);
  __quantum__qis__swap__body(q0, q1);
  __quantum__qis__swap__body(q1, q2);
  auto state = Runtime::getInstance().takeState();
  const auto values = state.edge.getVector();
  ASSERT_EQ(values.size(), 8);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], i == 4 ? 1. : 0.);
  }
  state.dd->decRef(state.edge);
}

TEST_F(QIRRuntimeTest, ReusesReleasedWiresWithoutReusingHandles) {
  __quantum__rt__initialize(nullptr);
  auto* first = __quantum__rt__qubit_allocate(nullptr);
  __quantum__qis__x__body(first);
  __quantum__rt__qubit_release(first);
  for (size_t i = 0; i < 32; ++i) {
    auto* qubit = __quantum__rt__qubit_allocate(nullptr);
    EXPECT_NE(qubit, first);
    __quantum__qis__mz__body(qubit, nullptr);
    EXPECT_FALSE(__quantum__rt__read_result(nullptr));
    __quantum__qis__x__body(qubit);
    __quantum__rt__qubit_release(qubit);
  }
  EXPECT_DEATH(__quantum__qis__x__body(first), ".+");
  auto state = Runtime::getInstance().takeState();
  EXPECT_EQ(state.numQubits, 1);
  state.dd->decRef(state.edge);
}

TEST_F(QIRRuntimeTest, ReleasedSwappedWireIsResetBeforeReuse) {
  __quantum__rt__initialize(nullptr);
  auto* a = __quantum__rt__qubit_allocate(nullptr);
  auto* b = __quantum__rt__qubit_allocate(nullptr);
  __quantum__qis__x__body(a);
  __quantum__qis__swap__body(a, b);
  __quantum__rt__qubit_release(b);
  auto* c = __quantum__rt__qubit_allocate(nullptr);
  __quantum__qis__mz__body(c, nullptr);
  EXPECT_FALSE(__quantum__rt__read_result(nullptr));
  __quantum__qis__x__body(a);
  __quantum__qis__mz__body(c, nullptr);
  EXPECT_FALSE(__quantum__rt__read_result(nullptr));
  auto state = Runtime::getInstance().takeState();
  EXPECT_EQ(state.numQubits, 2);
  state.dd->decRef(state.edge);
}

TEST_F(QIRRuntimeTest, ReleasingUnusedQubitsDoesNotEnlargeState) {
  __quantum__rt__initialize(nullptr);
  for (size_t i = 0; i <= dd::Package::MAX_POSSIBLE_QUBITS; ++i) {
    __quantum__rt__qubit_release(__quantum__rt__qubit_allocate(nullptr));
  }
  auto state = Runtime::getInstance().takeState();
  EXPECT_EQ(state.numQubits, 0);
}

TEST_F(QIRRuntimeTest, DisabledTextOutputStillRecordsResults) {
  auto& runtime = Runtime::getInstance();
  __quantum__rt__initialize(nullptr);
  runtime.disableOutput();
  ::mqt::test::value(runtime.outputProgramHeader());
  ::mqt::test::value(runtime.outputShotStart());
  __quantum__qis__x__body(nullptr);
  __quantum__qis__mz__body(nullptr, nullptr);
  __quantum__rt__result_record_output(nullptr, "one");
  std::array<Result*, 2> results{nullptr, nullptr};
  __quantum__rt__result_array_record_output(2, results.data(), "pair");
  __quantum__rt__bool_record_output(true, nullptr);
  __quantum__rt__int_record_output(42, nullptr);
  __quantum__rt__double_record_output(0.3, nullptr);
  __quantum__rt__tuple_record_output(1, nullptr);
  __quantum__rt__array_record_output(1, nullptr);
  ::mqt::test::value(runtime.outputShotEnd());
  EXPECT_EQ(runtime.getMeasurements(), "1111");
  EXPECT_TRUE(sink.str().empty());
  runtime.setOstream(sink);
  __quantum__rt__result_record_output(nullptr, "one");
  EXPECT_FALSE(sink.str().empty());
}

} // namespace qir

TEST(QIRRuntimeGrowth, PreservesStateAndPhaseAcrossGrowthAndTransfer) {
  constexpr size_t width = 65;
  const std::array<std::complex<dd::fp>, 4> x{0., 1., 1., 0.};
  for (const bool dynamic : {false, true}) {
    qir::Runtime runtime(42);
    for (size_t job = 0; job < 2; ++job) {
      runtime.applyGlobalPhase(dd::PI);
      for (size_t q = 0; q < width; ++q) {
        auto* qubit = dynamic ? ::mqt::test::value(runtime.qAlloc())
                              : reinterpret_cast<Qubit*>(q);
        const std::array targets{qubit};
        ::mqt::test::value(runtime.apply(x, {}, targets));
      }
      auto state = runtime.takeState();
      EXPECT_EQ(state.numQubits, width);
      EXPECT_GE(state.dd->qubits(), width);
      EXPECT_LT(state.dd->qubits(), 2 * width);
      const auto amplitude = ::mqt::test::value(
          state.edge.getValueByPath(width, std::string(width, '1')));
      EXPECT_NEAR(amplitude.real(), -1., 1e-12);
      EXPECT_NEAR(amplitude.imag(), 0., 1e-12);
      state.dd->decRef(state.edge);
    }
  }
}

TEST(QIRRuntimeErrors, ReportsStreamFailuresWithoutThrowing) {
  qir::Runtime runtime{0};
  std::ostringstream stream;
  stream.setstate(std::ios::badbit);
  runtime.setOstream(stream);
  EXPECT_THAT(
      ::mqt::test::errorMessage([&] { return runtime.outputProgramHeader(); }),
      ::testing::HasSubstr("Failed to write QIR output"));
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return runtime.outputBool(true, "result"); }),
            ::mqt::ErrorCategory::IO);
  EXPECT_EQ(::mqt::test::errorKind([&] { return runtime.outputShotEnd(); }),
            ::mqt::ErrorCategory::IO);
  EXPECT_TRUE(stream.str().empty());
  stream.clear();
  stream.exceptions(std::ios::badbit);
  const auto diagnostic =
      ::mqt::test::diagnostic([&] { return runtime.outputShotStart(); });
  ASSERT_TRUE(diagnostic);
  EXPECT_EQ(diagnostic->category, ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(diagnostic->severity, ::mqt::DiagnosticSeverity::Error);
  EXPECT_FALSE(diagnostic->status);
  EXPECT_THAT(diagnostic->message, ::testing::HasSubstr("exceptions disabled"));
  stream.exceptions(std::ios::goodbit);
  ::mqt::test::value(runtime.outputShotStart());
  ::mqt::test::value(runtime.outputBool(true, "result"));
  ::mqt::test::value(runtime.outputShotEnd());
  EXPECT_THAT(stream.str(), ::testing::HasSubstr("START"));
  EXPECT_THAT(stream.str(),
              ::testing::HasSubstr("OUTPUT\tBOOL\ttrue\tresult\nEND\t0\n"));
}

TEST(QIRRuntimeErrors, ReportsResourceFailuresAtTheCall) {
  qir::Runtime runtime{0};
  auto* qubit = ::mqt::test::value(runtime.qAlloc());
  auto* result = ::mqt::test::value(runtime.rAlloc());
  ::mqt::test::value(runtime.qFree(qubit));
  const std::array controls{qubit};
  const std::array<std::complex<dd::fp>, 4> x{0., 1., 1., 0.};
  EXPECT_THAT(
      ::mqt::test::errorMessage([&] { return runtime.apply(x, controls, {}); }),
      ::testing::HasSubstr("Qubit not allocated"));
  EXPECT_EQ(
      ::mqt::test::errorKind([&] { return runtime.apply(x, {}, controls); }),
      ::mqt::ErrorCategory::OutOfRange);
  EXPECT_THAT(
      ::mqt::test::errorMessage([&] { return runtime.reset(controls); }),
      ::testing::HasSubstr("Qubit not allocated"));
  EXPECT_THAT(
      ::mqt::test::errorMessage([&] { return runtime.measure(qubit, result); }),
      ::testing::HasSubstr("Qubit not allocated"));
  auto* liveQubit = ::mqt::test::value(runtime.qAlloc());
  EXPECT_EQ(
      ::mqt::test::errorKind([&] { return runtime.swap(liveQubit, qubit); }),
      ::mqt::ErrorCategory::OutOfRange);
  EXPECT_EQ(::mqt::test::errorKind([&] { return runtime.qFree(qubit); }),
            ::mqt::ErrorCategory::OutOfRange);
  ::mqt::test::value(runtime.rFree(result));
  EXPECT_THAT(::mqt::test::errorMessage(
                  [&] { return runtime.measure(liveQubit, result); }),
              ::testing::HasSubstr("Result not allocated"));
  EXPECT_EQ(::mqt::test::errorKind([&] { return runtime.rFree(result); }),
            ::mqt::ErrorCategory::OutOfRange);
  auto* liveResult = ::mqt::test::value(runtime.rAlloc());
  ::mqt::test::value(runtime.apply(x, {}, std::array{liveQubit}));
  ::mqt::test::value(runtime.measure(liveQubit, liveResult));
  EXPECT_TRUE(::mqt::test::value(runtime.deref(liveResult))->r);
  ::mqt::test::value(runtime.qFree(liveQubit));
  ::mqt::test::value(runtime.rFree(liveResult));

  runtime.reset();
  ::mqt::test::value(runtime.apply(x, {}, std::array<Qubit*, 1>{nullptr}));
  EXPECT_EQ(::mqt::test::errorKind([&] { return runtime.qAlloc(); }),
            ::mqt::ErrorCategory::InvalidArgument);
  ::mqt::test::value(runtime.measure(static_cast<Qubit*>(nullptr),
                                     static_cast<Result*>(nullptr)));
  EXPECT_EQ(::mqt::test::errorKind([&] { return runtime.rAlloc(); }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_TRUE(::mqt::test::value(runtime.deref(nullptr))->r);
}

TEST(QIRRuntimeErrors, RejectsWrongGateMatrixDimensions) {
  qir::Runtime runtime{0};
  const std::array<Qubit*, 1> targets{nullptr};
  ::mqt::test::value(
      runtime.apply(mlir::qco::XOp::getUnitaryMatrix(), {}, targets));
  const std::array<std::complex<dd::fp>, 1> wrongSize{1.};
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return runtime.apply(wrongSize, {}, targets); }),
            ::mqt::ErrorCategory::InvalidArgument);
  auto state = runtime.takeState();
  EXPECT_EQ(state.numQubits, 1);
  EXPECT_EQ(::mqt::test::value(state.edge.getValueByPath(1, "1")),
            std::complex<dd::fp>(1., 0.));
  state.dd->decRef(state.edge);
}
