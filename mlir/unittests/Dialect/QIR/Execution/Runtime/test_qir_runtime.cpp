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
#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>

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
  EXPECT_THROW(__quantum__rt__array_create_1d(0, 1), std::invalid_argument);
  EXPECT_THROW(__quantum__rt__array_create_1d(sizeof(Qubit*), -1),
               std::invalid_argument);
  EXPECT_THROW(
      __quantum__rt__array_create_1d(2, std::numeric_limits<int64_t>::max()),
      std::length_error);
}

TEST(QIRRuntimeArgumentsTest, RejectsInvalidTupleSizes) {
  EXPECT_THROW(__quantum__rt__tuple_create(-1), std::invalid_argument);
  EXPECT_THROW(__quantum__rt__tuple_create(std::numeric_limits<int64_t>::max()),
               std::length_error);

  auto* controls = __quantum__rt__array_create_1d(sizeof(Qubit*), 0);
  auto* tuple = __quantum__rt__tuple_create(1);
  EXPECT_THROW(__quantum__qis__rx__ctl(controls, tuple), std::invalid_argument);
  __quantum__rt__tuple_update_reference_count(tuple, -1);
  __quantum__rt__array_update_reference_count(controls, -1);
}

TEST_F(QIRRuntimeTest, RejectsInvalidDynamicResourceUse) {
  __quantum__rt__initialize(nullptr);
  auto* qubit = __quantum__rt__qubit_allocate(nullptr);
  auto* result = __quantum__rt__result_allocate(nullptr);
  __quantum__rt__qubit_release(qubit);
  __quantum__rt__result_release(result);

  EXPECT_THROW(__quantum__qis__x__body(qubit), std::out_of_range);
  EXPECT_THROW(__quantum__rt__read_result(result), std::out_of_range);
  EXPECT_THROW(__quantum__rt__qubit_release(qubit), std::out_of_range);
  EXPECT_THROW(__quantum__rt__result_release(result), std::out_of_range);
}

TEST_F(QIRRuntimeTest, RejectsMixedStaticAndDynamicResourceManagement) {
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(nullptr);
  EXPECT_THROW(__quantum__rt__qubit_allocate(nullptr), std::logic_error);

  __quantum__rt__initialize(nullptr);
  __quantum__qis__mz__body(nullptr, nullptr);
  EXPECT_THROW(__quantum__rt__result_allocate(nullptr), std::logic_error);
}

TEST_F(QIRRuntimeTest, RejectsStaticQubitBeyondDDRange) {
  __quantum__rt__initialize(nullptr);
  auto* qubit = reinterpret_cast<Qubit*>(dd::Package::MAX_POSSIBLE_QUBITS);
  EXPECT_THROW(__quantum__qis__x__body(qubit), std::out_of_range);

  __quantum__rt__initialize(nullptr);
  constexpr std::array<dd::fp, 0> params{};
  std::array<Qubit*, 0> controls{};
  std::array<Qubit*, 1> targets{qubit};
  EXPECT_THROW(
      Runtime::getInstance().apply(dd::GateType::X, params, controls, targets),
      std::out_of_range);
}

TEST_F(QIRRuntimeTest, RejectsDynamicQubitBeyondDDRange) {
  Runtime runtime{0};
  for (size_t i = 0; i < dd::Package::MAX_POSSIBLE_QUBITS; ++i) {
    static_cast<void>(runtime.qAlloc());
  }
  EXPECT_THROW(runtime.qAlloc(), std::out_of_range);
}

} // namespace

// Any test that emits output relies on the runtime producing the spec-mandated
// HEADER/START/METADATA/END records around the per-shot OUTPUT block.
// The runtime picks @c Labeled as default output schema, which is why the
// the framing emits `labeled` in both HEADER and METADATA here.
TEST_F(QIRRuntimeTest, OutputFraming) {
  auto& runtime = Runtime::getInstance();
  runtime.outputProgramHeader();
  runtime.outputShotStart();
  runtime.outputShotEnd();
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
  runtime.outputProgramHeader();
  runtime.outputShotStart();
  runtime.outputBool(true, "bool_label");
  runtime.outputInt(42, "int_label");
  runtime.outputFloat(3.14, "double_label");
  runtime.outputTuple(2, "tuple_label");
  runtime.outputArray(3, "array_label");
  runtime.outputShotEnd();
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
  runtime.outputProgramHeader();
  runtime.outputShotStart();
  runtime.outputBool(true, "bool_label");
  runtime.outputInt(42, "int_label");
  runtime.outputFloat(3.14, "double_label");
  runtime.outputTuple(2, "tuple_label");
  runtime.outputArray(3, "array_label");
  runtime.outputShotEnd();
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
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
  __quantum__qis__x__body(q0);
  __quantum__qis__swap__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  Runtime::getInstance().outputShotEnd();
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t0\tr0\n"
           << "OUTPUT\tRESULT\t1\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
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
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  Runtime::getInstance().outputShotEnd();
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
}

TEST_F(QIRRuntimeTest, BellPairDynamic) {
  __quantum__rt__initialize(nullptr);
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
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
  Runtime::getInstance().outputShotEnd();
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
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
  __quantum__qis__h__body(q1);
  __quantum__qis__cx__body(q1, q0);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  Runtime::getInstance().outputShotEnd();
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
}

TEST_F(QIRRuntimeTest, BellPairDynamicReverse) {
  __quantum__rt__initialize(nullptr);
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
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
  Runtime::getInstance().outputShotEnd();
  std::ostringstream expected;
  expected << "OUTPUT\tRESULT\t" << m1 << "\tr0\n"
           << "OUTPUT\tRESULT\t" << m2 << "\tr1\n";
  EXPECT_THAT(sink.str(), ::testing::HasSubstr(expected.str()));
  __quantum__rt__result_release(r0);
  __quantum__rt__result_release(r1);
}

TEST_F(QIRRuntimeTest, GHZ4Static) {
  const std::array q = {
      reinterpret_cast<Qubit*>(0UL), reinterpret_cast<Qubit*>(1UL),
      reinterpret_cast<Qubit*>(2UL), reinterpret_cast<Qubit*>(3UL)};
  const std::array r = {
      reinterpret_cast<Result*>(0UL), reinterpret_cast<Result*>(1UL),
      reinterpret_cast<Result*>(2UL), reinterpret_cast<Result*>(3UL)};
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
      __quantum__rt__read_result(r[0]), __quantum__rt__read_result(r[1]),
      __quantum__rt__read_result(r[2]), __quantum__rt__read_result(r[3])};
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
  // dd::Package starts at 32 qubits.
  // Acting on qubit 32 forces qState.dd->resize.
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
  Runtime::getInstance().outputProgramHeader();
  Runtime::getInstance().outputShotStart();
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
  Runtime::getInstance().outputShotEnd();

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
