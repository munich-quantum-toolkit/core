/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "qir/runtime/QIR.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <streambuf>

#ifdef _WIN32
#define SYSTEM _wsystem
#else
#define SYSTEM std::system
#endif

namespace qir {

class QIRRuntimeTest : public testing::Test {
protected:
  std::stringstream buffer;
  std::streambuf* old = nullptr;
  void SetUp() override { old = std::cout.rdbuf(buffer.rdbuf()); }
  void TearDown() override { std::cout.rdbuf(old); }
};

TEST_F(QIRRuntimeTest, XGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__x__body(q0);
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
  __quantum__qis__sdg__body(q0);
}

TEST_F(QIRRuntimeTest, SXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sx__body(q0);
}

TEST_F(QIRRuntimeTest, SXdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sxdg__body(q0);
}

TEST_F(QIRRuntimeTest, SqrtXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sqrtx__body(q0);
}

TEST_F(QIRRuntimeTest, SqrtXdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__sqrtxdg__body(q0);
}

TEST_F(QIRRuntimeTest, TGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__t__body(q0);
}

TEST_F(QIRRuntimeTest, TdgGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__tdg__body(q0);
}

TEST_F(QIRRuntimeTest, RXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rx__body(q0, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ry__body(q0, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rz__body(q0, qc::PI_2);
}

TEST_F(QIRRuntimeTest, PGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__p__body(q0, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RXXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rxx__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RYYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__ryy__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RZZGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rzz__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, RZXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__rzx__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, UGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u__body(q0, qc::PI_2, 0, qc::PI_4);
}

TEST_F(QIRRuntimeTest, U3Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u3__body(q0, qc::PI_2, 0, qc::PI_4);
}

TEST_F(QIRRuntimeTest, U2Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u2__body(q0, qc::PI_2, 0);
}

TEST_F(QIRRuntimeTest, U1Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__u1__body(q0, qc::PI_2);
}

TEST_F(QIRRuntimeTest, CU1Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cu1__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, CU3Gate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cu3__body(q0, q1, qc::PI_2, 0, qc::PI_4);
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
  __quantum__qis__x__body(q0);
  __quantum__qis__swap__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_EQ(buffer.str(), "r0: 0\nr1: 1\n");
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
  __quantum__qis__crz__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, CRYGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cry__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, CRXGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__crx__body(q0, q1, qc::PI_2);
}

TEST_F(QIRRuntimeTest, CPGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__cp__body(q0, q1, qc::PI_2);
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

TEST_F(QIRRuntimeTest, MGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__m__body(q0);
}

TEST_F(QIRRuntimeTest, MeasureGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__measure__body(q0);
}

TEST_F(QIRRuntimeTest, MzGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__mz__body(q0, r0);
}

TEST_F(QIRRuntimeTest, ResetGate) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__reset__body(q0);
}

TEST_F(QIRRuntimeTest, BellPairStatic) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
}

TEST_F(QIRRuntimeTest, BellPairDynamic) {
  __quantum__rt__initialize(nullptr);
  auto* q0 = __quantum__rt__qubit_allocate();
  auto* q1 = __quantum__rt__qubit_allocate();
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  auto* r0 = __quantum__qis__m__body(q0);
  auto* r1 = __quantum__qis__m__body(q1);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
}

TEST_F(QIRRuntimeTest, BellPairStaticReverse) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q1);
  __quantum__qis__cx__body(q1, q0);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
}

TEST_F(QIRRuntimeTest, BellPairDynamicReverse) {
  __quantum__rt__initialize(nullptr);
  auto* q0 = __quantum__rt__qubit_allocate();
  auto* q1 = __quantum__rt__qubit_allocate();
  __quantum__qis__h__body(q1);
  __quantum__qis__cx__body(q1, q0);
  auto* r0 = __quantum__qis__m__body(q0);
  auto* r1 = __quantum__qis__m__body(q1);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
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
  auto* qArr = __quantum__rt__qubit_allocate_array(4);
  const std::array q = {*reinterpret_cast<Qubit**>(
                            __quantum__rt__array_get_element_ptr_1d(qArr, 0)),
                        *reinterpret_cast<Qubit**>(
                            __quantum__rt__array_get_element_ptr_1d(qArr, 1)),
                        *reinterpret_cast<Qubit**>(
                            __quantum__rt__array_get_element_ptr_1d(qArr, 2)),
                        *reinterpret_cast<Qubit**>(
                            __quantum__rt__array_get_element_ptr_1d(qArr, 3))};
  __quantum__qis__h__body(q[0]);
  __quantum__qis__cx__body(q[0], q[1]);
  __quantum__qis__cx__body(q[1], q[2]);
  __quantum__qis__cx__body(q[2], q[3]);
  auto* rArr = __quantum__rt__array_create_1d(sizeof(Result*), 4);
  std::array r = {*reinterpret_cast<Result**>(
                      __quantum__rt__array_get_element_ptr_1d(rArr, 0)),
                  *reinterpret_cast<Result**>(
                      __quantum__rt__array_get_element_ptr_1d(rArr, 1)),
                  *reinterpret_cast<Result**>(
                      __quantum__rt__array_get_element_ptr_1d(rArr, 2)),
                  *reinterpret_cast<Result**>(
                      __quantum__rt__array_get_element_ptr_1d(rArr, 3))};
  r[0] = __quantum__qis__m__body(q[0]);
  r[1] = __quantum__qis__m__body(q[1]);
  r[2] = __quantum__qis__m__body(q[2]);
  r[3] = __quantum__qis__m__body(q[3]);
  __quantum__rt__qubit_release_array(qArr);
  EXPECT_TRUE(__quantum__rt__result_equal(r[0], r[1]));
  EXPECT_TRUE(__quantum__rt__result_equal(r[1], r[2]));
  EXPECT_TRUE(__quantum__rt__result_equal(r[2], r[3]));
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
  __quantum__rt__result_update_reference_count(r[0], -1);
  __quantum__rt__result_update_reference_count(r[1], -1);
  __quantum__rt__result_update_reference_count(r[2], -1);
  __quantum__rt__result_update_reference_count(r[3], -1);
  __quantum__rt__array_update_reference_count(rArr, -1);
}

class QIRFilesTest : public ::testing::TestWithParam<std::filesystem::path> {};

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    QIRExecutablesTest, //< Custom instantiation name
    QIRFilesTest,       //< Test suite name
    // Parameters to test with
    ::testing::Values(TEST_EXECUTABLES),
    [](const testing::TestParamInfo<std::filesystem::path>& info) {
      // Extract the last part of the file path
      auto filename = info.param.stem().string();
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
