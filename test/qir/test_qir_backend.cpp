/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/QIR.h"

#include <algorithm>
#ifndef _WIN32
#include <cstdlib>
#endif // _WIN32
#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <streambuf>

namespace mqt {

class QIRBackendTest : public ::testing::Test {
protected:
  std::stringstream buffer;
  std::streambuf* old = nullptr;
  void SetUp() override { old = std::cout.rdbuf(buffer.rdbuf()); }
  void TearDown() override { std::cout.rdbuf(old); }
};

TEST_F(QIRBackendTest, BellPairStatic) {
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

TEST_F(QIRBackendTest, BellPairDynamic) {
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

TEST_F(QIRBackendTest, BellPairStaticReverse) {
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

TEST_F(QIRBackendTest, BellPairDynamicReverse) {
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

TEST_F(QIRBackendTest, GHZ4Static) {
  const std::array<Qubit*, 4> q = {
      reinterpret_cast<Qubit*>(0UL), reinterpret_cast<Qubit*>(1UL),
      reinterpret_cast<Qubit*>(2UL), reinterpret_cast<Qubit*>(3UL)};
  const std::array<Result*, 4> r = {
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

TEST_F(QIRBackendTest, GHZ4StaticReverse) {
  const std::array<Qubit*, 4> q = {
      reinterpret_cast<Qubit*>(0UL), reinterpret_cast<Qubit*>(1UL),
      reinterpret_cast<Qubit*>(2UL), reinterpret_cast<Qubit*>(3UL)};
  const std::array<Result*, 4> r = {
      reinterpret_cast<Result*>(0UL), reinterpret_cast<Result*>(1UL),
      reinterpret_cast<Result*>(2UL), reinterpret_cast<Result*>(3UL)};
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q[3]);
  __quantum__qis__cx__body(q[3], q[2]);
  __quantum__qis__cx__body(q[2], q[1]);
  __quantum__qis__cx__body(q[1], q[0]);
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

TEST_F(QIRBackendTest, GHZ4Dynamic) {
  __quantum__rt__initialize(nullptr);
  auto* qArr = __quantum__rt__qubit_allocate_array(4);
  const std::array<Qubit*, 4> q = {
      *reinterpret_cast<Qubit**>(
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
  auto* r0 = __quantum__qis__m__body(q[0]);
  auto* r1 = __quantum__qis__m__body(q[1]);
  auto* r2 = __quantum__qis__m__body(q[2]);
  auto* r3 = __quantum__qis__m__body(q[3]);
  __quantum__rt__qubit_release_array(qArr);
  const auto m0 = __quantum__rt__read_result(r0);
  const auto m1 = __quantum__rt__read_result(r1);
  const auto m2 = __quantum__rt__read_result(r2);
  const auto m3 = __quantum__rt__read_result(r3);
  EXPECT_EQ(m0, m1);
  EXPECT_EQ(m1, m2);
  EXPECT_EQ(m2, m3);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  __quantum__rt__result_record_output(r2, "r2");
  __quantum__rt__result_record_output(r3, "r3");
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
  __quantum__rt__result_update_reference_count(r2, -1);
  __quantum__rt__result_update_reference_count(r3, -1);
}

TEST_F(QIRBackendTest, GHZ4DynamicReverse) {
  __quantum__rt__initialize(nullptr);
  auto* qArr = __quantum__rt__qubit_allocate_array(4);
  const std::array<Qubit*, 4> q = {
      *reinterpret_cast<Qubit**>(
          __quantum__rt__array_get_element_ptr_1d(qArr, 0)),
      *reinterpret_cast<Qubit**>(
          __quantum__rt__array_get_element_ptr_1d(qArr, 1)),
      *reinterpret_cast<Qubit**>(
          __quantum__rt__array_get_element_ptr_1d(qArr, 2)),
      *reinterpret_cast<Qubit**>(
          __quantum__rt__array_get_element_ptr_1d(qArr, 3))};
  __quantum__qis__h__body(q[3]);
  __quantum__qis__cx__body(q[3], q[2]);
  __quantum__qis__cx__body(q[2], q[1]);
  __quantum__qis__cx__body(q[1], q[0]);
  auto* r0 = __quantum__qis__m__body(q[0]);
  auto* r1 = __quantum__qis__m__body(q[1]);
  auto* r2 = __quantum__qis__m__body(q[2]);
  auto* r3 = __quantum__qis__m__body(q[3]);
  __quantum__rt__qubit_release_array(qArr);
  const auto m0 = __quantum__rt__read_result(r0);
  const auto m1 = __quantum__rt__read_result(r1);
  const auto m2 = __quantum__rt__read_result(r2);
  const auto m3 = __quantum__rt__read_result(r3);
  EXPECT_EQ(m0, m1);
  EXPECT_EQ(m1, m2);
  EXPECT_EQ(m2, m3);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  __quantum__rt__result_record_output(r2, "r2");
  __quantum__rt__result_record_output(r3, "r3");
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
  __quantum__rt__result_update_reference_count(r2, -1);
  __quantum__rt__result_update_reference_count(r3, -1);
}

#ifndef _WIN32
class QIRFilesTest : public ::testing::TestWithParam<std::filesystem::path> {};

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    QIRExecutablesTest, //< Custom instantiation name
    QIRFilesTest,       //< Test suite name
    // Parameters to test with
    ::testing::Values(TEST_EXECUTABLES),
    [](const testing::TestParamInfo<std::filesystem::path>& info) {
      // Extract the last part of the file path
      auto filename = info.param.filename().string();
      // replace all '-' with '_'
      std::ranges::replace(filename, '-', '_');
      return filename;
    });

TEST_P(QIRFilesTest, Executables) {
  const auto& path = GetParam();
  const auto result = std::system(path.c_str());
  EXPECT_EQ(result, 0);
}
#endif // _WIN32
} // namespace mqt
