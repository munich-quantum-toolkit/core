/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * DDSIM QDMI Device - Results: statevector (dense/sparse)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstddef>
#include <vector>

#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
#include "qir/runtime/Runtime.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <sstream>
#include <string>
#include <string_view>
#endif

TEST(ResultsStatevector, DenseNormalizedAndBufferTooSmall) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  auto vec = qdmi_test::getDenseState(j.job);
  ASSERT_FALSE(vec.empty());
  auto norm = 0.0;
  for (const auto& v : vec) {
    norm += std::norm(v);
  }
  EXPECT_NEAR(norm, 1.0, 1e-6);

  const size_t sz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE);
  if (sz > 0) {
    std::vector<char> tooSmall(sz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsStatevector, SparseNormalizedAndBufferTooSmall) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  auto [keys, vals] = qdmi_test::getSparseState(j.job);
  ASSERT_EQ(keys.size(), vals.size());
  auto norm = 0.0;
  for (const auto& v : vals) {
    norm += std::norm(v);
  }
  EXPECT_NEAR(norm, 1.0, 1e-6);

  const size_t ksz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS);
  if (ksz > 0) {
    std::vector<char> tooSmall(ksz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
                  tooSmall.size(), tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  const size_t vsz =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES);
  if (vsz > 0) {
    std::vector<char> tooSmall(vsz - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
                  tooSmall.size(), tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsStatevector, HistogramRequestsInvalidWithShotsZero) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_KEYS, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_HIST_VALUES, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

#ifdef BUILD_MQT_CORE_QDMI_DDSIM_WITH_QIR
namespace {

class QIRStateExtractionTest : public testing::Test {
protected:
  std::ostringstream sink;
  void SetUp() override { qir::Runtime::getInstance().setOstream(sink); }
  void TearDown() override { qir::Runtime::getInstance().resetOstream(); }

  static std::string getProgram(const std::string_view file) {
    const std::filesystem::path path =
        std::filesystem::path(QIR_FILES_DIR) / file;
    std::ifstream ifs(path);
    EXPECT_TRUE(ifs.is_open()) << "Failed to open " << path.string();
    return {std::istreambuf_iterator<char>{ifs}, {}};
  }
};

TEST_F(QIRStateExtractionTest, BellPairStaticBaseStringYieldsBellState) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  const auto program = getProgram("BellPairStatic.ll");
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QIRBASESTRING, program),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const auto vec = qdmi_test::getDenseState(j.job);
  ASSERT_EQ(vec.size(), 4U);

  // Bell pair: amplitudes at |00> and |11> are 1/sqrt(2), |01> and |10> are 0.
  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  EXPECT_NEAR(std::abs(vec[0]), invSqrt2, 1e-6);
  EXPECT_NEAR(std::abs(vec[1]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[2]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(vec[3]), invSqrt2, 1e-6);
}

} // namespace
#endif
