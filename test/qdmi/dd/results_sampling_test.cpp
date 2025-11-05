/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * DDSIM QDMI Device - Results: sampling (histogram keys/values)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(ResultsSampling, HistogramKeysAndValuesSumToShots) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  constexpr size_t shots = 1024;
  ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  auto [keys, vals] = qdmi_test::getHistogram(j.job);
  ASSERT_EQ(keys.size(), vals.size());
  auto sum = 0U;
  for (const auto& v : vals) {
    sum += v;
  }
  EXPECT_EQ(sum, shots);
}

TEST(ResultsSampling, BufferTooSmallErrors) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 512), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  if (const size_t ks = qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_KEYS);
      ks > 0) {
    std::vector<char> tooSmall(ks - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_HIST_KEYS, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }

  if (const size_t vs =
          qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_VALUES);
      vs > 0) {
    std::vector<char> tooSmall(vs - 1);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_HIST_VALUES, tooSmall.size(),
                  tooSmall.data(), nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(ResultsSampling, StateAndProbRequestsAreInvalidWhenShotsPositive) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 32), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_results(
          j.job, QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, 0, nullptr,
                nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}
