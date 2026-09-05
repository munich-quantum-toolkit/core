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
 * DDSIM QDMI Device - Results: sampling (histogram keys/values)
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

namespace {

std::vector<std::string> getShots(MQT_DDSIM_QDMI_Device_Job job) {
  const size_t size = qdmi_test::querySize(job, QDMI_JOB_RESULT_SHOTS);
  std::string result(size, '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job, QDMI_JOB_RESULT_SHOTS,
                                                  size, result.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_FALSE(result.empty());
  if (!result.empty()) {
    EXPECT_EQ(result.back(), '\0');
    result.pop_back();
  }
  return qdmi_test::splitCSV(result);
}

} // namespace

TEST(ResultsSampling, QASM3Program) {
  constexpr size_t numShots = 1024;
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, numShots), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const auto shots = getShots(j.job);
  EXPECT_EQ(shots.size(), numShots);
  EXPECT_EQ(getShots(j.job), shots);

  const auto [keys, vals] = qdmi_test::getHistogram(j.job);
  ASSERT_EQ(keys.size(), vals.size());
  EXPECT_EQ(std::accumulate(vals.cbegin(), vals.cend(), size_t{0}), numShots);
  ASSERT_EQ(keys.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(
      keys, [](const auto& key) { return key == "00" || key == "11"; }));

  std::map<std::string, size_t> counts;
  for (const auto& shot : shots) {
    ++counts[shot];
  }
  ASSERT_EQ(counts.size(), keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(counts.at(keys[i]), vals.at(i));
  }
}

TEST(ResultsSampling, EmptyQASM3YieldsEmptyHistogram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3, "OPENQASM 3.0;"),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 4), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  constexpr std::array results{QDMI_JOB_RESULT_HIST_KEYS,
                               QDMI_JOB_RESULT_HIST_VALUES};
  char dummy{};
  for (const auto result : results) {
    size_t size = 1;
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(j.job, result, 0, nullptr, &size),
        QDMI_SUCCESS);
    EXPECT_EQ(size, 0U);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(j.job, result, 0, &dummy,
                                                    nullptr),
              QDMI_SUCCESS);
  }
}

TEST(ResultsSampling, BufferTooSmallErrors) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 512), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t shotsSize = qdmi_test::querySize(j.job, QDMI_JOB_RESULT_SHOTS);
  ASSERT_EQ(shotsSize, 512U * 3U);
  std::vector<char> shotsTooSmall(shotsSize - 1);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, QDMI_JOB_RESULT_SHOTS, shotsTooSmall.size(),
                shotsTooSmall.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

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
                                  qdmi_test::QASM3_BELL_SAMPLING),
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
