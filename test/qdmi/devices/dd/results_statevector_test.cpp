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
#include "mqt_ddsim_qdmi/device.h"

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <qdmi/constants.h>
#include <vector>

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
