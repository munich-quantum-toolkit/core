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
 * DDSIM QDMI Device - Concurrency tests
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <atomic>
#include <complex>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

TEST(Concurrency, ConcurrentStatevectorReads) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setPrograms(
                j.job, QDMI_PROGRAM_FORMAT_QASM3,
                {qdmi_test::QASM3_ZERO,
                 "OPENQASM 3; include \"stdgates.inc\"; qubit[2] q; x q;"}),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const auto worker = [&](const size_t index) {
    const size_t dimension = index == 0 ? 2 : 4;
    const size_t nonzero = index == 0 ? 0 : 3;
    std::vector<std::complex<double>> state(dimension);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, index, QDMI_JOB_RESULT_STATEVECTOR_DENSE,
                  state.size() * sizeof(state.front()), state.data(), nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(state[nonzero], std::complex<double>(1.0, 0.0));
    std::vector<double> probabilities(dimension);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, index, QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                  probabilities.size() * sizeof(probabilities.front()),
                  probabilities.data(), nullptr),
              QDMI_SUCCESS);
    EXPECT_DOUBLE_EQ(probabilities[nonzero], 1.0);
  };

  std::thread t1(worker, 0);
  std::thread t2(worker, 0);
  std::thread t3(worker, 1);
  std::thread t4(worker, 1);
  t1.join();
  t2.join();
  t3.join();
  t4.join();
}

TEST(Concurrency, ConcurrentHistogramReads) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 1024), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t keysSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_KEYS);
  const size_t valsSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_VALUES);

  auto const keysWorker = [&] {
    std::string buf(keysSize > 0 ? keysSize - 1 : 0, '\0');
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(
            j.job, 0, QDMI_JOB_RESULT_HIST_KEYS, keysSize, buf.data(), nullptr),
        QDMI_SUCCESS);
  };
  auto const valsWorker = [&] {
    std::vector<size_t> v(valsSize / sizeof(size_t));
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(
            j.job, 0, QDMI_JOB_RESULT_HIST_VALUES, valsSize, v.data(), nullptr),
        QDMI_SUCCESS);
  };

  std::thread t1(keysWorker);
  std::thread t2(keysWorker);
  std::thread t3(valsWorker);
  std::thread t4(valsWorker);
  t1.join();
  t2.join();
  t3.join();
  t4.join();
}

TEST(Concurrency, ConcurrentCheckDuringRun) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  constexpr size_t shots = 4096;
  ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);

  std::atomic<bool> done{false};
  std::thread poller([&] {
    QDMI_Job_Status s0 = QDMI_JOB_STATUS_CREATED;
    while (!done.load()) {
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &s0), QDMI_SUCCESS);
      if (s0 == QDMI_JOB_STATUS_DONE || s0 == QDMI_JOB_STATUS_FAILED) {
        break;
      }
    }
  });

  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
  done.store(true);
  poller.join();
}
