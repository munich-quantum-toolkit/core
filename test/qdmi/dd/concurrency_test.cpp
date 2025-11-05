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
 * DDSIM QDMI Device - Concurrency tests
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/device.h"

#include <atomic>
#include <cstddef>
#include <gtest/gtest.h>
#include <qdmi/constants.h>
#include <string>
#include <thread>
#include <vector>

TEST(Concurrency, ConcurrentStatevectorReads) {
  const qdmi_test::SessionGuard s{};
  qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_State),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t stateSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE);
  ASSERT_GT(stateSize, 0U);

  auto worker = [&]() {
    std::vector<double> buf(stateSize / sizeof(double));
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE, stateSize,
                  buf.data(), nullptr),
              QDMI_SUCCESS);
  };

  std::thread t1(worker);
  std::thread t2(worker);
  std::thread t3(worker);
  std::thread t4(worker);
  t1.join();
  t2.join();
  t3.join();
  t4.join();
}

TEST(Concurrency, ConcurrentHistogramReads) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 1024), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t keysSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_KEYS);
  const size_t valsSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_HIST_VALUES);

  auto keysWorker = [&]() {
    std::string buf(keysSize > 0 ? keysSize - 1 : 0, '\0');
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(j.job, QDMI_JOB_RESULT_HIST_KEYS,
                                              keysSize, buf.data(), nullptr),
        QDMI_SUCCESS);
  };
  auto valsWorker = [&]() {
    std::vector<size_t> v(valsSize / sizeof(size_t));
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(
            j.job, QDMI_JOB_RESULT_HIST_VALUES, valsSize, v.data(), nullptr),
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
                                  qdmi_test::QASM3_Bell_Sampling),
            QDMI_SUCCESS);
  constexpr size_t shots = 4096;
  ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);

  std::atomic<bool> done{false};
  std::thread poller([&]() {
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
