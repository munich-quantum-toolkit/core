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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <numeric>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

TEST(Concurrency, ConcurrentStatevectorReads) {
  const qdmi_test::SessionGuard s{};
  qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, qdmi_test::OPENQASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  const size_t stateSize =
      qdmi_test::querySize(j.job, QDMI_JOB_RESULT_STATEVECTOR_DENSE);
  ASSERT_GT(stateSize, 0U);

  auto const worker = [&] {
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
  ASSERT_EQ(qdmi_test::setProgram(j.job, qdmi_test::OPENQASM3,
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
        MQT_DDSIM_QDMI_device_job_get_results(j.job, QDMI_JOB_RESULT_HIST_KEYS,
                                              keysSize, buf.data(), nullptr),
        QDMI_SUCCESS);
  };
  auto const valsWorker = [&] {
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
  ASSERT_EQ(qdmi_test::setProgram(j.job, qdmi_test::OPENQASM3,
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

TEST(Concurrency, ConcurrentQIRJobsOwnTheirRuntimeState) {
  constexpr size_t numJobs = 4;
  constexpr size_t shots = 1024;
  const qdmi_test::SessionGuard session{};
  const auto program = qdmi_test::getQIRProgram("BellPairStatic.ll");
  std::vector<std::unique_ptr<qdmi_test::JobGuard>> jobs;
  jobs.reserve(numJobs);

  for (size_t i = 0; i < numJobs; ++i) {
    auto job = std::make_unique<qdmi_test::JobGuard>(session.session);
    ASSERT_EQ(
        qdmi_test::setProgram(job->job, qdmi_test::QIR21_BASE_TEXT, program),
        QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job->job, shots), QDMI_SUCCESS);
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(job->job), QDMI_SUCCESS);
    jobs.emplace_back(std::move(job));
  }

  for (const auto& job : jobs) {
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(job->job, 0), QDMI_SUCCESS);
    const auto [keys, values] = qdmi_test::getHistogram(job->job);
    ASSERT_EQ(keys.size(), values.size());
    EXPECT_EQ(std::accumulate(values.cbegin(), values.cend(), size_t{0}),
              shots);
    EXPECT_TRUE(std::ranges::all_of(
        keys, [](const auto& key) { return key == "00" || key == "11"; }));
  }
}
