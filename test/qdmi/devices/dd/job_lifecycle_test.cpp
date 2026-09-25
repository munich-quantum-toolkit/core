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
 * DDSIM QDMI Device - Job lifecycle (submit/cancel/check/wait/free)
 */
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "helpers/circuits.hpp"
#include "helpers/controlled_job.hpp"
#include "helpers/test_utils.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <cstddef>
#include <future>
#include <utility>

TEST(JobLifecycle, SubmitAndWaitSampling) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 256), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, SubmitAndWaitStatevector) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, WaitInvalidBeforeSubmitAndIdempotentAfterDone) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  // wait before submit is invalid
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_ERROR_BADSTATE);
  // now run a quick job
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 64), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  // waiting again succeeds
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, WaitTimeoutPath) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  qdmi_test::ControlledJob running{j.job};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 1), QDMI_ERROR_TIMEOUT);
  running.release();
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
}

TEST(JobLifecycle, CancelFromCreatedAndFromRunningAndFromDone) {
  // From CREATED
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(j.job), QDMI_SUCCESS);
  }
  // Cancellation terminates the running payload without waiting for its
  // barrier.
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    qdmi_test::ControlledJob running{j.job};
    std::promise<void> entered;
    auto canceled = std::async(std::launch::async, [&] {
      entered.set_value();
      return MQT_DDSIM_QDMI_device_job_cancel(j.job);
    });
    entered.get_future().wait();
    EXPECT_EQ(canceled.wait_for(std::chrono::seconds(5)),
              std::future_status::ready);
    running.canceled();
    EXPECT_EQ(canceled.get(), QDMI_SUCCESS);
    QDMI_Job_Status status{};
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &status), QDMI_SUCCESS);
    EXPECT_EQ(status, QDMI_JOB_STATUS_CANCELED);
  }
  // From DONE/FAILED → INVALIDARGUMENT
  {
    const qdmi_test::SessionGuard s{};
    const qdmi_test::JobGuard j{s.session};
    ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                    qdmi_test::QASM3_BELL_SAMPLING),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(j.job, 1), QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(j.job),
              QDMI_ERROR_INVALIDARGUMENT);
  }
}

TEST(JobLifecycle, FreeWhileRunningWaitsForCompletion) {
  const qdmi_test::SessionGuard s{};
  qdmi_test::JobGuard j{s.session};
  qdmi_test::ControlledJob running{j.job};
  std::promise<void> entered;
  auto freed =
      std::async(std::launch::async, [&, job = std::exchange(j.job, nullptr)] {
        entered.set_value();
        MQT_DDSIM_QDMI_device_job_free(job);
      });
  entered.get_future().wait();
  EXPECT_EQ(freed.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  running.release();
  freed.get();
}

TEST(JobLifecycle, ReusesWorkersAndDestroysJobGlobals) {
  const qdmi_test::SessionGuard session{};
  std::string previous;
  for (size_t i = 0; i < 3; ++i) {
    const qdmi_test::JobGuard job{session.session};
#ifdef _WIN32
    const std::string pidFunction = "_getpid";
#else
    const std::string pidFunction = "getpid";
#endif
    std::string program = "declare i32 @";
    program += pidFunction;
    program += "()\n";
    program += R"(
      @counter = global i64 0
      declare void @__quantum__rt__int_record_output(i64, ptr)
      define i64 @main() #0 {
        %previous = load i64, ptr @counter
        %counter = add i64 %previous, 1
        store i64 %counter, ptr @counter
        %pid = call i32 @)";
    program += pidFunction;
    program += R"(()
        %wide = sext i32 %pid to i64
        call void @__quantum__rt__int_record_output(i64 %wide, ptr null)
        ret i64 %previous
      }
      attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
    )";
    ASSERT_EQ(qdmi_test::setProgram(
                  job.job, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, program),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job.job, 1), QDMI_SUCCESS);
    const bool capture = true;
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  job.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM2, sizeof(capture),
                  &capture),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::submitAndWait(job.job, 10), QDMI_SUCCESS);
    const auto size = qdmi_test::querySize(job.job, QDMI_JOB_RESULT_CUSTOM1);
    ASSERT_GT(size, 1);
    std::string output(size, '\0');
    ASSERT_EQ(
        MQT_DDSIM_QDMI_device_job_get_results(
            job.job, 0, QDMI_JOB_RESULT_CUSTOM1, size, output.data(), nullptr),
        QDMI_SUCCESS);
    if (i != 0) {
      EXPECT_EQ(output, previous);
    }
    previous = std::move(output);
  }
}
