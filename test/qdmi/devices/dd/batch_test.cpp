/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr auto ZERO = "OPENQASM 3; qubit q; bit c; c = measure q;";
constexpr auto ONE =
    "OPENQASM 3; include \"stdgates.inc\"; qubit q; bit c; x q; c = measure q;";

class BatchJobTest : public testing::Test {
protected:
  qdmi_test::SessionGuard session_;
  qdmi_test::JobGuard job_{session_.session};

  [[nodiscard]] auto setPrograms(const std::vector<std::string>& programs) const
      -> int {
    std::vector<size_t> sizes;
    std::vector<const void*> pointers;
    for (const auto& program : programs) {
      sizes.emplace_back(program.size() + 1);
      pointers.emplace_back(program.c_str());
    }
    return MQT_DDSIM_QDMI_device_job_set_programs(
        job_.job, QDMI_PROGRAM_FORMAT_QASM3, programs.size(), sizes.data(),
        pointers.data());
  }

  [[nodiscard]] auto getProgram(const size_t index) const -> std::string {
    size_t size = 0;
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program(job_.job, index, 0, nullptr,
                                                    &size),
              QDMI_SUCCESS);
    std::string program(size, '\0');
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program(job_.job, index, size,
                                                    program.data(), nullptr),
              QDMI_SUCCESS);
    if (!program.empty()) {
      program.pop_back();
    }
    return program;
  }

  [[nodiscard]] auto getStatus(const size_t index) const -> QDMI_Job_Status {
    QDMI_Job_Status status{};
    EXPECT_EQ(
        MQT_DDSIM_QDMI_device_job_get_program_status(job_.job, index, &status),
        QDMI_SUCCESS);
    return status;
  }
};

TEST_F(BatchJobTest, CopiesAndAtomicallyReplacesPrograms) {
  std::vector<std::string> programs{ZERO, ONE};
  ASSERT_EQ(setPrograms(programs), QDMI_SUCCESS);
  programs[0] = "overwritten";
  EXPECT_EQ(getProgram(0), ZERO);
  EXPECT_EQ(getProgram(1), ONE);
  const std::string invalid("OPENQASM 3;\0garbage", 19);
  EXPECT_EQ(setPrograms({ONE, invalid}), QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(getProgram(0), ZERO);
  EXPECT_EQ(getProgram(1), ONE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                job_.job, QDMI_PROGRAM_FORMAT_QASM2, 4, nullptr, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(getProgram(1), ONE);
  size_t count = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job_.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, sizeof(count),
                &count, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(count, 2);
  ASSERT_EQ(setPrograms({ONE, ZERO, ONE}), QDMI_SUCCESS);
  EXPECT_EQ(getProgram(2), ONE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_program(job_.job, 3, 0, nullptr, nullptr),
      QDMI_ERROR_OUTOFRANGE);
}

TEST_F(BatchJobTest, KeepsSuccessfulResultsWhenAnotherProgramFails) {
  ASSERT_EQ(setPrograms({ZERO, "invalid OpenQASM", ONE}), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job_.job, 32), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(job_.job, 0), QDMI_SUCCESS);
  QDMI_Job_Status status{};
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job_.job, &status), QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_FAILED);
  EXPECT_EQ(getStatus(0), QDMI_JOB_STATUS_DONE);
  EXPECT_EQ(getStatus(1), QDMI_JOB_STATUS_FAILED);
  EXPECT_EQ(getStatus(2), QDMI_JOB_STATUS_DONE);
  for (const auto index : {size_t{0}, size_t{2}}) {
    std::array<char, 2> key{};
    size_t count = 0;
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  job_.job, index, QDMI_JOB_RESULT_HIST_KEYS, key.size(),
                  key.data(), nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  job_.job, index, QDMI_JOB_RESULT_HIST_VALUES, sizeof(count),
                  &count, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(std::string(key.data()), index == 0 ? "0" : "1");
    EXPECT_EQ(count, 32);
  }
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job_.job, 1, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                job_.job, 3, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_OUTOFRANGE);
}

TEST_F(BatchJobTest, KeepsStateAndCachesSeparateAcrossPrograms) {
  ASSERT_EQ(
      setPrograms(
          {ZERO, "OPENQASM 3; include \"stdgates.inc\"; qubit[2] q; x q;"}),
      QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job_.job, 0), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(job_.job, 0), QDMI_SUCCESS);
  const auto read = [&](const size_t index, const bool probabilities) {
    const auto result = probabilities ? QDMI_JOB_RESULT_PROBABILITIES_DENSE
                                      : QDMI_JOB_RESULT_STATEVECTOR_DENSE;
    size_t size = 0;
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(job_.job, index, result, 0,
                                                    nullptr, &size),
              QDMI_SUCCESS);
    const size_t dimension = index == 0 ? 2 : 4;
    if (probabilities) {
      std::vector<double> values(size / sizeof(double));
      ASSERT_EQ(values.size(), dimension);
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job_.job, index, result, size, values.data(), nullptr),
                QDMI_SUCCESS);
      EXPECT_DOUBLE_EQ(values[index == 0 ? 0 : 3], 1.0);
    } else {
      std::vector<std::complex<double>> values(size /
                                               sizeof(std::complex<double>));
      ASSERT_EQ(values.size(), dimension);
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job_.job, index, result, size, values.data(), nullptr),
                QDMI_SUCCESS);
      EXPECT_EQ(values[index == 0 ? 0 : 3], std::complex<double>(1.0, 0.0));
    }
  };
  std::array<std::thread, 4> readers{
      std::thread(read, 0, false),
      std::thread(read, 0, true),
      std::thread(read, 1, false),
      std::thread(read, 1, true),
  };
  for (auto& reader : readers) {
    reader.join();
  }
}

TEST_F(BatchJobTest, CancelsAllProgramsBeforeSubmission) {
  ASSERT_EQ(setPrograms({ZERO, ONE}), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_cancel(job_.job), QDMI_SUCCESS);
  EXPECT_EQ(getStatus(0), QDMI_JOB_STATUS_CANCELED);
  EXPECT_EQ(getStatus(1), QDMI_JOB_STATUS_CANCELED);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_wait(job_.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(job_.job), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(setPrograms({ONE}), QDMI_ERROR_BADSTATE);
}

TEST_F(BatchJobTest, CancellationReturnsOnlyAfterAllProgramsStop) {
  ASSERT_EQ(setPrograms({qdmi_test::QASM3_HEAVY_SAMPLING, ZERO, ONE}),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job_.job, 8192), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(job_.job), QDMI_SUCCESS);
  while (getStatus(0) == QDMI_JOB_STATUS_CREATED) {
    std::this_thread::yield();
  }
  const auto result = MQT_DDSIM_QDMI_device_job_cancel(job_.job);
  QDMI_Job_Status status{};
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job_.job, &status), QDMI_SUCCESS);
  if (result == QDMI_SUCCESS) {
    EXPECT_EQ(status, QDMI_JOB_STATUS_CANCELED);
  } else {
    EXPECT_EQ(result, QDMI_ERROR_INVALIDARGUMENT);
    EXPECT_EQ(status, QDMI_JOB_STATUS_DONE);
  }
  for (size_t i = 0; i < 3; ++i) {
    const auto programStatus = getStatus(i);
    EXPECT_TRUE(programStatus == QDMI_JOB_STATUS_CANCELED ||
                programStatus == QDMI_JOB_STATUS_DONE);
    if (programStatus == QDMI_JOB_STATUS_DONE) {
      EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                    job_.job, i, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
                QDMI_SUCCESS);
    }
  }
}
} // namespace
