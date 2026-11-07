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
 * DDSIM QDMI Device - Job parameters and properties
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <string>

TEST(JobParameters, SetAndQueryBasics) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 256), QDMI_SUCCESS);

  QDMI_Program_Format format{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, sizeof(format),
                &format, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(format, QDMI_PROGRAM_FORMAT_QASM3);
  size_t count = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, sizeof(count),
                &count, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(count, 1);
  size_t shots = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, sizeof(shots), &shots,
                nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(shots, 256);

  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, 0, nullptr, &size),
            QDMI_SUCCESS);
  EXPECT_EQ(size, strlen(qdmi_test::QASM3_BELL_SAMPLING) + 1);
  std::string program(size, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, size,
                                                  program.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_STREQ(program.c_str(), qdmi_test::QASM3_BELL_SAMPLING);
  EXPECT_EQ(program.back(), '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, size - 1,
                                                  program.data(), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);

  QDMI_Job_Status status{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program_status(j.job, 0, &status),
            QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_CREATED);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program_status(j.job, 0, &status),
            QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_DONE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
}

TEST(JobParameters, RequiresProgramsBeforeSubmissionAndQueries) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_ERROR_BADSTATE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  QDMI_Job_Status status{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program_status(j.job, 0, &status),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 0, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
}

TEST(JobParameters, ProgramReplacementIsAtomic) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(
      qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM2, "OPENQASM 2.0;"),
      QDMI_SUCCESS);
  constexpr std::array unterminated{'x'};
  constexpr auto interiorNull = std::to_array("OPENQASM 3.0;\0garbage");
  const std::array<const void*, 2> invalidPrograms{
      unterminated.data(),
      interiorNull.data(),
  };
  constexpr std::array sizes{unterminated.size(), interiorNull.size()};
  for (size_t i = 0; i < invalidPrograms.size(); ++i) {
    const void* data = invalidPrograms[i];
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                  j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, &sizes[i], &data),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, nullptr, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 2, nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  QDMI_Program_Format format{};
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, sizeof(format),
                &format, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(format, QDMI_PROGRAM_FORMAT_QASM2);
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, 0, nullptr, &size),
            QDMI_SUCCESS);
  std::string program(size, '\0');
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_program(j.job, 0, size,
                                                  program.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_STREQ(program.c_str(), "OPENQASM 2.0;");
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
}

TEST(JobParameters, ValidatesProgramListArgumentsAndIndices) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  const size_t size = 1;
  const void* data = "";
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                nullptr, QDMI_PROGRAM_FORMAT_QASM3, 1, &size, &data),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, nullptr, &data),
            QDMI_ERROR_INVALIDARGUMENT);
  const size_t zero = 0;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, &zero, &data),
            QDMI_ERROR_INVALIDARGUMENT);
  const void* null = nullptr;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, &size, &null),
            QDMI_ERROR_INVALIDARGUMENT);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                j.job, QDMI_PROGRAM_FORMAT_QASM3, 1, &size, &data),
            QDMI_SUCCESS);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_get_program(j.job, 1, 0, nullptr, nullptr),
      QDMI_ERROR_OUTOFRANGE);
  QDMI_Job_Status status{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program_status(j.job, 1, &status),
            QDMI_ERROR_OUTOFRANGE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_program_status(j.job, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                j.job, 1, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
            QDMI_ERROR_OUTOFRANGE);
}

TEST(JobParameters, ProgramFormatSupport) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  for (const auto format :
       {QDMI_PROGRAM_FORMAT_QASM2, QDMI_PROGRAM_FORMAT_QASM3}) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(j.job, format, 1, nullptr,
                                                     nullptr),
              QDMI_SUCCESS);
  }
  for (const auto format : {
           QDMI_PROGRAM_FORMAT_QIRBASESTRING,
           QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
           QDMI_PROGRAM_FORMAT_QPY,
           QDMI_PROGRAM_FORMAT_IQMJSON,
           QDMI_PROGRAM_FORMAT_CUSTOM1,
           QDMI_PROGRAM_FORMAT_CUSTOM2,
           QDMI_PROGRAM_FORMAT_CUSTOM3,
           QDMI_PROGRAM_FORMAT_CUSTOM4,
           QDMI_PROGRAM_FORMAT_CUSTOM5,
           // Removed formats retain their numeric values as reserved inputs.
           // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
           static_cast<QDMI_Program_Format>(6),
           static_cast<QDMI_Program_Format>(9),
           // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
       }) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(j.job, format, 1, nullptr,
                                                     nullptr),
              QDMI_ERROR_NOTSUPPORTED);
  }
}
