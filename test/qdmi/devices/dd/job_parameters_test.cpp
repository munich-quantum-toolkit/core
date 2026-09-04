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

namespace {
constexpr auto QASM3_FORMAT = QDMI_PROGRAM_FORMAT_QASM3;
constexpr auto QASM2_FORMAT = QDMI_PROGRAM_FORMAT_QASM2;
constexpr auto QIR_BINARY_FORMAT = QDMI_PROGRAM_FORMAT_QIRBASEMODULE;
} // namespace

TEST(JobParameters, SetAndQueryBasics) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);

  // Shots
  constexpr size_t shots = 256;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_set_parameter(
          j.job, QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM, sizeof(size_t), &shots),
      QDMI_SUCCESS);

  // Query properties reflect parameters
  QDMI_Program_Format fmtOut{};
  size_t size = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &fmtOut, &size),
            QDMI_SUCCESS);
  EXPECT_EQ(size, sizeof(QDMI_Program_Format));
  EXPECT_EQ(fmtOut, QDMI_PROGRAM_FORMAT_QASM3);

  size_t shotsOut = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, sizeof(size_t),
                &shotsOut, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(shotsOut, shots);

  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          j.job, QDMI_DEVICE_JOB_PROPERTY_QUEUEPOSITION, 0, nullptr, nullptr),
      QDMI_ERROR_NOTSUPPORTED);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_ID, 0, nullptr, &size),
            QDMI_SUCCESS);
  EXPECT_GT(size, 0U);
  std::string id(size - 1, '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_ID, size, id.data(), nullptr),
            QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, 0, nullptr, &size),
            QDMI_SUCCESS);
  EXPECT_EQ(size, strlen(qdmi_test::QASM3_BELL_SAMPLING) + 1);
  std::string program(size - 1, '\0');
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, size, program.data(),
                nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(program, qdmi_test::QASM3_BELL_SAMPLING);

  size_t programsNum = 0U;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, sizeof(size_t),
                &programsNum, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(programsNum, 1U);
}

TEST(JobParameters, RequiresACompleteProgramBeforeSubmission) {
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};

  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAMSNUM, 0, nullptr, nullptr),
      QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(job.job), QDMI_ERROR_BADSTATE);

  ASSERT_EQ(qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                job.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &QASM2_FORMAT),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, 0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(job.job), QDMI_ERROR_BADSTATE);
}

TEST(JobParameters, BinaryProgramRoundTripsExactly) {
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  constexpr std::array expected{std::byte{0}, std::byte{0xff}, std::byte{0x7f}};
  auto program = expected;

  const size_t size = program.size();
  const void* const data = program.data();
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(job.job, &QIR_BINARY_FORMAT,
                                                   1U, &size, &data),
            QDMI_SUCCESS);

  program.fill(std::byte{0});
  size_t resultSize = 0U;
  ASSERT_EQ(
      MQT_DDSIM_QDMI_device_job_query_property(
          job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, 0, nullptr, &resultSize),
      QDMI_SUCCESS);
  ASSERT_EQ(resultSize, program.size());
  std::array<std::byte, program.size()> result{};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                job.job, QDMI_DEVICE_JOB_PROPERTY_PROGRAM, result.size(),
                result.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(result, expected);
}

TEST(JobParameters, ProgramListsValidateAtomically) {
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  constexpr char program = '\0';
  constexpr std::array<size_t, 1> sizes{1U};
  const std::array<const void*, 1> programs{&program};

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                nullptr, &QASM3_FORMAT, 1U, sizes.data(), programs.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                job.job, nullptr, 1U, sizes.data(), programs.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                job.job, &QASM3_FORMAT, 0U, sizes.data(), programs.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  constexpr auto invalid = QDMI_PROGRAM_FORMAT_MAX;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                job.job, &invalid, 1U, sizes.data(), programs.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(job.job, &QASM3_FORMAT, 1U,
                                                   sizes.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(job.job, &QASM3_FORMAT, 1U,
                                                   nullptr, programs.data()),
            QDMI_ERROR_INVALIDARGUMENT);
  constexpr std::array<size_t, 2> twoSizes{1U, 1U};
  const std::array<const void*, 2> twoPrograms{&program, &program};
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(job.job, &QASM3_FORMAT, 2U,
                                                   twoSizes.data(),
                                                   twoPrograms.data()),
            QDMI_ERROR_NOTSUPPORTED);
}

TEST(JobParameters, RejectsUnterminatedTextProgram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  const size_t size = strlen(qdmi_test::QASM3_BELL_SAMPLING);
  const void* const program = qdmi_test::QASM3_BELL_SAMPLING;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(j.job, &QASM3_FORMAT, 1U,
                                                   &size, &program),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobParameters, RejectsInteriorNullInTextProgram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  constexpr auto program = std::to_array("OPENQASM 3.0;\0garbage");
  const size_t size = program.size();
  const void* const data = program.data();
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(j.job, &QASM3_FORMAT, 1U,
                                                   &size, &data),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobParameters, ProgramFormatSupport) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  /// Supported program formats.
  for (QDMI_Program_Format fmt : {
           QDMI_PROGRAM_FORMAT_QASM2,
           QDMI_PROGRAM_FORMAT_QASM3,
           QDMI_PROGRAM_FORMAT_QIRBASESTRING,
           QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
       }) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &fmt),
              QDMI_SUCCESS);
  }

  /// A valid but unsupported format is rejected.
  constexpr QDMI_Program_Format unsupported = QDMI_PROGRAM_FORMAT_QPY;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &unsupported),
            QDMI_ERROR_NOTSUPPORTED);

  constexpr auto invalid = QDMI_PROGRAM_FORMAT_MAX;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &invalid),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobParameters, SamplingSeed) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  EXPECT_EQ(qdmi_test::setSeed(j.job, 7), QDMI_SUCCESS);
  EXPECT_EQ(qdmi_test::setSeed(j.job, 0), QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(qdmi_test::setSeed(j.job, -1), QDMI_ERROR_INVALIDARGUMENT);

  constexpr bool wrongType = true;
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_CUSTOM1, sizeof(wrongType),
                &wrongType),
            QDMI_ERROR_INVALIDARGUMENT);
}
