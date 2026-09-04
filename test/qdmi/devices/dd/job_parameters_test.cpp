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

  // Program format QASM3
  constexpr QDMI_Program_Format fmt = qdmi_test::OPENQASM3;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &fmt),
            QDMI_SUCCESS);

  // Program string
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM,
                strlen(qdmi_test::QASM3_BELL_SAMPLING) + 1,
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
  EXPECT_EQ(fmtOut, qdmi_test::OPENQASM3);

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
}

TEST(JobParameters, RejectsUnterminatedTextProgram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  constexpr QDMI_Program_Format fmt = qdmi_test::OPENQASM3;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &fmt),
            QDMI_SUCCESS);

  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM,
                strlen(qdmi_test::QASM3_BELL_SAMPLING),
                qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobParameters, RejectsInteriorNullInTextProgram) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  constexpr QDMI_Program_Format fmt = qdmi_test::OPENQASM3;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &fmt),
            QDMI_SUCCESS);

  constexpr auto program = std::to_array("OPENQASM 3.0;\0garbage");
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAM, program.size(),
                program.data()),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobParameters, ProgramFormatSupport) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  /// Supported descriptors.
  for (QDMI_Program_Format fmt : {
           qdmi_test::OPENQASM2,
           qdmi_test::OPENQASM3,
           qdmi_test::QIR21_BASE_TEXT,
           qdmi_test::QIR21_BASE_BINARY,
           qdmi_test::QIR21_ADAPTIVE_TEXT,
           qdmi_test::QIR21_ADAPTIVE_BINARY,
       }) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &fmt),
              QDMI_SUCCESS);
  }

  /// An exact but unsupported descriptor is rejected.
  QDMI_Program_Format unsupported{
      .version = QDMI_MAKE_VERSION(1, 0, 0),
      .encoding = QDMI_PROGRAM_ENCODING_BINARY,
      .id = "qiskit.qpy",
      .profile = "",
  };
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                sizeof(QDMI_Program_Format), &unsupported),
            QDMI_ERROR_NOTSUPPORTED);

  auto invalid = qdmi_test::OPENQASM3;
  invalid.id[sizeof("openqasm")] = 'x';
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
