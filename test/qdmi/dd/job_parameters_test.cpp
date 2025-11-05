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
 * DDSIM QDMI Device - Job parameters and properties
 */
#include "helpers/circuits.hpp"
#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/device.h"

#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <qdmi/constants.h>

TEST(JobParameters, SetAndQueryBasics) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  // Program format QASM3
  constexpr QDMI_Program_Format fmt = QDMI_PROGRAM_FORMAT_QASM3;
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
  EXPECT_EQ(fmtOut, QDMI_PROGRAM_FORMAT_QASM3);

  size_t shotsOut = 0;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_query_property(
                j.job, QDMI_DEVICE_JOB_PROPERTY_SHOTSNUM, sizeof(size_t),
                &shotsOut, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(shotsOut, shots);

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
            QDMI_ERROR_NOTSUPPORTED);
}

TEST(JobParameters, ProgramFormatSupport) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};

  // Supported
  for (QDMI_Program_Format fmt :
       {QDMI_PROGRAM_FORMAT_QASM2, QDMI_PROGRAM_FORMAT_QASM3}) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &fmt),
              QDMI_SUCCESS);
  }

  // Unsupported → NOTSUPPORTED
  for (QDMI_Program_Format fmt : {
           QDMI_PROGRAM_FORMAT_QIRBASESTRING,
           QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
           QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
           QDMI_PROGRAM_FORMAT_CALIBRATION,
           QDMI_PROGRAM_FORMAT_CUSTOM1,
           QDMI_PROGRAM_FORMAT_CUSTOM2,
           QDMI_PROGRAM_FORMAT_CUSTOM3,
           QDMI_PROGRAM_FORMAT_CUSTOM4,
           QDMI_PROGRAM_FORMAT_CUSTOM5,
       }) {
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_set_parameter(
                  j.job, QDMI_DEVICE_JOB_PARAMETER_PROGRAMFORMAT,
                  sizeof(QDMI_Program_Format), &fmt),
              QDMI_ERROR_NOTSUPPORTED);
  }
}
