/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Slurm.hpp"

#include "support/TestSupport.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "qdmi/client.h"
#include "qdmi/constants.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace qdmi::slurm {
namespace {

using mqt::test::ScopedEnvironmentVariable;

} // namespace

TEST(SlurmAdapterTest, AcceptsImplicitAndExplicitUnitCounts) {
  for (const auto* const value : {"test.slurm.idle", "test.slurm.idle:1"}) {
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", value);
    EXPECT_EQ(::mqt::test::value(
                  ::mqt::test::value(openDeviceFromLicense()).getStatus()),
              QDMI_DEVICE_STATUS_IDLE);
  }
}

TEST(SlurmAdapterTest, AcceptsBusyDevice) {
  const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES",
                                           "test.slurm.busy");
  EXPECT_EQ(::mqt::test::value(
                ::mqt::test::value(openDeviceFromLicense()).getStatus()),
            QDMI_DEVICE_STATUS_BUSY);
}

TEST(SlurmAdapterTest, OpensRepeatedlyAndReportsUnknownDevice) {
  const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES",
                                           "test.slurm.idle:1");
  const auto first = ::mqt::test::value(openDeviceFromLicense());
  const auto second = ::mqt::test::value(openDeviceFromLicense());
  EXPECT_EQ(::mqt::test::value(first.getId()),
            ::mqt::test::value(second.getId()));
  const ScopedEnvironmentVariable unknown("SLURM_JOB_LICENSES",
                                          "test.slurm.unknown");
  const auto error =
      ::mqt::test::diagnostic([] { return openDeviceFromLicense(); });
  ASSERT_TRUE(error);
  EXPECT_THAT(error->message,
              testing::HasSubstr("Slurm license 'test.slurm.unknown' is not a "
                                 "registered QDMI device ID"));
}

TEST(SlurmAdapterTest, RejectsMissingAndMalformedValues) {
  const std::array<std::optional<std::string>, 13> invalidValues{
      std::nullopt,
      "",
      " test.slurm.grammar",
      "test.slurm.grammar ",
      ":1",
      "test.slurm.grammar:",
      "test.slurm.grammar:+1",
      "test.slurm.grammar:-1",
      "test.slurm.grammar:1x",
      "test.slurm.grammar:0",
      "test.slurm.grammar:2",
      "test.slurm.grammar:184467440737095516160",
      "test.slurm.grammar:1:1",
  };

  for (const auto& value : invalidValues) {
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", value);
    EXPECT_TRUE(::mqt::test::errorStatus([&] {
                  return openDeviceFromLicense();
                }).has_value())
        << "value: " << value.value_or("<unset>");
  }
}

TEST(SlurmAdapterTest, RejectsUnknownRemoteAndCompoundLicenses) {
  constexpr std::array invalidValues{
      "test.slurm.unknown",          "test.slurm.single@license-server:1",
      "test.slurm.single,unrelated", "unrelated,test.slurm.single",
      "test.slurm.single|unrelated", "unrelated|test.slurm.single",
  };

  for (const auto* const value : invalidValues) {
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", value);
    EXPECT_TRUE(::mqt::test::errorStatus([&] {
                  return openDeviceFromLicense();
                }).has_value())
        << "value: " << value;
  }
}

TEST(SlurmAdapterTest, RejectsUnavailableDeviceWithIdAndStatus) {
  constexpr std::array rejectedStates{
      std::pair{"offline", "OFFLINE"},
      std::pair{"error", "ERROR"},
      std::pair{"maintenance", "MAINTENANCE"},
      std::pair{"calibration", "CALIBRATION"},
      std::pair{"max", "UNKNOWN"},
  };

  for (const auto& [configuredStatus, reportedStatus] : rejectedStates) {
    const auto id = std::string{"test.slurm."} + configuredStatus;
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", id);
    const auto error =
        ::mqt::test::diagnostic([] { return openDeviceFromLicense(); });
    ASSERT_TRUE(error);
    EXPECT_THAT(error->message,
                testing::AllOf(testing::HasSubstr(id),
                               testing::HasSubstr(reportedStatus)));
  }
}

} // namespace qdmi::slurm
