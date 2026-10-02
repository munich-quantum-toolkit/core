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
#include "qdmi/driver/Driver.hpp"

#include "TestUtils.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "qdmi/client.h"
#include "qdmi/constants.h"

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace qdmi::slurm {
namespace {

using mqt::test::ScopedEnvironmentVariable;

void registerStatusDevice(const std::string& id,
                          const std::string& configuredStatus) {
  static_cast<void>(qdmi::Driver::get().registerDeviceIfAbsent({
      .id = id,
      .library = MQT_CORE_QDMI_SLURM_TEST_DEVICE,
      .prefix = "TEST_SESSION",
      .session = {.custom4 = configuredStatus},
  }));
}

} // namespace

TEST(SlurmAdapterTest, AcceptsImplicitAndExplicitUnitCounts) {
  registerStatusDevice("test.slurm.idle", "idle");
  for (const auto* const value : {"test.slurm.idle", "test.slurm.idle:1"}) {
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", value);
    EXPECT_EQ(openDeviceFromLicense().getStatus(), QDMI_DEVICE_STATUS_IDLE);
  }
}

TEST(SlurmAdapterTest, AcceptsBusyDevice) {
  registerStatusDevice("test.slurm.busy", "busy");
  const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES",
                                           "test.slurm.busy");
  EXPECT_EQ(openDeviceFromLicense().getStatus(), QDMI_DEVICE_STATUS_BUSY);
}

TEST(SlurmAdapterTest, KeepsSessionsFreshAndReportsUnknownDevice) {
  registerStatusDevice("test.slurm.fresh", "idle");
  const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES",
                                           "test.slurm.fresh:1");
  const auto first = openDeviceFromLicense();
  const auto second = openDeviceFromLicense();
  EXPECT_NE(static_cast<QDMI_Device>(first), static_cast<QDMI_Device>(second));
  const ScopedEnvironmentVariable unknown("SLURM_JOB_LICENSES",
                                          "test.slurm.unknown");
  EXPECT_THAT([] { return openDeviceFromLicense(); },
              testing::ThrowsMessage<std::runtime_error>(
                  testing::HasSubstr("Slurm license 'test.slurm.unknown' is "
                                     "not a registered QDMI device ID")));
}

TEST(SlurmAdapterTest, RejectsMissingAndMalformedValues) {
  registerStatusDevice("test.slurm.grammar", "idle");
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
    EXPECT_THROW(static_cast<void>(openDeviceFromLicense()), std::runtime_error)
        << "value: " << value.value_or("<unset>");
  }
}

TEST(SlurmAdapterTest, RejectsUnknownRemoteAndCompoundLicenses) {
  registerStatusDevice("test.slurm.single", "idle");
  constexpr std::array invalidValues{
      "test.slurm.unknown",          "test.slurm.single@license-server:1",
      "test.slurm.single,unrelated", "unrelated,test.slurm.single",
      "test.slurm.single|unrelated", "unrelated|test.slurm.single",
  };

  for (const auto* const value : invalidValues) {
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", value);
    EXPECT_THROW(static_cast<void>(openDeviceFromLicense()), std::runtime_error)
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
    registerStatusDevice(id, configuredStatus);
    const ScopedEnvironmentVariable licenses("SLURM_JOB_LICENSES", id);
    EXPECT_THAT(
        [] { return openDeviceFromLicense(); },
        testing::ThrowsMessage<std::runtime_error>(testing::AllOf(
            testing::HasSubstr(id), testing::HasSubstr(reportedStatus))));
  }
}

} // namespace qdmi::slurm
