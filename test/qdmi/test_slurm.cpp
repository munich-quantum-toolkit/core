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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/constants.h>

#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace qdmi::slurm {
namespace {

class ScopedSlurmLicenses {
public:
  explicit ScopedSlurmLicenses(const std::optional<std::string>& value)
      : originalValue(getValue()) {
    setValue(value);
  }

  ~ScopedSlurmLicenses() { setValue(originalValue); }

  ScopedSlurmLicenses(const ScopedSlurmLicenses&) = delete;
  ScopedSlurmLicenses& operator=(const ScopedSlurmLicenses&) = delete;
  ScopedSlurmLicenses(ScopedSlurmLicenses&&) = delete;
  ScopedSlurmLicenses& operator=(ScopedSlurmLicenses&&) = delete;

private:
  [[nodiscard]] static auto getValue() -> std::optional<std::string> {
    if (const auto* const value = std::getenv("SLURM_JOB_LICENSES")) {
      return value;
    }
    return std::nullopt;
  }

  static void setValue(const std::optional<std::string>& value) {
#ifdef _WIN32
    if (_putenv_s("SLURM_JOB_LICENSES", value.value_or("").c_str()) != 0) {
      std::abort();
    }
#else
    int result = 0;
    if (value.has_value()) {
      // NOLINTNEXTLINE(misc-include-cleaner)
      result = setenv("SLURM_JOB_LICENSES", value->c_str(), 1);
    } else {
      // NOLINTNEXTLINE(misc-include-cleaner)
      result = unsetenv("SLURM_JOB_LICENSES");
    }
    if (result != 0) {
      std::abort();
    }
#endif
  }

  std::optional<std::string> originalValue;
};

} // namespace

TEST(SlurmAdapterTest, AcceptsImplicitAndExplicitUnitCounts) {
  for (const auto* const value : {"test.slurm.idle", "test.slurm.idle:1"}) {
    const ScopedSlurmLicenses licenses(value);
    EXPECT_EQ(openDeviceFromLicense().getStatus(), QDMI_DEVICE_STATUS_IDLE);
  }
}

TEST(SlurmAdapterTest, AcceptsBusyDevice) {
  const ScopedSlurmLicenses licenses("test.slurm.busy");
  EXPECT_EQ(openDeviceFromLicense().getStatus(), QDMI_DEVICE_STATUS_BUSY);
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
    const ScopedSlurmLicenses licenses(value);
    EXPECT_THROW(static_cast<void>(openDeviceFromLicense()), std::runtime_error)
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
    const ScopedSlurmLicenses licenses(value);
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
    const ScopedSlurmLicenses licenses(id);
    EXPECT_THAT(
        [] { return openDeviceFromLicense(); },
        testing::ThrowsMessage<std::runtime_error>(testing::AllOf(
            testing::HasSubstr(id), testing::HasSubstr(reportedStatus))));
  }
}

} // namespace qdmi::slurm
