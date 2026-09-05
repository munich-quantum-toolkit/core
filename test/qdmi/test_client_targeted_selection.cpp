/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <stdexcept>

namespace qdmi {
namespace {

TEST(ClientTargetedSelectionTest, SuccessfulRawAllocationFixesSelection) {
  EXPECT_THAT(
      [] {
        return default_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{
                MQT_CORE_QDMI_TARGETED_INIT_FAILURE_DRIVER});
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Permission denied")));
  EXPECT_THAT(
      [] {
        return Session{SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER}};
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("already selected")));
}

} // namespace
} // namespace qdmi
