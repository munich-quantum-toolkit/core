/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>

namespace qir {
class QIRRunnerTest : public testing::TestWithParam<std::filesystem::path> {};

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    QIRRunnerTest, //< Custom instantiation name
    QIRRunnerTest, //< Test suite name
    // Parameters to test with
    ::testing::Values(QIR_FILES),
    [](const testing::TestParamInfo<std::filesystem::path>& info) {
      // Extract the last part of the file path
      auto filename = info.param.stem().string();
      return filename;
    });

TEST_P(QIRRunnerTest, QIRFile) {
  const auto& file = GetParam();
  std::ostringstream command;
  command << EXECUTABLE_PATH << " " << file;
  const auto result = std::system(command.str().c_str());
  EXPECT_EQ(result, 0);
}
} // namespace qir
