/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <array>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>

TEST(ExecutableTest, Version) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " --version";
  // Open a pipe to capture the output
  FILE* pipe = popen(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer;
  std::stringstream output;
  while (fgets(buffer.data(), sizeof(buffer), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = pclose(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  // Optionally, validate the output
  EXPECT_EQ(output.str(),
            // NOLINTNEXTLINE(misc-include-cleaner)
            "MQT QDMI NA Device Generator Version " MQT_CORE_VERSION "\n");
}
