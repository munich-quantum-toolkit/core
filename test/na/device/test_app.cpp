/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <_stdio.h>
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
  char buffer[128];
  std::stringstream output;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output << buffer;
  }
  // Close the pipe
  int returnCode = pclose(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  // Optionally, validate the output
  // NOLINTNEXTLINE(misc-include-cleaner)
  EXPECT_EQ(output.str(),
            "MQT QDMI NA Device Generator Version " MQT_CORE_VERSION "\n");
}
