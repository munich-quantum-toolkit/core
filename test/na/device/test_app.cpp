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

#ifdef _WIN32
#define PLATFORM_POPEN _popen
#define PLATFORM_PCLOSE _pclose
#else
// The following are included via <stdio> but the direct include are platform-
// specific, so ignore the corresponding warning for platform-agnostic includes.
// NOLINTBEGIN(misc-include-cleaner)
#define PLATFORM_POPEN popen
#define PLATFORM_PCLOSE pclose
// NOLINTEND(misc-include-cleaner)
#endif

TEST(ExecutableTest, Version) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " --version";
  // Open a pipe to capture the output
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer{};
  buffer.fill('\0');
  std::stringstream output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  // Validate the output
  EXPECT_EQ(output.str(),
            // NOLINTNEXTLINE(misc-include-cleaner)
            "MQT QDMI NA Device Generator Version " MQT_CORE_VERSION "\n");
}

TEST(ExecutableTest, Usage) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " --help";
  // Open a pipe to capture the output
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer{};
  buffer.fill('\0');
  std::stringstream output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_FALSE(output.str().empty());
}

TEST(ExecutableTest, SchemaUsage) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " schema --help";
  // Open a pipe to capture the output
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer{};
  buffer.fill('\0');
  std::stringstream output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_TRUE(output.str().rfind("Generates a JSON schema", 0) == 0);
}

TEST(ExecutableTest, ValidateUsage) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " validate --help";
  // Open a pipe to capture the output
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer{};
  buffer.fill('\0');
  std::stringstream output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_TRUE(output.str().rfind("Validates", 0) == 0);
}

TEST(ExecutableTest, GenerateUsage) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " generate --help";
  // Open a pipe to capture the output
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Read the output
  std::array<char, 128> buffer{};
  buffer.fill('\0');
  std::stringstream output;
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    output << buffer.data();
  }
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_TRUE(output.str().rfind("Generates a header file", 0) == 0);
}

TEST(ExecutableTest, RoundTrip) {
  std::string schema;
  // Capture the output of the schema command
  {
    // Command to execute
    // NOLINTNEXTLINE(misc-include-cleaner)
    const std::string command = EXECUTABLE_PATH " schema";
    // Open a pipe to capture the output
    FILE* pipe = PLATFORM_POPEN(command.c_str(), "r");
    ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
    // Read the output
    std::array<char, 128> buffer{};
    buffer.fill('\0');
    std::stringstream output;
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
      output << buffer.data();
    }
    // Close the pipe
    const int returnCode = PLATFORM_PCLOSE(pipe);
    ASSERT_EQ(returnCode, 0)
        << "Executable failed with return code: " << returnCode;
    // Print the captured output
    schema = output.str();
    std::cout << "Captured Output:\n" << schema << "\n";
  }
  // Validate the output
  {
    // Command to execute
    // NOLINTNEXTLINE(misc-include-cleaner)
    const std::string command = EXECUTABLE_PATH " validate";
    // Open a pipe to the executable with write mode
    FILE* pipe = PLATFORM_POPEN(command.c_str(), "w");
    ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
    // Write the schema to the executable's stdin
    fwrite(schema.c_str(), sizeof(char), schema.size(), pipe);
    // Close the pipe
    const int returnCode = PLATFORM_PCLOSE(pipe);
    ASSERT_EQ(returnCode, 0)
        << "Executable failed with return code: " << returnCode;
  }
}
