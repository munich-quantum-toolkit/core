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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
  // Validate the output
  EXPECT_EQ(output.str(),
            // NOLINTNEXTLINE(misc-include-cleaner)
            "MQT QDMI NA Device Generator (MQT Version " MQT_CORE_VERSION
            ")\n");
}

TEST(ExecutableTest, MissingSubcommand) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH;
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_NE(returnCode, 0);
}

TEST(ExecutableTest, UnknownSubcommand) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " unknown";
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_NE(returnCode, 0);
}

TEST(ExecutableTest, SchemaUnknownOption) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " schema --unknown-option";
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_NE(returnCode, 0);
}

TEST(ExecutableTest, SchemaMissingFile) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " schema --output";
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_NE(returnCode, 0);
}

TEST(ExecutableTest, ValidateInvalidJson) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " validate";
  // Open a pipe to the executable
  FILE* pipe = PLATFORM_POPEN(command.c_str(), "w");
  ASSERT_NE(pipe, nullptr) << "Failed to open pipe";
  // Write the schema to the executable's stdin
  fwrite("{", sizeof(char), 2, pipe);
  // Close the pipe
  const int returnCode = PLATFORM_PCLOSE(pipe);
  EXPECT_NE(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
}

TEST(ExecutableTest, GenerateMissingFile) {
  // Command to execute
  // NOLINTNEXTLINE(misc-include-cleaner)
  const std::string command = EXECUTABLE_PATH " generate --output";
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  EXPECT_NE(returnCode, 0);
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
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
  // Print the captured output
  std::cout << "Captured Output:\n" << output.str() << "\n";
  ASSERT_EQ(returnCode, 0) << "Executable failed with return code: "
                           << returnCode;
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
    schema = output.str();
    // Print the captured output
    std::cout << "Captured Output:\n" << schema << "\n";
    ASSERT_EQ(returnCode, 0)
        << "Executable failed with return code: " << returnCode;
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

TEST(ExecutableTest, RoundTripFile) {
  // Write schema to a file
  {
    // Command to execute
    // NOLINTNEXTLINE(misc-include-cleaner)
    const std::string command = EXECUTABLE_PATH " schema --output schema.json";
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
    // Print the captured output
    std::cout << "Captured Output:\n" << output.str() << "\n";
    ASSERT_EQ(returnCode, 0)
        << "Executable failed with return code: " << returnCode;
  }
  // Validate the output
  {
    // Command to execute
    // NOLINTNEXTLINE(misc-include-cleaner)
    const std::string command = EXECUTABLE_PATH " validate schema.json";
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
    // Print the captured output
    std::cout << "Captured Output:\n" << output.str() << "\n";
    EXPECT_EQ(returnCode, 0)
        << "Executable failed with return code: " << returnCode;
  }
}
