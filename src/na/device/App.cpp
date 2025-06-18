/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/device/Generator.hpp"

#include <iostream>
#include <optional>
#include <vector>

namespace {
/**
 * Prints the usage information for the command line tool.
 * @param programName is the name of the program executable.
 */
auto printUsage(const std::string& programName) -> void {
  std::cout
      << "Usage: " << programName
      << " [OPTION] [JSON_FILE]\n"
         "Parse the device configuration from a JSON file and output\n"
         "corresponding definitions in a C++ header file.\n\n"
         "Options:\n"
         "  -h, --help          Show this help message and exit.\n"
         "  -v, --version       Show version information and exit.\n"
         "  -o, --output FILE   Specify the output header file. If no output\n"
         "                      file is specified, the JSON file is just\n"
         "                      parsed and no output produced.\n"
         "  -s, --schema FILE   Write a JSON schema with default values.\n"
         "                      This option does not require a JSON file.\n";
}

/**
 * Prints the version information for the command line tool.
 */
auto printVersion() -> void {
  std::cout << "MQT QDMI NA Device Generator Version " MQT_CORE_VERSION "\n";
}

/// Struct to hold the parsed command line arguments.
struct Arguments {
  std::string programName; ///< Name of the program executable
  bool help = false;       ///< Flag to indicate if help is requested
  /// Flag to indicate if version information is requested
  bool version = false;
  /// Optional output file for the generated header file
  std::optional<std::string> outputFile;
  /// Optional schema file to write the JSON schema
  std::optional<std::string> schemaFile;
  /// Optional JSON file to parse the device configuration
  std::optional<std::string> jsonFile;
};

/**
 * Parses the command line arguments and returns an Arguments struct.
 * @param args is the vector of command line arguments.
 * @return Parsed arguments as an Arguments struct.
 * @throws std::invalid_argument if the value after an option is missing.
 */
auto parseArguments(const std::vector<std::string>& args) -> Arguments {
  Arguments arguments;
  arguments.programName = args.front();
  size_t i = 1;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      arguments.help = true;
      ++i;
    } else if (arg == "-v" || arg == "--version") {
      arguments.version = true;
      ++i;
    } else if (arg == "-o" || arg == "--output") {
      ++i;
      if (i < args.size()) {
        arguments.outputFile = args.at(i);
      } else {
        throw std::invalid_argument("Missing output file argument.");
      }
      ++i;
    } else if (arg == "-s" || arg == "--schema") {
      ++i;
      if (i < args.size()) {
        arguments.schemaFile = args.at(i);
      } else {
        throw std::invalid_argument("Missing schema file argument.");
      }
      ++i;
    } else {
      arguments.jsonFile = arg;
      ++i;
    }
  }
  return arguments;
}
} // namespace

/**
 * @brief Main function that parses command-line-arguments and processes the
 * JSON.
 * @details This function handles the command line arguments, checks for help
 * and version flags, and processes the JSON file or schema file as specified by
 * the user. Either a JSON file or a schema file must be provided. If no output
 * file is specified, the JSON file is parsed but no header file is generated.
 *
 * @param argc is the number of command line arguments.
 * @param argv is the array of command line arguments.
 */
int main(int argc, char* argv[]) {
  const auto& args =
      parseArguments(std::vector<std::string>(argv, argv + argc));
  if (args.help) {
    printUsage(args.programName);
    return 0;
  }
  if (args.version) {
    printVersion();
    return 0;
  }
  if (!args.jsonFile && !args.schemaFile) {
    std::cerr << "Error: No JSON file or schema file specified.\n";
    printUsage(args.programName);
    return 1;
  }
  if (args.schemaFile) {
    try {
      na::writeJsonSchema(*args.schemaFile);
    } catch (const std::runtime_error& e) {
      std::cerr << "Error writing JSON schema: " << e.what() << '\n';
      return 1;
    }
  }
  if (args.jsonFile) {
    na::Device device;
    try {
      device = na::readJsonFile(*args.jsonFile);
    } catch (const std::runtime_error& e) {
      std::cerr << "Error parsing JSON file: " << e.what() << '\n';
      return 1;
    }
    if (args.outputFile) {
      try {
        na::writeHeaderFile(device, *args.outputFile);
      } catch (const std::runtime_error& e) {
        std::cerr << "Error writing header file: " << e.what() << '\n';
        return 1;
      }
    }
  }
  return 0;
}
