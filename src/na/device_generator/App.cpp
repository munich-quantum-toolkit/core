/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/device_generator/Generator.hpp"

#include <iostream>
#include <optional>
#include <vector>

namespace {
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
auto printVersion() -> void {
  std::cout << "MQT QDMI NA Device Generator Version 1.0.0\n";
}
struct Arguments {
  std::string programName;
  bool help = false;
  bool version = false;
  std::optional<std::string> outputFile;
  std::optional<std::string> schemaFile;
  std::optional<std::string> jsonFile;
};
auto parseArguments(const std::vector<std::string>& args) -> Arguments {
  if (args.size() < 2) {
    printUsage(args.front());
  }
  Arguments arguments;
  arguments.programName = args.front();
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args.at(i);
    if (arg == "-h" || arg == "--help") {
      arguments.help = true;
    } else if (arg == "-v" || arg == "--version") {
      arguments.version = true;
    } else if (arg == "-o" || arg == "--output") {
      if (++i < args.size()) {
        arguments.outputFile = args.at(i);
      } else {
        throw std::invalid_argument("Missing output file argument.");
      }
    } else if (arg == "-s" || arg == "--schema") {
      if (++i < args.size()) {
        arguments.schemaFile = args.at(i);
      } else {
        throw std::invalid_argument("Missing schema file argument.");
      }
    } else {
      arguments.jsonFile = arg;
    }
  }
  return arguments;
}
} // namespace

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
