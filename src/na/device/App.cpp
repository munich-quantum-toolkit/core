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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
/**
 * Prints the usage information for the command line tool.
 * @param programName is the name of the program executable.
 */
auto printUsage(const std::string& programName) -> void {
  std::cout
      << "Generator for turning neutral atom computer JSON specifications into "
         "header files to be used as part of a neutral atom QDMI device "
         "implementation.\n"
         "\n"
         "Usage: "
      << programName
      << " [OPTIONS] <command> [ARGS]\n"
         "\n"
         "Commands:\n"
         "  schema      Generate a default JSON schema.\n"
         "  validate    Validate a JSON specification.\n"
         "  generate    Generate a header file from a JSON specification.\n"
         "\n"
         "Options:\n"
         "  -h, --help        Show this help message and exit.\n"
         "  -v, --version     Show version information and exit.\n";
}

/**
 * Prints the usage information for the schema sub-command.
 * @param programName is the name of the program executable.
 */
auto printSchemaUsage(const std::string& programName) -> void {
  std::cout << "Generates a JSON schema with default values.\n"
               "\n"
               "Usage: "
            << programName
            << " schema [options]\n"
               "\n"
               "Options:\n"
               "  -h, --help            Show this help message and exit.\n"
               "  -o, --output <file>   Specify the output file. If no output "
               "                        file is "
               "                        specified, the schema is printed to "
               "                        stdout.\n";
}

/**
 * Prints the usage information for the validate sub-command.
 * @param programName is the name of the program executable.
 */
auto printValidateUsage(const std::string& programName) -> void {
  std::cout << "Validates a JSON specification against the schema.\n"
               "\n"
               "Usage: "
            << programName
            << " validate [options] [<json_file>]\n"
               "\n"
               "Arguments:\n"
               "  json_file       the path to the JSON file to validate. If\n"
               "                  not specified, the JSON is read from stdin.\n"
               "\n"
               "Options:\n"
               "  -h, --help      Show this help message and exit.\n";
}

/**
 * Prints the usage information for the generate sub-command.
 * @param programName is the name of the program executable.
 */
auto printGenerateUsage(const std::string& programName) -> void {
  std::cout << "Generates a header file from a JSON specification.\n"
               "\n"
               "Usage: "
            << programName
            << " generate [options] <json_file>\n"
               "\n"
               "Arguments:\n"
               "  json_file       the path to the JSON file to generate the\n"
               "                  header file from. If not specified, the\n"
               "                  JSON is read from stdin.\n"
               "\n"
               "Options:\n"
               "  -h, --help            Show this help message and exit.\n"
               "  -o, --output <file>   Specify the output file for the\n"
               "                        generated header file. If no output\n"
               "                        file is specified, the header file is\n"
               "                        printed to stdout.\n";
}

/**
 * Prints the version information for the command line tool.
 */
auto printVersion() -> void {
  // NOLINTNEXTLINE(misc-include-cleaner)
  std::cout << "MQT QDMI NA Device Generator (MQT Version " MQT_CORE_VERSION
               ")\n";
}

/// Enum to represent the different commands that can be executed.
enum class Command : uint8_t {
  Schema,   ///< Command to generate a JSON schema
  Validate, ///< Command to validate a JSON specification
  Generate  ///< Command to generate a header file from a JSON specification
};

/// Struct to hold the parsed command line arguments.
struct Arguments {
  std::string programName; ///< Name of the program executable
  bool help = false;       ///< Flag to indicate if help is requested
  /// Flag to indicate if version information is requested
  bool version = false;
  std::optional<Command> command; ///< Command to execute
};

/// Struct to hold the parsed schema command line arguments.
struct SchemaArguments {
  bool help = false; ///< Flag to indicate if help is requested
  /// Optional output file for the schema
  std::optional<std::string> outputFile;
};

/// Struct to hold the parsed validate command line arguments.
struct ValidateArguments {
  bool help = false; ///< Flag to indicate if help is requested
  /// Optional JSON file to validate
  std::optional<std::string> jsonFile;
};

/// Struct to hold the parsed generate command line arguments.
struct GenerateArguments {
  bool help = false; ///< Flag to indicate if help is requested
  /// Optional output file for the generated header file
  std::optional<std::string> outputFile;
  /// Optional JSON file to parse the device configuration
  std::optional<std::string> jsonFile;
};

/**
 * Parses the command line arguments and returns an Arguments struct.
 * @param args is the vector of command line arguments.
 * @returns the parsed arguments as an Arguments struct and an index indicating
 * the position of the first sub-command argument.
 * @throws std::invalid_argument if the value after an option is missing.
 */
auto parseArguments(const std::vector<std::string>& args)
    -> std::pair<Arguments, size_t> {
  Arguments arguments;
  arguments.programName =
      args.empty() ? "mqt-core-na-device-gen" : args.front();
  size_t i = 1;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      arguments.help = true;
    } else if (arg == "-v" || arg == "--version") {
      arguments.version = true;
    } else if (arg == "schema") {
      arguments.command = Command::Schema;
      break; // No more arguments for schema command
    } else if (arg == "validate") {
      arguments.command = Command::Validate;
      break; // No more arguments for validate command
    } else if (arg == "generate") {
      arguments.command = Command::Generate;
      break; // No more arguments for generate command
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
    ++i;
  }
  return {arguments, i + 1};
}

/**
 * Parses the command line arguments for the schema command and returns a
 * SchemaArguments struct.
 * @param args is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p args
 * @return Parsed schema arguments as a SchemaArguments struct.
 */
auto parseSchemaArguments(const std::vector<std::string>& args, size_t i)
    -> SchemaArguments {
  SchemaArguments schemaArgs;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      schemaArgs.help = true;
    } else if (arg == "-o" || arg == "--output") {
      if (++i >= args.size()) {
        throw std::invalid_argument("Missing value for output option.");
      }
      schemaArgs.outputFile = args.at(i);
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
    ++i;
  }
  return schemaArgs;
}

/**
 * Parses the command line arguments for the validate command and returns a
 * ValidateArguments struct.
 * @param args is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p args
 * @return Parsed validate arguments as a ValidateArguments struct.
 */
auto parseValidateArguments(const std::vector<std::string>& args, size_t i)
    -> ValidateArguments {
  ValidateArguments validateArgs;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      validateArgs.help = true;
    } else {
      validateArgs.jsonFile = arg;
    }
    ++i;
  }
  return validateArgs;
}

/**
 * Parses the command line arguments for the generate command and returns a
 * GenerateArguments struct.
 * @param args is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p args
 * @return Parsed generate arguments as a GenerateArguments struct.
 */
auto parseGenerateArguments(const std::vector<std::string>& args, size_t i)
    -> GenerateArguments {
  GenerateArguments generateArgs;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      generateArgs.help = true;
    } else if (arg == "-o" || arg == "--output") {
      if (++i >= args.size()) {
        throw std::invalid_argument("Missing value for output option.");
      }
      generateArgs.outputFile = args.at(i);
    } else {
      generateArgs.jsonFile = arg;
    }
    ++i;
  }
  return generateArgs;
}

/**
 * Executes the schema command, generating a JSON schema and writing it to the
 * specified output file or stdout.
 * @param progName is the name of the program executable.
 * @param argVec is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p argVec
 * @return 0 on success, 1 on error.
 */
auto executeSchemaCommand(const std::string& progName,
                          const std::vector<std::string>& argVec,
                          const size_t i) -> int {
  SchemaArguments schemaArgs;
  // parse the rest of the command line arguments for the schema command
  try {
    schemaArgs = parseSchemaArguments(argVec, i);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error parsing schema arguments: {}", e.what());
    printSchemaUsage(progName);
    return 1;
  }
  // if the help flag is set, print the schema usage information and exit
  if (schemaArgs.help) {
    printSchemaUsage(progName);
    return 0;
  }
  // generate the JSON schema and write it to the output file or stdout
  try {
    if (schemaArgs.outputFile.has_value()) {
      na::writeJSONSchema(schemaArgs.outputFile.value());
    } else {
      na::writeJSONSchema(std::cout);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error generating JSON schema: {}", e.what());
    return 1;
  }
  return 0;
}

/**
 * Executes the validate command, validating a JSON file or JSON string from
 * stdin.
 * @param progName is the name of the program executable.
 * @param argVec is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p argVec
 * @return 0 on success, 1 on error.
 */
auto executeValidateCommand(const std::string& progName,
                            const std::vector<std::string>& argVec,
                            const size_t i) -> int {
  // parse the rest of the command line arguments for the validate command
  const ValidateArguments validateArgs = parseValidateArguments(argVec, i);
  //
  if (validateArgs.help) {
    printValidateUsage(progName);
    return 0;
  }
  // validate the JSON file or the JSON string from stdin
  try {
    if (validateArgs.jsonFile.has_value()) {
      std::ignore = na::readJSON(validateArgs.jsonFile.value());
    } else {
      std::ignore = na::readJSON(std::cin);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error validating JSON: {}", e.what());
    return 1;
  }
  return 0;
}

/**
 * Executes the generate command, generating a header file from a JSON file or
 * JSON string from stdin.
 * @param progName is the name of the program executable.
 * @param argVec is the vector of command line arguments.
 * @param i is the index to the first sub-command argument within @p argVec
 * @return 0 on success, 1 on error.
 */
auto executeGenerateCommand(const std::string& progName,
                            const std::vector<std::string>& argVec,
                            const size_t i) -> int {
  GenerateArguments generateArgs;
  // parse the rest of the command line arguments for the generate command
  try {
    generateArgs = parseGenerateArguments(argVec, i);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error parsing generate arguments: {}", e.what());
    printGenerateUsage(progName);
    return 1;
  }
  // if the help flag is set, print the generate usage information and exit
  if (generateArgs.help) {
    printGenerateUsage(progName);
    return 0;
  }
  // generate the header file from the JSON specification
  try {
    na::Device device;
    // read the JSON file or the JSON string from stdin
    if (generateArgs.jsonFile.has_value()) {
      device = na::readJSON(generateArgs.jsonFile.value());
    } else {
      device = na::readJSON(std::cin);
    }
    // write the header file to the output file or stdout
    if (generateArgs.outputFile.has_value()) {
      na::writeHeader(device, generateArgs.outputFile.value());
    } else {
      na::writeHeader(device, std::cout);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error generating header file: {}", e.what());
    return 1;
  }
  return 0;
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
  std::vector<std::string> argVec;
  std::pair<Arguments, size_t> parsedArgs;
  // `main` functions should not throw exceptions. Apparently, the
  // initialization of a vector can throw exceptions, so we catch them here.
  try {
    argVec = std::vector<std::string>(argv, argv + argc);
  } catch (std::exception& e) {
    SPDLOG_ERROR("Error parsing arguments into vector: {}", e.what());
    return 1;
  }
  // parse the command line arguments up to the first sub-command
  try {
    parsedArgs = parseArguments(argVec);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error parsing arguments: {}", e.what());
    printUsage(argVec.empty() ? "mqt-core-na-device-gen" : argVec.front());
    return 1;
  }
  // unpack the parsed arguments and the index of the first sub-command here
  // because structured bindings only work with fresh variables
  const auto& [args, i] = parsedArgs;
  // print help or version information if requested
  if (args.help) {
    printUsage(args.programName);
    return 0;
  }
  // if the version flag is set, print the version information and exit
  if (args.version) {
    printVersion();
    return 0;
  }
  // if no command is specified, print the usage information
  if (!args.command.has_value()) {
    printUsage(args.programName);
    return 1;
  }
  switch (*args.command) {
  case Command::Schema:
    return executeSchemaCommand(args.programName, argVec, i);
  case Command::Validate:
    return executeValidateCommand(args.programName, argVec, i);
  case Command::Generate:
    return executeGenerateCommand(args.programName, argVec, i);
  }
  return 0;
}
