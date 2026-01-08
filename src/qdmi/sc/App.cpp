/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/sc/Generator.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
/**
 * @brief Writes usage information and available commands and options to stdout.
 *
 * @param programName Program executable name inserted into the Usage line.
 */
auto printUsage(const std::string& programName) -> void {
  std::cout
      << "Generator for turning superconducting computer JSON specifications "
         "into "
         "header files to be used as part of a superconducting QDMI device "
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
               "  -o, --output <file>   Specify the output file. If not\n"
               "                        specified, prints to stdout.\n";
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
            << " generate [options] [<json_file>]\n"
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
 * @brief Writes the tool's version string to standard output.
 *
 * Prints the program name and the embedded MQT core version to stdout.
 */
auto printVersion() -> void {
  // NOLINTNEXTLINE(misc-include-cleaner)
  std::cout << "MQT QDMI SC Device Generator (MQT Version " MQT_CORE_VERSION
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
 * @brief Parse top-level command-line options and locate the chosen
 * sub-command.
 *
 * @param args Vector of command-line tokens (typically argv converted to
 * std::string), where args[0] is the program name.
 * @return std::pair<Arguments, size_t> The first element is the parsed
 * top-level Arguments; the second element is the index in `args` of the
 * first argument belonging to the chosen sub-command (i.e., one past the
 * sub-command token). If no sub-command is present, the returned index
 * will be `args.size() + 1`.
 */
auto parseArguments(const std::vector<std::string>& args)
    -> std::pair<Arguments, size_t> {
  Arguments arguments;
  arguments.programName =
      args.empty() ? "mqt-core-sc-device-gen" : args.front();
  size_t i = 1;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      arguments.help = true;
    } else if (arg == "-v" || arg == "--version") {
      arguments.version = true;
    } else if (arg == "schema") {
      arguments.command = Command::Schema;
      break; // Stop top-level parsing; remaining args handled by schema parser
    } else if (arg == "validate") {
      arguments.command = Command::Validate;
      // Stop top-level parsing; remaining args handled by validate parser
      break;
    } else if (arg == "generate") {
      arguments.command = Command::Generate;
      // Stop top-level parsing; remaining args handled by generate parser
      break;
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
    ++i;
  }
  return {arguments, i + 1};
}

/**
 * @brief Parse arguments for the "schema" sub-command.
 *
 * Parses options for the schema command and produces a SchemaArguments value
 * describing whether help was requested and which output file (if any) was set.
 *
 * @param args Vector of all command-line arguments.
 * @param i Index of the first argument belonging to the schema sub-command.
 * @return SchemaArguments Struct with `help` set if help was requested and
 *         `outputFile` containing the path provided with `-o|--output`, if any.
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
 * @brief Parses arguments for the "validate" subcommand.
 *
 * @param args Vector of command-line arguments.
 * @param i Index of the first validate subcommand argument within @p args.
 * @return ValidateArguments Parsed flags and optional JSON input file path:
 * `help` is set if -h/--help was present, `jsonFile` contains the positional
 * JSON file if provided.
 * @throws std::invalid_argument if multiple JSON files are specified.
 */
auto parseValidateArguments(const std::vector<std::string>& args, size_t i)
    -> ValidateArguments {
  ValidateArguments validateArgs;
  while (i < args.size()) {
    if (const std::string& arg = args.at(i); arg == "-h" || arg == "--help") {
      validateArgs.help = true;
    } else {
      if (validateArgs.jsonFile.has_value()) {
        throw std::invalid_argument("Multiple JSON files specified");
      }
      validateArgs.jsonFile = arg;
    }
    ++i;
  }
  return validateArgs;
}

/**
 * Parse arguments for the "generate" subcommand.
 *
 * Recognizes the following arguments:
 * - `-h`, `--help`: sets the help flag.
 * - `-o <file>`, `--output <file>`: sets the output header file path.
 * - `<jsonFile>` (positional): sets the input JSON file; if omitted, input is
 * read from stdin.
 *
 * @param args Vector of command-line arguments.
 * @param i Index of the first argument belonging to the subcommand within
 * `args`.
 * @return GenerateArguments Structure with `help`, optional `outputFile`, and
 * optional `jsonFile` populated.
 * @throws std::invalid_argument If an `-o`/`--output` option is provided
 * without a following value.
 * @throws std::invalid_argument if multiple JSON files are specified.
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
      if (generateArgs.jsonFile.has_value()) {
        throw std::invalid_argument("Multiple JSON files specified");
      }
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
      sc::writeJSONSchema(schemaArgs.outputFile.value());
    } else {
      sc::writeJSONSchema(std::cout);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error generating JSON schema: {}", e.what());
    return 1;
  }
  return 0;
}

/**
 * @brief Run the "validate" subcommand to validate a JSON input.
 *
 * Parses validate-specific arguments, prints subcommand usage if the help
 * flag is set, and validates JSON read from the provided file path or from
 * standard input.
 *
 * @param progName Name of the program executable (used for usage output).
 * @param argVec Full command-line argument vector.
 * @param i Index of the first argument belonging to the validate subcommand.
 * @return int `0` on successful validation or when help was printed, `1` on
 * error.
 */
auto executeValidateCommand(const std::string& progName,
                            const std::vector<std::string>& argVec,
                            const size_t i) -> int {
  ValidateArguments validateArgs;
  // parse the rest of the command line arguments for the validate command
  try {
    validateArgs = parseValidateArguments(argVec, i);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error parsing validate arguments: {}", e.what());
    printValidateUsage(progName);
    return 1;
  }

  // if the help flag is set, print the validate usage information and exit
  if (validateArgs.help) {
    printValidateUsage(progName);
    return 0;
  }
  // validate the JSON file or the JSON string from stdin
  try {
    if (validateArgs.jsonFile.has_value()) {
      std::ignore = sc::readJSON(validateArgs.jsonFile.value());
    } else {
      std::ignore = sc::readJSON(std::cin);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error validating JSON: {}", e.what());
    return 1;
  }
  return 0;
}

/**
 * @brief Generates a C++ header from a device JSON specification (file or
 * stdin).
 *
 * Parses generate-specific arguments from argVec starting at index i, reads a
 * sc::Device from the specified JSON file or from stdin, and writes a header to
 * the specified output file or to stdout.
 *
 * @param progName Program executable name (used for usage/help output).
 * @param argVec Full command-line argument vector.
 * @param i Index in argVec of the first generate sub-command argument.
 * @return int 0 on success, 1 on error.
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
  // if the help flag is set, print the 'generate' usage information and exit
  if (generateArgs.help) {
    printGenerateUsage(progName);
    return 0;
  }
  // generate the header file from the JSON specification
  try {
    sc::Device device;
    // read the JSON file or the JSON string from stdin
    if (generateArgs.jsonFile.has_value()) {
      device = sc::readJSON(generateArgs.jsonFile.value());
    } else {
      device = sc::readJSON(std::cin);
    }
    // write the header file to the output file or stdout
    if (generateArgs.outputFile.has_value()) {
      sc::writeHeader(device, generateArgs.outputFile.value());
    } else {
      sc::writeHeader(device, std::cout);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error generating header file: {}", e.what());
    return 1;
  }
  return 0;
}
} // namespace

/**
 * @brief Parses command-line arguments, dispatches the selected subcommand
 * (schema, validate, generate), and performs the requested operation.
 *
 * The function handles global flags (help, version), prints usage/version
 * information when requested, and forwards remaining arguments to the
 * appropriate subcommand executor which performs IO and error handling.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return int Exit code: `0` on success, `1` on error.
 */
int main(int argc, char* argv[]) {
  std::vector<std::string> argVec;
  std::pair<Arguments, size_t> parsedArgs;
  // `main` functions should not throw exceptions. Apparently, the
  // initialization of a vector can throw exceptions, so we catch them here.
  try {
    argVec.reserve(static_cast<size_t>(argc));
    for (const auto& arg : std::span(argv, static_cast<size_t>(argc))) {
      argVec.emplace_back(arg);
    }
  } catch (std::exception& e) {
    SPDLOG_ERROR("Error parsing arguments into vector: {}", e.what());
    return 1;
  }
  // parse the command line arguments up to the first sub-command
  try {
    parsedArgs = parseArguments(argVec);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error parsing arguments: {}", e.what());
    printUsage(argVec.empty() ? "mqt-core-sc-device-gen" : argVec.front());
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
  // LCOV_EXCL_START
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
  // LCOV_EXCL_STOP
}
