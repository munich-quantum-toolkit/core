/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sc {
/**
 * @brief Represents a superconducting device configuration.
 * @details This struct defines the schema for the JSON representation of a
 * superconducting device configuration. This struct, including all its
 * sub-structs, implements functions to serialize and deserialize to and from
 * JSON using the nlohmann::json library.
 */
struct Device {
  /// @brief The name of the device.
  std::string name;
  /// @brief The number of qubits in the device.
  uint64_t numQubits = 0;
  /// @brief The list of couplings the device supports.
  std::vector<std::pair<uint64_t, uint64_t>> couplings;

private:
  struct Operation {
    /// @brief The name of the operation.
    std::string name;
    /// @brief The number of parameters the operation takes.
    uint64_t numParameters = 0;
    /// @brief The number of qubits the operation takes.
    uint64_t numQubits = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Operation, name, numParameters,
                                                numQubits)
  };

public:
  /// @brief The list of operations the device supports.
  std::vector<Operation> operations;

  // NOLINTNEXTLINE(misc-include-cleaner)
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Device, name, numQubits,
                                              couplings, operations)
};

/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified output stream.
 * @param os is the output stream to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails.
 */
auto writeJSONSchema(std::ostream& os) -> void;

/**
 * @brief Writes a JSON schema with default values for the device configuration
 * to the specified path.
 * @param path The path to write the JSON schema to.
 * @throws std::runtime_error if the JSON conversion fails or the file cannot be
 * opened.
 */
auto writeJSONSchema(const std::string& path) -> void;

/**
 * @brief Parses the device configuration from an input stream.
 * @param is is the input stream containing the JSON representation of the
 * device configuration.
 * @returns The parsed device configuration as a @ref sc::Device object.
 * @throws std::runtime_error if the JSON cannot be parsed.
 */
[[nodiscard]] auto readJSON(std::istream& is) -> Device;

/**
 * @brief Parses the device configuration from a JSON file.
 * @param path is the path to the JSON file containing the device configuration.
 * @returns The parsed device configuration as a @ref sc::Device object.
 * @throws std::runtime_error if the JSON file does not exist, or the JSON file
 * cannot be parsed.
 */
[[nodiscard]] auto readJSON(const std::string& path) -> Device;

/**
 * @brief Writes a header file with the device configuration to the specified
 * output stream.
 * @param device is a parsed and in-memory representation of the device.
 * @param os is the output stream to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 * @note This implementation only supports multi-qubit gates up to two
 * qubits.
 */
auto writeHeader(const Device& device, std::ostream& os) -> void;

/**
 * @brief Writes a header file with the device configuration to the specified
 * path.
 * @param device is a parsed and in-memory representation of the device.
 * @param path is the path to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 * @note This implementation only supports multi-qubit gates up to two
 * qubits.
 */
auto writeHeader(const Device& device, const std::string& path) -> void;
} // namespace sc
