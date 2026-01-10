/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief The MQT QDMI device generator for superconducting devices.
 */

#include "qdmi/sc/Generator.hpp"

#include <cstdint>
#include <fstream>
#include <istream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sc {
namespace {
/**
 * @brief Ensure array fields of a Device contain default entries.
 *
 * Ensures arrays that must not be empty have default elements; in particular,
 * appends a default (empty) coupling to device.couplings.
 *
 * @param device Device instance whose array fields will be populated.
 */
auto populateArrayFields(Device& device) -> void {
  device.couplings.emplace_back();
  device.operations.emplace_back();
}

/**
 * @brief Writes a C preprocessor macro that initializes a variable with the
 * device's name.
 *
 * The macro emitted is `#define INITIALIZE_NAME(var) var = "<device.name>"`.
 */
auto writeName(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_NAME(var) var = \"" << device.name << "\"\n";
}

/**
 * @brief Emits a C macro that initializes the device's qubit count.
 *
 * @param device Device whose `numQubits` value will be embedded in the macro.
 * @param os Output stream to which the macro definition is written.
 */
auto writeQubitsNum(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_QUBITSNUM(var) var = " << device.numQubits
     << "ULL\n";
}

/**
 * @brief Generates a C preprocessor macro that initializes the device's qubit
 * sites and coupling map.
 *
 * Writes a macro `INITIALIZE_SITES(var)` which clears `var`, appends
 * `numQubits` unique sites (by index), and constructs a `_couplings` vector
 * reserved to the number of couplings and populated with pairs of site pointers
 * corresponding to `device.couplings`.
 *
 * @param device Device containing `numQubits` and `couplings`.
 * @param os Output stream to which the macro definition is written.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_SITES(var) var.clear()";
  for (uint64_t id = 0; id < device.numQubits; ++id) {
    os << ";\\\n  "
          "var.emplace_back(MQT_SC_QDMI_Site_impl_d::makeUniqueSite("
       << id << "ULL))";
  }
  os << ";\\\n  std::vector<MQT_SC_QDMI_Site> _singleQubitSites";
  os << ";\\\n  _singleQubitSites.reserve(var.size())";
  os << ";\\\n  std::ranges::transform(var, "
        "std::back_inserter(_singleQubitSites), [](const "
        "std::unique_ptr<MQT_SC_QDMI_Site_impl_d>& site) { return site.get(); "
        "})";
  os << ";\\\n  std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>> "
        "_couplings";
  os << ";\\\n  _couplings.reserve(" << device.couplings.size() << ")";
  for (const auto& [i1, i2] : device.couplings) {
    os << ";\\\n  "
          "_couplings.emplace_back(var.at("
       << i1 << ").get(), var.at(" << i2 << ").get())";
  }
  os << "\n";
}

/**
 * @brief Writes the operations from the device object.
 * @param device is the device object containing the operations.
 * @param os is the output stream to write the operations to.
 */
auto writeOperations(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_OPERATIONS(var) var.clear()";
  for (const auto& operation : device.operations) {
    if (operation.numQubits == 1) {
      os << ";\\\n"
            "  "
            "var.emplace_back(MQT_SC_QDMI_Operation_impl_d::"
            "makeUniqueSingleQubit(\""
         << operation.name << "\", " << operation.numParameters
         << ", _singleQubitSites))";
    } else if (operation.numQubits == 2) {
      os << ";\\\n"
            "  "
            "var.emplace_back(MQT_SC_QDMI_Operation_impl_d::"
            "makeUniqueTwoQubit(\""
         << operation.name << "\", " << operation.numParameters
         << ", _couplings))";
    } else {
      std::ostringstream ss;
      ss << "Got operation with " << operation.numQubits << " qubits but only "
         << "single- and two-qubit operations are supported.";
      throw std::runtime_error(ss.str());
    }
  }
  os << "\n";
}

/**
 * @brief Emits a macro to initialize the device coupling map.
 *
 * Writes the C preprocessor macro `INITIALIZE_COUPLINGMAP(var)` which assigns
 * `var = _couplings`.
 *
 * @note This macro depends on the `_couplings` variable created by
 *       the INITIALIZE_SITES macro from writeSites(). The macro
 *       INITIALIZE_SITES must be invoked before INITIALIZE_COUPLINGMAP.
 *
 * @param os Output stream to write the macro definition to.
 */
auto writeCouplingMap(const Device& /* unused */, std::ostream& os) -> void {
  os << "#define INITIALIZE_COUPLINGMAP(var) var = _couplings\n";
}
} // namespace

auto writeJSONSchema(std::ostream& os) -> void {
  // Create a default device configuration
  Device device;

  // Fill each array field with default values
  populateArrayFields(device);

  // Convert the device configuration to a JSON object
  // NOLINTNEXTLINE(misc-include-cleaner)
  const nlohmann::json json = device;

  // Write to the output stream
  os << json;
}

/**
 * @brief Write a default device JSON schema to the specified file path.
 *
 * Opens the file at `path` for writing and writes a JSON template representing
 * a default Device configuration. The function closes the file on completion.
 *
 * @param path Filesystem path where the JSON template will be written.
 * @throws std::runtime_error If the file at `path` cannot be opened for
 * writing.
 */
auto writeJSONSchema(const std::string& path) -> void {
  // Write to a file
  std::ofstream ofs(path);
  if (!ofs.good()) {
    std::stringstream ss;
    ss << "Failed to open file for writing: " << path;
    throw std::runtime_error(ss.str());
  }
  writeJSONSchema(ofs);
  ofs.close();
  SPDLOG_INFO("JSON template written to {}", path);
}

/**
 * @brief Parses a Device configuration from an input stream containing JSON.
 *
 * Reads JSON from the provided input stream and converts it into a Device.
 *
 * @param is Input stream that supplies the JSON representation of the Device.
 * @return Device constructed from the parsed JSON.
 * @throws std::runtime_error If JSON parsing fails; the exception message
 * contains parser error details.
 */
[[nodiscard]] auto readJSON(std::istream& is) -> Device {
  // Read the device configuration from the input stream
  nlohmann::json json;
  try {
    is >> json;
    // NOLINTNEXTLINE(misc-include-cleaner)
  } catch (const nlohmann::detail::parse_error& e) {
    std::stringstream ss;
    ss << "Failed to parse JSON string: " << e.what();
    throw std::runtime_error(ss.str());
  }
  return json;
}

/**
 * @brief Read a Device configuration from a JSON file.
 *
 * @param path Filesystem path to the JSON file containing the device
 * configuration.
 * @return Device parsed from the JSON file.
 * @throws std::runtime_error If the file cannot be opened or if parsing the
 * JSON fails.
 */
[[nodiscard]] auto readJSON(const std::string& path) -> Device {
  // Read the device configuration from a JSON file
  std::ifstream ifs(path);
  if (!ifs.good()) {
    throw std::runtime_error("Failed to open JSON file: " + std::string(path));
  }
  auto device = readJSON(ifs);
  ifs.close();
  return device;
}

/**
 * @brief Writes a C++ header snippet that initializes the provided Device as C
 * macros.
 *
 * Writes macros defining the device name, qubit count, site initializers, and
 * coupling map to the given output stream; the header begins with a pragma once
 * guard.
 *
 * @param device Device to serialize into header macros.
 * @param os Output stream to write the header content to.
 */
auto writeHeader(const Device& device, std::ostream& os) -> void {
  os << "#pragma once\n\n"
     << "#include <algorithm>\n"
     << "#include <iterator>\n"
     << "#include <memory>\n"
     << "#include <utility>\n"
     << "#include <vector>\n\n";
  writeName(device, os);
  writeQubitsNum(device, os);
  writeSites(device, os);
  writeCouplingMap(device, os);
  writeOperations(device, os);
}

/**
 * @brief Write a C++ header file that defines macros to initialize the given
 * Device.
 *
 * @param device Device to serialize into initialization macros (name, qubit
 * sites, coupling map).
 * @param path Filesystem path where the header will be created/overwritten.
 * @throws std::runtime_error if the file at `path` cannot be opened for
 * writing.
 */
auto writeHeader(const Device& device, const std::string& path) -> void {
  std::ofstream ofs(path);
  if (!ofs.good()) {
    std::stringstream ss;
    ss << "Failed to open header file for writing: " << path;
    throw std::runtime_error(ss.str());
  }
  writeHeader(device, ofs);
  ofs.close();
  SPDLOG_INFO("Header file written to {}", path);
}
} // namespace sc
