/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief The MQT QDMI device generator for neutral atom devices.
 */

#include "qdmi/sc/Generator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <istream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <ranges>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sc {
namespace {
/**
 * @brief Populates all array fields in the device object with default values.
 * @param device is the device object to populate.
 * @note This is a recursive auxiliary function used by @ref writeJSONSchema.
 */
auto populateArrayFields(Device& device) -> void {
  device.couplings.emplace_back();
}

/**
 * @brief Writes the name from the device object.
 * @param device is the device object containing the name.
 * @param os is the output stream to write the name to.
 */
auto writeName(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_NAME(var) var = \"" << device.name << "\"\n";
}

/**
 * @brief Writes the qubits number from the device object.
 * @param device is the device object containing the number of qubits.
 * @param os is the output stream to write the qubits number to.
 */
auto writeQubitsNum(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_QUBITSNUM(var) var = " << device.numQubits
     << "ULL\n";
}

/**
 * @brief Writes the sites from the device object.
 * @param device is the device object containing the number of sites.
 * @param os is the output stream to write the sites to.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_SITES(var) var.clear()";
  for (uint64_t id = 0; id < device.numQubits; ++id) {
    os << ";\\\n  "
          "var.emplace_back(MQT_SC_QDMI_Site_impl_d::makeUniqueSite("
       << id << "ULL))";
  }
  os << ";\\\n  std::vector<std::pair<MQT_SC_QDMI_Site, MQT_SC_QDMI_Site>> "
        "_couplings;";
  os << ";\\\n  _couplings.reserve(" << device.couplings.size() << ");";
  for (const auto& [i1, i2] : device.couplings) {
    os << ";\\\n  "
          "_couplings.emplace_back(var.at("
       << i1 << ").get(), var.at(" << i2 << ").get())";
  }
  os << "\n";
}

/**
 * @brief Writes the sites from the device object.
 * @param os is the output stream to write the sites to.
 */
auto writeCouplingMap(const Device& /* unused */, std::ostream& os) -> void {
  os << "#define INITIALIZE_COUPLINGMAP(var) var = std::move(_couplings)\n";
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

[[nodiscard]] auto readJSON(const std::string& path) -> Device {
  // Read the device configuration from a JSON file
  std::ifstream ifs(path);
  if (!ifs.good()) {
    throw std::runtime_error("Failed to open JSON file: " + std::string(path));
  }
  const auto& device = readJSON(ifs);
  ifs.close();
  return device;
}

auto writeHeader(const Device& device, std::ostream& os) -> void {
  os << "#pragma once\n\n";
  writeName(device, os);
  writeQubitsNum(device, os);
  writeSites(device, os);
  writeCouplingMap(device, os);
}

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
