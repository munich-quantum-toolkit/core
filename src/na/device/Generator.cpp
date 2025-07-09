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

#include "na/device/Generator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <nlohmann/detail/exceptions.hpp>
#include <nlohmann/json.hpp>
#include <ostream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace na {
namespace {
/**
 * @brief Populates all array fields of the JSON object with one default entry.
 * @param device is the JSON object representing the device.
 * @note This is a recursive auxiliary function used by @ref writeJSONSchema.
 */
auto populateArrayFields(Device& device) -> void {
  device.traps.emplace_back().sublatticeOffsets.emplace_back();
  device.globalMultiQubitOperations.emplace_back();
  device.globalSingleQubitOperations.emplace_back();
  device.localMultiQubitOperations.emplace_back();
  device.localSingleQubitOperations.emplace_back();
  device.shuttlingUnits.emplace_back();
}

/**
 * @brief Increments the indices in lexicographic order.
 * @details This function increments the first index that is less than its
 * limit, resets all previous indices to zero.
 * @param indices The vector of indices to increment.
 * @param limits The limits for each index.
 * @returns true if the increment was successful, false if all indices have
 * reached their limits.
 */
[[nodiscard]] auto increment(std::vector<int64_t>& indices,
                             const std::vector<int64_t>& limits) -> bool {
  size_t i = 0;
  for (; i < indices.size() && indices[i] == limits[i]; ++i) {
  }
  if (i == indices.size()) {
    // all indices are at their limits
    return false;
  }
  for (size_t j = 0; j < i; ++j) {
    indices[j] = 0; // Reset all previous indices
  }
  ++indices[i]; // Increment the next index
  return true;
}

/**
 * Computes the time unit factor based on the device configuration.
 * @param device is the Protobuf message containing the device configuration.
 * @returns a factor every time value must be multiplied with to convert it to
 * microseconds.
 */
[[nodiscard]] auto getTimeUnit(const Device& device) -> double {
  if (device.timeUnit.unit == "us") {
    return static_cast<double>(device.timeUnit.value);
  }
  if (device.timeUnit.unit == "ns") {
    return static_cast<double>(device.timeUnit.value) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported time unit: " << device.timeUnit.unit;
  throw std::runtime_error(ss.str());
}

/**
 * Computes the length unit factor based on the device configuration.
 * @param device is the Protobuf message containing the device configuration.
 * @returns a factor every length value must be multiplied with to convert it to
 * micrometers.
 */
[[nodiscard]] auto getLengthUnit(const Device& device) -> double {
  if (device.lengthUnit.unit == "um") {
    return static_cast<double>(device.lengthUnit.value);
  }
  if (device.lengthUnit.unit == "nm") {
    return static_cast<double>(device.lengthUnit.value) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported length unit: " << device.lengthUnit.unit;
  throw std::runtime_error(ss.str());
}

/**
 * @brief Writes the name from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 * @param os The output stream to write the sites to.
 */
auto writeName(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_NAME(var) var = \"" << device.name << "\"\n";
}

/**
 * @brief Writes the qubits number from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 * @param os The output stream to write the sites to.
 */
auto writeQubitsNum(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_QUBITSNUM(var) var = " << device.numQubits
     << "UL\n";
}

/**
 * @brief Writes the sites from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 * @param os The output stream to write the sites to.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  size_t count = 0;
  size_t moduleCount = 0;
  const auto lengthUnit = getLengthUnit(device);

  os << "#define INITIALIZE_SITES(var) var.clear();";
  for (const auto& lattice : device.traps) {
    size_t subModuleCount = 0;

    const auto latticeOriginX = lattice.latticeOrigin.x;
    const auto latticeOriginY = lattice.latticeOrigin.y;
    const auto baseVector1X = lattice.latticeVector1.x;
    const auto baseVector1Y = lattice.latticeVector1.y;
    const auto baseVector2X = lattice.latticeVector2.x;
    const auto baseVector2Y = lattice.latticeVector2.y;
    const auto extentOriginX = lattice.extent.origin.x;
    const auto extentOriginY = lattice.extent.origin.y;
    const auto extentWidth = static_cast<int64_t>(lattice.extent.size.width);
    const auto extentHeight = static_cast<int64_t>(lattice.extent.size.height);

    // approximate indices of the bottom left corner
    const auto& [bottomLeftI, bottomLeftJ] = solve2DLinearEquation<int64_t>(
        baseVector1X, baseVector2X, baseVector1Y, baseVector2Y,
        extentOriginX - latticeOriginX, extentOriginY - latticeOriginY);

    // approximate indices of the bottom right corner
    const auto& [bottomRightI, bottomRightJ] = solve2DLinearEquation<int64_t>(
        baseVector1X, baseVector2X, baseVector1Y, baseVector2Y,
        extentOriginX + extentWidth - latticeOriginX,
        extentOriginY - latticeOriginY);

    // approximate indices of the top left corner
    const auto& [topLeftI, topLeftJ] = solve2DLinearEquation<int64_t>(
        baseVector1X, baseVector2X, baseVector1Y, baseVector2Y,
        extentOriginX - latticeOriginX,
        extentOriginY + extentHeight - latticeOriginY);

    // approximate indices of the top right corner
    const auto& [topRightI, topRightJ] = solve2DLinearEquation<int64_t>(
        baseVector1X, baseVector2X, baseVector1Y, baseVector2Y,
        extentOriginX + extentWidth - latticeOriginX,
        extentOriginY + extentHeight - latticeOriginY);

    const auto minI = static_cast<int64_t>(
        std::floor(std::min({bottomLeftI, bottomRightI, topLeftI, topRightI})));
    const auto minJ = static_cast<int64_t>(
        std::floor(std::min({bottomLeftJ, bottomRightJ, topLeftJ, topRightJ})));
    const auto maxI = static_cast<int64_t>(
        std::floor(std::max({bottomLeftI, bottomRightI, topLeftI, topRightI})));
    const auto maxJ = static_cast<int64_t>(
        std::floor(std::max({bottomLeftJ, bottomRightJ, topLeftJ, topRightJ})));

    const std::vector limits{maxI, maxJ};
    std::vector indices{minI, minJ};
    for (bool loop = true; loop;
         loop = increment(indices, limits), ++subModuleCount) {
      // For every sublattice offset, add a site for repetition indices
      for (const auto& offset : lattice.sublatticeOffsets) {
        const auto id = count++;
        auto x = latticeOriginX + offset.x;
        auto y = latticeOriginY + offset.y;
        x += indices[0] * baseVector1X;
        y += indices[0] * baseVector1Y;
        x += indices[1] * baseVector2X;
        y += indices[1] * baseVector2Y;
        if (extentOriginX <= x && x < extentOriginX + extentWidth &&
            extentOriginY <= y && y < extentOriginY + extentHeight) {
          // Only add the site if it is within the extent of the lattice
          os << "\\\n  "
                "var.emplace_back(std::make_unique<MQT_NA_QDMI_Site_impl_d>("
                "MQT_NA_QDMI_Site_impl_d{"
             << id << ", " << moduleCount << ", " << subModuleCount << ", "
             << static_cast<double>(x) * lengthUnit << ", "
             << static_cast<double>(y) * lengthUnit << "}));";
        }
      }
    }
    ++moduleCount;
  }
  os << "\n";
}

/**
 * @brief Imports the operations from the Protobuf message into the device.
 * @param device The Protobuf message containing the device configuration.
 * @param timeUnit The time unit to use for the operations.
 * @param os The output stream to write the sites to.
 */
auto writeOperations(const Device& device, const double timeUnit,
                     std::ostream& os) -> void {
  os << "#define INITIALIZE_OPERATIONS(var) var.clear();";
  for (const auto& operation : device.globalSingleQubitOperations) {
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << "\", OperationType::GLOBAL_SINGLE_QUBIT, "
       << operation.numParameters << ", 1, "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << "}));";
  }
  for (const auto& operation : device.globalMultiQubitOperations) {
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << "\", OperationType::GLOBAL_MULTI_QUBIT, "
       << operation.numParameters << ", " << operation.numQubits << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << "}));";
  }
  for (const auto& operation : device.localSingleQubitOperations) {
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << "\", OperationType::LOCAL_SINGLE_QUBIT, "
       << operation.numParameters << ", 1, "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << "}));";
  }
  for (const auto& operation : device.localMultiQubitOperations) {
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << "\", OperationType::LOCAL_MULTI_QUBIT, "
       << operation.numParameters << ", " << operation.numQubits << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << "}));";
  }
  for (const auto& operation : device.shuttlingUnits) {
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << " (Load)\", OperationType::SHUTTLING_LOAD, "
       << operation.numParameters << ", 0, "
       << static_cast<double>(operation.loadDuration) * timeUnit << ", "
       << operation.loadFidelity << "}));";
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << " (Move)\", OperationType::SHUTTLING_MOVE, "
       << operation.numParameters << ", 0, 0, 0}));";
    os << "\\\n"
          "  var.emplace_back(std::make_unique<MQT_NA_QDMI_Operation_impl_d>("
          "MQT_NA_QDMI_Operation_impl_d{\""
       << operation.name << " (Store)\", OperationType::SHUTTLING_STORE, "
       << operation.numParameters << ", 0, "
       << static_cast<double>(operation.storeDuration) * timeUnit << ", "
       << operation.storeFidelity << "}));";
  }
  os << "\n";
}

/**
 * @brief Writes the decoherence times from the Protobuf message.
 * @param device The Protobuf message containing the device configuration.
 * @param timeUnit The time unit to use for the decoherence times.
 * @param os The output stream to write the sites to.
 */
auto writeDecoherenceTimes(const Device& device, const double timeUnit,
                           std::ostream& os) -> void {
  os << "#define INITIALIZE_T1(var) var = "
     << static_cast<double>(device.decoherenceTimes.t1) * timeUnit << ";\n";
  os << "#define INITIALIZE_T2(var) var = "
     << static_cast<double>(device.decoherenceTimes.t2) * timeUnit << ";\n";
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

  // Write to output stream
  os << json;
}

auto writeJSONSchema(const std::string& path) -> void {
  // Write to file
  std::ofstream ofs(path);
  if (ofs.is_open()) {
    writeJSONSchema(ofs);
    ofs.close();
    SPDLOG_INFO("JSON template written to {}", path);
  } else {
    std::stringstream ss;
    ss << "Failed to open file for writing: " << path;
    throw std::runtime_error(ss.str());
  }
}

[[nodiscard]] auto readJSON(std::istream& is) -> Device {
  // Read the device configuration from the input stream
  nlohmann::json json;
  try {
    is >> json;
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
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + std::string(path));
  }
  const auto& device = readJSON(ifs);
  ifs.close();
  return device;
}

auto writeHeader(const Device& device, std::ostream& os) -> void {
  os << "#pragma once\n\n";
  const auto timeUnit = getTimeUnit(device);
  writeName(device, os);
  writeQubitsNum(device, os);
  writeSites(device, os);
  writeOperations(device, timeUnit, os);
  writeDecoherenceTimes(device, timeUnit, os);
}

auto writeHeader(const Device& device, const std::string& path) -> void {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::stringstream ss;
    ss << "Failed to open header file for writing: " << path;
    throw std::runtime_error(ss.str());
  }
  writeHeader(device, ofs);
  ofs.close();
  SPDLOG_INFO("Header file written to {}", path);
}
} // namespace na
