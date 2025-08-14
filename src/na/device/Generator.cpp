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
 * @brief Populates all array fields in the device object with default values.
 * @param device is the device object to populate.
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
 * @param device is the device object containing the time unit.
 * @returns a factor every time value must be multiplied with to convert it to
 * microseconds.
 */
[[nodiscard]] auto getTimeUnit(const Device& device) -> double {
  if (device.timeUnit.unit == "us") {
    return static_cast<double>(device.timeUnit.scaleFactor);
  }
  if (device.timeUnit.unit == "ns") {
    return static_cast<double>(device.timeUnit.scaleFactor) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported time unit: " << device.timeUnit.unit;
  throw std::runtime_error(ss.str());
}

/**
 * Computes the length unit factor based on the device configuration.
 * @param device is the device object containing the length unit.
 * @returns a factor every length value must be multiplied with to convert it to
 * micrometers.
 */
[[nodiscard]] auto getLengthUnit(const Device& device) -> double {
  if (device.lengthUnit.unit == "um") {
    return static_cast<double>(device.lengthUnit.scaleFactor);
  }
  if (device.lengthUnit.unit == "nm") {
    return static_cast<double>(device.lengthUnit.scaleFactor) * 1e-3;
  }
  std::stringstream ss;
  ss << "Unsupported length unit: " << device.lengthUnit.unit;
  throw std::runtime_error(ss.str());
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
     << "UL\n";
}

/**
 * @brief Writes the length unit from the device object.
 * @param device is the device object containing the length unit.
 * @param os is the output stream to write the length unit to.
 */
auto writeLengthUnit(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_LENGTHUNIT(var) var = {\"" << device.lengthUnit.unit
     << "\", " << device.lengthUnit.scaleFactor << "}\n";
}

/**
 * @brief Writes the sites from the device object.
 * @param device is the device object containing the sites configuration.
 * @param os is the output stream to write the sites to.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  size_t count = 0;
  os << "#define INITIALIZE_SITES(var) var.clear()";
  // first write all zone sites
  for (const auto& operation : device.globalMultiQubitOperations) {
    const auto& region = operation.region;
    const auto id = count++;
    const auto x = region.origin.x;
    const auto y = region.origin.y;
    const auto width = region.size.width;
    const auto height = region.size.height;
    os << ";\\\n  "
          "const MQT_NA_QDMI_Site globalOp"
       << operation.name
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << ", " << x << ", " << y << ", " << width << ", " << height
       << ")).get()";
  }
  for (const auto& operation : device.globalSingleQubitOperations) {
    const auto& region = operation.region;
    const auto id = count++;
    const auto x = region.origin.x;
    const auto y = region.origin.y;
    const auto width = region.size.width;
    const auto height = region.size.height;
    os << ";\\\n  "
          "MQT_NA_QDMI_Site globalOp"
       << operation.name
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << "U, " << x << ", " << y << ", " << width << "U, " << height
       << "U)).get()";
  }
  for (const auto& shuttlingUnit : device.shuttlingUnits) {
    const auto& region = shuttlingUnit.region;
    const auto id = count++;
    const auto x = region.origin.x;
    const auto y = region.origin.y;
    const auto width = region.size.width;
    const auto height = region.size.height;
    os << ";\\\n  "
          "MQT_NA_QDMI_Site shuttlingUnit"
       << shuttlingUnit.id
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << "U, " << x << ", " << y << ", " << width << "U, " << height
       << "U)).get()";
  }
  for (const auto& operation : device.localSingleQubitOperations) {
    os << ";\\\n  std::unordered_set<MQT_NA_QDMI_Site> localOp"
       << operation.name << "Sites";
  }
  for (const auto& operation : device.localMultiQubitOperations) {
    os << ";\\\n  std::unordered_set<MQT_NA_QDMI_Site> localOp"
       << operation.name << "Sites";
  }
  // then write all regular sites
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
          os << ";\\\n  "
                "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueSite("
             << id << "U, " << moduleCount << "U, " << subModuleCount << "U, "
             << x << ", " << y << "))";
          for (const auto& operation : device.localSingleQubitOperations) {
            if (x >= operation.region.origin.x &&
                x <= operation.region.origin.x + operation.region.size.width &&
                y >= operation.region.origin.y &&
                y <= operation.region.origin.y + operation.region.size.height) {
              os << ";\\\n  localOp" << operation.name
                 << "Sites.emplace(var.back().get())";
            }
          }
        }
      }
    }
    ++moduleCount;
  }
  os << "\n";
}

/**
 * @brief Writes the operations from the device object.
 * @param device is the device object containing the operations configuration.
 * @param timeUnit is the time unit to use for the operations durations.
 * @param os is the output stream to write the operations to.
 */
auto writeOperations(const Device& device, const double timeUnit,
                     std::ostream& os) -> void {
  os << "#define INITIALIZE_OPERATIONS(var) var.clear()";
  for (const auto& operation : device.globalSingleQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueGlobalSingleQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << ", globalOp" << operation.name << "ZoneSite))";
  }
  for (const auto& operation : device.globalMultiQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueGlobalMultiQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.numQubits << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << ", " << operation.interactionRadius << ", "
       << operation.blockingRadius << ", globalOp" << operation.name
       << "ZoneSite))";
  }
  for (const auto& operation : device.localSingleQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueLocalSingleQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << ", localOp" << operation.name << "Sites))";
  }
  for (const auto& operation : device.localMultiQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueLocalMultiQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.numQubits << ", "
       << static_cast<double>(operation.duration) * timeUnit << ", "
       << operation.fidelity << "," << operation.interactionRadius << ", "
       << operation.blockingRadius << ", localOp" << operation.name
       << "Sites))";
  }
  for (const auto& shuttlingUnit : device.shuttlingUnits) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueShuttlingLoad(\"load<"
       << shuttlingUnit.id << ">\", " << shuttlingUnit.numParameters << ", "
       << static_cast<double>(shuttlingUnit.loadDuration) * timeUnit << ", "
       << shuttlingUnit.loadFidelity << ", shuttlingUnit" << shuttlingUnit.id
       << "ZoneSite))";
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueShuttlingMove(\"move<"
       << shuttlingUnit.id << ">\", " << shuttlingUnit.numParameters
       << ", shuttlingUnit" << shuttlingUnit.id << "ZoneSite))";
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueShuttlingStore(\"store<"
       << shuttlingUnit.id << ">\", " << shuttlingUnit.numParameters << ", "
       << static_cast<double>(shuttlingUnit.storeDuration) * timeUnit << ", "
       << shuttlingUnit.storeFidelity << ", shuttlingUnit" << shuttlingUnit.id
       << "ZoneSite))";
  }
  os << "\n";
}

/**
 * @brief Writes the decoherence times from the device object.
 * @param device is the device object containing the decoherence times.
 * @param timeUnit is the time unit to use for the decoherence times.
 * @param os is the output stream to write the sites to.
 */
auto writeDecoherenceTimes(const Device& device, const double timeUnit,
                           std::ostream& os) -> void {
  os << "#define INITIALIZE_DECOHERENCETIMES(var) var = {"
     << static_cast<double>(device.decoherenceTimes.t1) * timeUnit << ", "
     << static_cast<double>(device.decoherenceTimes.t2) * timeUnit << "}\n";
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
  const auto timeUnit = getTimeUnit(device);
  writeName(device, os);
  writeQubitsNum(device, os);
  writeLengthUnit(device, os);
  writeSites(device, os);
  writeOperations(device, timeUnit, os);
  writeDecoherenceTimes(device, timeUnit, os);
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
} // namespace na
