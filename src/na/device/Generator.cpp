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
 * @brief Writes the duration unit from the device object.
 * @param device is the device object containing the duration unit.
 * @param os is the output stream to write the duration unit to.
 */
auto writeDurationUnit(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_DURATIONUNIT(var) var = {\""
     << device.durationUnit.unit << "\", " << device.durationUnit.scaleFactor
     << "}\n";
}

/**
 * @brief Writes the minimum atom distance from the device object.
 * @param device is the device object containing the minimum atom distance.
 * @param os is the output stream to write the minimum atom distance to.
 */
auto writeMinAtomDistance(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_MINATOMDISTANCE(var) var = "
     << device.minAtomDistance << "\n";
}

/**
 * @brief Writes the sites from the device object.
 * @param device is the device object containing the sites.
 * @param os is the output stream to write the sites to.
 */
auto writeSites(const Device& device, std::ostream& os) -> void {
  size_t count = 0;
  os << "#define INITIALIZE_SITES(var) var.clear()";
  // first write all zone sites
  for (const auto& operation : device.globalMultiQubitOperations) {
    const auto& [origin, size] = operation.region;
    const auto id = count++;
    const auto x = origin.x;
    const auto y = origin.y;
    const auto width = size.width;
    const auto height = size.height;
    os << ";\\\n  "
          "MQT_NA_QDMI_Site globalOp"
       << operation.name
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << "U, " << x << ", " << y << ", " << width << "U, " << height
       << "U)).get()";
  }
  for (const auto& operation : device.globalSingleQubitOperations) {
    const auto& [origin, size] = operation.region;
    const auto id = count++;
    const auto x = origin.x;
    const auto y = origin.y;
    const auto width = size.width;
    const auto height = size.height;
    os << ";\\\n  "
          "MQT_NA_QDMI_Site globalOp"
       << operation.name
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << "U, " << x << ", " << y << ", " << width << "U, " << height
       << "U)).get()";
  }
  for (const auto& shuttlingUnit : device.shuttlingUnits) {
    const auto& [origin, size] = shuttlingUnit.region;
    const auto id = count++;
    const auto x = origin.x;
    const auto y = origin.y;
    const auto width = size.width;
    const auto height = size.height;
    os << ";\\\n  "
          "MQT_NA_QDMI_Site shuttlingUnit"
       << shuttlingUnit.id
       << "ZoneSite = "
          "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueZone("
       << id << "U, " << x << ", " << y << ", " << width << "U, " << height
       << "U)).get()";
  }
  for (const auto& operation : device.localSingleQubitOperations) {
    os << ";\\\n  std::vector<MQT_NA_QDMI_Site> localOp" << operation.name
       << "Sites";
  }
  for (const auto& operation : device.localMultiQubitOperations) {
    os << ";\\\n  std::vector<std::pair<MQT_NA_QDMI_Site, MQT_NA_QDMI_Site>> "
          "localOp"
       << operation.name << "Sites";
  }
  // then write all regular sites
  std::vector<std::tuple<size_t, int64_t, int64_t>> sites;
  forEachRegularSites(
      device.traps,
      [&sites, &os, &device](const uint64_t id, const int64_t x,
                             const int64_t y, const uint64_t moduleId,
                             const uint64_t subModuleId) {
        sites.emplace_back(id, x, y);
        os << ";\\\n  "
              "var.emplace_back(MQT_NA_QDMI_Site_impl_d::makeUniqueSite("
           << id << "U, " << moduleId << "U, " << subModuleId << "U, " << x
           << ", " << y << "))";
        for (const auto& operation : device.localSingleQubitOperations) {
          if (x >= operation.region.origin.x &&
              x <= operation.region.origin.x +
                       static_cast<int64_t>(operation.region.size.width) &&
              y >= operation.region.origin.y &&
              y <= operation.region.origin.y +
                       static_cast<int64_t>(operation.region.size.height)) {
            os << ";\\\n  localOp" << operation.name
               << "Sites.emplace_back(var.back().get())";
          }
        }
        // this generator (same as the device implementation) only supports
        // two-qubit local operations
        for (const auto& operation : device.localMultiQubitOperations) {
          if (operation.region.origin.x <= x &&
              x <= operation.region.origin.x +
                       static_cast<int64_t>(operation.region.size.width) &&
              operation.region.origin.y <= y &&
              y <= operation.region.origin.y +
                       static_cast<int64_t>(operation.region.size.height)) {
            for (const auto& [i2, x2, y2] :
                 sites | std::views::take(sites.size() - 1)) {
              if (operation.region.origin.x <= x2 &&
                  x2 <= operation.region.origin.x +
                            static_cast<int64_t>(operation.region.size.width) &&
                  operation.region.origin.y <= y2 &&
                  y2 <=
                      operation.region.origin.y +
                          static_cast<int64_t>(operation.region.size.height) &&
                  std::hypot(x2 - x, y2 - y) <=
                      static_cast<double>(operation.interactionRadius)) {
                os << ";\\\n  localOp" << operation.name
                   << "Sites.emplace_back(var.at(" << i2
                   << ").get(), var.back().get())";
              }
            }
          }
        }
      },
      count);
  os << "\n";
}

/**
 * @brief Writes the operations from the device object.
 * @param device is the device object containing the operations configuration.
 * @param os is the output stream to write the operations to.
 */
auto writeOperations(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_OPERATIONS(var) var.clear()";
  for (const auto& operation : device.globalSingleQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueGlobalSingleQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.duration << ", " << operation.fidelity << ", globalOp"
       << operation.name << "ZoneSite))";
  }
  for (const auto& operation : device.globalMultiQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueGlobalMultiQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.numQubits << ", " << operation.duration << ", "
       << operation.fidelity << ", " << operation.interactionRadius << ", "
       << operation.blockingRadius << ", " << operation.idlingFidelity
       << ", globalOp" << operation.name << "ZoneSite))";
  }
  for (const auto& operation : device.localSingleQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueLocalSingleQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.duration << ", " << operation.fidelity << ", localOp"
       << operation.name << "Sites))";
  }
  for (const auto& operation : device.localMultiQubitOperations) {
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueLocalTwoQubit(\""
       << operation.name << "\", " << operation.numParameters << ", "
       << operation.numQubits << ", " << operation.duration << ", "
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
       << shuttlingUnit.loadDuration << ", " << shuttlingUnit.loadFidelity
       << ", shuttlingUnit" << shuttlingUnit.id << "ZoneSite))";
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueShuttlingMove(\"move<"
       << shuttlingUnit.id << ">\", " << shuttlingUnit.numParameters
       << ", shuttlingUnit" << shuttlingUnit.id << "ZoneSite, "
       << shuttlingUnit.meanShuttlingSpeed << "))";
    os << ";\\\n"
          "  "
          "var.emplace_back(MQT_NA_QDMI_Operation_impl_d::"
          "makeUniqueShuttlingStore(\"store<"
       << shuttlingUnit.id << ">\", " << shuttlingUnit.numParameters << ", "
       << shuttlingUnit.storeDuration << ", " << shuttlingUnit.storeFidelity
       << ", shuttlingUnit" << shuttlingUnit.id << "ZoneSite))";
  }
  os << "\n";
}

/**
 * @brief Writes the decoherence times from the device object.
 * @param device is the device object containing the decoherence times.
 * @param os is the output stream to write the sites to.
 */
auto writeDecoherenceTimes(const Device& device, std::ostream& os) -> void {
  os << "#define INITIALIZE_DECOHERENCETIMES(var) var = {"
     << device.decoherenceTimes.t1 << ", " << device.decoherenceTimes.t2
     << "}\n";
}

/**
 * @brief Solves a 2D linear equation system.
 * @details The equation has the following form:
 * @code
 * x1 * i + x2 * j = x0
 * y1 * i + y2 * j = y0
 * @endcode
 * The free variables are i and j.
 * @param x1 Coefficient for x in the first equation.
 * @param x2 Coefficient for y in the first equation.
 * @param y1 Coefficient for x in the second equation.
 * @param y2 Coefficient for y in the second equation.
 * @param x0 Right-hand side of the first equation.
 * @param y0 Right-hand side of the second equation.
 * @returns A pair containing the solution (x, y).
 * @throws std::runtime_error if the system has no unique solution (determinant
 * is zero).
 */
template <typename T>
[[nodiscard]] auto solve2DLinearEquation(const T x1, const T x2, const T y1,
                                         const T y2, const T x0, const T y0)
    -> std::pair<double, double> {
  // Calculate the determinant
  const auto det = static_cast<double>((x1 * y2) - (x2 * y1));
  if (det == 0) {
    throw std::runtime_error("The system of equations has no unique solution.");
  }
  // Calculate the solution
  const auto detX = static_cast<double>((x0 * y2) - (x2 * y0));
  const auto detY = static_cast<double>((x1 * y0) - (x0 * y1));
  return {detX / det, detY / det};
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
  writeName(device, os);
  writeQubitsNum(device, os);
  writeLengthUnit(device, os);
  writeDurationUnit(device, os);
  writeMinAtomDistance(device, os);
  writeSites(device, os);
  writeOperations(device, os);
  writeDecoherenceTimes(device, os);
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
auto forEachRegularSites(const std::vector<Device::Lattice>& lattices,
                         const std::function<void(uint64_t, int64_t, int64_t,
                                                  uint64_t, uint64_t)>& f,
                         const size_t startId) -> void {
  size_t count = startId;
  size_t moduleCount = 0;
  for (const auto& [latticeOrigin, latticeVector1, latticeVector2,
                    sublatticeOffsets, extent] : lattices) {
    size_t subModuleCount = 0;
    const auto& [origin, size] = extent;
    const auto extentWidth = static_cast<int64_t>(size.width);
    const auto extentHeight = static_cast<int64_t>(size.height);

    // approximate indices of the bottom left corner
    const auto& [bottomLeftI, bottomLeftJ] = solve2DLinearEquation<int64_t>(
        latticeVector1.x, latticeVector2.x, latticeVector1.y, latticeVector2.y,
        origin.x - latticeOrigin.x, origin.y - latticeOrigin.y);

    // approximate indices of the bottom right corner
    const auto& [bottomRightI, bottomRightJ] = solve2DLinearEquation<int64_t>(
        latticeVector1.x, latticeVector2.x, latticeVector1.y, latticeVector2.y,
        origin.x + extentWidth - latticeOrigin.x, origin.y - latticeOrigin.y);

    // approximate indices of the top left corner
    const auto& [topLeftI, topLeftJ] = solve2DLinearEquation<int64_t>(
        latticeVector1.x, latticeVector2.x, latticeVector1.y, latticeVector2.y,
        origin.x - latticeOrigin.x, origin.y + extentHeight - latticeOrigin.y);

    // approximate indices of the top right corner
    const auto& [topRightI, topRightJ] = solve2DLinearEquation<int64_t>(
        latticeVector1.x, latticeVector2.x, latticeVector1.y, latticeVector2.y,
        origin.x + extentWidth - latticeOrigin.x,
        origin.y + extentHeight - latticeOrigin.y);

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
      for (const auto& [xOffset, yOffset] : sublatticeOffsets) {
        const auto id = count++;
        auto x = latticeOrigin.x + xOffset;
        auto y = latticeOrigin.y + yOffset;
        x += indices[0] * latticeVector1.x;
        y += indices[0] * latticeVector1.y;
        x += indices[1] * latticeVector2.x;
        y += indices[1] * latticeVector2.y;
        if (origin.x <= x && x <= origin.x + extentWidth && origin.y <= y &&
            y <= origin.y + extentHeight) {
          // Only add the site if it is within the extent of the lattice
          f(id, x, y, moduleCount, subModuleCount);
        }
      }
    }
    ++moduleCount;
  }
}
} // namespace na
