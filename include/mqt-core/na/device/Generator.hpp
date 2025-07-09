/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>
#include <istream>
#include <nlohmann/detail/macro_scope.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>

namespace na {
struct Device {
  std::string name;
  uint64_t numQubits = 0;

  struct Vector {
    int64_t x = 0;
    int64_t y = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Vector, x, y);
  };
  struct Region {
    Vector origin;

    struct Size {
      uint64_t width = 0;
      uint64_t height = 0;

      NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Size, width, height);
    };
    Size size;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Region, origin, size);
  };
  struct Lattice {
    Vector latticeOrigin;
    Vector latticeVector1;
    Vector latticeVector2;
    std::vector<Vector> sublatticeOffsets;
    Region extent;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Lattice, latticeOrigin,
                                                latticeVector1, latticeVector2,
                                                sublatticeOffsets, extent);
  };
  std::vector<Lattice> traps;
  uint64_t minAtomDistance = 0;

  struct GlobalSingleQubitOperation {
    std::string name;
    Region region;
    uint64_t duration = 0;
    double fidelity = 0.0;
    uint64_t numParameters = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(GlobalSingleQubitOperation,
                                                name, region, duration,
                                                fidelity, numParameters);
  };
  std::vector<GlobalSingleQubitOperation> globalSingleQubitOperations;

  struct GlobalMultiQubitOperation {
    std::string name;
    Region region;
    double interactionRadius = 0.0;
    double blockingRadius = 0.0;
    uint64_t duration = 0;
    double fidelity = 0.0;
    double idlingFidelity = 0.0;
    uint64_t numQubits = 0;
    uint64_t numParameters = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(GlobalMultiQubitOperation, name,
                                                region, interactionRadius,
                                                blockingRadius, duration,
                                                fidelity, idlingFidelity,
                                                numQubits, numParameters);
  };
  std::vector<GlobalMultiQubitOperation> globalMultiQubitOperations;

  struct LocalSingleQubitOperation {
    std::string name;
    Region region;
    uint64_t numRows = 0;
    uint64_t numColumns = 0;
    uint64_t duration = 0;
    double fidelity = 0.0;
    uint64_t numParameters = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LocalSingleQubitOperation, name,
                                                region, numRows, numColumns,
                                                duration, fidelity,
                                                numParameters);
  };
  std::vector<LocalSingleQubitOperation> localSingleQubitOperations;

  struct LocalMultiQubitOperation {
    std::string name;
    Region region;
    double interactionRadius = 0.0;
    double blockingRadius = 0.0;
    uint64_t numRows = 0;
    uint64_t numColumns = 0;
    uint64_t duration = 0;
    double fidelity = 0.0;
    uint64_t numQubits = 0;
    uint64_t numParameters = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(LocalMultiQubitOperation, name,
                                                region, interactionRadius,
                                                blockingRadius, numRows,
                                                numColumns, duration, fidelity,
                                                numQubits, numParameters);
  };
  std::vector<LocalMultiQubitOperation> localMultiQubitOperations;

  struct ShuttlingUnit {
    std::string name;
    Region region;
    uint64_t numRows = 0;
    uint64_t numColumns = 0;
    double movingSpeed = 0.0;
    uint64_t loadDuration = 0;
    uint64_t storeDuration = 0;
    double loadFidelity = 0.0;
    double storeFidelity = 0.0;
    uint64_t numParameters = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(ShuttlingUnit, name, region,
                                                numRows, numColumns,
                                                movingSpeed, loadDuration,
                                                storeDuration, loadFidelity,
                                                storeFidelity, numParameters);
  };
  std::vector<ShuttlingUnit> shuttlingUnits;

  struct DecoherenceTimes {
    uint64_t t1 = 0;
    uint64_t t2 = 0;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(DecoherenceTimes, t1, t2);
  };
  DecoherenceTimes decoherenceTimes;

  struct Unit {
    uint64_t value = 0;
    std::string unit;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Unit, value, unit);
  };

  Unit lengthUnit;
  Unit timeUnit;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      Device, name, numQubits, traps, minAtomDistance,
      globalSingleQubitOperations, globalMultiQubitOperations,
      localSingleQubitOperations, localMultiQubitOperations, shuttlingUnits,
      decoherenceTimes, lengthUnit, timeUnit);
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
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the JSON cannot be parsed.
 */
[[nodiscard]] auto readJSON(std::istream& is) -> Device;

/**
 * @brief Parses the device configuration from a JSON file.
 * @param path is the path to the JSON file containing the device configuration.
 * @returns The parsed device configuration as a Protobuf message.
 * @throws std::runtime_error if the JSON file does not exist, or the JSON file
 * cannot be parsed.
 */
[[nodiscard]] auto readJSON(const std::string& path) -> Device;

/**
 * @brief Writes a header file with the device configuration to the specified
 * output stream.
 * @param device is the protobuf representation of the device.
 * @param os is the output stream to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
auto writeHeader(const Device& device, std::ostream& os) -> void;

/**
 * @brief Writes a header file with the device configuration to the specified
 * path.
 * @param device is the protobuf representation of the device.
 * @param path is the path to write the header file to.
 * @throws std::runtime_error if the file cannot be opened or written to.
 */
auto writeHeader(const Device& device, const std::string& path) -> void;

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

} // namespace na
