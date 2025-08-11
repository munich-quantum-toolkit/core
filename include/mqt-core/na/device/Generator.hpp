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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>

namespace na {
/**
 * @brief Represents a neutral atom device configuration.
 * @details This struct defines the schema for the JSON representation of a
 * neutral atom device configuration. This struct, including all its
 * sub-structs, implements functions to serialize and deserialize to and from
 * JSON using the nlohmann::json library.
 * @note All duration and length values are in multiples of the time unit and
 * the length unit, respectively.
 */
struct Device {
  /// @brief The name of the device.
  std::string name;
  /// @brief The number of qubits in the device.
  uint64_t numQubits = 0;

  /// @brief Represents a 2D-vector.
  struct Vector {
    /// @brief The x-coordinate of the vector.
    int64_t x = 0;
    /// @brief The y-coordinate of the vector.
    int64_t y = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Vector, x, y)
  };
  /// @brief Represents a region in the device.
  struct Region {
    /// @brief The origin of the region.
    Vector origin;

    /// @brief The size of the region.
    struct Size {
      /// @brief The width of the region.
      uint64_t width = 0;
      /// @brief The height of the region.
      uint64_t height = 0;

      // NOLINTNEXTLINE(misc-include-cleaner)
      NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Size, width, height)
    };
    /// @brief The size of the region.
    Size size;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Region, origin, size)
  };
  /// @brief Represents a lattice of traps in the device.
  struct Lattice {
    /// @brief The origin of the lattice.
    Vector latticeOrigin;
    /**
     * @brief The first lattice vector.
     * @details Multiples of this vector are added to the lattice origin to
     * create the lattice structure.
     */
    Vector latticeVector1;
    /**
     * @brief The second lattice vector.
     * @details Multiples of this vector are added to the lattice origin and
     * multiples of the first lattice vector to create the lattice structure.
     */
    Vector latticeVector2;
    /**
     * @brief The offsets for each sublattice.
     * @details The actual locations of traps are calculated by adding the
     * each offset to the points in the lattice defined by the lattice
     * vectors, i.e., for each sublattice offset `offset` and indices `i` and
     * `j`, the trap location is `latticeOrigin + i * latticeVector1 + j *
     * latticeVector2 + offset`.
     */
    std::vector<Vector> sublatticeOffsets;
    /**
     * @brief The extent of the lattice.
     * @details The extent defines the boundary of the lattice in which traps
     * are placed. Only traps of the lattice that are within this extent
     * are considered valid.
     */
    Region extent;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Lattice, latticeOrigin,
                                                latticeVector1, latticeVector2,
                                                sublatticeOffsets, extent)
  };
  /// @brief The list of lattices (trap areas) in the device.
  std::vector<Lattice> traps;
  /**
   * @brief The minimum distance between atoms in the device that must be
   * respected.
   */
  uint64_t minAtomDistance = 0;

private:
  struct Operation {
    /// @brief The name of the operation.
    std::string name;
    /// @brief The region in which the operation can be performed.
    Region region;
    /// @brief The duration of the operation.
    uint64_t duration = 0;
    /// @brief The fidelity of the operation.
    double fidelity = 0.0;
    /// @brief The number of parameters the operation takes.
    uint64_t numParameters = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Operation, name, region,
                                                duration, fidelity);
  };

public:
  /// @brief Represents a global single-qubit operation.
  struct GlobalSingleQubitOperation : Operation {};
  /// @brief The list of global single-qubit operations supported by the device.
  std::vector<GlobalSingleQubitOperation> globalSingleQubitOperations;

  /// @brief Represents a global multi-qubit operation.
  struct GlobalMultiQubitOperation : Operation {
    /**
     * @brief The interaction radius of the operation within which two qubits
     * can interact.
     */
    double interactionRadius = 0.0;
    /**
     * @brief The blocking radius of the operation within which no other
     * operation can be performed to avoid interference.
     */
    double blockingRadius = 0.0;
    /// @brief The fidelity of the operation when no qubits are interacting.
    double idlingFidelity = 0.0;
    /// @brief The number of qubits involved in the operation.
    uint64_t numQubits = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_WITH_DEFAULT(
        GlobalMultiQubitOperation, Operation, interactionRadius, blockingRadius,
        idlingFidelity, numQubits)
  };
  /// @brief The list of global multi-qubit operations supported by the device.
  std::vector<GlobalMultiQubitOperation> globalMultiQubitOperations;

  /// @brief Represents a local single-qubit operation.
  struct LocalSingleQubitOperation : Operation {};
  /// @brief The list of local single-qubit operations supported by the device.
  std::vector<LocalSingleQubitOperation> localSingleQubitOperations;

  /// @brief Represents a local multi-qubit operation.
  struct LocalMultiQubitOperation : Operation {
    /**
     * @brief The interaction radius of the operation within which two qubits
     * can interact.
     */
    double interactionRadius = 0.0;
    /**
     * @brief The blocking radius of the operation within which no other
     * operation can be performed to avoid interference.
     */
    double blockingRadius = 0.0;
    /// @brief The number of qubits involved in the operation.
    uint64_t numQubits = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_WITH_DEFAULT(
        LocalMultiQubitOperation, Operation, interactionRadius, blockingRadius,
        numQubits)
  };
  /// @brief The list of local multi-qubit operations supported by the device.
  std::vector<LocalMultiQubitOperation> localMultiQubitOperations;

  /// @brief Represents a shuttling unit in the device.
  struct ShuttlingUnit {
    size_t id = 0; ///< @brief Unique identifier for the shuttling unit.
    /// @brief The region in which the shuttling unit operates.
    Region region;
    /// @brief The speed at which the shuttling unit moves.
    double movingSpeed = 0.0;
    /// @brief The duration of the load operation in the shuttling unit.
    uint64_t loadDuration = 0;
    /// @brief The duration of the store operation in the shuttling unit.
    uint64_t storeDuration = 0;
    /// @brief The fidelity of the load operation in the shuttling unit.
    double loadFidelity = 0.0;
    /// @brief The fidelity of the store operation in the shuttling unit.
    double storeFidelity = 0.0;
    /// @brief The number of parameters the shuttling unit takes.
    uint64_t numParameters = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(ShuttlingUnit, region,
                                                movingSpeed, loadDuration,
                                                storeDuration, loadFidelity,
                                                storeFidelity, numParameters)
  };
  /// @brief The list of shuttling units supported by the device.
  std::vector<ShuttlingUnit> shuttlingUnits;

  /// @brief Represents the decoherence times of the device.
  struct DecoherenceTimes {
    /// @brief The T1 time.
    uint64_t t1 = 0;
    /// @brief The T2 time.
    uint64_t t2 = 0;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(DecoherenceTimes, t1, t2)
  };
  /// @brief The decoherence times of the device.
  DecoherenceTimes decoherenceTimes;

  /// @brief Represents a unit of measurement for length and time.
  struct Unit {
    /// @brief The factor of the unit.
    double scaleFactor = 0;
    /// @brief The unit of measurement (e.g., "µm" for micrometers, "ns" for
    /// nanoseconds).
    std::string unit;

    // NOLINTNEXTLINE(misc-include-cleaner)
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Unit, scaleFactor, unit)
  };

  /// @brief The unit of measurement for lengths in the device.
  Unit lengthUnit;
  /// @brief The unit of measurement for time in the device.
  Unit timeUnit;

  // Before we used the macro NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT here,
  // too. Now, we added an id to shuttling units that must be initialized
  // in a custom routine, so we can only use the serialize macro.
  // NOLINTNEXTLINE(misc-include-cleaner)
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_ONLY_SERIALIZE(
      Device, name, numQubits, traps, minAtomDistance,
      globalSingleQubitOperations, globalMultiQubitOperations,
      localSingleQubitOperations, localMultiQubitOperations, shuttlingUnits,
      decoherenceTimes, lengthUnit, timeUnit)

  // the name of the following function is given by the nlohmann::json library
  // and must not be changed
  template <typename BasicJsonType>
  // NOLINTNEXTLINE(readability-identifier-naming)
  friend void from_json(const BasicJsonType& json, Device& device) {
    const Device defaultDevice{};
    device.name = !json.is_null() ? json.value("name", defaultDevice.name)
                                  : defaultDevice.name;
    device.numQubits = !json.is_null()
                           ? json.value("numQubits", defaultDevice.numQubits)
                           : defaultDevice.numQubits;
    device.traps = !json.is_null() ? json.value("traps", defaultDevice.traps)
                                   : defaultDevice.traps;
    device.minAtomDistance =
        !json.is_null()
            ? json.value("minAtomDistance", defaultDevice.minAtomDistance)
            : defaultDevice.minAtomDistance;
    device.globalSingleQubitOperations =
        !json.is_null() ? json.value("globalSingleQubitOperations",
                                     defaultDevice.globalSingleQubitOperations)
                        : defaultDevice.globalSingleQubitOperations;
    device.globalMultiQubitOperations =
        !json.is_null() ? json.value("globalMultiQubitOperations",
                                     defaultDevice.globalMultiQubitOperations)
                        : defaultDevice.globalMultiQubitOperations;
    device.localSingleQubitOperations =
        !json.is_null() ? json.value("localSingleQubitOperations",
                                     defaultDevice.localSingleQubitOperations)
                        : defaultDevice.localSingleQubitOperations;
    device.localMultiQubitOperations =
        !json.is_null() ? json.value("localMultiQubitOperations",
                                     defaultDevice.localMultiQubitOperations)
                        : defaultDevice.localMultiQubitOperations;
    device.shuttlingUnits =
        !json.is_null()
            ? json.value("shuttlingUnits", defaultDevice.shuttlingUnits)
            : defaultDevice.shuttlingUnits;
    std::ranges::for_each(device.shuttlingUnits,
                          [i = 0UL](auto& unit) mutable { unit.id = i++; });
    device.decoherenceTimes =
        !json.is_null()
            ? json.value("decoherenceTimes", defaultDevice.decoherenceTimes)
            : defaultDevice.decoherenceTimes;
    device.lengthUnit = !json.is_null()
                            ? json.value("lengthUnit", defaultDevice.lengthUnit)
                            : defaultDevice.lengthUnit;
    device.timeUnit = !json.is_null()
                          ? json.value("timeUnit", defaultDevice.timeUnit)
                          : defaultDevice.timeUnit;
  }
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
