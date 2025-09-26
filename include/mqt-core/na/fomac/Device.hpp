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

#include "fomac/FoMaC.hpp"
#include "na/device/Generator.hpp"

// NOLINTNEXTLINE(misc-include-cleaner)
#include <nlohmann/json.hpp>
#include <vector>

namespace na {

/**
 * @brief Class representing the FoMaC library with neutral atom extensions.
 * @see fomac::FoMaC
 */
class FoMaC : public fomac::FoMaC {
public:
  /**
   * @brief Class representing a quantum device with neutral atom extensions.
   * @see fomac::FoMaC::Device
   * @note Since it inherits from @ref na::Device, Device objects can be
   * converted to `nlohmann::json` objects.
   */
  class Device : public fomac::FoMaC::Device, na::Device {
    /**
     * @brief Calculate the extent of a set of sites.
     * @param sites The sites to calculate the extent for.
     * @returns A `Region` object representing the extent of the sites.
     */
    static auto calculateExtentFromSites(
        const std::vector<fomac::FoMaC::Device::Site>& sites) -> Region;

    /**
     * @brief Initializes the name from the underlying QDMI device.
     */
    auto initNameFromDevice() -> void;

    /**
     * @brief Initializes the minimum atom distance from the underlying QDMI
     * device.
     */
    auto initMinAtomDistanceFromDevice() -> void;

    /**
     * @brief Initializes the number of qubits from the underlying QDMI device.
     */
    auto initQubitsNumFromDevice() -> void;

    /**
     * @brief Initializes the length unit from the underlying QDMI device.
     */
    auto initLengthUnitFromDevice() -> void;

    /**
     * @brief Initializes the duration unit from the underlying QDMI device.
     */
    auto initDurationUnitFromDevice() -> void;

    /**
     * @brief Initializes the decoherence times from the underlying QDMI device.
     */
    auto initDecoherenceTimesFromDevice() -> void;

    /**
     * @brief Initializes the trap lattices from the underlying QDMI device.
     * @details It reconstructs the entire lattice structure from the
     * information retrieved from the QDMI device, including lattice vectors,
     * sublattice offsets, and extent.
     */
    auto initTrapsfromDevice() -> void;

    /**
     * @brief Initializes the global single-qubit operations from the
     * underlying QDMI device.
     */
    auto initGlobalSingleQubitOperationsFromDevice() -> void;

    /**
     * @brief Initializes the global multi-qubit operations from the
     * underlying QDMI device.
     */
    auto initGlobalMultiQubitOperationsFromDevice() -> void;

    /**
     * @brief Initializes the local single-qubit operations from the
     * underlying QDMI device.
     */
    auto initLocalSingleQubitOperationsFromDevice() -> void;

    /**
     * @brief Initializes the local multi-qubit operations from the
     * underlying QDMI device.
     */
    auto initLocalMultiQubitOperationsFromDevice() -> void;

    /**
     * @brief Initializes the shuttling units from the underlying QDMI device.
     */
    auto initShuttlingUnitsFromDevice() -> void;

  public:
    /**
     * @brief Constructs a Device object from a fomac::FoMaC::Device object.
     * @param device The fomac::FoMaC::Device object to wrap.
     */
    explicit Device(const fomac::FoMaC::Device& device);

    /// @returns the length unit of the device.
    [[nodiscard]] auto getLengthUnit() const -> const Unit& {
      return lengthUnit;
    }

    /// @returns the duration unit of the device.
    [[nodiscard]] auto getDurationUnit() const -> const Unit& {
      return durationUnit;
    }

    /// @returns the decoherence times of the device.
    [[nodiscard]] auto getDecoherenceTimes() const -> const DecoherenceTimes& {
      return decoherenceTimes;
    }

    /// @returns the list of trap lattices of the device.
    [[nodiscard]] auto getTraps() const -> const std::vector<Lattice>& {
      return traps;
    }

    // The following is the result of
    // NLOHMANN_DEFINE_DERIVED_TYPE_INTRUSIVE_ONLY_SERIALIZE(Device, na::Device)
    // without any new attributes, which is the reason the macro cannot be used.
    template <typename BasicJsonType>
    friend void to_json(BasicJsonType& nlohmannJsonJ,
                        const Device& nlohmannJsonT) {
      // NOLINTNEXTLINE(misc-include-cleaner)
      nlohmann::to_json(nlohmannJsonJ,
                        static_cast<const na::Device&>(nlohmannJsonT));
    }
  };

  /// @brief Deleted default constructor to prevent instantiation.
  FoMaC() = delete;

  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] static auto getDevices() -> std::vector<Device>;
};

} // namespace na
