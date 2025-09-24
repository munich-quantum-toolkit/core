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

#include "na/device/Generator.hpp"
#include "qdmi/FoMaC.hpp"

#include <nlohmann/json.hpp>
#include <vector>

namespace na {

/**
 * @brief Class representing the FoMaC library with neutral atom extensions.
 * @see qdmi::FoMaC
 */
class FoMaC : public qdmi::FoMaC {
public:
  /**
   * @brief Class representing a quantum device with neutral atom extensions.
   * @see qdmi::FoMaC::Device
   * @note Since it inherits from @ref na::Device, Device objects can be
   * converted to `nlohmann::json` objects.
   */
  class Device : public qdmi::FoMaC::Device, na::Device {
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
    auto initLatticesfromDevice() -> void;

  public:
    /**
     * @brief Constructs a Device object from a qdmi::FoMaC::Device object.
     * @param device The qdmi::FoMaC::Device object to wrap.
     */
    explicit Device(const qdmi::FoMaC::Device& device);

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
    friend void to_json(BasicJsonType& nlohmann_json_j,
                        const Device& nlohmann_json_t) {
      nlohmann::to_json(nlohmann_json_j,
                        static_cast<const na::Device&>(nlohmann_json_t));
    }
  };

  /// @brief Deleted default constructor to prevent instantiation.
  FoMaC() = delete;

  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] static auto getDevices() -> std::vector<Device>;
};

} // namespace na
