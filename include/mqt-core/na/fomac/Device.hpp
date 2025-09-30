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
#include <optional>
#include <utility>
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
     * @brief Initializes the name from the underlying QDMI device.
     */
    auto initNameFromDevice() -> void;

    /**
     * @brief Initializes the minimum atom distance from the underlying QDMI
     * device.
     */
    auto initMinAtomDistanceFromDevice() -> bool;

    /**
     * @brief Initializes the number of qubits from the underlying QDMI device.
     */
    auto initQubitsNumFromDevice() -> void;

    /**
     * @brief Initializes the length unit from the underlying QDMI device.
     */
    auto initLengthUnitFromDevice() -> bool;

    /**
     * @brief Initializes the duration unit from the underlying QDMI device.
     */
    auto initDurationUnitFromDevice() -> bool;

    /**
     * @brief Initializes the decoherence times from the underlying QDMI device.
     */
    auto initDecoherenceTimesFromDevice() -> bool;

    /**
     * @brief Initializes the trap lattices from the underlying QDMI device.
     * @details It reconstructs the entire lattice structure from the
     * information retrieved from the QDMI device, including lattice vectors,
     * sublattice offsets, and extent.
     */
    auto initTrapsfromDevice() -> bool;

    /**
     * @brief Initializes the all operations from the underlying QDMI device.
     */
    auto initOperationsFromDevice() -> bool;

    /**
     * @brief Constructs a Device object from a fomac::FoMaC::Device object.
     * @param device The fomac::FoMaC::Device object to wrap.
     */
    explicit Device(const fomac::FoMaC::Device& device);

  public:
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

    template <typename T> class GenericCompatibilityResult : std::optional<T> {
    public:
      GenericCompatibilityResult() = default;
      explicit GenericCompatibilityResult(T&& t)
          : std::optional<T>(std::move(t)) {}
      using std::optional<T>::operator bool;
      [[nodiscard]] auto createDevice() const -> T { return std::move(*this); }
    };
    using CompatibilityResult = GenericCompatibilityResult<Device>;

  private:
    static const CompatibilityResult INCOMPATIBLE;

  public:
    /**
     * @brief Checks if this class can be instantiated with the given device.
     * @param device The device to check.
     * @return An optional containing the instantiated Device if compatible,
     * std::nullopt otherwise.
     */
    [[nodiscard]] static auto
    checkCompatiblility(const fomac::FoMaC::Device& device)
        -> CompatibilityResult {
      Device d(device);
      if (!d.initMinAtomDistanceFromDevice()) {
        return INCOMPATIBLE;
      }
      if (!d.initLengthUnitFromDevice()) {
        return INCOMPATIBLE;
      }
      if (!d.initDurationUnitFromDevice()) {
        return INCOMPATIBLE;
      }
      if (!d.initDecoherenceTimesFromDevice()) {
        return INCOMPATIBLE;
      }
      if (!d.initTrapsfromDevice()) {
        return INCOMPATIBLE;
      }
      if (!d.initOperationsFromDevice()) {
        return INCOMPATIBLE;
      }
      d.initNameFromDevice();
      d.initQubitsNumFromDevice();
      return CompatibilityResult{std::move(d)};
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
