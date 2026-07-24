/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Device.hpp
/// @brief C++ object model for QDMI devices.

#pragma once

#include <qdmi/device.h>

#include <compare>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi {
namespace detail {
struct DeviceState;
struct JobState;
struct DeviceFactory;
} // namespace detail

using CustomJobParameter = std::variant<std::string, bool, int, double>;

/// Identifies an implementation-defined custom property or result slot.
enum class CustomProperty : std::uint8_t {
  Custom1 = 1,
  Custom2 = 2,
  Custom3 = 3,
  Custom4 = 4,
  Custom5 = 5,
};

template <typename T>
concept custom_property_value =
    std::same_as<T, std::string> || std::same_as<T, bool> ||
    std::same_as<T, int> || std::same_as<T, double> ||
    std::same_as<T, std::vector<std::byte>>;

namespace detail {
template <custom_property_value T>
[[nodiscard]] std::optional<T>
decodeCustomValue(const std::optional<std::vector<std::byte>>& bytes,
                  const std::string& description) {
  if (!bytes) {
    return std::nullopt;
  }
  if constexpr (std::same_as<T, std::vector<std::byte>>) {
    return *bytes;
  } else if constexpr (std::same_as<T, std::string>) {
    if (bytes->empty() || bytes->back() != std::byte{0}) {
      throw std::invalid_argument("Cannot decode " + description +
                                  " as a null-terminated string");
    }
    return std::string(reinterpret_cast<const char*>(bytes->data()),
                       bytes->size() - 1);
  } else {
    if (bytes->size() != sizeof(T)) {
      throw std::invalid_argument("Cannot decode " + description +
                                  ": unexpected byte size");
    }
    T value{};
    std::memcpy(&value, bytes->data(), sizeof(T));
    return value;
  }
}
} // namespace detail

class Job;
class Site;
class Operation;
class DeviceManager;

/// One initialized quantum-device session.
class Device {
public:
  [[nodiscard]] std::string getName() const;
  [[nodiscard]] std::string getVersion() const;
  [[nodiscard]] QDMI_Device_Status getStatus() const;
  [[nodiscard]] std::string getLibraryVersion() const;
  [[nodiscard]] size_t getQubitsNum() const;
  [[nodiscard]] std::vector<Site> getSites() const;
  [[nodiscard]] std::vector<Site> getRegularSites() const;
  [[nodiscard]] std::vector<Site> getZones() const;
  [[nodiscard]] std::vector<Operation> getOperations() const;
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getCouplingMap() const;
  [[nodiscard]] std::optional<size_t> getNeedsCalibration() const;
  [[nodiscard]] std::optional<std::string> getLengthUnit() const;
  [[nodiscard]] std::optional<double> getLengthScaleFactor() const;
  [[nodiscard]] std::optional<std::string> getDurationUnit() const;
  [[nodiscard]] std::optional<double> getDurationScaleFactor() const;
  [[nodiscard]] std::optional<uint64_t> getMinAtomDistance() const;
  [[nodiscard]] std::vector<QDMI_Program_Format>
  getSupportedProgramFormats() const;
  [[nodiscard]] std::vector<Device> getChildDevices() const;

  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom device property");
  }

  [[nodiscard]] Job submitJob(
      const std::string& program, QDMI_Program_Format format, size_t numShots,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  [[nodiscard]] auto operator<=>(const Device& other) const noexcept {
    return state_.get() <=> other.state_.get();
  }
  [[nodiscard]] bool operator==(const Device& other) const noexcept {
    return state_ == other.state_;
  }

private:
  explicit Device(std::shared_ptr<const detail::DeviceState> state)
      : state_(std::move(state)) {}
  [[nodiscard]] std::optional<std::vector<std::byte>>
  queryCustomPropertyBytes(CustomProperty property) const;
  std::shared_ptr<const detail::DeviceState> state_;
  friend class DeviceManager;
  friend struct detail::DeviceFactory;
};

/// A submitted job retaining its device session.
class Job {
public:
  [[nodiscard]] QDMI_Job_Status check() const;
  [[nodiscard]] bool wait(size_t timeout = 0) const;
  void cancel() const;
  [[nodiscard]] std::string getId() const;
  [[nodiscard]] QDMI_Program_Format getProgramFormat() const;
  [[nodiscard]] std::string getProgram() const;
  [[nodiscard]] size_t getNumShots() const;

  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom job property");
  }
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  getCustomResult(const CustomProperty property) const {
    return detail::decodeCustomValue<T>(getCustomResultBytes(property),
                                        "custom job result");
  }

  [[nodiscard]] std::vector<std::string> getShots() const;
  [[nodiscard]] std::map<std::string, size_t> getCounts() const;
  [[nodiscard]] std::vector<std::complex<double>> getDenseStateVector() const;
  [[nodiscard]] std::vector<double> getDenseProbabilities() const;
  [[nodiscard]] std::map<std::string, std::complex<double>>
  getSparseStateVector() const;
  [[nodiscard]] std::map<std::string, double> getSparseProbabilities() const;

  [[nodiscard]] auto operator<=>(const Job& other) const noexcept {
    return state_.get() <=> other.state_.get();
  }
  [[nodiscard]] bool operator==(const Job& other) const noexcept {
    return state_ == other.state_;
  }

private:
  explicit Job(std::shared_ptr<detail::JobState> state)
      : state_(std::move(state)) {}
  [[nodiscard]] std::optional<std::vector<std::byte>>
  queryCustomPropertyBytes(CustomProperty property) const;
  [[nodiscard]] std::optional<std::vector<std::byte>>
  getCustomResultBytes(CustomProperty property) const;
  std::shared_ptr<detail::JobState> state_;
  friend class Device;
};

/// A physical site or zone belonging to a device.
class Site {
public:
  [[nodiscard]] size_t getIndex() const;
  [[nodiscard]] std::optional<uint64_t> getT1() const;
  [[nodiscard]] std::optional<uint64_t> getT2() const;
  [[nodiscard]] std::optional<std::string> getName() const;
  [[nodiscard]] std::optional<int64_t> getXCoordinate() const;
  [[nodiscard]] std::optional<int64_t> getYCoordinate() const;
  [[nodiscard]] std::optional<int64_t> getZCoordinate() const;
  [[nodiscard]] bool isZone() const;
  [[nodiscard]] std::optional<uint64_t> getXExtent() const;
  [[nodiscard]] std::optional<uint64_t> getYExtent() const;
  [[nodiscard]] std::optional<uint64_t> getZExtent() const;
  [[nodiscard]] std::optional<uint64_t> getModuleIndex() const;
  [[nodiscard]] std::optional<uint64_t> getSubmoduleIndex() const;

  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom site property");
  }

  [[nodiscard]] auto operator<=>(const Site& other) const noexcept {
    if (const auto device = state_.get() <=> other.state_.get(); device != 0) {
      return device;
    }
    return handle_ <=> other.handle_;
  }
  [[nodiscard]] bool operator==(const Site& other) const noexcept {
    return state_ == other.state_ && handle_ == other.handle_;
  }

private:
  Site(std::shared_ptr<const detail::DeviceState> state, void* handle)
      : state_(std::move(state)), handle_(handle) {}
  [[nodiscard]] std::optional<std::vector<std::byte>>
  queryCustomPropertyBytes(CustomProperty property) const;
  std::shared_ptr<const detail::DeviceState> state_;
  void* handle_ = nullptr;
  friend class Device;
  friend class Operation;
};

/// An operation supported by a device.
class Operation {
public:
  [[nodiscard]] std::string
  getName(const std::vector<Site>& sites = {},
          const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<size_t>
  getQubitsNum(const std::vector<Site>& sites = {},
               const std::vector<double>& params = {}) const;
  [[nodiscard]] size_t
  getParametersNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<uint64_t>
  getDuration(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<double>
  getFidelity(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<uint64_t>
  getInteractionRadius(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<uint64_t>
  getBlockingRadius(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;
  [[nodiscard]] std::optional<double>
  getIdlingFidelity(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;
  [[nodiscard]] bool isZoned() const;
  [[nodiscard]] std::optional<std::vector<Site>> getSites() const;
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getSitePairs() const;
  [[nodiscard]] std::optional<uint64_t>
  getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                        const std::vector<double>& params = {}) const;

  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property,
                      const std::vector<Site>& sites = {},
                      const std::vector<double>& params = {}) const {
    return detail::decodeCustomValue<T>(
        queryCustomPropertyBytes(property, sites, params),
        "custom operation property");
  }

  [[nodiscard]] auto operator<=>(const Operation& other) const noexcept {
    if (const auto device = state_.get() <=> other.state_.get(); device != 0) {
      return device;
    }
    return handle_ <=> other.handle_;
  }
  [[nodiscard]] bool operator==(const Operation& other) const noexcept {
    return state_ == other.state_ && handle_ == other.handle_;
  }

private:
  [[nodiscard]] std::vector<void*>
  siteHandles(const std::vector<Site>& sites) const;
  Operation(std::shared_ptr<const detail::DeviceState> state, void* handle)
      : state_(std::move(state)), handle_(handle) {}
  [[nodiscard]] std::optional<std::vector<std::byte>>
  queryCustomPropertyBytes(CustomProperty property,
                           const std::vector<Site>& sites,
                           const std::vector<double>& params) const;
  std::shared_ptr<const detail::DeviceState> state_;
  void* handle_ = nullptr;
  friend class Device;
};
} // namespace qdmi
