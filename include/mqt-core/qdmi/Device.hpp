/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Device.hpp
 * @brief ABI-neutral C++ interface for QDMI devices.
 */

#pragma once

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

/** Device availability state independent of the provider ABI. */
enum class DeviceStatus : std::uint8_t {
  Offline,
  Idle,
  Busy,
  Error,
  Maintenance,
  Calibration,
};

/** Lifecycle state of a submitted job. */
enum class JobStatus : std::uint8_t {
  Created,
  Submitted,
  Queued,
  Running,
  Done,
  Canceled,
  Failed,
};

/** Program formats understood by the QDMI v1 adapter. */
enum class ProgramFormat : std::uint32_t {
  Qasm2 = 0,
  Qasm3 = 1,
  QirBaseString = 2,
  QirBaseModule = 3,
  QirAdaptiveString = 4,
  QirAdaptiveModule = 5,
  Calibration = 6,
  Qpy = 7,
  IqmJson = 8,
  BatchJob = 9,
  Custom1 = 999999995,
  Custom2 = 999999996,
  Custom3 = 999999997,
  Custom4 = 999999998,
  Custom5 = 999999999,
};

/** Identifies an implementation-defined custom property or result slot. */
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
[[nodiscard]] auto
decodeCustomValue(const std::optional<std::vector<std::byte>>& bytes,
                  const std::string& description) -> std::optional<T> {
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

/** One initialized quantum-device session. */
class Device {
public:
  [[nodiscard]] auto getName() const -> std::string;
  [[nodiscard]] auto getVersion() const -> std::string;
  [[nodiscard]] auto getStatus() const -> DeviceStatus;
  [[nodiscard]] auto getLibraryVersion() const -> std::string;
  [[nodiscard]] auto getQubitsNum() const -> size_t;
  [[nodiscard]] auto getSites() const -> std::vector<Site>;
  [[nodiscard]] auto getRegularSites() const -> std::vector<Site>;
  [[nodiscard]] auto getZones() const -> std::vector<Site>;
  [[nodiscard]] auto getOperations() const -> std::vector<Operation>;
  [[nodiscard]] auto getCouplingMap() const
      -> std::optional<std::vector<std::pair<Site, Site>>>;
  [[nodiscard]] auto getNeedsCalibration() const -> std::optional<size_t>;
  [[nodiscard]] auto getLengthUnit() const -> std::optional<std::string>;
  [[nodiscard]] auto getLengthScaleFactor() const -> std::optional<double>;
  [[nodiscard]] auto getDurationUnit() const -> std::optional<std::string>;
  [[nodiscard]] auto getDurationScaleFactor() const -> std::optional<double>;
  [[nodiscard]] auto getMinAtomDistance() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getSupportedProgramFormats() const
      -> std::vector<ProgramFormat>;
  [[nodiscard]] auto getChildDevices() const -> std::vector<Device>;

  template <custom_property_value T>
  [[nodiscard]] auto queryCustomProperty(const CustomProperty property) const
      -> std::optional<T> {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom device property");
  }

  [[nodiscard]] auto submitJob(
      const std::string& program, ProgramFormat format, size_t numShots,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const
      -> Job;

  [[nodiscard]] auto operator<=>(const Device& other) const noexcept {
    return state_.get() <=> other.state_.get();
  }
  [[nodiscard]] auto operator==(const Device& other) const noexcept -> bool {
    return state_ == other.state_;
  }

private:
  explicit Device(std::shared_ptr<detail::DeviceState> state)
      : state_(std::move(state)) {}
  [[nodiscard]] auto queryCustomPropertyBytes(CustomProperty property) const
      -> std::optional<std::vector<std::byte>>;
  std::shared_ptr<detail::DeviceState> state_;
  friend class DeviceManager;
  friend struct detail::DeviceFactory;
};

/** A submitted job retaining its device session. */
class Job {
public:
  [[nodiscard]] auto check() const -> JobStatus;
  [[nodiscard]] auto wait(size_t timeout = 0) const -> bool;
  void cancel() const;
  [[nodiscard]] auto getId() const -> std::string;
  [[nodiscard]] auto getProgramFormat() const -> ProgramFormat;
  [[nodiscard]] auto getProgram() const -> std::string;
  [[nodiscard]] auto getNumShots() const -> size_t;

  template <custom_property_value T>
  [[nodiscard]] auto queryCustomProperty(const CustomProperty property) const
      -> std::optional<T> {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom job property");
  }
  template <custom_property_value T>
  [[nodiscard]] auto getCustomResult(const CustomProperty property) const
      -> std::optional<T> {
    return detail::decodeCustomValue<T>(getCustomResultBytes(property),
                                        "custom job result");
  }

  [[nodiscard]] auto getShots() const -> std::vector<std::string>;
  [[nodiscard]] auto getCounts() const -> std::map<std::string, size_t>;
  [[nodiscard]] auto getDenseStateVector() const
      -> std::vector<std::complex<double>>;
  [[nodiscard]] auto getDenseProbabilities() const -> std::vector<double>;
  [[nodiscard]] auto getSparseStateVector() const
      -> std::map<std::string, std::complex<double>>;
  [[nodiscard]] auto getSparseProbabilities() const
      -> std::map<std::string, double>;

  [[nodiscard]] auto operator<=>(const Job& other) const noexcept {
    return state_.get() <=> other.state_.get();
  }
  [[nodiscard]] auto operator==(const Job& other) const noexcept -> bool {
    return state_ == other.state_;
  }

private:
  explicit Job(std::shared_ptr<detail::JobState> state)
      : state_(std::move(state)) {}
  [[nodiscard]] auto queryCustomPropertyBytes(CustomProperty property) const
      -> std::optional<std::vector<std::byte>>;
  [[nodiscard]] auto getCustomResultBytes(CustomProperty property) const
      -> std::optional<std::vector<std::byte>>;
  std::shared_ptr<detail::JobState> state_;
  friend class Device;
};

/** A physical site or zone belonging to a device. */
class Site {
public:
  [[nodiscard]] auto getIndex() const -> size_t;
  [[nodiscard]] auto getT1() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getT2() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getName() const -> std::optional<std::string>;
  [[nodiscard]] auto getXCoordinate() const -> std::optional<int64_t>;
  [[nodiscard]] auto getYCoordinate() const -> std::optional<int64_t>;
  [[nodiscard]] auto getZCoordinate() const -> std::optional<int64_t>;
  [[nodiscard]] auto isZone() const -> bool;
  [[nodiscard]] auto getXExtent() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getYExtent() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getZExtent() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getModuleIndex() const -> std::optional<uint64_t>;
  [[nodiscard]] auto getSubmoduleIndex() const -> std::optional<uint64_t>;

  template <custom_property_value T>
  [[nodiscard]] auto queryCustomProperty(const CustomProperty property) const
      -> std::optional<T> {
    return detail::decodeCustomValue<T>(queryCustomPropertyBytes(property),
                                        "custom site property");
  }

  [[nodiscard]] auto operator<=>(const Site& other) const noexcept {
    if (const auto device = state_.get() <=> other.state_.get(); device != 0) {
      return device;
    }
    return handle_ <=> other.handle_;
  }
  [[nodiscard]] auto operator==(const Site& other) const noexcept -> bool {
    return state_ == other.state_ && handle_ == other.handle_;
  }

private:
  Site(std::shared_ptr<detail::DeviceState> state, void* handle)
      : state_(std::move(state)), handle_(handle) {}
  [[nodiscard]] auto queryCustomPropertyBytes(CustomProperty property) const
      -> std::optional<std::vector<std::byte>>;
  std::shared_ptr<detail::DeviceState> state_;
  void* handle_ = nullptr;
  friend class Device;
  friend class Operation;
};

/** An operation supported by a device. */
class Operation {
public:
  [[nodiscard]] auto getName(const std::vector<Site>& sites = {},
                             const std::vector<double>& params = {}) const
      -> std::string;
  [[nodiscard]] auto getQubitsNum(const std::vector<Site>& sites = {},
                                  const std::vector<double>& params = {}) const
      -> std::optional<size_t>;
  [[nodiscard]] auto
  getParametersNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const -> size_t;
  [[nodiscard]] auto getDuration(const std::vector<Site>& sites = {},
                                 const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  [[nodiscard]] auto getFidelity(const std::vector<Site>& sites = {},
                                 const std::vector<double>& params = {}) const
      -> std::optional<double>;
  [[nodiscard]] auto
  getInteractionRadius(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  [[nodiscard]] auto
  getBlockingRadius(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  [[nodiscard]] auto
  getIdlingFidelity(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const
      -> std::optional<double>;
  [[nodiscard]] auto isZoned() const -> bool;
  [[nodiscard]] auto getSites() const -> std::optional<std::vector<Site>>;
  [[nodiscard]] auto getSitePairs() const
      -> std::optional<std::vector<std::pair<Site, Site>>>;
  [[nodiscard]] auto
  getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                        const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;

  template <custom_property_value T>
  [[nodiscard]] auto
  queryCustomProperty(const CustomProperty property,
                      const std::vector<Site>& sites = {},
                      const std::vector<double>& params = {}) const
      -> std::optional<T> {
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
  [[nodiscard]] auto operator==(const Operation& other) const noexcept -> bool {
    return state_ == other.state_ && handle_ == other.handle_;
  }

private:
  [[nodiscard]] static auto siteHandles(const std::vector<Site>& sites)
      -> std::vector<void*>;
  Operation(std::shared_ptr<detail::DeviceState> state, void* handle)
      : state_(std::move(state)), handle_(handle) {}
  [[nodiscard]] auto
  queryCustomPropertyBytes(CustomProperty property,
                           const std::vector<Site>& sites,
                           const std::vector<double>& params) const
      -> std::optional<std::vector<std::byte>>;
  std::shared_ptr<detail::DeviceState> state_;
  void* handle_ = nullptr;
  friend class Device;
};
} // namespace qdmi
