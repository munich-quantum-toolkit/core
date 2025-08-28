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

#include "ir/operations/OpType.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <qdmi/client.h>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace fomac {
/**
 * @brief Concept for ranges that are contiguous in memory and can be
 * constructed with a size.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept size_constructible_contiguous_range =
    std::ranges::contiguous_range<T> &&
    std::constructible_from<T, std::size_t> &&
    requires { typename T::value_type; } && requires(T t) {
      { t.data() } -> std::same_as<typename T::value_type*>;
    };
/**
 * @brief Concept for types that are either integral, floating point, bool,
 * std::string, or QDMI_Device_Status.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept value_or_string =
    std::integral<T> || std::floating_point<T> || std::is_same_v<T, bool> ||
    std::is_same_v<T, std::string> || std::is_same_v<T, QDMI_Device_Status>;

/**
 * @brief Concept for types that are either value_or_string or
 * size_constructible_contiguous_range.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept value_or_string_or_vector =
    value_or_string<T> || size_constructible_contiguous_range<T>;

/**
 * @brief Concept for types that are std::optional of value_or_string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept is_optional = requires { typename T::value_type; } &&
                      std::is_same_v<T, std::optional<typename T::value_type>>;

/**
 * @brief Concept for types that are either size_constructible_contiguous_range
 * or std::optional of size_constructible_contiguous_range.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Operation::queryProperty
 */
template <typename T>
concept maybe_optional_size_constructible_contiguous_range =
    size_constructible_contiguous_range<T> ||
    (is_optional<T> &&
     size_constructible_contiguous_range<typename T::value_type>);

/**
 * @brief Concept for types that are either value_or_string or std::optional of
 * value_or_string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Site::queryProperty
 */
template <typename T>
concept maybe_optional_value_or_string =
    value_or_string<T> ||
    (is_optional<T> && value_or_string<typename T::value_type>);

/**
 * @brief Concept for types that are either value_or_string_or_vector or
 * std::optional of value_or_string_or_vector.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Operation::queryProperty
 */
template <typename T>
concept maybe_optional_value_or_string_or_vector =
    value_or_string_or_vector<T> ||
    (is_optional<T> && value_or_string_or_vector<typename T::value_type>);

/**
 * @brief Concept for types that are either std::string or std::optional of
 * std::string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept string_or_optional_string =
    std::is_same_v<T, std::string> ||
    (is_optional<T> && std::is_same_v<typename T::value_type, std::string>);

/// @returns the string representation of the given QDMI_STATUS.
constexpr auto toString(QDMI_STATUS result) -> std::string {
  switch (result) {
  case QDMI_WARN_GENERAL:
    return "General warning.";
  case QDMI_SUCCESS:
    return "Success.";
  case QDMI_ERROR_FATAL:
    return "A fatal error.";
  case QDMI_ERROR_OUTOFMEM:
    return "Out of memory.";
  case QDMI_ERROR_NOTIMPLEMENTED:
    return "Not implemented.";
  case QDMI_ERROR_LIBNOTFOUND:
    return "Library not found.";
  case QDMI_ERROR_NOTFOUND:
    return "Element not found.";
  case QDMI_ERROR_OUTOFRANGE:
    return "Out of range.";
  case QDMI_ERROR_INVALIDARGUMENT:
    return "Invalid argument.";
  case QDMI_ERROR_PERMISSIONDENIED:
    return "Permission denied.";
  case QDMI_ERROR_NOTSUPPORTED:
    return "Not supported.";
  case QDMI_ERROR_BADSTATE:
    return " Bad state.";
  case QDMI_ERROR_TIMEOUT:
    return "Timeout.";
  default:
    return "Unknown status code.";
  }
}

/// @returns the string representation of the given QDMI_Site_Property.
constexpr auto toString(QDMI_Site_Property prop) -> std::string {
  switch (prop) {
  case QDMI_SITE_PROPERTY_INDEX:
    return "QDMI_SITE_PROPERTY_INDEX";
  case QDMI_SITE_PROPERTY_T1:
    return "QDMI_SITE_PROPERTY_T1";
  case QDMI_SITE_PROPERTY_T2:
    return "QDMI_SITE_PROPERTY_T2";
  case QDMI_SITE_PROPERTY_NAME:
    return "QDMI_SITE_PROPERTY_NAME";
  case QDMI_SITE_PROPERTY_XCOORDINATE:
    return "QDMI_SITE_PROPERTY_XCOORDINATE";
  case QDMI_SITE_PROPERTY_YCOORDINATE:
    return "QDMI_SITE_PROPERTY_YCOORDINATE";
  case QDMI_SITE_PROPERTY_ZCOORDINATE:
    return "QDMI_SITE_PROPERTY_ZCOORDINATE";
  case QDMI_SITE_PROPERTY_ISZONE:
    return "QDMI_SITE_PROPERTY_ISZONE";
  case QDMI_SITE_PROPERTY_XEXTENT:
    return "QDMI_SITE_PROPERTY_XEXTENT";
  case QDMI_SITE_PROPERTY_YEXTENT:
    return "QDMI_SITE_PROPERTY_YEXTENT";
  case QDMI_SITE_PROPERTY_ZEXTENT:
    return "QDMI_SITE_PROPERTY_ZEXTENT";
  case QDMI_SITE_PROPERTY_MODULEINDEX:
    return "QDMI_SITE_PROPERTY_MODULEINDEX";
  case QDMI_SITE_PROPERTY_SUBMODULEINDEX:
    return "QDMI_SITE_PROPERTY_SUBMODULEINDEX";
  default:
    return "QDMI_SITE_PROPERTY_UNKNOWN";
  }
}

/// @returns the string representation of the given QDMI_Operation_Property.
constexpr auto toString(QDMI_Operation_Property prop) -> std::string {
  switch (prop) {
  case QDMI_OPERATION_PROPERTY_NAME:
    return "QDMI_OPERATION_PROPERTY_NAME";
  case QDMI_OPERATION_PROPERTY_QUBITSNUM:
    return "QDMI_OPERATION_PROPERTY_QUBITSNUM";
  case QDMI_OPERATION_PROPERTY_PARAMETERSNUM:
    return "QDMI_OPERATION_PROPERTY_PARAMETERSNUM";
  case QDMI_OPERATION_PROPERTY_DURATION:
    return "QDMI_OPERATION_PROPERTY_DURATION";
  case QDMI_OPERATION_PROPERTY_FIDELITY:
    return "QDMI_OPERATION_PROPERTY_FIDELITY";
  case QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS:
    return "QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS";
  case QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS:
    return "QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS";
  case QDMI_OPERATION_PROPERTY_IDLINGFIDELITY:
    return "QDMI_OPERATION_PROPERTY_IDLINGFIDELITY";
  case QDMI_OPERATION_PROPERTY_ISZONED:
    return "QDMI_OPERATION_PROPERTY_ISZONED";
  case QDMI_OPERATION_PROPERTY_SITES:
    return "QDMI_OPERATION_PROPERTY_SITES";
  case QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED:
    return "QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED";
  default:
    return "QDMI_OPERATION_PROPERTY_UNKNOWN";
  }
}

/// @returns the string representation of the given QDMI_Device_Property.
constexpr auto toString(QDMI_Device_Property prop) -> std::string {
  switch (prop) {
  case QDMI_DEVICE_PROPERTY_NAME:
    return "QDMI_DEVICE_PROPERTY_NAME";
  case QDMI_DEVICE_PROPERTY_VERSION:
    return "QDMI_DEVICE_PROPERTY_VERSION";
  case QDMI_DEVICE_PROPERTY_STATUS:
    return "QDMI_DEVICE_PROPERTY_STATUS";
  case QDMI_DEVICE_PROPERTY_LIBRARYVERSION:
    return "QDMI_DEVICE_PROPERTY_LIBRARYVERSION";
  case QDMI_DEVICE_PROPERTY_QUBITSNUM:
    return "QDMI_DEVICE_PROPERTY_QUBITSNUM";
  case QDMI_DEVICE_PROPERTY_SITES:
    return "QDMI_DEVICE_PROPERTY_SITES";
  case QDMI_DEVICE_PROPERTY_OPERATIONS:
    return "QDMI_DEVICE_PROPERTY_OPERATIONS";
  case QDMI_DEVICE_PROPERTY_COUPLINGMAP:
    return "QDMI_DEVICE_PROPERTY_COUPLINGMAP";
  case QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION:
    return "QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION";
  case QDMI_DEVICE_PROPERTY_LENGTHUNIT:
    return "QDMI_DEVICE_PROPERTY_LENGTHUNIT";
  case QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR:
    return "QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR";
  case QDMI_DEVICE_PROPERTY_DURATIONUNIT:
    return "QDMI_DEVICE_PROPERTY_DURATIONUNIT";
  case QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR:
    return "QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR";
  case QDMI_DEVICE_PROPERTY_MINATOMDISTANCE:
    return "QDMI_DEVICE_PROPERTY_MINATOMDISTANCE";
  default:
    return "QDMI_DEVICE_PROPERTY_UNKNOWN";
  }
}

/// @returns the string representation of the given QDMI_Session_Property.
constexpr auto toString(QDMI_Session_Property prop) -> std::string {
  if (prop == QDMI_SESSION_PROPERTY_DEVICES) {
    return "QDMI_SESSION_PROPERTY_DEVICES";
  }
  return "QDMI_SESSION_PROPERTY_UNKNOWN";
}

/// Throws an exception corresponding to the given QDMI_STATUS code.
auto throwError(int result, const std::string& msg) -> void;

/// Throws an exception if the result indicates an error.
inline auto throwIfError(int result, const std::string& msg) -> void {
  switch (result) {
  case QDMI_SUCCESS:
    break;
  case QDMI_WARN_GENERAL:
    std::cerr << "Warning: " << msg << "\n";
  default:
    throwError(result, msg);
  }
}

class Site {
  friend class Device;
  friend class Operation;
  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;
  /// @brief The underlying QDMI_Site object.
  QDMI_Site site_;
  /**
   * @brief Constructs a Site object from a QDMI_Site handle.
   * @param site The QDMI_Site handle to wrap.
   */
  Site(QDMI_Device device, QDMI_Site site) : device_(device), site_(site) {}

  template <maybe_optional_value_or_string T>
  [[nodiscard]] auto queryProperty(QDMI_Site_Property prop) const -> T {
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      const auto result = QDMI_device_query_site_property(device_, site_, prop,
                                                          0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      std::string value(size - 1, '\0');
      throwIfError(QDMI_device_query_site_property(device_, site_, prop, size,
                                                   value.data(), nullptr),
                   "Querying " + toString(prop));
      return value;
    } else {
      T value{};
      const auto result = QDMI_device_query_site_property(
          device_, site_, prop, sizeof(T), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      return value;
    }
  }

public:
  /// @returns the underlying QDMI_Site object.
  [[nodiscard]] auto getQDMISite() const -> QDMI_Site { return site_; }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Site() const { return site_; }
  auto operator<=>(const Site&) const = default;
  /// @see QDMI_SITE_PROPERTY_INDEX
  [[nodiscard]] auto getIndex() const -> size_t;
  /// @see QDMI_SITE_PROPERTY_T1
  [[nodiscard]] auto getT1() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_T2
  [[nodiscard]] auto getT2() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_NAME
  [[nodiscard]] auto getName() const -> std::optional<std::string>;
  /// @see QDMI_SITE_PROPERTY_XCOORDINATE
  [[nodiscard]] auto getXCoordinate() const -> std::optional<int64_t>;
  /// @see QDMI_SITE_PROPERTY_YCOORDINATE
  [[nodiscard]] auto getYCoordinate() const -> std::optional<int64_t>;
  /// @see QDMI_SITE_PROPERTY_ZCOORDINATE
  [[nodiscard]] auto getZCoordinate() const -> std::optional<int64_t>;
  /// @see QDMI_SITE_PROPERTY_ISZONE
  [[nodiscard]] auto isZone() const -> std::optional<bool>;
  /// @see QDMI_SITE_PROPERTY_XEXTENT
  [[nodiscard]] auto getXExtent() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_YEXTENT
  [[nodiscard]] auto getYExtent() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_ZEXTENT
  [[nodiscard]] auto getZExtent() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_MODULEINDEX
  [[nodiscard]] auto getModuleIndex() const -> std::optional<uint64_t>;
  /// @see QDMI_SITE_PROPERTY_SUBMODULEINDEX
  [[nodiscard]] auto getSubmoduleIndex() const -> std::optional<uint64_t>;
};
class Operation {
  friend class Device;
  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;
  /// @brief The underlying QDMI_Operation object.
  QDMI_Operation operation_;
  /**
   * @brief Constructs an Operation object from a QDMI_Operation handle.
   * @param operation The QDMI_Operation handle to wrap.
   */
  Operation(QDMI_Device device, QDMI_Operation operation)
      : device_(device), operation_(operation) {}

  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] auto queryProperty(QDMI_Operation_Property prop,
                                   const std::vector<Site>& sites,
                                   const std::vector<double>& params) const
      -> T {
    std::vector<QDMI_Site> qdmiSites;
    qdmiSites.reserve(sites.size());
    std::ranges::transform(sites, std::back_inserter(qdmiSites),
                           [](const Site& site) -> QDMI_Site { return site; });
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      const auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      std::string value(size - 1, '\0');
      throwIfError(QDMI_device_query_operation_property(
                       device_, operation_, sites.size(), qdmiSites.data(),
                       params.size(), params.data(), prop, size, value.data(),
                       nullptr),
                   "Querying " + toString(prop));
      return value;
    } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                             T>) {
      size_t size = 0;
      const auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      T value(size / sizeof(typename T::value_type));
      throwIfError(QDMI_device_query_operation_property(
                       device_, operation_, sites.size(), qdmiSites.data(),
                       params.size(), params.data(), prop, size, value.data(),
                       nullptr),
                   "Querying " + toString(prop));
      return value;
    } else {
      T value{};
      const auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, sizeof(T), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      return value;
    }
  }

public:
  /// @returns the underlying QDMI_Operation object.
  [[nodiscard]] auto getQDMIOperation() const -> QDMI_Operation {
    return operation_;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Operation() const { return operation_; }
  auto operator<=>(const Operation&) const = default;
  /// @see QDMI_OPERATION_PROPERTY_NAME
  [[nodiscard]] auto getName(const std::vector<Site>& sites = {},
                             const std::vector<double>& params = {}) const
      -> std::string;
  /// @see QDMI_OPERATION_PROPERTY_QUBITSNUM
  [[nodiscard]] auto getQubitsNum(const std::vector<Site>& sites = {},
                                  const std::vector<double>& params = {}) const
      -> std::optional<size_t>;
  /// @see QDMI_OPERATION_PROPERTY_PARAMETERSNUM
  [[nodiscard]] auto
  getParametersNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const -> size_t;
  /// @see QDMI_OPERATION_PROPERTY_DURATION
  [[nodiscard]] auto getDuration(const std::vector<Site>& sites = {},
                                 const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  /// @see QDMI_OPERATION_PROPERTY_FIDELITY
  [[nodiscard]] auto getFidelity(const std::vector<Site>& sites = {},
                                 const std::vector<double>& params = {}) const
      -> std::optional<double>;
  /// @see QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS
  [[nodiscard]] auto
  getInteractionRadius(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  /// @see QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS
  [[nodiscard]] auto
  getBlockingRadius(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
  /// @see QDMI_OPERATION_PROPERTY_IDLINGFIDELITY
  [[nodiscard]] auto
  getIdlingFidelity(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const
      -> std::optional<double>;
  /// @see QDMI_OPERATION_PROPERTY_ISZONED
  [[nodiscard]] auto isZoned(const std::vector<Site>& sites = {},
                             const std::vector<double>& params = {}) const
      -> std::optional<bool>;
  /// @see QDMI_OPERATION_PROPERTY_SITES
  [[nodiscard]] auto getSites(const std::vector<Site>& sites = {},
                              const std::vector<double>& params = {})
      -> std::optional<std::vector<Site>>;
  /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
  [[nodiscard]] auto
  getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                        const std::vector<double>& params = {}) const
      -> std::optional<uint64_t>;
};
class Device {
  friend class FoMaC;
  /// @brief The underlying QDMI_Device object.
  QDMI_Device device_;
  /**
   * @brief Constructs a Device object from a QDMI_Device handle.
   * @param device The QDMI_Device handle to wrap.
   */
  explicit Device(QDMI_Device device) : device_(device) {}

  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] auto queryProperty(QDMI_Device_Property prop) const -> T {
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      const auto result =
          QDMI_device_query_device_property(device_, prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      std::string value(size - 1, '\0');
      throwIfError(QDMI_device_query_device_property(device_, prop, size,
                                                     value.data(), nullptr),
                   "Querying " + toString(prop));
      return value;
    } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                             T>) {
      size_t size = 0;
      const auto result =
          QDMI_device_query_device_property(device_, prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      T value(size / sizeof(typename T::value_type));
      throwIfError(QDMI_device_query_device_property(device_, prop, size,
                                                     value.data(), nullptr),
                   "Querying " + toString(prop));
      return value;
    } else {
      T value{};
      const auto result = QDMI_device_query_device_property(
          device_, prop, sizeof(T), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      throwIfError(result, "Querying " + toString(prop));
      return value;
    }
  }

public:
  /// @returns the underlying QDMI_Device object.
  [[nodiscard]] auto getQDMIDevice() const -> QDMI_Device { return device_; }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Device() const { return device_; }
  auto operator<=>(const Device&) const = default;
  /// @see QDMI_DEVICE_PROPERTY_NAME
  [[nodiscard]] auto getName() const -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_VERSION
  [[nodiscard]] auto getVersion() const -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_STATUS
  [[nodiscard]] auto getStatus() const -> QDMI_Device_Status;
  /// @see QDMI_DEVICE_PROPERTY_LIBRARYVERSION
  [[nodiscard]] auto getLibraryVersion() const -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_QUBITSNUM
  [[nodiscard]] auto getQubitsNum() const -> size_t;
  /// @see QDMI_DEVICE_PROPERTY_SITES
  [[nodiscard]] auto getSites() const -> std::vector<Site>;
  /// @see QDMI_DEVICE_PROPERTY_OPERATIONS
  [[nodiscard]] auto getOperations() const -> std::vector<Operation>;
  /// @see QDMI_DEVICE_PROPERTY_COUPLINGMAP
  [[nodiscard]] auto getCouplingMap() const
      -> std::optional<std::vector<std::pair<Site, Site>>>;
  /// @see QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION
  [[nodiscard]] auto getNeedsCalibration() const -> std::optional<size_t>;
  /// @see QDMI_DEVICE_PROPERTY_LENGTHUNIT
  [[nodiscard]] auto getLengthUnit() const -> std::optional<std::string>;
  /// @see QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR
  [[nodiscard]] auto getLengthScaleFactor() const -> std::optional<double>;
  /// @see QDMI_DEVICE_PROPERTY_DURATIONUNIT
  [[nodiscard]] auto getDurationUnit() const -> std::optional<std::string>;
  /// @see QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR
  [[nodiscard]] auto getDurationScaleFactor() const -> std::optional<double>;
  /// @see QDMI_DEVICE_PROPERTY_MINATOMDISTANCE
  [[nodiscard]] auto getMinAtomDistance() const -> std::optional<uint64_t>;
};
class FoMaC {
  QDMI_Session session_ = nullptr;

  FoMaC();
  static auto get() -> FoMaC& {
    static FoMaC instance;
    return instance;
  }
  template <size_constructible_contiguous_range T>
  [[nodiscard]] auto queryProperty(QDMI_Session_Property prop) const -> T {
    size_t size = 0;
    throwIfError(
        QDMI_session_query_session_property(session_, prop, 0, nullptr, &size),
        "Querying " + toString(prop));
    T value(size / sizeof(typename T::value_type));
    throwIfError(QDMI_session_query_session_property(session_, prop, size,
                                                     value.data(), nullptr),
                 "Querying " + toString(prop));
    return value;
  }

public:
  virtual ~FoMaC();
  // Delete copy constructors and assignment operators to prevent copying the
  // singleton instance.
  FoMaC(const FoMaC&) = delete;
  FoMaC& operator=(const FoMaC&) = delete;
  FoMaC(FoMaC&&) = default;
  FoMaC& operator=(FoMaC&&) = default;
  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] static auto getDevices() -> std::vector<Device>;
};
} // namespace fomac
