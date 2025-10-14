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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <qdmi/client.h>
#include <ranges>
#include <string>
#include <type_traits>
#include <unordered_map>
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

/// @see remove_optional_t
template <typename T> struct remove_optional {
  using type = T;
};

/// @see remove_optional_t
template <typename U> struct remove_optional<std::optional<U>> {
  using type = U;
};

/**
 * @brief Helper type to strip std::optional from a type if it is present.
 * @details This is useful for template metaprogramming when you want to work
 * with the underlying type of optional without caring about its optionality.
 * @tparam T The type to strip optional from.
 */
template <typename T>
using remove_optional_t = typename remove_optional<T>::type;

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
    size_constructible_contiguous_range<remove_optional_t<T>>;

/**
 * @brief Concept for types that are either value_or_string or std::optional of
 * value_or_string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Site::queryProperty
 */
template <typename T>
concept maybe_optional_value_or_string = value_or_string<remove_optional_t<T>>;

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
    value_or_string_or_vector<remove_optional_t<T>>;

/// @returns the string representation of the given QDMI_STATUS.
auto toString(QDMI_STATUS result) -> std::string;

/// @returns the string representation of the given QDMI_Site_Property.
auto toString(QDMI_Site_Property prop) -> std::string;

/// @returns the string representation of the given QDMI_Operation_Property.
auto toString(QDMI_Operation_Property prop) -> std::string;

/// @returns the string representation of the given QDMI_Device_Property.
auto toString(QDMI_Device_Property prop) -> std::string;

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
    break;
  default:
    throwError(result, msg);
  }
}

/**
 * @brief Class representing the FoMaC library.
 * @details This class provides methods to query available devices and
 * manage the QDMI session.
 * @note This class is a singleton.
 * @see QDMI_Session
 */
class FoMaC {
  /**
   * @brief Private token class.
   * @details Only the FoMaC class can create instances of this class.
   */
  class Token {
  public:
    Token() = default;
  };

public:
  /**
   * @brief Class representing a quantum device.
   * @details This class provides methods to query properties of the device,
   * its sites, and its operations.
   * @see QDMI_Device
   */
  class Device {
    /**
     * @brief Private token class.
     * @details Only the Device class can create instances of this class.
     */
    class Token {
    public:
      Token() = default;
    };

  public:
    /**
     * @brief Class representing a site (qubit) on the device.
     * @details This class provides methods to query properties of the site.
     * @see QDMI_Site
     */
    class Site {
      /// @brief The associated QDMI_Device object.
      QDMI_Device device_;
      /// @brief The underlying QDMI_Site object.
      QDMI_Site site_;

      template <maybe_optional_value_or_string T>
      [[nodiscard]] auto queryProperty(QDMI_Site_Property prop) const -> T {
        if constexpr (string_or_optional_string<T>) {
          size_t size = 0;
          const auto result = QDMI_device_query_site_property(
              device_, site_, prop, 0, nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          throwIfError(result, "Querying " + toString(prop));
          std::string value(size - 1, '\0');
          throwIfError(QDMI_device_query_site_property(
                           device_, site_, prop, size, value.data(), nullptr),
                       "Querying " + toString(prop));
          return value;
        } else {
          remove_optional_t<T> value{};
          const auto result = QDMI_device_query_site_property(
              device_, site_, prop, sizeof(remove_optional_t<T>), &value,
              nullptr);
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
      /**
       * @brief Constructs a Site object from a QDMI_Site handle.
       * @param device The associated QDMI_Device handle.
       * @param site The QDMI_Site handle to wrap.
       */
      Site(Token /* unused */, QDMI_Device device, QDMI_Site site)
          : device_(device), site_(site) {}
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
    /**
     * @brief Class representing an operation (gate) supported by the device.
     * @details This class provides methods to query properties of the
     * operation.
     * @see QDMI_Operation
     */
    class Operation {
      /// @brief The associated QDMI_Device object.
      QDMI_Device device_;
      /// @brief The underlying QDMI_Operation object.
      QDMI_Operation operation_;

      template <maybe_optional_value_or_string_or_vector T>
      [[nodiscard]] auto queryProperty(QDMI_Operation_Property prop,
                                       const std::vector<Site>& sites,
                                       const std::vector<double>& params) const
          -> T {
        std::vector<QDMI_Site> qdmiSites;
        qdmiSites.reserve(sites.size());
        std::ranges::transform(
            sites, std::back_inserter(qdmiSites),
            [](const Site& site) -> QDMI_Site { return site; });
        if constexpr (string_or_optional_string<T>) {
          size_t size = 0;
          const auto result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, 0, nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          throwIfError(result, "Querying " + toString(prop));
          std::string value(size - 1, '\0');
          throwIfError(QDMI_device_query_operation_property(
                           device_, operation_, sites.size(), qdmiSites.data(),
                           params.size(), params.data(), prop, size,
                           value.data(), nullptr),
                       "Querying " + toString(prop));
          return value;
        } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                                 T>) {
          size_t size = 0;
          const auto result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, 0, nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          throwIfError(result, "Querying " + toString(prop));
          remove_optional_t<T> value(
              size / sizeof(typename remove_optional_t<T>::value_type));
          throwIfError(QDMI_device_query_operation_property(
                           device_, operation_, sites.size(), qdmiSites.data(),
                           params.size(), params.data(), prop, size,
                           value.data(), nullptr),
                       "Querying " + toString(prop));
          return value;
        } else {
          remove_optional_t<T> value{};
          const auto result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, sizeof(remove_optional_t<T>),
              &value, nullptr);
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
      /**
       * @brief Constructs an Operation object from a QDMI_Operation handle.
       * @param device The associated QDMI_Device handle.
       * @param operation The QDMI_Operation handle to wrap.
       */
      Operation(Token /* unused */, QDMI_Device device,
                QDMI_Operation operation)
          : device_(device), operation_(operation) {}
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
      [[nodiscard]] auto
      getQubitsNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const
          -> std::optional<size_t>;
      /// @see QDMI_OPERATION_PROPERTY_PARAMETERSNUM
      [[nodiscard]] auto
      getParametersNum(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const -> size_t;
      /// @see QDMI_OPERATION_PROPERTY_DURATION
      [[nodiscard]] auto
      getDuration(const std::vector<Site>& sites = {},
                  const std::vector<double>& params = {}) const
          -> std::optional<uint64_t>;
      /// @see QDMI_OPERATION_PROPERTY_FIDELITY
      [[nodiscard]] auto
      getFidelity(const std::vector<Site>& sites = {},
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
                                  const std::vector<double>& params = {}) const
          -> std::optional<std::vector<Site>>;
      /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
      [[nodiscard]] auto
      getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                            const std::vector<double>& params = {}) const
          -> std::optional<uint64_t>;
    };

  private:
    /// @brief The underlying QDMI_Device object.
    QDMI_Device device_;

    /// @brief Mutex for thread-safe cache access.
    mutable std::mutex cacheMutex_;
    /// @brief Cached sites list.
    mutable std::optional<std::vector<Site>> cachedSites_;
    /// @brief Cached operations list.
    mutable std::optional<std::vector<Operation>> cachedOperations_;
    /// @brief Cached coupling map.
    mutable std::optional<std::optional<std::vector<std::pair<Site, Site>>>>
        cachedCouplingMap_;

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
        remove_optional_t<T> value(
            size / sizeof(typename remove_optional_t<T>::value_type));
        throwIfError(QDMI_device_query_device_property(device_, prop, size,
                                                       value.data(), nullptr),
                     "Querying " + toString(prop));
        return value;
      } else {
        remove_optional_t<T> value{};
        const auto result = QDMI_device_query_device_property(
            device_, prop, sizeof(remove_optional_t<T>), &value, nullptr);
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
    /**
     * @brief Constructs a Device object from a QDMI_Device handle.
     * @param device The QDMI_Device handle to wrap.
     */
    Device(FoMaC::Token /* unused */, QDMI_Device device) : device_(device) {}

    // Delete copy constructor and copy assignment (due to mutex member)
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // Define move constructor and move assignment
    Device(Device&& other) noexcept
        : device_(other.device_), cachedSites_(std::move(other.cachedSites_)),
          cachedOperations_(std::move(other.cachedOperations_)),
          cachedCouplingMap_(std::move(other.cachedCouplingMap_)) {}

    Device& operator=(Device&& other) noexcept {
      if (this != &other) {
        device_ = other.device_;
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cachedSites_ = std::move(other.cachedSites_);
        cachedOperations_ = std::move(other.cachedOperations_);
        cachedCouplingMap_ = std::move(other.cachedCouplingMap_);
      }
      return *this;
    }

    /// @returns the underlying QDMI_Device object.
    [[nodiscard]] auto getQDMIDevice() const -> QDMI_Device { return device_; }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator QDMI_Device() const { return device_; }

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

private:
  QDMI_Session session_ = nullptr;

  FoMaC();
  static auto get() -> FoMaC& {
    static FoMaC instance;
    return instance;
  }
  template <size_constructible_contiguous_range T>
  [[nodiscard]] auto queryProperty(const QDMI_Session_Property prop) const
      -> T {
    size_t size = 0;
    throwIfError(
        QDMI_session_query_session_property(session_, prop, 0, nullptr, &size),
        "Querying " + toString(prop));
    remove_optional_t<T> value(
        size / sizeof(typename remove_optional_t<T>::value_type));
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
