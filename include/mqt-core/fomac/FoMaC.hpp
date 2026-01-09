/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "qdmi/Common.hpp"

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <qdmi/client.h>
#include <ranges>
#include <string>
#include <type_traits>
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

/**
 * @brief Configuration structure for session authentication parameters.
 * @details All parameters are optional. Only set the parameters needed for
 * your authentication method. Parameters are validated when the session is
 * constructed.
 */
struct SessionConfig {
  /// Authentication token
  std::optional<std::string> token;
  /// Path to file containing authentication information
  std::optional<std::string> authFile;
  /// URL to authentication server
  std::optional<std::string> authUrl;
  /// Username for authentication
  std::optional<std::string> username;
  /// Password for authentication
  std::optional<std::string> password;
  /// Project ID for session
  std::optional<std::string> projectId;
  /// Custom configuration parameter 1
  std::optional<std::string> custom1;
  /// Custom configuration parameter 2
  std::optional<std::string> custom2;
  /// Custom configuration parameter 3
  std::optional<std::string> custom3;
  /// Custom configuration parameter 4
  std::optional<std::string> custom4;
  /// Custom configuration parameter 5
  std::optional<std::string> custom5;
};

/**
 * @brief Class representing the Session library.
 * @details This class provides methods to query available devices and
 * manage the QDMI session.
 * @see QDMI_Session
 */
class Session {
  /**
   * @brief Private token class.
   * @details Only the Session class can create instances of this class.
   */
  class Token {
  public:
    Token() = default;
  };

public:
  /**
   * @brief Class representing a submitted job.
   * @details This class provides methods to query job status and retrieve
   * results.
   * @see QDMI_Job
   */
  class Job {
    QDMI_Job job_;

  public:
    /**
     * @brief Constructs a Job object from a QDMI_Job handle.
     * @param job The QDMI_Job handle to wrap.
     */
    explicit Job(QDMI_Job job) : job_(job) {}
    /**
     * @brief Destructor that releases the underlying QDMI_Job resource.
     */
    ~Job() {
      if (job_ != nullptr) {
        QDMI_job_free(job_);
      }
    }
    // Delete copy constructor and copy assignment operator to prevent
    // pointer duplication and double-free
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
    // Default move constructor and move assignment operator to allow
    // safe ownership transfer
    Job(Job&& other) noexcept : job_(other.job_) { other.job_ = nullptr; }
    Job& operator=(Job&& other) noexcept {
      if (this != &other) {
        if (job_ != nullptr) {
          QDMI_job_free(job_);
        }
        job_ = other.job_;
        other.job_ = nullptr;
      }
      return *this;
    }
    /// @returns the underlying QDMI_Job object.
    [[nodiscard]] auto getQDMIJob() const -> QDMI_Job { return job_; }
    // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
    operator QDMI_Job() const { return job_; }
    /// @see QDMI_job_check
    [[nodiscard]] auto check() const -> QDMI_Job_Status;
    /**
     * @brief @see QDMI_job_wait
     * @param timeout The maximum time to wait in seconds. 0 (default) means
     * wait indefinitely.
     * @return true if the job completed successfully, false if it timed out
     */
    [[nodiscard]] auto wait(size_t timeout = 0) const -> bool;
    /// @see QDMI_job_cancel
    auto cancel() const -> void;
    /// Get the job ID
    [[nodiscard]] auto getId() const -> std::string;
    /// Get the program format
    [[nodiscard]] auto getProgramFormat() const -> QDMI_Program_Format;
    /// Get the program to be executed
    [[nodiscard]] auto getProgram() const -> std::string;
    /// Get the number of shots
    [[nodiscard]] auto getNumShots() const -> size_t;
    /**
     * @brief Returns the measurement shots as a vector of bitstrings.
     * @see QDMI_JOB_RESULT_SHOTS
     */
    [[nodiscard]] auto getShots() const -> std::vector<std::string>;
    /**
     * @brief Returns a map of measurement outcomes to their respective counts.
     * @see QDMI_JOB_RESULT_HIST_KEYS
     * @see QDMI_JOB_RESULT_HIST_VALUES
     */
    [[nodiscard]] auto getCounts() const -> std::map<std::string, size_t>;
    /**
     * @brief Returns the dense state vector as a vector of complex numbers.
     * @see QDMI_JOB_RESULT_STATEVECTOR_DENSE
     */
    [[nodiscard]] auto getDenseStateVector() const
        -> std::vector<std::complex<double>>;
    /**
     * @brief Returns the dense probabilities as a vector of doubles.
     * @see QDMI_JOB_RESULT_PROBABILITIES_DENSE
     */
    [[nodiscard]] auto getDenseProbabilities() const -> std::vector<double>;
    /**
     * @brief Returns the sparse state vector as a map of bitstrings to complex
     * amplitudes.
     * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS
     * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES
     */
    [[nodiscard]] auto getSparseStateVector() const
        -> std::map<std::string, std::complex<double>>;
    /**
     * @brief Returns the sparse probabilities as a map of bitstrings to
     * probabilities.
     * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS
     * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES
     */
    [[nodiscard]] auto getSparseProbabilities() const
        -> std::map<std::string, double>;
  };

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
      [[nodiscard]] auto queryProperty(const QDMI_Site_Property prop) const
          -> T {
        std::string msg = "Querying ";
        msg += qdmi::toString(prop);
        if constexpr (string_or_optional_string<T>) {
          size_t size = 0;
          auto result = QDMI_device_query_site_property(device_, site_, prop, 0,
                                                        nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          qdmi::throwIfError(result, msg);
          std::string value(size - 1, '\0');
          result = QDMI_device_query_site_property(device_, site_, prop, size,
                                                   value.data(), nullptr);
          qdmi::throwIfError(result, msg);
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
          qdmi::throwIfError(result, msg);
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
      // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
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
      [[nodiscard]] auto isZone() const -> bool;
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
      [[nodiscard]] auto queryProperty(const QDMI_Operation_Property prop,
                                       const std::vector<Site>& sites,
                                       const std::vector<double>& params) const
          -> T {
        std::string msg = "Querying ";
        msg += qdmi::toString(prop);
        std::vector<QDMI_Site> qdmiSites;
        qdmiSites.reserve(sites.size());
        std::ranges::transform(
            sites, std::back_inserter(qdmiSites),
            [](const Site& site) -> QDMI_Site { return site; });
        if constexpr (string_or_optional_string<T>) {
          size_t size = 0;
          auto result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, 0, nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          qdmi::throwIfError(result, msg);
          std::string value(size - 1, '\0');
          result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, size, value.data(), nullptr);
          qdmi::throwIfError(result, msg);
          return value;
        } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                                 T>) {
          size_t size = 0;
          auto result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, 0, nullptr, &size);
          if constexpr (is_optional<T>) {
            if (result == QDMI_ERROR_NOTSUPPORTED) {
              return std::nullopt;
            }
          }
          qdmi::throwIfError(result, msg);
          remove_optional_t<T> value(
              size / sizeof(typename remove_optional_t<T>::value_type));
          result = QDMI_device_query_operation_property(
              device_, operation_, sites.size(), qdmiSites.data(),
              params.size(), params.data(), prop, size, value.data(), nullptr);
          qdmi::throwIfError(result, msg);
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
          qdmi::throwIfError(result, msg);
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
      // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
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
      [[nodiscard]] auto isZoned() const -> bool;
      /// @see QDMI_OPERATION_PROPERTY_SITES
      [[nodiscard]] auto getSites() const -> std::optional<std::vector<Site>>;
      /**
       * @brief Returns the list of site pairs the local 2-qubit operation can
       * be performed on.
       * @details For local 2-qubit operations, this function interprets the
       * returned list of sites by QDMI as site pairs according to the QDMI
       * specification. Hence, this function facilitates easier iteration over
       * supported site pairs.
       * @return Optional vector of site pairs if this is a local 2-qubit
       * operation, std::nullopt otherwise.
       * @see QDMI_OPERATION_PROPERTY_SITES
       */
      [[nodiscard]] auto getSitePairs() const
          -> std::optional<std::vector<std::pair<Site, Site>>>;
      /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
      [[nodiscard]] auto
      getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                            const std::vector<double>& params = {}) const
          -> std::optional<uint64_t>;
    };

  private:
    /// @brief The underlying QDMI_Device object.
    QDMI_Device device_;

    template <maybe_optional_value_or_string_or_vector T>
    [[nodiscard]] auto queryProperty(const QDMI_Device_Property prop) const
        -> T {
      std::string msg = "Querying ";
      msg += qdmi::toString(prop);
      if constexpr (string_or_optional_string<T>) {
        size_t size = 0;
        auto result =
            QDMI_device_query_device_property(device_, prop, 0, nullptr, &size);
        if constexpr (is_optional<T>) {
          if (result == QDMI_ERROR_NOTSUPPORTED) {
            return std::nullopt;
          }
        }
        qdmi::throwIfError(result, msg);
        std::string value(size - 1, '\0');
        result = QDMI_device_query_device_property(device_, prop, size,
                                                   value.data(), nullptr);
        qdmi::throwIfError(result, msg);
        return value;
      } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                               T>) {
        size_t size = 0;
        auto result =
            QDMI_device_query_device_property(device_, prop, 0, nullptr, &size);
        if constexpr (is_optional<T>) {
          if (result == QDMI_ERROR_NOTSUPPORTED) {
            return std::nullopt;
          }
        }
        qdmi::throwIfError(result, msg);
        remove_optional_t<T> value(
            size / sizeof(typename remove_optional_t<T>::value_type));
        result = QDMI_device_query_device_property(device_, prop, size,
                                                   value.data(), nullptr);
        qdmi::throwIfError(result, msg);
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
        qdmi::throwIfError(result, msg);
        return value;
      }
    }

  public:
    /**
     * @brief Constructs a Device object from a QDMI_Device handle.
     * @param device The QDMI_Device handle to wrap.
     */
    Device(Session::Token /* unused */, QDMI_Device device) : device_(device) {}
    /**
     * @brief Creates a Device object from a QDMI_Device handle.
     * @param device The QDMI_Device handle to wrap.
     * @return A Device object wrapping the given handle.
     * @note This is a factory method for use in bindings where Token
     * construction is not accessible.
     */
    [[nodiscard]] static auto fromQDMIDevice(QDMI_Device device) -> Device {
      return Device(Session::Token{}, device);
    }
    /// @returns the underlying QDMI_Device object.
    [[nodiscard]] auto getQDMIDevice() const -> QDMI_Device { return device_; }
    // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
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
    /**
     * @brief Returns the list of regular sites (without zone sites) available
     * on the device.
     * @details Filters all sites and only returns regular sites, i.e., where
     * `isZone()` yields `false`. These represent actual potential physical
     * qubit locations on the device lattice.
     * @returns vector of regular sites
     * @see QDMI_DEVICE_PROPERTY_SITES
     */
    [[nodiscard]] auto getRegularSites() const -> std::vector<Site>;
    /**
     * @brief Returns the list of zone sites (without regular sites) available
     * on the device.
     * @details Filters all sites and only returns zone sites, i.e., where
     * `isZone()` yields `true`. These represent a zone, i.e., an extent where
     * zoned operations can be performed, not individual qubit locations.
     * @returns a vector of zone sites
     * @see QDMI_DEVICE_PROPERTY_SITES
     */
    [[nodiscard]] auto getZones() const -> std::vector<Site>;
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
    /// @see QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS
    [[nodiscard]] auto getSupportedProgramFormats() const
        -> std::vector<QDMI_Program_Format>;
    /// @see QDMI_job_submit
    [[nodiscard]] auto submitJob(const std::string& program,
                                 QDMI_Program_Format format,
                                 size_t numShots) const -> Job;
  };

private:
  QDMI_Session session_ = nullptr;

  template <size_constructible_contiguous_range T>
  [[nodiscard]] auto queryProperty(const QDMI_Session_Property prop) const
      -> T {
    std::string msg = "Querying ";
    msg += qdmi::toString(prop);
    size_t size = 0;
    auto result =
        QDMI_session_query_session_property(session_, prop, 0, nullptr, &size);
    qdmi::throwIfError(result, msg);
    remove_optional_t<T> value(
        size / sizeof(typename remove_optional_t<T>::value_type));
    result = QDMI_session_query_session_property(session_, prop, size,
                                                 value.data(), nullptr);
    qdmi::throwIfError(result, msg);
    return value;
  }

public:
  /**
   * @brief Constructs a new QDMI Session with optional authentication.
   * @param config Optional session configuration containing authentication
   * parameters. If not provided, uses default (no authentication).
   * @details Creates, allocates, and initializes a new QDMI session.
   */
  explicit Session(const SessionConfig& config = {});

  /**
   * @brief Destructor that releases the QDMI session.
   */
  ~Session();

  // Delete copy constructors and assignment operators
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Allow move semantics
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;

  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] auto getDevices() -> std::vector<Device>;
};
} // namespace fomac
