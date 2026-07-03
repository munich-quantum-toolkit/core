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

#include "qdmi/common/Common.hpp"
#include "qdmi/types.h"

#include <qdmi/client.h>

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
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
    std::ranges::contiguous_range<T> && std::constructible_from<T, size_t> &&
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
    std::integral<T> || std::floating_point<T> || std::same_as<T, bool> ||
    std::same_as<T, std::string> || std::same_as<T, QDMI_Device_Status>;

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
                      std::same_as<T, std::optional<typename T::value_type>>;

/**
 * @brief Concept for types that are either std::string or std::optional of
 * std::string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept string_or_optional_string =
    std::same_as<T, std::string> ||
    (is_optional<T> && std::same_as<typename T::value_type, std::string>);

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

class Job;
class Site;
class Device;
class Operation;

/**
 * @brief Class representing the Session library.
 * @details This class provides methods to query available devices and
 * manage the QDMI session.
 * @see QDMI_Session
 */
class Session {
public:
  /**
   * @brief Constructs a new QDMI Session with optional authentication.
   * @param config Optional session configuration containing authentication
   * parameters. If not provided, uses default (no authentication).
   * @details Creates, allocates, and initializes a new QDMI session.
   */
  explicit Session(const SessionConfig& config = {});

  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] std::vector<Device> getDevices();

private:
  /// Query a session property.
  template <size_constructible_contiguous_range T>
  [[nodiscard]] T queryProperty(const QDMI_Session_Property prop) const {
    std::string msg = "Querying ";
    msg += qdmi::toString(prop);
    size_t size = 0;
    auto result = QDMI_session_query_session_property(session_.get(), prop, 0,
                                                      nullptr, &size);
    qdmi::throwIfError(result, msg);
    remove_optional_t<T> value(
        size / sizeof(typename remove_optional_t<T>::value_type));
    result = QDMI_session_query_session_property(session_.get(), prop, size,
                                                 value.data(), nullptr);
    qdmi::throwIfError(result, msg);
    return value;
  }

  std::unique_ptr<QDMI_Session_impl_d, decltype(&QDMI_session_free)> session_{
      nullptr, QDMI_session_free};
};

static_assert(!std::is_copy_constructible<Session>());
static_assert(!std::is_copy_assignable<Session>());
static_assert(std::is_move_constructible<Session>());
static_assert(std::is_move_assignable<Session>());

/**
 * @brief Class representing a quantum device.
 * @details
 * This class provides methods to query properties of the device,
 * its sites, and its operations.
 *
 * The class can only be constructed by Session instances.
 *
 * @see QDMI_Device
 */
class Device {
public:
  /**
   * @brief Creates a Device object from a QDMI_Device handle.
   * @param device The QDMI_Device handle to wrap.
   * @return A Device object wrapping the given handle.
   * @note This is a factory method for use in bindings where a
   * session is not accessible.
   */
  [[nodiscard]] static Device fromQDMIDevice(QDMI_Device device) {
    return Device(device);
  }

  /// @returns the underlying QDMI_Device object.
  [[nodiscard]] QDMI_Device getQDMIDevice() const { return device_; }

  // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
  operator QDMI_Device() const { return device_; }

  /// @see QDMI_DEVICE_PROPERTY_NAME
  [[nodiscard]] std::string getName() const;

  /// @see QDMI_DEVICE_PROPERTY_VERSION
  [[nodiscard]] std::string getVersion() const;

  /// @see QDMI_DEVICE_PROPERTY_STATUS
  [[nodiscard]] QDMI_Device_Status getStatus() const;

  /// @see QDMI_DEVICE_PROPERTY_LIBRARYVERSION
  [[nodiscard]] std::string getLibraryVersion() const;

  /// @see QDMI_DEVICE_PROPERTY_QUBITSNUM
  [[nodiscard]] size_t getQubitsNum() const;

  /// @see QDMI_DEVICE_PROPERTY_SITES
  [[nodiscard]] std::vector<Site> getSites() const;

  /**
   * @brief Returns the list of regular sites (without zone sites) available
   * on the device.
   * @details Filters all sites and only returns regular sites, i.e., where
   * `isZone()` yields `false`. These represent actual potential physical
   * qubit locations on the device lattice.
   * @returns vector of regular sites
   * @see QDMI_DEVICE_PROPERTY_SITES
   */
  [[nodiscard]] std::vector<Site> getRegularSites() const;

  /**
   * @brief Returns the list of zone sites (without regular sites) available
   * on the device.
   * @details Filters all sites and only returns zone sites, i.e., where
   * `isZone()` yields `true`. These represent a zone, i.e., an extent where
   * zoned operations can be performed, not individual qubit locations.
   * @returns a vector of zone sites
   * @see QDMI_DEVICE_PROPERTY_SITES
   */
  [[nodiscard]] std::vector<Site> getZones() const;

  /// @see QDMI_DEVICE_PROPERTY_OPERATIONS
  [[nodiscard]] std::vector<Operation> getOperations() const;

  /// @see QDMI_DEVICE_PROPERTY_COUPLINGMAP
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getCouplingMap() const;

  /// @see QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION
  [[nodiscard]] std::optional<size_t> getNeedsCalibration() const;

  /// @see QDMI_DEVICE_PROPERTY_LENGTHUNIT
  [[nodiscard]] std::optional<std::string> getLengthUnit() const;

  /// @see QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR
  [[nodiscard]] std::optional<double> getLengthScaleFactor() const;

  /// @see QDMI_DEVICE_PROPERTY_DURATIONUNIT
  [[nodiscard]] std::optional<std::string> getDurationUnit() const;

  /// @see QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR
  [[nodiscard]] std::optional<double> getDurationScaleFactor() const;

  /// @see QDMI_DEVICE_PROPERTY_MINATOMDISTANCE
  [[nodiscard]] std::optional<uint64_t> getMinAtomDistance() const;

  /// @see QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS
  [[nodiscard]] std::vector<QDMI_Program_Format>
  getSupportedProgramFormats() const;

  /// @see QDMI_job_submit
  [[nodiscard]] Job submitJob(const std::string& program,
                              QDMI_Program_Format format,
                              size_t numShots) const;

  auto operator<=>(const Device&) const noexcept = default;

private:
  /**
   * @brief Constructs a Device object from a QDMI_Device handle.
   * @param device The QDMI_Device handle to wrap.
   */
  explicit Device(QDMI_Device device) : device_(device) {}

  /// Query a device property.
  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] T queryProperty(const QDMI_Device_Property prop) const {
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

  /// @brief The underlying QDMI_Device object.
  QDMI_Device device_;

  friend std::vector<Device> Session::getDevices();
};

/**
 * @brief Class representing a submitted job.
 * @details
 * This class provides methods to query job status and retrieve
 * results.
 *
 * The class can only be constructed by Device instances.
 *
 * @see QDMI_Job
 */
class Job {
public:
  /// @returns the underlying QDMI_Job object.
  [[nodiscard]] QDMI_Job getQDMIJob() const { return job_.get(); }

  // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
  operator QDMI_Job() const { return job_.get(); }

  /// @see QDMI_job_check
  [[nodiscard]] QDMI_Job_Status check() const;

  /**
   * @brief @see QDMI_job_wait
   * @param timeout The maximum time to wait in seconds. 0 (default) means
   * wait indefinitely.
   * @return true if the job completed successfully, false if it timed out
   */
  [[nodiscard]] bool wait(size_t timeout = 0) const;

  /// @see QDMI_job_cancel
  void cancel() const;

  /// Get the job ID
  [[nodiscard]] std::string getId() const;

  /// Get the program format
  [[nodiscard]] QDMI_Program_Format getProgramFormat() const;

  /// Get the program to be executed
  [[nodiscard]] std::string getProgram() const;

  /// Get the number of shots
  [[nodiscard]] size_t getNumShots() const;

  /**
   * @brief Returns the measurement shots as a vector of bitstrings.
   * @see QDMI_JOB_RESULT_SHOTS
   */
  [[nodiscard]] std::vector<std::string> getShots() const;

  /**
   * @brief Returns a map of measurement outcomes to their respective counts.
   * @see QDMI_JOB_RESULT_HIST_KEYS
   * @see QDMI_JOB_RESULT_HIST_VALUES
   */
  [[nodiscard]] std::map<std::string, size_t> getCounts() const;

  /**
   * @brief Returns the dense state vector as a vector of complex numbers.
   * @see QDMI_JOB_RESULT_STATEVECTOR_DENSE
   */
  [[nodiscard]] std::vector<std::complex<double>> getDenseStateVector() const;

  /**
   * @brief Returns the dense probabilities as a vector of doubles.
   * @see QDMI_JOB_RESULT_PROBABILITIES_DENSE
   */
  [[nodiscard]] std::vector<double> getDenseProbabilities() const;

  /**
   * @brief Returns the sparse state vector as a map of bitstrings to complex
   * amplitudes.
   * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS
   * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES
   */
  [[nodiscard]] std::map<std::string, std::complex<double>>
  getSparseStateVector() const;

  /**
   * @brief Returns the sparse probabilities as a map of bitstrings to
   * probabilities.
   * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS
   * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES
   */
  [[nodiscard]] std::map<std::string, double> getSparseProbabilities() const;

private:
  /**
   * @brief Constructs a Job object from a QDMI_Job handle.
   * @param job The QDMI_Job handle to wrap.
   */
  explicit Job(QDMI_Job job) : job_(job, QDMI_job_free) {}

  std::unique_ptr<QDMI_Job_impl_d, decltype(&QDMI_job_free)> job_{
      nullptr, QDMI_job_free};

  friend class Device;
};

static_assert(!std::is_copy_constructible<Job>());
static_assert(!std::is_copy_assignable<Job>());
static_assert(std::is_move_constructible<Job>());
static_assert(std::is_move_assignable<Job>());

/**
 * @brief Class representing a site (qubit) on the device.
 * @details
 * This class provides methods to query properties of the site.
 *
 * The class can only be constructed by Device and Operation instances.
 *
 * @see QDMI_Site
 */
class Site {
public:
  /// @returns the underlying QDMI_Site object.
  [[nodiscard]] QDMI_Site getQDMISite() const { return site_; }

  // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
  operator QDMI_Site() const { return site_; }

  /// @see QDMI_SITE_PROPERTY_INDEX
  [[nodiscard]] size_t getIndex() const;

  /// @see QDMI_SITE_PROPERTY_T1
  [[nodiscard]] std::optional<uint64_t> getT1() const;

  /// @see QDMI_SITE_PROPERTY_T2
  [[nodiscard]] std::optional<uint64_t> getT2() const;

  /// @see QDMI_SITE_PROPERTY_NAME
  [[nodiscard]] std::optional<std::string> getName() const;

  /// @see QDMI_SITE_PROPERTY_XCOORDINATE
  [[nodiscard]] std::optional<int64_t> getXCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_YCOORDINATE
  [[nodiscard]] std::optional<int64_t> getYCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_ZCOORDINATE
  [[nodiscard]] std::optional<int64_t> getZCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_ISZONE
  [[nodiscard]] bool isZone() const;

  /// @see QDMI_SITE_PROPERTY_XEXTENT
  [[nodiscard]] std::optional<uint64_t> getXExtent() const;

  /// @see QDMI_SITE_PROPERTY_YEXTENT
  [[nodiscard]] std::optional<uint64_t> getYExtent() const;

  /// @see QDMI_SITE_PROPERTY_ZEXTENT
  [[nodiscard]] std::optional<uint64_t> getZExtent() const;

  /// @see QDMI_SITE_PROPERTY_MODULEINDEX
  [[nodiscard]] std::optional<uint64_t> getModuleIndex() const;

  /// @see QDMI_SITE_PROPERTY_SUBMODULEINDEX
  [[nodiscard]] std::optional<uint64_t> getSubmoduleIndex() const;

  auto operator<=>(const Site&) const noexcept = default;

private:
  /**
   * @brief Constructs a Site object from a QDMI_Site handle.
   * @param device The associated QDMI_Device handle.
   * @param site The QDMI_Site handle to wrap.
   */
  Site(QDMI_Device device, QDMI_Site site) : device_(device), site_(site) {}

  /// Query a site property.
  template <maybe_optional_value_or_string T>
  [[nodiscard]] T queryProperty(const QDMI_Site_Property prop) const {
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
          device_, site_, prop, sizeof(remove_optional_t<T>), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      return value;
    }
  }

  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;

  /// @brief The underlying QDMI_Site object.
  QDMI_Site site_;

  friend class Device;
  friend class Operation;
};

/**
 * @brief Class representing an operation (gate) supported by the device.
 * @details
 * This class provides methods to query properties of the
 * operation.
 *
 * The class can only be constructed by Device instances.
 *
 * @see QDMI_Operation
 */
class Operation {
public:
  /// @returns the underlying QDMI_Operation object.
  [[nodiscard]] QDMI_Operation getQDMIOperation() const { return operation_; }

  // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
  operator QDMI_Operation() const { return operation_; }

  /// @see QDMI_OPERATION_PROPERTY_NAME
  [[nodiscard]] std::string
  getName(const std::vector<Site>& sites = {},
          const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_QUBITSNUM
  [[nodiscard]] std::optional<size_t>
  getQubitsNum(const std::vector<Site>& sites = {},
               const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_PARAMETERSNUM
  [[nodiscard]] size_t
  getParametersNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_DURATION
  [[nodiscard]] std::optional<uint64_t>
  getDuration(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_FIDELITY
  [[nodiscard]] std::optional<double>
  getFidelity(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS
  [[nodiscard]] std::optional<uint64_t>
  getInteractionRadius(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS
  [[nodiscard]] std::optional<uint64_t>
  getBlockingRadius(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_IDLINGFIDELITY
  [[nodiscard]] std::optional<double>
  getIdlingFidelity(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_ISZONED
  [[nodiscard]] bool isZoned() const;

  /// @see QDMI_OPERATION_PROPERTY_SITES
  [[nodiscard]] std::optional<std::vector<Site>> getSites() const;

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
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getSitePairs() const;

  /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
  [[nodiscard]] std::optional<uint64_t>
  getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                        const std::vector<double>& params = {}) const;

  auto operator<=>(const Operation&) const noexcept = default;

private:
  /**
   * @brief Constructs an Operation object from a QDMI_Operation handle.
   * @param device The associated QDMI_Device handle.
   * @param operation The QDMI_Operation handle to wrap.
   */
  Operation(QDMI_Device device, QDMI_Operation operation)
      : device_(device), operation_(operation) {}

  /// Query a operation property.
  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] T queryProperty(const QDMI_Operation_Property prop,
                                const std::vector<Site>& sites,
                                const std::vector<double>& params) const {
    std::string msg = "Querying ";
    msg += qdmi::toString(prop);
    std::vector<QDMI_Site> qdmiSites;
    qdmiSites.reserve(sites.size());
    std::ranges::transform(sites, std::back_inserter(qdmiSites),
                           [](const Site& site) -> QDMI_Site { return site; });
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      std::string value(size - 1, '\0');
      result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, size, value.data(), nullptr);
      qdmi::throwIfError(result, msg);
      return value;
    } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                             T>) {
      size_t size = 0;
      auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      remove_optional_t<T> value(
          size / sizeof(typename remove_optional_t<T>::value_type));
      result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, size, value.data(), nullptr);
      qdmi::throwIfError(result, msg);
      return value;
    } else {
      remove_optional_t<T> value{};
      const auto result = QDMI_device_query_operation_property(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, sizeof(remove_optional_t<T>), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      return value;
    }
  }

  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;

  /// @brief The underlying QDMI_Operation object.
  QDMI_Operation operation_;

  friend class Device;
};
} // namespace fomac
